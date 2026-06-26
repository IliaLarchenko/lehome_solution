"""LeRobot-format dataset writer for eval runs.

Only the main process should call add_episode_from_temp (one at a time) to avoid
corruption. Workers write trajectory to temp dir; main process loads and appends.

Reusable for RL: same module can append episodes from rollouts.
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import logging
import os
import pickle
import resource
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets import video_utils as _lerobot_video_utils


def _install_crf_override(crf: int) -> None:
    """Monkey-patch LeRobot's ``encode_video_frames`` so every mp4 it writes uses
    the given CRF value instead of the hard-coded 30.  Idempotent — if already
    patched to the same crf it's a no-op.  Applied once at module import time
    so every ``LeRobotDataset.save_episode`` call that eventually reaches
    ``encode_video_frames`` uses the override.

    We override at the *function* level (not LeRobotDataset class level) because
    LeRobotDataset's ``create`` does not expose a ``crf`` kwarg and its
    internal ``encode_episode_videos`` hard-codes no crf — so the only way to
    shift the encoder quality is to patch ``encode_video_frames`` directly.
    """
    orig = getattr(_lerobot_video_utils, "_lehome_orig_encode_video_frames",
                   _lerobot_video_utils.encode_video_frames)

    def _wrapped(*args, **kwargs):
        # Respect an explicit caller-provided crf (e.g. non-default unit test),
        # otherwise force our override.
        if "crf" not in kwargs:
            kwargs["crf"] = crf
        return orig(*args, **kwargs)

    _lerobot_video_utils._lehome_orig_encode_video_frames = orig
    _lerobot_video_utils.encode_video_frames = _wrapped
    # Also patch the LeRobotDataset module that already did `from video_utils
    # import encode_video_frames` at its top; the reimport binds to the patched
    # function for callers that reach it via that module-level name.
    try:
        from lerobot.datasets import lerobot_dataset as _lerobot_ds_mod
        if hasattr(_lerobot_ds_mod, "encode_video_frames"):
            _lerobot_ds_mod.encode_video_frames = _wrapped
    except Exception:
        pass


_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _get_rss_mb() -> float:
    """Current process resident-set size in MB.

    Uses /proc/self/statm so we get *current* RSS, not the peak RSS that
    getrusage(ru_maxrss) returns.  Peak RSS is monotonic and useless for
    measuring per-step deltas in a long-running process.
    """
    try:
        with open("/proc/self/statm", "r") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * _PAGE_SIZE / (1024 * 1024)
    except Exception:
        # Non-Linux fallback: peak RSS (better than nothing).
        # getrusage ru_maxrss is in KB on Linux, bytes on macOS.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _release_caches_and_log(label: str, ep_idx: int) -> None:
    """Force-release caches that are easy to mistake for leaks.

    Called once per episode write so the orchestrator's RSS reflects what's
    actually live, not what's been allocated and forgotten.

    What this does and why:
    - gc.collect(): trajectory dicts contain numpy arrays nested in lists nested
      in dicts. Python's gen-2 GC fires lazily; an explicit collect after each
      episode catches reference cycles before they accumulate.
    - pyarrow.default_memory_pool().release_unused(): LeRobot's add_frame /
      save_episode round-trip non-video columns through pyarrow. Pyarrow's
      memory pool caches buffers for reuse and DOES NOT return memory to the OS
      by default — to a process monitor this looks identical to a leak. The
      release_unused() call hands free buffers back to malloc / mmap.
    - Logging RSS at INFO level so we can SEE the trend across episodes
      instead of guessing.
    """
    gc.collect()
    try:
        import pyarrow as _pa
        pool = _pa.default_memory_pool()
        bytes_allocated_before = pool.bytes_allocated()
        pool.release_unused()
        bytes_allocated_after = pool.bytes_allocated()
        pa_freed_mb = (bytes_allocated_before - bytes_allocated_after) / (1024 * 1024)
    except Exception:
        pa_freed_mb = 0.0
    rss_mb = _get_rss_mb()
    logger.info(
        "[%s] post-write ep=%s rss=%.0fMB pyarrow_freed=%.0fMB",
        label, ep_idx, rss_mb, pa_freed_mb,
    )

# vcodec = libsvtav1 matches LeRobot's default (ecosystem compatibility).
# We rely on LeRobot's default non-streaming encoding path — no custom
# encoder, no monkey patching. Simpler, reliable, slightly slower per
# episode but never silently loses frames.
_DATASET_VCODEC = "libsvtav1"
# Override LeRobot's hard-coded crf=30 with crf=15: at CRF=30 the 320x240 rollouts
# are blurry enough that the success head learns sharpness as a "BC ⇒ success=1"
# shortcut (see scripts/diag_bc_param_sweep.py). Keep this load-bearing.
_DATASET_CRF = 15

# Install the CRF override once at module import, before any LeRobotDataset is
# constructed so every save_episode picks it up.
_install_crf_override(_DATASET_CRF)


def _assert_dataset_integrity(dataset: LeRobotDataset, root: Path) -> None:
    """Post-rollout sanity check: every per-chunk per-camera mp4 must have
    exactly sum(episode lengths) frames. Raises if the video files and the
    episode metadata disagree — prevents a corrupt rollout from being
    uploaded or used for training.

    Cheap: one ffprobe per (chunk, camera), ~dozens of calls per rollout.

    Grouping is per-camera by that camera's own (chunk_index, file_index):
    LeRobot rotates a video to file-NNN+1 once the current mp4 crosses a size
    threshold, independently per camera. top_rgb (overhead full-scene view)
    rotates first because it compresses worse than the wrist cams. Grouping by
    data/chunk+file and taking the first episode's video chunk/file would miss
    the rotated tail file and produce a spurious mismatch.
    """
    import pyarrow.parquet as pq
    import pandas as _pd
    ep_parquets = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
    if not ep_parquets:
        return
    ep_df = _pd.concat([pq.read_table(str(p)).to_pandas() for p in ep_parquets],
                       ignore_index=True)
    video_keys = list(dataset.meta.video_keys)
    problems: list[str] = []
    for vk in video_keys:
        vk_chunk = f"videos/{vk}/chunk_index"
        vk_file = f"videos/{vk}/file_index"
        for (vc, vf), grp in ep_df.groupby([vk_chunk, vk_file]):
            expected = int(grp["length"].sum())
            mp4 = Path(root) / dataset.meta.video_path.format(
                video_key=vk, chunk_index=int(vc), file_index=int(vf),
            )
            if not mp4.exists():
                problems.append(f"{mp4} missing")
                continue
            try:
                out = subprocess.check_output([
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-count_packets", "-show_entries", "stream=nb_read_packets",
                    "-of", "csv=p=0", str(mp4),
                ], text=True, timeout=120).strip()
                actual = int(out)
            except Exception as e:
                problems.append(f"{mp4} ffprobe failed: {e}")
                continue
            if actual != expected:
                problems.append(
                    f"chunk={vc} file={vf} cam={vk} mp4_frames={actual} "
                    f"parquet_frames={expected} ({mp4})"
                )
    if problems:
        raise RuntimeError(
            "Dataset integrity check FAILED:\n  - " + "\n  - ".join(problems)
        )
    logger.info("Dataset integrity check OK (%d episodes, %d video keys).",
                len(ep_df), len(video_keys))


from lehome_solution.eval.metadata import EvalRunMetadata, EpisodeMetadata
from lehome_solution.shared.eval_wrapper import resize_with_pad
from lehome_solution.constants import EMA_ALPHA as _DEFAULT_EMA_ALPHA


_OVERLAY_MARGIN = 10
_OVERLAY_CHART_H = 70
_BORDER_PX = 2


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 [0,255]. Handles float32 [0,1] and [0,255] ranges."""
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        if float(img.max()) > 1.5:
            return img.clip(0, 255).astype(np.uint8)
        else:
            return (img.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return img.clip(0, 255).astype(np.uint8)


def _compose_frame(top_raw: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose multi-camera view: [left|right] on top row, top-cam below. Matches policy _add_frame."""
    top = _to_uint8(np.asarray(top_raw))[..., :3]
    top = top[::-1, ::-1]       # 180° rotation (top camera is upside-down in raw feed)
    left = _to_uint8(np.asarray(left))[..., :3]
    right = _to_uint8(np.asarray(right))[..., :3]
    h, w = top.shape[:2]
    left_s = cv2.resize(left, (w // 2, h // 2))
    right_s = cv2.resize(right, (w // 2, h // 2))
    top_row = np.concatenate([left_s, right_s], axis=1)
    if top_row.shape[1] != w:
        top_row = cv2.resize(top_row, (w, top_row.shape[0]))
    frame = np.concatenate([top_row, top], axis=0)
    sh, sw = top_row.shape[0], top_row.shape[1] // 2
    frame[sh:sh + _BORDER_PX, :] = 0
    frame[:sh, sw:sw + _BORDER_PX] = 0
    return frame


def _draw_overlay(frame: np.ndarray, success_preds: list[float], advantages: list[float],
                  frame_idx: int, total_frames: int,
                  cumulative_rewards: list[float] | None = None,
                  completion_preds: list[float] | None = None,
                  ttc_preds: list[float] | None = None) -> np.ndarray:
    """Draw S (blue), A (orange), R (green), C (white), T (cyan) chart on the frame.

    S: EMA-smoothed P(success) from model (0-1 axis), blue
    A: GAE advantage (-1 to +1 range), orange  — includes GAE_completion when α_c>0
    R: cumulative collected return (0-1 axis), green
    C: completion prediction (0-1 axis), white
    T: time-to-completion (0-1 axis), cyan
    """
    if not success_preds and not cumulative_rewards:
        return frame
    h, w = frame.shape[:2]
    margin, ch = _OVERLAY_MARGIN, _OVERLAY_CHART_H
    cw = w - 2 * margin
    left, top = margin, margin
    tf = max(total_frames, 1)
    inner_h = ch - 2 * margin
    mid_y = top + margin + inner_h // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (left + cw, top + ch), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    n = len(success_preds) if success_preds else (len(cumulative_rewards) if cumulative_rewards else 0)
    # Dotted midline
    for x_dot in range(left + margin, left + cw - margin, 7):
        cv2.circle(frame, (x_dot, mid_y), 1, (140, 140, 140), -1)

    def _draw_series(series, color, vmin=0.0, vmax=1.0, thickness=2):
        pts = []
        for i, v in enumerate(series):
            v_clip = max(vmin, min(vmax, float(v)))
            x = left + margin + int((i / tf) * (cw - 2 * margin))
            frac = (v_clip - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            y = top + margin + inner_h - int(frac * inner_h)
            pts.append((x, y))
        if len(pts) >= 2:
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j + 1], color, thickness)

    # Green: cumulative reward R (0-1)
    if cumulative_rewards is not None:
        _draw_series(cumulative_rewards[:n], (0, 220, 0), 0.0, 1.0)

    # Blue: raw P(success) from model (0-1)
    if success_preds:
        _draw_series(success_preds[:n], (220, 160, 0), 0.0, 1.0)  # BGR: blue-ish

    # Orange: advantage A (-1 to +1)
    if advantages:
        _draw_series(advantages[:n], (0, 160, 255), -1.0, 1.0)

    # White: completion prediction C (0-1)
    if completion_preds is not None:
        _draw_series(completion_preds[:n], (255, 255, 255), 0.0, 1.0)

    # Cyan: time-to-completion T (0-1).
    if ttc_preds is not None:
        _draw_series(ttc_preds[:n], (220, 200, 0), 0.0, 1.0)

    cv2.rectangle(frame, (left, top), (left + cw, top + ch), (120, 120, 120), 1)
    last_r = float(cumulative_rewards[-1]) if cumulative_rewards and len(cumulative_rewards) > 0 else 0.0
    last_s = float(success_preds[-1]) if success_preds and len(success_preds) > 0 else 0.0
    last_a = float(advantages[min(n - 1, len(advantages) - 1)]) if advantages and n > 0 else 0.0
    label_text = f"S:{last_s:.2f} A:{last_a:+.2f} R:{last_r:.2f}"
    if completion_preds is not None and len(completion_preds) > 0:
        last_c = float(completion_preds[-1])
        label_text += f" C:{last_c:.2f}"
    if ttc_preds is not None and len(ttc_preds) > 0:
        last_ttc = float(ttc_preds[-1])
        label_text += f" T:{last_ttc:.2f}"
    cv2.putText(frame, label_text,
                (left + 5, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return frame


def render_episode_video(
    frames: list[dict],
    video_path: str | Path,
    success: bool | None,
    fps: int = 30,
    lam: float = 0.99,
    success_alpha: float = 0.5,
    completion_alpha: float = 0.5,
    value_tail_k: int = 30,
    dense_rewards: list[float] | None = None,
    camera_size: tuple[int, int] | None = (240, 320),
    top_camera_size: tuple[int, int] | None = None,
) -> bool:
    """Render multi-camera video with R + V + GAE overlay. Called by main process.

    The overlay uses GAE for approximate live visualization. Final training
    advantages come from recompute_advantages.py (segment-level, not GAE).

    Args:
        frames: trajectory frames from pkl, each with observation.images.{top,left,right}_rgb + value
        video_path: output .mp4 path
        success: true episode outcome (0/1); None falls back to last V as proxy
        fps: frames per second
        lam: GAE lambda (default 0.99)
        success_alpha: dampening coefficient for the P_success transition in GAE_success
        completion_alpha: dampening coefficient for the completion transition in GAE_completion
        dense_rewards: per-step dense rewards from compute_dense_reward (optional)
        camera_size: (h, w) per-camera target size before composition.  Default
            (240, 320) → composed grid 360x320.  Set to None to keep camera
            native resolution (much heavier at 640x480).
        top_camera_size: optional (h, w) override applied only to the top
            camera. Use e.g. (240, 426) for real-format runs where the top is
            16:9 (1280×720) — keeps the bottom strip at the source aspect
            ratio instead of squashing it to 4:3. ``None`` (default) means
            the top uses ``camera_size`` like the wrists.
    Returns True on success, False on error.
    """
    def _downsize_to(img, size):
        if size is None or img is None:
            return img
        ih, iw = img.shape[:2]
        th, tw = size
        if (ih, iw) == (th, tw):
            return img
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    _top_size = top_camera_size if top_camera_size is not None else camera_size

    def _downsize_top(img):
        return _downsize_to(img, _top_size)

    def _downsize(img):
        return _downsize_to(img, camera_size)
    if not frames:
        return False
    try:
        raw_success_preds = [float(f.get("success_pred", f.get("value", 0.0))) for f in frames]
        # EMA smoothing — same α and init as training GAE (first prediction)
        _ema_alpha = _DEFAULT_EMA_ALPHA
        _ema = raw_success_preds[0] if raw_success_preds else 0.5
        smoothed_preds: list[float] = []
        for v in raw_success_preds:
            _ema = _ema_alpha * v + (1.0 - _ema_alpha) * _ema
            smoothed_preds.append(_ema)
        # Apply the same tail-frame correction the GAE uses, so the displayed
        # blue S-curve matches what the value function actually consumes.
        if value_tail_k > 0 and success is not None and len(smoothed_preds) > value_tail_k + 1:
            _T = len(smoothed_preds)
            _K = int(value_tail_k)
            _anchor = smoothed_preds[_T - _K - 1]
            _target = 1.0 if success else 0.0
            for _i in range(1, _K + 1):
                _frac = _i / _K
                smoothed_preds[_T - _K - 1 + _i] = _anchor + (_target - _anchor) * _frac
        completion_preds: list[float] | None = None
        if any(f.get("completion_pred") is not None for f in frames):
            completion_preds = [float(f.get("completion_pred", 0.0)) for f in frames]
        ttc_preds: list[float] | None = None
        if any(f.get("ttc_pred") is not None for f in frames):
            ttc_preds = [float(f.get("ttc_pred", 0.0)) for f in frames]
        # Overlay mirrors training advantage structure: α_s-dampened success-GAE plus
        # a separate, always-on α_c-weighted GAE_completion over the completion head.
        # Overlay shows raw GAE_success + GAE_completion (no segment blend); training
        # uses w·GAE_success + (1-w)·segment + GAE_completion.
        success_label_opt = (1.0 if success else 0.0) if success is not None else None
        all_advs, _gae_values = compute_gae(
            raw_success_preds, dense_rewards=dense_rewards, lam=lam,
            success_alpha=success_alpha,
            success_label=success_label_opt,
            value_tail_k=value_tail_k,
        )
        if completion_alpha != 0.0 and completion_preds is not None:
            gae_c = compute_completion_gae(
                completion_preds, len(raw_success_preds),
                completion_alpha=completion_alpha,
                lam=lam,
                success_label=success_label_opt,
            )
            all_advs = [a + c for a, c in zip(all_advs, gae_c)]

        if dense_rewards is not None:
            cumulative_rewards: list[float] | None = []
            cum = 0.0
            for r in dense_rewards:
                cum += r
                cumulative_rewards.append(cum)
        else:
            cumulative_rewards = None

        first = frames[0]
        composed0 = _compose_frame(
            _downsize_top(first["observation.images.top_rgb"]),
            _downsize(first["observation.images.left_rgb"]),
            _downsize(first["observation.images.right_rgb"]),
        )
        h, w = composed0.shape[:2]

        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
            str(video_path),
        ]
        try:
            # Route ffmpeg output to logs/ subfolder so SVT-AV1 encoder messages
            # don't leak to the terminal.  start_new_session=True detaches from TTY.
            logs_dir = Path(video_path).parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg_log = logs_dir / (Path(video_path).stem + ".ffmpeg.log")
            ffmpeg_log_fh = open(ffmpeg_log, "w")
            proc = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE,
                stdout=ffmpeg_log_fh, stderr=ffmpeg_log_fh,
                start_new_session=True,
            )
            for i, frame_data in enumerate(frames):
                composed = _compose_frame(
                    _downsize_top(frame_data["observation.images.top_rgb"]),
                    _downsize(frame_data["observation.images.left_rgb"]),
                    _downsize(frame_data["observation.images.right_rgb"]),
                )
                out = _draw_overlay(composed, smoothed_preds[:i + 1], all_advs[:i + 1], i, len(frames),
                                    cumulative_rewards=cumulative_rewards[:i + 1] if cumulative_rewards else None,
                                    completion_preds=completion_preds[:i + 1] if completion_preds else None,
                                    ttc_preds=ttc_preds[:i + 1] if ttc_preds else None)
                proc.stdin.write(out.tobytes())
            proc.stdin.close()
            proc.wait()
            ffmpeg_log_fh.close()
        except FileNotFoundError:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for i, frame_data in enumerate(frames):
                composed = _compose_frame(
                    _downsize_top(frame_data["observation.images.top_rgb"]),
                    _downsize(frame_data["observation.images.left_rgb"]),
                    _downsize(frame_data["observation.images.right_rgb"]),
                )
                out = _draw_overlay(composed, smoothed_preds[:i + 1], all_advs[:i + 1], i, len(frames),
                                    cumulative_rewards=cumulative_rewards[:i + 1] if cumulative_rewards else None,
                                    completion_preds=completion_preds[:i + 1] if completion_preds else None,
                                    ttc_preds=ttc_preds[:i + 1] if ttc_preds else None)
                writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            writer.release()
        return True
    except Exception as e:
        logger.warning("render_episode_video failed for %s: %s", video_path, e)
        return False


# Expected completion order per garment type, derived from empirical log analysis.
# Each entry is a list of checkpoint groups (list of check indices).
# The first group is always the free spread checks (no reward).
# Each subsequent group is a reward-worthy checkpoint (1/num_checkpoints each).
# A checkpoint is reached when ALL checks in that group (and all prior) pass.
#
# top (5 cond): spread(3,4) → center fold(1) → both sleeve folds(0,2) together
#   Dominant success path: 00011 → 01011 → 11111
#   2 checkpoints → 1/2 each
# long-pant (7 cond): spread(1,2) → intermediate(4,5,6) → leg folds(0,3)
#   3 groups, 2 checkpoints → 1/2 each
# short-pant (4 cond): spread(3,2) → both edge folds(1,0) together
#   Dominant success path: 0011 → 1111
#   1 checkpoint → 1.0
GARMENT_CHECKPOINTS: dict[str, list[list[int]]] = {
    # Tops: superset checkpoints — second includes all conditions from first.
    # First checkpoint: collar alignment (1) + both spreads (3, 4).
    # Second checkpoint: all 5 conditions (superset — adds body fold 0, 2).
    "top-long-sleeve": [[], [1, 3, 4], [0, 1, 2, 3, 4]],
    "top-short-sleeve": [[], [1, 3, 4], [0, 1, 2, 3, 4]],
    # Long pants: first checkpoint = proximity checks (4,5,6); second = all 7 conditions.
    "long-pant": [[], [4, 5, 6], [0, 1, 2, 3, 4, 5, 6]],
    # Short pants: single checkpoint, ALL conditions must pass (folds + spreads).
    "short-pant": [[], [0, 1, 2, 3]],
}

# For shirts, the primary distance condition in the first checkpoint group.
# This is the only <= (proximity) condition in the group — the hard one that
# drives the gradual reward.  Index 1 = dist(p[2], p[3]) = collar/sleeve proximity.
FIRST_CP_PRIMARY_DISTANCE: dict[str, int] = {
    "top-long-sleeve": 1,
    "top-short-sleeve": 1,
}


def _garment_type_from_name(garment_name: str) -> str | None:
    """Derive canonical garment type from garment name (e.g. 'Top_Long_Unseen_0' → 'top-long-sleeve')."""
    prefix = "_".join(garment_name.split("_")[:2])
    return {
        "Top_Long": "top-long-sleeve",
        "Top_Short": "top-short-sleeve",
        "Pant_Short": "short-pant",
        "Pant_Long": "long-pant",
    }.get(prefix)


def compute_dense_reward(
    frames: list[dict], garment_name: str = "",
    success: bool | None = None,
) -> tuple[list[float], list[float]] | None:
    """Compute dense reward and checkpoint_held from check_status.

    Uses GARMENT_CHECKPOINTS to define ordered groups of conditions.  The first
    group (spread checks) gives no reward.  Each subsequent group is a checkpoint
    worth 1/num_checkpoints.  Groups can be completed in any order — skipping a
    middle checkpoint and completing a later one still earns reward.

    The first checkpoint can oscillate: if reached (+0.5) but later lost (-0.5),
    the reward reflects the regression. ``checkpoint_held`` tracks whether the
    first checkpoint condition is currently satisfied (0.0 or 1.0), expanded
    forward to fill frames between checks.

    **Failure withdrawal:** If ``success`` is False, all accumulated intermediate
    reward is withdrawn at the last step.  This ensures total episode reward is
    binary: 1.0 (success) or 0.0 (failure), aligning dense rewards with the
    final goal. Intermediate checkpoints still provide temporal credit within
    the episode, but the total reflects the true binary outcome.

    **Shirts only — gradual first checkpoint:**
    For tops, the first +0.5 is replaced with gradual reward proportional to how
    much the primary distance (sleeve proximity, index 1) decreases. Computed
    backwards after the episode, only for the first occurrence of the first
    checkpoint. Subsequent oscillations keep the binary +/-0.5 logic.

    Special cases:
    - Short pants: first checkpoint = success, so checkpoint_held mirrors success.

    Returns ``(dense_rewards, checkpoint_held)`` per frame, or None if no frame
    has check_status (backward compat).
    """
    # Find N from first frame with check_status
    N = 0
    for f in frames:
        cs = f.get("check_status")
        if cs is not None:
            N = len(cs)
            break
    if N == 0:
        return None

    # Get checkpoint groups for this garment type
    gtype = _garment_type_from_name(garment_name) if garment_name else None
    groups = GARMENT_CHECKPOINTS.get(gtype) if gtype else None
    if groups is None or not all(0 <= idx < N for g in groups for idx in g):
        # Fallback: first group is free (empty), each check is its own checkpoint
        groups = [[]] + [[i] for i in range(N)]

    num_checkpoints = len(groups) - 1  # first group is free
    if num_checkpoints <= 0:
        return [0.0] * len(frames), [0.0] * len(frames)

    # First checkpoint group is groups[1] (groups[0] is free/spread)
    first_cp_group = groups[1]
    # Other non-free groups (groups[2:]) — these remain monotonic
    other_groups = groups[2:]

    cp_reward = 1.0 / num_checkpoints  # reward for reaching/losing first checkpoint

    # --- Forward pass: binary rewards + oscillation ---
    rewards = [0.0] * len(frames)
    checkpoint_held = [0.0] * len(frames)
    prev_first_cp_held = False
    prev_other_completed = 0
    first_cp_frame = None  # frame index where first checkpoint was first reached
    first_cp_transitions = 0  # count of not-held → held transitions

    for t, f in enumerate(frames):
        cs = f.get("check_status")
        if cs is not None:
            first_cp_now = all(cs[idx] > 0.5 for idx in first_cp_group)
            other_completed = sum(
                1 for group in other_groups
                if all(cs[idx] > 0.5 for idx in group)
            )

            # First checkpoint oscillation: emit reward delta when state changes
            if first_cp_now and not prev_first_cp_held:
                rewards[t] += cp_reward
                first_cp_transitions += 1
                if first_cp_transitions == 1:
                    first_cp_frame = t
            elif not first_cp_now and prev_first_cp_held:
                rewards[t] -= cp_reward

            # Other groups: monotonic reward
            new_other = max(0, other_completed - prev_other_completed)
            if new_other > 0:
                rewards[t] += new_other / num_checkpoints

            prev_first_cp_held = first_cp_now
            prev_other_completed = max(prev_other_completed, other_completed)

            checkpoint_held[t] = 1.0 if first_cp_now else 0.0
        else:
            checkpoint_held[t] = 1.0 if prev_first_cp_held else 0.0

    # --- Backward pass: gradual first-checkpoint reward for shirts ---
    # Only applies to the first occurrence of the first checkpoint.
    # Requires check_distances data from Patch 6.
    primary_idx = FIRST_CP_PRIMARY_DISTANCE.get(gtype)
    if primary_idx is not None and first_cp_frame is not None:
        # Collect (frame_idx, distance_ratio) for the primary condition
        dist_samples: list[tuple[int, float]] = []
        for t, f in enumerate(frames):
            cd = f.get("check_distances")
            if cd is not None and len(cd) > primary_idx:
                dist_samples.append((t, float(cd[primary_idx])))
            if t >= first_cp_frame:
                break  # only need up to first checkpoint

        if len(dist_samples) >= 2:
            d_init = dist_samples[0][1]
            d_first_cp = dist_samples[-1][1]  # distance at/after first_cp_frame

            if d_init > d_first_cp + 1e-6:
                # Remove the binary +cp_reward from the first checkpoint frame
                rewards[first_cp_frame] -= cp_reward

                # Distribute cp_reward gradually based on distance reduction
                running_min = d_init
                prev_cumulative = 0.0
                for t, d in dist_samples:
                    running_min = min(running_min, d)
                    frac = (d_init - running_min) / (d_init - d_first_cp)
                    frac = max(0.0, min(1.0, frac))
                    cumulative = frac * cp_reward
                    increment = cumulative - prev_cumulative
                    if increment > 1e-9:
                        rewards[t] += increment
                    prev_cumulative = cumulative

    # --- Failure withdrawal: zero out total reward for failed episodes ---
    # Spread the withdrawal uniformly from the last peak reward frame to the end,
    # rather than a single spike at the terminal. This gives each post-checkpoint
    # step a small negative reward, creating smooth negative advantages throughout
    # the failing region.
    if success is False and len(rewards) > 0:
        cum = sum(rewards)
        if abs(cum) > 1e-9:
            # Find the last frame where a positive reward brought cumulative to its peak.
            # Only advance peak_frame on actual reward events (not frames where
            # cumulative passively stays at the peak level).
            running = 0.0
            peak = 0.0
            peak_frame = 0
            for t, r in enumerate(rewards):
                running += r
                if r > 1e-9 and running >= peak - 1e-9:
                    peak = max(peak, running)
                    peak_frame = t
            # Spread withdrawal uniformly from peak_frame+1 to end
            n_spread = len(rewards) - peak_frame - 1
            if n_spread > 0:
                per_step = -cum / n_spread
                for t in range(peak_frame + 1, len(rewards)):
                    rewards[t] += per_step
            else:
                rewards[-1] -= cum  # fallback: peak at last frame

    return rewards, checkpoint_held


def _ema_with_tail_correction(
    raw: list[float],
    ema_alpha: float,
    *,
    success_label: float | None = None,
    value_tail_k: int = 0,
) -> list[float]:
    """EMA-smooth ``raw`` in-order, optionally interpolating the last K frames toward
    ``success_label`` to cancel the near-terminal bias of the success head.

    EMA is seeded at ``raw[0]`` so ema[0] == raw[0].  ``value_tail_k <= 0`` or missing
    ``success_label`` disables the tail correction.
    """
    T = len(raw)
    if T == 0:
        return []
    out: list[float] = [0.0] * T
    out[0] = float(raw[0])
    for t in range(1, T):
        out[t] = ema_alpha * float(raw[t]) + (1.0 - ema_alpha) * out[t - 1]

    if value_tail_k > 0 and success_label is not None and T > value_tail_k + 1:
        K = int(value_tail_k)
        anchor_idx = T - K - 1
        anchor = out[anchor_idx]
        target = float(success_label)
        for i in range(1, K + 1):
            frac = i / K
            out[anchor_idx + i] = anchor + (target - anchor) * frac
    return out


def compute_gae(
    values: list[float],
    dense_rewards: list[float],
    lam: float = 0.99,
    gamma: float = 0.999,
    ema_alpha: float = _DEFAULT_EMA_ALPHA,
    success_alpha: float = 0.5,
    success_label: float | None = None,
    value_tail_k: int = 0,
    **_kwargs,
) -> tuple[list[float], list[float]]:
    """GAE for the success-prediction channel with an alpha-dampened value baseline.

    The value function here is the expected *remaining* reward, which — because the
    total return equals the success indicator — is

        V(s_t) = EMA(P_success)[t] − R_so_far[t],   R_so_far[t] = Σ_{τ<t} r[τ].

    The TD residual uses this full value, dampened by ``success_alpha`` (a CUPED-style
    control-variate coefficient: full subtraction is optimal only for a perfect
    predictor, so we attenuate the imperfect estimate). With V̂_t = α_s · V(s_t):

        δ[t] = r[t] + γ · V̂_{t+1} − V̂_t
        A[t] = δ[t] + γ·λ · A[t+1]

    Expanding V̂ (using R_so_far[t+1] = R_so_far[t] + r[t]) gives

        δ[t] = r[t]·(1 − α_s·γ) + α_s·(γ · P[t+1] − P[t]) + α_s·(1−γ)·R_so_far[t],

    so the reward is damped by (1 − α_s·γ) — the dampened return-subtraction — which
    is why checkpoint moments keep a positive (but attenuated) advantage. For γ≈1 the
    R_so_far term vanishes and only the (1 − α_s·γ) reward damping remains.

    Terminal P_success[T] is enforced from ``success_label`` (1 on success, 0 on
    failure) when provided — otherwise we extrapolate from the last smoothed
    prediction. The final-step reward r[T-1] is still included.

    ``value_tail_k`` linearly interpolates the tail of EMA(P_success) toward the
    true success label to cancel the "almost-success looks like failure" bias
    introduced by the marginal distribution the success head learns.

    Returns ``(advantages, V)`` where ``V[t] = EMA(P_success)[t] − R_so_far[t]`` is
    the *undampened* value (for reporting/overlay; can go negative). The
    completion-based GAE_completion component is computed separately by
    ``compute_completion_gae`` and added on top of the blend in
    ``compute_gae_advantages``.
    """
    T = len(values)
    r = [float(dr) for dr in dense_rewards]

    ema_s_arr = _ema_with_tail_correction(
        [float(v) for v in values],
        ema_alpha,
        success_label=success_label,
        value_tail_k=value_tail_k,
    )

    # Undampened value V[t] = EMA(P_success)[t] − R_so_far[t] (reporting/overlay),
    # and R_so_far[t] = Σ_{τ<t} r[τ], reused below to build the dampened baseline.
    V = [0.0] * T
    cum_before = [0.0] * T
    cum_reward = 0.0
    for t in range(T):
        cum_before[t] = cum_reward
        V[t] = ema_s_arr[t] - cum_reward
        cum_reward += r[t]
    cum_total = cum_reward  # R_so_far[T] = full cumulative reward = Σ_{τ<T} r[τ]

    if T == 0:
        return [], V

    # Terminal P_success[T]: enforce from success_label when provided; otherwise
    # extrapolate from the last smoothed prediction (no extra info available).
    p_terminal = float(success_label) if success_label is not None else ema_s_arr[-1]

    # α-dampened value baseline V̂_t = α_s · (P[t] − R_so_far[t]), including the
    # terminal V̂_T = α_s · (P_terminal − R_so_far[T]). δ_t = r_t + γ·V̂_{t+1} − V̂_t.
    v_hat = [success_alpha * (ema_s_arr[t] - cum_before[t]) for t in range(T)]
    v_hat_terminal = success_alpha * (p_terminal - cum_total)

    if T == 1:
        delta = r[0] + gamma * v_hat_terminal - v_hat[0]
        return [delta], V

    gae = [0.0] * T
    gae[-1] = r[-1] + gamma * v_hat_terminal - v_hat[-1]
    gl = gamma * lam
    for t in range(T - 2, -1, -1):
        delta = r[t] + gamma * v_hat[t + 1] - v_hat[t]
        gae[t] = delta + gl * gae[t + 1]
    return gae, V


def compute_completion_gae(
    completion_preds: list[float] | None,
    length: int,
    *,
    completion_alpha: float,
    gamma: float = 0.999,
    lam: float = 0.99,
    ema_alpha: float = _DEFAULT_EMA_ALPHA,
    success_label: float | None = None,
) -> list[float]:
    """GAE_completion: progress signal decoupled from the success prediction.

    Per-step delta uses ONLY the completion channel (no reward):
        δ_c[t] = (γ · EMA(C)[t+1] − EMA(C)[t]) · completion_alpha
        GAE_c[t] = δ_c[t] + γ·λ · GAE_c[t+1]

    Terminal C[T] is enforced from ``success_label`` (1 on success, 0 on failure)
    when provided — matching the fact that a successful episode's completion is
    100% and a failed one's is effectively 0 (we do not trust the head past T).

    Returns zeros when ``completion_alpha == 0`` or ``completion_preds`` is
    None / has the wrong length — so episodes without predictions contribute 0
    automatically.
    """
    T = int(length)
    if T == 0:
        return []
    if (
        completion_alpha == 0.0
        or completion_preds is None
        or len(completion_preds) != T
    ):
        return [0.0] * T

    ema_c_arr: list[float] = [0.0] * T
    ema_c_arr[0] = float(completion_preds[0])
    for t in range(1, T):
        ema_c_arr[t] = (
            ema_alpha * float(completion_preds[t])
            + (1.0 - ema_alpha) * ema_c_arr[t - 1]
        )

    c_terminal = float(success_label) if success_label is not None else ema_c_arr[-1]

    if T == 1:
        return [completion_alpha * (gamma * c_terminal - ema_c_arr[0])]

    gae_c = [0.0] * T
    gae_c[-1] = completion_alpha * (gamma * c_terminal - ema_c_arr[-1])
    gl = gamma * lam
    for t in range(T - 2, -1, -1):
        delta_c = completion_alpha * (gamma * ema_c_arr[t + 1] - ema_c_arr[t])
        gae_c[t] = delta_c + gl * gae_c[t + 1]
    return gae_c


TYPE_NUM_CHECKPOINTS = {
    "top_long": 2, "top_short": 2,
    "pant_long": 2, "pant_short": 1,
}

# Map garment_type from metadata to TYPE_NUM_CHECKPOINTS key
_GTYPE_TO_KEY = {
    "top-long-sleeve": "top_long", "top-short-sleeve": "top_short",
    "long-pant": "pant_long", "short-pant": "pant_short",
    "top_long": "top_long", "top_short": "top_short",
    "pant_long": "pant_long", "pant_short": "pant_short",
}


def _weighted_mean_std(arr: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Compute weighted mean and std."""
    w = weights / weights.sum()
    mean = float(np.dot(w, arr))
    var = float(np.dot(w, (arr - mean) ** 2))
    std = float(np.sqrt(var))
    return mean, std


def _garment_baselines(
    ep_indices: list[int],
    returns: list[float],
    episodes_info: list[dict],
    weights: list[float],
    type_mean: float,
    type_n: int,
    min_episodes: int = 5,
    baseline_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-episode baselines with garment → type → 0.5 fallback.

    For each episode, the baseline is the weighted mean return of its exact garment.
    Falls back to type_mean if the garment has < min_episodes, and to 0.5 if
    the type itself has < min_episodes.

    ``baseline_mask``: if provided, boolean array (same length as ep_indices).
    Only episodes where mask is True contribute to mean computation. All episodes
    still receive a baseline value.
    """
    n_eps = len(ep_indices)
    if baseline_mask is None:
        baseline_mask = np.ones(n_eps, dtype=bool)

    # Group by exact garment name
    garment_groups: dict[str, list[int]] = {}  # garment -> list of local indices
    for local_i, idx in enumerate(ep_indices):
        garment = episodes_info[idx].get("garment", "")
        garment_groups.setdefault(garment, []).append(local_i)

    baselines = np.full(n_eps, 0.5)

    # Type-level fallback (if enough type-level data)
    if type_n >= min_episodes:
        baselines[:] = type_mean

    # Garment-level override (if enough garment-level data)
    for _garment, local_indices in garment_groups.items():
        eligible = [i for i in local_indices if baseline_mask[i]]
        if len(eligible) >= min_episodes:
            g_returns = np.array([returns[i] for i in eligible])
            g_weights = np.array([weights[i] for i in eligible])
            g_mean, _ = _weighted_mean_std(g_returns, g_weights)
            for i in local_indices:
                baselines[i] = g_mean

    return baselines


def compute_segment_advantages(
    episodes_info: list[dict],
    clip_min: float = -2.0,
    clip_max: float = 2.0,
    min_baseline_weight: float = 0.5,
    beta: float = 1.0,
    **_kwargs,
) -> list[list[float]]:
    """Compute episode-level advantages: (dense_return - garment_baseline) / global_std.

    One advantage number per episode, all frames get the same value.
    Baseline fallback: garment mean (≥5 eps) → type mean (≥5 eps) → 0.5.
    Std computed globally across all garment types for consistent scaling.

    Only unbiased rollout types with sampling_share >= min_baseline_weight
    contribute to baseline/std computation. All episodes still get advantages.

    Each episode dict must have:
        garment_type, garment, dense_return, num_frames, success.
    Optional: sampling_share (default 1.0).
    """
    groups: dict[str, list[int]] = {}
    for i, ep in enumerate(episodes_info):
        key = _GTYPE_TO_KEY.get(ep["garment_type"], ep["garment_type"])
        groups.setdefault(key, []).append(i)

    # First pass: compute per-type residual stds, then RMS for global std
    # RMS of per-type stds gives each garment type equal influence on normalization.
    type_stds: list[float] = []
    group_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # gtype -> (ret_arr, w_arr, baselines)

    for gtype, ep_indices in groups.items():
        returns = [episodes_info[idx]["dense_return"] for idx in ep_indices]
        weights = [episodes_info[idx].get("sampling_share", 1.0) for idx in ep_indices]
        ret_arr = np.array(returns)
        w_arr = np.array(weights)

        # Mask: only unbiased rollouts with sufficient weight for baseline/std
        baseline_mask = np.array([
            episodes_info[idx].get("rollout_type", "normal") in _BASELINE_ROLLOUT_TYPES
            and episodes_info[idx].get("sampling_share", 1.0) >= min_baseline_weight
            for idx in ep_indices
        ], dtype=bool)
        n_eligible = int(baseline_mask.sum())

        # Compute type mean from eligible episodes only
        if n_eligible >= 5:
            type_mean, _ = _weighted_mean_std(ret_arr[baseline_mask], w_arr[baseline_mask])
        else:
            type_mean = 0.5  # fallback

        baselines = _garment_baselines(
            ep_indices, returns, episodes_info, weights,
            type_mean=type_mean, type_n=n_eligible,
            baseline_mask=baseline_mask,
        )
        residuals = ret_arr - baselines
        group_data[gtype] = (ret_arr, w_arr, baselines)
        # Std from eligible episodes only (but applied to all)
        if n_eligible >= 2:
            _, type_std = _weighted_mean_std(residuals[baseline_mask], w_arr[baseline_mask])
        else:
            _, type_std = _weighted_mean_std(residuals, w_arr)
        type_stds.append(type_std)

    # Global std = RMS of per-type stds (equal weight per type), scaled by AWR beta
    global_std = float(np.sqrt(np.mean(np.array(type_stds) ** 2))) if type_stds else 1.0
    if global_std < 1e-8:
        global_std = 1.0
    raw_std = global_std
    global_std *= beta
    logger.info("  Segment global_std=%.4f (raw=%.4f × beta=%.1f, RMS of %d type stds: %s)",
                global_std, raw_std, beta, len(type_stds), [f"{s:.4f}" for s in type_stds])

    # Second pass: normalize and assign
    result: list[list[float]] = [[] for _ in episodes_info]

    for gtype, ep_indices in groups.items():
        ret_arr, w_arr, baselines = group_data[gtype]
        type_mean = float(np.dot(w_arr / w_arr.sum(), ret_arr))
        advs = np.clip((ret_arr - baselines) / global_std, clip_min, clip_max)

        logger.info(
            "  Segment [%s] (%d eps): type_mean=%.4f global_std=%.4f "
            "baselines: mean=%.4f min=%.4f max=%.4f",
            gtype, len(ep_indices), type_mean, global_std,
            baselines.mean(), baselines.min(), baselines.max(),
        )

        for i, idx in enumerate(ep_indices):
            n = episodes_info[idx]["num_frames"]
            result[idx] = [float(advs[i])] * n

    return result


_BASELINE_ROLLOUT_TYPES = {"normal", "random", "full", "curriculum"}


def compute_gae_advantages(
    episodes_info: list[dict],
    lam: float = 0.99,
    gamma: float = 0.999,
    clip_min: float = -2.0,
    clip_max: float = 2.0,
    min_baseline_weight: float = 0.5,
    garment_success_rates: dict[str, float] | None = None,
    beta: float = 1.0,
    success_alpha: float = 0.5,
    completion_alpha: float = 0.5,
    value_tail_k: int = 0,
    **_kwargs,
) -> list[list[float]]:
    """Dataset-weighted blend of success-GAE and segment baselines, plus GAE_completion.

    Per-step advantage:
        A_t = w · GAE_success_t + (1-w) · segment_per_step_t + GAE_completion_t

    where w = min(sampling_share, 1.0) per episode. GAE_success uses the exact
    alpha-dampened value baseline V̂_t = success_alpha · (EMA(P_success)[t] − R_so_far[t]):
        δ_s[t] = r[t] + γ · V̂[t+1] − V̂[t]
    (see ``compute_gae`` — this includes the cumulative-reward term, so the reward is
    partially cancelled by the dampened return-subtraction). GAE_completion is a
    separate, potential-based shaping term over the completion head:
        δ_c[t] = (γ · EMA(C)[t+1] − EMA(C)[t]) · completion_alpha

    GAE_completion is decoupled from the segment blend so the task-progress signal
    reaches segment-dominated samples (old data, w≈0) at full strength. Episodes
    without completion predictions get GAE_completion=0 naturally.

    Both GAE_success and segment are RAW (unnormalized) before mixing. GAE_completion
    is added to the raw mix BEFORE normalization, so the global STD covers it too.
    """
    garment_sr = garment_success_rates or {}

    # ── Step 1: Per-garment baselines (reuse segment logic) ──
    groups: dict[str, list[int]] = {}
    for i, ep in enumerate(episodes_info):
        key = _GTYPE_TO_KEY.get(ep["garment_type"], ep["garment_type"])
        groups.setdefault(key, []).append(i)

    # ── Step 2: Per-garment segment baselines (checkpoint rate + conditional SR) ──
    # Segment 1 (before checkpoint): baseline = cp_rate × 0.5
    # Segment 2 (after checkpoint):  baseline = cond_SR - 0.5
    #   where cond_SR = SR / cp_rate (P(success | reached checkpoint))
    # Episodes without checkpoint: single segment, baseline = success_rate

    # Compute per-garment checkpoint rates from eligible episodes
    garment_cp_data: dict[str, dict] = {}  # {garment: {n_cp, n_total, n_success_with_cp}}
    for ep in episodes_info:
        if ep.get("rollout_type", "normal") not in _BASELINE_ROLLOUT_TYPES:
            continue
        if ep.get("sampling_share", 1.0) < min_baseline_weight:
            continue
        garment = ep.get("garment", "")
        if garment not in garment_cp_data:
            garment_cp_data[garment] = {"n_cp": 0, "n_total": 0, "n_success_with_cp": 0}
        gcd = garment_cp_data[garment]
        gcd["n_total"] += 1
        drs = ep.get("dense_rewards", [])
        reached_cp = False
        cum = 0.0
        for r in drs:
            cum += r
            if cum > 0.4:
                reached_cp = True
                break
        if reached_cp:
            gcd["n_cp"] += 1
            if ep.get("success", False):
                gcd["n_success_with_cp"] += 1

    # Build per-garment baselines for each segment
    garment_baselines: dict[str, dict] = {}  # {garment: {b_total, b_seg1, b_seg2}}
    for garment, gcd in garment_cp_data.items():
        cp_rate = gcd["n_cp"] / max(gcd["n_total"], 1)
        sr = garment_sr.get(garment, 0.5)
        cond_sr = (sr / cp_rate) if cp_rate > 0.01 else 0.5
        cond_sr = min(cond_sr, 1.0)
        garment_baselines[garment] = {
            "b_total": sr,                # E[total_return] = success_rate
            "b_seg1": cp_rate * 0.5,      # E[seg1_return] = P(reach_cp) × 0.5
            "b_seg2": cond_sr - 0.5,      # E[seg2_return | reached_cp]
            "cp_rate": cp_rate,
            "cond_sr": cond_sr,
        }
    logger.info("  Segment baselines: %s",
                {g: f"cp={v['cp_rate']:.2f} csr={v['cond_sr']:.2f} b1={v['b_seg1']:.3f} b2={v['b_seg2']:.3f}"
                 for g, v in sorted(garment_baselines.items())})

    # ── Step 3: Raw GAE_success + raw segment per step, then blend, then add GAE_completion ──
    gl = gamma * lam  # γλ ≈ 0.9890 for γ=0.999, λ=0.99
    mixed_advs: list[list[float]] = [[] for _ in episodes_info]

    # Debug accumulators (before normalization)
    _dbg_gae_all: list[float] = []
    _dbg_seg_all: list[float] = []
    _dbg_mix_all: list[float] = []
    _dbg_comp_all: list[float] = []
    _eps_with_completion = 0

    def _geom(length):
        return (1.0 - gl ** max(length, 1)) / (1.0 - gl) if abs(gl - 1.0) > 1e-12 else float(length)

    for i, ep in enumerate(episodes_info):
        n = ep["num_frames"]
        dense_return = ep["dense_return"]
        garment = ep.get("garment", "")
        w = 0.0 if ep.get("force_segment") else min(float(ep.get("sampling_share", 1.0)), 1.0)
        gb = garment_baselines.get(garment, {"b_total": 0.5, "b_seg1": 0.25, "b_seg2": 0.0})

        drs = ep.get("dense_rewards", [0.0] * n)
        comp_preds = ep.get("completion_preds")
        if comp_preds is not None and len(comp_preds) != n:
            raise RuntimeError(
                f"compute_gae_advantages: completion_preds length mismatch for "
                f"episode {ep.get('ep_idx')} garment={ep.get('garment')} "
                f"(len={len(comp_preds)}, expected {n}). Upstream frame drop?"
            )
        success_label = 1.0 if ep.get("success") else 0.0

        # Raw success GAE (alpha-dampened P_success transition + full-weight reward).
        if ep.get("values"):
            gae_raw, _ = compute_gae(
                ep["values"], drs, lam=lam, gamma=gamma,
                success_alpha=success_alpha,
                success_label=success_label,
                value_tail_k=value_tail_k,
            )
        else:
            gae_raw = [0.0] * n

        # Find first checkpoint frame
        cp_frame = None
        cum = 0.0
        for t_dr in range(len(drs)):
            cum += drs[t_dr]
            if cum > 0.4:
                cp_frame = t_dr
                break

        # Segment advantage per step with proper per-segment baselines
        seg_values = [0.0] * n
        if cp_frame is not None and 0 < cp_frame < n - 1:
            n1 = cp_frame + 1
            n2 = n - n1
            R1 = sum(drs[:n1])
            R2 = sum(drs[n1:])
            seg1 = (R1 - gb["b_seg1"]) * _geom(n1) / max(n1, 1)
            seg2 = (R2 - gb["b_seg2"]) * _geom(n2) / max(n2, 1)
            for t in range(n1):
                seg_values[t] = seg1
            for t in range(n1, n):
                seg_values[t] = seg2
        else:
            # No checkpoint: single segment, baseline = success_rate
            geom_sum = _geom(n)
            seg_per_step = (dense_return - gb["b_total"]) * geom_sum / max(n, 1)
            seg_values = [seg_per_step] * n

        # Blend: w * raw_GAE_success + (1-w) * raw_segment_per_step
        mixed = [w * g + (1.0 - w) * s for g, s in zip(gae_raw, seg_values)]

        # GAE_completion applied at full strength (independent of blend weight w).
        # Only eligible when completion predictions are available — BC/DAgger episodes
        # without predictions get GAE_completion=0 automatically.
        if completion_alpha != 0.0 and comp_preds is not None:
            gae_c = compute_completion_gae(
                comp_preds, n,
                completion_alpha=completion_alpha,
                gamma=gamma, lam=lam,
                success_label=success_label,
            )
            mixed = [m + c for m, c in zip(mixed, gae_c)]
            _dbg_comp_all.extend(gae_c)
            _eps_with_completion += 1

        mixed_advs[i] = mixed

        # Debug accumulation
        _dbg_seg_all.extend(seg_values)
        _dbg_gae_all.extend(gae_raw)
        _dbg_mix_all.extend(mixed)

    # ── Step 3: Global STD from eligible episodes ──
    eligible_vals: list[float] = []
    for i, ep in enumerate(episodes_info):
        if (ep.get("rollout_type", "normal") in _BASELINE_ROLLOUT_TYPES
                and ep.get("sampling_share", 1.0) >= min_baseline_weight):
            eligible_vals.extend(mixed_advs[i])

    global_std = float(np.std(eligible_vals)) if len(eligible_vals) > 1 else 1.0
    if global_std < 1e-8:
        global_std = 1.0
    raw_std = global_std
    global_std *= beta

    # ── Debug logging (before normalization) ──
    def _stats(vals):
        if not vals:
            return "empty"
        a = np.array(vals)
        return f"mean={a.mean():.6f} std={a.std():.6f} min={a.min():.6f} max={a.max():.6f}"

    logger.info("  [DEBUG] γλ=%.6f, 1-γλ=%.6f (γ=%.4f, λ=%.2f)", gl, 1.0 - gl, gamma, lam)
    logger.info("  [DEBUG] RAW segment per-step: %s", _stats(_dbg_seg_all))
    logger.info("  [DEBUG] RAW GAE_success per-step (α_s=%.3f): %s",
                success_alpha, _stats(_dbg_gae_all))
    if completion_alpha != 0.0:
        logger.info("  [DEBUG] RAW GAE_completion per-step: %s  (α_c=%.3f, %d/%d episodes)",
                    _stats(_dbg_comp_all), completion_alpha,
                    _eps_with_completion, len(episodes_info))
    logger.info("  [DEBUG] RAW blended+completion per-step: %s", _stats(_dbg_mix_all))
    logger.info("  [DEBUG] global_std=%.6f (raw=%.6f × beta=%.1f, from %d eligible values, %d episodes)",
                global_std, raw_std, beta, len(eligible_vals),
                sum(1 for ep in episodes_info
                    if ep.get("rollout_type", "normal") in _BASELINE_ROLLOUT_TYPES
                    and ep.get("sampling_share", 1.0) >= min_baseline_weight))

    # ── Step 4: Normalize by global STD and clip ──
    result: list[list[float]] = [[] for _ in episodes_info]
    for i in range(len(episodes_info)):
        result[i] = [max(clip_min, min(clip_max, a / global_std)) for a in mixed_advs[i]]

    return result


logger = logging.getLogger(__name__)

# Default camera/dataset image shape (H, W, C). Everything from recording through
# dataset storage uses this resolution; only the model's 224×224 resize is separate.
# Override per-run by passing image_shape to EvalDatasetWriter / KeyframeDatasetWriter.
DEFAULT_IMAGE_SHAPE = (480, 640, 3)

# Optional features when value model is used during eval/RL
VALUE_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["success_pred"],
}
# P(reward >= 0.5) prediction — checkpoint probability head
CHECKPOINT_PRED_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["checkpoint_pred"],
}
# Advantage A(s_t) — placeholder during collection, recomputed by recompute_advantages.py
ADVANTAGE_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["advantage"],
}
# Binary success label (0.0 or 1.0) — used as value head training target
SUCCESS_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["success"],
}
# Dense reward per step from success check prefix progression
DENSE_REWARD_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["dense_reward"],
}
# Per-garment average success rate (set by recompute_advantages.py, placeholder 0.5)
GARMENT_AVG_SUCCESS_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["garment_avg_success"],
}
# Per-garment average checkpoint reach rate (set by recompute_advantages.py, placeholder 0.5)
GARMENT_AVG_CHECKPOINT_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["garment_avg_checkpoint"],
}
# Whether the first checkpoint condition is currently satisfied (0.0 or 1.0).
# Oscillates with the checkpoint state; forward-filled between checks.
CHECKPOINT_HELD_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["checkpoint_held"],
}
# %completion prediction (step fraction within episode)
COMPLETION_PRED_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["completion_pred"],
}
# Time-to-completion prediction: -steps_left/600 (success) or -1-steps_left/600 (failure)
TTC_PRED_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["ttc_pred"],
}

# Max number of success-check conditions across all garment types (long-pant = 7).
# Per-garment active counts: short-top=5, long-top=5, short-pant=4, long-pant=7.
# Shorter arrays are right-padded with NaN so downstream masking is unambiguous.
MAX_CHECK_CONDITIONS = 7

# Per-condition keypoint distance values, already normalized by success threshold
# (value/threshold; >1 means condition not yet met). Float32, length MAX_CHECK_CONDITIONS,
# NaN-padded. Populated from launcher Patch 5 via pkl frames.
CHECK_DISTANCES_FEATURE = {
    "dtype": "float32",
    "shape": (MAX_CHECK_CONDITIONS,),
    "names": [f"cond_{i}" for i in range(MAX_CHECK_CONDITIONS)],
}

# Keypoint-distance head prediction (Head 1): 21-wide per-garment-slice vector.
# Mirrors KEYPOINT_HEAD_WIDTH from lehome_solution.constants — kept inlined here
# to avoid circular import at module load.
KEYPOINT_DISTANCES_PRED_WIDTH = 21
KEYPOINT_DISTANCES_PRED_FEATURE = {
    "dtype": "float32",
    "shape": (KEYPOINT_DISTANCES_PRED_WIDTH,),
    "names": [f"slot_{i}" for i in range(KEYPOINT_DISTANCES_PRED_WIDTH)],
}
# WM-flow head predictions (Head 3) — monitoring-only cond + uncond.
# success / completion are scalars; keypoint is a 21-wide vector.
WM_FLOW_SCALAR_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["wm_flow"],
}
WM_FLOW_KEYPOINT_FEATURE = {
    "dtype": "float32",
    "shape": (KEYPOINT_DISTANCES_PRED_WIDTH,),
    "names": [f"slot_{i}" for i in range(KEYPOINT_DISTANCES_PRED_WIDTH)],
}

# Best-of-N candidate-spread diagnostics. Populated per-chunk (same value
# across all frames of the chunk) when the policy server was started with
# ``num_rollout_candidates > 1``; all-NaN otherwise.
BON_SCALAR_FEATURE = {
    "dtype": "float32",
    "shape": (1,),
    "names": ["best_of_n_score"],
}
BON_INT_FEATURE = {
    "dtype": "int32",
    "shape": (1,),
    "names": ["best_of_n_n_valid"],
}


def _pad_check_distances(raw: Any) -> np.ndarray:
    """Right-pad ``raw`` to MAX_CHECK_CONDITIONS with NaN; return all-NaN when absent.

    The launcher produces per-garment arrays of length 5 (tops), 4 (short-pant),
    or 7 (long-pant); we normalize to a single fixed width so the dataset
    column has a static schema.
    """
    out = np.full(MAX_CHECK_CONDITIONS, np.nan, dtype=np.float32)
    if raw is None:
        return out
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    n = min(arr.shape[0], MAX_CHECK_CONDITIONS)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _fixed_width_pred(raw: Any, width: int) -> np.ndarray:
    """Coerce a raw per-frame prediction to shape ``(width,)`` float32.

    Truncates / NaN-pads as needed. Used for keypoint & wm-flow keypoint heads
    where every frame must have the same column width.
    """
    out = np.full(width, np.nan, dtype=np.float32)
    if raw is None:
        return out
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    n = min(arr.shape[0], width)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _scalar_pred(raw: Any) -> np.ndarray:
    """Coerce a raw per-frame scalar prediction to shape ``(1,)`` float32.
    Returns all-NaN when absent, matching wm_flow scalar-column schema."""
    if raw is None:
        return np.array([np.nan], dtype=np.float32)
    v = np.asarray(raw, dtype=np.float32).reshape(-1)
    return np.array([float(v[0]) if v.size else np.nan], dtype=np.float32)

# Shared constants for feature definitions
_JOINT_NAMES = [
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
]


def _camera_features(image_shape: tuple[int, int, int]) -> dict:
    return {
        f"observation.images.{cam}_rgb": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
        for cam in ("top", "left", "right")
    }


def _eval_features(image_shape: tuple[int, int, int]) -> dict:
    """Build LeHome eval feature dict for the given image shape."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": list(_JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (12,),
            "names": list(_JOINT_NAMES),
        },
        **_camera_features(image_shape),
    }


TEMP_EPISODES_DIR = "temp_episodes"
EVAL_RUN_INFO = "meta/eval_run_info.json"
EVAL_EPISODE_META_DIR = "meta/eval_episode_metadata"
GARMENT_INFO_PATH = "meta/garment_info.json"


def _resize_image(img: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize image to target_shape (H, W, C) if needed."""
    if img is None:
        return img
    arr = np.asarray(img)
    if arr.shape[:2] == target_shape[:2]:
        return arr
    return resize_with_pad(arr, target_shape[0], target_shape[1])


def sanitize_basename(s: str) -> str:
    """Sanitize a string for use as a filename component."""
    for c in r'/\:*?"<>|':
        s = s.replace(c, "_")
    return s.strip(" .") or "unknown"


def _pose_to_list(pose: Any) -> list | None:
    """Normalize object_initial_pose to list format for garment_info.json (same as record.append_episode_initial_pose)."""
    if pose is None:
        return None
    if isinstance(pose, dict):
        if "Garment" in pose:
            pose = pose["Garment"]
        else:
            pose = list(pose.values())[0] if pose else None
    if pose is None:
        return None
    if hasattr(pose, "tolist"):
        return pose.tolist()
    if isinstance(pose, (list, tuple)):
        return list(pose)
    return None


def _merge_garment_info_entry(
    root: Path,
    garment_name: str,
    episode_index: int,
    garment_info: dict[str, Any],
) -> None:
    """Merge one episode's garment_info into meta/garment_info.json (LeRobot/challenge format)."""
    path = root / GARMENT_INFO_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    # Proxy's garment_name is authoritative: after env.switch_garment() the
    # Isaac-side payload's garment_name lags by one garment, so trusting it
    # files episodes under the *previous* garment's key.
    gname = garment_name or garment_info.get("garment_name")
    if gname not in data:
        data[gname] = {}
    pose_list = _pose_to_list(garment_info.get("object_initial_pose"))
    episode_rec = {"object_initial_pose": pose_list if pose_list is not None else []}
    if garment_info.get("garment_name"):
        episode_rec["garment_name"] = garment_info["garment_name"]
    scale = garment_info.get("scale")
    if scale is not None:
        episode_rec["scale"] = np.asarray(scale).tolist() if hasattr(scale, "tolist") else list(scale)
    data[gname][str(episode_index)] = episode_rec
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug("Wrote garment_info entry %s episode %s", gname, episode_index)


def _temp_episode_path(
    video_dir: str | Path, garment: str, seed: int,
    rollout_idx: int | None = None, ep_idx: int | None = None,
) -> Path:
    base = sanitize_basename(garment)
    name = f"{base}_seed{seed}_ep{ep_idx if ep_idx is not None else 0}"
    if rollout_idx is not None:
        name += f"_r{rollout_idx}"
    return Path(video_dir) / TEMP_EPISODES_DIR / f"{name}.pkl"


def _temp_episode_path_with_fallback(
    video_dir: str | Path, garment: str, seed: int,
    rollout_idx: int | None = None, ep_idx: int | None = None,
) -> Path | None:
    """Return path to temp episode file. Tries primary path then lehome-challenge fallback.

    Replay/hard-mining tasks can produce retry attempts; the worker writes
    them as ``..._retry{N}.pkl``. The exact-match path won't see those, so on
    miss we glob for ``..._ep{ep_idx}_retry*.pkl`` and return the most
    recently modified match. This is what carries successful retries
    (early-stopped on first success) into the dataset.
    """
    primary = _temp_episode_path(video_dir, garment, seed, rollout_idx, ep_idx)
    if primary.exists():
        return primary

    base = sanitize_basename(garment)
    ep = ep_idx if ep_idx is not None else 0
    stem = f"{base}_seed{seed}_ep{ep}"
    if rollout_idx is not None:
        stem += f"_r{rollout_idx}"

    # Retry-suffix glob in the primary temp dir.
    try:
        primary_dir = Path(video_dir) / TEMP_EPISODES_DIR
        if primary_dir.exists():
            retries = sorted(
                primary_dir.glob(f"{stem}_retry*.pkl"),
                key=lambda p: p.stat().st_mtime,
            )
            if retries:
                return retries[-1]
    except Exception:
        pass

    # Worker may have written under lehome-challenge (cwd) when video_dir was relative
    try:
        repo_root = Path(__file__).resolve().parents[3]
        run_name = Path(video_dir).name
        fallback_dir = repo_root / "lehome-challenge" / "outputs" / "eval_videos" / run_name / TEMP_EPISODES_DIR
        fallback = fallback_dir / f"{stem}.pkl"
        if fallback.exists():
            return fallback
        if fallback_dir.exists():
            retries = sorted(
                fallback_dir.glob(f"{stem}_retry*.pkl"),
                key=lambda p: p.stat().st_mtime,
            )
            if retries:
                return retries[-1]
    except Exception:
        pass
    return None


class EvalDatasetWriter:
    """Writes eval trajectories to a LeRobot dataset. Main process only.

    When use_value=True, the dataset includes a "success_pred" feature (P(success) per frame).
    Workers must have received "success_pred" from the server and stored it in each trajectory frame.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str = "lehome_eval",
        fps: int = 30,
        features: dict[str, Any] | None = None,
        use_value: bool = False,
        image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
        save_debug_video: bool = True,
        debug_video_camera_size: tuple[int, int] = (240, 320),
        # GAE overlay params (kept in sync with pipeline advantage config so
        # the debug video matches what the trainer actually uses).
        gae_lambda: float = 0.99,
        success_alpha: float = 0.5,
        completion_alpha: float = 0.5,
        value_tail_k: int = 30,
    ):
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = fps
        self.use_value = use_value
        self.image_shape = image_shape
        self.features = features or _eval_features(image_shape)
        self.gae_lambda = gae_lambda
        self.success_alpha = success_alpha
        self.completion_alpha = completion_alpha
        self.value_tail_k = value_tail_k
        if use_value:
            self.features = {
                **self.features,
                "success_pred": VALUE_FEATURE,
                "checkpoint_pred": CHECKPOINT_PRED_FEATURE,
                "success": SUCCESS_FEATURE,
                "advantage": ADVANTAGE_FEATURE,
                "dense_reward": DENSE_REWARD_FEATURE,
                "garment_avg_success": GARMENT_AVG_SUCCESS_FEATURE,
                "garment_avg_checkpoint": GARMENT_AVG_CHECKPOINT_FEATURE,
                "checkpoint_held": CHECKPOINT_HELD_FEATURE,
                "garment_type_pred": {
                    "dtype": "int32",
                    "shape": (1,),
                    "names": ["garment_type_pred"],
                },
                "completion_pred": COMPLETION_PRED_FEATURE,
                "ttc_pred": TTC_PRED_FEATURE,
                "check_distances": CHECK_DISTANCES_FEATURE,
                # Keypoint distance predictions (Head 1) — 21-wide.
                "keypoint_distances_pred": KEYPOINT_DISTANCES_PRED_FEATURE,
                # WM-flow head predictions (Head 3) — cond + uncond pairs.
                "wm_flow_success_cond": WM_FLOW_SCALAR_FEATURE,
                "wm_flow_completion_cond": WM_FLOW_SCALAR_FEATURE,
                "wm_flow_keypoint_cond": WM_FLOW_KEYPOINT_FEATURE,
                "wm_flow_success_uncond": WM_FLOW_SCALAR_FEATURE,
                "wm_flow_completion_uncond": WM_FLOW_SCALAR_FEATURE,
                "wm_flow_keypoint_uncond": WM_FLOW_KEYPOINT_FEATURE,
                # Per-frame Δsuccess target (terminal S − V̂(s_t)). Correlate
                # wm_flow_success_{cond,uncond} against this, NOT raw success,
                # since the head predicts the residual over V̂.
                "success_minus_vhat": WM_FLOW_SCALAR_FEATURE,
                # Best-of-N candidate-spread diagnostics.
                "best_of_n_score_chosen": BON_SCALAR_FEATURE,
                "best_of_n_score_mean": BON_SCALAR_FEATURE,
                "best_of_n_score_min": BON_SCALAR_FEATURE,
                "best_of_n_score_max": BON_SCALAR_FEATURE,
                "best_of_n_score_std": BON_SCALAR_FEATURE,
                "best_of_n_score_spread": BON_SCALAR_FEATURE,
                "best_of_n_n_valid": BON_INT_FEATURE,
            }
        self._dataset: LeRobotDataset | None = None
        self._episode_meta_paths: list[Path] = []
        self.save_debug_video = save_debug_video
        # (h, w) for each camera before composition; the composed grid is
        # (h + h//2, w).  At the default 240x320 the encoded video is 360x320.
        self.debug_video_camera_size = debug_video_camera_size
        # Lazy: only allocated if save_debug_video is True (otherwise no thread pool, no extra threads).
        self._video_pool: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(max_workers=4)
            if save_debug_video else None
        )
        self._video_futures: list[concurrent.futures.Future] = []

    @property
    def dataset(self) -> LeRobotDataset:
        if self._dataset is None:
            raise RuntimeError("Dataset not created; call create() first.")
        return self._dataset

    def create(self) -> EvalDatasetWriter:
        """Create the LeRobot dataset on disk. Call once at start of eval."""
        # Only create parent; LeRobotDataset.create() creates root with exist_ok=False
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.root,
            features=self.features,
            use_videos=True,
            image_writer_threads=0,
            image_writer_processes=0,
            batch_encoding_size=1,
            streaming_encoding=False,
            vcodec=_DATASET_VCODEC,
        )
        (self.root / EVAL_EPISODE_META_DIR).mkdir(parents=True, exist_ok=True)
        logger.info("Eval dataset created at %s (non-streaming, vcodec=%s, crf=%d)",
                    self.root, _DATASET_VCODEC, _DATASET_CRF)
        return self

    def add_episode_from_temp(
        self,
        video_dir: str | Path,
        episode_result: dict[str, Any],
        *,
        task: str | None = None,
        rollout_idx: int | None = None,
        delete_temp: bool = True,
        render_video: bool = True,
    ) -> bool:
        """
        Load trajectory from worker-written temp file and append one episode.

        Only the main process should call this, one episode at a time.
        Temp path is derived from episode_result["garment"], ["seed"], and optional rollout_idx.
        Set delete_temp=False to keep the temp file (e.g. when multiple writers read the same file).

        Returns True if episode was added, False if temp file missing or error.
        """
        video_dir = Path(video_dir)
        garment = episode_result.get("garment", "")
        seed = episode_result.get("seed", 0)
        ep_idx = episode_result.get("ep_idx", None)
        ridx = rollout_idx if rollout_idx is not None else episode_result.get("rollout_idx")
        temp_path = _temp_episode_path_with_fallback(video_dir, garment, seed, ridx, ep_idx)

        if temp_path is None:
            logger.info(
                "No temp episode for %s seed=%s (looked in %s and lehome-challenge fallback)",
                garment, seed, Path(video_dir) / TEMP_EPISODES_DIR,
            )
            return False

        # Per-step memory probes — leak hunting.  Each `_p()` call records the
        # current process RSS so we can see exactly which step (pickle.load,
        # add_frame loop, save_episode, post-del cleanup) is responsible for
        # the growth.  Cheap (one getrusage call per step).
        _probes: list[tuple[str, float]] = []
        def _p(label: str) -> None:
            _probes.append((label, _get_rss_mb()))
        _p("enter")

        try:
            with open(temp_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load temp episode %s: %s", temp_path, e)
            try:
                temp_path.unlink()
            except OSError:
                pass
            return False
        _p("after_pickle_load")

        frames = payload.get("frames")
        if not frames:
            logger.warning("Temp episode %s has no frames", temp_path)
            try:
                temp_path.unlink()
            except OSError:
                pass
            return False

        # Reject episodes where actions are invalid (NaN or extreme values)
        _ACTION_ABS_MAX = 10.0
        has_bad_action = any(
            np.isnan(raw.get("action", np.zeros(1))).any()
            or np.abs(raw.get("action", np.zeros(1))).max() > _ACTION_ABS_MAX
            for raw in frames
        )
        if has_bad_action:
            logger.warning(
                "Temp episode %s has bad actions (NaN or extreme values), skipping",
                temp_path,
            )
            try:
                temp_path.unlink()
            except OSError:
                pass
            return False

        task_str = task or garment or "eval"

        # Compute dense rewards and GAE for video overlay (approximate visualization).
        # Final advantages are recomputed by recompute_advantages.py after collection.
        ep_success = episode_result.get("success", None)
        reward_result = compute_dense_reward(frames, garment_name=garment, success=ep_success)
        if reward_result is not None:
            dense_rewards, checkpoint_held_values = reward_result
        else:
            dense_rewards, checkpoint_held_values = None, None
        # Store dense return on the episode dict so the eval summary can report
        # partial-progress reward statistics (not just binary success).
        if dense_rewards is not None:
            episode_result["dense_return"] = sum(dense_rewards)
            # max_reward: peak cumulative reward (before withdrawal), for checkpoint detection
            cum = 0.0
            max_cum = 0.0
            for r in dense_rewards:
                cum += r
                max_cum = max(max_cum, cum)
            episode_result["max_reward"] = max_cum
        # Render debug video in background thread.  Entirely skipped when
        # save_debug_video is False — no compose, no GAE, no ffmpeg, no thread pool.
        if self.save_debug_video and render_video and frames:
            ep_idx = episode_result.get("ep_idx", 0)
            seed = episode_result.get("seed", 0)
            ridx = rollout_idx if rollout_idx is not None else episode_result.get("rollout_idx")
            vname = sanitize_basename(garment)
            vname += f"_ep{ep_idx:02d}_seed{seed}"
            if ridx is not None:
                vname += f"_r{ridx}"
            vname += ".mp4"
            video_path = Path(video_dir) / vname
            fut = self._video_pool.submit(
                render_episode_video, frames, video_path,
                success=ep_success, dense_rewards=dense_rewards,
                camera_size=self.debug_video_camera_size,
                lam=self.gae_lambda,
                success_alpha=self.success_alpha,
                completion_alpha=self.completion_alpha,
                value_tail_k=self.value_tail_k,
            )
            # Prune completed futures to avoid holding frame data in memory
            self._video_futures = [f for f in self._video_futures if not f.done()]
            self._video_futures.append(fut)

        # Write frames: value=per-frame V(s) prediction, success=binary,
        # advantage=0 placeholder (recompute_advantages.py will overwrite)
        binary_success = float(ep_success) if ep_success is not None else 0.0
        for i, raw in enumerate(frames):
            frame = {
                "observation.state": raw.get("observation.state"),
                "action": raw.get("action"),
                "observation.images.top_rgb": _resize_image(raw.get("observation.images.top_rgb"), self.image_shape),
                "observation.images.left_rgb": _resize_image(raw.get("observation.images.left_rgb"), self.image_shape),
                "observation.images.right_rgb": _resize_image(raw.get("observation.images.right_rgb"), self.image_shape),
                "task": raw.get("task", task_str),
            }
            if self.use_value:
                frame["success_pred"] = np.array([float(raw.get("success_pred", raw.get("value", 0.0)))], dtype=np.float32)
                frame["checkpoint_pred"] = np.array([float(raw.get("checkpoint_pred", raw.get("checkpoint_value", 0.0)))], dtype=np.float32)
                frame["success"] = np.array([binary_success], dtype=np.float32)
                frame["advantage"] = np.array([0.0], dtype=np.float32)
                frame["dense_reward"] = np.array(
                    [dense_rewards[i] if dense_rewards else 0.0], dtype=np.float32
                )
                frame["checkpoint_held"] = np.array(
                    [checkpoint_held_values[i] if checkpoint_held_values else 0.0], dtype=np.float32
                )
                frame["garment_avg_success"] = np.array([0.5], dtype=np.float32)  # placeholder, set by recompute_advantages.py
                frame["garment_avg_checkpoint"] = np.array([0.5], dtype=np.float32)  # placeholder, set by recompute_advantages.py
                frame["garment_type_pred"] = np.array([int(raw.get("garment_type_pred", -1))], dtype=np.int32)
                frame["completion_pred"] = np.array([float(raw.get("completion_pred", 0.0))], dtype=np.float32)
                frame["ttc_pred"] = np.array([float(raw.get("ttc_pred", 0.0))], dtype=np.float32)
                frame["check_distances"] = _pad_check_distances(raw.get("check_distances"))
                # Keypoint + WM-flow predictions (per-chunk, propagated per-frame
                # by eval_worker). NaN-padded when not produced.
                frame["keypoint_distances_pred"] = _fixed_width_pred(
                    raw.get("keypoint_distances_pred"), KEYPOINT_DISTANCES_PRED_WIDTH
                )
                frame["wm_flow_success_cond"] = _scalar_pred(raw.get("wm_flow_success_cond"))
                frame["wm_flow_completion_cond"] = _scalar_pred(raw.get("wm_flow_completion_cond"))
                frame["wm_flow_keypoint_cond"] = _fixed_width_pred(
                    raw.get("wm_flow_keypoint_cond"), KEYPOINT_DISTANCES_PRED_WIDTH
                )
                frame["wm_flow_success_uncond"] = _scalar_pred(raw.get("wm_flow_success_uncond"))
                frame["wm_flow_completion_uncond"] = _scalar_pred(raw.get("wm_flow_completion_uncond"))
                frame["wm_flow_keypoint_uncond"] = _fixed_width_pred(
                    raw.get("wm_flow_keypoint_uncond"), KEYPOINT_DISTANCES_PRED_WIDTH
                )
                _sp_raw = raw.get("success_pred", raw.get("value"))
                if _sp_raw is None or not np.isfinite(float(_sp_raw)):
                    frame["success_minus_vhat"] = np.array([np.nan], dtype=np.float32)
                else:
                    frame["success_minus_vhat"] = np.array(
                        [binary_success - float(_sp_raw)], dtype=np.float32
                    )
                # Best-of-N diagnostics — NaN / 0 when server ran with N=1.
                for _k in (
                    "best_of_n_score_chosen", "best_of_n_score_mean",
                    "best_of_n_score_min", "best_of_n_score_max",
                    "best_of_n_score_std", "best_of_n_score_spread",
                ):
                    frame[_k] = _scalar_pred(raw.get(_k))
                _n_valid_raw = raw.get("best_of_n_n_valid")
                frame["best_of_n_n_valid"] = np.array(
                    [int(_n_valid_raw) if _n_valid_raw is not None else 0],
                    dtype=np.int32,
                )
            self.dataset.add_frame(frame)
        _p("after_add_frame_loop")

        self.dataset.save_episode(parallel_encoding=False)
        _p("after_save_episode")

        ep_idx = self.dataset.meta.total_episodes - 1

        # Save episode metadata (includes garment_type_id, distinct from LeRobot task_index)
        ep_meta = EpisodeMetadata.from_episode_result(episode_result)
        meta_dict = ep_meta.to_dict()
        # Store inference optimization config if present in PKL payload
        if payload.get("inference_config"):
            meta_dict["inference_config"] = payload["inference_config"]
        if payload.get("augmentation_info"):
            meta_dict["augmentation_info"] = payload["augmentation_info"]
        meta_path = self.root / EVAL_EPISODE_META_DIR / f"episode_{ep_idx:06d}.json"
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2)
        self._episode_meta_paths.append(meta_path)

        # Merge garment_info (object_initial_pose, garment_name) into meta/garment_info.json
        if payload.get("garment_info"):
            _merge_garment_info_entry(
                self.root,
                garment,
                ep_idx,
                payload["garment_info"],
            )

        if delete_temp:
            try:
                temp_path.unlink()
            except OSError as e:
                logger.warning("Could not remove temp %s: %s", temp_path, e)

        logger.info("Added episode %s seed=%s to eval dataset (%d frames)", garment, seed, len(frames))
        # Drop frames reference now (the closure of pickle.load result is the
        # only thing that was holding ~1.7 GB of image arrays). The local
        # variable `frames` would normally not be released until function
        # return; explicit `del` lets the next gc.collect() reclaim it.
        del frames
        del payload
        _p("after_del")
        gc.collect()
        _p("after_gc")
        # Print the per-step delta so the leak source is visible at a glance.
        deltas = []
        for i in range(1, len(_probes)):
            label, rss = _probes[i]
            prev_rss = _probes[i - 1][1]
            deltas.append(f"{label}=+{int(rss - prev_rss):d}MB")
        logger.info("[eval] ep=%s rss_steps: enter=%dMB %s",
                    episode_result.get("ep_idx"),
                    int(_probes[0][1]),
                    " ".join(deltas))
        _release_caches_and_log("eval", episode_result.get("ep_idx"))
        return True

    def finalize(self) -> None:
        """Close writers, wait for pending video renders, and write metadata. Call at end of eval."""
        # Wait for all background video renders to complete
        pending = [f for f in self._video_futures if not f.done()]
        if pending:
            logger.info("Waiting for %d video render(s) to finish...", len(pending))
            for f in concurrent.futures.as_completed(pending):
                try:
                    f.result()
                except Exception as e:
                    logger.warning("Video render failed: %s", e)
        self._video_futures.clear()
        if self._video_pool is not None:
            self._video_pool.shutdown(wait=False)

        if self._dataset is not None:
            self._dataset.finalize()
            logger.info("Eval dataset finalized at %s", self.root)
            _assert_dataset_integrity(self._dataset, self.root)

    def write_run_metadata(self, run_metadata: EvalRunMetadata) -> None:
        """Write run-level metadata to meta/eval_run_info.json."""
        path = self.root / EVAL_RUN_INFO
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(run_metadata.to_dict(), f, indent=2)
        logger.info("Eval run metadata written to %s", path)


def _keyframe_features(image_shape: tuple[int, int, int]) -> dict:
    """Build keyframe dataset feature dict for the given image shape."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": list(_JOINT_NAMES),
        },
        **_camera_features(image_shape),
        "source_frame_index": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["index"],
        },
        "success_pred": VALUE_FEATURE,
        "checkpoint_pred": CHECKPOINT_PRED_FEATURE,
    }


class KeyframeDatasetWriter:
    """Writes keyframe-only episodes (frames where infer_chunk was called).

    Stored alongside the full eval dataset at ``{root}_keyframes/``.
    ~20x fewer frames, same video format as the full dataset.
    Enables fast value re-prediction by loading only the frames that
    were actually sent to the policy server.

    Each keyframe episode corresponds 1:1 to a full dataset episode
    (same episode_index).  ``source_frame_index`` maps each keyframe
    back to its position in the full episode.
    """

    def __init__(self, root: str | Path, *, repo_id: str = "lehome_eval_keyframes", fps: int = 30,
                 image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE):
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = fps
        self.image_shape = image_shape
        self._dataset: LeRobotDataset | None = None

    @property
    def dataset(self) -> LeRobotDataset:
        if self._dataset is None:
            raise RuntimeError("Dataset not created; call create() first.")
        return self._dataset

    def create(self) -> "KeyframeDatasetWriter":
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.root,
            features=_keyframe_features(self.image_shape),
            use_videos=True,
            image_writer_threads=0,
            image_writer_processes=0,
            batch_encoding_size=1,
            # Default non-streaming encoding — see EvalDatasetWriter.create.
            streaming_encoding=False,
            vcodec=_DATASET_VCODEC,
        )
        logger.info("Keyframe dataset created at %s (non-streaming, vcodec=%s, crf=%d)",
                    self.root, _DATASET_VCODEC, _DATASET_CRF)
        return self

    def add_episode_from_temp(
        self,
        video_dir: str | Path,
        episode_result: dict[str, Any],
        *,
        task: str | None = None,
        rollout_idx: int | None = None,
    ) -> bool:
        """Filter keyframes from temp pkl and write a mini-episode.

        Returns True if keyframes were added, False if temp missing or no keyframes.
        Does NOT delete the temp file (the main EvalDatasetWriter handles that).
        """
        video_dir = Path(video_dir)
        garment = episode_result.get("garment", "")
        seed = episode_result.get("seed", 0)
        ep_idx = episode_result.get("ep_idx", None)
        ridx = rollout_idx if rollout_idx is not None else episode_result.get("rollout_idx")
        temp_path = _temp_episode_path_with_fallback(video_dir, garment, seed, ridx, ep_idx)

        if temp_path is None:
            return False

        try:
            with open(temp_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("Keyframe: failed to load temp %s: %s", temp_path, e)
            return False

        frames = payload.get("frames")
        if not frames:
            return False

        task_str = task or garment or "eval"

        keyframe_count = 0
        for full_idx, raw in enumerate(frames):
            if not raw.get("is_keyframe"):
                continue
            frame = {
                "observation.state": raw.get("observation.state"),
                "observation.images.top_rgb": _resize_image(raw.get("observation.images.top_rgb"), self.image_shape),
                "observation.images.left_rgb": _resize_image(raw.get("observation.images.left_rgb"), self.image_shape),
                "observation.images.right_rgb": _resize_image(raw.get("observation.images.right_rgb"), self.image_shape),
                "source_frame_index": np.array([full_idx], dtype=np.int32),
                "success_pred": np.array([float(raw.get("success_pred", raw.get("value", 0.0)))], dtype=np.float32),
                "checkpoint_pred": np.array([float(raw.get("checkpoint_pred", raw.get("checkpoint_value", 0.0)))], dtype=np.float32),
                "task": raw.get("task", task_str),
            }
            self.dataset.add_frame(frame)
            keyframe_count += 1

        if keyframe_count == 0:
            return False

        self.dataset.save_episode(parallel_encoding=False)
        logger.info(
            "Keyframe: added %s seed=%s (%d keyframes from %d total frames)",
            garment, seed, keyframe_count, len(frames),
        )
        del frames
        del payload
        _release_caches_and_log("keyframe", episode_result.get("ep_idx"))
        return True

    def finalize(self) -> None:
        if self._dataset is not None:
            self._dataset.finalize()
            logger.info("Keyframe dataset finalized at %s", self.root)
