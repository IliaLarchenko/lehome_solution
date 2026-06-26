#!/usr/bin/env python3
"""Replay a real-robot BC episode inside Isaac Sim for sim/real alignment validation.

Loads action sequence + initial-state from the real LeRobot dataset, converts
the degree-mode values to sim radians (``real_units_to_sim_radians``) with a cubic
20→30 Hz resample, then drives Isaac Sim with the resulting 30-Hz action
stream and captures the three cameras (top / left / right). The garment is fixed to a default
(``Top_Long_Seen_0``) — we are NOT trying to reproduce the real trajectory; we
just want a visual side-by-side of "what does the same action sequence look
like in sim" vs. "what does it look like on the real robot".

The script wires every production knob it can so the comparison is honest:

* The Isaac Sim run inherits ``augmentation.top_camera_pos_offset`` /
  ``augmentation.top_camera_rot_offset_deg`` from
  ``configs/rl_pipeline_sim_to_real.yaml`` by default (override via ``--top_cam_pos_offset``
  / ``--top_cam_rot_offset_deg`` / ``--no_camera_offset``). All OTHER
  augmentations are disabled (pattern/colour/jitter zeroed; per-step tint off)
  so the comparison isolates the camera framing.
* The real-side 3-cam video pads the 16:9 top frame to 4:3 with a local
  ``letterbox_top_to_4_3`` helper (display-only).
* Real degree-mode states are converted to sim radians via
  ``real_units_to_sim_radians`` (the production conversion).

Outputs (under ``--out_dir``, default ``outputs/replay_real_in_sim/``):
  sim_3cam_30fps.mp4      sim replay, 3 cams stacked at 30 fps
  real_3cam_30fps.mp4     real episode, 3 cams stacked at 30 fps (20→30 dup)
  side_by_side.mp4        sim row over real row (state-diff overlay top-right)
  overlay.mp4             sim alpha-blended onto real (50% transparency)

Run:
    uv run python scripts/replay_real_in_sim.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy.interpolate import CubicSpline

REPO_ROOT = Path(__file__).parent.parent
LEHOME_CHALLENGE_DIR = REPO_ROOT / "lehome-challenge"
LEHOME_VENV_PYTHON = LEHOME_CHALLENGE_DIR / ".venv" / "bin" / "python"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lehome_solution.eval import (  # noqa: E402
    ensure_isaacsim_env,
)
from lehome_solution.eval.remote_command_server import (  # noqa: E402
    RemoteCommandServer,
    decode_observation,
)
from lehome_solution.training.real_data_transforms import (  # noqa: E402
    real_units_to_sim_radians,
)


class _ReplayServer(RemoteCommandServer):
    """Drive the sim through one episode of a fixed 30-Hz action stream.

    Feeds the pre-recorded actions one per ``/infer`` call and records the
    three camera streams + states from each posted observation (replacing the
    old NPZ-in / pickle-out file dance + fork-side recording policy).
    """

    def __init__(self, actions: np.ndarray, garment_name: str, garment_type: str,
                 seed: int):
        super().__init__()
        self._actions = actions
        self._task = {
            "garment_name": garment_name,
            "garment_type": garment_type,
            "seed": seed,
            "stop_on_success": False,  # replay a fixed stream, never early-stop
        }
        self._task_sent = False
        self._step = 0
        self.top: list[np.ndarray] = []
        self.left: list[np.ndarray] = []
        self.right: list[np.ndarray] = []
        self.states: list[np.ndarray] = []

    def handle(self, endpoint: str, body: dict, session_id: str):
        if endpoint == "/next_task":
            if self._task_sent:
                return {"shutdown": True}
            self._task_sent = True
            return self._task
        if endpoint == "/infer":
            self._collect(decode_observation(body))
            if self._step >= len(self._actions):
                return {"stop": True}
            action = self._actions[self._step]
            self._step += 1
            return {"actions": np.asarray(action, dtype=np.float32).tolist()}
        # /reset, /snapshot, /update_check_status need no data back.
        return {}

    def _collect(self, obs: dict):
        for buf, key in (
            (self.top, "observation.images.top_rgb"),
            (self.left, "observation.images.left_rgb"),
            (self.right, "observation.images.right_rgb"),
        ):
            img = obs.get(key)
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                buf.append(np.ascontiguousarray(img).copy())
        st = obs.get("observation.state")
        if st is not None:
            self.states.append(np.asarray(st).copy())


def letterbox_top_to_4_3(img, target_aspect: float = 4 / 3):
    """Pad an (H, W, C) image with white bars on top to ``target_aspect``.

    Display-only helper (the real top cam is 720x1280 = 16:9; sim renders
    4:3). The whole bar goes on top so the table foreground sits low,
    matching the sim bird's-eye baseline.
    """
    h, w, c = img.shape
    if w / h <= target_aspect:
        return img
    fill = 255 if np.issubdtype(img.dtype, np.integer) else 1.0
    new_h = int(round(w / target_aspect))
    out = np.full((new_h, w, c), fill, dtype=img.dtype)
    out[new_h - h:, :, :] = img
    return out

ensure_isaacsim_env(REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NO_OUTPUT_TIMEOUT = 300
SIM_FPS = 30
REAL_FPS = 20
DEFAULT_GARMENT_NAME = "Top_Long_Seen_0"
DEFAULT_GARMENT_TYPE = "top_long"
DEFAULT_PIPELINE_YAML = REPO_ROOT / "configs" / "rl_pipeline_sim_to_real.yaml"

# Output canonical (post-display-rotation) cam resolutions. Sim's launcher emits
# at these dims by default; real frames are resized to match so both rows have
# identical shape and the overlay / vstack are trivial.
DISPLAY_TOP_W = 640
DISPLAY_TOP_H = 480
DISPLAY_WRIST_W = 640
DISPLAY_WRIST_H = 480


def load_real_episode(
    dataset_root: Path, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load (actions, states) for one real episode in native degree-mode units.

    Returns:
        actions: (N, 12) actions at real-time 1/20 … N/20 s.
        states:  (N+1, 12) observation.state at real-time 0, 1/20, …, N/20 s.
                 We include the t=0 anchor explicitly so callers can spline from it.
    """
    pf = sorted((dataset_root / "data").rglob("*.parquet"))[0]
    table = pq.read_table(
        str(pf), columns=["episode_index", "observation.state", "action", "frame_index"]
    ).to_pandas()
    ep_mask = table["episode_index"].values == episode_index
    ep_rows = table[ep_mask].sort_values("frame_index").reset_index(drop=True)
    if len(ep_rows) == 0:
        raise ValueError(f"Episode {episode_index} not found in {pf}")
    actions = np.stack(ep_rows["action"].values).astype(np.float32)
    states = np.stack(ep_rows["observation.state"].values).astype(np.float32)
    return actions, states


def resample_actions_20_to_30(
    actions_deg: np.ndarray, state0_deg: np.ndarray
) -> np.ndarray:
    """Full-episode cubic resample. 20-Hz source → 30-Hz target, in sim radians.

    Source anchors: [state(t=0), action(t=1/20), …, action(t=N/20)] = N+1 points
    spanning [0, N/20] sec. Target: actions at [1/30, 2/30, …, M/30] sec where
    M = floor(N * 30/20). Output stays inside the source span → no extrapolation.

    Input is the native degree-mode convention; converted to sim radians via
    ``real_units_to_sim_radians`` before the spline so we interpolate in the
    action space the sim replay drives.
    """
    n_src_actions = actions_deg.shape[0]
    src_x = np.empty(n_src_actions + 1, dtype=np.float64)
    src_x[0] = 0.0
    src_x[1:] = np.arange(1, n_src_actions + 1) / REAL_FPS
    src_y_deg = np.empty((n_src_actions + 1, actions_deg.shape[1]), dtype=np.float32)
    src_y_deg[0] = state0_deg
    src_y_deg[1:] = actions_deg
    src_y_rad = real_units_to_sim_radians(src_y_deg)

    m = int(np.floor(n_src_actions * SIM_FPS / REAL_FPS))
    target_x = np.arange(1, m + 1, dtype=np.float64) / SIM_FPS
    spline = CubicSpline(src_x, src_y_rad, axis=0, bc_type="natural")
    return spline(target_x).astype(np.float32)


def resample_states_20_to_30(states_deg: np.ndarray) -> np.ndarray:
    """Cubic-resample real observation.state series 20→30 Hz, in sim radians.

    Input: (N+1, 12) states at t = 0, 1/20, …, N/20 s. Output: (M, 12) at
    t = 1/30, 2/30, …, M/30 s where M = floor(N * 30/20). The first sim step
    sees the state AT the first action's effective time (t=1/30), aligning with
    how the replay server records the observation BEFORE acting in step i.
    """
    if states_deg.shape[0] < 2:
        raise ValueError(f"need ≥2 states for spline, got {states_deg.shape}")
    n_actions = states_deg.shape[0] - 1
    src_x = np.arange(0, n_actions + 1, dtype=np.float64) / REAL_FPS
    src_y_rad = real_units_to_sim_radians(states_deg)
    m = int(np.floor(n_actions * SIM_FPS / REAL_FPS))
    target_x = np.arange(1, m + 1, dtype=np.float64) / SIM_FPS
    spline = CubicSpline(src_x, src_y_rad, axis=0, bc_type="natural")
    return spline(target_x).astype(np.float32)


def load_camera_offset_from_yaml(
    yaml_path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Read top-camera offsets + focal scale from a pipeline YAML.

    Returns (pos_offset_xyz_m, rot_offset_euler_deg, focal_scale). Missing keys
    default to zero offsets and focal_scale=1.0. Returns the all-defaults
    triple if the YAML is missing entirely.
    """
    if not yaml_path.exists():
        logger.warning(f"Pipeline YAML {yaml_path} not found; using identity offsets")
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    aug = raw.get("augmentation", {}) or {}
    pos = tuple(aug.get("top_camera_pos_offset", (0.0, 0.0, 0.0)))
    rot = tuple(aug.get("top_camera_rot_offset_deg", (0.0, 0.0, 0.0)))
    focal = float(aug.get("top_camera_focal_scale", 1.0))
    if len(pos) != 3 or len(rot) != 3:
        raise ValueError(
            f"top_camera_*_offset must be 3-vectors; got pos={pos}, rot={rot}"
        )
    return tuple(float(x) for x in pos), tuple(float(x) for x in rot), focal


def launch_sim_replay(
    actions_30hz_rad: np.ndarray,
    out_dir: Path,
    *,
    garment_name: str = DEFAULT_GARMENT_NAME,
    garment_type: str = DEFAULT_GARMENT_TYPE,
    camera_width: int = 640,
    camera_height: int = 480,
    top_camera_width: int | None = None,
    top_camera_height: int | None = None,
    seed: int = 0,
    top_cam_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    top_cam_rot_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    top_cam_focal_scale: float = 1.0,
) -> Path:
    """Run Isaac Sim driven by ``_ReplayServer``, return path to the frames pkl.

    If either offset is non-zero, sets ``LEHOME_GARMENT_AUGMENTATION=1`` and
    feeds the launcher a minimal ``LEHOME_AUG_CONFIG`` that activates ONLY the
    constant top-camera offset (all other knobs zeroed, ``step_color_tint``
    explicitly off so we don't get the default per-step tint).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_pkl = (out_dir / "_replay_frames.pkl").resolve()
    if frames_pkl.exists():
        frames_pkl.unlink()

    # HTTP server that feeds the action stream and records the camera frames.
    # The sim runs the generic `remote` policy and drives this server.
    server = _ReplayServer(actions_30hz_rad, garment_name, garment_type, seed).start()

    eval_env = {
        **os.environ,
        "LEHOME_DISABLE_KEYBOARD": "1",
        "PYTHONUNBUFFERED": "1",
        "LEHOME_NO_DEPTH": "1",
        "LEHOME_CAMERA_WIDTH": str(camera_width),
        "LEHOME_CAMERA_HEIGHT": str(camera_height),
        "LEHOME_REMOTE_URL": server.url,
        "LEHOME_WORKER_LABEL": "W0",
    }
    if top_camera_width is not None and top_camera_height is not None:
        eval_env["LEHOME_TOP_CAMERA_WIDTH"] = str(top_camera_width)
        eval_env["LEHOME_TOP_CAMERA_HEIGHT"] = str(top_camera_height)
        logger.info(
            f"Top camera render override: {top_camera_width}x{top_camera_height} "
            f"(wrist stays at {camera_width}x{camera_height})"
        )
    # Camera offset wiring. Only enable the augmentation pathway when at least
    # one offset is non-zero; otherwise leave it off so the run is identical to
    # the legacy zero-aug baseline (no extra physics step, no augmentation log
    # spam).
    has_offset = (
        any(abs(v) > 0 for v in top_cam_pos_offset)
        or any(abs(v) > 0 for v in top_cam_rot_offset_deg)
        or abs(top_cam_focal_scale - 1.0) > 1e-6
    )
    if has_offset:
        # All other augs zeroed. step_color_tint defaults to True in the fork's
        # visual_augmentation engine — must be explicitly disabled to keep the
        # comparison visually clean.
        aug_cfg = {
            "top_camera_pos_offset": list(top_cam_pos_offset),
            "top_camera_rot_offset_deg": list(top_cam_rot_offset_deg),
            "top_camera_focal_scale": float(top_cam_focal_scale),
            "step_color_tint": False,
        }
        eval_env["LEHOME_GARMENT_AUGMENTATION"] = "1"
        eval_env["LEHOME_AUG_CONFIG"] = json.dumps(aug_cfg)
        logger.info(
            f"Sim camera offset: pos={top_cam_pos_offset}, "
            f"rot_deg={top_cam_rot_offset_deg}, focal_scale={top_cam_focal_scale:.3f}"
        )
    else:
        logger.info("Sim camera offset: disabled (all zeros)")

    sim_cmd = [
        str(LEHOME_VENV_PYTHON), "-u", "-m", "scripts.eval",
        "--policy_type", "remote",
        "--remote_url", server.url,
        "--garment_type", garment_type,
        "--garment_name", garment_name,
        "--num_episodes", "1",
        "--max_steps", str(len(actions_30hz_rad) + 30),
        "--headless",
        "--enable_cameras",
        "--device", "cpu",
        "--seed", str(seed),
        "--same_seed",
        "--camera_width", str(camera_width),
        "--camera_height", str(camera_height),
    ]
    if top_camera_width is not None and top_camera_height is not None:
        sim_cmd += [
            "--top_camera_width", str(top_camera_width),
            "--top_camera_height", str(top_camera_height),
        ]

    log_path = out_dir / "replay_isaac.log"
    log_fh = open(log_path, "w")
    sim_proc = None
    t0 = time.time()
    try:
        sim_proc = subprocess.Popen(
            sim_cmd,
            cwd=str(LEHOME_CHALLENGE_DIR),
            env=eval_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        # Wait for Isaac Sim to come up and finish env init.
        start_wait = time.time()
        while time.time() - start_wait < 240:
            ready, _, _ = select.select([sim_proc.stdout], [], [], 1.0)
            if ready:
                line = sim_proc.stdout.readline()
                if not line:
                    break
                log_fh.write(line)
                log_fh.flush()
                if "Evaluating:" in line or "remote-driven" in line:
                    break

        # Run loop until process exits or output stalls.
        while True:
            ready, _, _ = select.select([sim_proc.stdout], [], [], NO_OUTPUT_TIMEOUT)
            if not ready:
                logger.warning(f"No output for {NO_OUTPUT_TIMEOUT}s, killing")
                break
            line = sim_proc.stdout.readline()
            if not line:
                break
            log_fh.write(line)
            log_fh.flush()
            if "Episode " in line and "Return=" in line:
                logger.info(line.strip())
        sim_proc.wait(timeout=60)
        logger.info(f"Isaac Sim exited (rc={sim_proc.returncode}) in {time.time() - t0:.0f}s")
    finally:
        if sim_proc and sim_proc.poll() is None:
            try:
                os.killpg(os.getpgid(sim_proc.pid), 9)
            except Exception:
                pass
        log_fh.close()
        server.stop()

    # Write the frames the server recorded (same schema the renderers read).
    if not server.top and not server.states:
        raise RuntimeError(f"Replay recorded no frames — see {log_path}")
    with open(frames_pkl, "wb") as f:
        pickle.dump(
            {
                "top": server.top,
                "left": server.left,
                "right": server.right,
                "states": server.states,
                "num_steps": len(server.states),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logger.info(
        f"Replay recorded {len(server.top)}/{len(server.left)}/{len(server.right)} "
        f"top/L/R frames -> {frames_pkl}"
    )
    return frames_pkl


# ---------------------------------------------------------------------------
# Video compositing
# ---------------------------------------------------------------------------
def _to_display_top(img: np.ndarray, *, was_pre_rotated: bool) -> np.ndarray:
    """Resize a (possibly letterboxed) top frame to DISPLAY_TOP_H × DISPLAY_TOP_W.

    The input may already be 4:3 (sim native, or real after letterbox) or
    pre-rotated upside-down. We undo the pre-rotation so the displayed frame is
    canonical (right-side-up), which is what humans expect to see.
    """
    import cv2

    img = img[..., :3]
    if was_pre_rotated:
        img = np.ascontiguousarray(img[::-1, ::-1, :])
    if img.shape[0] != DISPLAY_TOP_H or img.shape[1] != DISPLAY_TOP_W:
        img = cv2.resize(
            img, (DISPLAY_TOP_W, DISPLAY_TOP_H), interpolation=cv2.INTER_AREA
        )
    return img


def _to_display_wrist(img: np.ndarray) -> np.ndarray:
    import cv2

    img = img[..., :3]
    if img.shape[0] != DISPLAY_WRIST_H or img.shape[1] != DISPLAY_WRIST_W:
        img = cv2.resize(
            img, (DISPLAY_WRIST_W, DISPLAY_WRIST_H), interpolation=cv2.INTER_AREA
        )
    return img


def _draw_state_diff(img: np.ndarray, diff: np.ndarray | None) -> np.ndarray:
    """Overlay 12-dim state-diff text in the top-right corner of an RGB frame.

    Format: two columns × 6 rows (left arm above the gap, right arm below — same
    layout as the 12-vector ordering [L0..L5, R0..R5]). Each cell renders as
    ``J0: +0.123`` where the number is in sim radians. Returns a new array.
    """
    import cv2

    if diff is None:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    arr = np.asarray(diff).reshape(-1)
    if arr.size != 12:
        return out
    # Layout: top-right corner, two columns side-by-side, six rows.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    line_h = 14
    cell_w = 90
    pad = 4
    total_w = cell_w * 2 + pad * 3
    total_h = line_h * 6 + pad * 2
    x0 = w - total_w - 6
    y0 = 6
    # Semi-transparent black background panel for readability.
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + total_w, y0 + total_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0)
    labels = ("J0", "J1", "J2", "J3", "J4", "J5")
    for row in range(6):
        ly = y0 + pad + line_h * (row + 1) - 3
        # Left arm column
        v_l = float(arr[row])
        cv2.putText(
            out,
            f"L{labels[row]}:{v_l:+.2f}",
            (x0 + pad, ly),
            font, scale, (0, 255, 0), thickness, cv2.LINE_AA,
        )
        # Right arm column
        v_r = float(arr[6 + row])
        cv2.putText(
            out,
            f"R{labels[row]}:{v_r:+.2f}",
            (x0 + pad + cell_w + pad, ly),
            font, scale, (255, 255, 0), thickness, cv2.LINE_AA,
        )
    return out


def _write_3cam_video(
    top: list[np.ndarray],
    left: list[np.ndarray],
    right: list[np.ndarray],
    *,
    output_path: Path,
    fps: int,
    rotate_top_180: bool,
    letterbox_top: bool,
    label: str,
    state_diffs: np.ndarray | None = None,
) -> int:
    """Write a 3-cam side-by-side mp4 (libx264, yuv420p) at fixed display res.

    All output rows are ``DISPLAY_TOP_H × (DISPLAY_TOP_W + 2*DISPLAY_WRIST_W)``
    so sim and real videos are byte-aligned for vstack / overlay.

    Args:
        top/left/right: per-frame RGB arrays. ``top`` may be 16:9 (real native);
            set ``letterbox_top=True`` to pad to 4:3 via the production helper.
            ``rotate_top_180=True`` is for sim's raw upside-down convention —
            we ALWAYS display canonical (right-side-up).
        state_diffs: optional (N, 12) per-frame state diff to overlay top-right.
    """
    if not top:
        raise RuntimeError(f"No top frames to write for {label}")
    n = min(len(top), len(left), len(right))
    if state_diffs is not None:
        n = min(n, state_diffs.shape[0])
    if n != len(top):
        logger.warning(
            f"{label}: trimming to {n} frames "
            f"(top={len(top)}, L={len(left)}, R={len(right)}"
            + (f", diffs={state_diffs.shape[0]}" if state_diffs is not None else "")
            + ")"
        )

    row_w = DISPLAY_TOP_W + DISPLAY_WRIST_W + DISPLAY_WRIST_W
    row_h = DISPLAY_TOP_H
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="replay_compose_"))
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{row_w}x{row_h}", "-r", str(fps),
            "-i", "pipe:",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for i in range(n):
            t = top[i]
            if t.ndim == 3 and t.shape[2] == 4:
                t = t[:, :, :3]
            if letterbox_top:
                # Production helper: pad 16:9 → 4:3 with white bar on top.
                t = letterbox_top_to_4_3(t)
            # ALL frames displayed canonical. Sim raw is upside-down; real after
            # letterbox is canonical. If the caller pre-rotated the top frames
            # 180°, pass rotate_top_180=True so we undo it.
            t = _to_display_top(t, was_pre_rotated=rotate_top_180)
            l = _to_display_wrist(left[i])
            r = _to_display_wrist(right[i])
            row = np.concatenate([t, l, r], axis=1)
            diff = state_diffs[i] if state_diffs is not None else None
            row = _draw_state_diff(row, diff)
            proc.stdin.write(row.astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    logger.info(f"Wrote {label} -> {output_path} ({n} frames @ {fps} fps, {row_w}x{row_h})")
    return n


def render_sim_3cam_video(
    frames_pkl: Path,
    output_path: Path,
    *,
    state_diffs: np.ndarray | None = None,
) -> int:
    """Compose sim 3-cam at 30 fps. Top is upside down in sim → rotate 180°."""
    with open(frames_pkl, "rb") as f:
        d = pickle.load(f)
    return _write_3cam_video(
        d["top"], d["left"], d["right"],
        output_path=output_path,
        fps=SIM_FPS,
        rotate_top_180=True,
        letterbox_top=False,
        label="sim 3-cam",
        state_diffs=state_diffs,
    )


def render_real_3cam_video(
    real_root: Path,
    episode_index: int,
    output_path: Path,
    *,
    state_diffs: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Decode real episode's 3 cameras (right_front + left/right wrist) and compose
    at sim's 30 fps. Real is stored at 20 fps; we use ffmpeg's fps filter to
    duplicate frames so wall-clock duration matches sim playback.

    The top frame goes through ``letterbox_top_to_4_3`` before display resize.
    """
    ep_meta_pf = real_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_meta = pq.read_table(str(ep_meta_pf)).to_pandas()
    row = ep_meta[ep_meta["episode_index"] == episode_index].iloc[0]

    cam_names = [
        "observation.images.right_front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    decoded = {}
    raw_native_top_hw: tuple[int, int] | None = None
    for cam in cam_names:
        ck = int(row[f"videos/{cam}/chunk_index"])
        fi = int(row[f"videos/{cam}/file_index"])
        from_ts = float(row[f"videos/{cam}/from_timestamp"])
        to_ts = float(row[f"videos/{cam}/to_timestamp"])
        mp4 = real_root / "videos" / cam / f"chunk-{ck:03d}" / f"file-{fi:03d}.mp4"
        dur = to_ts - from_ts
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{from_ts:.6f}",
                "-i", str(mp4),
                "-t", f"{dur:.6f}",
                "-vf", f"fps={SIM_FPS}",
                "-c:v", "rawvideo", "-pix_fmt", "rgb24",
                "-f", "rawvideo",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "default=nw=1", str(mp4)],
                capture_output=True, text=True,
            )
            wh = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in probe.stdout.strip().splitlines()}
            w, h = wh["width"], wh["height"]
            raw = np.frombuffer(tmp_path.read_bytes(), dtype=np.uint8).reshape(-1, h, w, 3)
            decoded[cam] = raw.copy()
            if cam.endswith("right_front"):
                raw_native_top_hw = (h, w)
        finally:
            tmp_path.unlink(missing_ok=True)

    n_top = decoded["observation.images.right_front"].shape[0]
    n_left = decoded["observation.images.left_wrist"].shape[0]
    n_right = decoded["observation.images.right_wrist"].shape[0]
    logger.info(
        f"Decoded real cams @ {SIM_FPS} fps: top={n_top}, L={n_left}, R={n_right}; "
        f"native top hw={raw_native_top_hw}"
    )
    top = list(decoded["observation.images.right_front"])
    left = list(decoded["observation.images.left_wrist"])
    right = list(decoded["observation.images.right_wrist"])
    n_written = _write_3cam_video(
        top, left, right,
        output_path=output_path,
        fps=SIM_FPS,
        rotate_top_180=False,   # real top is already canonical, no rotation
        letterbox_top=True,     # 16:9 → 4:3 via production helper
        label="real 3-cam (20→30 fps)",
        state_diffs=state_diffs,
    )
    return n_top, n_left, n_written


def _decode_real_top_native(
    real_root: Path, episode_index: int, target_fps: int = SIM_FPS,
) -> np.ndarray:
    """Decode the real top cam (right_front) at native 720×1280 resolution,
    resampled to ``target_fps`` via ffmpeg's fps filter (20→30 = duplicate).

    Returns (N, 720, 1280, 3) uint8 RGB. Real top is already canonical, no
    rotation or letterbox needed — this is what gets written to the 16:9
    comparison videos.
    """
    ep_meta_pf = real_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_meta = pq.read_table(str(ep_meta_pf)).to_pandas()
    row = ep_meta[ep_meta["episode_index"] == episode_index].iloc[0]

    cam = "observation.images.right_front"
    ck = int(row[f"videos/{cam}/chunk_index"])
    fi = int(row[f"videos/{cam}/file_index"])
    from_ts = float(row[f"videos/{cam}/from_timestamp"])
    to_ts = float(row[f"videos/{cam}/to_timestamp"])
    mp4 = real_root / "videos" / cam / f"chunk-{ck:03d}" / f"file-{fi:03d}.mp4"
    dur = to_ts - from_ts
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{from_ts:.6f}",
                "-i", str(mp4),
                "-t", f"{dur:.6f}",
                "-vf", f"fps={target_fps}",
                "-c:v", "rawvideo", "-pix_fmt", "rgb24",
                "-f", "rawvideo",
                str(tmp_path),
            ],
            check=True, capture_output=True,
        )
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "default=nw=1", str(mp4)],
            capture_output=True, text=True,
        )
        wh = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in probe.stdout.strip().splitlines()}
        w, h = wh["width"], wh["height"]
        raw = np.frombuffer(tmp_path.read_bytes(), dtype=np.uint8).reshape(-1, h, w, 3).copy()
    finally:
        tmp_path.unlink(missing_ok=True)
    return raw


def render_top_16_9_outputs(
    frames_pkl: Path,
    real_root: Path,
    episode_index: int,
    out_dir: Path,
    *,
    overlay_alpha: float = 0.5,
    state_diffs: np.ndarray | None = None,
) -> dict[str, Path]:
    """Top-cam-only 16:9 outputs at native (720, 1280, 3).

    Sim top is upside-down in raw capture → rotated 180° to canonical.
    Real top is already canonical → used as-is. NO letterbox / NO 4:3 pad.
    Both are written at exactly the real BC dataset shape so the overlay /
    side-by-side videos verify framing & focal length match real for the
    real-format writer's outbound resolution.

    Returns mapping {kind: path}.
    """
    import cv2

    with open(frames_pkl, "rb") as f:
        d = pickle.load(f)
    sim_top_raw = d["top"]
    if not sim_top_raw:
        raise RuntimeError("no sim top frames in pkl")
    real_top = _decode_real_top_native(real_root, episode_index, target_fps=SIM_FPS)

    n = min(len(sim_top_raw), real_top.shape[0])
    if state_diffs is not None:
        n = min(n, state_diffs.shape[0])
    H, W = 720, 1280  # real BC right_front shape

    def _canonical_sim(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)[..., :3]
        # Sim renders upside-down; rotate to canonical (matches real).
        arr = np.ascontiguousarray(arr[::-1, ::-1, :])
        if arr.shape[:2] != (H, W):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
        return arr

    def _ensure_real(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)[..., :3]
        if arr.shape[:2] != (H, W):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
        return arr

    out_dir.mkdir(parents=True, exist_ok=True)
    sim_only = out_dir / "top_16_9_sim.mp4"
    real_only = out_dir / "top_16_9_real.mp4"
    side_by_side = out_dir / "top_16_9_side_by_side.mp4"
    overlay = out_dir / "top_16_9_overlay.mp4"

    def _open_writer(path: Path, w: int, h: int) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(SIM_FPS),
                "-i", "pipe:",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    sim_w = _open_writer(sim_only, W, H)
    real_w = _open_writer(real_only, W, H)
    sbs_w = _open_writer(side_by_side, W, H * 2 + 8)  # 8-px white separator
    ovl_w = _open_writer(overlay, W, H)
    sep = np.full((8, W, 3), 255, dtype=np.uint8)
    a = float(overlay_alpha)

    try:
        for i in range(n):
            s = _canonical_sim(sim_top_raw[i])
            r = _ensure_real(real_top[i])
            diff = state_diffs[i] if state_diffs is not None else None
            s_overlaid = _draw_state_diff(s, diff)
            r_overlaid = _draw_state_diff(r, diff)
            sim_w.stdin.write(s_overlaid.astype(np.uint8).tobytes())
            real_w.stdin.write(r_overlaid.astype(np.uint8).tobytes())
            stacked = np.concatenate([s_overlaid, sep, r_overlaid], axis=0)
            sbs_w.stdin.write(stacked.astype(np.uint8).tobytes())
            blend = (a * s.astype(np.float32) + (1.0 - a) * r.astype(np.float32)).clip(0, 255).astype(np.uint8)
            blend = _draw_state_diff(blend, diff)
            ovl_w.stdin.write(blend.tobytes())
        for w in (sim_w, real_w, sbs_w, ovl_w):
            w.stdin.close()
            w.wait()
    finally:
        for w in (sim_w, real_w, sbs_w, ovl_w):
            if w.poll() is None:
                w.terminate()

    logger.info(
        f"Wrote 16:9 top-only outputs ({n} frames @ {SIM_FPS} fps, {W}×{H} native): "
        f"{sim_only.name}, {real_only.name}, {side_by_side.name}, {overlay.name}"
    )
    return {"sim": sim_only, "real": real_only, "side_by_side": side_by_side, "overlay": overlay}


def render_side_by_side(sim_path: Path, real_path: Path, output_path: Path) -> None:
    """Stack sim row over real row vertically into a single composite mp4.

    Both inputs are now produced at identical resolution by ``_write_3cam_video``
    so this is a pure vstack — no scaling needed.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(sim_path),
        "-i", str(real_path),
        "-filter_complex",
        "[0:v]pad=iw:ih+8:0:0:white[s];[s][1:v]vstack=inputs=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Wrote side-by-side -> {output_path}")


def render_overlay(
    sim_path: Path,
    real_path: Path,
    output_path: Path,
    alpha: float = 0.5,
) -> None:
    """Alpha-blend sim over real. Single 3-cam strip at the shared display res.

    ``alpha`` is the SIM weight (1 - alpha is real). Both inputs are already at
    the same dimensions so the ``blend`` filter is direct.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(sim_path),
        "-i", str(real_path),
        "-filter_complex",
        f"[0:v][1:v]blend=all_mode=normal:all_opacity={alpha:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Wrote overlay -> {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real_root", type=Path, default=Path("data/lehome_real/four_types_merged"))
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--garment_name", default=DEFAULT_GARMENT_NAME)
    ap.add_argument("--garment_type", default=DEFAULT_GARMENT_TYPE)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/replay_real_in_sim"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera_width", type=int, default=DISPLAY_TOP_W)
    ap.add_argument("--camera_height", type=int, default=DISPLAY_TOP_H)
    ap.add_argument(
        "--top_camera_width", type=int, default=None,
        help="Override TOP cam render width (real BC = 1280). Wrists keep --camera_width. "
             "Sim still renders at this res; the comparison video is downscaled to "
             f"{DISPLAY_TOP_W}x{DISPLAY_TOP_H} for vstack/overlay.",
    )
    ap.add_argument(
        "--top_camera_height", type=int, default=None,
        help="Override TOP cam render height (real BC = 720).",
    )
    ap.add_argument(
        "--config_yaml", type=Path, default=DEFAULT_PIPELINE_YAML,
        help="Pipeline YAML to source augmentation.top_camera_*_offset defaults from.",
    )
    ap.add_argument(
        "--top_cam_pos_offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="Override top-camera pos offset in meters (base-local frame). Overrides yaml.",
    )
    ap.add_argument(
        "--top_cam_rot_offset_deg", type=float, nargs=3, default=None, metavar=("RX", "RY", "RZ"),
        help="Override top-camera rotation offset in degrees. Overrides yaml.",
    )
    ap.add_argument(
        "--top_cam_focal_scale", type=float, default=None,
        help="Multiplier on top-camera focal length (1.0 = no change). Defaults "
             "to the yaml's augmentation.top_camera_focal_scale (or 1.0 if absent).",
    )
    ap.add_argument(
        "--no_camera_offset", action="store_true",
        help="Force zero top-camera offset (ignores yaml and --top_cam_*_offset).",
    )
    ap.add_argument(
        "--no_state_diff_overlay", action="store_true",
        help="Skip the 12-dim state-diff text overlay in the top-right corner.",
    )
    ap.add_argument(
        "--max_frames", type=int, default=None,
        help="Clip the action stream to the first N frames (for fast sweeps).",
    )
    ap.add_argument(
        "--overlay_alpha", type=float, default=0.5,
        help="Sim opacity in the overlay video (0=real only, 1=sim only).",
    )
    ap.add_argument("--skip_sim", action="store_true",
                    help="Skip Isaac Sim run (use pre-existing _replay_frames.pkl in out_dir)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve top-camera offset (CLI > yaml > zero). ---
    if args.no_camera_offset:
        top_pos_off = (0.0, 0.0, 0.0)
        top_rot_off = (0.0, 0.0, 0.0)
        top_focal_scale = 1.0
    else:
        yaml_pos, yaml_rot, yaml_focal = load_camera_offset_from_yaml(args.config_yaml)
        top_pos_off = tuple(args.top_cam_pos_offset) if args.top_cam_pos_offset is not None else yaml_pos
        top_rot_off = tuple(args.top_cam_rot_offset_deg) if args.top_cam_rot_offset_deg is not None else yaml_rot
        top_focal_scale = args.top_cam_focal_scale if args.top_cam_focal_scale is not None else yaml_focal

    actions_deg, states_deg = load_real_episode(args.real_root, args.episode_index)
    state0_deg = states_deg[0]
    logger.info(
        f"Loaded real ep {args.episode_index}: {actions_deg.shape[0]} actions @ {REAL_FPS} fps "
        f"({actions_deg.shape[0] / REAL_FPS:.2f} s); "
        f"degree range [{actions_deg.min():.1f}, {actions_deg.max():.1f}]"
    )
    actions_rad_30 = resample_actions_20_to_30(actions_deg, state0_deg)
    real_states_rad_30 = resample_states_20_to_30(states_deg)
    if args.max_frames is not None and args.max_frames < actions_rad_30.shape[0]:
        logger.info(f"Clipping action stream {actions_rad_30.shape[0]} -> {args.max_frames}")
        actions_rad_30 = actions_rad_30[: args.max_frames]
        real_states_rad_30 = real_states_rad_30[: args.max_frames]
    logger.info(
        f"Resampled to {actions_rad_30.shape[0]} actions @ {SIM_FPS} fps "
        f"({actions_rad_30.shape[0] / SIM_FPS:.2f} s) — "
        f"range [{actions_rad_30.min():.3f}, {actions_rad_30.max():.3f}] rad"
    )

    frames_pkl = args.out_dir / "_replay_frames.pkl"
    if not args.skip_sim:
        frames_pkl = launch_sim_replay(
            actions_rad_30,
            args.out_dir,
            garment_name=args.garment_name,
            garment_type=args.garment_type,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            top_camera_width=args.top_camera_width,
            top_camera_height=args.top_camera_height,
            seed=args.seed,
            top_cam_pos_offset=top_pos_off,
            top_cam_rot_offset_deg=top_rot_off,
            top_cam_focal_scale=top_focal_scale,
        )

    # --- Build state-diff array (sim - real, in sim radians). ---
    state_diffs: np.ndarray | None = None
    if not args.no_state_diff_overlay:
        with open(frames_pkl, "rb") as f:
            d = pickle.load(f)
        sim_states = np.asarray(d["states"], dtype=np.float32)  # (n_sim, 12)
        n = min(sim_states.shape[0], real_states_rad_30.shape[0])
        if n > 0 and sim_states.shape[1] == 12 and real_states_rad_30.shape[1] == 12:
            state_diffs = sim_states[:n] - real_states_rad_30[:n]
            logger.info(
                f"State-diff overlay: {n} frames; "
                f"|diff| max={np.abs(state_diffs).max():.3f}, "
                f"mean={np.abs(state_diffs).mean():.3f} rad"
            )
        else:
            logger.warning(
                f"Skipping state-diff overlay: sim_states.shape={sim_states.shape}, "
                f"real_states.shape={real_states_rad_30.shape}"
            )

    sim_video = args.out_dir / "sim_3cam_30fps.mp4"
    real_video = args.out_dir / "real_3cam_30fps.mp4"
    side_by_side = args.out_dir / "side_by_side.mp4"
    overlay_video = args.out_dir / "overlay.mp4"

    sim_n = render_sim_3cam_video(frames_pkl, sim_video, state_diffs=state_diffs)
    _, _, real_n = render_real_3cam_video(
        args.real_root, args.episode_index, real_video, state_diffs=state_diffs
    )
    logger.info(f"Frame counts: sim={sim_n}, real(written)={real_n}")
    render_side_by_side(sim_video, real_video, side_by_side)
    render_overlay(sim_video, real_video, overlay_video, alpha=args.overlay_alpha)

    # Top-only 16:9 outputs at real-format native shape (720, 1280).
    # These are the primary verification target for the real-format writer:
    # no letterbox, no 4:3 squash, sim rotated 180° to canonical so both rows
    # are oriented the same way as real BC `right_front`.
    top_16_9 = render_top_16_9_outputs(
        frames_pkl, args.real_root, args.episode_index,
        args.out_dir, overlay_alpha=args.overlay_alpha,
        state_diffs=state_diffs,
    )

    print()
    print(f"  sim           : {sim_video}")
    print(f"  real          : {real_video}")
    print(f"  side-by-side  : {side_by_side}")
    print(f"  overlay       : {overlay_video}")
    print()
    print(f"  16:9 sim top  : {top_16_9['sim']}")
    print(f"  16:9 real top : {top_16_9['real']}")
    print(f"  16:9 side     : {top_16_9['side_by_side']}")
    print(f"  16:9 overlay  : {top_16_9['overlay']}")


if __name__ == "__main__":
    main()
