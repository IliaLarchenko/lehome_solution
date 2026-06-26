"""Real-BC-format dataset writer for sim success replays.

Mirrors the on-disk schema of ``data/lehome_real/four_types_merged/`` so a sim
replay run can be dropped into a real-first training mix without any custom
load path. Differences vs ``EvalDatasetWriter``:

* ``fps=20`` (real cadence) instead of 30.
* Per-camera shapes: ``observation.images.right_front`` is (720, 1280, 3),
  ``left_wrist`` / ``right_wrist`` are (480, 640, 3). EvalDatasetWriter forces
  a single shape across all cams.
* Per-frame state/action units are configurable via ``units``: ``"radians"``
  (default — sim USD radians, unchanged) or ``"degrees"`` (the real-robot
  degree-mode convention: arm joints in degrees, gripper 0-100 — converted
  via ``sim_radians_to_real_units``). Motor-pct is not supported.
* Top camera is rotated 180° at write time so it sits in real-canonical
  orientation (real-first: no letterbox, no pre-rotation cancellation needed
  at training time).
* No aux columns — only the real schema fields (state, action, images, task,
  the LeRobot-stock frame/episode/index metadata).
* Each episode's task string is one of the 4 garment-class names from
  ``scripts/enrich_garment_type.py --domain real``, derived from the sim garment name.
  That maps 1:1 to ``task_index ∈ [0..3]`` and matches the real BC schema
  exactly (real ds also encodes garment via task_index, not a side column).
* Per-episode time-axis stretch: 30 Hz sim frames are cubic-resampled to a
  20 Hz frame grid that doubles the wall-clock duration, matching real teleop
  cadence (~2× slower than sim). State+action use a cubic spline; cameras use
  nearest-neighbor over the same time map.

Only the main process should call ``add_episode_from_temp`` (one episode at
a time), same constraint as EvalDatasetWriter.
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome_solution.eval.dataset_writer import (
    _DATASET_CRF,
    _DATASET_VCODEC,
    _JOINT_NAMES,
    _assert_dataset_integrity,
    _get_rss_mb,
    _release_caches_and_log,
    _temp_episode_path_with_fallback,
    compute_dense_reward,
    render_episode_video,
    sanitize_basename,
)
from lehome_solution.training.real_data_transforms import sim_radians_to_real_units

logger = logging.getLogger(__name__)


# Garment task names — order MUST match scripts/enrich_garment_type.py --domain real
# so task_index in this dataset matches task_index in four_types_merged.
GARMENT_TASKS = ["Top_Long", "Top_Short", "Pant_Long", "Pant_Short"]


# Real BC dataset cadence + per-camera shapes (from
# data/lehome_real/four_types_merged/meta/info.json).
REAL_FPS = 20
SIM_FPS = 30
# Real teleop is ~2× slower than sim. The writer stretches sim wall-clock
# by this factor so the saved dataset has real-style cadence; the per-source
# training-time ``speed_factor`` collapses the stretch back when sampling.
SIM_TO_REAL_TIME_STRETCH = 2.0

# Supported state/action unit conventions for the written dataset.
DATASET_UNITS = ("radians", "degrees")

TOP_SHAPE = (720, 1280, 3)
WRIST_SHAPE = (480, 640, 3)


def _real_features() -> dict[str, Any]:
    """LeRobot feature dict matching four_types_merged/meta/info.json."""
    # Real ds joint names carry the ".pos" suffix used by LeRobot's BiSOFollower.
    # Match exactly so a strict-schema loader treating both datasets as the
    # same source doesn't trip.
    joint_names_real = [f"{n}.pos" for n in _JOINT_NAMES]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": joint_names_real,
        },
        "action": {
            "dtype": "float32",
            "shape": (12,),
            "names": joint_names_real,
        },
        "observation.images.right_front": {
            "dtype": "video",
            "shape": TOP_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.left_wrist": {
            "dtype": "video",
            "shape": WRIST_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": WRIST_SHAPE,
            "names": ["height", "width", "channels"],
        },
    }


def _garment_task(garment_name: str) -> str:
    """Map a sim garment name (e.g. 'Top_Long_Unseen_0') to the 4-class task
    string used by enrich_garment_type.py (--domain real).

    Falls back to ``GARMENT_TASKS[0]`` for unknown names — caller logs at the
    call site, so the writer never silently drops an episode."""
    if not garment_name:
        return GARMENT_TASKS[0]
    prefix = "_".join(garment_name.split("_")[:2])
    for t in GARMENT_TASKS:
        if t == prefix:
            return t
    logger.warning("Unrecognized garment '%s' — defaulting task to %s", garment_name, GARMENT_TASKS[0])
    return GARMENT_TASKS[0]


def _ensure_top_shape(img: np.ndarray) -> np.ndarray:
    """Top cam must be (720, 1280, 3) — raise if not. Strip alpha if present."""
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"top image must be 3-D HWC, got shape {arr.shape}")
    if arr.shape[-1] not in (3, 4):
        raise ValueError(f"top image last axis must be 3 or 4 (RGB[A]), got shape {arr.shape}")
    arr = arr[..., :3]
    if arr.shape[:2] != TOP_SHAPE[:2]:
        raise ValueError(
            f"top image shape {arr.shape} != {TOP_SHAPE} — make sure sim renders top at "
            f"1280x720 (set top_camera_width/top_camera_height in the pipeline config)."
        )
    return arr


def _ensure_wrist_shape(img: np.ndarray, side: str) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"{side}_wrist image must be HWC RGB[A], got shape {arr.shape}")
    arr = arr[..., :3]
    if arr.shape[:2] != WRIST_SHAPE[:2]:
        raise ValueError(
            f"{side}_wrist image shape {arr.shape} != {WRIST_SHAPE}. Check camera_width/height "
            f"in the pipeline config (should be 640x480)."
        )
    return arr


def _rotate_top_180(img: np.ndarray) -> np.ndarray:
    """Sim renders the top cam upside-down vs real. Flip both axes once at
    write time so the saved dataset is in real-canonical orientation."""
    return np.ascontiguousarray(img[::-1, ::-1, :])


def _resample_episode_to_real_grid(
    states: np.ndarray, actions: np.ndarray, frames: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Cubic-resample state+action and nearest-pick camera indices.

    Inputs are length-T sim arrays at 30 Hz. Output is length-T_real arrays
    at 20 Hz that span ``SIM_TO_REAL_TIME_STRETCH × (T / SIM_FPS)`` seconds.

    With the default 2.0× stretch:
        T_real = round(T * SIM_TO_REAL_TIME_STRETCH * REAL_FPS / SIM_FPS)
               = round(T * 4 / 3)

    Returns (states_resampled, actions_resampled, src_idx_per_real_frame).
    The camera index array is nearest-neighbor — image interpolation is bad,
    and 30 Hz adjacent frames are near-identical so the visual jitter is
    sub-pixel.
    """
    T = states.shape[0]
    if T < 2:
        raise ValueError(f"episode too short to resample (T={T})")

    duration_real = (T - 1) * SIM_TO_REAL_TIME_STRETCH / SIM_FPS
    T_real = int(round(duration_real * REAL_FPS)) + 1
    if T_real < 2:
        raise ValueError(f"resampled episode too short (T_real={T_real})")

    # Source anchors at normalized time [0, 1/(T-1), ..., 1]
    src_x = np.linspace(0.0, 1.0, T, dtype=np.float64)
    # Target evaluation at [0, 1/(T_real-1), ..., 1]
    tgt_x = np.linspace(0.0, 1.0, T_real, dtype=np.float64)

    state_spline = CubicSpline(src_x, states.astype(np.float64), axis=0, bc_type="natural")
    action_spline = CubicSpline(src_x, actions.astype(np.float64), axis=0, bc_type="natural")
    states_r = state_spline(tgt_x).astype(np.float32)
    actions_r = action_spline(tgt_x).astype(np.float32)

    # Camera: nearest-neighbor index into the source frame list.
    src_idx = np.clip(np.round(tgt_x * (T - 1)).astype(int), 0, T - 1).tolist()
    if len(frames) != T:
        raise RuntimeError(f"frames len {len(frames)} != state len {T}")
    return states_r, actions_r, src_idx


class RealFormatDatasetWriter:
    """Writes sim success-replay trajectories in the real BC schema.

    Drop-in API mirror of ``EvalDatasetWriter`` so ``run_eval.py`` can swap
    implementations behind a single flag. Only the main process should call
    ``add_episode_from_temp`` (one episode at a time).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str = "lehome_sim_replays_real_format",
        fps: int = REAL_FPS,
        time_stretch: float = SIM_TO_REAL_TIME_STRETCH,
        # State/action units of the written dataset: "radians" (default, sim
        # USD radians pass through unchanged) or "degrees" (real-robot
        # degree-mode convention — use when collecting for a real training mix).
        units: str = "radians",
        # Accepted-and-honored knobs (signature mirrors EvalDatasetWriter).
        use_value: bool = False,
        image_shape: tuple[int, int, int] | None = None,
        save_debug_video: bool = True,
        debug_video_camera_size: tuple[int, int] = (240, 320),
        gae_lambda: float = 0.99,
        success_alpha: float = 0.5,
        completion_alpha: float = 0.5,
        value_tail_k: int = 30,
        features: dict[str, Any] | None = None,
    ):
        del use_value, image_shape
        if fps != REAL_FPS:
            logger.warning("RealFormatDatasetWriter fps=%d (real ds uses %d)", fps, REAL_FPS)
        if units not in DATASET_UNITS:
            raise ValueError(f"units must be one of {DATASET_UNITS}, got {units!r}")
        self.units = units
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = fps
        self.time_stretch = float(time_stretch)
        self.features = features or _real_features()
        self._dataset: LeRobotDataset | None = None
        # Debug-video state: 3-cam composite + GAE overlay, rendered in a
        # background thread (same pattern as EvalDatasetWriter). The composite
        # is built from the RAW sim frames (sim cam keys top_rgb/left_rgb/
        # right_rgb), so it reads the temp pkl directly — unaffected by the
        # real-format conversions we apply for the parquet write.
        self.save_debug_video = save_debug_video
        self.debug_video_camera_size = debug_video_camera_size
        self.gae_lambda = gae_lambda
        self.success_alpha = success_alpha
        self.completion_alpha = completion_alpha
        self.value_tail_k = value_tail_k
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

    def create(self) -> "RealFormatDatasetWriter":
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.root,
            features=self.features,
            use_videos=True,
            robot_type="bi_so_follower",
            image_writer_threads=0,
            image_writer_processes=0,
            batch_encoding_size=1,
            streaming_encoding=False,
            vcodec=_DATASET_VCODEC,
        )
        logger.info(
            "Real-format dataset created at %s (fps=%d, stretch=%.2fx, units=%s, vcodec=%s, crf=%d)",
            self.root, self.fps, self.time_stretch, self.units, _DATASET_VCODEC, _DATASET_CRF,
        )
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
        video_dir = Path(video_dir)
        garment = episode_result.get("garment", "")
        seed = episode_result.get("seed", 0)
        ep_idx = episode_result.get("ep_idx", None)
        ridx = rollout_idx if rollout_idx is not None else episode_result.get("rollout_idx")
        temp_path = _temp_episode_path_with_fallback(video_dir, garment, seed, ridx, ep_idx)

        if temp_path is None:
            logger.info("No temp episode for %s seed=%s", garment, seed)
            return False

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

        frames = payload.get("frames")
        if not frames:
            logger.warning("Temp episode %s has no frames", temp_path)
            try:
                temp_path.unlink()
            except OSError:
                pass
            return False

        # Same NaN / blow-up filter as EvalDatasetWriter.
        _ACTION_ABS_MAX = 10.0
        if any(
            np.isnan(raw.get("action", np.zeros(1))).any()
            or np.abs(raw.get("action", np.zeros(1))).max() > _ACTION_ABS_MAX
            for raw in frames
        ):
            logger.warning("Temp episode %s has bad actions, skipping", temp_path)
            try:
                temp_path.unlink()
            except OSError:
                pass
            return False

        # Debug video: 3-cam composite + GAE overlay (same as EvalDatasetWriter).
        # Runs in a background thread so it doesn't block the dataset write.
        # Submit BEFORE resampling/conversion because render_episode_video
        # reads the RAW sim frames (sim cam keys) directly.
        if self.save_debug_video and render_video and self._video_pool is not None:
            ep_success = episode_result.get("success", None)
            reward_result = compute_dense_reward(frames, garment_name=garment, success=ep_success)
            dense_rewards = reward_result[0] if reward_result is not None else None
            vname = sanitize_basename(garment)
            vname += f"_ep{ep_idx if ep_idx is not None else 0:02d}_seed{seed}"
            if ridx is not None:
                vname += f"_r{ridx}"
            vname += ".mp4"
            video_path = video_dir / vname
            # Top cam is rendered at 1280×720 (16:9). Keep that aspect in the
            # bottom strip of the debug composite — wrists are stretched to
            # match (240×426 layout, total composite 360×426). camera_size
            # stays at the 4:3 wrist default for memory efficiency.
            #
            # Playback rate: SIM_FPS / time_stretch (default 30/2 = 15 fps).
            # Frames are RAW sim 30 Hz; playing them at 15 fps stretches the
            # wall-clock by 2× so the debug video's tempo matches what the
            # real_bc dataset will deliver to training (20 fps × 2× stretch
            # over the same trajectory).
            debug_fps = max(1, int(round(SIM_FPS / max(self.time_stretch, 1e-6))))
            fut = self._video_pool.submit(
                render_episode_video, frames, video_path,
                success=ep_success, dense_rewards=dense_rewards,
                fps=debug_fps,
                camera_size=self.debug_video_camera_size,
                top_camera_size=(240, 426),
                lam=self.gae_lambda,
                success_alpha=self.success_alpha,
                completion_alpha=self.completion_alpha,
                value_tail_k=self.value_tail_k,
            )
            self._video_futures = [f for f in self._video_futures if not f.done()]
            self._video_futures.append(fut)

        # Per-episode collect → unit conversion (optional) → resample → write.
        states = np.stack([np.asarray(raw["observation.state"], dtype=np.float32) for raw in frames])
        actions = np.stack([np.asarray(raw["action"], dtype=np.float32) for raw in frames])
        if self.units == "degrees":
            states = sim_radians_to_real_units(states)
            actions = sim_radians_to_real_units(actions)

        states_r, actions_r, src_idx = _resample_episode_to_real_grid(states, actions, frames)

        task_str = task or _garment_task(garment)
        rss_before = _get_rss_mb()

        for i, k in enumerate(src_idx):
            raw = frames[k]
            top = _ensure_top_shape(raw["observation.images.top_rgb"])
            top = _rotate_top_180(top)
            left = _ensure_wrist_shape(raw["observation.images.left_rgb"], "left")
            right = _ensure_wrist_shape(raw["observation.images.right_rgb"], "right")
            frame = {
                "observation.state": states_r[i],
                "action": actions_r[i],
                "observation.images.right_front": top,
                "observation.images.left_wrist": left,
                "observation.images.right_wrist": right,
                "task": task_str,
            }
            self.dataset.add_frame(frame)

        self.dataset.save_episode(parallel_encoding=False)
        ep_global_idx = self.dataset.meta.total_episodes - 1

        if delete_temp:
            try:
                temp_path.unlink()
            except OSError as e:
                logger.warning("Could not remove temp %s: %s", temp_path, e)

        logger.info(
            "Added real-format episode %d garment=%s seed=%s "
            "(sim_frames=%d → real_frames=%d, rss=%.0fMB)",
            ep_global_idx, garment, seed, len(frames), len(src_idx), rss_before,
        )
        del frames
        del payload
        gc.collect()
        _release_caches_and_log("eval-real", ep_idx)
        return True

    def _canonicalize_task_indices(self) -> None:
        """Remap task_index in tasks.parquet + data + per-episode metadata so
        the index ordering matches scripts/enrich_garment_type.py --domain real:
            Top_Long=0, Top_Short=1, Pant_Long=2, Pant_Short=3.

        LeRobotDataset assigns task_index in first-seen order, which depends
        on which garment the first episode happens to be. We rewrite it here
        so the resulting dataset is bit-compatible with a real BC dataset that
        has gone through enrich_garment_type.py (--domain real) — the training-time
        ``garment_name_to_type()`` lookup will see the same task strings at the
        same indices in both sources.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        tasks_path = self.root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return

        canonical_order = list(GARMENT_TASKS)  # ["Top_Long","Top_Short","Pant_Long","Pant_Short"]
        canonical_idx = {name: i for i, name in enumerate(canonical_order)}

        tasks_tbl = pq.read_table(str(tasks_path))
        tasks_df = tasks_tbl.to_pandas()
        # Real format uses task name as the index, "task_index" as the only column.
        # Detect whichever LeRobot wrote and normalize.
        if "task_index" not in tasks_df.columns:
            logger.warning("tasks.parquet has no task_index column; skipping canonicalize")
            return
        # LeRobot writes tasks.parquet with the task NAME as the pandas index
        # (label varies across versions: "task", None, "__index_level_0__").
        # Treat the index as the name column when no explicit "task" column exists.
        if "task" in tasks_df.columns:
            name_to_old = {str(row["task"]): int(row["task_index"]) for _, row in tasks_df.iterrows()}
        else:
            name_to_old = {str(name): int(row["task_index"]) for name, row in tasks_df.iterrows()}

        # Build old→new map for tasks present in this dataset; tasks not in the
        # canonical list keep their original index (best-effort) but get a
        # warning logged.
        unknown = [n for n in name_to_old if n not in canonical_idx]
        if unknown:
            logger.warning(
                "tasks.parquet has %d task(s) not in canonical order %r: %r — "
                "these are kept with their original indices and may break "
                "compatibility with real BC training.",
                len(unknown), canonical_order, unknown,
            )
        old_to_new: dict[int, int] = {}
        for name, old in name_to_old.items():
            if name in canonical_idx:
                old_to_new[old] = canonical_idx[name]
        # Reassign unknowns to slots above the canonical 0..3.
        next_slot = max([canonical_idx[c] for c in canonical_order]) + 1
        for name, old in name_to_old.items():
            if name not in canonical_idx:
                old_to_new[old] = next_slot
                canonical_idx[name] = next_slot
                next_slot += 1
        needs_remap = not all(old == new for old, new in old_to_new.items())
        if needs_remap:
            logger.info("Remapping task_index: %s", old_to_new)

            # 1. data/*.parquet — rewrite per-frame task_index column.
            for pf in sorted((self.root / "data").rglob("*.parquet")):
                tbl = pq.read_table(str(pf))
                old_idx = tbl["task_index"].to_pylist()
                new_idx = [old_to_new.get(int(v), int(v)) for v in old_idx]
                col_idx = tbl.column_names.index("task_index")
                tbl = tbl.drop("task_index")
                tbl = tbl.add_column(col_idx, "task_index", pa.array(new_idx, type=pa.int64()))
                pq.write_table(tbl, str(pf))

            # 2. meta/episodes/*.parquet — rewrite per-episode `tasks` list[str].
            ep_dir = self.root / "meta" / "episodes"
            for pf in sorted(ep_dir.rglob("*.parquet")):
                tbl = pq.read_table(str(pf))
                ep_df = tbl.to_pandas()
                if "tasks" in ep_df.columns:
                    ep_df["tasks"] = [list(x) if x is not None else x for x in ep_df["tasks"]]
                pq.write_table(pa.Table.from_pandas(ep_df, preserve_index=False), str(pf))
        else:
            logger.info("task_index already in canonical order; skipping data remap.")

        # Always rewrite tasks.parquet to ensure all 4 canonical entries
        # are registered, even when this batch only saw a subset.
        import pandas as pd
        full_canonical = list(canonical_order)
        unknown_tail = [n for n in name_to_old if n not in canonical_order]
        canonical_present = full_canonical + unknown_tail
        for i, n in enumerate(full_canonical):
            canonical_idx.setdefault(n, i)
        new_tasks_df = pd.DataFrame(
            {"task_index": [canonical_idx[n] for n in canonical_present]},
            index=pd.Index(canonical_present, name="task"),
        )
        pq.write_table(pa.Table.from_pandas(new_tasks_df, preserve_index=True), str(tasks_path))

        # 4. meta/info.json — patch total_tasks + robot_type + joint names.
        #    Also patch robot_type + ".pos"-suffixed joint names so info.json
        #    matches what enrich_garment_type.py (--domain real) produces from real data
        #    end-to-end. (For new runs these are set correctly at LeRobotDataset
        #    creation time; this branch only fixes pre-existing datasets where
        #    we want to back-fill the schema.)
        info_path = self.root / "meta" / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            info["total_tasks"] = len(canonical_present)
            if not info.get("robot_type"):
                info["robot_type"] = "bi_so_follower"
            joint_names_real = [f"{n}.pos" for n in _JOINT_NAMES]
            for fname in ("observation.state", "action"):
                feat = info.get("features", {}).get(fname)
                if isinstance(feat, dict) and feat.get("names") and not feat["names"][0].endswith(".pos"):
                    feat["names"] = list(joint_names_real)
            info_path.write_text(json.dumps(info, indent=4))
        logger.info(
            "Canonicalized task_index → %s (total_tasks=%d)",
            {canonical_idx[n]: n for n in canonical_present}, len(canonical_present),
        )

    def finalize(self) -> None:
        # Drain any in-flight debug-video renders before shutting down.
        pending = [f for f in self._video_futures if not f.done()]
        if pending:
            logger.info("Waiting for %d debug video render(s) to finish...", len(pending))
            for f in concurrent.futures.as_completed(pending):
                try:
                    f.result()
                except Exception as e:
                    logger.warning("Debug video render failed: %s", e)
        self._video_futures.clear()
        if self._video_pool is not None:
            self._video_pool.shutdown(wait=False)
        if self._dataset is not None:
            self._dataset.finalize()
            self._canonicalize_task_indices()
            logger.info("Real-format dataset finalized at %s", self.root)
            _assert_dataset_integrity(self._dataset, self.root)

    def write_run_metadata(self, run_metadata: Any) -> None:
        """API parity with EvalDatasetWriter. Real BC schema has no per-run
        metadata file, so we just dump the dict alongside the dataset for
        provenance."""
        path = self.root / "meta" / "sim_replay_run_info.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = run_metadata.to_dict() if hasattr(run_metadata, "to_dict") else dict(run_metadata)
        except Exception:
            payload = {"_unserializable": repr(run_metadata)}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Real-format run metadata written to %s", path)
