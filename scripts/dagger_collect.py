#!/usr/bin/env python3
"""DAgger-style recovery data collection from failure states.

Architecture:
    SO101 Reader (challenge venv) --stdout--> Controller (main venv) --WebSocket--> Isaac Sim (challenge venv)

Loads failure NPZ files (saved during eval rollouts), restores physics state in
Isaac Sim, lets the user teleoperate with SO101 leaders to recover from failures.
Successful recoveries are saved as training data in the same PKL format as eval.

Usage:
    # Config-driven (pulls failure states + dagger data from HF, uploads results):
    uv run python scripts/dagger_collect.py --config configs/rl_pipeline_sim.yaml

    # Manual mode (local failure dir + output dir):
    uv run python scripts/dagger_collect.py \\
        --failure_dir outputs/eval_videos/rl_XXX/physics_states/failure \\
        --output_dir outputs/dagger_episodes/session_001
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import pickle
import queue
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

import cv2
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
LEHOME_CHALLENGE_DIR = REPO_ROOT / "lehome-challenge"
LEHOME_VENV_PYTHON = LEHOME_CHALLENGE_DIR / ".venv" / "bin" / "python"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arm_viz import draw_arm_side_view, JOINT_LIMITS_RAD as ARM_JOINT_LIMITS_RAD
from lehome_solution.constants import GARMENT_TYPE_TO_ID
from lehome_solution.eval import (
    ensure_isaacsim_env,
    garment_name_to_type,
)
from lehome_solution.eval.dataset_writer import (
    EvalDatasetWriter,
    KeyframeDatasetWriter,
    compute_dense_reward,
    render_episode_video,
    sanitize_basename,
    _resize_image,
)
from lehome_solution.eval.physics_states import save_physics_state_npz
from lehome_solution.eval.remote_command_server import RemoteCommandServer

ensure_isaacsim_env(REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Boot deadline for wait_for_sim_ready. A healthy boot reaches the ready
# marker 30-60s after launch; a wedged Isaac never reaches it at all, so a
# generous deadline only delays the reboot retry.
SIM_BOOT_TIMEOUT = 150
# Deadline for a single restore (env.reset + settle + particle restore). A
# healthy restore finishes in 5-30s; a wedged sim never finishes.
RESTORE_TIMEOUT = 120
# A sim whose HTTP long-poll sits idle for minutes can abort (native abort ->
# IsaacLab's recursive close handler) the moment it is next touched. Feed every
# waiting sim a hold-position step at this interval so no poll goes stale.
KEEPALIVE_IDLE = 20.0


def log_proc_diag(pid: int, label: str):
    """Log kernel-side thread states of a stuck process (no ptrace needed).

    For a hang with idle CPU this tells us WHAT it is blocked on:
    futex_wait = deadlock, poll/select/sock = network or pipe I/O,
    io_schedule = disk. One line per non-sleeping/interesting thread.
    """
    import glob as _glob
    try:
        rows = []
        for tdir in sorted(_glob.glob(f"/proc/{pid}/task/*"))[:64]:
            tid = tdir.rsplit("/", 1)[1]
            try:
                comm = open(f"{tdir}/comm").read().strip()
                wchan = open(f"{tdir}/wchan").read().strip() or "-"
                state = open(f"{tdir}/stat").read().split(")")[-1].split()[0]
            except OSError:
                continue
            rows.append((tid, comm, state, wchan))
        main = [r for r in rows if r[0] == str(pid)]
        interesting = [r for r in rows
                       if r[2] not in ("S",) or r[3] not in ("-", "futex_wait_queue")]
        logger.warning("[diag %s] pid=%d threads=%d", label, pid, len(rows))
        for tid, comm, state, wchan in (main + interesting)[:20]:
            logger.warning("[diag %s]   tid=%s comm=%-18s state=%s wchan=%s",
                           label, tid, comm, state, wchan)
        # Histogram of wchans across all threads — the blocked majority shows here
        from collections import Counter
        hist = Counter(f"{r[2]}:{r[3]}" for r in rows)
        logger.warning("[diag %s]   wchan histogram: %s", label,
                       dict(hist.most_common(8)))
    except Exception as e:
        logger.warning("[diag %s] failed: %s", label, e)
# How many failure states to try before giving up on starting a session. A
# corrupt/mismatched NPZ should cost one skip, not the whole run.
_FIRST_RESTORE_ATTEMPTS = 5
# Reboot retries for a sim that wedges during Isaac startup (nondeterministic;
# consecutive wedges happen). Each failed attempt costs SIM_BOOT_TIMEOUT.
_SIM_START_ATTEMPTS = 4
DAGGER_BASE_DIR = REPO_ROOT / "outputs" / "dagger_episodes"
DAGGER_HF_PREFIX = "dagger"  # folder prefix in HF dataset repo


def build_dagger_frame(
    raw: dict,
    *,
    writer_image_shape: tuple[int, int, int],
    default_task: str,
    binary_success: float,
    garment_type_id: int,
    dense_reward: float,
    checkpoint_held: float,
) -> dict:
    """Build a LeRobot frame dict for a dagger episode.

    Must exactly match the schema of ``EvalDatasetWriter(use_value=True)`` in
    ``dataset_writer.py`` — missing or extra keys make LeRobot's ``add_frame``
    raise. Tested in ``tests/test_dagger.py::TestDaggerFrameSchema``.
    """
    state = raw.get("observation.state")
    if state is not None:
        state = np.asarray(state, dtype=np.float32)
    action = raw.get("action")
    if action is not None:
        action = np.asarray(action, dtype=np.float32)
    nan = float("nan")
    return {
        "observation.state": state,
        "action": action,
        "observation.images.top_rgb": _resize_image(
            raw.get("observation.images.top_rgb"), writer_image_shape
        ),
        "observation.images.left_rgb": _resize_image(
            raw.get("observation.images.left_rgb"), writer_image_shape
        ),
        "observation.images.right_rgb": _resize_image(
            raw.get("observation.images.right_rgb"), writer_image_shape
        ),
        "task": raw.get("task", default_task),
        "success_pred": np.array([nan], dtype=np.float32),
        "checkpoint_pred": np.array([nan], dtype=np.float32),
        "success": np.array([binary_success], dtype=np.float32),
        "advantage": np.array([0.0], dtype=np.float32),
        "dense_reward": np.array([dense_reward], dtype=np.float32),
        "checkpoint_held": np.array([checkpoint_held], dtype=np.float32),
        "garment_avg_success": np.array([0.5], dtype=np.float32),
        "garment_avg_checkpoint": np.array([0.5], dtype=np.float32),
        "garment_type_pred": np.array([garment_type_id], dtype=np.int32),
        "completion_pred": np.array([nan], dtype=np.float32),
        "ttc_pred": np.array([nan], dtype=np.float32),
        # DAgger is teleop — no success-check distances are ever computed; write
        # all-NaN so downstream training masks the keypoint losses on these frames.
        "check_distances": np.full(7, nan, dtype=np.float32),
        # Keypoint + WM-flow head predictions — never produced during teleop.
        # NaN placeholders keep the LeRobot schema fixed.
        "keypoint_distances_pred": np.full(21, nan, dtype=np.float32),
        "wm_flow_success_cond": np.array([nan], dtype=np.float32),
        "wm_flow_completion_cond": np.array([nan], dtype=np.float32),
        "wm_flow_keypoint_cond": np.full(21, nan, dtype=np.float32),
        "wm_flow_success_uncond": np.array([nan], dtype=np.float32),
        "wm_flow_completion_uncond": np.array([nan], dtype=np.float32),
        "wm_flow_keypoint_uncond": np.full(21, nan, dtype=np.float32),
        # Δsuccess target — DAgger has no V̂(s_t) baseline (teleop, no model
        # forward pass), so leave NaN; wm_flow_success loss masks NaN targets.
        "success_minus_vhat": np.array([nan], dtype=np.float32),
        # Best-of-N diagnostics — not applicable to DAgger (teleop).
        "best_of_n_score_chosen": np.array([nan], dtype=np.float32),
        "best_of_n_score_mean": np.array([nan], dtype=np.float32),
        "best_of_n_score_min": np.array([nan], dtype=np.float32),
        "best_of_n_score_max": np.array([nan], dtype=np.float32),
        "best_of_n_score_std": np.array([nan], dtype=np.float32),
        "best_of_n_score_spread": np.array([nan], dtype=np.float32),
        "best_of_n_n_valid": np.array([0], dtype=np.int32),
    }

# Per-arm joint order (matches MOTOR_NAMES in shared/real_robot_config.py).
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


# ---------------------------------------------------------------------------
# HF integration for dagger data
# ---------------------------------------------------------------------------


def load_pipeline_config(config_path: str) -> dict:
    """Load RL pipeline YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_next_session_dir() -> Path:
    """Auto-increment session number based on existing sessions."""
    DAGGER_BASE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DAGGER_BASE_DIR.glob("session_*"))
    if not existing:
        return DAGGER_BASE_DIR / "session_001"
    last_num = max(int(d.name.split("_")[1]) for d in existing if d.is_dir())
    return DAGGER_BASE_DIR / f"session_{last_num + 1:03d}"


def download_failure_states_from_hf(repo_id: str, local_dir: Path) -> Path:
    """Download failure_states/ from HF dataset repo."""
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id,
        allow_patterns="failure_states/*.npz",
        local_dir=str(local_dir),
        repo_type="dataset",
    )
    fs_dir = local_dir / "failure_states"
    n = len(list(fs_dir.glob("*.npz"))) if fs_dir.exists() else 0
    logger.info("Downloaded %d failure states from %s", n, repo_id)
    return fs_dir


def load_dagger_manifest_from_hf(repo_id: str) -> dict:
    """Download and load dagger_manifest.json from HF. Returns empty dict if not found."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id,
            filename=f"{DAGGER_HF_PREFIX}/dagger_manifest.json",
            repo_type="dataset",
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.info("No existing dagger manifest on HF — starting fresh")
        return {}


def get_solved_npz_names(manifest: dict) -> set[str]:
    """Get NPZ filenames that should be excluded from future dagger sessions.

    Includes: successfully solved, saved as semi-success, and discarded.
    """
    _DONE_OUTCOMES = {"success", "semi_success", "discarded"}
    return {
        name for name, info in manifest.items()
        if isinstance(info, dict) and info.get("outcome") in _DONE_OUTCOMES
    }


def _verify_hf_parquets(api, repo_id: str, local_dir: Path, hf_prefix: str) -> bool:
    """Verify uploaded parquets match local files (size + PAR1 footer).

    Downloads each parquet from HF and checks:
    1. File size matches local copy
    2. PAR1 magic bytes present in footer (not truncated)

    Returns True if all parquets verify, False if any mismatch found.
    """
    from huggingface_hub import hf_hub_download

    local_parquets = list(local_dir.glob("**/data/chunk-*/*.parquet"))
    if not local_parquets:
        return True

    all_ok = True
    for local_pq in local_parquets:
        rel = local_pq.relative_to(local_dir)
        hf_path = f"{hf_prefix}/{rel}"
        try:
            downloaded = hf_hub_download(
                repo_id, filename=hf_path, repo_type="dataset", force_download=True,
            )
            local_size = local_pq.stat().st_size
            remote_size = Path(downloaded).stat().st_size
            if local_size != remote_size:
                logger.error(
                    "Parquet size mismatch: %s (local=%d, HF=%d)",
                    hf_path, local_size, remote_size,
                )
                all_ok = False
                continue

            with open(downloaded, "rb") as f:
                f.seek(-4, 2)
                footer = f.read(4)
            if footer != b"PAR1":
                logger.error("Parquet truncated on HF (no PAR1 footer): %s", hf_path)
                all_ok = False
        except Exception as e:
            logger.error("Failed to verify parquet %s: %s", hf_path, e)
            all_ok = False

    return all_ok


def upload_dagger_session_to_hf(
    session_dir: Path,
    manifest: dict,
    repo_id: str,
    corrupted_npzs: set[str] | None = None,
    physics_states_dirs: dict[str, Path] | None = None,
) -> bool:
    """Upload dagger session + updated manifest to HF dataset repo.

    Skips session upload if eval_dataset_success has no episodes (no parquet data).
    Always uploads the manifest (tracks attempted failures even without success).
    Removes corrupted/discarded failure state NPZs from HF.
    Uploads success/semi_success physics states to HF.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    session_name = session_dir.name
    hf_prefix = f"{DAGGER_HF_PREFIX}/{session_name}"

    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/access repo %s: %s", repo_id, e)
        return False

    success = True

    # Remove corrupted/discarded failure states from HF
    if corrupted_npzs:
        try:
            from huggingface_hub import CommitOperationDelete
            operations = [
                CommitOperationDelete(path_in_repo=f"failure_states/{name}")
                for name in sorted(corrupted_npzs)
            ]
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Remove {len(corrupted_npzs)} failure states (corrupted/discarded)",
            )
            logger.info("Removed %d failure states from HF", len(corrupted_npzs))
        except Exception as e:
            logger.warning("Failed to remove failure states: %s", e)

    # Upload physics states (success, semi_success) to HF
    if physics_states_dirs:
        for kind, states_dir in physics_states_dirs.items():
            if not states_dir.exists():
                continue
            npz_files = list(states_dir.glob("*.npz"))
            if not npz_files:
                continue
            hf_states_prefix = f"{kind}_states"
            try:
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=str(states_dir),
                    path_in_repo=hf_states_prefix,
                    commit_message=f"DAgger {kind} states from {session_name} ({len(npz_files)} files)",
                )
                logger.info(
                    "Uploaded %d %s states to %s/%s",
                    len(npz_files), kind, repo_id, hf_states_prefix,
                )
            except Exception as e:
                logger.warning("Failed to upload %s states: %s", kind, e)

    # Only upload session if it has successful episodes (non-empty dataset)
    success_ds = session_dir / "eval_dataset_success"
    has_success_data = (
        success_ds.exists()
        and any(success_ds.glob("data/chunk-*/*.parquet"))
    )

    if has_success_data:
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=str(session_dir),
                    path_in_repo=hf_prefix,
                    commit_message=f"DAgger session: {session_name}",
                )
                logger.info("Uploaded session %s to %s/%s", session_name, repo_id, hf_prefix)
            except Exception as e:
                logger.warning("Failed to upload session %s: %s", session_name, e)
                success = False
                break

            # Verify parquets are not truncated
            if _verify_hf_parquets(api, repo_id, session_dir, hf_prefix):
                break
            elif attempt < max_attempts - 1:
                logger.warning("Parquet verification failed — retrying upload")
            else:
                logger.error("Parquet verification failed after %d attempts", max_attempts)
                success = False
    else:
        logger.info("No successful episodes in %s — skipping session upload", session_name)

    # Upload updated manifest
    manifest_path = session_dir.parent / "dagger_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"{DAGGER_HF_PREFIX}/dagger_manifest.json",
            commit_message=f"Update dagger manifest after {session_name}",
        )
        logger.info("Uploaded dagger manifest to %s", repo_id)
    except Exception as e:
        logger.warning("Failed to upload manifest: %s", e)
        success = False

    return success


# ---------------------------------------------------------------------------
# Joint conversion
# ---------------------------------------------------------------------------

def convert_bimanual_to_action(reading: dict) -> np.ndarray:
    """Convert SO101 reader output to a 12-dim action array in sim radians.

    ``scripts/so101_reader.py`` emits the project-wide degree-mode convention
    (arm joints in degrees, gripper 0-100); the shared
    ``real_units_to_sim_radians`` affine maps that to sim USD radians.
    """
    from lehome_solution.training.real_data_transforms import real_units_to_sim_radians

    vec12 = np.array(
        [float(reading["left"].get(n, 0.0)) for n in JOINT_NAMES]
        + [float(reading["right"].get(n, 0.0)) for n in JOINT_NAMES],
        dtype=np.float32,
    )
    return real_units_to_sim_radians(vec12)


# ---------------------------------------------------------------------------
# Failure queue
# ---------------------------------------------------------------------------

class FailureQueue:
    """Manages the queue of failure state NPZ files."""

    def __init__(self, failure_dir: str, exclude_npzs: set[str] | None = None,
                 success_rates: dict | None = None,
                 garment_filter: str | None = None):
        self.failure_dir = Path(failure_dir)
        self._items: list[dict] = []
        self._done: list[dict] = []
        self._success_rates = success_rates
        self._garment_filter = garment_filter
        self._idx = 0
        self._excluded = exclude_npzs or set()
        self._load()

    def _load(self):
        npz_files = sorted(self.failure_dir.glob("*.npz"))
        if not npz_files:
            logger.warning(f"No NPZ files in {self.failure_dir}")
            return

        # Filter out already-solved failures
        if self._excluded:
            before = len(npz_files)
            npz_files = [p for p in npz_files if p.name not in self._excluded]
            skipped = before - len(npz_files)
            if skipped:
                logger.info(f"Skipped {skipped} already-solved failures")

        # Filter by garment name pattern (fnmatch-style glob)
        if self._garment_filter:
            import fnmatch
            before = len(npz_files)
            # Quick pre-filter by filename (garment name is the prefix before _seed)
            npz_files = [
                p for p in npz_files
                if fnmatch.fnmatch(p.stem.rsplit("_seed", 1)[0], self._garment_filter)
            ]
            filtered = before - len(npz_files)
            if filtered:
                logger.info(
                    "Garment filter %r: kept %d, skipped %d",
                    self._garment_filter, len(npz_files), filtered,
                )

        # Group by garment type, load all items
        by_type: dict[str, list[dict]] = {}
        for p in npz_files:
            data = np.load(p, allow_pickle=True)
            meta = json.loads(str(data["metadata"]))
            item = {
                "path": str(p),
                "npz_filename": p.name,
                "garment": meta.get("garment", "unknown"),
                "garment_type": meta.get("garment_type", "unknown"),
                "seed": meta.get("seed", 0),
                "failure_frame": meta.get("failure_frame", -1),
                "garment_points": data.get("garment_points"),
                "garment_velocities": data.get("garment_velocities"),
                "garment_pos": data.get("garment_pos"),
                "garment_ori": data.get("garment_ori"),
                "left_joint_pos": data.get("left_joint_pos"),
                "right_joint_pos": data.get("right_joint_pos"),
                "garment_info": meta.get("garment_info"),
                "augmentation": meta.get("augmentation"),
            }
            gt = garment_name_to_type(item["garment"])
            by_type.setdefault(gt, []).append(item)

        n_garments = sum(len(v) for v in by_type.values())

        # FR-proportional interleaving: sample items so type distribution
        # reflects FR regardless of pool size per type
        if self._success_rates and by_type:
            sr_by_type = self._success_rates.get("by_type", {})
            type_fr = {}
            for t in by_type:
                sr = sr_by_type.get(t, 0.0)
                type_fr[t] = max(1.0 - sr, 0.05)
            total_fr = sum(type_fr.values())
            # Target count per type proportional to FR
            type_targets = {t: max(1, int(round(n_garments * type_fr[t] / total_fr)))
                           for t in by_type}
            for t in by_type:
                np.random.shuffle(by_type[t])
            # Build queue: take up to target from each type, then append leftovers
            for t in sorted(by_type, key=lambda x: -type_fr[x]):
                target = type_targets[t]
                items = by_type[t]
                self._items.extend(items[:target])
                by_type[t] = items[target:]  # leftovers
            # Append remaining items (so nothing is lost)
            for t in sorted(by_type, key=lambda x: -type_fr.get(x, 0.5)):
                self._items.extend(by_type[t])
            logger.info("FailureQueue: FR-proportional ordering — targets: %s",
                        {t: type_targets[t] for t in sorted(type_targets)})
        else:
            all_items = [item for items in by_type.values() for item in items]
            np.random.shuffle(all_items)
            self._items = all_items

        logger.info(
            f"Loaded {len(self._items)} failure states from {n_garments} garments"
        )

    @property
    def total(self) -> int:
        return len(self._items)

    @property
    def remaining(self) -> int:
        return len(self._items) - self._idx

    @property
    def done_count(self) -> int:
        return len(self._done)

    @property
    def success_count(self) -> int:
        return sum(1 for d in self._done if d.get("success"))

    def current(self) -> dict | None:
        if self._idx < len(self._items):
            return self._items[self._idx]
        return None

    def next(self) -> dict | None:
        self._idx += 1
        return self.current()

    def skip(self):
        """Skip current without recording."""
        item = self.current()
        if item:
            item["success"] = False
            item["skipped"] = True
            self._done.append(item)
            self._idx += 1

    def mark_done(self, success: bool):
        item = self.current()
        if item:
            item["success"] = success
            self._done.append(item)
            self._idx += 1

    def retry(self):
        """Retry current — don't advance the index."""
        pass  # no-op, current() still returns the same item

    def peek(self, offset: int = 1) -> dict | None:
        """Look ahead at a future item without advancing the index."""
        idx = self._idx + offset
        return self._items[idx] if idx < len(self._items) else None

    def position_str(self) -> str:
        return f"{self._idx + 1}/{self.total}"


class RandomEpisodeQueue:
    """Queue of fresh-episode items for a single garment (no failure NPZs).

    Mirrors ``FailureQueue``'s public surface so ``DaggerController`` and the
    sim pool can consume it interchangeably. Each item carries
    ``garment_points=None`` so ``_build_remote_task`` omits ``restore_data``
    (a fresh episode) and the sim skips particle restore.
    """

    def __init__(self, garment_name: str, num_episodes: int):
        gt = garment_name_to_type(garment_name)
        if gt == "unknown":
            raise ValueError(
                f"Unknown garment type for {garment_name!r}; expected name like "
                f"'Top_Short_Unseen_0', 'Pant_Long_Seen_3', etc."
            )
        # Seeds need only be unique per item (used for filename + recorder
        # metadata; env randomness comes from env.reset()'s internal RNG).
        base_seed = int(time.time())
        self._items: list[dict] = [
            {
                "path": None,
                "npz_filename": None,
                "garment": garment_name,
                "garment_type": gt,
                "seed": base_seed + i,
                "failure_frame": -1,
                "garment_points": None,
                "garment_velocities": None,
                "garment_pos": None,
                "garment_ori": None,
                "left_joint_pos": None,
                "right_joint_pos": None,
                "garment_info": None,
                "augmentation": None,
            }
            for i in range(num_episodes)
        ]
        self._done: list[dict] = []
        self._idx = 0
        logger.info(
            "RandomEpisodeQueue: %d fresh episodes for %s (type=%s)",
            num_episodes, garment_name, gt,
        )

    @property
    def total(self) -> int:
        return len(self._items)

    @property
    def remaining(self) -> int:
        return len(self._items) - self._idx

    @property
    def done_count(self) -> int:
        return len(self._done)

    @property
    def success_count(self) -> int:
        return sum(1 for d in self._done if d.get("success"))

    def current(self) -> dict | None:
        if self._idx < len(self._items):
            return self._items[self._idx]
        return None

    def next(self) -> dict | None:
        self._idx += 1
        return self.current()

    def skip(self):
        item = self.current()
        if item:
            item["success"] = False
            item["skipped"] = True
            self._done.append(item)
            self._idx += 1

    def mark_done(self, success: bool):
        item = self.current()
        if item:
            item["success"] = success
            self._done.append(item)
            self._idx += 1

    def retry(self):
        pass  # no-op, current() still returns the same item

    def peek(self, offset: int = 1) -> dict | None:
        idx = self._idx + offset
        return self._items[idx] if idx < len(self._items) else None

    def position_str(self) -> str:
        return f"{self._idx + 1}/{self.total}"


# ---------------------------------------------------------------------------
# SO101 reader subprocess
# ---------------------------------------------------------------------------

class SO101ReaderProcess:
    """Manages the SO101 reader subprocess and provides latest reading."""

    def __init__(self, left_port: str, right_port: str):
        self._left_port = left_port
        self._right_port = right_port
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        cmd = [
            str(LEHOME_VENV_PYTHON), "-u",
            str(REPO_ROOT / "scripts" / "so101_reader.py"),
            "--left_port", self._left_port,
            "--right_port", self._right_port,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(LEHOME_CHALLENGE_DIR),
        )
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        # Wait for ready signal
        deadline = time.time() + 10
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    break
            time.sleep(0.05)
        else:
            logger.warning("SO101 reader did not send ready signal in 10s")

    def _reader_loop(self):
        while self._running and self._proc and self._proc.poll() is None:
            line = self._proc.stdout.readline()
            if not line:
                break
            line_s = line.strip()
            try:
                data = json.loads(line_s)
                if "status" in data:
                    # Ready signal — store a dummy to unblock start()
                    with self._lock:
                        self._latest = self._latest or {}
                    continue
                with self._lock:
                    self._latest = data
            except json.JSONDecodeError:
                # Log non-JSON output (errors, warnings from subprocess)
                if line_s:
                    logger.warning(f"SO101 reader: {line_s}")
                continue
        rc = self._proc.poll() if self._proc else None
        if rc is not None and rc != 0:
            logger.error(f"SO101 reader exited with code {rc}")
        self._running = False

    def get_action(self) -> np.ndarray | None:
        """Get latest 12-dim action in radians, or None if no reading."""
        with self._lock:
            reading = self._latest
        if reading is None or "left" not in reading:
            return None
        return convert_bimanual_to_action(reading)

    @property
    def is_alive(self) -> bool:
        return self._running and self._proc is not None and self._proc.poll() is None

    def stop(self):
        self._running = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ---------------------------------------------------------------------------
# Isaac Sim thin client management
# ---------------------------------------------------------------------------

def start_sim(
    first_garment: str,
    first_garment_type: str,
    seed: int,
    output_dir: Path,
    camera_width: int,
    camera_height: int,
    remote_url: str = "",
    session_id: str = "default",
    sim_id: int = 0,
) -> subprocess.Popen:
    """Start the Isaac Sim subprocess running the generic ``remote`` policy.

    The sim is an HTTP client that connects back to the shared DAgger server at
    ``remote_url`` and identifies itself with ``session_id``; per-step batching
    is sent in each action (n_steps), so the old LEHOME_DAGGER_* knobs are gone.
    """
    eval_env = {
        **os.environ,
        "LEHOME_DISABLE_KEYBOARD": "1",
        "PYTHONUNBUFFERED": "1",
        "LEHOME_NO_DEPTH": "1",
        "LEHOME_CHECK_INTERVAL": "30",
        "LEHOME_GARMENT_AUGMENTATION": "1",  # augmentor must be active to re-apply saved augs
        # ...but nothing may be randomized on top of it during manual collection:
        # the human teleoperates what they see, so a per-step colour tint would
        # both distract the operator and inject visual noise into the recording.
        # ``step_color_tint`` defaults to True in the augmentor, so turn it off
        # explicitly (all other knobs already default to 0.0 / disabled).
        "LEHOME_AUG_CONFIG": json.dumps({"step_color_tint": False}),
        "LEHOME_REMOTE_URL": remote_url,
        "LEHOME_WORKER_LABEL": f"W{sim_id}",
    }
    if camera_width and camera_height:
        eval_env["LEHOME_CAMERA_WIDTH"] = str(camera_width)
        eval_env["LEHOME_CAMERA_HEIGHT"] = str(camera_height)

    sim_cmd = [
        str(LEHOME_VENV_PYTHON), "-u", "-m", "scripts.eval",
        "--policy_type", "remote",
        "--remote_url", remote_url,
        "--session_id", session_id,
        "--garment_type", first_garment_type,
        "--garment_name", first_garment,
        "--num_episodes", "1",
        "--max_steps", "9999",
        "--headless",
        "--enable_cameras",
        "--device", "cpu",
        "--seed", str(seed),
        "--same_seed",
    ]
    if camera_width and camera_height:
        sim_cmd += ["--camera_width", str(camera_width), "--camera_height", str(camera_height)]

    log_path = output_dir / f"isaac_sim_{sim_id}.log"
    log_fh = open(log_path, "w")

    proc = subprocess.Popen(
        sim_cmd,
        cwd=str(LEHOME_CHALLENGE_DIR),
        env=eval_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    start_wait = time.time()
    while time.time() - start_wait < 120:
        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if not line:
                break
            log_fh.write(line)
            log_fh.flush()
            if ("Loaded" in line and "garments" in line) or "Evaluating:" in line:
                logger.info("Sim %d garment loaded", sim_id)
                break

    return proc, log_fh


def wait_for_sim_ready(proc: subprocess.Popen, log_fh, on_wait=None) -> bool:
    """Wait for Isaac Sim to finish loading. Returns True if ready.

    ``on_wait`` (optional) is called about once per second while waiting so a
    foreground caller can keep the UI repainting during a ~60s boot. Must only
    be passed from the main thread (it runs OpenCV calls).
    """
    t0 = time.time()
    last_line = ""
    next_note = 30.0
    n_err_logged = 0
    while time.time() - t0 < SIM_BOOT_TIMEOUT:
        if on_wait is not None:
            on_wait()
        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if not line:
                logger.error("Sim boot: stdout closed (proc exit=%s, %.0fs, last: %s)",
                             proc.poll(), time.time() - t0, last_line[:120])
                return False
            log_fh.write(line)
            log_fh.flush()
            if line.strip():
                last_line = line.strip()
            if "remote-driven session" in line:
                logger.info(f"Isaac Sim ready ({time.time() - t0:.0f}s)")
                return True
            if ("Error" in line and "PATCH" not in line
                    and "attachShape" not in line):  # attachShape is benign boot noise
                n_err_logged += 1
                if n_err_logged <= 10:
                    logger.warning(f"Sim error: {line.strip()}")
                elif n_err_logged == 11:
                    logger.warning("Sim error: (further error lines suppressed; "
                                   "see the isaac_sim_*.log)")
        elapsed = time.time() - t0
        if elapsed >= next_note:
            logger.info("Sim boot: %.0fs elapsed, proc=%s, last output: %s",
                        elapsed, "alive" if proc.poll() is None else f"exit={proc.poll()}",
                        last_line[:120])
            next_note += 30.0
    logger.error(f"Sim did not become ready in {SIM_BOOT_TIMEOUT}s "
                 f"(last output: {last_line[:160]})")
    log_proc_diag(proc.pid, "boot-wedge")
    return False


# ---------------------------------------------------------------------------
# Restore message helper
# ---------------------------------------------------------------------------

def _build_remote_task(failure: dict) -> dict:
    """Build a ``/next_task`` response (remote protocol) from a failure dict.

    Two modes:
    - NPZ restore (default): ``restore_data`` carries the saved particle/arm
      state; the sim lifts + settles it (``restore_mode="lifted"``).
    - Fresh episode (``failure["garment_points"] is None``): no ``restore_data``;
      the sim only does switch_garment + reset + stabilize (``--random_garment``).

    DAgger never auto-stops on success (the operator decides) and always wants
    the post-restore physics snapshot back (for saving success/semi NPZs).
    """
    def _to_list(v):
        return v.tolist() if isinstance(v, np.ndarray) else v

    task = {
        "garment_name": failure["garment"],
        "garment_type": failure["garment_type"],
        "seed": failure["seed"],
        "stop_on_success": False,
        "want_snapshot": True,
    }
    if failure.get("garment_points") is not None:
        rd = {
            "garment_points": _to_list(failure["garment_points"]),
            "garment_velocities": _to_list(failure.get("garment_velocities")),
            "left_joint_pos": _to_list(failure["left_joint_pos"]),
            "right_joint_pos": _to_list(failure["right_joint_pos"]),
        }
        if failure.get("garment_pos") is not None:
            rd["garment_pos"] = _to_list(failure["garment_pos"])
        if failure.get("garment_ori") is not None:
            rd["garment_ori"] = _to_list(failure["garment_ori"])
        task["restore_data"] = rd
        task["restore_mode"] = "lifted"
        if failure.get("augmentation"):
            task["augmentation"] = _strip_visual_augmentation(failure["augmentation"])
    return task


# Saved-augmentation keys that only affect how the garment *looks*. Everything
# else in the dict (pos_offset/rot_offset/scale_factor, camera_jitter,
# arm_shift, ...) is geometric: the failure state's particle positions were
# recorded with it applied, so dropping those would desync the restore.
_VISUAL_AUG_KEYS = ("pattern_swap", "pattern_indices", "color_remap", "color_seed")


def _strip_visual_augmentation(aug: dict) -> dict:
    """Drop texture swap / colour remap from a saved augmentation.

    Both rebind textures, and RTX loads textures asynchronously — so the first
    rendered frame still shows the original material and the swap only lands a
    frame or two later. During autonomous rollouts nobody is looking; during
    manual collection it reads as the garment changing colour the moment you
    start recording. The operator teleoperates what they see, so the visuals
    are pinned to the default material and only the geometry is replayed.
    """
    return {k: v for k, v in aug.items() if k not in _VISUAL_AUG_KEYS}


def _dagger_success(check_status, garment_type: str) -> bool:
    """Success from the per-condition check status (long-pant uses the first 4)."""
    if check_status is None:
        return False
    cs = np.asarray(check_status, dtype=np.float32)
    if cs.size == 0:
        return False
    if garment_type == "long-pant":
        return bool(cs[:4].min() > 0.5)
    return bool(cs.min() > 0.5)


class _DaggerServer(RemoteCommandServer):
    """One HTTP server shared by all sims; routes by ``session_id`` to the
    :class:`SimInstance` driving that sim. The sim is an HTTP client (``remote``
    policy) that polls for the next task and per-step actions; this server hands
    each request to the matching SimInstance's coordination queues so the
    controller keeps its synchronous ``send_restore`` / ``send_step`` API."""

    def __init__(self):
        super().__init__()
        self.sessions: dict[str, "SimInstance"] = {}

    def handle(self, endpoint: str, body: dict, session_id: str):
        sim = self.sessions.get(session_id)
        if sim is None:
            return {"shutdown": True}
        if endpoint == "/next_task":
            return sim._take_task()
        if endpoint == "/reset":
            sim._on_reset(body)
            return {}
        if endpoint == "/infer":
            return sim._on_infer(body)
        # /snapshot, /update_check_status: nothing to record here.
        return {}


def _failure_key(failure: dict) -> tuple:
    """Hashable identity for a failure state (npz_filename + garment + seed)."""
    return (failure.get("npz_filename") or failure.get("path", ""),
            failure["garment"], failure["seed"])


# ---------------------------------------------------------------------------
# Multi-sim instances
# ---------------------------------------------------------------------------

class SimInstance:
    """Wraps one Isaac Sim subprocess + WebSocket connection."""

    def __init__(self, sim_id: int, server: "_DaggerServer"):
        self.sim_id = sim_id
        self.session_id = f"w{sim_id}"
        self.server = server
        self.state = "idle"  # idle|loading|ready|active|error|dead
        self.proc: subprocess.Popen | None = None
        self.log_fh = None
        self.current_obs: dict | None = None
        self.restore_snapshot: dict | None = None
        self.current_failure: dict | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()
        self._ping_running = False
        self._ping_thread: threading.Thread | None = None
        self._log_thread: threading.Thread | None = None
        self._log_running = False
        # Coordination with the HTTP server's request threads (remote protocol):
        #   _task_q  : controller -> /next_task  (next episode, or shutdown)
        #   _action_q: controller -> /infer      (next action, or {"stop": True})
        #   _obs_q   : /infer     -> controller  (resulting observation body)
        self._task_q: "queue.Queue" = queue.Queue()
        self._action_q: "queue.Queue" = queue.Queue()
        self._obs_q: "queue.Queue" = queue.Queue()
        self._pending_snapshot: dict | None = None
        self._active = False
        self._skip = False
        # Serializes protocol exchanges (keepalive thread vs controller); the
        # generation counter invalidates zombie background loads after reboot.
        self._io_lock = threading.Lock()
        self._gen = 0
        self._last_exchange = time.time()
        server.sessions[self.session_id] = self

    def start(
        self, first_garment: str, first_garment_type: str, seed: int,
        output_dir: Path, args, on_wait=None,
    ) -> bool:
        """Start the Isaac Sim subprocess (it connects back to the shared server)."""
        logger.info("Booting sim %d: garment=%s seed=%s",
                    self.sim_id, first_garment, seed)
        t_boot = time.time()
        try:
            self.proc, self.log_fh = start_sim(
                first_garment=first_garment,
                first_garment_type=first_garment_type,
                seed=seed,
                output_dir=output_dir,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
                remote_url=self.server.url,
                session_id=self.session_id,
                sim_id=self.sim_id,
            )
            if not wait_for_sim_ready(self.proc, self.log_fh, on_wait=on_wait):
                logger.error("Sim %d failed to start", self.sim_id)
                # Reap it. A sim that missed its startup deadline is usually
                # still alive and still holding ~12 GB plus its GPU context —
                # leaving it running starves the sims that did come up, which
                # is what turns one slow start into a cascade of them.
                self._kill_proc()
                return False

            # Start background log reader
            self._log_running = True
            self._log_thread = threading.Thread(
                target=self._read_log, daemon=True,
            )
            self._log_thread.start()

            # Start liveness watcher
            self.start_ping()
            logger.info("Sim %d boot complete in %.0fs", self.sim_id, time.time() - t_boot)

            logger.info("Sim %d ready (session %s)", self.sim_id, self.session_id)
            return True
        except Exception as e:
            logger.error("Sim %d start failed: %s", self.sim_id, e)
            self.state = "dead"
            return False

    # -- server-thread handlers (called from the HTTP worker thread) --------

    def _take_task(self) -> dict:
        """/next_task: block until the controller assigns a failure (or shutdown)."""
        return self._task_q.get()

    def _on_reset(self, body: dict):
        """/reset: stash the post-restore snapshot; flag a skip on mismatch."""
        self._pending_snapshot = body.get("restore_snapshot")
        if body.get("restore_ok") is False:
            self._skip = True
            self._obs_q.put({"_restore_skip": True})
        else:
            self._skip = False

    def _on_infer(self, body: dict) -> dict:
        """/infer: deliver the observation, return the next action (or stop)."""
        if self._skip:
            self._skip = False
            return {"stop": True}
        self._obs_q.put(body)
        return self._action_q.get()

    # -- controller-thread API (unchanged signatures) -----------------------

    def send_restore(self, failure: dict, on_wait=None) -> bool:
        """Assign a failure to this sim, block until its first observation.

        Returns True on success, False on error/skip. Same semantics as before;
        the transport is now the remote protocol (the sim drives, this answers).

        ``on_wait`` is called repeatedly while blocked. A restore can take tens
        of seconds, and the caller runs the OpenCV event loop, so without this
        the window stops repainting and the WM greys it out as hung.
        """
        t_restore = time.time()
        logger.debug("Sim %d: restore of %s starting (state=%s, proc=%s)",
                     self.sim_id, failure["garment"], self.state,
                     "alive" if self.proc_alive() else "DEAD")
        self.current_failure = failure
        task = _build_remote_task(failure)
        # Drain any stale observation left by a timed-out earlier exchange —
        # consuming it later as this restore's response would silently hand
        # the controller the wrong garment's state.
        while True:
            try:
                self._obs_q.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            if self._active:
                # End the current episode so the sim loops back to /next_task.
                self._action_q.put({"stop": True})
                self._active = False
            self._task_q.put(task)
        try:
            if on_wait is None:
                obs_body = self._obs_q.get(timeout=RESTORE_TIMEOUT)
            else:
                deadline = time.time() + RESTORE_TIMEOUT
                while True:
                    try:
                        obs_body = self._obs_q.get(timeout=0.05)
                        break
                    except queue.Empty:
                        if time.time() >= deadline:
                            raise
                        on_wait()
        except queue.Empty:
            self.load_error = "restore timeout"
            logger.warning("Sim %d: restore of %s TIMED OUT after %.0fs (proc=%s)",
                           self.sim_id, failure["garment"], time.time() - t_restore,
                           "alive" if self.proc_alive() else "DEAD")
            if self.proc_alive():
                log_proc_diag(self.proc.pid, f"restore-wedge-sim{self.sim_id}")
            return False
        if obs_body.get("_restore_skip"):
            self.load_error = "particle_mismatch"
            logger.warning(
                "Sim %d restore skipped for %s (particle mismatch)",
                self.sim_id, failure["garment"],
            )
            return False
        self._last_exchange = time.time()
        logger.info("Sim %d: restore of %s done in %.1fs",
                    self.sim_id, failure["garment"], time.time() - t_restore)
        self.current_obs = decode_observation(obs_body)
        self.restore_snapshot = self._pending_snapshot
        self.current_failure = failure
        self.load_error = None
        self._active = True
        return True

    def hold_step(self) -> bool:
        """One hold-position step (keepalive). Returns False on failure."""
        if self.current_obs is None:
            return True  # nothing to hold against; nothing to keep alive either
        state = self.current_obs.get("observation.state")
        if state is None:
            return True
        if not bool(np.all(np.isfinite(np.asarray(state, dtype=np.float64)))):
            logger.warning("Sim %d: non-finite state in keepalive — physics "
                           "blew up while idle", self.sim_id)
            return False
        try:
            action = np.asarray(state, dtype=np.float32).flatten()[:12]
            resp = self.send_step(action, 1)
            self.current_obs = decode_observation(resp)
            return True
        except Exception as e:
            logger.warning("Sim %d: keepalive step FAILED (%s)", self.sim_id, e)
            if self.proc_alive():
                log_proc_diag(self.proc.pid, f"keepalive-sim{self.sim_id}")
            return False

    def send_step(self, action: np.ndarray, n_steps: int = 1,
                  on_wait=None) -> dict:
        """Apply an action, return the resulting observation (+ success/steps_done).

        ``on_wait`` is called while a slow step is pending (a physics-heavy
        frame right after a restore can take seconds) so the UI keeps
        repainting instead of freezing on the operator.
        """
        t_step = time.time()
        self._action_q.put({"actions": action.tolist(), "n_steps": n_steps})
        if on_wait is None:
            obs_body = self._obs_q.get(timeout=30)
        else:
            deadline = time.time() + 30
            while True:
                try:
                    obs_body = self._obs_q.get(timeout=0.5)
                    break
                except queue.Empty:
                    if time.time() >= deadline:
                        raise
                    on_wait()
        self._last_exchange = time.time()
        if time.time() - t_step > 2.0:
            logger.warning("Sim %d: slow step %.1fs", self.sim_id, time.time() - t_step)
        resp = dict(obs_body)
        resp["steps_done"] = n_steps
        gt = (self.current_failure or {}).get("garment_type", "")
        resp["success"] = _dagger_success(obs_body.get("check_status"), gt)
        return resp

    def start_ping(self):
        """Background liveness watcher (HTTP is connectionless — just watch the proc)."""
        self._ping_running = True

        def _loop():
            while self._ping_running:
                time.sleep(10)
                if self.proc and self.proc.poll() is not None:
                    logger.warning(
                        "Sim %d process died (exit code %s)",
                        self.sim_id, self.proc.returncode,
                    )
                    self.state = "dead"
                    # Unblock any waiting controller call so it doesn't hang.
                    self._obs_q.put({"_restore_skip": True})
                    break

        self._ping_thread = threading.Thread(target=_loop, daemon=True)
        self._ping_thread.start()

    def proc_alive(self) -> bool:
        return bool(self.proc) and self.proc.poll() is None

    def _kill_proc(self):
        """SIGTERM then SIGKILL this sim's process group, if still alive."""
        if not (self.proc and self.proc.poll() is None):
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=3)
            except Exception:
                pass
        except Exception:
            pass

    def reset_for_reboot(self):
        """Make this instance reusable after ``shutdown()``.

        ``shutdown()`` leaves three landmines for a rebooted process: the
        session is deregistered from the server (its HTTP calls would 404),
        and the task/action queues hold the poison ``{"shutdown"}`` /
        ``{"stop"}`` messages — a fresh sim would consume them and exit
        immediately. Drain everything and re-register.
        """
        # Fresh queue objects (not just drained): zombie threads from the old
        # incarnation still block on the old queues and must never steal a
        # message meant for the new process.
        self._gen += 1
        self._task_q = queue.Queue()
        self._action_q = queue.Queue()
        self._obs_q = queue.Queue()
        self._pending_snapshot = None
        self._active = False
        self._skip = False
        self.load_error = None
        self.current_failure = None
        self.current_obs = None
        self.state = "idle"
        self.server.sessions[self.session_id] = self

    def shutdown(self):
        """Shutdown this sim instance."""
        self._ping_running = False
        # Stop log reader before killing process (suppresses abort loop spam)
        self._log_running = False
        # Tell the sim to exit at its next poll (end episode, then shutdown).
        try:
            self._action_q.put({"stop": True})
            self._task_q.put({"shutdown": True})
        except Exception:
            pass
        self.server.sessions.pop(self.session_id, None)
        self._kill_proc()
        if self.log_fh:
            self.log_fh.close()
            self.log_fh = None
        self.state = "dead"

    def _read_log(self):
        """Background thread to drain sim stdout."""
        while self._log_running and self.proc and self.proc.poll() is None:
            try:
                ready, _, _ = select.select([self.proc.stdout], [], [], 1.0)
                if ready:
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    if self.log_fh:
                        self.log_fh.write(line)
                        self.log_fh.flush()
                    if "[Dagger]" in line:
                        logger.info("SIM%d: %s", self.sim_id, line.strip())
            except Exception:
                break


class SimPool:
    """Manages N SimInstance objects with round-robin pre-loading."""

    def __init__(self, num_sims: int, args):
        # One HTTP server shared by all sims; each sim connects back to it and
        # is routed by session_id.
        self.server = _DaggerServer().start()
        self.sims = [SimInstance(i, self.server) for i in range(num_sims)]
        self.num_sims = num_sims
        self.args = args
        self._active_idx = 0
        self._load_threads: dict[int, threading.Thread] = {}
        # ensure_preloaded is called from both the main loop and the background
        # boot thread (start_rest_async); the lock keeps queue-peek + sim
        # assignment atomic so two callers can't assign the same failure twice.
        self._preload_lock = threading.Lock()
        self._boot_thread: threading.Thread | None = None
        self._abort_boot = False
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, daemon=True, name="sim_keepalive")
        self._keepalive_thread.start()

    def _keepalive_loop(self):
        """Hold-step every pre-loaded ("ready") sim before its poll goes stale.

        The active sim is kept alive by the controller's own wait loops (main
        thread); this thread only touches "ready" sims, under the sim's
        _io_lock so a concurrent switch-to-active can never interleave.
        """
        while not self._abort_boot:
            time.sleep(5)
            for sim in self.sims:
                if sim.state != "ready" or not sim.proc_alive():
                    continue
                if time.time() - sim._last_exchange < KEEPALIVE_IDLE:
                    continue
                with sim._io_lock:
                    if sim.state != "ready":
                        continue
                    if not sim.hold_step():
                        sim.state = "error"

    def _start_one(self, sim: SimInstance, first_failure: dict,
                   output_dir: Path, on_wait=None) -> bool:
        """Start a single sim, retrying once if Isaac wedges during startup.

        Isaac occasionally finishes scene setup, then goes silent and never
        reaches the ready marker, while its siblings do the identical work in
        ~30s. It's not reproducible against a particular sim index, so retry
        before giving up: a lost sim costs a pre-load slot for the session.
        """
        garment = first_failure["garment"]
        garment_type = first_failure["garment_type"]
        seed = first_failure["seed"]
        for attempt in range(_SIM_START_ATTEMPTS):
            if self._abort_boot:
                return False
            if sim.start(garment, garment_type, seed, output_dir, self.args,
                         on_wait=on_wait):
                return True
            if attempt + 1 < _SIM_START_ATTEMPTS and not self._abort_boot:
                logger.warning("Sim %d failed to start — retrying (%d/%d)",
                               sim.sim_id, attempt + 2, _SIM_START_ATTEMPTS)
        sim.state = "dead"
        return False

    def start_first(self, first_failure: dict, output_dir: Path) -> bool:
        """Start sim 0 only (foreground). Returns True on success.

        The remaining sims boot afterwards via ``start_rest_async`` — the
        operator should not wait several minutes of sequential Isaac boots
        before the first episode, and a sim left idling in its ``/next_task``
        long-poll for that long has been observed to die the moment the task
        finally arrives. Booting one sim and immediately giving it work keeps
        every idle window short.
        """
        ok = self._start_one(self.sims[0], first_failure, output_dir)
        logger.info("SimPool: sim 0 %s", "started" if ok else "FAILED")
        return ok

    def start_rest_async(self, first_failure: dict, output_dir: Path,
                         queue: 'FailureQueue'):
        """Boot sims 1..N-1 in a background thread (sequential, like before).

        As each sim becomes ready it is *immediately* given the next failure
        from the queue (``ensure_preloaded``), so no sim ever sits idle in its
        ``/next_task`` long-poll — that idle window is what killed sims when
        all boots happened before the first restore.
        """
        if self.num_sims <= 1:
            return

        def _boot():
            ok_count = 1 if self.sims[0].state != "dead" else 0
            for sim in self.sims[1:]:
                if self._abort_boot:
                    return
                if self._start_one(sim, first_failure, output_dir):
                    ok_count += 1
                    # Hand this sim work right away (offset 1: sim 0 holds
                    # queue.current()). Thread-safe via _preload_lock.
                    self.ensure_preloaded(queue, start_offset=1)
            logger.info("SimPool: %d/%d sims started", ok_count, self.num_sims)

        self._boot_thread = threading.Thread(
            target=_boot, daemon=True, name="sim_boot")
        self._boot_thread.start()

    def get_active(self) -> SimInstance:
        return self.sims[self._active_idx]

    def load_in_background(self, sim: SimInstance, failure: dict):
        """Start background restore on a sim."""
        if not sim.proc_alive():
            logger.warning("Sim %d has no live process — not assigning %s",
                           sim.sim_id, failure["garment"])
            return
        logger.debug("Sim %d: assigning background pre-load of %s",
                     sim.sim_id, failure["garment"])
        sim.state = "loading"
        sim.current_failure = failure
        sim.load_error = None
        gen = sim._gen

        def _do_load():
            ok = sim.send_restore(failure)
            if sim._gen != gen:
                logger.info("Sim %d: stale background load discarded (%s)",
                            sim.sim_id, failure["garment"])
                return
            if ok:
                sim.state = "ready"
                logger.info(
                    "Sim %d pre-loaded: %s", sim.sim_id, failure["garment"],
                )
            else:
                sim.state = "error"
                logger.warning(
                    "Sim %d background load failed: %s",
                    sim.sim_id, sim.load_error,
                )

        t = threading.Thread(target=_do_load, daemon=True)
        t.start()
        self._load_threads[sim.sim_id] = t

    def advance(self, target_failure: dict, timeout: float = 60,
                on_wait=None) -> SimInstance | None:
        """Switch to a sim that has target_failure pre-loaded. Returns new active sim or None.

        ``on_wait`` is pumped while polling so the UI stays responsive; see
        ``SimInstance.send_restore``.
        """
        target_key = _failure_key(target_failure)
        t0 = time.time()
        while time.time() - t0 < timeout:
            # Search round-robin for a ready sim with the CORRECT failure
            for offset in range(1, self.num_sims + 1):
                idx = (self._active_idx + offset) % self.num_sims
                sim = self.sims[idx]
                if (sim.state == "ready" and sim.current_failure is not None
                        and _failure_key(sim.current_failure) == target_key):
                    with sim._io_lock:
                        if sim.state != "ready":
                            continue  # keepalive just failed it
                        sim.state = "active"
                    self._active_idx = idx
                    logger.info("Switched to pre-loaded sim %d (%s) in %.1fs",
                                sim.sim_id, target_failure["garment"], time.time() - t0)
                    return sim
            # Check if any sim is loading the target
            any_loading_target = any(
                s.state == "loading" and s.current_failure is not None
                and _failure_key(s.current_failure) == target_key
                for s in self.sims
            )
            if not any_loading_target:
                # Nobody is loading our target — bail immediately
                return None
            if on_wait is not None:
                on_wait()  # repaints the window, and blocks ~50ms doing so
            else:
                time.sleep(0.2)  # poll
        logger.error("SimPool: target sim not ready in %.0fs", timeout)
        return None

    def ensure_preloaded(self, queue: 'FailureQueue', start_offset: int = 1):
        """Ensure all idle/non-active alive sims are pre-loading the correct failures.

        Args:
            queue: failure queue to peek into
            start_offset: queue offset to start assigning from. Use 0 when there
                is no active sim (between episodes), 1 when an active sim already
                has queue.current() loaded.
        """
        with self._preload_lock:
            self._ensure_preloaded_locked(queue, start_offset)

    def _ensure_preloaded_locked(self, queue: 'FailureQueue', start_offset: int):
        # Collect available sims (not active, not dead/error). proc_alive()
        # matters: sims boot in the background now, and an unbooted sim also
        # reads state=="idle" — assigning it a task posts into a queue nobody
        # is polling, and the sim later swallows the stale task on its first
        # poll after boot (observed: preload timeout + wrong-garment restore).
        available = [s for s in self.sims
                     if s.state in ("idle", "ready", "loading") and s.proc_alive()]

        if not available:
            return

        # Collect failures we need pre-loaded
        target_failures = []
        for offset in range(start_offset, start_offset + len(available)):
            f = queue.peek(offset=offset) if offset > 0 else queue.current()
            if f is not None:
                target_failures.append(f)

        # Identify sims already loading/ready with correct failures
        assigned_keys: set[tuple] = set()
        unassigned_sims = []
        for sim in available:
            if sim.current_failure is not None and sim.state in ("ready", "loading"):
                sk = _failure_key(sim.current_failure)
                needed = any(_failure_key(f) == sk for f in target_failures)
                if needed and sk not in assigned_keys:
                    assigned_keys.add(sk)
                    continue  # sim is already doing the right thing
            unassigned_sims.append(sim)

        # Assign remaining sims to unassigned failures
        unassigned_failures = [f for f in target_failures
                               if _failure_key(f) not in assigned_keys]
        for sim, failure in zip(unassigned_sims, unassigned_failures):
            self.load_in_background(sim, failure)

    def shutdown_all(self):
        """Shutdown all sim instances and the shared server."""
        self._abort_boot = True
        if self._boot_thread and self._boot_thread.is_alive():
            # Give the boot thread a moment to notice; a sim mid-boot is
            # killed below regardless (shutdown() kills the process group).
            self._boot_thread.join(timeout=10)
        for sim in self.sims:
            if sim.state != "dead":
                sim.shutdown()
        self.server.stop()

    def alive_count(self) -> int:
        return sum(1 for s in self.sims if s.state not in ("dead", "error"))


# ---------------------------------------------------------------------------
# Observation decoding
# ---------------------------------------------------------------------------

def decode_observation(msg: dict) -> dict:
    """Decode base64 images (JPEG or raw) in a WebSocket response."""
    obs = {}
    for k, v in msg.items():
        if isinstance(v, dict) and "jpeg" in v:
            raw = base64.b64decode(v["jpeg"])
            buf = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            obs[k] = bgr[:, :, ::-1].copy()  # BGR -> RGB
        elif isinstance(v, dict) and "base64" in v:
            raw = base64.b64decode(v["base64"])
            arr = np.frombuffer(raw, dtype=np.dtype(v["dtype"])).reshape(v["shape"])
            obs[k] = arr
        elif isinstance(v, list):
            obs[k] = np.array(v)
        else:
            obs[k] = v
    return obs


# ---------------------------------------------------------------------------
# UI display
# ---------------------------------------------------------------------------

# Foot-pedal arrow keys, same convention as record_real_dagger.py:
#   left pedal   -> LEFT arrow  -> drop this episode, straight on to the next
#   middle pedal -> SPACE       -> pause / resume
#   right pedal  -> RIGHT arrow -> save and advance to the next failure state
# The raw code differs per OpenCV GUI backend (Qt: 0x250000/0x270000,
# GTK: 65361/65363), so both are accepted.
_PEDAL_DROP_CODES = frozenset({0x250000, 65361})
_PEDAL_SAVE_NEXT_CODES = frozenset({0x270000, 65363})

# F11 toggles fullscreen. Qt reports Qt::Key_F11, GTK reports GDK_KEY_F11
# (0xFFC8) — which is what this build actually sends, despite linking Qt5.
_FULLSCREEN_CODES = frozenset({0x01000034, 0xFFC8})

# Sentinel returned for fullscreen; outside both the 0-255 ASCII range that
# every other binding lives in and the -1 "no key" value, so it can never
# collide with a real binding.
KEY_FULLSCREEN = 0x0F11

_unknown_keys_seen: set[int] = set()


def _normalize_key(code: int) -> int:
    """Fold the pedal's arrow keys and F11 onto the plain ASCII key bindings.

    Reading them needs ``waitKeyEx`` — plain ``waitKey() & 0xFF`` collapses
    the arrows to 0 on the Qt backend. Everything else keeps the historical
    ``& 0xFF`` behaviour so the rest of the controller still compares against
    ordinary ASCII.
    """
    if code < 0:
        return -1
    if code in _PEDAL_DROP_CODES:
        return ord("f")  # KEY_SKIP
    if code in _PEDAL_SAVE_NEXT_CODES:
        return ord("s")  # KEY_SAVE_SEMI
    if code in _FULLSCREEN_CODES:
        return KEY_FULLSCREEN
    if code > 0xFF and code not in _unknown_keys_seen:
        # Log once per code so an unrecognised pedal/keyboard can be mapped
        # without guessing at backend-specific key codes.
        _unknown_keys_seen.add(code)
        logger.info("Unmapped key code %d (0x%X) — ignoring", code, code)
    return code & 0xFF


class DaggerUI:
    """OpenCV window for camera display and status."""

    WINDOW_NAME = "DAgger Recovery Collection"
    BORDER_PX = 6

    # Colors (BGR for OpenCV)
    COLOR_RECORDING = (0, 0, 255)    # Red
    COLOR_PAUSED = (0, 255, 255)     # Yellow
    COLOR_SUCCESS = (0, 255, 0)      # Green
    COLOR_RESTORING = (255, 165, 0)  # Orange
    COLOR_IDLE = (128, 128, 128)     # Gray

    def __init__(self, camera_width: int, camera_height: int):
        self.cam_w = camera_width
        self.cam_h = camera_height
        self.pool = None  # set by the controller once the SimPool exists
        self._state = "IDLE"
        self._success_timer = 0.0
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._fullscreen = True
        self._fullscreen_applied = False
        self._apply_fullscreen()
        # Detect screen resolution for image scaling
        # Fallback to 1920x1080 if detection fails
        self._screen_w = 1920
        self._screen_h = 1080
        try:
            import subprocess as _sp
            out = _sp.check_output(["xrandr"], text=True, stderr=_sp.DEVNULL)
            for line in out.splitlines():
                if "*" in line:  # current resolution
                    res = line.split()[0]
                    w, h = res.split("x")
                    self._screen_w, self._screen_h = int(w), int(h)
                    break
        except Exception:
            pass

    def _apply_fullscreen(self):
        if self._fullscreen:
            # Setting the property to the value it already holds is a no-op in
            # the backend, so the window would stay decorated. Bounce it
            # through NORMAL to force the WM to act on the change.
            cv2.setWindowProperty(
                self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL,
        )

    def toggle_fullscreen(self):
        """F11: flip between fullscreen and a normal resizable window."""
        self._fullscreen = not self._fullscreen
        self._apply_fullscreen()
        logger.info("Fullscreen %s", "on" if self._fullscreen else "off")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val
        if val == "SUCCESS":
            self._success_timer = time.time()

    def _border_color(self) -> tuple:
        if self._state == "SUCCESS":
            # Green for 3 seconds after success
            if time.time() - self._success_timer < 3.0:
                return self.COLOR_SUCCESS
            return self.COLOR_IDLE
        return {
            "RECORDING": self.COLOR_RECORDING,
            "PAUSED": self.COLOR_PAUSED,
            "RESTORING": self.COLOR_RESTORING,
        }.get(self._state, self.COLOR_IDLE)

    def update(
        self,
        obs: dict | None,
        garment_name: str,
        frame_idx: int,
        queue_pos: str,
        check_status: np.ndarray | None,
        success_count: int,
        done_count: int,
        time_remaining: float | None = None,
        fps: float | None = None,
        joint_state: np.ndarray | None = None,
    ) -> int:
        """Display cameras + arm panels + status bar. Returns cv2.waitKey result."""
        if obs is None:
            # Show blank frame
            cam_frame = np.zeros((self.cam_h + self.cam_h // 2, self.cam_w, 3), dtype=np.uint8)
        else:
            # Compose multi-camera view
            top = obs.get("observation.images.top_rgb")
            left = obs.get("observation.images.left_rgb")
            right = obs.get("observation.images.right_rgb")

            if top is None or left is None or right is None:
                cam_frame = np.zeros((self.cam_h + self.cam_h // 2, self.cam_w, 3), dtype=np.uint8)
            else:
                # Ensure uint8
                top = self._to_uint8(top)
                left = self._to_uint8(left)
                right = self._to_uint8(right)

                # Strip alpha if present
                if top.shape[2] == 4:
                    top = top[:, :, :3]
                if left.shape[2] == 4:
                    left = left[:, :, :3]
                if right.shape[2] == 4:
                    right = right[:, :, :3]

                # Rotate top 180 degrees
                top = top[::-1, ::-1].copy()

                h, w = top.shape[:2]
                left_s = cv2.resize(left, (w // 2, h // 2))
                right_s = cv2.resize(right, (w // 2, h // 2))
                top_row = np.concatenate([left_s, right_s], axis=1)
                if top_row.shape[1] != w:
                    top_row = cv2.resize(top_row, (w, top_row.shape[0]))
                cam_frame = np.concatenate([top_row, top], axis=0)

        # Arm visualization panels
        cam_h_total = cam_frame.shape[0]
        arm_panel_w = int(cam_frame.shape[1] * 0.35)  # each arm panel ~35% camera width

        left_panel = np.zeros((cam_h_total, arm_panel_w, 3), dtype=np.uint8)
        right_panel = np.zeros((cam_h_total, arm_panel_w, 3), dtype=np.uint8)

        if (joint_state is not None and len(joint_state) >= 12
                and bool(np.all(np.isfinite(np.asarray(joint_state[:12],
                                                       dtype=np.float64))))):
            left_joints = np.asarray(joint_state[:6], dtype=np.float64)
            right_joints = np.asarray(joint_state[6:12], dtype=np.float64)

            # Gripper openness normalized to 0-1
            grip_lo = ARM_JOINT_LIMITS_RAD["gripper"][0]
            grip_hi = ARM_JOINT_LIMITS_RAD["gripper"][1]
            grip_range = grip_hi - grip_lo
            left_grip = float(np.clip((left_joints[5] - grip_lo) / grip_range, 0, 1))
            right_grip = float(np.clip((right_joints[5] - grip_lo) / grip_range, 0, 1))

            draw_arm_side_view(left_panel, left_joints,
                               (0, 0, arm_panel_w, cam_h_total),
                               label="LEFT", gripper_open=left_grip, flip=False)
            draw_arm_side_view(right_panel, right_joints,
                               (0, 0, arm_panel_w, cam_h_total),
                               label="RIGHT", gripper_open=right_grip, flip=True)

        frame = np.concatenate([left_panel, cam_frame, right_panel], axis=1)

        # Timer overlay — top-left
        _font = cv2.FONT_HERSHEY_SIMPLEX
        _ov_scale = 0.5
        _ov_thick = 1
        if time_remaining is not None:
            secs = max(0, int(time_remaining))
            timer_text = f"{secs}s"
            if secs < 10:
                timer_color = (0, 0, 255)
            elif secs < 20:
                timer_color = (0, 255, 255)
            else:
                timer_color = (255, 255, 255)
            (tw, th), _ = cv2.getTextSize(timer_text, _font, _ov_scale, _ov_thick)
            cv2.rectangle(frame, (4, 4), (10 + tw, 10 + th), (0, 0, 0), -1)
            cv2.putText(frame, timer_text, (7, 7 + th), _font, _ov_scale, timer_color, _ov_thick)

        # FPS overlay — top-right
        if fps and fps > 0:
            fps_text = f"{fps:.0f} fps"
            (tw, th), _ = cv2.getTextSize(fps_text, _font, _ov_scale, _ov_thick)
            fw = frame.shape[1]
            cv2.rectangle(frame, (fw - tw - 10, 4), (fw - 4, 10 + th), (0, 0, 0), -1)
            cv2.putText(frame, fps_text, (fw - tw - 7, 7 + th), _font, _ov_scale, (200, 200, 200), _ov_thick)

        # Status bar
        bar_h = 30
        status_bar = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
        status_bar[:] = (30, 30, 30)

        sim_time = frame_idx / 30.0
        sims_txt = ""
        if self.pool is not None:
            usable = sum(1 for s in self.pool.sims
                         if s.state in ("active", "ready") and s.proc_alive())
            sims_txt = f" | sims:{usable}/{len(self.pool.sims)}"
        text = (
            f"{self._state} | {garment_name} | "
            f"{sim_time:.1f}s | Q:{queue_pos} | "
            f"D:{done_count} S:{success_count}{sims_txt}"
        )
        cv2.putText(
            status_bar, text, (5, 20),
            _font, 0.4, (200, 200, 200), 1,
        )
        frame = np.concatenate([frame, status_bar], axis=0)

        # Border
        color = self._border_color()
        b = self.BORDER_PX
        frame[:b, :] = color
        frame[-b:, :] = color
        frame[:, :b] = color
        frame[:, -b:] = color

        # Convert RGB to BGR, scale up, center on black canvas at screen resolution
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fh, fw = frame_bgr.shape[:2]
        scale = min(self._screen_w / fw, self._screen_h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Center on black canvas
        canvas = np.zeros((self._screen_h, self._screen_w, 3), dtype=np.uint8)
        y_off = (self._screen_h - new_h) // 2
        x_off = (self._screen_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = frame_bgr
        cv2.imshow(self.WINDOW_NAME, canvas)

        wait_ms = 50 if self._state == "PAUSED" else 1
        key = _normalize_key(cv2.waitKeyEx(wait_ms))

        if not self._fullscreen_applied:
            # Window managers ignore the fullscreen hint set before the window
            # is first mapped, which is why it otherwise opens as a small
            # floating window. Re-apply once the first frame has been shown.
            self._fullscreen_applied = True
            self._apply_fullscreen()

        if key == KEY_FULLSCREEN:
            self.toggle_fullscreen()
            return -1
        return key

    @staticmethod
    def _to_uint8(img):
        if img.dtype == np.uint8:
            return img
        if np.issubdtype(img.dtype, np.floating):
            if float(img.max()) > 1.5:
                return img.clip(0, 255).astype(np.uint8)
            return (img.clip(0, 1) * 255).astype(np.uint8)
        return img.clip(0, 255).astype(np.uint8)

    def close(self):
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Episode recorder
# ---------------------------------------------------------------------------

class EpisodeRecorder:
    """Accumulates frames for one DAgger episode."""

    def __init__(self, garment_name: str, garment_type: str, seed: int,
                 output_dir: Path, session_idx: int = 0,
                 npz_filename: str | None = None):
        self.garment_name = garment_name
        self.garment_type = garment_type
        self.seed = seed
        self.output_dir = output_dir
        self.session_idx = session_idx
        self.npz_filename = npz_filename
        self.frames: list[dict] = []

    def add_frame(self, obs: dict, action: np.ndarray):
        frame = {
            "observation.state": obs.get("observation.state"),
            "action": action.copy(),
            "observation.images.top_rgb": obs.get("observation.images.top_rgb"),
            "observation.images.left_rgb": obs.get("observation.images.left_rgb"),
            "observation.images.right_rgb": obs.get("observation.images.right_rgb"),
            "check_status": obs.get("check_status"),
            "task": self.garment_name,
            "timestamp": len(self.frames) / 30.0,
            "is_keyframe": True,  # every DAgger frame has a real human action
            "success_pred": float("nan"),  # no policy value for teleop
        }
        self.frames.append(frame)

    def save(self, success: bool) -> tuple[Path | None, Path | None]:
        """Save PKL + video to success/ or failure/ subdir. Returns (pkl_path, video_path)."""
        if not self.frames:
            return None, None
        subdir = "success" if success else "failure"
        pkl_dir = self.output_dir / subdir
        pkl_dir.mkdir(parents=True, exist_ok=True)

        basename = sanitize_basename(self.garment_name)
        filename = f"{basename}_dagger_s{self.session_idx}"

        # Save PKL
        pkl_path = pkl_dir / f"{filename}.pkl"
        payload = {
            "frames": self.frames,
            "garment_info": {
                "garment_name": self.garment_name,
            },
            "dagger_metadata": {
                "session_idx": self.session_idx,
                "success": success,
                "collection_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "npz_filename": self.npz_filename,
            },
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Saved PKL: {subdir}/{pkl_path.name} ({len(self.frames)} frames)")

        # Save video
        video_path = pkl_dir / f"{filename}.mp4"
        render_episode_video(
            self.frames, video_path,
            success=success,
            dense_rewards=(compute_dense_reward(self.frames, self.garment_name) or (None, None))[0],
        )
        logger.info(f"Saved video: {subdir}/{video_path.name}")
        return pkl_path, video_path


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class DaggerController:
    """Orchestrates the DAgger collection process."""

    # Key bindings. The three pedal keys are folded onto these in
    # ``_normalize_key``: LEFT arrow -> KEY_SKIP, RIGHT arrow -> KEY_SAVE_SEMI.
    KEY_PAUSE = ord(" ")       # SPACE / middle pedal: pause/resume recording
    KEY_SKIP = ord("f")        # F / left pedal: drop episode, on to the next
                               #   (nothing saved, failure NPZ left in the pool)
    KEY_RETRY = ord("r")       # R: re-record the same failure state
    KEY_SAVE_SEMI = ord("s")   # S / right pedal: save to semi_success and advance
    KEY_DISCARD = ord("d")     # D: discard AND delete the failure NPZ from disk
    KEY_QUIT = 27              # ESC: quit session

    def __init__(self, args):
        self.args = args
        self._hf_repo: str | None = None
        self._dagger_manifest: dict = {}
        self._shared_states_dir: Path | None = None

        random_garment = getattr(args, "random_garment", None)

        # Config-driven mode: derive dirs/HF repo/shared states dir from pipeline
        # config. Works for both failure-state replay AND random-garment mode —
        # in random mode we skip the failure-state download but keep everything
        # else (HF upload, shared NPZ saving, auto session dir, manifest).
        if getattr(args, "config", None):
            cfg = load_pipeline_config(args.config)
            self._hf_repo = cfg.get("hf_dataset_repo")

            # Derive shared states dir from pipeline config
            config_name = cfg.get("config_name", "pi_modified_bc_rl")
            exp_name = cfg.get("exp_name", "bc_rl_v1")
            self._shared_states_dir = (
                REPO_ROOT / "outputs" / "checkpoints" / config_name / exp_name
            ).resolve()
            logger.info("Shared states dir: %s", self._shared_states_dir)

            # Failure-state pool only needed when NOT in random mode.
            failure_dir = None
            if not random_garment:
                local_failure_dir = self._shared_states_dir / "failure_states"
                if local_failure_dir.exists() and any(local_failure_dir.glob("*.npz")):
                    failure_dir = local_failure_dir
                    n_local = len(list(local_failure_dir.glob("*.npz")))
                    logger.info(
                        "Using local failure states: %s (%d files)",
                        local_failure_dir, n_local,
                    )
                elif self._hf_repo:
                    # Fall back to HF download (different machine)
                    logger.info("No local failure states — downloading from HF")
                    dl_dir = DAGGER_BASE_DIR / "_hf_download"
                    failure_dir = download_failure_states_from_hf(self._hf_repo, dl_dir)
                else:
                    raise ValueError(
                        f"No failure states at {local_failure_dir} and no hf_dataset_repo in config"
                    )

            # Load existing manifest (used for failure-state dedup; harmless in
            # random mode since random items have no npz_filename to look up).
            if self._hf_repo:
                self._dagger_manifest = load_dagger_manifest_from_hf(self._hf_repo)
            local_manifest = self._shared_states_dir / "dagger_manifest.json"
            if local_manifest.exists():
                try:
                    with open(local_manifest) as f:
                        local_m = json.load(f)
                    self._dagger_manifest.update(local_m)
                except (json.JSONDecodeError, OSError):
                    pass
            solved = get_solved_npz_names(self._dagger_manifest)
            if solved and not random_garment:
                logger.info("Already solved: %d failures", len(solved))

            # Auto-determine output dir
            self.output_dir = get_next_session_dir()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Session directory: %s", self.output_dir)

            if random_garment:
                # Random mode: skip success-rates / FR ordering — single garment.
                self.queue = RandomEpisodeQueue(random_garment, args.num_episodes)
            else:
                # Download success rates for FR-proportional ordering
                _sr = None
                local_sr = self._shared_states_dir / "success_rates.json"
                if local_sr.exists():
                    try:
                        with open(local_sr) as _f:
                            _sr = json.load(_f)
                        logger.info("Loaded local success rates from %s", local_sr)
                    except (json.JSONDecodeError, OSError):
                        pass
                if _sr is None:
                    hf_model_repo = cfg.get("hf_model_repo")
                    if hf_model_repo:
                        try:
                            from lehome_solution.training.hf_upload import download_model_asset
                            dl_dir = DAGGER_BASE_DIR / "_hf_download"
                            dl_dir.mkdir(parents=True, exist_ok=True)
                            sr_path = str(dl_dir / "success_rates.json")
                            if download_model_asset(hf_model_repo, "success_rates.json", sr_path):
                                with open(sr_path) as _f:
                                    _sr = json.load(_f)
                        except Exception as e:
                            logger.warning("Failed to download success rates: %s", e)
                self.queue = FailureQueue(
                    str(failure_dir), exclude_npzs=solved, success_rates=_sr,
                    garment_filter=getattr(args, "garment_filter", None),
                )
        else:
            # Manual mode: use provided dirs (no config / no HF integration).
            self.output_dir = Path(args.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if random_garment:
                self.queue = RandomEpisodeQueue(random_garment, args.num_episodes)
            else:
                self.queue = FailureQueue(
                    args.failure_dir,
                    garment_filter=getattr(args, "garment_filter", None),
                )
            if getattr(args, "shared_states_dir", None):
                self._shared_states_dir = Path(args.shared_states_dir).resolve()

        self.ui = DaggerUI(args.camera_width, args.camera_height)
        self.so101 = SO101ReaderProcess(args.left_port, args.right_port)

        self._num_sims = getattr(args, "num_sims", 3)
        self._sim_pool: SimPool | None = None
        self._session_idx = 0
        self._corrupted_npzs: set[str] = set()  # NPZs with particle count mismatch
        self._discarded_npzs: set[str] = set()   # NPZs user chose to discard

        # Post-settle snapshot captured during restore (for saving success/semi-success NPZs)
        self._restore_snapshot: dict | None = None

        # LeRobot dataset writers (success + failure)
        image_shape = (args.camera_height, args.camera_width, 3)
        self._ds_success = EvalDatasetWriter(
            root=self.output_dir / "eval_dataset_success",
            repo_id="dagger_success",
            use_value=True,
            image_shape=image_shape,
        ).create()
        self._ds_failure = EvalDatasetWriter(
            root=self.output_dir / "eval_dataset_failure",
            repo_id="dagger_failure",
            use_value=True,
            image_shape=image_shape,
        ).create()
        self._kf_success = KeyframeDatasetWriter(
            root=self.output_dir / "eval_dataset_success_keyframes",
            repo_id="dagger_success_keyframes",
            image_shape=image_shape,
        ).create()
        self._keyframe_sample_every = 10

        # Background save queue — dataset writes, PKL/video saves, NPZ saves
        # happen in a single background thread so they don't block sim switching.
        # LeRobot dataset objects are NOT thread-safe, so all dataset writes
        # are serialized through this single-threaded executor.
        from concurrent.futures import ThreadPoolExecutor
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bg_save")
        self._save_futures: list = []

    def _drain_save_futures(self):
        """Check completed background saves for errors."""
        still_pending = []
        for fut in self._save_futures:
            if fut.done():
                try:
                    fut.result()  # raises if background save failed
                except Exception as e:
                    logger.error("Background save failed: %s", e)
            else:
                still_pending.append(fut)
        self._save_futures = still_pending

    def _flush_save_futures(self, timeout: float = 60):
        """Wait for all pending background saves to complete."""
        for fut in self._save_futures:
            try:
                fut.result(timeout=timeout)
            except Exception as e:
                logger.error("Background save failed: %s", e)
        self._save_futures.clear()

    def run(self):
        """Main entry point."""
        if self.queue.remaining == 0:
            logger.error("No failure states to process")
            return

        first = self.queue.current()
        logger.info(
            f"Starting DAgger collection: {self.queue.total} failures, "
            f"{self._num_sims} sims"
        )

        try:
            # Start sim 0 only; the first restore goes out the moment it is
            # ready. Sims 1..N-1 boot in the background (start_rest_async in
            # _main_loop, once the queue's head is pinned by the first restore).
            self._sim_pool = SimPool(self._num_sims, self.args)
            self.ui.pool = self._sim_pool
            if not self._sim_pool.start_first(first, self.output_dir):
                logger.error("Failed to start Isaac Sim 0")
                return

            # Start SO101 reader
            if self.so101:
                self.so101.start()
                logger.info("SO101 reader started")

            # Run the main loop
            self._main_loop()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()
            self._print_summary()

    @staticmethod
    def _get_joint_state(obs: dict | None) -> np.ndarray | None:
        """Extract 12-dim joint state from observation for arm viz."""
        if obs is None:
            return None
        state = obs.get("observation.state")
        if state is None:
            return None
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        return state if len(state) >= 12 else None

    def _save_episode_background(self, recorder: EpisodeRecorder, success: bool,
                                  restore_snapshot: dict | None, session_idx: int,
                                  save_npz_subdir: str | None = None):
        """Submit all episode save work to the background thread.

        Args:
            recorder: completed episode recorder (frames already accumulated)
            success: whether episode was successful
            restore_snapshot: snapshot dict for NPZ saving (captured at restore time)
            session_idx: session index for NPZ naming
            save_npz_subdir: "success" or "semi_success" for NPZ saving, None to skip
        """
        # Drain completed futures (log errors)
        self._drain_save_futures()

        # Capture all state needed by the background thread to avoid races
        # with the main thread overwriting self._restore_snapshot etc.
        shared_states_dir = self._shared_states_dir
        output_dir = self.output_dir

        def _do_save():
            try:
                recorder.save(success=success)
                self._write_episode_to_dataset(recorder, success=success)
                if save_npz_subdir and restore_snapshot:
                    self._save_state_npz_impl(
                        recorder, save_npz_subdir, restore_snapshot,
                        session_idx, output_dir, shared_states_dir,
                    )
            except Exception:
                logger.exception("Background save error for %s", recorder.garment_name)

        fut = self._save_executor.submit(_do_save)
        self._save_futures.append(fut)
        logger.info("Queued background save: %s (%d frames, %s)",
                     recorder.garment_name, len(recorder.frames),
                     "success" if success else "failure")

    def _write_episode_to_dataset(self, recorder: EpisodeRecorder, success: bool):
        """Write recorded frames to the appropriate LeRobot dataset."""
        if not recorder.frames:
            return
        writer = self._ds_success if success else self._ds_failure
        binary_success = 1.0 if success else 0.0
        garment_type_id = GARMENT_TYPE_TO_ID.get(recorder.garment_type, -1)
        reward_result = compute_dense_reward(recorder.frames, recorder.garment_name)
        if reward_result is not None:
            dense_rewards, checkpoint_held_values = reward_result
        else:
            dense_rewards, checkpoint_held_values = None, None
        kf_count = 0
        for i, raw in enumerate(recorder.frames):
            frame = build_dagger_frame(
                raw=raw,
                writer_image_shape=writer.image_shape,
                default_task=recorder.garment_name,
                binary_success=binary_success,
                garment_type_id=garment_type_id,
                dense_reward=dense_rewards[i] if dense_rewards else 0.0,
                checkpoint_held=checkpoint_held_values[i] if checkpoint_held_values else 0.0,
            )
            writer.dataset.add_frame(frame)

            # Write every Nth frame to keyframe dataset (success only)
            if success and i % self._keyframe_sample_every == 0:
                kf_frame = {
                    "observation.state": frame["observation.state"],
                    "observation.images.top_rgb": frame["observation.images.top_rgb"],
                    "observation.images.left_rgb": frame["observation.images.left_rgb"],
                    "observation.images.right_rgb": frame["observation.images.right_rgb"],
                    "source_frame_index": np.array([i], dtype=np.int32),
                    "success_pred": np.array([float("nan")], dtype=np.float32),
                    "checkpoint_pred": np.array([float("nan")], dtype=np.float32),
                    "task": raw.get("task", recorder.garment_name),
                }
                self._kf_success.dataset.add_frame(kf_frame)
                kf_count += 1

        writer.dataset.save_episode(parallel_encoding=False)
        if success and kf_count > 0:
            self._kf_success.dataset.save_episode(parallel_encoding=False)
        ep_idx = writer.dataset.meta.total_episodes - 1
        label = "success" if success else "failure"
        kf_info = f", {kf_count} keyframes" if success and kf_count > 0 else ""
        logger.info(
            f"Wrote episode {ep_idx} to {label} dataset ({len(recorder.frames)} frames{kf_info})"
        )

    @staticmethod
    def _save_state_npz_impl(
        recorder: EpisodeRecorder, states_subdir: str,
        restore_snapshot: dict | None, session_idx: int,
        output_dir: Path, shared_states_dir: Path | None,
    ) -> Path | None:
        """Save physics state NPZ for success or semi-success replay.

        Uses the post-settle snapshot captured at restore time as the starting
        state, and all recorded actions as the replay sequence.
        Thread-safe: uses only passed-in state, no self references.
        """
        if not restore_snapshot:
            logger.warning("No restore snapshot available — cannot save state NPZ")
            return None
        if not recorder.frames:
            logger.warning("No frames recorded — cannot save state NPZ")
            return None

        actions = np.array(
            [f["action"] for f in recorder.frames], dtype=np.float32,
        )
        base = sanitize_basename(recorder.garment_name)
        extra = {"dagger": True}

        # Save to local session dir
        local_dir = output_dir / "physics_states" / states_subdir
        result = save_physics_state_npz(
            save_dir=local_dir,
            basename=base,
            seed=recorder.seed,
            ep_idx=session_idx,
            snapshot=restore_snapshot,
            garment_name=recorder.garment_name,
            garment_type=recorder.garment_type,
            actions=actions,
            snapshot_frame=0,
            n_frames=len(recorder.frames),
            metadata_extra=extra,
            min_actions=0,  # always save dagger states regardless of length
            label="dagger",
        )

        # Also save to shared persistent dir if configured
        if shared_states_dir:
            shared_dir = shared_states_dir / f"{states_subdir}_states"
            save_physics_state_npz(
                save_dir=shared_dir,
                basename=base,
                seed=recorder.seed,
                ep_idx=session_idx,
                snapshot=restore_snapshot,
                garment_name=recorder.garment_name,
                garment_type=recorder.garment_type,
                actions=actions,
                snapshot_frame=0,
                n_frames=len(recorder.frames),
                metadata_extra=extra,
                min_actions=0,
                label="dagger-shared",
            )

        return result

    def _delete_failure_npz(self, failure: dict):
        """Delete a failure NPZ + JSON sidecar from disk."""
        npz_path = failure.get("path")
        if not npz_path:
            return
        npz_path = Path(npz_path)
        try:
            if npz_path.exists():
                npz_path.unlink()
                logger.info("Deleted failure NPZ: %s", npz_path.name)
            json_path = npz_path.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
        except OSError as e:
            logger.warning("Failed to delete failure NPZ %s: %s", npz_path.name, e)

    def _main_loop(self):
        """Main loop: manage sim pool, process failures with pre-loading."""
        pool = self._sim_pool

        # Restore first failure on sim[0] (foreground — user waits for the first one)
        active_sim = pool.sims[0]
        first_failure = self.queue.current()
        if first_failure is None:
            return

        active_sim.state = "active"
        if not self._first_restore_with_recovery(active_sim):
            logger.error("Failed to load first failure")
            return
        first_failure = self.queue.current()  # may have advanced past bad NPZs

        # Sim 0 is live and the operator can start working. Boot the remaining
        # sims now, in the background; each is preloaded the moment it's ready.
        pool.start_rest_async(first_failure, self.output_dir, self.queue)

        consecutive_errors = 0
        while True:
            failure = self.queue.current()
            if failure is None:
                logger.info("All failures processed")
                break

            # Drain completed background saves (log errors)
            self._drain_save_futures()

            result = self._process_failure(failure, active_sim)
            logger.info("Episode done: %s -> %s [sims: %s]",
                        failure["garment"], result,
                        ",".join(f"{sm.sim_id}:{sm.state}" for sm in pool.sims))

            # Track consecutive errors (timeouts, step failures)
            if result == "error":
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    logger.error("3 consecutive errors — finishing")
                    break
                self.queue.skip()
            else:
                consecutive_errors = 0

            if result == "quit":
                logger.info("Quit requested — shutting down all sims...")
                break
            elif result == "skip":
                self.queue.skip()
            elif result == "retry":
                self.queue.retry()
                # Re-restore same failure in background on current sim
                active_sim.state = "idle"
                pool.load_in_background(active_sim, failure)
                # Also load on another idle sim for faster switch
                for s in pool.sims:
                    if s is not active_sim and s.state == "idle":
                        pool.load_in_background(s, failure)
                        break
                # Fall through to sim-switching logic below
            elif result == "success":
                self.queue.mark_done(True)
                self._update_manifest(failure, "success")
            elif result == "semi_success":
                self.queue.mark_done(False)
                self._update_manifest(failure, "semi_success")
            elif result == "discard":
                self._delete_failure_npz(failure)
                npz_name = failure.get("npz_filename")
                if npz_name:
                    self._discarded_npzs.add(npz_name)
                self.queue.mark_done(False)
                self._update_manifest(failure, "discarded")

            # --- Switch to next pre-loaded sim ---
            next_failure = self.queue.current()
            if next_failure is None:
                break  # queue exhausted

            old_sim = active_sim
            old_sim.state = "idle"

            # First: check if any sim already has next_failure ready (from
            # earlier pre-loading). Don't kick off new loads yet — that would
            # cause all sims to restore simultaneously, competing for GPU.
            new_sim = pool.advance(next_failure, timeout=0.1)
            if new_sim is None:
                # No sim has next_failure ready. Kick off a load for it on
                # exactly ONE sim, then wait for it.
                # Pick best candidate: prefer a sim that already has the right
                # garment loaded (no switch needed), else any idle sim.
                candidate = None
                next_key = _failure_key(next_failure)
                # Check if any sim is already loading it
                already_loading = any(
                    s.state == "loading" and s.current_failure is not None
                    and _failure_key(s.current_failure) == next_key
                    for s in pool.sims
                )
                if not already_loading:
                    for s in pool.sims:
                        if s.state in ("idle", "ready"):
                            candidate = s
                            break
                    if candidate:
                        pool.load_in_background(candidate, next_failure)
                    else:
                        # All sims dead/error/loading other things — foreground
                        logger.warning("No available sim for %s — foreground on sim %d",
                                       next_failure["garment"], old_sim.sim_id)
                        if self._revive_and_restore(old_sim, next_failure):
                            active_sim = old_sim
                        else:
                            logger.error("Foreground restore failed")
                            break
                        # Pre-load future failures now that active is set
                        pool.ensure_preloaded(self.queue, start_offset=1)
                        continue

                # Wait for the target sim to become ready
                new_sim = pool.advance(
                    next_failure, timeout=60,
                    on_wait=lambda: self._pump_ui(next_failure["garment"]))
                if new_sim is None:
                    logger.warning("Sim loading %s timed out — foreground on sim %d",
                                   next_failure["garment"], old_sim.sim_id)
                    if self._revive_and_restore(old_sim, next_failure):
                        active_sim = old_sim
                    else:
                        logger.error("Foreground restore failed")
                        break
                    pool.ensure_preloaded(self.queue, start_offset=1)
                    continue

            active_sim = new_sim

            # NOW that active sim is secured, pre-load future failures on
            # remaining idle sims (these won't compete with the active restore)
            pool.ensure_preloaded(self.queue, start_offset=1)

        pool.shutdown_all()

    def _revive_and_restore(self, sim: SimInstance, failure: dict) -> bool:
        """Foreground fallback: make ``sim`` usable for ``failure``, whatever
        state it is in — reboot first when it is mid-load, erroring, or dead
        (a sim stuck "loading" is almost always in the abort-spin)."""
        pump = lambda g=failure["garment"]: self._pump_ui(g)  # noqa: E731
        if sim.state in ("loading", "error", "dead") or not sim.proc_alive():
            logger.warning("Sim %d unusable (state=%s, proc=%s) — rebooting",
                           sim.sim_id, sim.state,
                           "alive" if sim.proc_alive() else "dead")
            sim.shutdown()
            sim.reset_for_reboot()
            if not self._sim_pool._start_one(sim, failure, self.output_dir,
                                             on_wait=pump):
                return False
        sim.state = "active"
        return sim.send_restore(failure, on_wait=pump)

    def _first_restore_with_recovery(self, sim: SimInstance) -> bool:
        """Load the session's first failure state, surviving Isaac wedges.

        Isaac occasionally wedges or silently dies during boot or the first
        ``env.reset`` — the process either never reaches the ready marker or
        goes quiet mid-reset. Both leave the recorder with nothing to show.
        Recovery: reboot the sim and retry the SAME state (a wedge says
        nothing about the NPZ). Only a genuine particle-count mismatch — sim
        alive, restore explicitly skipped — advances the queue. The UI keeps
        repainting throughout.
        """
        for attempt in range(1, _FIRST_RESTORE_ATTEMPTS + 1):
            failure = self.queue.current()
            if failure is None:
                return False  # queue exhausted
            garment = failure["garment"]
            pump = lambda g=garment: self._pump_ui(g)  # noqa: E731

            if not sim.proc_alive():
                logger.info("Rebooting sim %d (attempt %d)...", sim.sim_id, attempt)
                sim.shutdown()
                sim.reset_for_reboot()
                if not self._sim_pool._start_one(sim, failure, self.output_dir,
                                                 on_wait=pump):
                    logger.error("Sim %d would not reboot", sim.sim_id)
                    return False
                sim.state = "active"

            logger.info("Loading first failure %s on sim %d (attempt %d)...",
                        garment, sim.sim_id, attempt)
            if sim.send_restore(failure, on_wait=pump):
                return True

            if sim.proc_alive() and sim.load_error == "particle_mismatch":
                # Real NPZ/garment mismatch — the state is bad, the sim is fine.
                logger.warning("Skipping %s (particle mismatch)", garment)
                self.queue.skip()
                continue

            # Wedged (timeout, proc alive) or dead — reboot and retry same state.
            logger.warning(
                "First restore of %s failed (%s, proc %s) — rebooting sim",
                garment, sim.load_error,
                "alive" if sim.proc_alive() else "dead",
            )
            sim.shutdown()
        return False

    def _pump_ui(self, garment: str):
        """Repaint the window while a blocking sim restore/advance is running.

        Passed as ``on_wait`` to ``send_restore``/``advance``. Without it the
        window is frozen for the whole restore — which with ``--num_sims 1`` is
        every single episode transition, since there is no pre-loaded sim to
        switch to.
        """
        self.ui.state = "RESTORING"
        self.ui.update(
            None, garment, 0, self.queue.position_str(), None,
            self.queue.success_count, self.queue.done_count,
            joint_state=None,
        )

    def _handle_key_pre_recording(self, key, recorder=None) -> str | None:
        """Handle key press in pre-recording or paused state.

        Returns result string to propagate, or None to continue.
        """
        if key == self.KEY_PAUSE:
            return "start_recording"
        elif key == self.KEY_SKIP:
            return "skip"
        elif key == self.KEY_RETRY:
            return "retry"
        elif key == self.KEY_QUIT:
            return "quit"
        elif key == self.KEY_DISCARD:
            return "discard"
        elif key == self.KEY_SAVE_SEMI:
            # S during recording/pause: save to semi_success
            if recorder and recorder.frames:
                self._save_episode_background(
                    recorder, success=False,
                    restore_snapshot=self._restore_snapshot,
                    session_idx=self._session_idx,
                    save_npz_subdir="semi_success",
                )
                self._session_idx += 1
                logger.info(f"Queued {len(recorder.frames)} frames to semi_success")
                return "semi_success"
            return "skip"  # nothing to save
        return None

    def _process_failure(self, failure: dict, sim: SimInstance) -> str:
        """Process one failure using the given sim instance.

        If sim already has this failure pre-loaded (state=ready), skips restore.
        Returns: 'success', 'skip', 'retry', 'quit', 'discard', 'semi_success', 'error'.
        """
        garment = failure["garment"]
        garment_type = failure["garment_type"]
        seed = failure["seed"]

        # Use pre-loaded state if available, otherwise foreground restore
        if (sim.state in ("ready", "active") and sim.current_failure is not None
                and _failure_key(sim.current_failure) == _failure_key(failure)):
            obs = sim.current_obs
            self._restore_snapshot = sim.restore_snapshot
        else:
            # Foreground restore (first episode or retry)
            self.ui.state = "RESTORING"
            self.ui.update(
                None, garment, 0, self.queue.position_str(), None,
                self.queue.success_count, self.queue.done_count,
                joint_state=None,
            )
            if not sim.send_restore(
                    failure, on_wait=lambda: self._pump_ui(garment)):
                if sim.load_error == "particle_mismatch":
                    npz_name = failure.get("npz_filename")
                    if npz_name:
                        self._corrupted_npzs.add(npz_name)
                return "skip" if sim.load_error == "particle_mismatch" else "error"
            obs = sim.current_obs
            self._restore_snapshot = sim.restore_snapshot

        sim.state = "active"

        # Show initial observation, wait for user
        self.ui.state = "PAUSED"
        logger.info(
            f"Ready: {garment} [sim{sim.sim_id}] "
            "(SPACE/mid-pedal=start, RIGHT/right-pedal=save+next, "
            "LEFT/left-pedal=drop+next, R=re-record, D=delete NPZ, "
            "F11=fullscreen, ESC=quit)"
        )

        # Pre-recording wait loop. The hold-step keeps the ACTIVE sim's
        # long-poll fresh while the operator lines up — an idle sim aborts on
        # the next touch (see KEEPALIVE_IDLE). It also live-updates the view.
        _last_hold = time.time()
        while True:
            key = self.ui.update(
                obs, garment, 0, self.queue.position_str(),
                obs.get("check_status"), self.queue.success_count, self.queue.done_count,
                joint_state=self._get_joint_state(obs),
            )
            if time.time() - _last_hold >= KEEPALIVE_IDLE:
                if not sim.hold_step():
                    return "error"
                obs = sim.current_obs or obs
                _last_hold = time.time()
            result = self._handle_key_pre_recording(key)
            if result == "start_recording":
                break
            elif result is not None:
                return result

        # Recording loop
        self.ui.state = "RECORDING"
        recorder = EpisodeRecorder(
            garment, garment_type, seed,
            self.output_dir, self._session_idx,
            npz_filename=failure.get("npz_filename"),
        )

        max_sim_seconds = self.args.episode_timeout  # sim-time limit
        max_steps = int(max_sim_seconds * 30)  # 30 fps sim
        steps_per_batch = self.args.steps_per_batch
        sim_step = 0
        fps_t0 = time.time()
        fps_frames = 0
        fps_val = 0.0
        while sim_step < max_steps:

            # Get action from the SO101 leader arms
            action = self.so101.get_action() if self.so101 else None

            if action is None:
                # No reading yet — hold position using current state
                state = obs.get("observation.state")
                if isinstance(state, list):
                    state = np.array(state, dtype=np.float32)
                action = state if state is not None else np.zeros(12, dtype=np.float32)

            # Send step to sim
            try:
                resp = sim.send_step(
                    action, steps_per_batch,
                    on_wait=lambda: self.ui.update(
                        obs, garment, sim_step, self.queue.position_str(),
                        None, self.queue.success_count, self.queue.done_count,
                        joint_state=self._get_joint_state(obs),
                    ))
            except Exception as e:
                logger.error(f"Step failed on sim {sim.sim_id}: {e}")
                return "error"

            obs = decode_observation(resp)
            _st = obs.get("observation.state")
            if _st is not None and not bool(np.all(np.isfinite(
                    np.asarray(_st, dtype=np.float64)))):
                logger.warning(
                    "Sim %d returned non-finite joint state at step %d — "
                    "physics blew up, aborting episode", sim.sim_id, sim_step)
                return "error"
            check_status = obs.get("check_status")
            if isinstance(check_status, list):
                check_status = np.array(check_status, dtype=np.float32)

            # Record N frames (same action repeated, obs from last step)
            n_done = resp.get("steps_done", steps_per_batch)
            for _ in range(n_done):
                recorder.add_frame(obs, action)
            sim_step += n_done

            # FPS counter
            fps_frames += n_done
            fps_elapsed = time.time() - fps_t0
            if fps_elapsed >= 1.0:
                fps_val = fps_frames / fps_elapsed
                fps_frames = 0
                fps_t0 = time.time()

            # Check success
            if resp.get("success"):
                self.ui.state = "SUCCESS"
                logger.info(f"SUCCESS detected at step {sim_step}!")
                for _ in range(40):
                    self.ui.update(
                        obs, garment, sim_step, self.queue.position_str(),
                        check_status, self.queue.success_count, self.queue.done_count,
                        joint_state=self._get_joint_state(obs),
                    )

                self._save_episode_background(
                    recorder, success=True,
                    restore_snapshot=self._restore_snapshot,
                    session_idx=self._session_idx,
                    save_npz_subdir="success",
                )
                self._session_idx += 1
                return "success"

            # Handle keyboard
            time_remaining = (max_steps - sim_step) / 30.0
            key = self.ui.update(
                obs, garment, sim_step, self.queue.position_str(),
                check_status, self.queue.success_count, self.queue.done_count,
                time_remaining=time_remaining,
                fps=fps_val,
                joint_state=self._get_joint_state(obs),
            )

            if key == self.KEY_PAUSE:
                if self.ui.state == "PAUSED":
                    self.ui.state = "RECORDING"
                else:
                    self.ui.state = "PAUSED"
                    _last_hold = time.time()
                    while self.ui.state == "PAUSED":
                        key2 = self.ui.update(
                            obs, garment, sim_step, self.queue.position_str(),
                            check_status, self.queue.success_count, self.queue.done_count,
                            joint_state=self._get_joint_state(obs),
                        )
                        if time.time() - _last_hold >= KEEPALIVE_IDLE:
                            if not sim.hold_step():
                                return "error"
                            obs = sim.current_obs or obs
                            _last_hold = time.time()
                        result = self._handle_key_pre_recording(key2, recorder)
                        if result == "start_recording":
                            self.ui.state = "RECORDING"
                            break
                        elif result is not None:
                            return result
            elif key == self.KEY_SKIP:
                return "skip"
            elif key == self.KEY_DISCARD:
                return "discard"
            elif key == self.KEY_QUIT:
                return "quit"
            elif key == self.KEY_RETRY:
                return "retry"
            elif key == self.KEY_SAVE_SEMI:
                if recorder.frames:
                    self._save_episode_background(
                        recorder, success=False,
                        restore_snapshot=self._restore_snapshot,
                        session_idx=self._session_idx,
                        save_npz_subdir="semi_success",
                    )
                    self._session_idx += 1
                    logger.info(f"Queued {len(recorder.frames)} frames to semi_success")
                    return "semi_success"

        # Max steps reached — skip (no auto-save)
        logger.info(f"Sim timeout ({max_sim_seconds}s / {max_steps} steps)")
        return "skip"

    def _update_manifest(self, failure: dict, outcome: str):
        """Track NPZ outcome in the dagger manifest and persist to disk."""
        npz_name = failure.get("npz_filename")
        if not npz_name:
            return
        self._dagger_manifest[npz_name] = {
            "outcome": outcome,
            "session": self.output_dir.name,
            "garment": failure.get("garment"),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # Persist manifest incrementally (atomic write via temp + rename)
        manifest_path = self.output_dir / "dagger_manifest.json"
        tmp_path = manifest_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._dagger_manifest, f, indent=2)
            tmp_path.rename(manifest_path)
        except OSError as e:
            logger.warning("Failed to persist manifest to disk: %s", e)

    def _cleanup(self):
        """Cleanup all resources."""
        import shutil

        # Wait for all background saves to complete before finalizing datasets
        logger.info("Waiting for background saves to finish...")
        self._flush_save_futures(timeout=120)
        self._save_executor.shutdown(wait=True)

        # Finalize all datasets (writes meta/episodes parquet — required for valid LeRobot dataset)
        for label, writer in [("success", self._ds_success), ("failure", self._ds_failure)]:
            try:
                writer.dataset.finalize()
            except Exception as e:
                logger.warning("Failed to finalize %s dataset: %s", label, e)
        self._kf_success.finalize()

        # Log dataset stats
        for label, writer in [("success", self._ds_success), ("failure", self._ds_failure)]:
            n = writer.dataset.meta.total_episodes
            if n > 0:
                logger.info(f"{label} dataset: {n} episodes at {writer.root}")
        kf_n = self._kf_success.dataset.meta.total_episodes if self._kf_success._dataset else 0
        if kf_n > 0:
            logger.info(f"success keyframes: {kf_n} episodes at {self._kf_success.root}")

        # Remove raw temp PKL files — data is already in LeRobot datasets
        # Keep videos (useful for review), remove only .pkl files
        for subdir in ("success", "failure"):
            temp_dir = self.output_dir / subdir
            if temp_dir.exists():
                for pkl in temp_dir.glob("*.pkl"):
                    pkl.unlink()
                    logger.info("Removed temp PKL: %s", pkl.name)

        # Remove launcher debug files (keep sim logs for debugging)
        for i in range(self._num_sims):
            p = self.output_dir / f"launcher_debug_{i}.py"
            p.unlink(missing_ok=True)

        # Upload to HF if config-driven
        npzs_to_remove = self._corrupted_npzs | self._discarded_npzs
        if self._hf_repo and (self._dagger_manifest or npzs_to_remove):
            logger.info("Uploading dagger session to HF...")
            if self._corrupted_npzs:
                logger.info("Will remove %d corrupted failure states", len(self._corrupted_npzs))
            if self._discarded_npzs:
                logger.info("Will remove %d discarded failure states", len(self._discarded_npzs))
            upload_dagger_session_to_hf(
                self.output_dir, self._dagger_manifest, self._hf_repo,
                corrupted_npzs=npzs_to_remove,
                physics_states_dirs={
                    "success": self.output_dir / "physics_states" / "success",
                    "semi_success": self.output_dir / "physics_states" / "semi_success",
                },
            )

        self.ui.close()

        if self.so101:
            self.so101.stop()

        # Shutdown sim pool
        if self._sim_pool:
            self._sim_pool.shutdown_all()

    def _print_summary(self):
        """Print collection summary."""
        print(f"\n{'=' * 60}")
        print("DAGGER COLLECTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total failures: {self.queue.total}")
        print(f"Processed: {self.queue.done_count}")
        print(f"Remaining: {self.queue.remaining}")
        print(f"Successes: {self.queue.success_count}")
        if self.queue.done_count > 0:
            sr = self.queue.success_count / self.queue.done_count * 100
            print(f"Success rate: {sr:.1f}%")
        print(f"Success PKL: {self.output_dir / 'success'}")
        print(f"Failure PKL: {self.output_dir / 'failure'}")
        ds_s = self._ds_success.dataset.meta.total_episodes
        ds_f = self._ds_failure.dataset.meta.total_episodes
        print(f"Success dataset: {self._ds_success.root} ({ds_s} episodes)")
        print(f"Failure dataset: {self._ds_failure.root} ({ds_f} episodes)")
        if self._corrupted_npzs:
            print(f"Corrupted NPZs (removed from HF): {len(self._corrupted_npzs)}")
            for name in sorted(self._corrupted_npzs):
                print(f"  {name}")

        # Save summary JSON
        summary = {
            "total": self.queue.total,
            "processed": self.queue.done_count,
            "remaining": self.queue.remaining,
            "successes": self.queue.success_count,
            "output_dir": str(self.output_dir),
        }
        summary_path = self.output_dir / "dagger_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DAgger-style recovery data collection from failure states"
    )
    # Config-driven mode (pulls from HF)
    parser.add_argument(
        "--config", default=None,
        help="RL pipeline YAML config (pulls failure states + dagger data from HF)",
    )
    # Manual mode
    parser.add_argument(
        "--failure_dir", default=None,
        help="Directory with failure state NPZ files (manual mode)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory for PKL episodes and videos (manual mode)",
    )
    parser.add_argument("--left_port", default="/dev/ttyACM0", help="Left SO101 leader serial port")
    parser.add_argument("--right_port", default="/dev/ttyACM1", help="Right SO101 leader serial port")
    parser.add_argument("--camera_width", type=int, default=320, help="Camera width")
    parser.add_argument("--camera_height", type=int, default=240, help="Camera height")
    parser.add_argument("--episode_timeout", type=float, default=60,
                        help="Sim-time limit per episode in seconds (60s = 1800 steps, auto-resets)")
    parser.add_argument("--jpeg_quality", type=int, default=80,
                        help="JPEG quality for image compression (0-100, lower=faster)")
    parser.add_argument("--steps_per_batch", type=int, default=1,
                        help="Sim steps per WS round-trip (3=~3x faster, same action repeated)")
    parser.add_argument("--render_every_n", type=int, default=1,
                        help="Render every N-th frame (1 = every frame, 3 = 10Hz camera)")
    parser.add_argument("--shared_states_dir", default=None,
                        help="Shared persistent dir for success/semi-success states (manual mode)")
    parser.add_argument("--garment_filter", default=None,
                        help="Garment name filter (fnmatch glob, e.g. 'Top_Short*' or 'Pant_Long_Unseen_*')")
    parser.add_argument("--num_sims", type=int, default=3,
                        help="Number of parallel Isaac Sim instances (default 3, reduces wait between episodes)")
    # Random-garment mode (no failure states; just spawn fresh episodes)
    parser.add_argument("--random_garment", default=None,
                        help="Single garment name (e.g. 'Top_Short_Unseen_0') to "
                             "collect fresh random episodes for. Bypasses failure-state "
                             "loading; mutually exclusive with --config and --failure_dir.")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to collect in --random_garment mode (default 50).")

    args = parser.parse_args()

    if args.random_garment:
        if args.failure_dir:
            parser.error("--random_garment cannot be combined with --failure_dir "
                         "(random mode has no failure-state pool to consume)")
        if not args.config and not args.output_dir:
            parser.error("--random_garment requires --config (preferred — same HF/shared "
                         "dir as the pipeline) or --output_dir (manual mode)")
        if args.num_episodes <= 0:
            parser.error("--num_episodes must be > 0")
    elif not args.config and not args.failure_dir:
        parser.error("Either --config, --failure_dir, or --random_garment is required")
    elif not args.config and not args.output_dir:
        parser.error("--output_dir is required in manual mode (without --config)")

    controller = DaggerController(args)
    controller.run()


if __name__ == "__main__":
    main()
