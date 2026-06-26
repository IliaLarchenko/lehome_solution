"""Shared utilities for saving/loading physics state NPZ files.

Used by eval_worker (success/failure state saving) and dagger_collect
(success/semi-success state saving from teleop episodes).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Minimum actions for a success/semi-success state to be worth saving (2s at 30Hz)
MIN_SUCCESS_STATE_ACTIONS = 60

# Physics state keys stored in NPZ files
_PHYSICS_KEYS = (
    "garment_points", "garment_velocities",
    "garment_pos", "garment_ori",
    "left_joint_pos", "right_joint_pos",
)


def _safe_garment_info(garment_info: dict) -> dict:
    """Convert garment_info values to JSON-safe types."""
    safe = {}
    for k, v in garment_info.items():
        if isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        elif hasattr(v, "tolist"):
            safe[k] = v.tolist()
        else:
            safe[k] = v
    return safe


def save_physics_state_npz(
    save_dir: Path,
    basename: str,
    seed: int,
    ep_idx: int,
    snapshot: dict,
    garment_name: str,
    garment_type: str,
    *,
    actions: np.ndarray | None = None,
    snapshot_frame: int | None = None,
    failure_frame: int | None = None,
    n_frames: int = 0,
    garment_info: dict | None = None,
    augmentation_info: dict | None = None,
    metadata_extra: dict | None = None,
    top_rgb: np.ndarray | None = None,
    min_actions: int = MIN_SUCCESS_STATE_ACTIONS,
    label: str = "",
) -> Path | None:
    """Save a physics state NPZ file with JSON metadata sidecar.

    Args:
        save_dir: Directory to write the NPZ+JSON pair.
        basename: Base filename prefix (garment name, sanitized).
        seed: Episode seed.
        ep_idx: Episode index.
        snapshot: Dict with physics state arrays (garment_points, velocities, etc.).
        garment_name: Human-readable garment name.
        garment_type: Garment type key (top_long, top_short, etc.).
        actions: Optional action array for success/semi-success states.
        snapshot_frame: Frame index of the snapshot (for success states).
        failure_frame: Frame index of the failure (for failure states).
        n_frames: Total frames in the episode.
        garment_info: Optional garment metadata dict.
        augmentation_info: Optional augmentation metadata dict.
        metadata_extra: Extra metadata fields (e.g. {"dagger": True}).
        top_rgb: Optional camera frame for visual reference (failure states).
        min_actions: Minimum action count for states with actions (skip if fewer).
        label: Log label prefix (e.g. "worker0").

    Returns:
        Path to the saved NPZ, or None if skipped/failed.
    """
    # Validate actions length if provided
    if actions is not None and len(actions) < min_actions:
        logger.info(
            "%s ep%d: skipping state save — too short (%d actions < %d minimum)",
            label, ep_idx, len(actions), min_actions,
        )
        return None

    # Validate particle count
    garment_points = snapshot.get("garment_points")
    if garment_points is not None:
        pts = np.array(garment_points, dtype=np.float32)
        if len(pts) == 0:
            logger.warning("%s ep%d: skipping state save — zero particles", label, ep_idx)
            return None
    else:
        logger.warning("%s ep%d: skipping state save — no garment_points", label, ep_idx)
        return None

    particle_count = len(pts)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{basename}_seed{seed}_ep{ep_idx}"

    # Build save dict — physics state arrays
    save_data = {}
    for key in _PHYSICS_KEYS:
        val = snapshot.get(key)
        if val is not None:
            save_data[key] = np.array(val, dtype=np.float32)

    # Actions (success/semi-success states)
    if actions is not None:
        save_data["actions"] = np.array(actions, dtype=np.float32)

    # Top camera frame (failure states, visual reference)
    if top_rgb is not None:
        save_data["top_rgb"] = np.asarray(top_rgb)

    # Metadata
    meta: dict = {
        "garment": garment_name,
        "garment_type": garment_type,
        "seed": seed,
        "ep_idx": ep_idx,
        "particle_count": particle_count,
        "n_frames": n_frames,
        "status": "active",
    }
    if snapshot_frame is not None:
        meta["snapshot_frame"] = snapshot_frame
    if failure_frame is not None:
        meta["failure_frame"] = failure_frame
    if actions is not None:
        meta["n_actions"] = len(actions)
    if garment_info:
        meta["garment_info"] = _safe_garment_info(garment_info)
    if augmentation_info:
        meta["augmentation"] = augmentation_info
    if metadata_extra:
        meta.update(metadata_extra)

    save_data["metadata"] = np.array(json.dumps(meta))

    try:
        npz_path = save_dir / f"{fname}.npz"
        np.savez_compressed(npz_path, **save_data)

        # JSON sidecar
        sidecar: dict = {
            "garment": garment_name,
            "garment_type": garment_type,
            "seed": seed,
            "particle_count": particle_count,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if actions is not None:
            sidecar["n_actions"] = len(actions)
        if n_frames:
            sidecar["n_frames"] = n_frames
        if metadata_extra:
            sidecar.update(metadata_extra)

        with open(npz_path.with_suffix(".json"), "w") as f:
            json.dump(sidecar, f, indent=2)

        # Log
        parts = []
        if failure_frame is not None:
            parts.append(f"failure frame {failure_frame}")
        if snapshot_frame is not None:
            parts.append(f"snapshot frame {snapshot_frame}")
        if actions is not None:
            parts.append(f"{len(actions)} actions")
        parts.append(f"{particle_count} particles")
        detail = ", ".join(parts)
        rel_path = f"{save_dir.name}/{fname}.npz"
        logger.info("%s ep%d: saved state (%s) -> %s", label, ep_idx, detail, rel_path)

        return npz_path
    except Exception as e:
        logger.warning("%s ep%d: failed to save state: %s", label, ep_idx, e)
        return None
