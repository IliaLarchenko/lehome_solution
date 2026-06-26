#!/usr/bin/env python3
"""Recompute advantages using cross-episode segment-level normalization.

Replaces per-episode GAE advantages with segment-level advantages that properly
handle partial-success episodes. Leaves value column (per-frame V(s) predictions) untouched.

Usage:
    uv run python scripts/recompute_advantages.py \
        --eval_dirs \
            outputs/eval_videos/rl1500_20260302_174908/eval_dataset \
            outputs/eval_videos/rl2500_20260303_084825/eval_dataset
"""

import argparse
import collections
import dataclasses
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lehome_solution.constants import BC_FR_SCALE
from lehome_solution.eval.dataset_writer import (
    EVAL_EPISODE_META_DIR,
    compute_segment_advantages,
    compute_gae_advantages,
)
from lehome_solution.utils.state_persistence import (
    STATE_FILENAME,
    load_state,
    save_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _garment_name_to_type(garment: str) -> str:
    """Derive garment_type from garment name (e.g. 'Pant_Long_Seen_1' -> 'pant_long')."""
    parts = garment.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}".lower()
    return garment.lower()


def _load_meta_from_lerobot(dataset_dir: Path) -> dict[int, dict]:
    """Fallback: derive episode metadata from LeRobot episode parquet + data parquet.

    Used for datasets (e.g. dagger) that don't have eval_episode_metadata JSON files.
    """
    ep_meta: dict[int, dict] = {}

    # Try LeRobot v3 episode metadata for garment names
    ep_parquets = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet")) if (dataset_dir / "meta" / "episodes").exists() else []
    garment_by_ep: dict[int, str] = {}
    for pq_path in ep_parquets:
        try:
            table = pq.read_table(str(pq_path))
        except Exception as e:
            logger.warning("Failed to read %s: %s (skipping)", pq_path, e)
            continue
        for i in range(table.num_rows):
            ep_idx = table["episode_index"][i].as_py()
            tasks = table["tasks"][i].as_py()
            if tasks:
                garment_by_ep[ep_idx] = tasks[0] if isinstance(tasks, (list, np.ndarray)) else tasks

    # Get success from data parquet
    data_parquets = sorted((dataset_dir / "data").rglob("*.parquet"))
    for pq_path in data_parquets:
        try:
            table = pq.read_table(str(pq_path), columns=["episode_index", "success"])
        except Exception as e:
            logger.warning("Failed to read %s: %s (skipping)", pq_path, e)
            continue
        ep_indices = table["episode_index"].to_pylist()
        successes = table["success"].to_pylist()
        for ep_idx, s in zip(ep_indices, successes):
            if ep_idx not in ep_meta:
                garment = garment_by_ep.get(ep_idx, "unknown")
                ep_meta[ep_idx] = {
                    "garment": garment,
                    "garment_type": _garment_name_to_type(garment),
                    "success": bool(s == 1.0) if not np.isnan(s) else False,
                }

    if ep_meta:
        logger.info("Derived metadata from LeRobot format for %d episodes", len(ep_meta))
    return ep_meta


def _ffprobe_nb_frames(path: Path) -> int:
    """Return number of decodable frames in a video file, or -1 if unreadable."""
    if not path.exists():
        return -1
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        return int(r.stdout.strip())
    except Exception:
        return -2


def validate_and_repair_dataset(dataset_dir: Path) -> bool:
    """Validate that video frame counts match parquet expectations.

    For LeRobot v3 datasets where multiple episodes are packed into one .mp4,
    AV1 encoding can occasionally drop trailing frames, leaving the video
    shorter than the parquet expects. Training then crashes with
    "Invalid frame index=X, must be less than Y".

    Handles both single-file and multi-file layouts:
      - multiple mp4s per camera (camera file-rotation once an mp4 hits the
        size threshold; cameras rotate independently)
      - multiple data parquets and/or multiple episodes parquets

    Per-(camera, chunk, file), ffprobe the video and tail-drop any episodes
    that overflow. Union the drops across cameras, then rewrite data /
    episodes parquets with recomputed global `index`, `dataset_from_index`,
    and `dataset_to_index` columns. Orphaned mp4s (no surviving episodes
    reference them) are deleted.
    """
    info_p = dataset_dir / "meta" / "info.json"
    if not info_p.exists():
        return True  # not a lerobot dataset, skip validation
    try:
        info = json.loads(info_p.read_text())
    except Exception as e:
        logger.warning("validate: cannot read %s: %s — skipping dataset", info_p, e)
        return False

    vid_features = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    if not vid_features:
        return True  # nothing to validate

    ep_files = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        return True
    ep_table = pa.concat_tables([pq.read_table(f) for f in ep_files])
    n_eps = ep_table.num_rows
    if n_eps == 0:
        return True

    lengths = ep_table.column("length").to_pylist()
    ep_indices = ep_table.column("episode_index").to_pylist()

    # Per camera, per (chunk, file): tail-drop episodes that overflow the mp4.
    # Each camera rotates files independently, so grouping must be per-camera
    # (not unioned across cameras, which would misattribute episodes).
    to_drop: set[int] = set()
    broken_reports: list[tuple[str, int, int, int, int]] = []
    for vf in vid_features:
        ci_col = f"videos/{vf}/chunk_index"
        fi_col = f"videos/{vf}/file_index"
        if ci_col not in ep_table.column_names:
            continue
        ci = ep_table.column(ci_col).to_pylist()
        fi = ep_table.column(fi_col).to_pylist()

        cam_files: dict[tuple[int, int], list[tuple[int, int]]] = collections.defaultdict(list)
        for row, (c, f) in enumerate(zip(ci, fi)):
            if c is None or f is None:
                continue
            cam_files[(c, f)].append((ep_indices[row], lengths[row]))

        for (c, f), eps in cam_files.items():
            vp = dataset_dir / f"videos/{vf}/chunk-{c:03d}/file-{f:03d}.mp4"
            actual = _ffprobe_nb_frames(vp)
            if actual < 0:
                logger.error("validate: %s: missing/unreadable video %s — dropping dataset",
                             dataset_dir.name, vp)
                return False
            expected = sum(L for _, L in eps)
            if actual >= expected:
                continue
            broken_reports.append((vf, c, f, actual, expected))
            offset = 0
            dropping = False
            for ep_idx, L in eps:
                if dropping or offset + L > actual:
                    to_drop.add(ep_idx)
                    dropping = True
                offset += L

    for vf, c, f, actual, expected in broken_reports:
        logger.warning("validate: %s %s (chunk=%d, file=%d): video has %d frames, "
                       "parquet expects %d — will drop overflowing episodes",
                       dataset_dir.name, vf, c, f, actual, expected)

    if not to_drop:
        return True
    if len(to_drop) >= n_eps:
        logger.error("validate: %s would drop all episodes — dropping dataset",
                     dataset_dir.name)
        return False

    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not data_files:
        logger.error("validate: %s has no data parquet — dropping", dataset_dir.name)
        return False

    # Rewrite data parquets: filter dropped episodes and renumber the global
    # `index` column so it stays contiguous 0..total_frames-1 across files.
    global_idx = 0
    for df_path in data_files:
        t = pq.read_table(df_path)
        col = t.column("episode_index").to_pylist()
        keep = pa.array([i not in to_drop for i in col])
        filtered = t.filter(keep)
        if filtered.num_rows == 0:
            df_path.unlink()
            continue
        if "index" in filtered.column_names:
            new_index = pa.array(
                list(range(global_idx, global_idx + filtered.num_rows)),
                type=t.column("index").type,
            )
            filtered = filtered.set_column(
                filtered.column_names.index("index"), "index", new_index,
            )
        global_idx += filtered.num_rows
        pq.write_table(filtered, df_path)
    total_kept_frames = global_idx

    # Rewrite episodes parquets: recompute dataset_from_index/dataset_to_index
    # (global row offsets into the concatenated data parquets) for surviving
    # episodes, walking in (data_chunk, data_file, original_from) order.
    ep_table_inputs: list[tuple[Path, pa.Table]] = [(p, pq.read_table(p)) for p in ep_files]

    all_rows: list[tuple[int, int, int, int, int]] = []
    for _, t in ep_table_inputs:
        dci = t.column("data/chunk_index").to_pylist()
        dfi = t.column("data/file_index").to_pylist()
        dfrom = t.column("dataset_from_index").to_pylist()
        eidx = t.column("episode_index").to_pylist()
        lens = t.column("length").to_pylist()
        for i in range(t.num_rows):
            if eidx[i] in to_drop:
                continue
            all_rows.append((dci[i], dfi[i], dfrom[i], eidx[i], lens[i]))
    all_rows.sort(key=lambda r: (r[0], r[1], r[2]))
    new_from: dict[int, int] = {}
    new_to: dict[int, int] = {}
    running = 0
    for _, _, _, ep_idx, L in all_rows:
        new_from[ep_idx] = running
        running += L
        new_to[ep_idx] = running

    if running != total_kept_frames:
        logger.error("validate: %s post-repair frame mismatch (data=%d, episodes=%d) — "
                     "dropping dataset", dataset_dir.name, total_kept_frames, running)
        return False

    filtered_ep_tables: list[pa.Table] = []
    for ep_path, t in ep_table_inputs:
        col = t.column("episode_index").to_pylist()
        keep_mask = [i not in to_drop for i in col]
        filtered = t.filter(pa.array(keep_mask))
        if filtered.num_rows == 0:
            ep_path.unlink()
            continue
        kept_eps = [e for e, k in zip(col, keep_mask) if k]
        from_type = t.column("dataset_from_index").type
        to_type = t.column("dataset_to_index").type
        filtered = filtered.set_column(
            filtered.column_names.index("dataset_from_index"),
            "dataset_from_index",
            pa.array([new_from[e] for e in kept_eps], type=from_type),
        )
        filtered = filtered.set_column(
            filtered.column_names.index("dataset_to_index"),
            "dataset_to_index",
            pa.array([new_to[e] for e in kept_eps], type=to_type),
        )
        pq.write_table(filtered, ep_path)
        filtered_ep_tables.append(filtered)

    # Remove eval_episode_metadata json files for dropped episodes
    md_dir = dataset_dir / EVAL_EPISODE_META_DIR
    if md_dir.exists():
        for ep_idx in sorted(to_drop):
            md = md_dir / f"episode_{ep_idx:06d}.json"
            if md.exists():
                md.unlink()

    # Delete orphan mp4s: (camera, chunk, file) no longer referenced by any
    # surviving episode. The video file is still valid on disk but unused.
    referenced: set[tuple[str, int, int]] = set()
    for t in filtered_ep_tables:
        for vf in vid_features:
            ci_col = f"videos/{vf}/chunk_index"
            fi_col = f"videos/{vf}/file_index"
            if ci_col not in t.column_names:
                continue
            ci = t.column(ci_col).to_pylist()
            fi = t.column(fi_col).to_pylist()
            for c, f in zip(ci, fi):
                if c is None or f is None:
                    continue
                referenced.add((vf, c, f))

    for vf in vid_features:
        vdir = dataset_dir / f"videos/{vf}"
        if not vdir.exists():
            continue
        for mp4 in vdir.rglob("*.mp4"):
            try:
                c = int(mp4.parent.name.split("-")[1])
                f = int(mp4.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            if (vf, c, f) not in referenced:
                mp4.unlink()
                logger.info("validate: %s: removed orphan video %s",
                            dataset_dir.name, mp4.relative_to(dataset_dir))

    new_total_eps = n_eps - len(to_drop)
    info["total_episodes"] = new_total_eps
    info["total_frames"] = total_kept_frames
    if "splits" in info and "train" in info["splits"]:
        info["splits"]["train"] = f"0:{new_total_eps}"
    info_p.write_text(json.dumps(info, indent=4))

    logger.warning("validate: %s repaired — dropped %d episodes, kept %d frames",
                   dataset_dir.name, len(to_drop), total_kept_frames)
    return True


def _find_pipeline_state_for(dataset_dir: Path) -> Path | None:
    """Walk up parent directories looking for a pipeline_state.json."""
    for parent in dataset_dir.resolve().parents:
        candidate = parent / STATE_FILENAME
        if candidate.exists():
            return candidate
    return None


def remove_dataset_from_pipeline_state(dataset_dir: Path) -> bool:
    """Remove `dataset_dir` from rl_datasets / dagger_datasets / consumed_rollout_ids
    in the enclosing pipeline_state.json. Idempotent. Returns True if state was modified."""
    state_path = _find_pipeline_state_for(dataset_dir)
    if state_path is None:
        return False
    try:
        state = load_state(state_path)
    except Exception as e:
        logger.warning("Could not load pipeline state at %s: %s", state_path, e)
        return False

    # Match by resolved absolute path AND by recorded (often-relative) string form
    target_resolved = str(dataset_dir.resolve())
    target_relative = str(dataset_dir)

    def _matches(root_str: str) -> bool:
        if root_str == target_relative or root_str == target_resolved:
            return True
        try:
            return str(Path(root_str).resolve()) == target_resolved
        except Exception:
            return False

    modified = False
    for key in ("rl_datasets", "dagger_datasets"):
        old = state.get(key, [])
        new = [d for d in old if not _matches(d.get("root", ""))]
        if len(new) != len(old):
            state[key] = new
            modified = True
            logger.warning("Pipeline state %s: removed %d entries from %s for %s",
                           state_path.name, len(old) - len(new), key, dataset_dir.name)

    consumed = state.get("consumed_rollout_ids", [])
    new_consumed = [r for r in consumed if not _matches(r)]
    if len(new_consumed) != len(consumed):
        state["consumed_rollout_ids"] = new_consumed
        modified = True

    if modified:
        save_state(state, state_path)
    return modified


def load_episodes_from_dataset(
    dataset_dir: Path,
    load_values: bool = False,
    load_check_distances: bool = False,
) -> list[dict]:
    """Load episode info from metadata + parquet for advantage computation.

    Returns list of dicts with keys needed by compute_segment_advantages / compute_gae_advantages,
    plus parquet_path and row_indices for writing back.

    Args:
        load_values: If True, also load per-frame value predictions (needed for GAE mode).
        load_check_distances: If True, store the last-frame ``check_distances``
            vector on each episode as ``last_check_distances`` (used by the
            precision boost).
    """
    meta_dir = dataset_dir / EVAL_EPISODE_META_DIR
    data_dir = dataset_dir / "data"

    # LeRobot-derived base (garment/garment_type/success from tasks+data parquet).
    # Always loaded so per-episode JSON gaps get filled from the authoritative
    # LeRobot sidecar instead of silently defaulting to garment="", garment_type="unknown".
    ep_meta: dict[int, dict] = _load_meta_from_lerobot(dataset_dir)

    # Overlay per-episode eval_episode_metadata/*.json on top of LeRobot-derived base.
    # JSON values win when present (they carry rollout_type, inference_config, etc.).
    meta_files = sorted(meta_dir.glob("episode_*.json")) if meta_dir.exists() else []
    json_idxs: set[int] = set()
    for mf in meta_files:
        ep_idx = int(mf.stem.split("_")[1])
        with open(mf) as f:
            json_meta = json.load(f)
        json_idxs.add(ep_idx)
        if ep_idx in ep_meta:
            ep_meta[ep_idx].update(json_meta)
        else:
            ep_meta[ep_idx] = json_meta

    # Warn on missing per-episode JSONs (dataset writer crashed between
    # save_episode and JSON write — data is fine but rollout_type is unknown)
    lerobot_only = sorted(set(ep_meta) - json_idxs)
    if lerobot_only and meta_files:
        logger.warning(
            "Dataset %s: %d episodes missing eval_episode_metadata JSON "
            "(filled from LeRobot tasks): %s",
            dataset_dir, len(lerobot_only),
            lerobot_only[:10] + (["..."] if len(lerobot_only) > 10 else []),
        )

    # Load parquet data
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    # Build per-episode frame data from parquet
    ep_frames: dict[int, dict] = {}  # ep_idx -> {rows, dense_rewards, values, pq_path}

    for pq_path in parquet_files:
        table = pq.read_table(str(pq_path))
        if table.num_rows == 0:
            continue

        ep_indices = table["episode_index"].to_pylist()
        has_dr = "dense_reward" in table.column_names
        dense_rewards = table["dense_reward"].to_pylist() if has_dr else None
        val_col = "success_pred" if "success_pred" in table.column_names else "value" if "value" in table.column_names else None
        has_val = val_col is not None and load_values
        value_preds = table[val_col].to_pylist() if has_val else None
        has_comp = "completion_pred" in table.column_names and load_values
        comp_preds = table["completion_pred"].to_pylist() if has_comp else None
        has_cd = "check_distances" in table.column_names and load_check_distances
        cd_values = table["check_distances"].to_pylist() if has_cd else None
        frame_indices = table["frame_index"].to_pylist() if has_cd else None

        for row_idx in range(table.num_rows):
            ep_idx = ep_indices[row_idx]
            if ep_idx not in ep_frames:
                ep_frames[ep_idx] = {
                    "row_indices": [],
                    "dense_rewards": [],
                    "values": [],
                    "completion_preds": [],
                    "pq_path": pq_path,
                }
            ep_frames[ep_idx]["row_indices"].append(row_idx)
            if dense_rewards is not None:
                dr = dense_rewards[row_idx]
                ep_frames[ep_idx]["dense_rewards"].append(
                    float(dr[0]) if isinstance(dr, list) else float(dr)
                )
            if value_preds is not None:
                v = value_preds[row_idx]
                if v is not None:
                    ep_frames[ep_idx]["values"].append(
                        float(v[0]) if isinstance(v, list) else float(v)
                    )
            if cd_values is not None:
                # Track (frame_index, check_distances) so we can pick the max-
                # frame_index row at episode-build time regardless of parquet order.
                cd_bucket = ep_frames[ep_idx].setdefault("_cd_frames", [])
                cd_bucket.append((frame_indices[row_idx], cd_values[row_idx]))
            if comp_preds is not None:
                cp_v = comp_preds[row_idx]
                if cp_v is None:
                    raise RuntimeError(
                        f"completion_pred is None at row {row_idx} of {pq_path} "
                        f"(episode {ep_idx}). Every frame must have a completion "
                        f"prediction — upstream writer or propagation is broken."
                    )
                ep_frames[ep_idx]["completion_preds"].append(
                    float(cp_v[0]) if isinstance(cp_v, list) else float(cp_v)
                )

    # Build episodes_info list
    episodes = []
    for ep_idx in sorted(ep_frames.keys()):
        meta = ep_meta.get(ep_idx, {})
        frame_data = ep_frames[ep_idx]
        drs = frame_data["dense_rewards"]
        success = meta.get("success", False)

        garment_type = meta.get("garment_type", "unknown")

        # Max cumulative reward reached during episode (before any withdrawal).
        # Used for checkpoint detection (did the episode reach the first checkpoint?).
        max_cum = 0.0
        cum = 0.0
        for dr in drs:
            cum += dr
            max_cum = max(max_cum, cum)

        # Find checkpoint_frame: first frame with cumulative reward > 0.
        checkpoint_frame = None
        cum = 0.0
        for fi, dr in enumerate(drs):
            cum += dr
            if cum > 0.4:
                checkpoint_frame = fi
                break

        # dense_return = binary: 1.0 (success) or 0.0 (failure).
        # With failure withdrawal, intermediate checkpoints don't affect the total.
        dense_return = 1.0 if success else 0.0

        ep_info = {
            "ep_idx": ep_idx,
            "garment_type": garment_type,
            "dense_return": dense_return,
            "max_reward": max_cum,  # peak reward reached (for checkpoint detection)
            "num_frames": len(frame_data["row_indices"]),
            "checkpoint_frame": checkpoint_frame,
            "success": success,
            "garment": meta.get("garment", ""),
            "rollout_type": meta.get("rollout_type", "normal"),
            "pq_path": frame_data["pq_path"],
            "row_indices": frame_data["row_indices"],
            "dataset_dir": str(dataset_dir),
        }
        cd_frames = frame_data.get("_cd_frames")
        if cd_frames:
            last = max(cd_frames, key=lambda t: t[0])[1]
            ep_info["last_check_distances"] = last
        if load_values:
            # NOTE: failure withdrawal is already applied by dataset_writer during
            # episode collection (compute_dense_reward).  Do NOT re-apply here —
            # double withdrawal would make failed episode rewards negative instead
            # of zero.
            ep_info["dense_rewards"] = drs
            values = frame_data["values"]
            # Replace NaN values with success label (e.g. dagger data has no value predictions)
            if values and any(np.isnan(v) for v in values):
                fill_val = 1.0 if success else 0.0
                values = [fill_val if np.isnan(v) else v for v in values]
            ep_info["values"] = values
            # Completion predictions for PBRS shaping. Must cover every frame with
            # finite values — a partial/NaN column means upstream propagation is broken.
            comp_list = frame_data.get("completion_preds", [])
            if comp_list:
                if len(comp_list) != len(values):
                    raise RuntimeError(
                        f"completion_preds length mismatch for episode {ep_idx} in "
                        f"{dataset_dir}: comp={len(comp_list)}, values={len(values)}"
                    )
                if any(np.isnan(v) for v in comp_list):
                    raise RuntimeError(
                        f"completion_preds contains NaN for episode {ep_idx} in "
                        f"{dataset_dir} — upstream writer or propagation is broken."
                    )
                ep_info["completion_preds"] = comp_list
        episodes.append(ep_info)

    return episodes


# Only these rollout types are used for success rate / baseline computation (unbiased)
_SR_INCLUDED_ROLLOUT_TYPES = {"normal", "random", "full", "curriculum"}
# Minimum sampling_share for an episode to be included in SR estimation.
# Datasets with weight below this are too old to reflect current policy performance.
_SR_MIN_WEIGHT = 0.9


@dataclasses.dataclass
class GarmentStats:
    """Centralized per-garment statistics from unbiased, recent episodes.

    Computed once by ``compute_garment_stats`` and reused by advantage
    computation, debiasing, success rate saving, and statistics display.
    """
    # Per-garment averages (garment_name -> float)
    avg_success: dict[str, float]
    avg_checkpoint: dict[str, float]
    # Per-garment episode counts
    episode_counts: dict[str, int]
    # garment_name -> garment_type key
    garment_to_type: dict[str, str]
    # Per-type SR (garment-level average)
    type_sr: dict[str, float]
    # Number of filtered (eligible) episodes
    n_eligible: int
    # Number of episodes skipped due to low weight
    n_skipped_weight: int


def compute_garment_stats(all_episodes: list[dict]) -> GarmentStats:
    """Single-pass computation of per-garment stats from unbiased, recent episodes.

    Only includes rollout types in _SR_INCLUDED_ROLLOUT_TYPES and episodes with
    sampling_share >= _SR_MIN_WEIGHT. All downstream consumers should use the
    returned GarmentStats instead of recomputing.
    """
    from lehome_solution.eval.dataset_writer import _GTYPE_TO_KEY

    garment_successes: dict[str, list[float]] = {}
    garment_checkpoints: dict[str, list[float]] = {}
    garment_to_type: dict[str, str] = {}
    n_skipped_weight = 0

    for ep in all_episodes:
        rollout_type = ep.get("rollout_type", "normal")
        if rollout_type not in _SR_INCLUDED_ROLLOUT_TYPES:
            continue
        if ep.get("sampling_share", 1.0) < _SR_MIN_WEIGHT:
            n_skipped_weight += 1
            continue
        garment = ep.get("garment", "unknown")
        garment_successes.setdefault(garment, []).append(float(ep.get("success", 0.0)))
        max_r = ep.get("max_reward", ep.get("dense_return", 0.0))
        cp = 1.0 if (max_r >= 0.5 or ep.get("success", False)) else 0.0
        garment_checkpoints.setdefault(garment, []).append(cp)
        if garment not in garment_to_type:
            garment_to_type[garment] = _GTYPE_TO_KEY.get(ep["garment_type"], ep["garment_type"])

    avg_success = {g: float(np.mean(v)) for g, v in garment_successes.items()}
    avg_checkpoint = {g: float(np.mean(v)) for g, v in garment_checkpoints.items()}
    episode_counts = {g: len(v) for g, v in garment_successes.items()}
    n_eligible = sum(episode_counts.values())

    # Type SR = simple average of garment SRs within each type
    type_garment_srs: dict[str, list[float]] = {}
    for garment, sr in avg_success.items():
        gtype = garment_to_type[garment]
        type_garment_srs.setdefault(gtype, []).append(sr)
    type_sr = {k: float(np.mean(v)) for k, v in type_garment_srs.items()}

    # Logging
    if n_skipped_weight:
        logger.info("Garment stats: skipped %d episodes with weight < %.2f",
                     n_skipped_weight, _SR_MIN_WEIGHT)
    logger.info("Per-garment avg success rates (%d garments, %d episodes from %s): %s",
                len(avg_success), n_eligible, _SR_INCLUDED_ROLLOUT_TYPES,
                {k: f"{v:.3f}" for k, v in sorted(avg_success.items())})
    logger.info("Per-garment avg checkpoint rates (%d garments): %s",
                len(avg_checkpoint),
                {k: f"{v:.3f}" for k, v in sorted(avg_checkpoint.items())})
    logger.info("Per-type success rates (garment-level avg): %s",
                {k: f"{v:.1%}" for k, v in sorted(type_sr.items())})
    for gtype in sorted(type_garment_srs.keys()):
        garments_of_type = [(g, avg_success[g], episode_counts[g])
                            for g in sorted(avg_success) if garment_to_type[g] == gtype]
        details = ", ".join(f"{g}={sr:.1%}({n}ep)" for g, sr, n in garments_of_type)
        type_total = sum(n for _, _, n in garments_of_type)
        logger.info("  %s (%d episodes, %d garments): %s",
                     gtype, type_total, len(garments_of_type), details)

    return GarmentStats(
        avg_success=avg_success,
        avg_checkpoint=avg_checkpoint,
        episode_counts=episode_counts,
        garment_to_type=garment_to_type,
        type_sr=type_sr,
        n_eligible=n_eligible,
        n_skipped_weight=n_skipped_weight,
    )


# Cap on debias weight; prevents huge multipliers when one outcome is rare.
_MAX_DEBIAS_WEIGHT = 2.0


def _effective_episode_weight(
    ep: dict,
    rl_advantages: list[float] | None,
    garment_sr: dict[str, float],
    bc_fr_scale: float,
) -> float:
    """Effective training weight contributed by this episode.

    Reflects the expected number of frames sampled from this episode during
    training, so per-garment success/checkpoint ratios match training reality.

    RL:       sampling_share * Σ_{frames} exp(clip(adv_frame))
              (advantage-prioritised sampling; sum correlates with outcome).
    BC/DAg:   sampling_share * ep_len * exp(fr_scale * (1 - SR_g))
              (FR-proportional sampling; constant across frames).
    Fallback: sampling_share * ep_len (uniform).
    """
    sampling_share = float(ep.get("sampling_share", 1.0))
    rollout_type = ep.get("rollout_type", "normal")

    if rollout_type in ("bc", "dagger"):
        ep_len = int(ep.get("ep_len", 1) or 1)
        garment = ep.get("garment", "unknown")
        sr = garment_sr.get(garment, 0.5)
        fr_boost = float(np.exp(bc_fr_scale * (1.0 - sr)))
        return sampling_share * ep_len * fr_boost

    # RL (any rollout type): use per-frame exp(clip(adv)) if available.
    if rl_advantages is not None and len(rl_advantages) > 0:
        advs = np.asarray(rl_advantages, dtype=np.float32)
        # Advantages are already clipped before being written; clamp defensively.
        return sampling_share * float(np.exp(np.clip(advs, -2.0, 2.0)).sum())

    ep_len = int(ep.get("ep_len", 1) or 1)
    return sampling_share * ep_len


def compute_garment_debias(
    all_train_episodes: list[dict],
    all_advantages: list[list[float]] | None,
    garment_stats: GarmentStats,
    bc_fr_scale: float,
) -> dict[str, dict[str, float]]:
    """Per-garment debias weights: success and checkpoint multipliers.

    Target SR/CP per garment come from the pre-computed ``garment_stats``
    (unbiased RL subset). Weighted counts use every training source (RL +
    BC + DAgger), weighted by the expected training-sampling frequency so the
    debiased rate matches training reality.

    Returns:
        {garment_name: {"success_w": float, "checkpoint_w": float}}
    """
    per_garment_counts: dict[str, dict[str, float]] = {}

    for i, ep in enumerate(all_train_episodes):
        garment = ep.get("garment", "unknown")
        c = per_garment_counts.setdefault(garment, {
            "s_all": 0.0, "f_all": 0.0, "cp_all": 0.0, "ncp_all": 0.0,
        })
        advs = all_advantages[i] if all_advantages is not None and i < len(all_advantages) else None
        ew = _effective_episode_weight(ep, advs, garment_stats.avg_success, bc_fr_scale)

        success = bool(ep.get("success", False))
        max_r = ep.get("max_reward", ep.get("dense_return", 0.0))
        cp_reached = bool(max_r >= 0.5 or success)

        if success:
            c["s_all"] += ew
        else:
            c["f_all"] += ew
        if cp_reached:
            c["cp_all"] += ew
        else:
            c["ncp_all"] += ew

    debias: dict[str, dict[str, float]] = {}
    for garment, c in per_garment_counts.items():
        # Target SR/CP from the pre-computed unbiased RL subset.
        # Fallback to 0.5 if a garment was never seen in random/full rollouts.
        sr = garment_stats.avg_success.get(garment, 0.5)
        cp = garment_stats.avg_checkpoint.get(garment, 0.5)

        s_all, f_all = c["s_all"], c["f_all"]
        cp_all, ncp_all = c["cp_all"], c["ncp_all"]

        # Debias formula: w = SR · f_all / ((1 − SR) · s_all), capped at _MAX_DEBIAS_WEIGHT.
        # Degenerate cases: SR ≥ 1 or s_all = 0 → formula diverges, clamp to max cap.
        #                   f_all = 0 → formula = 0, successes can't be balanced against
        #                               nonexistent failures; neutral weight 1.0 is safest.
        if s_all <= 0.0 or sr >= 1.0:
            w_success = _MAX_DEBIAS_WEIGHT
        elif f_all <= 0.0 or sr <= 0.0:
            w_success = 1.0
        else:
            w_success = (sr * f_all) / ((1.0 - sr) * s_all)

        if cp_all <= 0.0 or cp >= 1.0:
            w_checkpoint = _MAX_DEBIAS_WEIGHT
        elif ncp_all <= 0.0 or cp <= 0.0:
            w_checkpoint = 1.0
        else:
            w_checkpoint = (cp * ncp_all) / ((1.0 - cp) * cp_all)

        debias[garment] = {
            "success_w": float(min(w_success, _MAX_DEBIAS_WEIGHT)),
            "checkpoint_w": float(min(w_checkpoint, _MAX_DEBIAS_WEIGHT)),
        }

    if debias:
        sw = [v["success_w"] for v in debias.values()]
        cw = [v["checkpoint_w"] for v in debias.values()]
        logger.info(
            "Garment debias: %d garments, success_w mean=%.3f min=%.3f max=%.3f, "
            "checkpoint_w mean=%.3f min=%.3f max=%.3f",
            len(debias),
            float(np.mean(sw)), float(np.min(sw)), float(np.max(sw)),
            float(np.mean(cw)), float(np.min(cw)), float(np.max(cw)),
        )
        for g in sorted(debias):
            logger.info(
                "  %s: s_w=%.3f cp_w=%.3f (target SR=%.3f, CP=%.3f)",
                g, debias[g]["success_w"], debias[g]["checkpoint_w"],
                garment_stats.avg_success.get(g, 0.5),
                garment_stats.avg_checkpoint.get(g, 0.5),
            )
    return debias


def save_garment_debias(all_episodes: list[dict], debias: dict[str, dict[str, float]]) -> None:
    """Write garment_debias.json next to success_rates.json in every dataset dir."""
    dataset_dirs: set[str] = set()
    for ep in all_episodes:
        pq_path = Path(ep.get("pq_path", ""))
        if pq_path:
            dataset_dirs.add(str(pq_path.parent.parent.parent))

    for ds_dir in dataset_dirs:
        out_path = Path(ds_dir) / "garment_debias.json"
        with open(out_path, "w") as f:
            json.dump(debias, f, indent=2)
        logger.info("Saved garment_debias.json to %s", ds_dir)


def load_bc_dagger_episodes(root: Path, rollout_type: str, sampling_share: float) -> list[dict]:
    """Load BC/DAgger episode stubs (success=True, checkpoint=reached) for debias.

    Only reads meta/episodes parquets — full data parquets are not needed.
    Each episode returns a dict with: garment, garment_type, ep_len, success=True,
    max_reward=1.0, dense_return=1.0, sampling_share, rollout_type ∈ {"bc","dagger"}.
    """
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        logger.warning("BC/DAgger meta/episodes not found at %s — skipping", meta_dir)
        return []

    # meta/episodes/*.parquet contain (episode_index, tasks[], length, ...)
    ep_parquets = sorted(meta_dir.rglob("*.parquet"))
    episodes: list[dict] = []
    for pq_path in ep_parquets:
        try:
            table = pq.read_table(str(pq_path))
        except Exception as e:
            logger.warning("Failed to read %s: %s (skipping)", pq_path, e)
            continue
        cols = table.column_names
        if "episode_index" not in cols or "length" not in cols:
            logger.warning("%s missing expected columns (need episode_index, length)", pq_path)
            continue
        tasks_col = table["tasks"].to_pylist() if "tasks" in cols else [None] * table.num_rows
        lengths = table["length"].to_pylist()
        for ep_idx, tasks, length in zip(
            table["episode_index"].to_pylist(), tasks_col, lengths
        ):
            task0 = ""
            if tasks:
                task0 = tasks[0] if isinstance(tasks, (list, tuple, np.ndarray)) else str(tasks)
            garment = task0 or "unknown"
            gtype = _garment_name_to_type(garment)
            episodes.append({
                "garment": garment,
                "garment_type": gtype,
                "ep_len": int(length) if length is not None else 0,
                "success": True,
                "max_reward": 1.0,
                "dense_return": 1.0,
                "sampling_share": sampling_share,
                "rollout_type": rollout_type,
                # pq_path left empty so save_garment_debias won't try to write into
                # the BC/DAgger dataset dir (garment_debias.json belongs with RL data).
                "pq_path": "",
            })
    return episodes


def update_parquet_files(
    all_episodes: list[dict],
    all_advantages: list[list[float]],
    garment_avg_success: dict[str, float] | None = None,
    garment_avg_checkpoint: dict[str, float] | None = None,
) -> None:
    """Write advantages + per-garment label smoothing baselines to parquet files.

    Per-garment debias weights are now stored in garment_debias.json (one per
    dataset dir), loaded by the data loader at init time — no per-episode
    weight columns are written to parquet anymore.
    """
    # Pre-compute per-episode garment averages
    ep_avg_success: dict[int, float] = {}
    ep_avg_checkpoint: dict[int, float] = {}
    for i, ep in enumerate(all_episodes):
        garment = ep.get("garment", "unknown")
        if garment_avg_success is not None:
            ep_avg_success[i] = garment_avg_success.get(garment, 0.5)
        if garment_avg_checkpoint is not None:
            ep_avg_checkpoint[i] = garment_avg_checkpoint.get(garment, 0.5)

    # Group updates by parquet path
    updates_by_pq: dict[str, list[tuple[int, list[int], list[float]]]] = {}
    for i, (ep, advs) in enumerate(zip(all_episodes, all_advantages)):
        pq_key = str(ep["pq_path"])
        updates_by_pq.setdefault(pq_key, []).append(
            (i, ep["row_indices"], advs)
        )

    for pq_path_str, updates in updates_by_pq.items():
        pq_path = Path(pq_path_str)
        table = pq.read_table(pq_path_str)

        has_advantage = "advantage" in table.column_names

        if not has_advantage:
            logger.warning("Missing advantage column in %s, skipping", pq_path)
            continue

        old_advantages = table["advantage"].to_pylist()
        new_advantages = list(old_advantages)

        # Prepare garment_avg_success / garment_avg_checkpoint columns
        # (backward compat: also handle legacy type_avg_reward column)
        def _init_column(table, col_name, old_advantages):
            if col_name in table.column_names:
                return list(table[col_name].to_pylist())
            is_list = isinstance(old_advantages[0], list) if old_advantages else True
            return [[0.5] if is_list else 0.5] * table.num_rows

        new_gas = _init_column(table, "garment_avg_success", old_advantages)
        new_gac = _init_column(table, "garment_avg_checkpoint", old_advantages)
        for ep_i, row_indices, advs in updates:
            is_list = isinstance(old_advantages[row_indices[0]], list) if row_indices else True
            for local_fi, row_idx in enumerate(row_indices):
                adv = advs[local_fi] if local_fi < len(advs) else 0.0
                new_advantages[row_idx] = [adv] if is_list else adv
                if ep_i in ep_avg_success:
                    v = ep_avg_success[ep_i]
                    new_gas[row_idx] = [v] if is_list else v
                if ep_i in ep_avg_checkpoint:
                    v = ep_avg_checkpoint[ep_i]
                    new_gac[row_idx] = [v] if is_list else v

        # Replace advantage column
        col_names = table.column_names
        adv_idx = col_names.index("advantage")
        table = table.drop("advantage")
        table = table.add_column(adv_idx, "advantage", pa.array(new_advantages))

        # Add/replace garment_avg columns
        for col_name, new_vals in [
            ("garment_avg_success", new_gas), ("garment_avg_checkpoint", new_gac),
        ]:
            if col_name in table.column_names:
                idx = table.column_names.index(col_name)
                table = table.drop(col_name)
                table = table.add_column(idx, col_name, pa.array(new_vals))
            else:
                table = table.append_column(col_name, pa.array(new_vals))

        # Drop legacy columns (migrated to JSON / per-garment files).
        # Any column NOT explicitly replaced/dropped above (e.g. check_distances,
        # success, dense_reward, checkpoint_held, ...) passes through unchanged —
        # keep this list tightly scoped so additive features survive recompute.
        for legacy in ("type_avg_reward", "success_debias_weight", "checkpoint_debias_weight"):
            if legacy in table.column_names:
                table = table.drop(legacy)

        # Atomic write: write to temp file then rename to avoid race with data loader
        tmp_path = pq_path_str + ".tmp"
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, pq_path_str)
        ep_count = len(updates)
        logger.info("Updated %s: %d episodes, %d rows", pq_path.name, ep_count, table.num_rows)


def _update_info_json_features(all_episodes: list[dict]) -> None:
    """Add garment_avg_success/garment_avg_checkpoint to info.json features if missing."""
    dataset_roots: set[str] = set()
    for ep in all_episodes:
        pq_path = Path(ep["pq_path"])
        dataset_roots.add(str(pq_path.parent.parent.parent))

    new_features = {
        "garment_avg_success": {"dtype": "float64", "shape": [1], "names": ["garment_avg_success"]},
        "garment_avg_checkpoint": {"dtype": "float64", "shape": [1], "names": ["garment_avg_checkpoint"]},
    }
    legacy_features = ("type_avg_reward", "success_debias_weight", "checkpoint_debias_weight")

    for root in dataset_roots:
        info_path = Path(root) / "meta" / "info.json"
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        features = info.get("features", {})
        updated = False
        for feat_name, feat_spec in new_features.items():
            if feat_name not in features:
                features[feat_name] = feat_spec
                updated = True
        for feat_name in legacy_features:
            if feat_name in features:
                del features[feat_name]
                updated = True
        if updated:
            info["features"] = features
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)
            logger.info("Updated %s: garment_avg_success/checkpoint features", info_path)


def success_rates_from_stats(stats: GarmentStats) -> dict:
    """Extract success rates dict from pre-computed GarmentStats.

    Returns dict with:
        "by_type": {garment_type -> SR}
        "by_garment": {garment_name -> SR}
    """
    return {"by_type": stats.type_sr, "by_garment": stats.avg_success}


def save_success_rates(all_episodes: list[dict], success_rates: dict) -> None:
    """Save success_rates.json to each eval dataset directory."""
    from lehome_solution.eval.rollout_strategies import SUCCESS_RATES_FILE

    # Find unique dataset dirs from episodes
    dataset_dirs: set[str] = set()
    for ep in all_episodes:
        pq_path = Path(ep["pq_path"])
        # pq files are at data/chunk-NNN/file.parquet, dataset dir is 2 levels up
        ds_dir = pq_path.parent.parent.parent
        dataset_dirs.add(str(ds_dir))

    for ds_dir in dataset_dirs:
        out_path = Path(ds_dir) / SUCCESS_RATES_FILE
        with open(out_path, "w") as f:
            json.dump(success_rates, f, indent=2)
        logger.info("Saved %s to %s", SUCCESS_RATES_FILE, ds_dir)


def print_statistics(
    all_episodes: list[dict],
    all_advantages: list[list[float]],
    stats: GarmentStats,
    mode: str = "segment",
) -> None:
    """Print per-type advantage statistics.

    SR/CP stats come from the pre-computed GarmentStats (unbiased, recent episodes).
    Advantage stats use all episodes.
    """
    from lehome_solution.eval.dataset_writer import _GTYPE_TO_KEY, TYPE_NUM_CHECKPOINTS

    # Group by garment type
    groups: dict[str, list[int]] = {}
    for i, ep in enumerate(all_episodes):
        key = _GTYPE_TO_KEY.get(ep["garment_type"], ep["garment_type"])
        groups.setdefault(key, []).append(i)

    logger.info("=" * 70)
    logger.info("ADVANTAGE STATISTICS (mode=%s)", mode)
    logger.info("=" * 70)

    all_flat_advs = []
    for gtype, indices in sorted(groups.items()):
        num_cp = TYPE_NUM_CHECKPOINTS.get(gtype, 1)
        advs = [all_advantages[i] for i in indices]

        # SR from pre-computed stats
        type_sr = stats.type_sr.get(gtype, 0.0)
        garments_of_type = [g for g, t in stats.garment_to_type.items() if t == gtype]
        n_sr_eps = sum(stats.episode_counts.get(g, 0) for g in garments_of_type)

        # --- Advantage stats (all episodes) ---
        flat_advs = [a for adv_list in advs for a in adv_list]
        all_flat_advs.extend(flat_advs)
        flat_arr = np.array(flat_advs) if flat_advs else np.array([0.0])

        logger.info("")
        logger.info("  %s (%d episodes total, %d for SR, %d garments, %d checkpoints):",
                     gtype, len(indices), n_sr_eps, len(garments_of_type), num_cp)
        logger.info("    Success rate: %.1f%% (garment-avg over %d garments)",
                     100 * type_sr, len(garments_of_type))
        logger.info("    Advantage:     mean=%.3f std=%.3f min=%.3f max=%.3f",
                     flat_arr.mean(), flat_arr.std(), flat_arr.min(), flat_arr.max())
        pct_pos = (flat_arr > 0).mean() * 100
        pct_neg = (flat_arr < 0).mean() * 100
        logger.info("    Advantage %%:   positive=%.1f%% negative=%.1f%%", pct_pos, pct_neg)

        # Show a few examples
        for i, idx in enumerate(indices[:3]):
            ep = all_episodes[idx]
            adv = all_advantages[idx]
            adv_arr = np.array(adv) if adv else np.array([0.0])
            logger.info("      ep%d: %s success=%s max_r=%.3f cp_frame=%s adv min=%.4f max=%.4f mean=%.4f med=%.4f",
                         ep["ep_idx"], ep["garment"], ep["success"],
                         ep.get("max_reward", ep["dense_return"]), ep.get("checkpoint_frame"),
                         adv_arr.min(), adv_arr.max(), adv_arr.mean(), float(np.median(adv_arr)))

    all_flat_arr = np.array(all_flat_advs) if all_flat_advs else np.array([0.0])
    logger.info("")
    logger.info("  OVERALL (%d episodes, %d frames):", len(all_episodes), len(all_flat_advs))
    logger.info("    Advantage: mean=%.4f std=%.4f min=%.4f max=%.4f",
                 all_flat_arr.mean(), all_flat_arr.std(), all_flat_arr.min(), all_flat_arr.max())
    # Overall SR from centralized stats (unbiased, recent only)
    if stats.type_sr:
        overall_sr = float(np.mean(list(stats.type_sr.values())))
        logger.info("    Filtered SR: %.1f%% (type-avg over %d types, %d episodes)",
                     100 * overall_sr, len(stats.type_sr), stats.n_eligible)
    logger.info("=" * 70)


def _rl_dataset_dirs_from_config(config_name: str) -> list[tuple[str, float]]:
    """Extract non-BC eval dataset root paths and sampling shares from a training config.

    Returns list of (root_path, sampling_share) tuples.
    """
    from lehome_solution.training.config import get_config

    config = get_config(config_name)
    data_cfg = config.data
    if not hasattr(data_cfg, "data_sources") or data_cfg.data_sources is None:
        logger.error("Config %s has no data_sources", config_name)
        return []

    dirs = []
    for spec in data_cfg.data_sources:
        if spec.is_bc or getattr(spec, "is_real", False):
            continue
        if not Path(spec.root).exists():
            logger.warning("Data source root not found (skipping): %s", spec.root)
            continue
        dirs.append((spec.root, spec.sampling_share))
    return dirs


def _bc_dagger_dirs_from_config(config_name: str) -> list[tuple[str, str, float]]:
    """Extract BC + DAgger dataset roots from a training config.

    Returns list of (root, rollout_type, sampling_share) — rollout_type is
    one of {"bc", "dagger"}.
    """
    from lehome_solution.training.config import get_config

    config = get_config(config_name)
    data_cfg = config.data
    if not hasattr(data_cfg, "data_sources") or data_cfg.data_sources is None:
        return []

    out: list[tuple[str, str, float]] = []
    for spec in data_cfg.data_sources:
        if not (spec.is_bc or getattr(spec, "is_dagger", False)):
            continue
        if getattr(spec, "is_real", False):
            # Real BC sources have no garment_avg_success / debias state; they
            # never enter the BC/DAgger debias recompute path.
            continue
        if not Path(spec.root).exists():
            logger.warning("BC/DAgger data source root not found (skipping): %s", spec.root)
            continue
        rollout_type = "dagger" if getattr(spec, "is_dagger", False) else "bc"
        out.append((spec.root, rollout_type, spec.sampling_share))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Recompute advantages using segment-level or GAE normalization"
    )
    parser.add_argument(
        "--eval_dirs", type=str, nargs="*", default=None,
        help="Eval dataset directories (e.g. outputs/eval_videos/rl_XXX/eval_dataset)",
    )
    parser.add_argument(
        "--config_name", type=str, default=None,
        help="Training config name. Auto-discovers all non-BC data_sources. "
             "Can be combined with --eval_dirs (union of both).",
    )
    parser.add_argument(
        "--sampling_shares", type=float, nargs="*", default=None,
        help="Sampling weights for --eval_dirs (one per dir, same order). "
             "Used for weighted baseline in advantage normalization. "
             "If --config_name is used, weights come from data_sources.",
    )
    parser.add_argument(
        "--mode", type=str, default="segment", choices=["segment", "gae"],
        help="Advantage computation mode: 'segment' (cross-episode segment-level) "
             "or 'gae' (per-episode GAE with value predictions). Default: segment.",
    )
    parser.add_argument("--clip_min", type=float, default=-2.0)
    parser.add_argument("--clip_max", type=float, default=2.0)
    parser.add_argument("--gae_lambda", type=float, default=0.99,
                        help="GAE lambda (only used in gae mode). Default: 0.99.")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="Discount factor (only used in gae mode). Default: 0.999.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="AWR temperature: global_std is multiplied by beta before "
                             "normalizing advantages. Higher = more uniform sampling. Default: 1.0.")
    parser.add_argument("--success_alpha", type=float, default=0.5,
                        help="Dampening coefficient on the full value baseline "
                             "V̂ = α_s·(P_success − R_so_far) in the GAE_success delta: "
                             "δ = r + γ·V̂' − V̂. Equivalently r is damped by (1−α_s·γ) "
                             "while the value transition is scaled by α_s. Only used in "
                             "gae mode. Default: 0.5.")
    parser.add_argument("--completion_alpha", type=float, default=0.5,
                        help="Weight on the completion shaping term, a separate "
                             "potential-based shaping over the completion head: "
                             "δ_c = (γ·C' − C)·α_c. Added to the blended advantage at full "
                             "strength (decoupled from GAE/segment blend). 0.0 disables. "
                             "Only used in gae mode. Default: 0.5.")
    parser.add_argument("--value_tail_k", type=int, default=0,
                        help="Linearly interpolate EMA(P_success) over the last K frames "
                             "to the true success label. Cancels the 'almost-success' bias "
                             "near episode end. 0 disables. Only used in gae mode. Default: 0.")
    parser.add_argument(
        "--config_file", type=str, default=None,
        help="Path to pickled AdvantageConfig (alternative to individual CLI args). "
             "CLI args still override values from the config file.",
    )
    parser.add_argument(
        "--force_segment_dirs", type=str, nargs="*", default=None,
        help="Dataset directories to force segment-only advantages (sampling_share=0). "
             "Used for init datasets with stale value predictions.",
    )
    parser.add_argument(
        "--precision_boost", type=float, default=0.0,
        help="Additive advantage applied to every frame of the top-K%% of "
             "successful episodes PER GARMENT, ranked by binding-condition "
             "margin on the final check_distances. 0 disables.",
    )
    parser.add_argument(
        "--precision_top_k", type=float, default=0.20,
        help="Per-garment fraction of successful episodes to boost.",
    )
    parser.add_argument(
        "--precision_min_successes", type=int, default=5,
        help="Skip precision boost for garments with fewer successes than this.",
    )
    args = parser.parse_args()

    # Load pickled AdvantageConfig if provided (CLI args override)
    if args.config_file:
        import pickle
        with open(args.config_file, "rb") as f:
            adv_cfg = pickle.load(f)
        # Apply defaults from config, let explicit CLI args win
        for attr in ("mode", "gamma", "gae_lambda", "clip_max", "beta",
                     "success_alpha", "completion_alpha", "value_tail_k"):
            cli_default = parser.get_default(attr if attr != "mode" else attr)
            cli_val = getattr(args, attr if attr != "mode" else attr, None)
            cfg_val = getattr(adv_cfg, attr, None)
            if cfg_val is not None and cli_val == cli_default:
                setattr(args, attr, cfg_val)
        # weighting_clipping tuple -> clip_min/clip_max
        wc = getattr(adv_cfg, "weighting_clipping", None)
        if wc is not None:
            if args.clip_min == parser.get_default("clip_min"):
                args.clip_min = wc[0]
            if args.clip_max == parser.get_default("clip_max"):
                args.clip_max = wc[1]

    load_values = args.mode == "gae"

    # Collect eval dataset dirs from both sources
    # Map resolved path -> sampling_share (from config or default 1.0)
    dir_weights: dict[str, float] = {}
    eval_dirs = args.eval_dirs or []
    cli_weights = args.sampling_shares or []
    for i, d in enumerate(eval_dirs):
        w = cli_weights[i] if i < len(cli_weights) else 1.0
        dir_weights[str(Path(d).resolve())] = w

    if args.config_name:
        config_dirs = _rl_dataset_dirs_from_config(args.config_name)
        if config_dirs:
            logger.info("Config %s: found %d RL dataset dirs", args.config_name, len(config_dirs))
            for d, share in config_dirs:
                logger.info("  %s (share=%.3f)", d, share)
        # Add config dirs, avoiding duplicates (config share takes priority)
        for d, share in config_dirs:
            resolved = str(Path(d).resolve())
            dir_weights[resolved] = share

    if not dir_weights:
        logger.error("No eval dirs specified. Use --eval_dirs and/or --config_name.")
        return

    # Resolve force_segment_dirs for matching
    force_segment_resolved = set()
    if args.force_segment_dirs:
        for d in args.force_segment_dirs:
            force_segment_resolved.add(str(Path(d).resolve()))
        logger.info("Force segment-only for %d dirs: %s", len(force_segment_resolved),
                     [str(d) for d in force_segment_resolved])

    # Load all episodes across datasets, attaching sampling_share
    all_episodes = []
    for eval_dir, weight in dir_weights.items():
        p = Path(eval_dir)
        if not p.exists():
            logger.error("Dataset not found: %s", p)
            continue
        if not validate_and_repair_dataset(p):
            logger.error("Dataset failed validation, skipping: %s", p)
            remove_dataset_from_pipeline_state(p)
            continue
        try:
            eps = load_episodes_from_dataset(
                p,
                load_values=load_values,
                load_check_distances=(args.precision_boost > 0),
            )
        except Exception as e:
            logger.warning("Skipping dataset %s: %s", p, e)
            continue
        # Attach sampling_share (used for SR/baseline filtering) and force_segment flag
        # (used only for GAE blend weight). force_segment does NOT affect SR eligibility.
        is_force_segment = str(p.resolve()) in force_segment_resolved
        for ep in eps:
            ep["sampling_share"] = weight
            if is_force_segment:
                ep["force_segment"] = True
        if is_force_segment:
            logger.info("Loaded %d episodes from %s (SEGMENT-ONLY, weight=%.3f)", len(eps), p, weight)
        else:
            logger.info("Loaded %d episodes from %s (weight=%.3f)", len(eps), p, weight)
        all_episodes.extend(eps)

    if not all_episodes:
        logger.error("No episodes loaded")
        return

    logger.info("Total: %d episodes across %d datasets (mode=%s)", len(all_episodes), len(dir_weights), args.mode)
    if args.mode == "gae":
        eps_with_comp = sum(1 for ep in all_episodes if ep.get("completion_preds"))
        logger.info(
            "GAE alphas: success_alpha=%.3f, completion_alpha=%.3f "
            "(%d/%d episodes have completion predictions)",
            args.success_alpha, args.completion_alpha, eps_with_comp, len(all_episodes),
        )
    if args.mode == "gae" and args.value_tail_k > 0:
        logger.info("Value tail correction enabled: K=%d frames", args.value_tail_k)

    # Centralized per-garment stats FIRST (needed for GAE EMA init)
    garment_stats = compute_garment_stats(all_episodes)

    # Compute advantages
    if args.mode == "gae":
        # Verify all episodes have values
        missing_vals = sum(1 for ep in all_episodes if not ep.get("values"))
        if missing_vals > 0:
            logger.warning("%d/%d episodes have no value predictions. "
                          "Use --mode segment for those datasets.",
                          missing_vals, len(all_episodes))
        all_advantages = compute_gae_advantages(
            all_episodes,
            lam=args.gae_lambda,
            gamma=args.gamma,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            garment_success_rates=garment_stats.avg_success,
            beta=args.beta,
            success_alpha=args.success_alpha,
            completion_alpha=args.completion_alpha,
            value_tail_k=args.value_tail_k,
        )
    else:
        all_advantages = compute_segment_advantages(
            all_episodes,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            beta=args.beta,
        )

    # Per-garment precision boost (additive advantage on final-margin top-K%).
    # Runs AFTER advantage computation and BEFORE debias so the boosted
    # sampling distribution flows through _effective_episode_weight.
    if args.precision_boost > 0 and args.precision_top_k > 0:
        try:
            from lehome_solution.training.precision_boost import (
                apply_precision_boost,
                load_garment_success_distances,
            )
            gsd = load_garment_success_distances()
            n_boosted, pb_info = apply_precision_boost(
                all_episodes, all_advantages, gsd,
                top_k=args.precision_top_k,
                boost=args.precision_boost,
                min_successes=args.precision_min_successes,
            )
            logger.info(
                "Precision boost: +%.2f adv to %d episodes "
                "(top %.0f%% per garment, min %d successes)",
                args.precision_boost, n_boosted,
                100 * args.precision_top_k, args.precision_min_successes,
            )
            skipped = sorted(g for g, v in pb_info.items() if v.get("skipped"))
            if skipped:
                logger.info("  skipped (< %d successes): %s",
                            args.precision_min_successes, skipped)
            for g in sorted(pb_info):
                v = pb_info[g]
                if v.get("threshold") is None:
                    continue
                logger.info("  %-22s n_success=%3d  thr=%+.3f  boosted=%2d",
                            g, v["n_success"], v["threshold"], v["n_boosted"])
        except Exception as e:
            logger.error("Precision boost failed: %s", e, exc_info=True)

    # Save success rates for curriculum sampling
    success_rates = success_rates_from_stats(garment_stats)
    save_success_rates(all_episodes, success_rates)

    # Print statistics
    print_statistics(all_episodes, all_advantages, garment_stats, mode=args.mode)

    # Per-garment debias weights (includes BC/DAgger if config is provided).
    # BC/DAgger episodes are treated as always-success/always-checkpoint-reached.
    bc_dagger_episodes: list[dict] = []
    if args.config_name:
        for root, rollout_type, share in _bc_dagger_dirs_from_config(args.config_name):
            eps = load_bc_dagger_episodes(Path(root), rollout_type, share)
            logger.info(
                "Loaded %d %s episodes from %s (share=%.3f)",
                len(eps), rollout_type.upper(), root, share,
            )
            bc_dagger_episodes.extend(eps)

    debias_inputs = all_episodes + bc_dagger_episodes
    # Align advantage lists: BC/DAgger contribute nothing to the per-frame
    # exp(clip(adv)) term, so pass None for those slots.
    debias_advantages = list(all_advantages) + [None] * len(bc_dagger_episodes)
    garment_debias = compute_garment_debias(
        all_train_episodes=debias_inputs,
        all_advantages=debias_advantages,
        garment_stats=garment_stats,
        bc_fr_scale=BC_FR_SCALE,
    )
    save_garment_debias(all_episodes, garment_debias)

    # Update parquet files (advantages + per-garment label smoothing baselines)
    update_parquet_files(
        all_episodes, all_advantages,
        garment_stats.avg_success, garment_stats.avg_checkpoint,
    )

    # Update info.json features
    _update_info_json_features(all_episodes)

    logger.info("Done. Parquet files updated with %s advantages + garment avg rates.", args.mode)


if __name__ == "__main__":
    main()
