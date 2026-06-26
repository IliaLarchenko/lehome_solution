"""Shared utilities for loading LeHome action data from parquet files.

Provides episode-aware action chunk generation that correctly handles
episode boundaries (chunks never span across episodes).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import openpi.transforms as transforms

from lehome_solution.training import config as _config


def get_delta_transform_from_config(config_name: str):
    """Get the delta action transform from the config (registry name or .yaml path)."""
    from lehome_solution.training.yaml_train_config import resolve_train_config

    config = resolve_train_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    delta_transform = None
    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, transforms.DeltaActions):
            delta_transform = transform
            break

    if delta_transform is None:
        raise ValueError("No DeltaActions transform found in config")

    return delta_transform


def apply_delta_transform(state: np.ndarray, action: np.ndarray, mask) -> np.ndarray:
    """Apply delta action transform: delta = action - state for masked dimensions.

    Args:
        state: Current state, shape [D].
        action: Single absolute action, shape [D].
        mask: Boolean mask from DeltaActions transform.

    Returns:
        Delta action, shape [D].
    """
    if mask is None:
        return action

    delta = action.copy()
    mask = np.asarray(mask)
    dims = mask.shape[-1]
    delta[:dims] = np.where(mask, action[:dims] - state[:dims], action[:dims])
    return delta


def find_parquet_files(data_root: str | Path) -> list[Path]:
    """Find LeHome parquet data files, supporting multiple directory layouts."""
    data_root = Path(data_root)

    # Try LeRobot v3 format first (file-*.parquet)
    files = sorted(data_root.glob("data/chunk-*/file-*.parquet"))
    if files:
        return files

    # Fallback to older per-episode format
    files = sorted(data_root.glob("data/chunk-*/episode_*.parquet"))
    if files:
        return files

    # Fallback to task-based format
    files = sorted(data_root.glob("data/task-*/episode_*.parquet"))
    if files:
        return files

    raise FileNotFoundError(
        f"No parquet data files found in {data_root}. "
        f"Checked: data/chunk-*/file-*.parquet, data/chunk-*/episode_*.parquet, "
        f"data/task-*/episode_*.parquet"
    )


# ---------------------------------------------------------------------------
# Low-level: process a single parquet file (used by parallel workers)
# ---------------------------------------------------------------------------

def _process_parquet_file(args):
    """Process a single parquet file and return delta action chunks.

    Handles LeRobot v3 format where multiple episodes live in one file.
    Chunks never span episode boundaries.

    Args (passed as tuple for ProcessPoolExecutor):
        parquet_path, delta_mask, action_horizon, sample_fraction,
        speed_factor.  Older callers may omit trailing fields (default:
        speed_factor=1.0). Units flow through natively — no conversion.

    Returns:
        dict with keys:
            "chunks": np.ndarray [N, H, D] of delta action chunks (or None).
            "states": np.ndarray [M, D] of all states.
            "actions": np.ndarray [M, D] of all delta actions (per-step).
    """
    if len(args) == 4:
        parquet_path, delta_mask, action_horizon, sample_fraction = args
        speed_factor = 1.0
    else:
        parquet_path, delta_mask, action_horizon, sample_fraction, speed_factor = args

    try:
        df = pd.read_parquet(parquet_path)

        all_states = np.array(
            [np.array(row["observation.state"]) for _, row in df.iterrows()]
        )
        all_raw_actions = np.array(
            [np.array(row["action"]) for _, row in df.iterrows()]
        )

        if len(all_states) == 0:
            return None

        # Per-step delta actions (each action relative to its own state)
        all_delta_actions = np.array([
            apply_delta_transform(all_states[i], all_raw_actions[i], delta_mask)
            for i in range(len(all_states))
        ])

        # Detect episode boundaries
        if "episode_index" in df.columns:
            episode_indices = df["episode_index"].values
        else:
            episode_indices = np.zeros(len(df), dtype=int)

        unique_episodes = np.unique(episode_indices)

        # Native frames needed per chunk when speed_factor != 1.0.
        # Mirrors MultiDataset.__init__: ceil(action_horizon * K) + 1.
        n_native = int(np.ceil(action_horizon * float(speed_factor))) + 1
        need_resample = abs(speed_factor - 1.0) > 1e-6

        if need_resample:
            from lehome_solution.training.data_loader import (
                _resample_native_chunk_to_horizon,
            )

        chunks = []
        for ep_idx in unique_episodes:
            ep_mask = episode_indices == ep_idx
            ep_states = all_states[ep_mask]
            ep_raw_actions = all_raw_actions[ep_mask]

            if len(ep_states) < n_native:
                continue

            for i in range(len(ep_states) - n_native + 1):
                current_state = ep_states[i]
                native_actions = ep_raw_actions[i : i + n_native]

                if need_resample:
                    resampled, _ = _resample_native_chunk_to_horizon(
                        native_actions, speed_factor, action_horizon,
                    )
                else:
                    resampled = native_actions[:action_horizon]

                delta_chunk = np.zeros_like(resampled)
                for t in range(action_horizon):
                    delta_chunk[t] = apply_delta_transform(
                        current_state, resampled[t], delta_mask
                    )
                chunks.append(delta_chunk)

        if not chunks:
            return {
                "chunks": None,
                "states": all_states,
                "actions": all_delta_actions,
            }

        chunks = np.array(chunks)  # [N, H, D]

        if sample_fraction < 1.0:
            n_keep = max(1, int(len(chunks) * sample_fraction))
            rng = np.random.RandomState(hash(str(parquet_path)) % (2**31))
            idx = rng.choice(len(chunks), size=n_keep, replace=False)
            chunks = chunks[idx]

        return {
            "chunks": chunks,
            "states": all_states,
            "actions": all_delta_actions,
        }

    except Exception as e:
        print(f"Error processing {parquet_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def load_action_chunks(
    data_root: str | Path,
    delta_mask,
    action_horizon: int,
    *,
    sample_fraction: float = 1.0,
    max_files: int | None = None,
    num_workers: int | None = None,
    speed_factor: float = 1.0,
) -> dict:
    """Load delta action chunks from LeHome parquet files.

    Correctly handles episode boundaries — chunks never span episodes.
    When ``speed_factor != 1.0``, applies the same cubic-spline time-axis
    resample that training uses (``_resample_native_chunk_to_horizon``),
    so the resulting stats match the distribution the model sees.

    Args:
        data_root: Path to dataset root (contains data/ subdirectory).
        delta_mask: Boolean mask from DeltaActions transform.
        action_horizon: Number of timesteps per action chunk.
        sample_fraction: Fraction of chunks to keep per file (1.0 = all).
        max_files: Limit on number of parquet files to process.
        num_workers: Parallel workers (None = auto).
        speed_factor: Time-axis rescale (>1 = source slower than primary).

    Returns:
        dict with:
            "chunks": np.ndarray [N, H, D] — delta action chunks.
            "states": np.ndarray [M, D] — all states (flat, for global stats).
            "actions": np.ndarray [M, D] — all per-step delta actions (flat).
    """
    parquet_files = find_parquet_files(data_root)
    sf_tag = f"  [speed_factor={speed_factor}]" if abs(speed_factor - 1.0) > 1e-6 else ""
    print(f"Found {len(parquet_files)} parquet data files in {data_root}" + sf_tag)

    if max_files is not None:
        parquet_files = parquet_files[:max_files]
        print(f"Limited to {len(parquet_files)} files")

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(parquet_files))
    num_workers = max(1, num_workers)
    print(f"Processing with {num_workers} workers, sample_fraction={sample_fraction}")

    args_list = [
        (f, delta_mask, action_horizon, sample_fraction, speed_factor) for f in parquet_files
    ]

    all_chunks = []
    all_states = []
    all_actions = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_parquet_file, a): a[0] for a in args_list
        }
        for future in as_completed(futures):
            parquet_path = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing {parquet_path}: {e}")
                continue
            if result is None:
                continue
            if result["chunks"] is not None:
                all_chunks.append(result["chunks"])
            all_states.append(result["states"])
            all_actions.append(result["actions"])

    states = np.concatenate(all_states, axis=0) if all_states else np.empty((0,))
    actions = np.concatenate(all_actions, axis=0) if all_actions else np.empty((0,))
    chunks = np.concatenate(all_chunks, axis=0) if all_chunks else None

    n_chunks = len(chunks) if chunks is not None else 0
    print(f"Loaded {len(states)} timesteps, {n_chunks} action chunks")

    return {
        "chunks": chunks,
        "states": states,
        "actions": actions,
    }
