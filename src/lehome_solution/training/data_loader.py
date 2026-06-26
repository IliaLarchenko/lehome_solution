"""Data loading for LeHome dataset.

Based on OpenPI training infrastructure, adapted for LeHome Challenge data.
Uses lerobot 0.4.x API (lerobot.datasets.lerobot_dataset).

Note: We intentionally do NOT import from openpi.training.data_loader at module level
because it imports lerobot.common.datasets (old 0.1.x API) which doesn't exist in lerobot 0.4.x.
Instead we copy the few utility classes we need (Protocol definitions, TransformedDataset, TorchDataLoader).
"""

import dataclasses
import logging
import multiprocessing
import os
import queue
import threading
import time
import typing
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as _transforms

from lehome_solution.models.observation import Observation
from lehome_solution.transforms_normalize import NormalizeWithPerTimestamp
from lehome_solution.eval.eval_utils import garment_name_to_type
from lehome_solution.constants import (
    GARMENT_TYPE_TO_ID,
    SUCCESS_TAIL_K,
    SUCCESS_TAIL_BOOST,
    BC_FR_SCALE,
    DAGGER_GARMENT_MAX_BOOST,
)

# Filename written by recompute_advantages.py into each dataset dir.
GARMENT_DEBIAS_FILENAME = "garment_debias.json"

T_co = TypeVar("T_co", covariant=True)



# ── Protocol definitions (copied from openpi.training.data_loader) ──────────

class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""
    def __getitem__(self, index: SupportsIndex) -> T_co: ...
    def __len__(self) -> int: ...


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""
    def data_config(self): ...
    def __iter__(self) -> Iterator[T_co]: ...


# ── Utility classes (copied from openpi.training.data_loader) ───────────────

class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transforms_list = list(transforms)
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raw = self._dataset[index]
        # Debug: run transforms one by one to find which introduces None
        result = raw
        for t in self._transforms_list:
            prev_keys_with_none = {k for k, v in result.items() if v is None} if isinstance(result, dict) else set()
            result = t(result)
            if isinstance(result, dict):
                new_nones = {k for k, v in result.items() if v is None} - prev_keys_with_none
                if new_nones:
                    logging.error(
                        "Transform %s introduced None for keys: %s (idx=%s)",
                        type(t).__name__, new_nones, index,
                    )
        return result

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)
        return {**observation.to_dict(), "actions": action}

    def __len__(self) -> int:
        return self._num_samples


@dataclasses.dataclass(frozen=True)
class InitialStuckConfig:
    """Suppress sampling of frames stuck near the initial position.

    Scans each episode's state/action parquet and identifies contiguous
    runs of frames that are (a) near the initial home position and (b)
    exhibit near-zero movement (|action − state| small). Leading runs get
    weight 0 (never sampled); mid-episode runs get ``mid_weight``;
    trailing runs get ``trailing_weight``.

    The initial-position envelope is derived per-source from the first
    frame of every episode in that source. A frame is "near initial" if
    every joint is within ``position_std_factor * per_joint_std`` of the
    per-source mean initial position (floored at ``min_position_range``
    degrees per joint).
    """
    movement_threshold: float = 3.0
    position_std_factor: float = 3.0
    min_position_range: float = 5.0
    min_leading_frames: int = 5
    mid_weight: float = 0.2
    trailing_weight: float = 0.3
    apply_to_buckets: tuple[str, ...] = ("primary_real", "home_real")


@dataclasses.dataclass(frozen=True)
class SimGarmentSkipConfig:
    """Frame-level skip rule for sim sources with a specific garment type.

    Motivation: sim rollouts of certain garments (e.g. Pant_Long) execute
    the early part of the task very differently from real teleop, which
    confuses the model. This rule lets us zero out the bulk of those
    early frames while keeping the first few (which carry the
    garment-bootstrap signal the inference-time detection logic depends
    on).

    For every episode of ``garment_type_name`` in a source matching
    ``apply_to_buckets``: frames at positions ``[first_n_keep, ep_len *
    skip_until_frac)`` get excluded from the source's valid-frame pool
    (they're never sampled). Frame 0 ... first_n_keep-1 and frame
    ep_len*skip_until_frac ... ep_len-1 stay sampled normally.
    """
    garment_type_name: str = "Pant_Long"
    first_n_keep: int = 10
    skip_until_frac: float = 0.5
    apply_to_buckets: tuple[str, ...] = ("home_sim",)


@dataclasses.dataclass(frozen=True)
class DaggerSamplingConfig:
    """Per-frame sampling probability rules for DAgger sources.

    Frames are weighted (then normalised to a sampling distribution) by:

      * Human-leader frames (``task_is_policy=0``):
            weight = ``human_weight`` (1.0 by default).
      * Auto-policy frames (``task_is_policy=1``) more than
        ``pre_intervention_window_seconds`` from the next human frame:
            weight = ``auto_weight`` (0.5 by default).
      * Auto-policy frames within the pre-intervention window:
            weight ramps LINEARLY from ``auto_weight`` (at window edge)
            down to ``pre_intervention_min_weight`` (at the intervention).

    All four numbers configurable; the design intent is to teach the policy
    on the auto frames it produced, EXCEPT the ones immediately leading up
    to a human takeover — those represent failure modes the policy itself
    should not be encouraged to repeat.
    """
    human_weight: float = 1.0
    auto_weight: float = 0.5
    pre_intervention_min_weight: float = 0.0
    pre_intervention_window_seconds: float = 5.0


@dataclasses.dataclass
class DataSourceConfig:
    """Runtime configuration for one data source in a MultiDataset."""
    repo_id: str
    root: str
    sampling_share: float = 1.0
    is_bc: bool = False
    is_dagger: bool = False  # DAgger: fixed advantage, uniform sampling
    # Real-robot source: real schema (degree units — see real_data_transforms),
    # native 20-Hz cadence, real camera column names. Data flows through with
    # NO unit conversion; only the per-source ``speed_factor`` time-axis
    # resample applies. Skips the BC/DAgger field-forcing block (no advantage /
    # success injection) and selects the degree-scaled bad-action threshold.
    is_real: bool = False
    # Bucket label for batch-mix control. Sources sharing the same bucket are
    # weighted together against ``bucket_shares`` (set on the dataset wrapper
    # at construction). Empty string disables bucket-rebalance for this source.
    bucket: str = ""
    # Optional garment-name override: when the parquet ``task`` column doesn't
    # carry a canonical garment name (e.g. all rows say "Fold the Garment"),
    # set this to the canonical name (``Pant_Long`` / ``Pant_Short`` /
    # ``Top_Long`` / ``Top_Short``). MultiDataset uses it in place of the
    # per-row task string when deriving ``garment_type_id``.
    garment_type_name: str = ""
    # Time-axis rescale for real sources. >1.0 means source is slower than
    # primary — grabs more native frames and compresses into action_horizon
    # (larger per-step deltas). <1.0 = faster source (fewer frames, smaller
    # deltas). Cubic-resamples onto the 30-action grid.
    speed_factor: float = 1.0
    # DAgger-only speed factors: human-leader frames train at one speed,
    # auto-policy frames at another. The chunk's resample factor is picked
    # from these two based on the t=0 frame's ``task_is_policy`` value
    # (for the whole chunk; per-frame within-chunk speed is intentionally
    # uniform for simplicity).
    dagger_human_speed_factor: float = 2.0
    dagger_auto_speed_factor: float = 2.0
    bc_advantage_raw: float | None = None  # injected for BC/dagger samples
    use_prioritized_sampling: bool = True   # sample proportional to exp(clip(advantage, clip_min, clip_max))
    success_rates_file: str | None = None   # FR-proportional sampling for BC/dagger data
    advantage_weighting_clipping: tuple[float, float] = (-2.0, 2.0)  # (min, max) for advantage clipping
    # When False, samples from this source contribute NaN labels to the
    # success, checkpoint, and ttc heads (which all use NaN masking in their
    # losses), so they train only on action + FAST + completion + garment_type.
    # Defaults True to preserve existing behaviour.  Typically set False for
    # BC / DAgger / golden / real sources when forcing `success=1` on them
    # would create a sharpness/quality shortcut (or — for real — there is no
    # failure signal in the data at all).
    use_for_success_labels: bool = True


class _MultiEpisodesAccessor:
    """Routes episodes[global_ep_idx] to the correct sub-dataset's meta.episodes."""

    def __init__(self, datasets: list, episode_offsets: list[int]):
        self._datasets = datasets
        self._episode_offsets = episode_offsets
        self._total = sum(len(ds.meta.episodes) for ds in datasets)

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, global_idx: int) -> dict:
        if global_idx < 0 or global_idx >= self._total:
            raise IndexError(f"Episode index {global_idx} out of range [0, {self._total})")
        # Find which sub-dataset owns this episode
        for i in range(len(self._datasets) - 1, -1, -1):
            if global_idx >= self._episode_offsets[i]:
                local_idx = global_idx - self._episode_offsets[i]
                return self._datasets[i].meta.episodes[local_idx]
        raise IndexError(f"Episode index {global_idx} could not be routed")


class _MultiDatasetMeta:
    """Composite meta object routing to per-source sub-dataset metas."""

    def __init__(self, datasets: list, episode_offsets: list[int]):
        self.episodes = _MultiEpisodesAccessor(datasets, episode_offsets)
        # Expose tasks from the first dataset (they should be identical across sources)
        if hasattr(datasets[0].meta, 'tasks'):
            self.tasks = datasets[0].meta.tasks


_ACTION_ABS_MAX = 10.0  # episodes with |action| > this are considered corrupted (radians)
_REAL_ACTION_ABS_MAX_DEG = 200.0  # degree-mode sources; nominal range ±180 (wrist_roll)

# Native sampling rate of all real (degree-mode) sources.
_RAW_REAL_FPS = 20


def _resample_native_chunk_to_horizon(
    native_actions: np.ndarray,
    speed_factor: float,
    action_horizon: int,
    fps: int = _RAW_REAL_FPS,
    action_is_pad: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic-resample a native-rate action chunk onto the model's
    ``action_horizon``-action grid using a ``speed_factor`` time rescale.
    Pure time-axis resample — units flow through unchanged.

    Native chunk has N actions at times [0, 1/fps, …, (N-1)/fps] seconds.
    The model expects ``action_horizon`` actions.  With speed_factor K,
    each model step should cover K× more native wall-clock time than the
    baseline (1/fps seconds).  So the 30-action model chunk spans
    ``action_horizon * K / fps`` native-seconds total.

    K > 1 (speed-up): target window is LONGER than baseline — grab more
    native frames and compress into action_horizon points.  Per-step
    Δaction magnitudes grow ~×K.

    K < 1 (slow-down): target window is SHORTER — use fewer native
    frames.  Per-step deltas shrink by ~×K.

    K = 1 (no rescale): identity if N == action_horizon, otherwise still
    interpolated to action_horizon to match the model's expected length.

    Caller must fetch at least ``ceil(action_horizon * K)`` native frames
    (otherwise the spline would extrapolate at the tail).
    """
    from scipy.interpolate import CubicSpline

    if native_actions.ndim != 2:
        raise ValueError(f"native_actions must be (N, D), got {native_actions.shape}")
    n_src = native_actions.shape[0]
    # Native frame times in seconds.
    src_t = np.arange(n_src, dtype=np.float64) / fps
    # Target frame times: each model step covers K/fps native-seconds,
    # so the full chunk spans action_horizon * K / fps native-seconds.
    target_span = action_horizon * float(speed_factor) / fps
    target_t = np.linspace(0.0, target_span, action_horizon, endpoint=False, dtype=np.float64)
    # Guard: never extrapolate. If caller didn't fetch enough native frames,
    # clamp target_t to the last src_t value. Spline still well-defined.
    target_t = np.minimum(target_t, src_t[-1])

    spline = CubicSpline(src_t, native_actions.astype(np.float64, copy=False), axis=0, bc_type="natural")
    out_y = spline(target_t).astype(native_actions.dtype, copy=False)

    if action_is_pad is None:
        out_pad = np.zeros(action_horizon, dtype=bool)
    else:
        pad = np.asarray(action_is_pad, dtype=bool).reshape(-1)
        if pad.shape[0] != n_src:
            raise ValueError(f"action_is_pad must be length {n_src}, got {pad.shape}")
        if pad.any():
            first_pad_src = int(np.argmax(pad))
            first_pad_t = first_pad_src / fps
            # Mark any target_t >= first_pad_t as padded.
            out_pad = target_t >= first_pad_t
        else:
            out_pad = np.zeros(action_horizon, dtype=bool)
    return out_y, out_pad


def _find_bad_action_episodes(dataset_root: str, *, is_real: bool = False) -> set[int]:
    """Scan parquets for episodes where actions contain NaN or extreme values.

    Sim parquets store actions in radians (~[-1.75, 1.77]); real-robot parquets
    store lerobot degree-mode values (arm joints in degrees, observed up to
    ~±130; gripper [0, 100]). Pick the threshold accordingly so we still catch
    corrupted episodes without rejecting the entire real dataset.
    """
    import pandas as pd
    from pathlib import Path

    root = Path(dataset_root)
    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        return set()

    threshold = _REAL_ACTION_ABS_MAX_DEG if is_real else _ACTION_ABS_MAX
    bad_episodes: set[int] = set()
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["episode_index", "action"])
        for ep_idx, group in df.groupby("episode_index"):
            actions = np.stack(group["action"].values)
            if np.isnan(actions).any() or np.abs(actions).max() > threshold:
                bad_episodes.add(int(ep_idx))
    return bad_episodes


class MultiDataset:
    """Wraps N LeRobotDataset instances with virtual index mapping.

    Global indices 0..total_len-1 are a concatenation of per-source effective lengths.
    When sampling_share < 1.0, each access randomly samples from the full valid pool
    (sampling with replacement), so different frames are seen across epochs.
    BC column injection and episode index remapping happen in __getitem__.
    """

    def __init__(
        self,
        sources: list[DataSourceConfig],
        action_horizon: int,
        action_sequence_keys: Sequence[str],
        seed: int | None = None,
        pct_real: float = 0.0,
        bucket_shares: dict[str, float] | None = None,
        dagger_sampling: DaggerSamplingConfig | None = None,
        sim_garment_skip: SimGarmentSkipConfig | None = None,
        initial_stuck: InitialStuckConfig | None = None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if not 0.0 <= pct_real <= 1.0:
            raise ValueError(f"pct_real must be in [0, 1], got {pct_real}")
        self._pct_real = pct_real
        # Bucket-shares is a more general replacement for ``pct_real`` — when
        # set, every source must carry a non-empty ``bucket`` label and shares
        # must sum to ~1.0. Sources in the same bucket split that bucket's
        # allocation proportional to their natural effective_len. Disabling
        # ``pct_real`` when bucket_shares is set is intentional: the two
        # systems compete for the same effective_lens after this init.
        # DAgger sampling config — controls per-frame weight in DAgger sources.
        self._dagger_sampling = dagger_sampling or DaggerSamplingConfig()
        # Sim-source per-garment frame skip rule (e.g. Pant_Long first-half).
        self._sim_garment_skip = sim_garment_skip
        self._initial_stuck = initial_stuck
        # Per-source action_horizon (constant; the model only knows one
        # horizon, but we may fetch more native frames upstream to support
        # speed_factor<1 sources).
        self._action_horizon = int(action_horizon)
        self._bucket_shares: dict[str, float] | None = None
        if bucket_shares:
            total = sum(bucket_shares.values())
            if abs(total - 1.0) > 1e-3:
                raise ValueError(
                    f"bucket_shares must sum to 1.0, got {total} ({bucket_shares})"
                )
            self._bucket_shares = dict(bucket_shares)
        # _sources is built inside the loop in lockstep with _datasets, so index
        # mapping stays correct when empty datasets are skipped. Do NOT assign
        # the raw `sources` here — that caused src/ds misalignment when any
        # source was skipped (BC/DAG override would fire on the wrong source).
        self._sources: list = []
        self._rng = np.random.default_rng(seed)
        self._datasets: list = []
        self._valid_indices: list[np.ndarray] = []  # all valid indices per source
        self._effective_lens: list[int] = []  # virtual length per source
        self._is_subsampled: list[bool] = []  # True when share < 1.0
        self._boundaries: list[int] = []  # cumulative effective lengths
        self._episode_offsets: list[int] = []
        self._sampling_probs: list[np.ndarray | None] = []  # prioritized sampling probs
        self._episode_dense_returns: list[dict[int, float]] = []  # per-dataset ep_idx -> dense_return
        self._episode_max_rewards: list[dict[int, float]] = []  # per-dataset ep_idx -> max cumulative reward
        # per-dataset ep_idx -> ndarray[ep_len, 7] of per-condition normalized distances.
        # Empty dict for sources without the column (BC / DAgger / older datasets).
        self._episode_check_distances: list[dict[int, np.ndarray]] = []
        # per-dataset ep_idx -> ndarray[ep_len] of past ttc_pred values (from the
        # checkpoint that collected this rollout). Used to bootstrap the TTC
        # target at t via ttc_pred(t + 30). Empty dict when the column is
        # absent (BC / DAgger / older rollouts without the head); consumers
        # fall back to the current analytical TTC formula in that case.
        self._episode_ttc_preds: list[dict[int, np.ndarray]] = []
        # Per-source per-garment debias table: {garment_name: (success_w, checkpoint_w)}.
        # Loaded from garment_debias.json (written by recompute_advantages.py).
        self._garment_debias: list[dict[str, tuple[float, float]]] = []

        sim_delta_timestamps = {
            key: [t / 30.0 for t in range(action_horizon)]
            for key in action_sequence_keys
        }

        cumulative_len = 0
        cumulative_episodes = 0

        for src in sources:
            # Skip empty datasets (no parquet data files)
            src_data_dir = Path(src.root) / "data"
            if not src_data_dir.exists() or not list(src_data_dir.rglob("*.parquet")):
                logging.warning(f"MultiDataset: skipping empty dataset {src.root} (no parquet files)")
                continue

            # Bypass _check_cached_episodes_sufficient to prevent LeRobotDataset
            # from trying to download from HF with fake repo_ids (e.g. "lehome_rl_70").
            # All our data is local — the video check can fail for incomplete downloads
            # but we only need parquet data for training.
            _orig_check = LeRobotDataset._check_cached_episodes_sufficient
            LeRobotDataset._check_cached_episodes_sufficient = lambda self: True
            try:
                if src.is_real:
                    # Per-source native delta_timestamps. Choose enough
                    # frames to cover the FASTEST speed_factor (= largest
                    # K, longest native window). For DAgger, that's
                    # max(human_K, auto_K) since either can fire per chunk.
                    if src.is_dagger:
                        max_k = max(src.dagger_human_speed_factor, src.dagger_auto_speed_factor)
                    else:
                        max_k = src.speed_factor
                    # +1 safety frame so the cubic spline always has a
                    # right anchor, and we never hit the strict last src_t.
                    n_native = int(np.ceil(action_horizon * float(max_k))) + 1
                    src_delta_timestamps = {
                        key: [t / float(_RAW_REAL_FPS) for t in range(n_native)]
                        for key in action_sequence_keys
                    }
                else:
                    src_delta_timestamps = sim_delta_timestamps
                ds = LeRobotDataset(
                    repo_id=src.repo_id,
                    root=src.root,
                    delta_timestamps=src_delta_timestamps,
                    video_backend="torchcodec",
                )
            finally:
                LeRobotDataset._check_cached_episodes_sufficient = _orig_check
            self._datasets.append(ds)
            self._sources.append(src)

            # Precompute per-episode dense_return + max_reward for checkpoint_target
            ep_dense_returns, ep_max_rewards = self._precompute_dense_returns(src.root)
            self._episode_dense_returns.append(ep_dense_returns)
            self._episode_max_rewards.append(ep_max_rewards)

            # Precompute per-episode check_distances arrays (for future-horizon lookup
            # used by WM/Q heads). Empty when the column is absent — consumers handle.
            self._episode_check_distances.append(self._precompute_check_distances(src.root))

            # Precompute per-episode ttc_pred arrays (for the TTC bootstrap
            # target: ttc(t) ≈ ttc_pred(t+30) − 30/600 for success / = ttc_pred
            # for failure). Empty when the column is absent.
            self._episode_ttc_preds.append(self._precompute_ttc_preds(src.root))

            # Scan parquets for episodes with NaN or extreme actions
            bad_episodes = _find_bad_action_episodes(src.root, is_real=src.is_real)

            # Also detect orphan-episode frames: rows in the data parquet
            # whose ``episode_index`` is >= len(meta.episodes). Caused by
            # interrupted recordings where one extra episode got appended
            # to the data file but the episodes meta wasn't updated. LeRobot
            # would crash on those frames (IndexError looking up the missing
            # meta entry); we exclude them upfront.
            n_meta_eps = len(ds.meta.episodes)
            orphan_frames: set[int] = set()
            try:
                import pyarrow.parquet as pq
                data_pqs = sorted((Path(src.root) / "data").glob("**/*.parquet"))
                global_row = 0
                for pf in data_pqs:
                    eps = pq.read_table(pf, columns=["episode_index"])["episode_index"].to_pylist()
                    for ep_val in eps:
                        ep_i = int(ep_val) if not isinstance(ep_val, list) else int(ep_val[0])
                        if ep_i >= n_meta_eps:
                            orphan_frames.add(global_row)
                        global_row += 1
            except Exception as e:
                logging.warning("Orphan-frame scan failed for %s: %r", src.root, e)
            if orphan_frames:
                logging.warning(
                    "MultiDataset [%s]: excluded %d orphan frames "
                    "(episode_index >= meta size %d)",
                    src.root, len(orphan_frames), n_meta_eps,
                )

            # Sim per-garment first-half skip (configured via
            # SimGarmentSkipConfig on the data config). Applies only to
            # sources whose bucket is in apply_to_buckets, only to episodes
            # of the configured garment type. Frames [first_n_keep,
            # ep_len * skip_until_frac) get excluded.
            skip_frames: set[int] = set()
            sgs = self._sim_garment_skip
            if sgs is not None and src.bucket in sgs.apply_to_buckets:
                n_eps_skipped = 0
                n_frames_skipped = 0
                for ep_idx in range(len(ds.meta.episodes)):
                    ep_meta = ds.meta.episodes[ep_idx]
                    # Episode's garment from the tasks list.
                    tasks = ep_meta.get("tasks", [])
                    if isinstance(tasks, (list, tuple)):
                        ep_task = str(tasks[0]) if tasks else ""
                    else:
                        ep_task = str(tasks)
                    if sgs.garment_type_name not in ep_task:
                        continue
                    n_eps_skipped += 1
                    from_idx = int(ep_meta.get("dataset_from_index", 0))
                    to_idx = int(ep_meta.get("dataset_to_index", from_idx))
                    ep_len = to_idx - from_idx
                    if ep_len <= sgs.first_n_keep:
                        continue
                    skip_start = from_idx + sgs.first_n_keep
                    skip_end = from_idx + int(round(ep_len * sgs.skip_until_frac))
                    if skip_end <= skip_start:
                        continue
                    for fi in range(skip_start, skip_end):
                        skip_frames.add(fi)
                    n_frames_skipped += (skip_end - skip_start)
                if skip_frames:
                    logging.warning(
                        "MultiDataset [%s]: SimGarmentSkip '%s' — excluded "
                        "%d frames across %d episodes (kept first %d, skipped to "
                        "%.0f%% of ep_len)",
                        src.root, sgs.garment_type_name, n_frames_skipped,
                        n_eps_skipped, sgs.first_n_keep, sgs.skip_until_frac * 100,
                    )

            # Build valid frame indices (exclude bad episodes + orphan frames + sim-skip)
            n = len(ds)
            if bad_episodes or orphan_frames or skip_frames:
                # Build set of frame indices belonging to bad episodes
                bad_frames = set(orphan_frames) | set(skip_frames)
                for ep_idx in bad_episodes:
                    ep = ds.meta.episodes[ep_idx]
                    for fi in range(ep["dataset_from_index"], ep["dataset_to_index"]):
                        bad_frames.add(fi)
                all_indices = np.array([i for i in range(n) if i not in bad_frames], dtype=np.int64)
                if bad_episodes:
                    logging.warning(
                        f"MultiDataset [{src.root}]: excluded {len(bad_episodes)} episodes "
                        f"with bad actions (NaN or |action|>{_ACTION_ABS_MAX}) "
                        f"({sum(1 for i in bad_frames if i not in orphan_frames)} frames): {sorted(bad_episodes)}"
                    )
            else:
                all_indices = np.arange(n, dtype=np.int64)

            # Compute effective length for this source
            effective_len = int(len(all_indices) * src.sampling_share)
            effective_len = max(1, min(effective_len, len(all_indices)))
            subsampled = effective_len < len(all_indices)

            # Build prioritized sampling probabilities
            sampling_probs = None
            if src.is_dagger and src.is_real:
                # Real DAgger path: per-frame weights from task_is_policy +
                # distance-to-next-intervention. Human-leader frames get full
                # weight; auto-policy frames default to 0.5 and ramp DOWN
                # toward the pre-intervention frames the policy itself
                # failed on.
                sampling_probs = self._load_dagger_priorities(
                    ds, src.root, all_indices, self._dagger_sampling,
                )
            elif not (src.is_bc or src.is_dagger or src.is_real) and src.use_prioritized_sampling:
                # RL sources: sample proportional to exp(clip(advantage, clip_min, clip_max))
                sampling_probs = self._load_priorities(src.root, all_indices, src.advantage_weighting_clipping)
            elif (src.is_bc or src.is_dagger) and src.success_rates_file:
                # BC and dagger sources: sample proportional to garment failure rate.
                # DAgger pools are dominated by easy garments (semi-success episodes
                # are correlated with high SR), so deweight by per-garment episode
                # count to keep the FR exponential as the only intended re-weight.
                sampling_probs = self._load_garment_fr_priorities(
                    ds, src.success_rates_file, all_indices,
                    divide_by_episode_count=src.is_dagger,
                )

            # Initial-stuck suppression: zero out leading stuck-at-home
            # frames, downweight mid/trailing stuck frames.
            if self._initial_stuck is not None and src.bucket in self._initial_stuck.apply_to_buckets:
                stuck_w = self._compute_initial_stuck_weights(
                    src.root, all_indices, self._initial_stuck,
                )
                if stuck_w is not None:
                    if sampling_probs is not None:
                        sampling_probs = sampling_probs * stuck_w
                        total = sampling_probs.sum()
                        if total > 0:
                            sampling_probs = sampling_probs / total
                        else:
                            logging.warning(
                                "InitialStuck zeroed all probs for %s — falling back to uniform",
                                src.root,
                            )
                            sampling_probs = None
                    else:
                        total = stuck_w.sum()
                        if total > 0:
                            sampling_probs = stuck_w / total

            # Validate sizes match — fail fast instead of silently using mismatched arrays
            if sampling_probs is not None and len(sampling_probs) != len(all_indices):
                raise ValueError(
                    f"SIZE MISMATCH at init: source={src.root}, ds_len={n}, "
                    f"all_indices.len={len(all_indices)}, "
                    f"all_indices.max={int(all_indices.max()) if len(all_indices) > 0 else -1}, "
                    f"sampling_probs.len={len(sampling_probs)}, "
                    f"hf_dataset_len={len(ds.hf_dataset)}"
                )

            self._valid_indices.append(all_indices)
            self._effective_lens.append(effective_len)
            self._is_subsampled.append(subsampled)
            self._sampling_probs.append(sampling_probs)

            # Load per-garment debias table (shared per training run).
            # Prefer src.root/garment_debias.json, fall back to sibling of
            # success_rates_file (so BC/DAgger can find the same table).
            debias_map = self._load_garment_debias(src)
            self._garment_debias.append(debias_map)

            cumulative_len += effective_len
            self._boundaries.append(cumulative_len)

            self._episode_offsets.append(cumulative_episodes)
            cumulative_episodes += len(ds.meta.episodes)

            n_episodes = len(ds.meta.episodes)
            tag = (
                "REAL" if src.is_real
                else "BC" if src.is_bc
                else "DAG" if src.is_dagger
                else "RL"
            )
            if sampling_probs is not None:
                mode = "FR-prioritized" if (src.is_bc or src.is_dagger) else "prioritized"
            else:
                mode = "random-sample" if subsampled else "full"
            logging.info(
                f"MultiDataset source [{tag}] {src.root}: "
                f"{n} frames -> {effective_len} effective (share={src.sampling_share:.2f}, {mode}), "
                f"{n_episodes} episodes (offset={self._episode_offsets[-1]})"
            )

        # ── bucket_shares rebalance (generalised replacement for pct_real) ──
        # Each source must carry a `bucket` label. We compute per-bucket
        # natural mass (sum of effective_lens within the bucket), then scale
        # each bucket so its mass / total_mass matches the requested share.
        # Sources within a bucket keep their relative natural ratios — so a
        # bucket containing multiple folders splits its allocation in
        # proportion to natural length.
        #
        # Implementation: anchor on the bucket whose ratio (natural_mass /
        # natural_total) most exceeds its target_share — that one stays
        # fixed; the others scale up. This avoids blowing past available
        # samples (sampling-with-replacement is fine, but we minimise the
        # over-sampling factor).
        if self._bucket_shares and self._sources:
            # Validate every source has a bucket label.
            missing = [i for i, s in enumerate(self._sources) if not s.bucket]
            if missing:
                names = ", ".join(self._sources[i].root for i in missing)
                raise ValueError(
                    f"bucket_shares set but {len(missing)} source(s) have no bucket: {names}"
                )
            unknown = set(s.bucket for s in self._sources) - set(self._bucket_shares.keys())
            if unknown:
                raise ValueError(
                    f"sources reference unknown buckets: {sorted(unknown)} "
                    f"(declared: {sorted(self._bucket_shares.keys())})"
                )
            # Per-bucket source-indices + natural mass.
            buckets: dict[str, list[int]] = {}
            for i, s in enumerate(self._sources):
                buckets.setdefault(s.bucket, []).append(i)
            natural_mass = {b: sum(self._effective_lens[i] for i in idxs) for b, idxs in buckets.items()}
            empty_buckets = [b for b, m in natural_mass.items() if m <= 0]
            if empty_buckets:
                raise ValueError(f"empty buckets after init: {empty_buckets}")

            # Pick the anchor bucket: the one with the smallest
            # target_share / natural_share ratio. Scaling it would require
            # the LEAST growth in the others.
            natural_total = sum(natural_mass.values())
            ratios = {
                b: self._bucket_shares[b] / (natural_mass[b] / natural_total)
                for b in buckets
            }
            anchor_bucket = min(ratios, key=ratios.get)
            anchor_share = self._bucket_shares[anchor_bucket]
            # After rebalance: anchor stays at natural_mass[anchor]. Total
            # length is anchor_mass / anchor_share. Each bucket's new mass
            # is share_b * total.
            target_total = natural_mass[anchor_bucket] / anchor_share
            for b, idxs in buckets.items():
                target_b = self._bucket_shares[b] * target_total
                scale = target_b / natural_mass[b]
                for i in idxs:
                    new_len = max(1, int(round(self._effective_lens[i] * scale)))
                    # Any source whose effective_len now differs from its
                    # valid-frame pool needs random re-sampling in __getitem__.
                    if new_len != len(self._valid_indices[i]):
                        self._is_subsampled[i] = True
                    self._effective_lens[i] = new_len
            # Rebuild boundaries.
            self._boundaries = []
            running = 0
            for ln in self._effective_lens:
                running += ln
                self._boundaries.append(running)
            cumulative_len = running

            # Diagnostic log so the achieved share is visible.
            new_mass = {b: sum(self._effective_lens[i] for i in idxs) for b, idxs in buckets.items()}
            new_total = sum(new_mass.values())
            for b in sorted(buckets):
                share = new_mass[b] / new_total
                logging.info(
                    "MultiDataset bucket[%s]: target=%.3f achieved=%.3f mass=%d (%d source(s); anchor=%s)",
                    b, self._bucket_shares[b], share, new_mass[b],
                    len(buckets[b]),
                    "yes" if b == anchor_bucket else "no",
                )

        # ── pct_real rebalance ────────────────────────────────────────────
        # Scale per-source effective_lens so real sources together occupy
        # exactly pct_real of the global virtual length. No-op when either
        # bucket is empty or pct_real == 0.  Within each bucket, the existing
        # per-source sampling_share ratios are preserved.
        # Skipped when bucket_shares is active (that controls mix instead).
        if self._bucket_shares is None and self._pct_real > 0.0 and self._sources:
            real_idx = [i for i, s in enumerate(self._sources) if s.is_real]
            sim_idx = [i for i, s in enumerate(self._sources) if not s.is_real]
            if real_idx and sim_idx:
                s_real = sum(self._effective_lens[i] for i in real_idx)
                s_sim = sum(self._effective_lens[i] for i in sim_idx)
                if s_real > 0 and s_sim > 0:
                    # Hold the LARGER bucket fixed and scale the smaller — keeps
                    # total dataset size close to the natural sum and avoids
                    # blowing past available samples.
                    natural_pct_real = s_real / (s_real + s_sim)
                    if natural_pct_real < self._pct_real:
                        # Need MORE real → upscale real, sim stays.
                        target_real = int(round(s_sim * self._pct_real / (1.0 - self._pct_real)))
                        scale = target_real / s_real if s_real > 0 else 1.0
                        for i in real_idx:
                            self._effective_lens[i] = max(1, int(round(self._effective_lens[i] * scale)))
                    else:
                        # Need LESS real → downscale sim, real stays.
                        target_sim = int(round(s_real * (1.0 - self._pct_real) / self._pct_real))
                        scale = target_sim / s_sim if s_sim > 0 else 1.0
                        for i in sim_idx:
                            self._effective_lens[i] = max(1, int(round(self._effective_lens[i] * scale)))
                    # Rebuild boundaries.
                    self._boundaries = []
                    running = 0
                    for ln in self._effective_lens:
                        running += ln
                        self._boundaries.append(running)
                    cumulative_len = running
                    # Any source whose effective_len now differs from its
                    # valid-frame pool needs to use random re-sampling in
                    # __getitem__ (over-sampling with replacement when
                    # effective_len > len(valid), under-sampling when <).
                    # Otherwise the direct-index branch `valid[idx - offset]`
                    # in __getitem__ blows past the end of the valid array.
                    for i in range(len(self._sources)):
                        if self._effective_lens[i] != len(self._valid_indices[i]):
                            self._is_subsampled[i] = True
                    new_pct = sum(self._effective_lens[i] for i in real_idx) / cumulative_len
                    logging.info(
                        f"MultiDataset pct_real rebalance: target={self._pct_real:.2f}, "
                        f"natural={natural_pct_real:.2f}, achieved={new_pct:.2f}, "
                        f"new effective_lens={self._effective_lens}"
                    )
            elif real_idx and not sim_idx:
                logging.info("MultiDataset pct_real set but no sim sources — pct_real ignored.")
            elif sim_idx and not real_idx:
                logging.info("MultiDataset pct_real set but no real sources — pct_real ignored.")

        self._total_len = cumulative_len
        self._meta = _MultiDatasetMeta(self._datasets, self._episode_offsets)
        logging.info(
            f"MultiDataset total: {self._total_len} frames, "
            f"{cumulative_episodes} episodes from {len(sources)} sources"
        )

        # TTC training policy per source:
        #   - last ``TTC_RL_TAIL`` RL datasets → full weight (1.0) + bootstrap
        #     target when ttc_pred(t + 30) is available; analytical fallback
        #     only at the terminal boundary where ``ttc_pred_future`` is NaN.
        #   - older RL datasets → analytical target only (no bootstrap), and
        #     the loss is down-weighted by ``TTC_OLD_WEIGHT`` so stale reward
        #     profiles don't drag the head around.
        #   - BC / DAgger → ttc_weight_scale = 1.0 but the target is NaN-masked
        #     downstream via ``success`` (BC forcing path), so the scale has
        #     no effect.
        TTC_RL_TAIL = 50
        TTC_OLD_WEIGHT = 0.01
        rl_indices = [
            i for i, s in enumerate(self._sources)
            if not (s.is_bc or s.is_dagger or s.is_real)
        ]
        full_weight_indices = set(rl_indices[-TTC_RL_TAIL:]) if rl_indices else set()
        # Map source_idx → per-sample weight scale for TTC loss.
        self._ttc_weight_scale_by_source: dict[int, float] = {}
        for i in range(len(self._sources)):
            if i in full_weight_indices:
                self._ttc_weight_scale_by_source[i] = 1.0
            elif i in set(rl_indices):
                self._ttc_weight_scale_by_source[i] = float(TTC_OLD_WEIGHT)
            else:
                # BC / DAgger: scale doesn't matter (target NaN), keep 1.0.
                self._ttc_weight_scale_by_source[i] = 1.0
        logging.info(
            "TTC: %d/%d RL sources at full weight (last %d); older RL at %.3fx; "
            "full-weight source indices=%s",
            len(full_weight_indices), len(rl_indices),
            TTC_RL_TAIL, TTC_OLD_WEIGHT, sorted(full_weight_indices),
        )

    @staticmethod
    def _load_priorities(
        root: str,
        valid_indices: np.ndarray,
        clip_range: tuple[float, float] = (-2.0, 2.0),
    ) -> np.ndarray | None:
        """Load advantage column from parquet and compute sampling probabilities.

        Returns normalized probabilities proportional to exp(clip(advantage, clip_min, clip_max)),
        or None if advantage column is not found or clip_range is (0, 0) (uniform).
        """
        from pathlib import Path
        import pyarrow.parquet as pq

        parquet_files = sorted(Path(root).glob("data/**/*.parquet"))
        if not parquet_files:
            return None

        # Read advantage column from all parquets
        all_advs = []
        for pf in parquet_files:
            try:
                table = pq.read_table(str(pf), columns=["advantage"])
            except Exception:
                return None
            if "advantage" not in table.column_names:
                return None
            col = table["advantage"].to_pylist()
            for v in col:
                if isinstance(v, (list, tuple)):
                    all_advs.append(float(v[0]))
                else:
                    all_advs.append(float(v))

        if len(all_advs) == 0:
            return None

        all_advs = np.array(all_advs, dtype=np.float32)
        # Bounds check: valid_indices must be within the range of all_advs
        if len(valid_indices) > 0 and valid_indices.max() >= len(all_advs):
            logging.warning(
                "valid_indices max (%d) >= all_advs length (%d), skipping prioritized sampling",
                valid_indices.max(), len(all_advs),
            )
            return None
        # Extract advantages for valid indices only
        valid_advs = all_advs[valid_indices]
        # Clip and exponentiate
        clip_min, clip_max = clip_range
        if clip_min == 0.0 and clip_max == 0.0:
            return None  # uniform sampling
        clipped = np.clip(valid_advs, clip_min, clip_max)
        priorities = np.exp(clipped)
        # Normalize to probability distribution
        priorities /= priorities.sum()
        uniform = 1.0 / len(priorities)
        logging.info(
            f"Prioritized sampling: {len(valid_advs)} frames, "
            f"advantage range [{valid_advs.min():.2f}, {valid_advs.max():.2f}], "
            f"priority max/min ratio {priorities.max() / max(priorities.min(), 1e-30):.1f}x, "
            f"max/uniform {priorities.max() / uniform:.1f}x"
        )
        return priorities

    @staticmethod
    def _load_dagger_priorities(
        dataset,
        root: str,
        valid_indices: np.ndarray,
        cfg: "DaggerSamplingConfig",
    ) -> np.ndarray | None:
        """Per-frame weights for a raw-real DAgger source.

        Reads ``task_is_policy`` from every parquet, groups frames by
        ``episode_index``, and for each frame computes a weight per the
        rules in :class:`DaggerSamplingConfig`.

        Returns a probability distribution over ``valid_indices`` (always
        summing to 1.0), or None if the column isn't present.
        """
        import pyarrow.parquet as pq

        parquet_files = sorted(Path(root).glob("data/**/*.parquet"))
        if not parquet_files:
            return None

        # Read task_is_policy + episode_index across all parquets, in file order
        # (matches LeRobotDataset's internal frame ordering).
        eps_all, pol_all = [], []
        for pf in parquet_files:
            try:
                tbl = pq.read_table(pf, columns=["task_is_policy", "episode_index"])
            except Exception:
                return None
            if "task_is_policy" not in tbl.column_names:
                logging.warning(
                    "DAgger source %s missing 'task_is_policy' column — falling back to uniform weights",
                    root,
                )
                return None
            eps_all.extend(tbl["episode_index"].to_pylist())
            pol_all.extend(tbl["task_is_policy"].to_pylist())
        ep_arr = np.asarray(eps_all, dtype=np.int64)
        pol_arr = np.asarray(pol_all, dtype=np.float32)
        n_total = ep_arr.shape[0]

        # Per-episode, walk backward computing distance (in frames) to the
        # next human-leader frame within the same episode. Frames with no
        # future human frame in their episode get +inf → auto_weight.
        next_human_dist = np.full(n_total, np.inf, dtype=np.float32)
        unique_eps = np.unique(ep_arr)
        for ep_idx in unique_eps:
            ep_mask = ep_arr == int(ep_idx)
            ep_pos = np.where(ep_mask)[0]  # indices in n_total
            ep_pol = pol_arr[ep_pos]
            # next_human_offset[i] = number of frames forward (within ep) to
            # next human frame; +inf if none.
            running = np.inf
            for j in range(len(ep_pos) - 1, -1, -1):
                if ep_pol[j] < 0.5:  # human-leader frame
                    running = 0.0
                next_human_dist[ep_pos[j]] = running
                running = running + 1.0  # one frame further as we step back

        # Convert distances to seconds.
        fps = 20.0  # all raw-real datasets are 20 fps
        secs = next_human_dist / fps

        # Apply the weight rule.
        weights = np.full(n_total, cfg.auto_weight, dtype=np.float32)
        # Human-leader frames always at human_weight.
        is_human = pol_arr < 0.5
        weights[is_human] = cfg.human_weight
        # Auto frames within window: linear ramp.
        is_auto = ~is_human
        window = float(cfg.pre_intervention_window_seconds)
        if window > 0:
            in_window = is_auto & (secs < window)
            # t = secs / window ∈ [0, 1). At t=1 → auto_weight; at t=0 → min.
            t = np.clip(secs[in_window] / window, 0.0, 1.0)
            weights[in_window] = (
                cfg.pre_intervention_min_weight
                + t * (cfg.auto_weight - cfg.pre_intervention_min_weight)
            )

        # Slice to valid_indices, normalise to a distribution.
        if len(valid_indices) > 0 and valid_indices.max() >= n_total:
            logging.warning(
                "DAgger valid_indices max (%d) >= n_total (%d), skipping",
                valid_indices.max(), n_total,
            )
            return None
        w_valid = weights[valid_indices]
        total = w_valid.sum()
        if total <= 0:
            logging.warning("DAgger: all weights zero for %s — falling back to uniform", root)
            return None
        probs = w_valid / total
        n_human = int(is_human.sum())
        n_auto = int(is_auto.sum())
        n_ramp = int((is_auto & (secs < window)).sum()) if window > 0 else 0
        logging.info(
            "DAgger priorities [%s]: %d frames (human=%d, auto=%d, pre-intervention=%d), "
            "weight range [%.3f, %.3f], avg %.3f",
            Path(root).name, len(w_valid), n_human, n_auto, n_ramp,
            w_valid.min(), w_valid.max(), w_valid.mean(),
        )
        return probs

    @staticmethod
    def _compute_initial_stuck_weights(
        root: str,
        valid_indices: np.ndarray,
        cfg: "InitialStuckConfig",
    ) -> np.ndarray | None:
        """Per-frame multiplier weights that suppress stuck-at-initial frames.

        Returns an array of length ``len(valid_indices)`` with values in
        [0, 1] that should be MULTIPLIED onto the existing sampling
        weights (or used directly when no other weights exist).
        """
        import pyarrow.parquet as pq

        parquet_files = sorted(Path(root).glob("data/**/*.parquet"))
        if not parquet_files:
            return None

        state_all, action_all, ep_all = [], [], []
        for pf in parquet_files:
            try:
                tbl = pq.read_table(pf, columns=["observation.state", "action", "episode_index"])
            except Exception:
                logging.warning("InitialStuck: cannot read %s — skipping", root)
                return None
            state_all.extend(tbl["observation.state"].to_pylist())
            action_all.extend(tbl["action"].to_pylist())
            ep_all.extend(tbl["episode_index"].to_pylist())

        states = np.array(state_all, dtype=np.float32)
        actions = np.array(action_all, dtype=np.float32)
        ep_arr = np.asarray(ep_all, dtype=np.int64)
        n_total = len(states)

        unique_eps = np.unique(ep_arr)
        initial_states = []
        for ep_idx in unique_eps:
            ep_mask = ep_arr == int(ep_idx)
            first = np.where(ep_mask)[0][0]
            initial_states.append(states[first])
        initial_states = np.stack(initial_states)
        mean_init = np.mean(initial_states, axis=0)
        std_init = np.std(initial_states, axis=0)
        pos_range = np.maximum(cfg.position_std_factor * std_init, cfg.min_position_range)

        per_frame_movement = np.mean(np.abs(actions - states), axis=1)
        near_initial = np.all(np.abs(states - mean_init) < pos_range, axis=1)
        low_movement = per_frame_movement < cfg.movement_threshold
        stuck = near_initial & low_movement

        weights = np.ones(n_total, dtype=np.float32)
        n_leading_zeroed = 0
        n_mid_down = 0
        n_trailing_down = 0

        for ep_idx in unique_eps:
            ep_mask = ep_arr == int(ep_idx)
            ep_pos = np.where(ep_mask)[0]
            ep_stuck = stuck[ep_pos]
            ep_len = len(ep_pos)

            leading = 0
            for i in range(ep_len):
                if ep_stuck[i]:
                    leading += 1
                else:
                    break

            trailing = 0
            for i in range(ep_len - 1, -1, -1):
                if ep_stuck[i]:
                    trailing += 1
                else:
                    break

            if leading >= cfg.min_leading_frames:
                for i in range(leading):
                    weights[ep_pos[i]] = 0.0
                n_leading_zeroed += leading

            start = leading if leading >= cfg.min_leading_frames else 0
            end = ep_len - trailing if trailing > 0 else ep_len
            for i in range(start, end):
                if ep_stuck[i]:
                    weights[ep_pos[i]] = cfg.mid_weight
                    n_mid_down += 1

            if trailing > 0:
                for i in range(ep_len - trailing, ep_len):
                    if ep_stuck[i]:
                        weights[ep_pos[i]] = cfg.trailing_weight
                        n_trailing_down += 1

        w_valid = weights[valid_indices]
        logging.info(
            "InitialStuck [%s]: %d total frames, %d leading zeroed (%.1f%%), "
            "%d mid downweighted (%.1f%%), %d trailing downweighted (%.1f%%)",
            Path(root).name, n_total,
            n_leading_zeroed, n_leading_zeroed / n_total * 100,
            n_mid_down, n_mid_down / n_total * 100,
            n_trailing_down, n_trailing_down / n_total * 100,
        )
        return w_valid

    @staticmethod
    def _load_garment_fr_priorities(
        dataset,
        success_rates_file: str,
        valid_indices: np.ndarray,
        fr_scale: float = BC_FR_SCALE,
        divide_by_episode_count: bool = False,
        max_boost_over_avg: float = DAGGER_GARMENT_MAX_BOOST,
    ) -> np.ndarray | None:
        """Compute exponential FR-weighted sampling for BC/dagger data.

        Maps each frame to its episode's garment, then weights by
        exp(fr_scale * (1 - garment_SR)).  With fr_scale=3 (default
        BC_FR_SCALE), hard garments still dominate but less aggressively
        than earlier, higher scales.

        Garment names come from episode tasks field (RL/dagger datasets) or
        garment_info.json (BC datasets where tasks is a generic description).

        With ``divide_by_episode_count=True`` (use for DAgger), each frame's
        weight is additionally divided by ``max(n_eps_for_garment, guard)``
        where ``guard = avg_eps_per_garment / max_boost_over_avg``. Intent:
        equalize per-garment total mass (so easy garments — which dominate
        the DAgger pool because semi-success episodes correlate with SR —
        don't crowd out rare ones), while capping the per-episode boost so
        a single rare-garment episode can't absorb all the mass.

        Semantics of the cap:
          * Per-episode weight = ``exp(fr_scale·fr) · min(1/n_ep_g,
            max_boost_over_avg/avg_eps)``. Each rare-garment episode is
            sampled at most ``max_boost_over_avg``× more often than the
            average episode.
          * Per-garment total mass is fully equalized for garments with
            ``n_ep_g ≥ guard``; rarer garments fall short of full
            equalization by a factor ``n_ep_g/guard`` (the price of the
            per-episode cap).

        The guard scales with dataset size and is a no-op for uniform pools.

        Returns normalized probability array over valid_indices, or None on error.
        """
        from lehome_solution.eval.rollout_strategies import load_success_rates_from_file

        sr_data = load_success_rates_from_file(success_rates_file)
        if not sr_data:
            raise FileNotFoundError(
                f"FR-prioritized sampling requires success rates but "
                f"failed to load from {success_rates_file!r}. "
                f"Run recompute_advantages.py first or remove success_rates_file from config."
            )

        sr_by_garment = sr_data.get("by_garment", {})
        sr_by_type = sr_data.get("by_type", {})

        n_frames = len(dataset)
        n_episodes = len(dataset.meta.episodes)
        frame_fr = np.full(n_frames, 0.5, dtype=np.float64)

        # Build episode -> garment mapping
        # Try 1: episode tasks field (works for RL/dagger datasets)
        # Try 2: garment_info.json (BC datasets: 25 episodes per garment, ordered)
        ep_garments: list[str] = []

        # Check if tasks contain garment names (not generic descriptions)
        sample_ep = dataset.meta.episodes[0]
        sample_tasks = sample_ep.get("tasks", [])
        sample_task = sample_tasks[0] if isinstance(sample_tasks, (list, tuple)) and sample_tasks else ""
        tasks_have_garments = sample_task and "_Seen_" in sample_task

        if tasks_have_garments:
            for ep_idx in range(n_episodes):
                ep = dataset.meta.episodes[ep_idx]
                tasks = ep.get("tasks", [])
                garment = tasks[0] if isinstance(tasks, (list, tuple)) and tasks else ""
                ep_garments.append(garment)
        else:
            # Fallback: use garment_info.json (25 episodes per garment, ordered by garment name)
            dataset_root = Path(dataset.root) if hasattr(dataset, 'root') else None
            gi_path = dataset_root / "meta" / "garment_info.json" if dataset_root else None
            if gi_path and gi_path.exists():
                import json
                with open(gi_path) as f:
                    garment_info = json.load(f)
                for garment_name, seeds in garment_info.items():
                    n_seeds = len(seeds)
                    ep_garments.extend([garment_name] * n_seeds)
                if len(ep_garments) != n_episodes:
                    raise ValueError(
                        f"garment_info.json has {len(ep_garments)} episodes but dataset has {n_episodes}. "
                        f"Dataset and garment_info.json are out of sync."
                    )
                logging.info(f"FR sampling: using garment_info.json ({len(garment_info)} garments × seeds)")
            else:
                raise FileNotFoundError(
                    f"FR-prioritized sampling requires garment info but no garment names "
                    f"found in tasks and no garment_info.json in dataset root. "
                    f"Enrich the dataset with `scripts/enrich_garment_type.py --domain sim` or remove success_rates_file."
                )

        # Per-garment episode counts — used to deweight overrepresented
        # garments when ``divide_by_episode_count`` is set (DAgger). Guard
        # caps the per-episode boost: each rare-garment episode is sampled
        # at most ``max_boost_over_avg``× more often than the average
        # episode (not per-garment; per-garment mass is equalized only for
        # garments with n_ep_g ≥ guard, rarer ones fall short by n_ep_g/guard).
        ep_count_by_garment: dict[str, int] = {}
        for g in ep_garments:
            if g:
                ep_count_by_garment[g] = ep_count_by_garment.get(g, 0) + 1
        if divide_by_episode_count and ep_count_by_garment:
            avg_eps = sum(ep_count_by_garment.values()) / len(ep_count_by_garment)
            episode_count_guard = max(1.0, avg_eps / max(max_boost_over_avg, 1e-6))
        else:
            episode_count_guard = 1.0

        try:
            assigned = 0
            for ep_idx in range(n_episodes):
                garment = ep_garments[ep_idx] if ep_idx < len(ep_garments) else ""
                if not garment:
                    continue

                garment_type = garment_name_to_type(garment)
                if garment in sr_by_garment:
                    fr = 1.0 - sr_by_garment[garment]
                elif garment_type in sr_by_type:
                    fr = 1.0 - sr_by_type[garment_type]
                else:
                    fr = 0.5
                fr = max(fr, 0.01)

                weight = float(np.exp(fr_scale * fr))
                if divide_by_episode_count:
                    weight /= max(ep_count_by_garment.get(garment, 1), episode_count_guard)

                ep = dataset.meta.episodes[ep_idx]
                from_idx = ep.get("dataset_from_index", 0)
                to_idx = ep.get("dataset_to_index", 0)
                frame_fr[from_idx:to_idx] = weight
                assigned += 1
        except Exception as e:
            raise RuntimeError(f"Failed to build garment FR weights: {e}") from e

        if assigned == 0:
            raise ValueError(
                "No garment assignments made during FR-prioritized sampling. "
                "Check that dataset episodes have valid garment names."
            )

        # Extract weights for valid indices
        if len(valid_indices) > 0 and valid_indices.max() >= n_frames:
            logging.warning(
                "valid_indices max (%d) >= n_frames (%d), skipping FR sampling",
                valid_indices.max(), n_frames,
            )
            return None

        valid_weights = frame_fr[valid_indices]
        total = valid_weights.sum()
        if total == 0:
            logging.warning("All FR weights are zero, falling back to uniform sampling")
            return None
        valid_weights /= total

        n_garments = len(set(sr_by_garment.keys()))
        logging.info(
            f"FR-prioritized sampling: {len(valid_weights)} frames, "
            f"weight range [{valid_weights.min():.4f}, {valid_weights.max():.4f}], "
            f"max/min ratio {valid_weights.max() / max(valid_weights.min(), 1e-30):.1f}x "
            f"({n_garments} garments in SR)"
        )
        return valid_weights

    @staticmethod
    def _load_garment_debias(src: "DataSourceConfig") -> dict[str, tuple[float, float]]:
        """Load per-garment (success_w, checkpoint_w) debias map.

        Looks first at ``{src.root}/garment_debias.json``; if absent, falls back
        to the sibling of ``src.success_rates_file``. Missing file → empty dict
        (all debias weights default to 1.0).
        """
        import json

        candidates: list[Path] = [Path(src.root) / GARMENT_DEBIAS_FILENAME]
        if src.success_rates_file:
            candidates.append(Path(src.success_rates_file).parent / GARMENT_DEBIAS_FILENAME)

        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        raw = json.load(f)
                except Exception as e:
                    logging.warning("Failed to read %s: %s — using default debias=1", path, e)
                    return {}
                result: dict[str, tuple[float, float]] = {}
                for garment, rec in raw.items():
                    sw = float(rec.get("success_w", 1.0))
                    cw = float(rec.get("checkpoint_w", 1.0))
                    result[garment] = (sw, cw)
                logging.info("Loaded garment debias (%d garments) from %s", len(result), path)
                return result

        logging.info(
            "garment_debias.json not found for %s (searched: %s). "
            "Using default debias=1.0 (no per-garment correction).",
            src.root, ", ".join(str(p) for p in candidates),
        )
        return {}

    @staticmethod
    def _precompute_dense_returns(root: str) -> tuple[dict[int, float], dict[int, float]]:
        """Precompute per-episode dense_return and max_reward from parquets.

        dense_return: sum of all dense_rewards (binary 0/1 with withdrawal).
        max_reward: peak cumulative reward reached (for checkpoint detection).

        Returns (dense_returns, max_rewards) dicts keyed by episode index.
        """
        import pyarrow.parquet as pq

        parquet_files = sorted(Path(root).glob("data/**/*.parquet"))
        if not parquet_files:
            return {}, {}

        # Accumulate per-episode: {ep_idx: list of (global_row_idx, reward)}
        # Track row index to guarantee frame ordering for cumulative max.
        ep_rewards_raw: dict[int, list[tuple[int, float]]] = {}
        global_row = 0
        for pf in parquet_files:
            try:
                table = pq.read_table(str(pf), columns=["episode_index", "dense_reward"])
            except Exception:
                continue
            if "dense_reward" not in table.column_names:
                global_row += len(table)
                continue
            ep_indices = table["episode_index"].to_pylist()
            drs = table["dense_reward"].to_pylist()
            for ep_idx, dr in zip(ep_indices, drs):
                val = float(dr[0]) if isinstance(dr, list) else float(dr)
                ep_rewards_raw.setdefault(ep_idx, []).append((global_row, val))
                global_row += 1

        ep_returns: dict[int, float] = {}
        ep_max_rewards: dict[int, float] = {}
        for ep_idx, indexed_rewards in ep_rewards_raw.items():
            # Sort by row index to ensure frame order (critical for merged datasets
            # where parquet files may interleave episodes from different sources)
            indexed_rewards.sort(key=lambda x: x[0])
            rewards = [r for _, r in indexed_rewards]
            ep_returns[ep_idx] = sum(rewards)
            cum = 0.0
            max_cum = 0.0
            for r in rewards:
                cum += r
                max_cum = max(max_cum, cum)
            ep_max_rewards[ep_idx] = max_cum

        # Validate: check for unexpected episode_index gaps or duplicates
        if ep_rewards_raw:
            max_ep = max(ep_rewards_raw.keys())
            n_episodes = len(ep_rewards_raw)
            if max_ep >= n_episodes * 2:
                logging.warning(
                    "Dense return precomputation: sparse episode indices in %s "
                    "(max_ep=%d, n_episodes=%d). This may indicate corrupted "
                    "episode_index values after dataset merging.",
                    root, max_ep, n_episodes,
                )
        return ep_returns, ep_max_rewards

    @staticmethod
    def _precompute_check_distances(root: str) -> dict[int, np.ndarray]:
        """Precompute per-episode check_distances arrays from parquet.

        Returns ``{episode_index: ndarray[ep_len, 7]}`` where each row is the
        per-condition normalized distance for that frame (NaN for unused slots).

        Empty when the column is absent (BC / DAgger / older datasets); callers
        must fall back to all-NaN targets in that case.

        Memory footprint: ~28 bytes/frame → <3 MB for a 100k-frame dataset.
        """
        import pyarrow.parquet as pq

        parquet_files = sorted(Path(root).glob("data/**/*.parquet"))
        if not parquet_files:
            return {}

        ep_distances_raw: dict[int, list[tuple[int, np.ndarray]]] = {}
        global_row = 0
        any_column_found = False
        for pf in parquet_files:
            try:
                table = pq.read_table(str(pf), columns=["episode_index", "check_distances"])
            except Exception:
                # Column absent → skip this parquet but advance row counter using a
                # bare read of episode_index so ordering across files is consistent.
                try:
                    idx_only = pq.read_table(str(pf), columns=["episode_index"])
                    global_row += len(idx_only)
                except Exception:
                    pass
                continue
            if "check_distances" not in table.column_names:
                global_row += len(table)
                continue
            any_column_found = True
            ep_indices = table["episode_index"].to_pylist()
            cds = table["check_distances"].to_pylist()
            for ep_idx, cd in zip(ep_indices, cds):
                arr = np.asarray(cd, dtype=np.float32).reshape(-1)
                ep_distances_raw.setdefault(ep_idx, []).append((global_row, arr))
                global_row += 1

        if not any_column_found:
            return {}

        ep_distances: dict[int, np.ndarray] = {}
        for ep_idx, indexed in ep_distances_raw.items():
            indexed.sort(key=lambda x: x[0])
            out = np.full((len(indexed), 7), np.nan, dtype=np.float32)
            for i, (_, arr) in enumerate(indexed):
                n = min(arr.shape[0], 7)
                out[i, :n] = arr[:n]
            ep_distances[ep_idx] = out
        return ep_distances

    @staticmethod
    def _precompute_ttc_preds(root: str) -> dict[int, np.ndarray]:
        """Precompute per-episode ``ttc_pred`` arrays from parquets.

        Returns ``{episode_index: ndarray[ep_len]}`` of past TTC predictions
        (from the checkpoint that collected the rollout). Empty dict when
        the column is absent (BC / DAgger / older datasets) — callers
        should fall back to the analytical TTC target.

        Memory footprint: 4 bytes/frame → <400 kB for a 100k-frame dataset.
        """
        import pyarrow.parquet as pq

        parquet_files = sorted(Path(root).glob("data/**/*.parquet"))
        if not parquet_files:
            return {}

        ep_preds_raw: dict[int, list[tuple[int, float]]] = {}
        global_row = 0
        any_column_found = False
        for pf in parquet_files:
            try:
                table = pq.read_table(str(pf), columns=["episode_index", "ttc_pred"])
            except Exception:
                # Column absent → advance global_row so ordering stays correct.
                try:
                    idx_only = pq.read_table(str(pf), columns=["episode_index"])
                    global_row += len(idx_only)
                except Exception:
                    pass
                continue
            if "ttc_pred" not in table.column_names:
                global_row += len(table)
                continue
            any_column_found = True
            ep_indices = table["episode_index"].to_pylist()
            vals = table["ttc_pred"].to_pylist()
            for ep_idx, v in zip(ep_indices, vals):
                scalar = float(v[0]) if isinstance(v, (list, tuple)) else float(v)
                ep_preds_raw.setdefault(ep_idx, []).append((global_row, scalar))
                global_row += 1

        if not any_column_found:
            return {}

        ep_preds: dict[int, np.ndarray] = {}
        for ep_idx, indexed in ep_preds_raw.items():
            indexed.sort(key=lambda x: x[0])
            out = np.array([v for _, v in indexed], dtype=np.float32)
            ep_preds[ep_idx] = out
        return ep_preds

    @property
    def meta(self) -> _MultiDatasetMeta:
        return self._meta

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__() if hasattr(index, '__index__') else int(index)
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"Index {idx} out of range [0, {self._total_len})")

        # Find which source owns this global index
        source_idx = 0
        for i, boundary in enumerate(self._boundaries):
            if idx < boundary:
                source_idx = i
                break

        valid = self._valid_indices[source_idx]
        probs = self._sampling_probs[source_idx]
        # IS weight: 1/(N*p). Same definition for every source (RL / BC / DAgger);
        # ep-len normalisation is applied below so the final is_weight field is
        # 1/(N*p*ep_len) — the full unbiased per-episode weight used by all aux losses.
        if probs is not None:
            chosen = self._rng.choice(len(valid), p=probs)
            local_mapped_idx = valid[chosen]
            raw_is_weight = np.float32(1.0 / (len(valid) * probs[chosen]))
        elif self._is_subsampled[source_idx]:
            local_mapped_idx = valid[self._rng.integers(len(valid))]
            raw_is_weight = np.float32(1.0)
        else:
            offset = self._boundaries[source_idx - 1] if source_idx > 0 else 0
            local_mapped_idx = valid[idx - offset]
            raw_is_weight = np.float32(1.0)

        ds = self._datasets[source_idx]
        src = self._sources[source_idx]
        if int(local_mapped_idx) >= len(ds):
            logging.error(
                "INDEX OUT OF BOUNDS: local_mapped_idx=%d >= ds_len=%d, "
                "source=%d/%s, valid.max=%d, probs=%s, global_idx=%d",
                int(local_mapped_idx), len(ds), source_idx,
                Path(src.root).name, int(valid.max()),
                "yes" if probs is not None else "no", idx,
            )

        item = ds[int(local_mapped_idx)]

        # Per-source time-axis rescale for real sources (native degree units,
        # 20-Hz cadence — no unit conversion anywhere on this path; camera
        # renames are handled by the RepackTransform mapping in
        # `LeRobotLeHomeDataConfig`).
        # We fetched ``ceil(action_horizon * max_K) + 1`` native frames;
        # resample is unconditional — at K=1.0 the spline is identity and
        # truncates to action_horizon.
        # For DAgger, pick K based on whether the t=0 frame is human-leader
        # (task_is_policy=0) or auto-policy.
        if src.is_real:
            if src.is_dagger:
                t0_is_policy = float(item.get("task_is_policy", 1.0)) > 0.5
                k = (
                    src.dagger_auto_speed_factor
                    if t0_is_policy
                    else src.dagger_human_speed_factor
                )
            else:
                k = src.speed_factor
            native_actions = np.asarray(item["action"], dtype=np.float32)
            pad = item.get("action_is_pad")
            new_actions, new_pad = _resample_native_chunk_to_horizon(
                native_actions,
                speed_factor=k,
                action_horizon=self._action_horizon,
                fps=_RAW_REAL_FPS,
                action_is_pad=np.asarray(pad) if pad is not None else None,
            )
            item["action"] = new_actions
            item["action_is_pad"] = new_pad

        # Remap episode_index to global space
        if "episode_index" in item:
            raw_ep = item["episode_index"]
            if hasattr(raw_ep, 'item'):
                raw_ep = raw_ep.item()
            item["episode_index"] = np.int64(int(raw_ep) + self._episode_offsets[source_idx])

        # Backward compat: migrate old column names from existing parquets
        if "value" in item and "success_pred" not in item:
            item["success_pred"] = item.pop("value")
        if "checkpoint_value" in item and "checkpoint_pred" not in item:
            item["checkpoint_pred"] = item.pop("checkpoint_value")

        # BC/DAgger: fixed advantage.  What gets written to success/max_reward/
        # dense_return depends on `src.use_for_success_labels`:
        #   True  (default): episodes are treated as successful (success=1, max_r=1,
        #     dense_return=1) so they feed the success/checkpoint/ttc BCE+MSE losses
        #     with a positive label.  Used when the BC/DAgger visual distribution
        #     matches RL rollouts (e.g. similar resolution / compression).
        #   False: success/max_reward/dense_return set to NaN.  The success,
        #     checkpoint, and ttc losses all mask out NaN targets, so these
        #     sources contribute ONLY to action + FAST + completion + garment_type
        #     losses (the ones whose targets are source-independent).  Use this
        #     when BC/DAgger images visually differ from rollouts enough that
        #     forcing `success=1` on them creates a shortcut for the head.
        if (src.is_bc or src.is_dagger) and not src.is_real:
            if src.bc_advantage_raw is not None:
                item["advantage"] = np.float32(src.bc_advantage_raw)
            item["success_pred"] = np.float32(np.nan)   # model never predicted for BC/DAgger/Real
            if src.use_for_success_labels:
                item["success"] = np.float32(1.0)
                item["max_reward"] = np.float32(1.0)
                item["dense_return"] = np.float32(1.0)
            else:
                # NaN propagates through ComputeSuccessTarget / ComputeAuxTargets
                # to success_target / checkpoint_target / ttc_target; every
                # downstream loss masks via jnp.isfinite.  completion_target
                # is frame_idx/ep_len (source-independent) — still computed.
                item["success"] = np.float32(np.nan)
                item["max_reward"] = np.float32(np.nan)
                item["dense_return"] = np.float32(np.nan)

        def _missing(key):
            return key not in item or item[key] is None

        if _missing("success_pred"):
            item["success_pred"] = np.float32(np.nan)
        if _missing("success"):
            item["success"] = np.float32(np.nan)
        # Real sources skip both the BC/DAgger forcing block and the RL parquet
        # lookup, so these are absent from the dict entirely; default them to
        # NaN so the RepackTransform doesn't KeyError.
        if _missing("max_reward"):
            item["max_reward"] = np.float32(np.nan)
        if _missing("dense_return"):
            item["dense_return"] = np.float32(np.nan)

        # Inject default garment avg rates if missing (older datasets without columns)
        if _missing("garment_avg_success"):
            item["garment_avg_success"] = item.pop("type_avg_reward", np.float32(0.5))
        else:
            item.pop("type_avg_reward", None)
        if _missing("garment_avg_checkpoint"):
            item["garment_avg_checkpoint"] = np.float32(0.5)

        if _missing("checkpoint_pred"):
            item["checkpoint_pred"] = np.float32(0.0)
        if _missing("checkpoint_held"):
            item["checkpoint_held"] = np.float32(0.0)

        # Per-condition keypoint distances (value / threshold, NaN-padded to width 7).
        # Populated from the launcher's Patch 5 via eval_worker → pkl → dataset_writer.
        # BC / DAgger / older datasets don't have this column — fill with NaN so
        # downstream masking via isfinite() zeros the loss cleanly.
        _CD_WIDTH = 7
        raw_cd = item.get("check_distances")
        if raw_cd is None:
            item["check_distances"] = np.full(_CD_WIDTH, np.nan, dtype=np.float32)
        else:
            arr = np.asarray(raw_cd, dtype=np.float32).reshape(-1)
            if arr.shape[0] < _CD_WIDTH:
                padded = np.full(_CD_WIDTH, np.nan, dtype=np.float32)
                padded[: arr.shape[0]] = arr
                item["check_distances"] = padded
            elif arr.shape[0] > _CD_WIDTH:
                item["check_distances"] = arr[:_CD_WIDTH]
            else:
                item["check_distances"] = arr

        # Garment type id from task name. A per-source override
        # ``garment_type_name`` takes precedence — used when the dataset's
        # task column doesn't carry a canonical garment name (e.g. the
        # home-real folders all carry "Fold the Garment", with garment type
        # encoded in the folder name).
        if src.garment_type_name:
            garment_name = src.garment_type_name
        else:
            task_name = item.get("task", "")
            if isinstance(task_name, (bytes, np.bytes_)):
                task_name = task_name.decode("utf-8", errors="replace")
            garment_name = task_name or ""
        if garment_name:
            gtype = garment_name_to_type(garment_name)
            item["garment_type_id"] = np.int32(GARMENT_TYPE_TO_ID.get(gtype, -1))
        else:
            item["garment_type_id"] = np.int32(-1)

        # completion_frac / ep_len / per-ep dense stats — ep_len is required; any
        # non-corrupted episode must have a valid length and resolvable frame index.
        raw_ep_idx = item.get("episode_index")
        if raw_ep_idx is None:
            raise RuntimeError(
                f"data_loader: missing episode_index for sample from source "
                f"{src.root} (is_bc={src.is_bc}, is_dagger={src.is_dagger})"
            )
        local_ep = int(raw_ep_idx) - self._episode_offsets[source_idx]
        # Only pull dense_return/max_reward from parquets for RL; BC/DAgger/Real got 1.0/NaN above.
        if not (src.is_bc or src.is_dagger or src.is_real):
            ep_dr = self._episode_dense_returns[source_idx].get(local_ep)
            if ep_dr is not None:
                item["dense_return"] = np.float32(ep_dr)
            ep_mr = self._episode_max_rewards[source_idx].get(local_ep)
            if ep_mr is not None:
                item["max_reward"] = np.float32(ep_mr)
        if local_ep >= len(ds.meta.episodes):
            raise RuntimeError(
                f"data_loader: episode_index {raw_ep_idx} (local {local_ep}) out of range "
                f"for source {src.root} (episodes={len(ds.meta.episodes)})"
            )
        ep_meta = ds.meta.episodes[local_ep]
        ep_len_raw = ep_meta.get("length")
        if ep_len_raw is None or int(ep_len_raw) <= 0:
            raise RuntimeError(
                f"data_loader: invalid ep_len={ep_len_raw} for episode {raw_ep_idx} "
                f"in {src.root}"
            )
        ep_len = int(ep_len_raw)
        from_idx = ep_meta.get("dataset_from_index", 0)
        frame_idx = item.get("index")
        if frame_idx is None:
            raise RuntimeError(
                f"data_loader: missing frame index for episode {raw_ep_idx} in {src.root}"
            )
        frame_in_ep = int(frame_idx) - from_idx
        if not (0 <= frame_in_ep < ep_len):
            raise RuntimeError(
                f"data_loader: frame_in_ep={frame_in_ep} out of range [0, {ep_len}) "
                f"for episode {raw_ep_idx} in {src.root}"
            )
        item["completion_frac"] = np.float32(frame_in_ep / ep_len)
        item["ep_len"] = np.int32(ep_len)

        # Future targets at t + WM_FUTURE_HORIZON (chunk_len). Targets for
        # post-FAST / post-flow world-modeling / Q heads. ALL future lookups
        # stay within the CURRENT episode — when the horizon would overshoot,
        # we clamp to the last frame (no cross-episode leakage). The episode
        # boundary is enforced here and in the per-episode check_distances
        # cache; LeRobot's ``delta_timestamps`` pads the action horizon the
        # same way (last action repeated; ``action_is_pad`` flags it).
        #   - success_future: episode-terminal success (constant per episode).
        #   - completion_future: min(frame + 30, ep_len - 1) / ep_len — at the
        #     last frame this is (ep_len - 1) / ep_len ≈ 0.997, which is
        #     intentional: it matches the current-time ``completion_frac`` at
        #     that same frame so training targets are consistent.
        #   - check_distances_future: reads the per-episode cache at the
        #     clamped ``future_in_ep``; NaN[7] for sources without the column.
        WM_FUTURE_HORIZON = 30
        future_in_ep = min(frame_in_ep + WM_FUTURE_HORIZON, ep_len - 1)
        item["completion_future"] = np.float32(future_in_ep / ep_len)
        _succ_raw = item.get("success")
        item["success_future"] = (
            np.float32(float(_succ_raw)) if _succ_raw is not None else np.float32(np.nan)
        )
        # Future check_distances (t + 30 frames). Sourced from the per-source
        # cache for THIS episode only — no cross-episode access. NaN[7] when
        # the source lacks the column (BC / DAgger / older datasets).
        ep_cd_map = self._episode_check_distances[source_idx] if self._episode_check_distances else {}
        ep_cd_arr = ep_cd_map.get(local_ep)
        if ep_cd_arr is not None and 0 <= future_in_ep < ep_cd_arr.shape[0]:
            item["check_distances_future"] = np.asarray(ep_cd_arr[future_in_ep], dtype=np.float32).copy()
        else:
            item["check_distances_future"] = np.full(7, np.nan, dtype=np.float32)

        # Bootstrap anchor for the TTC head: past ttc_pred at t + 30 (from the
        # checkpoint that collected this rollout). Set NaN when:
        #   - the column is absent (BC / DAgger / older sources),
        #   - the exact t + 30 overshoots the episode (terminal-boundary
        #     region; analytical fallback keeps the last-frame anchor tight),
        #   - the source is older than the last-N RL tail (bootstrap anchors
        #     from stale checkpoints drift, so we force analytical there and
        #     down-weight the whole TTC loss via ``ttc_weight_scale`` below).
        ttc_weight_scale = float(self._ttc_weight_scale_by_source.get(source_idx, 1.0))
        strict_future_in_ep = frame_in_ep + WM_FUTURE_HORIZON
        ep_ttc_map = self._episode_ttc_preds[source_idx] if self._episode_ttc_preds else {}
        ep_ttc_arr = ep_ttc_map.get(local_ep)
        if (
            ttc_weight_scale >= 1.0
            and ep_ttc_arr is not None
            and 0 <= strict_future_in_ep < ep_ttc_arr.shape[0]
        ):
            item["ttc_pred_future"] = np.float32(ep_ttc_arr[strict_future_in_ep])
        else:
            item["ttc_pred_future"] = np.float32(np.nan)

        # Per-sample loss scale applied to the TTC aux loss in train.py. 1.0
        # for BC / DAgger (NaN-masked anyway) and the last-N RL datasets;
        # 0.01 for older RL so they contribute a small regularising signal
        # without dominating the head's gradient.
        item["ttc_weight_scale"] = np.float32(ttc_weight_scale)

        # Episode-level IS weight: 1/(N*p) / ep_len. Single field used by every aux loss.
        item["is_weight"] = raw_is_weight / np.float32(ep_len)

        # Per-garment debias lookup (shared table loaded once at init).
        # For RL: success=1 uses success_w, success=0 uses 1.0; same for checkpoint (via max_reward).
        # For BC/DAgger: success=1 (forced above) → always uses success_w; checkpoint_reached=True.
        debias = self._garment_debias[source_idx].get(garment_name, (1.0, 1.0))
        success_w, checkpoint_w = debias
        succ_val = item.get("success")
        is_success = succ_val is not None and np.isfinite(float(succ_val)) and float(succ_val) > 0.5
        max_r_val = item.get("max_reward")
        cp_reached = is_success or (
            max_r_val is not None
            and np.isfinite(float(max_r_val))
            and float(max_r_val) >= 0.5
        )
        item["success_debias_weight"] = np.float32(success_w if is_success else 1.0)
        item["checkpoint_debias_weight"] = np.float32(checkpoint_w if cp_reached else 1.0)

        # Tail boost on success-head BCE only (folded into success_debias_weight).
        # Successful episodes' last K frames carry label=1 against a training-time
        # majority of "almost-success looks like failure" frames — boost corrects
        # the marginal-frequency bias in the BCE gradient.
        if is_success and frame_in_ep >= ep_len - SUCCESS_TAIL_K:
            item["success_debias_weight"] = np.float32(
                float(item["success_debias_weight"]) * SUCCESS_TAIL_BOOST
            )

        # Detect None values leaving __getitem__ (before transforms)
        for _dk, _dv in item.items():
            if _dv is None:
                logging.error(
                    "None BEFORE transforms: key=%s, source=%d/%s, "
                    "is_bc=%s, is_dagger=%s, local_idx=%d, global_idx=%d",
                    _dk, source_idx, Path(src.root).parent.name,
                    src.is_bc, src.is_dagger, int(local_mapped_idx), idx,
                )

        return item


def _collate_fn(items):
    """Collate batch elements into batched numpy arrays."""
    # Replace None values with dtype-appropriate defaults to prevent object dtype crash
    # from stale HF cache.  Use NaN for floats, -1 for ints (sentinel).
    _INT_KEYS = {"episode_index", "garment_type_id", "index", "frame_index", "task_index"}
    for i, item in enumerate(items):
        for key in list(item.keys()):
            if item[key] is None:
                logging.error("None in collation: item=%d/%d, key=%s", i, len(items), key)
                if key in _INT_KEYS:
                    item[key] = np.int64(-1)
                else:
                    item[key] = np.float32(np.nan)
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate GPU memory."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class TorchDataLoader:
    """Torch data loader implementation (copied from openpi)."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        prefetch: int = 2,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) > dataset size ({len(dataset)}).")

        self._sharding = sharding
        if sharding is None:
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._prefetch = prefetch

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def _raw_iter(self):
        """Iterate over raw batches (no prefetch)."""
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch
                )

    def __iter__(self):
        if self._prefetch <= 0:
            yield from self._raw_iter()
            return

        buf: queue.Queue = queue.Queue(maxsize=self._prefetch)
        _DONE = object()

        def _producer():
            try:
                for batch in self._raw_iter():
                    buf.put(batch)
            except Exception as e:
                buf.put(e)
            finally:
                buf.put(_DONE)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()
        while True:
            item = buf.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item


# ── LeHome-specific code ────────────────────────────────────────────────────

class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens.

    Yields (Observation, actions) tuples. Importance weighting is handled by
    prioritized sampling in MultiDataset, not per-sample weights.
    """

    def __init__(self, data_config, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self):
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            obs = Observation.from_dict(batch)
            actions = batch["actions"]
            yield obs, actions


def create_lehome_dataset(data_config, action_horizon: int, seed: int | None = None) -> Dataset:
    """Create a LeHome dataset for training.

    Uses standard LeRobotDataset (0.4.x) for efficient loading of LeHome data.
    When data_config.multi_sources is set, creates a MultiDataset wrapping N
    sub-datasets with virtual index mapping and BC column injection.
    """
    # Multi-source path
    if getattr(data_config, 'multi_sources', None):
        if seed is None:
            seed = int(time.time() * 1000) % (2**32)
            logging.info(f"Using random seed for MultiDataset: {seed}")
        dataset = MultiDataset(
            sources=data_config.multi_sources,
            action_horizon=action_horizon,
            action_sequence_keys=data_config.action_sequence_keys,
            seed=seed,
            pct_real=float(getattr(data_config, "pct_real", 0.0) or 0.0),
            bucket_shares=getattr(data_config, "bucket_shares", None),
            dagger_sampling=getattr(data_config, "dagger_sampling", None),
            sim_garment_skip=getattr(data_config, "sim_garment_skip", None),
            initial_stuck=getattr(data_config, "initial_stuck", None),
        )
        return dataset

    # Single-source path (backward compatible)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for LeRobotDataset: {seed}")

    dataset = LeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.lehome_dataset_root,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        video_backend="torchcodec",
    )

    return dataset


def transform_dataset(dataset: Dataset, data_config, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform dataset with LeHome-specific per-timestamp normalization support.

    CRITICAL: This overrides the base transform_dataset to pass use_per_timestamp to Normalize.
    The base version doesn't support per-timestamp normalization which causes huge action losses!
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Use custom Normalize with per-timestamp support
        NormalizeWithPerTimestamp(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        ),
    ]

    transforms_list.extend(data_config.model_transforms.inputs)

    return TransformedDataset(dataset, transforms_list)


def extract_episode_lengths_from_dataset(dataset) -> dict[int, float]:
    """Extract episode lengths from LeHome dataset metadata.

    Works with lerobot 0.4.x which provides meta.episodes with 'length' column.
    """
    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
        episodes = dataset.meta.episodes
        episode_lengths = {}
        for i in range(len(episodes)):
            ep = episodes[i]
            episode_lengths[i] = float(ep['length'])
        logging.info(f"Extracted {len(episode_lengths)} episode lengths from dataset")
        return episode_lengths

    raise ValueError("Dataset must have meta.episodes attribute (lerobot 0.4.x)")


def create_lehome_data_loader(
    config,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    """Create a data loader for LeHome training."""
    data_config = config.data.create(config.assets_dirs, config.model)

    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")

    dataset = create_lehome_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )

    return DataLoaderImpl(data_config, data_loader)
