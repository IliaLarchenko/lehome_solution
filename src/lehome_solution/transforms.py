"""Data transforms for LeHome dataset.

Standard transforms imported from OpenPI.
TokenizeFASTActions, ComputeSuccessTarget, AdvantageToWeight.

Based on OpenPI transforms, adapted for LeHome Challenge.
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np

# Import all standard transforms from OpenPI
from openpi.transforms import (
    DataTransformFn,
    DataDict,
)

from lehome_solution.shared.normalize import NormStats

# Module-level cache for FAST tokenizers, keyed by tokenizer_path.
# Avoids object.__setattr__ on frozen dataclass instances.
_TOKENIZER_CACHE: dict[str, object] = {}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTActions(DataTransformFn):
    """Tokenize actions to FAST discrete tokens for PiModified auxiliary training.
    
    Converts continuous actions to discrete tokens using a trained FAST tokenizer.
    This is used for auxiliary training to improve action quality.
    
    Note: Uses GLOBAL normalization even if per-timestamp normalization is enabled
    elsewhere, to preserve temporal smoothness for better DCT compression.
    
    Args:
        tokenizer_path: Path to trained FAST tokenizer directory
        encoded_dim_ranges: List of (start, end) tuples for dimensions to encode
        max_fast_tokens: Maximum number of tokens (truncate/pad)
        norm_stats: Normalization stats for denorm/renorm if per-timestamp is used
        use_per_timestamp: Whether per-timestamp normalization is being used
    
    Assumes:
    - data["actions"] exists and is normalized
    - data["state"] exists (for delta computation)
    
    Creates:
    - data["fast_tokens"]: Discrete token indices [max_fast_tokens]
    - data["fast_token_mask"]: Boolean mask for valid tokens [max_fast_tokens]
    """
    
    # Path to FAST tokenizer
    tokenizer_path: str
    # Action dimensions to encode: [(start, end), ...]
    encoded_dim_ranges: list[tuple[int, int]]
    # Max tokens to predict
    max_fast_tokens: int = 180
    # Normalization stats (for denorm/renorm if per-timestamp is used)
    norm_stats: dict[str, NormStats] | None = None
    # Whether per-timestamp normalization is being used
    use_per_timestamp: bool = False
    
    def _get_tokenizer(self):
        """Lazy load tokenizer (called in each worker process)."""
        if self.tokenizer_path in _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[self.tokenizer_path]

        import importlib.util

        tokenizer_dir = Path(self.tokenizer_path)
        processor_file = tokenizer_dir / "processing_action_tokenizer.py"

        if not processor_file.exists():
            raise FileNotFoundError(f"Tokenizer processing file not found at {processor_file}")

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("processing_action_tokenizer", processor_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the UniversalActionProcessor class
        UniversalActionProcessor = module.UniversalActionProcessor

        # Load the tokenizer using from_pretrained
        tokenizer = UniversalActionProcessor.from_pretrained(str(tokenizer_dir))

        _TOKENIZER_CACHE[self.tokenizer_path] = tokenizer
        return tokenizer
    
    def __call__(self, data: DataDict) -> DataDict:
        # Only tokenize if actions are present (training time)
        if "actions" not in data:
            return data
        
        actions = data["actions"]  # [H, D] - may be per-timestamp normalized
        
        # Extract encoded dimensions
        encoded_chunks = []
        for start, end in self.encoded_dim_ranges:
            encoded_chunks.append(actions[:, start:end])
        encoded_actions = np.concatenate(encoded_chunks, axis=-1)  # [H, D_encoded]

        if np.isnan(encoded_actions).any():
            raise ValueError(
                "NaN in actions passed to TokenizeFASTActions — "
                "dataset contains failed rollouts that should have been filtered out"
            )

        # Convert to global QUANTILE normalization for FAST (always, regardless of per_timestamp)
        # This clips outliers and provides consistent [-1, 1] range for DCT compression
        if self.norm_stats is not None:
            action_stats = self.norm_stats['actions']

            # Extract stats for encoded dimensions only
            # Build dimension indices from ranges
            encoded_dims = []
            for start, end in self.encoded_dim_ranges:
                encoded_dims.extend(range(start, end))
            encoded_dims = np.array(encoded_dims)

            # Denormalize to get back to raw delta actions
            if self.use_per_timestamp:
                # Denormalize from per-timestamp MEAN/STD normalization
                per_ts_mean = action_stats.per_timestamp_mean[:, encoded_dims]  # [H, D_encoded]
                per_ts_std = action_stats.per_timestamp_std[:, encoded_dims]    # [H, D_encoded]
                actions_delta = encoded_actions * (per_ts_std + 1e-6) + per_ts_mean
            else:
                # Denormalize from global MEAN/STD normalization
                global_mean = action_stats.mean[encoded_dims]  # [D_encoded]
                global_std = action_stats.std[encoded_dims]    # [D_encoded]
                actions_delta = encoded_actions * (global_std + 1e-6) + global_mean

            # Renormalize using GLOBAL QUANTILE normalization (for FAST)
            # This clips to [q01, q99] and maps to [-1, 1]
            q01 = action_stats.q01[encoded_dims]  # [D_encoded]
            q99 = action_stats.q99[encoded_dims]  # [D_encoded]
            actions_delta = np.clip(actions_delta, q01, q99)
            q_range = q99 - q01
            # When quantile range is near-zero (constant dimension), output 0 instead of huge values
            valid_range = q_range > 1e-4
            encoded_actions = np.where(
                valid_range,
                2.0 * (actions_delta - q01) / np.maximum(q_range, 1e-4) - 1.0,
                0.0,
            )

        # Tokenize
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(encoded_actions[None])[0]
        
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        
        # Truncate or pad to max_fast_tokens
        if len(tokens) > self.max_fast_tokens:
            tokens = tokens[:self.max_fast_tokens]
            mask = np.ones(self.max_fast_tokens, dtype=bool)
        else:
            mask = np.concatenate([
                np.ones(len(tokens), dtype=bool),
                np.zeros(self.max_fast_tokens - len(tokens), dtype=bool)
            ])
            tokens = np.pad(tokens, (0, self.max_fast_tokens - len(tokens)), constant_values=0)
        
        data["fast_tokens"] = tokens.astype(np.int32)
        data["fast_token_mask"] = mask

        return data


@dataclasses.dataclass(frozen=True)
class ComputeSuccessTarget(DataTransformFn):
    """Use binary success as value_target for the value head, with optional label smoothing.

    Label smoothing (when alpha > 0):
        target = success * (1 - alpha) + garment_avg_success * alpha
    Smooths toward the per-garment average success rate.

    Must still run BEFORE AdvantageToWeight.
    """

    label_smoothing: float = 0.0  # alpha for label smoothing (0 = disabled)

    def __call__(self, data: DataDict) -> DataDict:
        success = data.get("success")
        advantage = data.get("advantage")
        if advantage is None:
            return data
        if success is None:
            data["success_target"] = np.float32(np.nan)
            return data
        target = np.asarray(success, dtype=np.float32)
        if self.label_smoothing > 0:
            garment_avg = data.get("garment_avg_success")
            baseline = float(garment_avg) if garment_avg is not None and np.isfinite(garment_avg) else 0.5
            target = target * (1 - self.label_smoothing) + baseline * self.label_smoothing
        data["success_target"] = target
        return data


@dataclasses.dataclass(frozen=True)
class ComputeAuxTargets(DataTransformFn):
    """Compute auxiliary prediction targets: checkpoint, completion, garment_type_id.

    All targets share NaN masking: NaN means "skip this sample for this loss".
    Must run BEFORE AdvantageToWeight (needs raw advantage to detect RL samples).

    Targets:
        checkpoint_target: 1.0 if dense_reward >= 0.5 (or success=1), else 0.0.
            With label smoothing: target * (1-alpha) + garment_avg_checkpoint * alpha.
            NaN for BC/masked samples.
        completion_target: frame_position / episode_length. Emitted for all data,
            but train.py filters the loss to successful episodes only
            (success_target > 0.5; BC/DAgger count via forced success=1).
            NaN if completion_frac is missing.
        garment_type_id: int (0-3) for all data. -1 if unknown.
    """

    label_smoothing: float = 0.0  # shared with ComputeSuccessTarget

    def __call__(self, data: DataDict) -> DataDict:
        from lehome_solution.constants import GARMENT_TYPE_TO_ID

        garment_type_id = data.get("garment_type_id")
        if garment_type_id is None:
            garment_type_id = np.int32(-1)
        else:
            garment_type_id = np.int32(garment_type_id)
        data["garment_type_id"] = garment_type_id

        # completion_target: emitted for all data; train.py applies the
        # success-only filter at loss time (BC/DAgger pass via forced success=1).
        completion_frac = data.get("completion_frac")
        if completion_frac is not None:
            data["completion_target"] = np.float32(completion_frac)
        else:
            data["completion_target"] = np.float32(np.nan)

        # ttc_target: uses the TTC-bootstrap target when the past ttc_pred at
        # t + 30 is available, else the analytical formula.
        #
        #   analytical (current formula):
        #       success:  1 − steps_left/600
        #       failure:  0
        #
        #   bootstrap (preferred, N-step TD with N=30):
        #       success:  ttc_pred(t + 30) − 30/600
        #       failure:  ttc_pred(t + 30)
        #
        # The bootstrap is equivalent to the analytical semantics under
        # TD(0) with implicit per-step reward −1/600 (success) / 0 (failure)
        # plus the terminal +1 (success) / 0 (failure). Using the past-
        # checkpoint prediction as the value baseline lets the target track
        # the policy's own value landscape instead of anchoring to a fixed
        # analytical formula.
        #
        # The data loader only emits ``ttc_pred_future`` when t + 30 < ep_len
        # (no clamp), so the 30/600 offset is always exact. At the terminal
        # boundary (last 30 frames) the field is NaN → analytical fallback
        # keeps the terminal anchor tight (ttc = 1 / 0 at t = T − 1).
        ep_len_raw = data.get("ep_len")
        success_raw = data.get("success")
        # ``ttc_pred_future`` is already gated by the data loader: it is NaN
        # for samples that must use the analytical formula (BC / DAgger,
        # terminal boundary, and older RL datasets where bootstrap anchors
        # are stale — those are down-weighted separately via
        # ``ttc_weight_scale`` in train.py rather than NaN-masked here).
        ttc_pred_future_raw = data.get("ttc_pred_future")
        have_core = (
            completion_frac is not None
            and np.isfinite(float(completion_frac))
            and ep_len_raw is not None
            and int(ep_len_raw) > 0
            and success_raw is not None
            and hasattr(success_raw, '__float__')
            and np.isfinite(float(success_raw))
        )
        if have_core:
            is_success = float(success_raw) > 0.5
            bootstrap_finite = (
                ttc_pred_future_raw is not None
                and np.isfinite(float(ttc_pred_future_raw))
            )
            if bootstrap_finite:
                tpf = float(ttc_pred_future_raw)
                if is_success:
                    data["ttc_target"] = np.float32(tpf - 30.0 / 600.0)
                else:
                    data["ttc_target"] = np.float32(tpf)
                data["ttc_used_bootstrap"] = np.bool_(True)
            else:
                if is_success:
                    ep_len_val = int(ep_len_raw)
                    frame_in_ep = round(float(completion_frac) * ep_len_val)
                    steps_left = ep_len_val - 1 - frame_in_ep
                    data["ttc_target"] = np.float32(1.0 - steps_left / 600.0)
                else:
                    data["ttc_target"] = np.float32(0.0)
                data["ttc_used_bootstrap"] = np.bool_(False)
        else:
            data["ttc_target"] = np.float32(np.nan)
            data["ttc_used_bootstrap"] = np.bool_(False)

        advantage = data.get("advantage")
        if advantage is None:
            data["checkpoint_target"] = np.float32(np.nan)
            return data

        success = data.get("success")
        if success is None or (hasattr(success, '__float__') and np.isnan(float(success))):
            data["checkpoint_target"] = np.float32(np.nan)
            return data

        success_val = float(success)

        # checkpoint_target: episode-level label — did this episode reach the first checkpoint?
        # Use max_reward (peak cumulative reward before withdrawal). If absent,
        # fall back to the binary success label.
        max_reward = data.get("max_reward")
        if max_reward is not None:
            checkpoint_reached = float(max_reward) >= 0.5
        else:
            checkpoint_reached = success_val > 0.5
        cp_target = 1.0 if checkpoint_reached else 0.0
        if self.label_smoothing > 0:
            garment_avg = data.get("garment_avg_checkpoint")
            baseline = float(garment_avg) if garment_avg is not None else 0.5
            cp_target = cp_target * (1 - self.label_smoothing) + baseline * self.label_smoothing
        data["checkpoint_target"] = np.float32(cp_target)

        return data


@dataclasses.dataclass(frozen=True)
class ComputeKeypointDistanceTargets(DataTransformFn):
    """Route raw per-condition keypoint distances into the 21-wide head layout.

    Reads ``check_distances`` [7] + ``check_distances_future`` [7] from the data
    dict (data loader fills both) and ``garment_type_id``. Produces:

        keypoint_distance_target        [21]  — current-frame target (raw N values
                                                placed into that garment's slice)
        keypoint_distance_mask          [21]  — bool: True in the garment's slice
                                                ONLY where the raw value is finite
        keypoint_distance_future_target [21]  — same for t + WM_FUTURE_HORIZON
        keypoint_distance_future_mask   [21]

    BC / DAgger / older samples have all-NaN raw distances → all-False masks,
    so MSE contributions get zeroed downstream.
    """

    def __call__(self, data: DataDict) -> DataDict:
        from lehome_solution.constants import (
            KEYPOINT_HEAD_WIDTH,
            KEYPOINT_SLICES,
            KEYPOINT_DISTANCE_CAP,
        )

        gtype_raw = data.get("garment_type_id")
        gtype = int(gtype_raw) if gtype_raw is not None else -1

        def _build(raw):
            target = np.full(KEYPOINT_HEAD_WIDTH, np.nan, dtype=np.float32)
            mask = np.zeros(KEYPOINT_HEAD_WIDTH, dtype=np.bool_)
            if raw is None or gtype not in KEYPOINT_SLICES:
                return target, mask
            raw_arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            start, width = KEYPOINT_SLICES[gtype]
            n = min(width, raw_arr.shape[0])
            vals = raw_arr[:n]
            finite = np.isfinite(vals)
            # Cap targets at KEYPOINT_DISTANCE_CAP — anything materially above
            # threshold is "very far" and the MSE should not chase fine-grained
            # structure in the tail. np.clip preserves NaN, so invalid slots
            # stay NaN. We do NOT touch the raw parquet column, only the target.
            clipped = np.clip(vals, 0.0, KEYPOINT_DISTANCE_CAP)
            target[start : start + n] = np.where(finite, clipped, np.nan)
            mask[start : start + n] = finite
            return target, mask

        cd_now = data.get("check_distances")
        cd_fut = data.get("check_distances_future")
        t_now, m_now = _build(cd_now)
        t_fut, m_fut = _build(cd_fut)
        data["keypoint_distance_target"] = t_now
        data["keypoint_distance_mask"] = m_now
        data["keypoint_distance_future_target"] = t_fut
        data["keypoint_distance_future_mask"] = m_fut
        return data


@dataclasses.dataclass(frozen=True)
class ComputeFutureAuxTargets(DataTransformFn):
    """Future (t + WM_FUTURE_HORIZON) supervision for the post-FAST and
    post-flow Q/WM heads.

    Coverage is deliberately decoupled from keypoint (``check_distances``)
    availability: ``success_future_target`` and ``completion_future_target``
    are supervised everywhere their current-time counterparts are supervised,
    so these heads train on ~100% of the batch (not just the ~3% that have
    check_distances). Only the keypoint future target is sparse.

    Implementation:
      * ``success_future_target`` mirrors ``success_target`` (label-smoothed
        episode outcome, NaN for sources with ``use_for_success_labels=False``).
        Correct because ``success`` is constant across all frames in this
        codebase — "success at t+30" equals "success at t" equals the
        terminal episode outcome.
      * ``completion_future_target`` = ``completion_future`` from the data
        loader (``min(frame_in_ep + H, ep_len - 1) / ep_len``). Always finite.

    Must run AFTER ``ComputeSuccessTarget`` so ``success_target`` is already
    populated with the label-smoothed value.
    """

    def __call__(self, data: DataDict) -> DataDict:
        # success_future_target mirrors success_target directly.
        st = data.get("success_target")
        if st is not None:
            data["success_future_target"] = np.asarray(st, dtype=np.float32)
        else:
            data["success_future_target"] = np.float32(np.nan)

        # completion_future_target from data-loader's t+H completion frac.
        cf = data.get("completion_future")
        if cf is not None and np.isfinite(float(cf)):
            data["completion_future_target"] = np.float32(cf)
        else:
            # Fall back to current completion_target if future wasn't produced.
            ct = data.get("completion_target")
            if ct is not None:
                data["completion_future_target"] = np.asarray(ct, dtype=np.float32)
            else:
                data["completion_future_target"] = np.float32(np.nan)
        return data


@dataclasses.dataclass(frozen=True)
class AdvantageToWeight:
    """Clip raw advantages for advantage embedding conditioning.

    Reads the ``advantage`` key, clips to [clip_min, clip_max], and stores:
    - ``advantage``:  clipped for advantage embedding conditioning

    Importance weighting is handled by prioritized sampling in MultiDataset
    (sampling proportional to exp(clipped_advantage)), not per-sample weights.
    """

    clip_min: float = -2.0
    clip_max: float = 2.0

    def __call__(self, data: DataDict) -> DataDict:
        adv_raw = data.get("advantage")
        if adv_raw is None:
            return data

        data["advantage"] = np.clip(
            np.asarray(adv_raw, dtype=np.float32), self.clip_min, self.clip_max
        ).astype(np.float32)
        return data
