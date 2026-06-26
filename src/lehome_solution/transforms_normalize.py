"""Custom Normalize transform with per-timestamp support.
"""

import dataclasses
import numpy as np

from openpi.transforms import DataTransformFn, DataDict, apply_tree, pad_to_dim
from openpi.shared import array_typing as at
from lehome_solution.shared.normalize import NormStats


@dataclasses.dataclass(frozen=True)
class NormalizeWithPerTimestamp(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
    strict: bool = False
    use_per_timestamp: bool = False
    # Always quantile-normalize 1D inputs (state) to [-1, 1], matching OpenPi's
    # convention for state discretization bins.  Actions (2D) are unaffected.
    state_quantile_norm: bool = True

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            for stats in at.tree_leaves(self.norm_stats):
                if isinstance(stats, NormStats) and (stats.q01 is None or stats.q99 is None):
                    raise ValueError("Quantile normalization requires q01 and q99 in norm_stats")

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        # State (1D): always use quantile normalization to [-1, 1] so that
        # digitize(bins=linspace(-1, 1, 257)) covers the full range.
        if x.ndim < 2 and self.state_quantile_norm and stats.q01 is not None:
            q01 = stats.q01[..., : x.shape[-1]]
            q99 = stats.q99[..., : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

        # Actions (2D+): z-score, optionally per-timestamp
        if self.use_per_timestamp and stats.per_timestamp_mean is not None and x.ndim >= 2:
            mean = pad_to_dim(stats.per_timestamp_mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(stats.per_timestamp_std, x.shape[-1], axis=-1, value=1.0)
            T = x.shape[-2]
            if T > mean.shape[0]:
                raise ValueError(
                    f"Per-timestamp normalization: input has {T} timesteps but "
                    f"stats only cover {mean.shape[0]} (action_horizon mismatch?)"
                )
            mean = mean[:T, :]
            std = std[:T, :]
            return (x - mean) / (std + 1e-6)
        else:
            mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
            return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile normalization requires q01 and q99")

        # Check if this is actions and we should use per-timestamp normalization
        if self.use_per_timestamp and stats.per_timestamp_q01 is not None and x.ndim >= 2:
            T = x.shape[-2]
            if T > stats.per_timestamp_q01.shape[0]:
                raise ValueError(
                    f"Per-timestamp quantile normalization: input has {T} timesteps "
                    f"but stats only cover {stats.per_timestamp_q01.shape[0]}"
                )
            q01 = stats.per_timestamp_q01[:T, : x.shape[-1]]
            q99 = stats.per_timestamp_q99[:T, : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            q01 = pad_to_dim(stats.q01, x.shape[-1], axis=-1, value=0.0)
            q99 = pad_to_dim(stats.q99, x.shape[-1], axis=-1, value=1.0)
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class UnnormalizeWithPerTimestamp(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
    use_per_timestamp: bool = False
    # Must match NormalizeWithPerTimestamp.state_quantile_norm — always invert
    # quantile normalization for 1D inputs (state).
    state_quantile_norm: bool = True

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            for stats in at.tree_leaves(self.norm_stats):
                if isinstance(stats, NormStats) and (stats.q01 is None or stats.q99 is None):
                    raise ValueError("Quantile normalization requires q01 and q99 in norm_stats")

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        # State (1D): invert the quantile normalization applied by _normalize
        if x.ndim < 2 and self.state_quantile_norm and stats.q01 is not None:
            q01 = stats.q01[..., : x.shape[-1]]
            q99 = stats.q99[..., : x.shape[-1]]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

        # Check if this is actions and we should use per-timestamp normalization
        if self.use_per_timestamp and stats.per_timestamp_mean is not None and x.ndim >= 2:
            # x has shape [..., action_horizon, action_dim] for actions
            # stats.per_timestamp_mean has shape [action_horizon, action_dim]
            mean = pad_to_dim(stats.per_timestamp_mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(stats.per_timestamp_std, x.shape[-1], axis=-1, value=1.0)
            # Ensure we only use the appropriate timesteps
            mean = mean[: x.shape[-2], :]
            std = std[: x.shape[-2], :]
            return x * (std + 1e-6) + mean
        else:
            # Regular unnormalization
            mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
            return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile unnormalization requires q01 and q99")
        
        # Check if this is actions and we should use per-timestamp normalization
        if self.use_per_timestamp and stats.per_timestamp_q01 is not None and x.ndim >= 2:
            # x has shape [..., action_horizon, action_dim] for actions
            # stats.per_timestamp_q01 has shape [action_horizon, action_dim]
            q01 = stats.per_timestamp_q01[: x.shape[-2], : x.shape[-1]]
            q99 = stats.per_timestamp_q99[: x.shape[-2], : x.shape[-1]]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        else:
            # Regular quantile unnormalization
            q01, q99 = stats.q01, stats.q99
            if (dim := q01.shape[-1]) < x.shape[-1]:
                return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
