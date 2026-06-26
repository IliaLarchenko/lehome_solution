"""Normalization statistics.

Supports per-timestamp normalization with smooth parametric fits: the std is
fit as f(t) = a + s*sqrt(t + e) with a learnable e > 0 (``fit_sqrt_norm``) and
the mean is fit linearly as f(t) = offset + t*slope (``fit_linear_norm``).
"""

import json
import pathlib
from typing import NamedTuple

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile
    
    # Per-timestamp normalization stats (shape: [action_horizon, action_dim])
    # If None, use regular normalization
    per_timestamp_mean: numpydantic.NDArray | None = None
    per_timestamp_std: numpydantic.NDArray | None = None
    per_timestamp_q01: numpydantic.NDArray | None = None
    per_timestamp_q99: numpydantic.NDArray | None = None
    
    # Action correlation matrix for correlated noise generation (legacy - full matrix)
    # Shape: [action_horizon * action_dim, action_horizon * action_dim]
    # Cholesky decomposition of the empirical covariance matrix
    action_correlation_cholesky: numpydantic.NDArray | None = None


class SmoothFitResult(NamedTuple):
    """Result of fitting a smooth parametric function per dimension."""
    fitted_values: np.ndarray   # shape: [H, D] - fitted values at each timestep
    offsets: np.ndarray         # shape: [D] - intercept for each dimension
    slopes: np.ndarray          # shape: [D] - coefficient for each dimension
    epsilons: np.ndarray        # shape: [D] - shift parameter (for sqrt fit), 0 for linear
    r_squared: np.ndarray       # shape: [D] - R² per dimension
    max_abs_error: np.ndarray   # shape: [D] - max absolute error per dimension


def _fit_with_basis(basis: np.ndarray, raw_values: np.ndarray) -> SmoothFitResult:
    """Fit raw_values = basis @ [offset, slope] via least squares.

    Args:
        basis: shape [H, 2] – design matrix (column 0 = ones, column 1 = feature).
        raw_values: shape [H, D] – raw per-timestamp values.

    Returns:
        SmoothFitResult with fitted values and diagnostics.
    """
    D = raw_values.shape[1]
    coeffs, _, _, _ = np.linalg.lstsq(basis, raw_values, rcond=None)

    offsets = coeffs[0]  # [D]
    slopes = coeffs[1]   # [D]
    fitted = basis @ coeffs  # [H, D]

    # R² per dimension
    ss_res = np.sum((raw_values - fitted) ** 2, axis=0)   # [D]
    ss_tot = np.sum((raw_values - raw_values.mean(axis=0)) ** 2, axis=0)  # [D]
    r_squared = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 1.0)

    max_abs_error = np.max(np.abs(raw_values - fitted), axis=0)  # [D]

    return SmoothFitResult(
        fitted_values=fitted,
        offsets=offsets,
        slopes=slopes,
        epsilons=np.zeros(D),
        r_squared=r_squared,
        max_abs_error=max_abs_error,
    )


def fit_sqrt_norm(raw_values: np.ndarray) -> SmoothFitResult:
    """Fit f(t) = a + s * sqrt(t + e) with learnable e > 0, per dimension.

    Best for **std**: delta actions accumulate like a random walk, so
    variance grows ~ t and std grows ~ sqrt(t).  The learnable epsilon
    controls curvature at early timesteps — small e gives a steep initial
    rise, large e flattens it.

    Uses nonlinear least squares (scipy) to jointly optimize (a, s, e)
    per dimension.

    Args:
        raw_values: shape [H, D] – raw per-timestamp values (typically std).
    """
    from scipy.optimize import minimize

    H, D = raw_values.shape
    t = np.arange(H, dtype=np.float64)

    fitted = np.zeros_like(raw_values)
    offsets = np.zeros(D)
    slopes = np.zeros(D)
    epsilons = np.zeros(D)
    r_squared = np.zeros(D)
    max_abs_error = np.zeros(D)

    # Get a warm start from the fixed e=1 linear least squares
    basis_e1 = np.column_stack([np.ones(H), np.sqrt(t + 1.0)])
    coeffs_e1, _, _, _ = np.linalg.lstsq(basis_e1, raw_values, rcond=None)

    for d in range(D):
        y = raw_values[:, d]
        ss_tot = np.sum((y - y.mean()) ** 2)

        # Parameterize as (a, s, log_e) to keep e > 0
        a0 = coeffs_e1[0, d]
        s0 = coeffs_e1[1, d]
        log_e0 = 0.0  # log(1) = 0, i.e. start with e=1

        def loss(params):
            a, s, log_e = params
            e = np.exp(log_e)
            pred = a + s * np.sqrt(t + e)
            return np.sum((y - pred) ** 2)

        res = minimize(loss, x0=[a0, s0, log_e0], method="L-BFGS-B")

        a_opt, s_opt, log_e_opt = res.x
        e_opt = np.exp(log_e_opt)

        pred = a_opt + s_opt * np.sqrt(t + e_opt)
        ss_res = np.sum((y - pred) ** 2)

        offsets[d] = a_opt
        slopes[d] = s_opt
        epsilons[d] = e_opt
        fitted[:, d] = pred
        r_squared[d] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        max_abs_error[d] = np.max(np.abs(y - pred))

    return SmoothFitResult(
        fitted_values=fitted,
        offsets=offsets,
        slopes=slopes,
        epsilons=epsilons,
        r_squared=r_squared,
        max_abs_error=max_abs_error,
    )


def fit_linear_norm(raw_values: np.ndarray) -> SmoothFitResult:
    """Fit f(t) = offset + t * slope to per-timestamp statistics.

    Best for **mean**: the average drift from the current state grows
    roughly linearly with time (constant velocity bias).

    Args:
        raw_values: shape [H, D] – raw per-timestamp values (typically mean).
    """
    H = raw_values.shape[0]
    t = np.arange(H, dtype=np.float64)
    basis = np.column_stack([np.ones(H), t])
    return _fit_with_basis(basis, raw_values)


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
