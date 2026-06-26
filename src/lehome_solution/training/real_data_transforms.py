"""Joint-unit conversion between real-robot degrees and sim USD radians.

Units convention:

* **Real-robot datasets are recorded in lerobot DEGREES mode** (`use_degrees=True`
  on both followers and leaders). The five arm joints store degrees around the
  calibration midpoint, which the calibration procedure aligns with the sim
  USD zero pose. The **gripper is the one exception**: lerobot always
  normalizes it to ``RANGE_0_100`` (0 = closed, 100 = open) regardless of
  degrees mode.
* **Sim datasets are in USD radians** (30 Hz). The sim gripper joint spans
  [-10°, 100°] in USD degrees.

So the conversion is per-joint:

* arm joints (indices 0-4 and 6-10): plain deg ↔ rad,
* gripper (indices 5 and 11): affine [0, 100] ↔ [-10°, 100°], then deg ↔ rad.

Vector layout: indices [0..5] are the LEFT arm, [6..11] the RIGHT arm, each
``(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)``.

Motor-pct (lerobot ``RANGE_M100_100``) is intentionally NOT supported anywhere
anymore — wrong-unit datasets are discarded, not converted.
"""

from __future__ import annotations

import numpy as np

# Gripper normalization: lerobot RANGE_0_100 ↔ sim USD degree range.
_GRIPPER_PCT_RANGE = (0.0, 100.0)
_GRIPPER_USD_DEG_RANGE = (-10.0, 100.0)
_GRIPPER_INDICES = (5, 11)


def _build_deg_to_rad_affine() -> tuple[np.ndarray, np.ndarray]:
    """Per-dim (slope, intercept) such that ``rad = value * slope + intercept``.

    Identity (× π/180) for the arm joints; the gripper dims additionally fold
    in the RANGE_0_100 → USD-degree affine.
    """
    slopes_deg = np.ones(12, dtype=np.float64)
    intercepts_deg = np.zeros(12, dtype=np.float64)
    p_min, p_max = _GRIPPER_PCT_RANGE
    d_min, d_max = _GRIPPER_USD_DEG_RANGE
    g_slope = (d_max - d_min) / (p_max - p_min)
    g_intercept = d_min - p_min * g_slope
    for gi in _GRIPPER_INDICES:
        slopes_deg[gi] = g_slope
        intercepts_deg[gi] = g_intercept
    deg_to_rad = np.pi / 180.0
    return (
        (slopes_deg * deg_to_rad).astype(np.float32),
        (intercepts_deg * deg_to_rad).astype(np.float32),
    )


_DEG_TO_RAD_SLOPE, _DEG_TO_RAD_INTERCEPT = _build_deg_to_rad_affine()


def real_units_to_sim_radians(values_deg: np.ndarray) -> np.ndarray:
    """Convert real-robot degree-mode values (state or actions) to sim USD radians.

    Arm joints: deg → rad. Gripper dims (5, 11): RANGE_0_100 → USD radians.
    Supports any leading shape; last axis must be 12 (left arm 6 + right arm 6).
    """
    values_deg = np.asarray(values_deg, dtype=np.float32)
    if values_deg.shape[-1] != 12:
        raise ValueError(f"expected last axis size 12, got shape {values_deg.shape}")
    return (values_deg * _DEG_TO_RAD_SLOPE + _DEG_TO_RAD_INTERCEPT).astype(np.float32)


def sim_radians_to_real_units(values_rad: np.ndarray) -> np.ndarray:
    """Inverse of :pyfunc:`real_units_to_sim_radians`.

    Converts sim USD radians into the real-robot degree-mode convention
    (arm joints in degrees, gripper in RANGE_0_100) that lerobot's
    ``SOFollower.send_action`` consumes with ``use_degrees=True``.
    Supports any leading shape; last axis must be 12.
    """
    values_rad = np.asarray(values_rad, dtype=np.float32)
    if values_rad.shape[-1] != 12:
        raise ValueError(f"expected last axis size 12, got shape {values_rad.shape}")
    return ((values_rad - _DEG_TO_RAD_INTERCEPT) / _DEG_TO_RAD_SLOPE).astype(np.float32)
