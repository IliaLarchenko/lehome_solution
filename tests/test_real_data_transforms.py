"""Unit tests for `lehome_solution.training.real_data_transforms`.

Covers the degrees ↔ sim-radians conversion pair (arm joints: plain deg↔rad;
gripper dims 5/11: RANGE_0_100 ↔ the sim USD [-10°, 100°] range).
"""

from __future__ import annotations

import numpy as np
import pytest

from lehome_solution.training import real_data_transforms as rdt


def test_real_units_to_sim_radians_arm_joints_are_plain_deg_to_rad():
    """All non-gripper dims convert with the identity deg→rad mapping."""
    deg = np.zeros(12, dtype=np.float32)
    deg[0] = 45.0    # left shoulder_pan
    deg[1] = -90.0   # left shoulder_lift
    deg[4] = 180.0   # left wrist_roll (full rotation)
    deg[9] = -95.0   # right wrist_flex
    out = rdt.real_units_to_sim_radians(deg)
    arm_idx = [i for i in range(12) if i not in (5, 11)]
    np.testing.assert_allclose(out[arm_idx], np.radians(deg[arm_idx]), atol=1e-6)


def test_real_units_to_sim_radians_gripper_affine():
    """Gripper dims map RANGE_0_100 onto the USD [-10°, 100°] span."""
    deg = np.zeros(12, dtype=np.float32)
    out = rdt.real_units_to_sim_radians(deg)
    # gripper=0 (closed) -> -10° -> -π/18 rad
    np.testing.assert_allclose(out[5], np.radians(-10.0), atol=1e-6)
    np.testing.assert_allclose(out[11], np.radians(-10.0), atol=1e-6)
    # gripper=100 (open) -> 100°
    deg[5] = 100.0
    deg[11] = 100.0
    out = rdt.real_units_to_sim_radians(deg)
    np.testing.assert_allclose(out[5], np.radians(100.0), atol=1e-5)
    np.testing.assert_allclose(out[11], np.radians(100.0), atol=1e-5)


def test_real_units_to_sim_radians_supports_leading_shapes():
    deg = np.zeros((4, 7, 12), dtype=np.float32)
    out = rdt.real_units_to_sim_radians(deg)
    assert out.shape == (4, 7, 12)
    assert out.dtype == np.float32


def test_real_units_to_sim_radians_validates_last_axis():
    with pytest.raises(ValueError):
        rdt.real_units_to_sim_radians(np.zeros(10, dtype=np.float32))


def test_sim_radians_to_real_units_roundtrip():
    """Forward then inverse on the same array reproduces the input."""
    rng = np.random.default_rng(0)
    deg = rng.uniform(-180.0, 180.0, size=(4, 12)).astype(np.float32)
    deg[..., 5] = rng.uniform(0.0, 100.0, size=4)
    deg[..., 11] = rng.uniform(0.0, 100.0, size=4)
    rad = rdt.real_units_to_sim_radians(deg)
    back = rdt.sim_radians_to_real_units(rad)
    np.testing.assert_allclose(back, deg, atol=1e-3)


def test_sim_radians_to_real_units_validates_last_axis():
    with pytest.raises(ValueError):
        rdt.sim_radians_to_real_units(np.zeros(10, dtype=np.float32))


def test_sim_radians_to_real_units_known_values():
    """Spot-check the inverse direction against hand-computed values."""
    rad = np.zeros(12, dtype=np.float32)
    rad[1] = np.pi / 2          # left shoulder_lift 90°
    rad[5] = np.radians(-10.0)  # left gripper fully closed
    rad[11] = np.radians(100.0)  # right gripper fully open
    out = rdt.sim_radians_to_real_units(rad)
    np.testing.assert_allclose(out[1], 90.0, atol=1e-4)
    np.testing.assert_allclose(out[5], 0.0, atol=1e-4)
    np.testing.assert_allclose(out[11], 100.0, atol=1e-4)
