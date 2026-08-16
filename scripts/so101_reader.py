#!/usr/bin/env python3
"""Minimal SO101 dual-arm leader reader — outputs JSON lines to stdout.

Runs in the lehome-challenge venv (needs scservo_sdk / FeetechMotorsBus).
Reads both leader arms at ~100Hz and prints one JSON line per cycle:

    {"left": {"shoulder_pan": -12.3, ...}, "right": {...},
     "ts": 1234567890.123}

Joint values are in the project-wide degree-mode convention
(``MotorNormMode.DEGREES`` for the arm joints, gripper ``RANGE_0_100``).
The DAgger controller (main venv) converts degrees → sim radians via
``lehome_solution.training.real_data_transforms.real_units_to_sim_radians``.
Motor-pct is not used anywhere.

Usage:
    lehome-challenge/.venv/bin/python scripts/so101_reader.py \
        --left_port /dev/ttyACM0 --right_port /dev/ttyACM1
"""

import argparse
import json
import os
import sys
import time

# Import motor classes directly by path to avoid triggering
# lehome.devices.__init__ which imports SO101Leader -> isaaclab -> pxr.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LEROBOT_DIR = os.path.join(
    REPO_ROOT, "lehome-challenge", "source", "lehome",
    "lehome", "devices", "lerobot",
)
sys.path.insert(0, LEROBOT_DIR)

from common.motors import (
    FeetechMotorsBus,
    Motor,
    MotorCalibration,
    MotorNormMode,
    OperatingMode,
)

# Degree-mode normalization (project-wide convention): arm joints in degrees
# around the calibration midpoint, gripper always RANGE_0_100.
MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

CALIBRATION_DIR = os.path.join(
    REPO_ROOT, "lehome-challenge", "source", "lehome",
    "lehome", "devices", "lerobot", ".cache",
)

# The real-robot runner (record_real_dagger.py) drives these same leader arms
# from lerobot's default calibration directory, and that copy is the one kept
# in sync with the hardware. The legacy per-repo ``.cache`` copy has drifted:
# its gripper ranges are ~3x narrower than the arms' real travel, which shows
# up in sim as a gripper reading half-open while physically closed. Prefer the
# real-robot calibration and fall back to the old cache only if it is absent.
_LEROBOT_LEADER_CALIBRATION = {
    "left": "bimanual_leader_left.json",
    "right": "bimanual_leader_right.json",
}


def _lerobot_calibration_dir() -> str:
    """Resolve lerobot's teleoperator calibration dir without importing lerobot."""
    base = os.environ.get("HF_LEROBOT_CALIBRATION")
    if not base:
        hf_home = os.environ.get(
            "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        )
        base = os.path.join(hf_home, "lerobot", "calibration")
    return os.path.join(base, "teleoperators", "so_leader")


def default_calibration(side: str) -> str:
    path = os.path.join(_lerobot_calibration_dir(), _LEROBOT_LEADER_CALIBRATION[side])
    if os.path.exists(path):
        return path
    return f"{side}_so101_leader.json"


def load_calibration(filename_or_path: str) -> dict[str, MotorCalibration]:
    # Treat as absolute path if it looks like one, otherwise resolve against
    # the default leader-arm calibration cache (back-compat with old callers
    # that pass a bare filename like "left_so101_leader.json").
    if os.path.isabs(filename_or_path) or os.path.sep in filename_or_path:
        path = filename_or_path
    else:
        path = os.path.join(CALIBRATION_DIR, filename_or_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            "Run calibration first via: python -m scripts.dataset_sim record "
            "--teleop_device bi-so101leader --recalibrate"
        )
    with open(path) as f:
        data = json.load(f)
    calibration = {}
    for name, d in data.items():
        calibration[name] = MotorCalibration(
            id=int(d["id"]),
            drive_mode=int(d["drive_mode"]),
            homing_offset=int(d["homing_offset"]),
            range_min=int(d["range_min"]),
            range_max=int(d["range_max"]),
        )
    return calibration


def create_bus(port: str, calibration: dict) -> FeetechMotorsBus:
    bus = FeetechMotorsBus(
        port=port,
        motors=dict(MOTORS),  # copy since bus may mutate
        calibration=calibration,
    )
    bus.connect()
    bus.disable_torque()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
    return bus


def read_arm(bus: FeetechMotorsBus) -> dict[str, float]:
    """Read normalized motor positions for one arm."""
    return bus.sync_read("Present_Position")


def read_arm_raw(bus: FeetechMotorsBus) -> dict[str, int]:
    """Read raw encoder ticks (pre-calibration) for one arm."""
    return bus.sync_read("Present_Position", normalize=False)


def main():
    parser = argparse.ArgumentParser(description="SO101 dual-arm leader reader")
    parser.add_argument("--left_port", default="/dev/ttyACM0")
    parser.add_argument("--right_port", default="/dev/ttyACM1")
    parser.add_argument("--left_calibration", default=default_calibration("left"),
                        help="Path to left-arm calibration JSON, or bare filename in default .cache/. "
                             "Defaults to the real-robot (lerobot) calibration when present.")
    parser.add_argument("--right_calibration", default=default_calibration("right"),
                        help="Path to right-arm calibration JSON, or bare filename in default .cache/. "
                             "Defaults to the real-robot (lerobot) calibration when present.")
    parser.add_argument("--hz", type=int, default=100, help="Reading rate")
    parser.add_argument("--max_readings", type=int, default=0,
                        help="Stop after N readings (0 = run until SIGINT).")
    args = parser.parse_args()

    left_cal = load_calibration(args.left_calibration)
    right_cal = load_calibration(args.right_calibration)
    print(f"[so101_reader] left calibration:  {args.left_calibration}", file=sys.stderr, flush=True)
    print(f"[so101_reader] right calibration: {args.right_calibration}", file=sys.stderr, flush=True)

    left_bus = create_bus(args.left_port, left_cal)
    right_bus = create_bus(args.right_port, right_cal)

    # Surface the calibration range_min / range_max + homing_offset for each
    # joint so callers can see exactly which encoder ticks the normalization
    # was anchored against (helpful for diagnosing saturation when a motor
    # has been mechanically replaced).
    def _cal_dump(cal: dict[str, MotorCalibration]) -> dict[str, dict[str, int]]:
        return {
            name: {
                "id": c.id,
                "homing_offset": c.homing_offset,
                "range_min": c.range_min,
                "range_max": c.range_max,
                "drive_mode": c.drive_mode,
            }
            for name, c in cal.items()
        }

    calibration_info = {"left": _cal_dump(left_cal), "right": _cal_dump(right_cal)}

    # Print ready signal
    print(json.dumps({"status": "ready"}), flush=True)

    period = 1.0 / args.hz
    n = 0
    try:
        while True:
            t0 = time.monotonic()
            left = read_arm(left_bus)
            right = read_arm(right_bus)
            left_raw = read_arm_raw(left_bus)
            right_raw = read_arm_raw(right_bus)
            msg = {
                "left": left,
                "right": right,
                "left_raw": left_raw,
                "right_raw": right_raw,
                "calibration": calibration_info,
                "ts": time.time(),
            }
            print(json.dumps(msg), flush=True)
            n += 1
            if args.max_readings and n >= args.max_readings:
                break
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            left_bus.disconnect()
        except Exception:
            pass
        try:
            right_bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
