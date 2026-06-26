"""Shared helpers for real-robot tooling.

Two responsibilities, intentionally split so the module imports cleanly
from either venv:

1. **Pure-data constants and dict<->vec adapters** — usable from anywhere
   (the main JAX venv as well as the lehome-challenge venv used by the
   real-robot scripts). No lerobot imports at module level.

2. **lerobot config builders** — lazy-import ``lerobot.*`` so the module
   stays importable without lerobot. Calls raise ``ImportError`` if invoked
   without lerobot on path.

The motor-name list, the 12-vec ordering, and the camera-key conventions
match the lerobot ``BiSOFollower`` observation/action dicts (and therefore the
``data/lehome_real`` dataset schema) exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Constants — must match the lerobot BiSOFollower motor naming / ordering.
# ---------------------------------------------------------------------------

MOTOR_NAMES: tuple[str, ...] = (
    "left_shoulder_pan.pos",  "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos",    "left_wrist_roll.pos",    "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos",   "right_wrist_roll.pos",   "right_gripper.pos",
)
ACTION_DIM = len(MOTOR_NAMES)  # 12

# Mapping of yaml.cameras.<key> → (which_arm, which_arm_camera_key) so that
# BiSOFollower's left_/right_ prefix yields the names the eval protocol expects.
# yaml `top` belongs to the right arm with sub-key `front` ⇒ emits `right_front`.
_CAMERA_ROUTING: dict[str, tuple[str, str]] = {
    "top":         ("right", "front"),
    "left_wrist":  ("left",  "wrist"),
    "right_wrist": ("right", "wrist"),
}


# ---------------------------------------------------------------------------
# Pure-Python dict <-> vec12 adapters (no hardware deps).
# ---------------------------------------------------------------------------

def state_dict_to_vec12(obs: dict[str, Any]) -> np.ndarray:
    """Pull the 12 ``*.pos`` floats out of a robot observation dict.

    Works for both the gRPC ``TimedObservation.observation`` payload and
    ``BiSOFollower.get_observation()`` output — both use the canonical
    motor-name keys defined in :data:`MOTOR_NAMES`.
    """
    return np.asarray(
        [float(obs.get(k, 0.0)) for k in MOTOR_NAMES], dtype=np.float32,
    )


def vec12_to_action_dict(vec12: Iterable[float]) -> dict[str, float]:
    """Inverse of :func:`state_dict_to_vec12` — wrap a 12-vec into the dict
    shape ``BiSOFollower.send_action`` accepts."""
    arr = np.asarray(vec12, dtype=np.float32).reshape(-1)
    if arr.shape != (ACTION_DIM,):
        raise ValueError(f"expected shape ({ACTION_DIM},), got {arr.shape}")
    return {name: float(arr[i]) for i, name in enumerate(MOTOR_NAMES)}


# ---------------------------------------------------------------------------
# lerobot config builders (lazy imports — see module docstring).
# ---------------------------------------------------------------------------

def build_camera_config(cam_cfg: dict[str, Any]):
    """One ``configs/real_robot.yaml`` camera entry → a lerobot ``CameraConfig``.

    Supports ``backend: v4l2``/``opencv`` (→ :class:`OpenCVCameraConfig`) and
    ``backend: pyrealsense2`` (→ :class:`RealSenseCameraConfig`). Imports
    lazily so the surrounding module loads without lerobot.
    """
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    backend = (cam_cfg.get("backend") or "v4l2").lower()
    width = int(cam_cfg["width"])
    height = int(cam_cfg["height"])
    fps = int(cam_cfg["fps"])

    if backend in ("v4l2", "opencv"):
        device = cam_cfg.get("device") or cam_cfg.get("index_or_path")
        if device is None:
            raise KeyError("OpenCV camera entry missing 'device' / 'index_or_path'")
        if isinstance(device, str) and device.startswith("/dev/"):
            device = Path(device)
        return OpenCVCameraConfig(
            index_or_path=device,
            width=width,
            height=height,
            fps=fps,
            fourcc=cam_cfg.get("fourcc"),
        )

    if backend == "pyrealsense2":
        serial = cam_cfg.get("serial") or cam_cfg.get("serial_number_or_name") or ""
        if not serial:
            raise ValueError(
                "RealSense camera entry must specify 'serial' "
                "(lerobot RealSenseCameraConfig requires serial_number_or_name)."
            )
        return RealSenseCameraConfig(
            serial_number_or_name=serial,
            fps=fps,
            width=width,
            height=height,
            use_depth=bool(cam_cfg.get("use_depth", False)),
        )

    raise ValueError(f"unsupported camera backend: {backend!r}")


def ensure_default_calibration(kind: str, device_name: str, device_id: str) -> None:
    """Fail fast if calibration JSONs are missing from lerobot's default dir.

    ``kind`` is ``"robots"`` or ``"teleoperators"``; ``device_name`` the lerobot
    class name (``so_follower`` / ``so_leader``); ``device_id`` the bimanual id
    (``bimanual_follower`` / ``bimanual_leader``). lerobot resolves per-arm
    calibration at ``{HF_LEROBOT_CALIBRATION}/{kind}/{device_name}/{device_id}_{side}.json``
    and silently launches an interactive re-calibration on connect when the
    file is absent — raising here keeps that from happening mid-session.
    """
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    cal_dir = HF_LEROBOT_CALIBRATION / kind / device_name
    for side in ("left", "right"):
        fpath = cal_dir / f"{device_id}_{side}.json"
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Calibration JSON missing: {fpath}. Copy this rig's committed "
                f"file from configs/calibration/{device_id}_{side}.json, or run "
                f"the lerobot calibration once for new hardware. "
                f"See configs/calibration/README.md."
            )


def build_bi_so_follower_config(
    yaml_cfg: dict[str, Any],
    robot_id: str,
    *,
    skip_cameras: bool = False,
    use_degrees: bool = True,
):
    """Translate ``configs/real_robot.yaml`` to ``BiSOFollowerConfig``.

    ``use_degrees=True`` (default — the project-wide convention): arm joints
    speak degrees, gripper 0-100. Motor-pct (``use_degrees=False``) is kept
    only as a lerobot escape hatch and should not be used.

    Camera routing (see :data:`_CAMERA_ROUTING`):

    * ``yaml.cameras.top``         → ``right_arm.cameras["front"]`` ⇒ ``right_front``
    * ``yaml.cameras.left_wrist``  → ``left_arm.cameras["wrist"]``  ⇒ ``left_wrist``
    * ``yaml.cameras.right_wrist`` → ``right_arm.cameras["wrist"]`` ⇒ ``right_wrist``

    Calibration comes from lerobot's default directory
    (``~/.cache/huggingface/lerobot/calibration/robots/so_follower/``); missing
    JSONs raise immediately instead of letting lerobot silently fall through
    to an interactive re-calibration. See configs/calibration/README.md.

    ``skip_cameras=True`` skips camera config (e.g. open-loop replay where
    only motors are driven, or hardware bring-up before cameras are wired).
    """
    from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

    arms = yaml_cfg["arms"]

    # Validate calibration before cameras — calibration failures are usually
    # actionable (drop the JSON in place), camera ones need hardware on-hand.
    ensure_default_calibration("robots", "so_follower", robot_id)

    side_cams: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
    if not skip_cameras:
        cams_yaml = yaml_cfg["cameras"]
        for yaml_key, (side, arm_subkey) in _CAMERA_ROUTING.items():
            if yaml_key not in cams_yaml:
                raise KeyError(f"configs/real_robot.yaml missing cameras.{yaml_key}")
            side_cams[side][arm_subkey] = build_camera_config(cams_yaml[yaml_key])

    return BiSOFollowerConfig(
        id=robot_id,
        left_arm_config=SOFollowerConfig(
            port=arms["left"]["port"],
            cameras=side_cams["left"],
            use_degrees=use_degrees,
        ),
        right_arm_config=SOFollowerConfig(
            port=arms["right"]["port"],
            cameras=side_cams["right"],
            use_degrees=use_degrees,
        ),
    )
