#!/usr/bin/env python
"""Record a real-robot dataset — DAgger (manual+autonomous) or manual-only.

DAgger mode (default): combines teleop recording with autonomous policy rollouts.
SPACE switches between manual and autonomous mid-episode.  Requires a running
gRPC policy server.

Manual-only mode (``--manual_only``): pure teleop recording with background video
encoding.  No policy server needed.  Produces the standard
``data/lehome_real/four_types_merged``-style dataset format without blocking on
video encode after each episode.

Runs in the lehome-challenge venv (needs lerobot + hardware stack).

    source lehome-challenge/.venv/bin/activate

    # DAgger mode — needs policy server running in another terminal:
    python scripts/record_real_dagger.py \\
        --config configs/real_robot.yaml \\
        --garment top_long \\
        --server localhost:8080

    # Manual-only mode — no server needed:
    python scripts/record_real_dagger.py \\
        --config configs/real_robot.yaml \\
        --garment top_long \\
        --manual_only

Keyboard controls (physical keyboard, any focus):
    Space        switch manual <-> autonomous (DAgger only; ignored in manual-only)
    Right arrow  save episode -> manual reset phase -> next episode
    Left arrow   discard episode and re-record immediately
    Esc          stop the session
"""

from __future__ import annotations

import argparse
import logging
import os
import base64
import json
import select
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.cameras.realsense import camera_realsense as _rs_cam_mod
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.teleoperators.bi_so_leader.config_bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig
from websockets.sync.client import connect as ws_connect

from lehome_solution.shared.real_robot_config import (
    build_bi_so_follower_config,
    ensure_default_calibration,
    state_dict_to_vec12,
    vec12_to_action_dict,
)

log = logging.getLogger("record_real_dagger")

VALID_GARMENTS = ("top_long", "top_short", "pant_long", "pant_short")

# Terminal colors
_GREEN = "\033[1;32m"
_YELLOW = "\033[1;33m"
_MAGENTA = "\033[1;35m"
_CYAN = "\033[1;36m"
_WHITE = "\033[1;37m"
_BG_RED = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_BLUE = "\033[44m"
_BG_MAGENTA = "\033[45m"
_BG_CYAN = "\033[46m"
_BG_YELLOW = "\033[43m"
_RESET = "\033[0m"
_BAR = "=" * 60


def _banner(color: str, icon: str, text: str) -> None:
    print(f"\n{color}{_BAR}\n  {icon}  {text}\n{_BAR}{_RESET}\n", flush=True)


def _status(color: str, icon: str, text: str) -> None:
    print(f"{color}  {icon}  {text}{_RESET}", flush=True)

# Safe rest pose in the robot's native degree-mode units (lerobot
# use_degrees=True: arm joints in degrees, gripper 0-100). Captured from a
# live `robot.get_observation()` at the rest pose — re-capture if the rig or
# calibration changes.
SAFE_REST_DEGREES = np.array([
    2.24, -97.58, 100.0, 28.47, -2.27, 2.35,
    -1.16, -96.10, 99.55, 35.15, 1.73, 0.68,
], dtype=np.float32)



# ---------------------------------------------------------------------------
# RealSense patches (D435 bgr8 format workaround + cold-start timeout bump)
# ---------------------------------------------------------------------------

def _patch_realsense() -> None:
    """bgr8 request + elevated timeout + connect-retry for the D435."""
    import cv2
    import pyrealsense2 as rs

    def patched_configure(self, rs_config):
        rs.config.enable_device(rs_config, self.serial_number)
        if self.width and self.height and self.fps:
            rs_config.enable_stream(rs.stream.color, self.capture_width,
                                    self.capture_height, rs.format.bgr8, self.fps)
            if self.use_depth:
                rs_config.enable_stream(rs.stream.depth, self.capture_width,
                                        self.capture_height, rs.format.z16, self.fps)
        else:
            rs_config.enable_stream(rs.stream.color)
            if self.use_depth:
                rs_config.enable_stream(rs.stream.depth)

    _rs_cam_mod.RealSenseCamera._configure_rs_pipeline_config = patched_configure

    def patched_read(self, color_mode=None, timeout_ms=200):
        timeout_ms = max(timeout_ms, 3000)
        ret, frame = self.rs_pipeline.try_wait_for_frames(timeout_ms=timeout_ms)
        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")
        bgr = np.asanyarray(frame.get_color_frame().get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._postprocess_image(rgb, color_mode)

    _rs_cam_mod.RealSenseCamera.read = patched_read

    if not hasattr(_rs_cam_mod.RealSenseCamera, "_orig_connect"):
        _rs_cam_mod.RealSenseCamera._orig_connect = _rs_cam_mod.RealSenseCamera.connect

    def patched_connect(self, warmup=True):
        try:
            return _rs_cam_mod.RealSenseCamera._orig_connect(self, warmup=warmup)
        except (RuntimeError, ConnectionError) as e:
            self.rs_pipeline = None
            self.rs_profile = None
            try:
                import pyrealsense2 as rs2
                serial = str(getattr(self, "serial_number", "") or "")
                for dev in rs2.context().devices:
                    if serial and dev.get_info(rs2.camera_info.serial_number) != serial:
                        continue
                    log.warning("RealSense connect failed (%s); hardware_reset + retry", e)
                    dev.hardware_reset()
                    break
            except Exception:
                pass
            time.sleep(3.0)
            return _rs_cam_mod.RealSenseCamera._orig_connect(self, warmup=warmup)

    _rs_cam_mod.RealSenseCamera.connect = patched_connect


def _hardware_reset_realsense(yaml_cfg: dict) -> None:
    top = yaml_cfg.get("cameras", {}).get("top")
    if not top or (top.get("backend") or "").lower() != "pyrealsense2":
        return
    serial = str(top.get("serial") or "")
    try:
        import pyrealsense2 as rs
    except ImportError:
        return
    try:
        for dev in rs.context().devices:
            if serial and dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            log.info("hardware_reset() RealSense %s (4s settle)...",
                     dev.get_info(rs.camera_info.serial_number))
            dev.hardware_reset()
            time.sleep(4.0)
            return
    except Exception as e:
        log.warning("hardware_reset() raised: %s — continuing", e)


# ---------------------------------------------------------------------------
# Keyboard listener (evdev, works regardless of terminal focus)
# ---------------------------------------------------------------------------

class KeyboardListener:
    _KEY_MAP = {
        "KEY_SPACE": "SPACE", "KEY_LEFT": "LEFT",
        "KEY_RIGHT": "RIGHT", "KEY_ESC": "ESC",
    }

    def __init__(self) -> None:
        import evdev
        self._evdev = evdev
        self._events: Queue[str] = Queue()
        self._stop = threading.Event()
        self._devices: list = []
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        ecodes = self._evdev.ecodes
        watch = {getattr(ecodes, k) for k in self._KEY_MAP}
        code_map = {getattr(ecodes, k): v for k, v in self._KEY_MAP.items()}
        for path in self._evdev.list_devices():
            try:
                dev = self._evdev.InputDevice(path)
            except (OSError, PermissionError):
                continue
            if not watch.intersection(dev.capabilities().get(ecodes.EV_KEY, [])):
                dev.close()
                continue
            self._devices.append(dev)
            t = threading.Thread(target=self._loop, args=(dev, code_map),
                                 daemon=True, name=f"kb-{Path(path).name}")
            t.start()
            self._threads.append(t)
            log.info("Keyboard: %s (%s)", path, dev.name)
        if not self._devices:
            raise RuntimeError("No keyboard found under /dev/input/event*")

    def _loop(self, device, code_map: dict) -> None:
        ecodes = self._evdev.ecodes
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([device.fd], [], [], 0.2)
                if not ready:
                    continue
                for ev in device.read():
                    if ev.type == ecodes.EV_KEY and ev.value == 1:
                        name = code_map.get(ev.code)
                        if name:
                            self._events.put(name)
        except OSError:
            pass

    def poll(self) -> str | None:
        try:
            return self._events.get_nowait()
        except Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self._events.get_nowait()
            except Empty:
                return

    def stop(self) -> None:
        self._stop.set()
        for d in self._devices:
            try:
                d.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Policy client (WebSocket to scripts/serve.py)
# ---------------------------------------------------------------------------
# The policy server is the same generic one the sim eval uses
# (`scripts/serve.py`, stateless `infer_chunk` protocol). All real-robot
# session logic lives here on the client side:
#   * rolling inpaint anchor — `next_initial_actions` from each response is
#     sent back as `initial_actions` with the next request, so consecutive
#     chunks stay aligned;
#   * garment-type bootstrap — the first chunk of a session latches
#     `garment_type_pred` and is replaced by a hold-position chunk so the
#     robot stays still while the one-chunk delay flushes through; the
#     latched id is sent with every subsequent request.
# Everything else is a pure pass-through: states and actions travel in the
# robot's native degree-mode units (use_degrees=True; gripper 0-100) with no
# per-joint corrections or overrides.

# Diffusion knobs for real-robot inference (match the sim-eval defaults).
INFERENCE_CONFIG = {
    "noise_temperature": 0.9,
    "cfg_scale": 5.0,
    "time_threshold_inpaint": 0.5,
}


def _b64_image(img: np.ndarray) -> dict:
    img = np.ascontiguousarray(img)
    return {
        "base64": base64.b64encode(img.tobytes()).decode("ascii"),
        "shape": list(img.shape),
        "dtype": str(img.dtype),
    }


class PolicyClient:
    def __init__(self, server_address: str, garment_type_id: int | None = None) -> None:
        self._url = (
            server_address if server_address.startswith("ws")
            else f"ws://{server_address}"
        )
        # Optional fixed garment id 0..3; None = latch from the first chunk.
        self._forced_garment_type_id = garment_type_id
        self._garment_type_id = garment_type_id
        self._initial_actions: np.ndarray | None = None
        self._ws = None
        self._connect()

    def _connect(self) -> None:
        for attempt in range(20):
            try:
                self._ws = ws_connect(
                    self._url, max_size=200 * 1024 * 1024, open_timeout=30)
                log.info("PolicyClient connected to %s", self._url)
                return
            except Exception as e:
                log.warning("Policy server connect %d/20: %s", attempt + 1, e)
                time.sleep(3)
        raise ConnectionError(f"Could not connect to policy server at {self._url}")

    def reset_session(self) -> None:
        """New episode / mode switch: drop the inpaint anchor and re-run the
        garment bootstrap (unless a fixed garment id was forced)."""
        self._initial_actions = None
        self._garment_type_id = self._forced_garment_type_id

    def get_action_chunk(self, obs_dict: dict) -> list[np.ndarray]:
        state = state_dict_to_vec12(obs_dict)
        msg: dict = {
            "type": "infer_chunk",
            "session_id": "real_dagger",
            "observation.state": state.tolist(),
            "observation.images.top_rgb": _b64_image(obs_dict["right_front"]),
            "observation.images.left_rgb": _b64_image(obs_dict["left_wrist"]),
            "observation.images.right_rgb": _b64_image(obs_dict["right_wrist"]),
            "inference_config": dict(INFERENCE_CONFIG),
        }
        if self._garment_type_id is not None:
            msg["garment_type_id"] = int(self._garment_type_id)
        if self._initial_actions is not None:
            msg["initial_actions"] = self._initial_actions.tolist()

        try:
            self._ws.send(json.dumps(msg))
            resp = json.loads(self._ws.recv())
        except Exception as e:
            log.warning("Policy server connection lost (%s) — reconnecting", e)
            self._ws = None
            self._connect()
            self._ws.send(json.dumps(msg))
            resp = json.loads(self._ws.recv())
        if "error" in resp:
            raise RuntimeError(f"Policy server error: {resp['error']}")

        next_init = resp.get("next_initial_actions")
        self._initial_actions = (
            np.asarray(next_init, dtype=np.float32) if next_init is not None else None
        )
        chunk = np.asarray(resp["actions"], dtype=np.float32)

        # Garment-type bootstrap: latch the prediction from the first chunk
        # and hold position while the one-chunk delay flushes through. The
        # held pose has nothing to do with the drawn trajectory, so the
        # inpaint anchor is dropped too.
        if self._garment_type_id is None:
            gt_pred = resp.get("garment_type_pred")
            if gt_pred is not None and 0 <= int(gt_pred) < 4:
                self._garment_type_id = int(gt_pred)
                log.info(
                    "Garment-type bootstrap: latched garment_type_id=%d "
                    "(hold-position chunk emitted)", self._garment_type_id)
            else:
                log.warning(
                    "Garment-type bootstrap: garment_type_pred=%r — "
                    "skipping latch (next call retries)", gt_pred)
            self._initial_actions = None
            hold = np.broadcast_to(state[None, :], chunk.shape).copy()
            return [hold[i] for i in range(hold.shape[0])]

        return [chunk[i] for i in range(chunk.shape[0])]

    def close(self) -> None:
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Leader arm torque helpers
# ---------------------------------------------------------------------------

def _enable_leader_torque(teleop, obs: dict) -> None:
    _mirror_to_leaders(teleop, obs)
    teleop.left_arm.bus.enable_torque()
    teleop.right_arm.bus.enable_torque()
    log.info("Leader torque ENABLED")


def _disable_leader_torque(teleop) -> None:
    teleop.left_arm.bus.disable_torque()
    teleop.right_arm.bus.disable_torque()
    log.info("Leader torque DISABLED")


def _mirror_to_leaders(teleop, obs: dict) -> None:
    left = {k.removeprefix("left_").removesuffix(".pos"): v
            for k, v in obs.items()
            if k.startswith("left_") and k.endswith(".pos")}
    right = {k.removeprefix("right_").removesuffix(".pos"): v
             for k, v in obs.items()
             if k.startswith("right_") and k.endswith(".pos")}
    teleop.left_arm.bus.sync_write("Goal_Position", left)
    teleop.right_arm.bus.sync_write("Goal_Position", right)


# ---------------------------------------------------------------------------
# Safe rest ramp
# ---------------------------------------------------------------------------

def _ramp_to_safe_rest(robot, ramp_time: float = 4.0, hz: int = 30) -> None:
    cur = state_dict_to_vec12(robot.get_observation()).astype(np.float64)
    target = SAFE_REST_DEGREES.astype(np.float64)
    n = max(1, int(round(ramp_time * hz)))
    dt = 1.0 / hz
    for step in range(1, n + 1):
        interp = (cur + (step / n) * (target - cur)).astype(np.float32)
        robot.send_action(vec12_to_action_dict(interp))
        time.sleep(dt)
    robot.send_action(vec12_to_action_dict(SAFE_REST_DEGREES))
    log.info("At safe rest.")


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def _next_session_dir(root: Path, garment: str, manual_only: bool = False) -> Path:
    if manual_only:
        n = 1 + len(list(root.glob(f"{garment}_[0-9][0-9][0-9]")))
        return root / f"{garment}_{n:03d}"
    n = 1 + len(list(root.glob(f"{garment}_dagger_*")))
    return root / f"{garment}_dagger_{n:03d}"


def _create_dataset(robot, session_dir: Path, fps: int, hf_user: str,
                    manual_only: bool = False):
    teleop_proc, _, obs_proc = make_default_processors()
    obs_feats = aggregate_pipeline_dataset_features(
        pipeline=obs_proc,
        initial_features=create_initial_features(
            observation=robot.observation_features),
        use_videos=True,
    )
    act_feats = aggregate_pipeline_dataset_features(
        pipeline=teleop_proc,
        initial_features=create_initial_features(
            action=robot.action_features),
        use_videos=True,
    )
    features = combine_feature_dicts(obs_feats, act_feats)
    if not manual_only:
        features["task_is_policy"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_policy"],
        }
    dataset = LeRobotDataset.create(
        repo_id=f"{hf_user}/{session_dir.name}",
        fps=fps,
        root=session_dir,
        robot_type=robot.name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
    )
    return dataset


# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Background episode saver
# ---------------------------------------------------------------------------

class _BackgroundSaver:
    """Sequential background queue for dataset.save_episode() calls."""

    def __init__(self) -> None:
        self._queue: Queue = Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="episode-saver")
        self._thread.start()
        self._saved = 0
        self._errors: list[str] = []

    def submit(self, dataset, episode_data: dict, label: str = "") -> None:
        self._queue.put((dataset, episode_data, label))

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            dataset, episode_data, label = item
            try:
                t0 = time.perf_counter()
                dataset.save_episode(episode_data=episode_data)
                dt = time.perf_counter() - t0
                self._saved += 1
                _status(_GREEN, "OK",
                        f"Episode {label} encoded in {dt:.1f}s "
                        f"(queue: {self._queue.qsize()} pending)")
            except Exception as e:
                self._errors.append(f"{label}: {e}")
                log.error("Background save failed for %s: %s", label, e)

    def wait_all(self) -> None:
        """Block until all queued saves complete."""
        self._queue.put(None)
        self._thread.join()
        if self._errors:
            log.error("Background saver had %d errors: %s",
                      len(self._errors), self._errors)

    @property
    def pending(self) -> int:
        return self._queue.qsize()


def _freeze_position(robot, teleop, duration_s: float) -> None:
    """Hold followers + mirror leaders for ``duration_s`` (leaders stay torqued)."""
    obs = robot.get_observation()
    hold = {k: v for k, v in obs.items() if k.endswith(".pos")}
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        robot.send_action(hold)
        try:
            _mirror_to_leaders(teleop, obs)
        except Exception:
            pass
        try:
            robot.get_observation()
        except Exception:
            pass
        time.sleep(0.05)


def _reset_phase(robot, teleop, kb, timeout_s: float, fps: int) -> str:
    """Manual teleop for garment reset. Returns ``"next"`` or ``"stop"``."""
    _banner(_BG_YELLOW + _WHITE,
            "RESET ENV",
            f"Teleop active — reset the garment.  RIGHT to continue  ({timeout_s:.0f}s)")
    t_end = time.perf_counter() + timeout_s
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        ev = kb.poll()
        if ev in ("RIGHT", "SPACE"):
            _status(_GREEN, ">>", "Reset done — starting next episode")
            return "next"
        elif ev == "ESC":
            return "stop"
        try:
            action_dict = teleop.get_action()
            robot.send_action(action_dict)
        except Exception:
            pass
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, 1.0 / fps - elapsed))
    _status(_GREEN, ">>", "Reset time elapsed — starting next episode")
    return "next"


def _cleanup_temp_images(dataset_root: Path) -> None:
    """Remove leftover ``images/`` dir that lerobot leaves after video encoding."""
    images_dir = dataset_root / "images"
    if not images_dir.is_dir():
        return
    import shutil
    size_mb = sum(f.stat().st_size for f in images_dir.rglob("*") if f.is_file()) / 1e6
    n_files = sum(1 for _ in images_dir.rglob("*") if _.is_file())
    shutil.rmtree(images_dir)
    _status(_GREEN, "OK", f"Cleaned up temp images ({n_files} files, {size_mb:.0f} MB)")


_HW_ERRORS = Exception  # catch ANY error during recording → save what we have


def _record_episodes(robot, teleop, dataset, policy_client, kb, args) -> None:
    saver = _BackgroundSaver()
    pending_ep_idx = dataset.meta.total_episodes
    recorded = 0
    hw_crash = False
    manual_only = args.manual_only

    with VideoEncodingManager(dataset):
      try:
        while recorded < args.episodes:
            if manual_only:
                mode = "manual"
                _banner(_BG_MAGENTA + _WHITE, "RECORDING",
                        f"Episode {recorded + 1}/{args.episodes}  —  MANUAL")
            else:
                mode = "autonomous"
                _banner(_BG_CYAN + _WHITE, "RECORDING",
                        f"Episode {recorded + 1}/{args.episodes}  —  AUTONOMOUS")
            dataset.episode_buffer = dataset.create_episode_buffer(
                episode_index=pending_ep_idx)
            if not manual_only:
                policy_client.reset_session()
            action_queue: list[np.ndarray] = []
            frame_count = 0

            obs = robot.get_observation()
            if manual_only:
                _disable_leader_torque(teleop)
            else:
                _enable_leader_torque(teleop, obs)
            kb.drain()

            outcome = "continue"

            while outcome == "continue":
                t0 = time.perf_counter()

                ev = kb.poll()
                if ev == "ESC":
                    _banner(_BG_RED + _WHITE, "STOPPED",
                            "Session ended by operator")
                    dataset.clear_episode_buffer()
                    outcome = "stop"
                    break
                elif ev == "RIGHT":
                    if frame_count == 0:
                        _status(_YELLOW, "!!", "No frames yet — keep recording")
                    else:
                        outcome = "save"
                        break
                elif ev == "LEFT":
                    outcome = "discard"
                    break
                elif ev == "SPACE" and not manual_only:
                    new_mode = "autonomous" if mode == "manual" else "manual"
                    if new_mode == "manual":
                        _banner(_BG_MAGENTA + _WHITE, "HANDOVER",
                                f"AUTO -> MANUAL  ({args.mode_switch_delay:.0f}s — grab leaders)")
                    else:
                        _banner(_BG_BLUE + _WHITE, "HANDOVER",
                                f"MANUAL -> AUTO  ({args.mode_switch_delay:.0f}s — release leaders)")
                    if new_mode == "autonomous":
                        obs = robot.get_observation()
                        _enable_leader_torque(teleop, obs)
                    _freeze_position(robot, teleop, args.mode_switch_delay)
                    if new_mode == "manual":
                        _disable_leader_torque(teleop)
                    mode = new_mode
                    if mode == "manual":
                        _banner(_BG_MAGENTA + _WHITE, "MANUAL",
                                "Teleop active — you have control")
                    else:
                        _banner(_BG_CYAN + _WHITE, "AUTONOMOUS",
                                "Policy active — hands off")
                    action_queue.clear()
                    policy_client.reset_session()
                    kb.drain()
                    continue

                # ---- Record one frame ----
                if mode == "manual":
                    action_dict = teleop.get_action()
                    robot.send_action(action_dict)
                    obs = robot.get_observation()
                else:
                    if not action_queue:
                        obs_for_policy = robot.get_observation()
                        t_inf = time.perf_counter()
                        try:
                            chunk = policy_client.get_action_chunk(obs_for_policy)
                        except Exception as e:
                            log.error("Policy inference failed: %s", e)
                            continue
                        log.info("Got %d actions (%.0fms)",
                                 len(chunk), (time.perf_counter() - t_inf) * 1000)
                        action_queue = list(chunk)
                        t0 = time.perf_counter()

                    action_vec = action_queue.pop(0)
                    action_dict = vec12_to_action_dict(action_vec)
                    robot.send_action(action_dict)
                    obs = robot.get_observation()
                    try:
                        _mirror_to_leaders(teleop, obs)
                    except Exception:
                        pass

                obs_frame = build_dataset_frame(
                    dataset.features, obs, prefix="observation")
                act_frame = build_dataset_frame(
                    dataset.features, action_dict, prefix="action")
                frame = {
                    **obs_frame,
                    **act_frame,
                    "task": args.task,
                }
                if not manual_only:
                    frame["task_is_policy"] = np.array(
                        [1.0 if mode == "autonomous" else 0.0],
                        dtype=np.float32)
                dataset.add_frame(frame)
                frame_count += 1

                if not args.no_display:
                    log_rerun_data(observation=obs, action=action_dict)

                elapsed = time.perf_counter() - t0
                sleep_s = 1.0 / args.fps - elapsed
                if sleep_s < -0.01:
                    log.warning("Loop at %.1f Hz (target %d Hz)",
                                1.0 / max(elapsed, 1e-6), args.fps)
                time.sleep(max(0.0, sleep_s))

            if outcome == "stop":
                break

            if outcome == "save":
                label = f"{recorded + 1}/{args.episodes}"
                _banner(_BG_GREEN + _WHITE, "SAVED",
                        f"Episode {label}  —  {frame_count} frames  (encoding in background)")
                ep_data = dataset.episode_buffer
                saver.submit(dataset, ep_data, label=label)
                recorded += 1
                pending_ep_idx += 1
                _disable_leader_torque(teleop)
                if recorded < args.episodes:
                    kb.drain()
                    if _reset_phase(robot, teleop, kb,
                                    args.reset_time_s, args.fps) == "stop":
                        break

            elif outcome == "discard":
                _banner(_BG_RED + _WHITE, "DISCARDED",
                        "Episode thrown away — retrying")
                dataset.clear_episode_buffer()

      except _HW_ERRORS as exc:
        hw_crash = True
        _banner(_BG_RED + _WHITE, "HARDWARE ERROR",
                f"{type(exc).__name__}: {exc}")
        _status(_YELLOW, "!!",
                "Discarding in-progress episode, saving completed ones...")
        try:
            dataset.clear_episode_buffer()
        except Exception:
            pass

      finally:
        if saver.pending > 0:
            _status(_YELLOW, "..",
                    f"Waiting for {saver.pending} background encode(s)...")
        saver.wait_all()
        _cleanup_temp_images(dataset.root)

    if hw_crash:
        _banner(_BG_RED + _WHITE, "CRASHED",
                f"Hardware disconnected — {recorded} episodes saved before failure")
    else:
        _banner(_BG_GREEN + _WHITE, "DONE",
                f"Session complete — {recorded} episodes recorded")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _leader_ports(yaml_cfg: dict) -> tuple[str, str]:
    ports = []
    for side in ("left", "right"):
        port = yaml_cfg["arms"].get(side, {}).get("leader_port")
        if not port:
            raise SystemExit(
                f"configs/real_robot.yaml missing arms.{side}.leader_port")
        if not Path(port).exists():
            raise SystemExit(f"arms.{side}.leader_port={port} does not exist")
        ports.append(port)
    return ports[0], ports[1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path("configs/real_robot.yaml"))
    ap.add_argument("--garment", required=True, choices=VALID_GARMENTS)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--manual_only", action="store_true",
                    help="Pure teleop recording (no policy server), with "
                         "background video encoding.")
    ap.add_argument("--server", default="ws://localhost:8000",
                    help="WebSocket policy server URL — scripts/serve.py "
                         "(ignored with --manual_only)")
    ap.add_argument("--garment_type_id", type=int, default=None,
                    help="Optional fixed garment type id 0..3. Unset = latch "
                         "from the policy's first-chunk prediction.")
    ap.add_argument("--datasets_root", type=Path,
                    default=Path("data/lehome_real/recorded"))
    ap.add_argument("--task", default="Fold the Garment")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--mode_switch_delay", type=float, default=2.0,
                    help="Seconds to pause when switching modes (default: 2)")
    ap.add_argument("--reset_time_s", type=float, default=300.0,
                    help="Reset window between episodes in seconds (default: 300). "
                         "Press RIGHT to skip remaining time.")
    ap.add_argument("--no_display", action="store_true",
                    help="Disable Rerun visualization (enabled by default).")
    ap.add_argument("--hf_user",
                    default=os.environ.get("HF_USER", "local"))
    ap.add_argument("--robot_id", default="bimanual_follower")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True)

    if not args.config.exists():
        ap.error(f"Config not found: {args.config}")
    with args.config.open() as fh:
        yaml_cfg = yaml.safe_load(fh)

    # Followers are validated inside build_bi_so_follower_config; leaders here.
    ensure_default_calibration("teleoperators", "so_leader", "bimanual_leader")

    left_leader, right_leader = _leader_ports(yaml_cfg)

    args.datasets_root.mkdir(parents=True, exist_ok=True)
    session_dir = _next_session_dir(args.datasets_root, args.garment,
                                    manual_only=args.manual_only)
    if session_dir.exists():
        ap.error(f"Session dir exists: {session_dir}")

    # Patches must come before robot.connect()
    _patch_realsense()
    _hardware_reset_realsense(yaml_cfg)

    # Build configs. use_degrees=True so both arms speak the same units as the
    # organizer dataset (data/lehome_real/four_types_merged), which was recorded
    # in degrees. Mixing motor-pct and degrees on wrist_roll halves its
    # apparent range — see lerobot issue #3176.
    robot_cfg = build_bi_so_follower_config(yaml_cfg, args.robot_id, use_degrees=True)
    leader_cfg = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(port=left_leader, use_degrees=True),
        right_arm_config=SOLeaderConfig(port=right_leader, use_degrees=True),
        id="bimanual_leader",
    )

    # Connect hardware
    log.info("Connecting follower arms + cameras...")
    robot = make_robot_from_config(robot_cfg)
    robot.connect()

    log.info("Connecting leader arms...")
    teleop = make_teleoperator_from_config(leader_cfg)
    teleop.connect()

    # Dataset
    dataset = _create_dataset(robot, session_dir, args.fps, args.hf_user,
                              manual_only=args.manual_only)

    # Policy server (skip in manual-only mode)
    policy_client = None
    if not args.manual_only:
        policy_client = PolicyClient(
            args.server, garment_type_id=args.garment_type_id)

    # Keyboard
    kb = KeyboardListener()
    kb.start()

    session_name = "manual_recording" if args.manual_only else "dagger_recording"
    if not args.no_display:
        init_rerun(session_name=session_name)

    if args.manual_only:
        print(
            f"\n{'='*60}\n"
            f"  Manual recording ready — {args.garment}\n"
            f"  Pure teleop — no policy server\n"
            f"  RIGHT: save episode -> reset -> next episode\n"
            f"  LEFT: discard episode   ESC: stop\n"
            f"  Dataset: {session_dir}\n"
            f"{'='*60}\n",
            flush=True)
    else:
        print(
            f"\n{'='*60}\n"
            f"  DAgger recording ready — {args.garment}\n"
            f"  Each episode starts in AUTONOMOUS mode\n"
            f"  SPACE: switch to manual (corrections) / back to auto\n"
            f"  RIGHT: save episode -> reset -> next episode\n"
            f"  LEFT: discard episode   ESC: stop\n"
            f"  Server: {args.server}\n"
            f"  Dataset: {session_dir}\n"
            f"{'='*60}\n",
            flush=True)

    try:
        _record_episodes(robot, teleop, dataset, policy_client, kb, args)
    finally:
        log.info("Cleaning up...")
        # Suppress lerobot background thread tracebacks during disconnect
        # (RealSense _read_loop races with disconnect and prints AttributeError)
        threading.excepthook = lambda _args: None
        try:
            _disable_leader_torque(teleop)
        except Exception:
            pass
        kb.stop()
        if policy_client is not None:
            policy_client.close()
        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass

    log.info("Dataset saved: %s", session_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
