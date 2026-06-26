#!/usr/bin/env python3
"""Real-robot camera alignment overlay tool.

Picks a random frame from the real teleop dataset
(``data/lehome_real/four_types_merged``), drives the physical robot to that
12-vec state over 2 s, waits 0.5 s, then captures the three cameras from the
real robot and overlays them on top of the original dataset images. Use this
to validate that your physical camera placement matches the source dataset's
setup — if the cameras are aligned, the overlay shows ghosting only on
moving parts (the cloth), not on static features (table edges, robot bases).

The composite is written to ``outputs/real_camera_align/overlay.png`` (override
with ``--out_dir``); keep that file open in any auto-refreshing viewer (VS
Code Remote's image preview, ``feh --auto-reload``, etc.) — no GUI display
on this side, the script reads single chars from your terminal.

The overlay refreshes automatically at ~30 Hz — no manual recapture needed.

Keys (typed in the terminal where the script is running, no Enter required):

  n         pick a new random frame, ramp the robot, wait, resume auto-capture
  q / ESC   exit (ramps back to ``SAFE_REST_DEGREES`` before disconnect)

The robot stack is ``BimanualClient`` → ``BiSOFollower`` → lerobot — no
server-side code is involved. Run *in the lehome-challenge venv* (needs
lerobot + Feetech + cameras):

    source lehome-challenge/.venv/bin/activate
    python scripts/real_camera_align.py
"""

from __future__ import annotations

import argparse
import logging
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from lehome_solution.shared.real_robot_config import (  # noqa: E402
    build_bi_so_follower_config,
    state_dict_to_vec12,
    vec12_to_action_dict,
)

# These require the lehome-challenge venv.
from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower  # noqa: E402
from lerobot.cameras.realsense import camera_realsense as _rs_cam_mod  # noqa: E402

log = logging.getLogger("real_camera_align")


# ---------------------------------------------------------------------------
# Robot client + RealSense workarounds
# ---------------------------------------------------------------------------
# Two D435 quirks lerobot doesn't handle:
#   1) read() timeout_ms=200 — too tight after a cold open / hardware_reset.
#      Bumped to 2000 ms.
#   2) connect() raises if the first warmup read doesn't get a frame — but the
#      D435 sometimes needs a hardware_reset() to start streaming. Wrap connect
#      so a failed first attempt triggers one reset + retry.
def _patched_rs_read(self, color_mode=None, timeout_ms: int = 2000):
    return _rs_cam_mod.RealSenseCamera._orig_read(
        self, color_mode=color_mode, timeout_ms=timeout_ms)


def _patched_rs_connect(self, warmup: bool = True):
    try:
        return _rs_cam_mod.RealSenseCamera._orig_connect(self, warmup=warmup)
    except (RuntimeError, ConnectionError) as e:
        # Clean up any half-started pipeline (mirrors lerobot's own teardown).
        self.rs_pipeline = None
        self.rs_profile = None
        import pyrealsense2 as rs  # noqa: PLC0415
        target_serial = str(getattr(self, "serial_number", "") or "")
        try:
            ctx = rs.context()
            for dev in ctx.devices:
                if target_serial and dev.get_info(rs.camera_info.serial_number) != target_serial:
                    continue
                log.warning("RealSense connect failed (%s); issuing hardware_reset() and retrying", e)
                dev.hardware_reset()
                break
        except Exception:
            pass
        time.sleep(3.0)
        return _rs_cam_mod.RealSenseCamera._orig_connect(self, warmup=warmup)


if not hasattr(_rs_cam_mod.RealSenseCamera, "_orig_read"):
    _rs_cam_mod.RealSenseCamera._orig_read = _rs_cam_mod.RealSenseCamera.read
    _rs_cam_mod.RealSenseCamera.read = _patched_rs_read
if not hasattr(_rs_cam_mod.RealSenseCamera, "_orig_connect"):
    _rs_cam_mod.RealSenseCamera._orig_connect = _rs_cam_mod.RealSenseCamera.connect
    _rs_cam_mod.RealSenseCamera.connect = _patched_rs_connect


# Fixed "resting" pose to ramp the arms to before disconnect — the natural
# gravity-rest pose with torque disabled. Both elbow_flex sit at the +100
# range_max because the arm physically rests against its hardware stop;
# commanding +100 just keeps it there.
SAFE_REST_DEGREES = np.array([
    # LEFT [0:6]:  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    2.24, -97.58, 100.0, 28.47, -2.27, 2.35,
    # RIGHT [6:12]
    -1.16, -96.10, 99.55, 35.15, 1.73, 0.68,
], dtype=np.float32)


def _hardware_reset_realsense(yaml_cfg: dict) -> None:
    """Issue ``hardware_reset()`` on the configured RealSense and sleep ~3s.

    Does NOT open a pipeline (that caused lerobot's subsequent open to read
    no frames). Just resets the device so lerobot's fresh ``connect()``
    starts from a known-good state — and crucially, runs BEFORE any arm
    serial bus opens, so the right arm doesn't sit idle while the reset
    happens.
    """
    cams = yaml_cfg.get("cameras", {})
    top = cams.get("top")
    if not top or (top.get("backend") or "").lower() != "pyrealsense2":
        return
    serial = str(top.get("serial") or "")
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError:
        return
    try:
        ctx = rs.context()
        for dev in ctx.devices:
            if serial and dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            log.info("hardware_reset() RealSense %s (3s settle) ...",
                     dev.get_info(rs.camera_info.serial_number))
            dev.hardware_reset()
            time.sleep(3.0)
            return
        log.warning("RealSense %s not found; skipping pre-reset.", serial or "(any)")
    except Exception as e:
        log.warning("hardware_reset() raised: %s — continuing", e)


class BimanualClient:
    """Minimal wrapper around ``BiSOFollower`` for sequential joint control
    in degree-mode units (use_degrees=True; gripper 0-100).

    Uses lerobot directly so the wire format matches the eval / data-collection
    stack byte-for-byte.
    """

    def __init__(self, yaml_path: Path, robot_id: str = "bimanual_follower",
                 skip_cameras: bool = False, use_degrees: bool = True):
        with yaml_path.open() as fh:
            yaml_cfg = yaml.safe_load(fh)
        self._yaml_cfg = yaml_cfg
        self._skip_cameras = skip_cameras
        self._cfg = build_bi_so_follower_config(
            yaml_cfg, robot_id,
            skip_cameras=skip_cameras, use_degrees=use_degrees,
        )
        self.robot = BiSOFollower(self._cfg)

    def __enter__(self) -> "BimanualClient":
        # Always hardware_reset() the RealSense BEFORE any arm bus opens.
        # The reset alone takes ~3s; opening it inline (during right-arm
        # connect) leaves the right arm's serial bus idle long enough that
        # Feetech motors stop responding to writes. Reset first → motors only
        # see a fast camera open path → enable_torque write succeeds.
        if not self._skip_cameras:
            _hardware_reset_realsense(self._yaml_cfg)
        self.robot.connect(calibrate=True)
        return self

    def __exit__(self, *_exc) -> None:
        self.robot.disconnect()

    def state(self) -> np.ndarray:
        """Read current 12-vec joint state in degree-mode units."""
        return state_dict_to_vec12(self.robot.get_observation())

    def action(self, vec12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Send one 12-vec degree-mode target. Returns ``(state_after, sent_after_clip)``."""
        sent_dict = self.robot.send_action(vec12_to_action_dict(vec12))
        sent_vec = np.asarray(
            [float(sent_dict[k]) for k in vec12_to_action_dict(vec12).keys()],
            dtype=np.float32,
        )
        state_after = self.state()
        return state_after, sent_vec

    def ramp_to(self, target_12: np.ndarray, ramp_time: float, hz: int = 30) -> None:
        """Linearly interpolate from current state to ``target_12`` over
        ``ramp_time`` seconds at ``hz`` command rate."""
        if target_12.shape != (12,):
            raise ValueError(f"target must be (12,), got {target_12.shape}")
        n_steps = max(1, int(round(ramp_time * hz)))
        dt = 1.0 / hz
        start = self.state().astype(np.float64)
        delta = target_12.astype(np.float64) - start
        log.info(
            "Ramping to target over %.2fs (%d steps); max |Δ|=%.1f deg",
            ramp_time, n_steps, float(np.abs(delta).max()),
        )
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            interp = (start + alpha * delta).astype(np.float32)
            self.action(interp)
            time.sleep(dt)
        # Land one extra command at the exact target so the final pose holds.
        self.action(target_12.astype(np.float32))

REAL_DATA_ROOT_DEFAULT = REPO_ROOT / "data" / "lehome_real" / "four_types_merged"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "real_camera_align"


def _setup_cbreak() -> tuple[int, list] | tuple[None, None]:
    """Put stdin into cbreak mode (single-char reads, no Enter). Returns
    ``(fd, old_termios)`` so the caller can restore on exit, or ``(None, None)``
    when stdin isn't a TTY (e.g. piped — :func:`_poll_key` then no-ops)."""
    if not sys.stdin.isatty():
        return None, None
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old


def _restore_terminal(fd: int | None, old) -> None:
    if fd is None or old is None:
        return
    import termios
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _poll_key() -> str:
    """Non-blocking single-char read. Returns '' if no key is queued.

    Assumes :func:`_setup_cbreak` has already been called. Lowercases ASCII
    letters so callers can match on `'q'` / `'n'` directly.
    """
    if not sys.stdin.isatty():
        return ""
    import select
    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not rlist:
        return ""
    ch = sys.stdin.read(1)
    return ch.lower() if ch.isalpha() else ch
REAL_FPS = 20.0  # source teleop rate

# Cameras as they appear in BiSOFollower.get_observation() (BiSOFollower adds
# the ``left_`` / ``right_`` prefix; the per-arm sub-keys are configured via
# build_bi_so_follower_config). Same names show up under
# ``observation.images.*`` in the LeRobot dataset.
CAMERA_KEYS = ("right_front", "left_wrist", "right_wrist")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_states_table(ds_root: Path) -> pd.DataFrame:
    """Load (episode_index, frame_index, observation.state) for all frames in the BC parquet."""
    pf = sorted((ds_root / "data").rglob("*.parquet"))[0]
    return pq.read_table(
        str(pf), columns=["episode_index", "frame_index", "observation.state"]
    ).to_pandas()


def load_episode_meta(ds_root: Path) -> pd.DataFrame:
    """Load video chunk/file/timestamp metadata, one row per episode."""
    ep_meta_pf = ds_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    return pq.read_table(str(ep_meta_pf)).to_pandas()


def decode_frame_images(
    ds_root: Path,
    episode_meta: pd.DataFrame,
    ep: int,
    fr: int,
) -> dict[str, np.ndarray]:
    """Extract a single RGB frame from each of the 3 cameras at (ep, fr).

    Real teleop ran at 20 Hz, so frame ``fr`` within the episode is at offset
    ``fr / REAL_FPS`` seconds from the per-episode ``from_timestamp`` in the
    mp4. ffmpeg seeks accurately with the ``-ss`` flag *after* ``-i``.
    """
    row = episode_meta[episode_meta["episode_index"] == ep].iloc[0]
    images: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        cam_dataset_key = f"observation.images.{key}"
        ck = int(row[f"videos/{cam_dataset_key}/chunk_index"])
        fi = int(row[f"videos/{cam_dataset_key}/file_index"])
        from_ts = float(row[f"videos/{cam_dataset_key}/from_timestamp"])
        mp4 = ds_root / "videos" / cam_dataset_key / f"chunk-{ck:03d}" / f"file-{fi:03d}.mp4"
        target_ts = from_ts + fr / REAL_FPS
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{target_ts:.6f}",
                "-i", str(mp4),
                "-frames:v", "1",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            img_bgr = cv2.imread(str(tmp_path))
            if img_bgr is None:
                raise RuntimeError(f"failed to decode frame for {key} at ts={target_ts:.3f}")
            images[key] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        finally:
            tmp_path.unlink(missing_ok=True)
    return images


def pick_random_frame(
    states_df: pd.DataFrame,
    ds_root: Path,
    episode_meta: pd.DataFrame,
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, np.ndarray], int, int]:
    """Pick a (state_motor, images, ep, fr) at random from the dataset."""
    idx = rng.randint(0, len(states_df) - 1)
    state = np.asarray(states_df["observation.state"].iloc[idx], dtype=np.float32)
    ep = int(states_df["episode_index"].iloc[idx])
    fr = int(states_df["frame_index"].iloc[idx])
    images = decode_frame_images(ds_root, episode_meta, ep, fr)
    return state, images, ep, fr


# ---------------------------------------------------------------------------
# Robot camera capture
# ---------------------------------------------------------------------------

def _report_calibration(client: BimanualClient, *, force_write: bool) -> None:
    """Log the calibration file paths + per-motor diff between JSON and motor registers.

    This proves which JSON is actually being used and surfaces any drift between
    the file and the motors' onboard values. If ``force_write`` is set, pushes
    the file values to the motors after the diff print regardless of lerobot's
    own ``is_calibrated`` check.
    """
    for side, arm in (("left", client.robot.left_arm), ("right", client.robot.right_arm)):
        fpath = arm.calibration_fpath
        if not fpath.is_file():
            log.warning("[calibration] %s: file MISSING at %s", side, fpath)
            continue
        log.info("[calibration] %s file: %s", side, fpath)
        try:
            motor_cal = arm.bus.read_calibration()
        except Exception as e:
            log.warning("[calibration] %s read_calibration() failed: %s", side, e)
            motor_cal = None
        if motor_cal is None:
            continue
        any_diff = False
        for name, file_cal in arm.calibration.items():
            mcal = motor_cal.get(name)
            if mcal is None:
                log.warning("[calibration] %s.%s: present in file, missing on motor", side, name)
                any_diff = True
                continue
            d_min = file_cal.range_min - mcal.range_min
            d_max = file_cal.range_max - mcal.range_max
            d_off = file_cal.homing_offset - mcal.homing_offset
            if d_min or d_max or d_off:
                any_diff = True
                log.warning(
                    "[calibration] %s.%-13s DIFFERS: file=(min=%d max=%d off=%+d)  motor=(min=%d max=%d off=%+d)",
                    side, name,
                    file_cal.range_min, file_cal.range_max, file_cal.homing_offset,
                    mcal.range_min, mcal.range_max, mcal.homing_offset,
                )
        if not any_diff:
            log.info("[calibration] %s: file values exactly match motor registers ✓", side)
        if force_write:
            try:
                arm.bus.write_calibration(arm.calibration)
                log.info("[calibration] %s: --force_calibration → wrote JSON values to motors", side)
            except Exception as e:
                log.error("[calibration] %s write_calibration() failed: %s", side, e)


def capture_images_from_robot(client: BimanualClient) -> dict[str, np.ndarray]:
    """One frame from each of the 3 cameras via BiSOFollower.get_observation().

    lerobot's OpenCV/RealSense camera defaults to ColorMode.RGB, matching the
    dataset's storage convention, so no channel swap is needed.
    """
    obs = client.robot.get_observation()
    images: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        if key not in obs:
            raise KeyError(f"camera obs key missing from robot: {key} (have: {sorted(obs)})")
        img = np.asarray(obs[key])
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        images[key] = img.copy()
    return images


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def compose_overlay(
    orig: dict[str, np.ndarray],
    captured: dict[str, np.ndarray],
    *,
    alpha: float,
    panel_h: int = 480,
) -> np.ndarray:
    """3-panel RGB composite suitable for matplotlib ``imshow``.

    Each panel is the per-pixel alpha-blend of the original dataset frame
    (weight ``alpha``) and the live-robot capture (weight ``1 - alpha``),
    resized to a common display height while preserving aspect ratio so we
    can ``hstack`` them.
    """
    panels: list[np.ndarray] = []
    for key in CAMERA_KEYS:
        o = orig.get(key)
        c = captured.get(key)
        if o is None or c is None:
            panels.append(np.zeros((panel_h, int(panel_h * 4 / 3), 3), dtype=np.uint8))
            continue
        if c.shape[:2] != o.shape[:2]:
            c = cv2.resize(c, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA)
        blended = cv2.addWeighted(o, alpha, c, 1.0 - alpha, 0.0)
        h, w = blended.shape[:2]
        if h != panel_h:
            new_w = int(round(w * panel_h / h))
            blended = cv2.resize(blended, (new_w, panel_h), interpolation=cv2.INTER_AREA)
        panels.append(blended)
    return np.concatenate(panels, axis=1)


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--real_data_root", type=Path, default=REAL_DATA_ROOT_DEFAULT,
        help=f"Real BC dataset root (default: {REAL_DATA_ROOT_DEFAULT}).",
    )
    ap.add_argument(
        "--config", type=Path, default=Path("configs/real_robot.yaml"),
        help="Hardware mapping yaml.",
    )
    ap.add_argument("--robot_id", default="bimanual_follower")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional RNG seed for reproducible frame picks.")
    ap.add_argument("--ramp_time", type=float, default=2.0,
                    help="Seconds to interpolate from current pose to the target (default 2.0).")
    ap.add_argument("--settle_time", type=float, default=0.5,
                    help="Seconds to wait after the ramp before capturing cameras (default 0.5).")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Original-image opacity in the overlay (1.0 = original only, 0.0 = "
                         "robot capture only). Default 0.5.")
    ap.add_argument("--refresh_hz", type=float, default=30.0,
                    help="Auto-recapture rate while idle (Hz). Default 30.")
    ap.add_argument("--force_calibration", action="store_true",
                    help="Unconditionally write the JSON calibration values to the "
                         "motors after connect (so the JSON is guaranteed source of "
                         "truth, even if lerobot's mismatch check passed).")
    ap.add_argument(
        "--out_dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Where to write the live overlay PNG (default: {DEFAULT_OUT_DIR}).",
    )
    ap.add_argument("--dry_run", action="store_true",
                    help="Compute everything but do not touch the robot — useful for "
                         "smoke-testing the image decoding off-bench.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    rng = random.Random(args.seed)

    log.info("Loading dataset metadata from %s ...", args.real_data_root)
    states_df = load_states_table(args.real_data_root)
    episode_meta = load_episode_meta(args.real_data_root)
    n_eps = int(states_df["episode_index"].nunique())
    log.info("Dataset has %d frames across %d episodes.", len(states_df), n_eps)

    if args.dry_run:
        log.warning("--dry_run set: skipping robot connect; just decoding one frame.")
        state, images, ep, fr = pick_random_frame(states_df, args.real_data_root, episode_meta, rng)
        log.info("Dry-run frame: ep=%d fr=%d state_max=%.2f", ep, fr, float(np.abs(state).max()))
        return 0

    out_path = (args.out_dir / "overlay.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # use_degrees=True: four_types_merged is recorded in degrees, so the
    # follower must speak the same units when we ramp_to() those states.
    with BimanualClient(args.config, args.robot_id, use_degrees=True) as client:
        _report_calibration(client, force_write=args.force_calibration)
        log.info("Robot connected. Picking initial frame ...")
        state, original_images, ep, fr = pick_random_frame(
            states_df, args.real_data_root, episode_meta, rng,
        )
        log.info("Initial frame ep=%d fr=%d — ramping over %.2fs ...", ep, fr, args.ramp_time)
        client.ramp_to(state, args.ramp_time)
        log.info("Settling for %.2fs ...", args.settle_time)
        time.sleep(args.settle_time)
        captured = capture_images_from_robot(client)

        def _write_overlay() -> None:
            composite_rgb = compose_overlay(
                original_images, captured, alpha=args.alpha,
            )
            cv2.imwrite(
                str(out_path), cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR),
            )

        _write_overlay()
        log.info(
            "Initial overlay written. ep=%d fr=%d. Auto-refresh @ %.0fHz.",
            ep, fr, args.refresh_hz,
        )
        print(
            f"\n>>> Open this file in any auto-refreshing viewer (VS Code "
            f"Remote, feh --auto-reload, etc.):\n        {out_path}\n"
            f">>> Keys: n=next random frame   q/ESC=quit (no Enter needed)\n",
            flush=True,
        )

        period = 1.0 / max(1e-3, args.refresh_hz)
        fd, old_termios = _setup_cbreak()
        frame_counter = 0
        last_log = time.perf_counter()
        try:
            while True:
                t0 = time.perf_counter()
                # Auto-recapture each iteration; the loop rate-limits below.
                captured = capture_images_from_robot(client)
                _write_overlay()
                frame_counter += 1
                # Heartbeat every ~3s so the operator sees the loop is alive
                # without spamming 30 lines/s.
                if t0 - last_log >= 3.0:
                    fps = frame_counter / (t0 - last_log)
                    log.info(
                        "live: ep=%d fr=%d  refresh ~%.1fHz  → %s",
                        ep, fr, fps, out_path.name,
                    )
                    frame_counter = 0
                    last_log = t0

                key = _poll_key()
                if key in ("q", "\x1b", "\x03"):  # q, ESC, Ctrl-C
                    log.info("Quit key pressed.")
                    break
                if key == "n":
                    # Restore terminal during ramp so the user can Ctrl-C cleanly.
                    _restore_terminal(fd, old_termios)
                    fd, old_termios = None, None
                    state, original_images, ep, fr = pick_random_frame(
                        states_df, args.real_data_root, episode_meta, rng,
                    )
                    log.info(
                        "New frame ep=%d fr=%d — ramping over %.2fs ...",
                        ep, fr, args.ramp_time,
                    )
                    client.ramp_to(state, args.ramp_time)
                    time.sleep(args.settle_time)
                    fd, old_termios = _setup_cbreak()
                elif key and key not in ("\n", "\r"):
                    log.debug("unbound key: %r", key)

                # Sleep the remainder of the target period.
                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            _restore_terminal(fd, old_termios)
            log.info("Ramping back to SAFE_REST over 4.0s ...")
            try:
                client.ramp_to(SAFE_REST_DEGREES.astype(np.float32), 4.0)
            except Exception as e:
                log.warning("ramp-to-rest failed: %s", e)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
