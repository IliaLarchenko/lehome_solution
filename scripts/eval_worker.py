#!/usr/bin/env python3
"""Workers that run eval episodes against a shared policy server.

Architecture
============
Isaac Sim subprocess  ←→  Worker proxy (this file)  ←→  Policy server (serve.py)
                                                     ←→  Value server  (serve.py)

The sim runs the generic ``remote`` policy (RemotePolicy, an HTTP client) that
only serialises observations and POSTs them, plus per-episode commands, to the
worker's per-worker HTTP proxy server. The proxy thread (running inside the
worker thread):
  - serves the sim's commands over HTTP (/next_task, /reset, /infer, /snapshot,
    /update_check_status), driving the async gateway via a request queue
  - holds a persistent WebSocket connection to the shared policy server
  - accumulates trajectory frames per episode
  - writes temp pkl files (same format/naming as before) on every episode boundary
    and on sim disconnect (last episode)

The orchestrator's background dataset thread picks those pkls up via
add_episode_from_temp as before — nothing changes downstream.

run_worker_session(): Primary eval worker.
"""

import base64
import json
import logging
import os
import pickle
from collections import Counter
import queue
import re
import select
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from lehome_solution.eval import (
    ensure_isaacsim_env,
    garment_name_to_type,
)
from lehome_solution.eval.dataset_writer import sanitize_basename
from lehome_solution.utils.logging_config import log_path as _log_path
from lehome_solution.utils import logging_config as logcfg
from lehome_solution.constants import (
    FPS as _FPS,
    STUCK_WINDOW as _STUCK_WINDOW,
    STUCK_THRESHOLD as _STUCK_THRESHOLD,
    STUCK_KEEP as _STUCK_KEEP,
    EMA_ALPHA as _EMA_ALPHA,
    GARMENT_TYPE_TO_ID,
)

# Bypass marker for fast-warmup: setting policy_call_count >= this value
# skips the chunk-shrinking branch (mirrors _FAST_WARMUP_CALLS in the loop).
_FAST_WARMUP_CALLS_BYPASS = 5

REPO_ROOT = Path(__file__).parent.parent
LEHOME_CHALLENGE_DIR = REPO_ROOT / "lehome-challenge"
LEHOME_VENV_PYTHON = LEHOME_CHALLENGE_DIR / ".venv" / "bin" / "python"

ensure_isaacsim_env(REPO_ROOT)

NO_OUTPUT_TIMEOUT = 300
SIM_INIT_TIMEOUT = 600  # Simulation init (scene creation + physics) can be slow with concurrent instances
ABORT_LOOP_THRESHOLD = 20
GARMENT_LIST_WAIT_TIMEOUT = 180

# Proxy server: each worker W listens on localhost:PROXY_BASE_PORT+W
PROXY_BASE_PORT = 9000



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kill_pg(pid: int):
    """Kill entire process group and wait for it to die."""
    try:
        os.killpg(os.getpgid(pid), 9)
    except (ProcessLookupError, PermissionError):
        return
    # Wait briefly for process to exit to avoid zombies and port reuse races
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def _signal_server(server_url: str, msg: dict, timeout: float = 10, recv_timeout: float = 30) -> dict | None:
    """Send a one-shot control message to the policy server and return the response."""
    try:
        from websockets.sync.client import connect as ws_connect
        ws = ws_connect(server_url, open_timeout=timeout)
        ws.send(json.dumps(msg))
        resp = json.loads(ws.recv(timeout=recv_timeout))
        ws.close()
        return resp
    except Exception as e:
        logger.warning(f"Failed to signal server: {e}")
        return None


# sanitize_basename imported from dataset_writer


# Global cap on concurrent _flush_episode calls across all worker proxies.
# Each in-flight flush pins one full trajectory snapshot in RAM (~1.7 GB at
# 640x480x3 / 600 frames).  Without a cap, all 5 workers can be mid-flush
# simultaneously, accumulating ~8.5 GB of dead trajectory data on top of the
# 5 workers' live in-flight trajectories — which is what OOM-killed the
# orchestrator at iteration 19000 (40 GB anon-rss).
#
# 2 was chosen to bound the worst case to ~3.4 GB of flushing snapshots while
# still parallelizing pickle.dump across two workers (which is fast enough
# that the cap rarely binds in steady state).  The semaphore is acquired in
# the proxy gateway loop BEFORE submitting to the executor, so producers
# block instead of queuing more snapshots.  Released by `_flush_episode_release`
# when the executor task completes.
_MAX_CONCURRENT_FLUSHES = 2
_FLUSH_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_FLUSHES)


def _decode_obs_images(msg: dict, image_blob: bytes | None = None) -> dict:
    """Decode image fields in an obs message dict, return decoded dict.

    Two transport formats are supported:
    - Binary framing (preferred): image fields are dicts with
      {binary: True, shape, dtype, offset, size}, and the raw image bytes
      are passed in via `image_blob` (the trailing chunk of a binary ws frame).
      We `.copy()` each view so the underlying blob can be GC'd at end of step.
    - Legacy base64 (kept for back-compat with thin clients still sending JSON
      text frames): image fields are dicts with {base64, shape, dtype}.
    """
    obs = {}
    for k, v in msg.items():
        if isinstance(v, dict) and v.get("binary"):
            dtype = np.dtype(v["dtype"])
            count = v["size"] // dtype.itemsize
            arr = np.frombuffer(
                image_blob, dtype=dtype, count=count, offset=v["offset"]
            ).reshape(v["shape"])
            # Copy so the (per-step) image_blob bytes object can be released
            # instead of being pinned for the entire episode.
            obs[k] = arr.copy()
        elif isinstance(v, dict) and "base64" in v:
            raw = base64.b64decode(v["base64"])
            arr = np.frombuffer(raw, dtype=np.dtype(v["dtype"])).reshape(v["shape"])
            obs[k] = arr
        elif isinstance(v, list):
            obs[k] = np.array(v)
        else:
            obs[k] = v
    return obs




def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """Ensure an image array is uint8 in [0, 255].

    The policy's resize_with_pad (JAX) expects either uint8 [0,255] or float32 [-1,1].
    Isaac Sim cameras may return float32 at non-standard resolutions.  This function
    normalises any format to uint8 so the policy always gets the dtype it was trained on.
    """
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        vmax = float(img.max())
        if vmax > 1.5:
            # float32 in [0, 255] range — direct cast
            return img.clip(0, 255).astype(np.uint8)
        else:
            # float32 in [0, 1] range — scale up
            return (img.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    # Fallback for other integer dtypes (e.g. int32)
    return img.clip(0, 255).astype(np.uint8)


def _encode_obs_for_policy(
    decoded_msg: dict,
    session_id: str,
    garment_type_input_id: int | None = None,
) -> dict:
    """Re-encode an already-decoded obs dict into the format the policy server expects.
    Images are normalised to uint8 [0,255] to match the training data dtype."""
    msg: dict = {"type": "infer_chunk", "session_id": session_id}
    if garment_type_input_id is not None:
        msg["garment_type_id"] = garment_type_input_id
    for k, v in decoded_msg.items():
        if k in ("type", "session_id"):
            continue
        if isinstance(v, np.ndarray):
            if v.ndim == 3:
                img = _to_uint8_image(v)
                msg[k] = {
                    "base64": base64.b64encode(img.tobytes()).decode("ascii"),
                    "shape": list(img.shape),
                    "dtype": str(img.dtype),
                }
            else:
                msg[k] = v.tolist()
        else:
            msg[k] = v
    return msg


# ---------------------------------------------------------------------------
# Proxy server (runs in a background thread inside each worker thread)
# ---------------------------------------------------------------------------

class _ProxyServer:
    """Per-worker WebSocket gateway between Isaac Sim (thin client) and policy/value servers.

    Lifecycle:
        server = _ProxyServer(...)
        server.start()           # starts listening thread
        ...sim runs...
        server.join()            # blocks until sim disconnects and last episode is flushed
        pkls = server.pkl_paths  # list of pkl Paths written (one per episode)
    """

    FPS = _FPS

    # Stuck detection: 4-second rolling window, trim keeping 1s of stuck state
    STUCK_WINDOW = _STUCK_WINDOW
    STUCK_THRESHOLD = _STUCK_THRESHOLD

    # Crash detection: episodes with fewer frames than this are discarded and re-queued.
    # This catches Isaac Sim crashes that produce 1-frame garbage episodes.
    MIN_EPISODE_FRAMES = 10
    STUCK_KEEP = _STUCK_KEEP

    # EMA on P(success): α matches value function (dataset_writer.py).
    # Used for failure drop detection AND near-success detection.
    EMA_ALPHA = _EMA_ALPHA    # EMA smoothing on chunk-level values
    EMA_SKIP_CHUNKS = 3       # skip first ~3 chunks (~2 seconds) to avoid initial spike
    EMA_DROP_THRESH = 0.12    # EMA must drop this much from running max
    EMA_PEAK_THRESH = 0.25    # running max must exceed this before drop counts

    # Early snapshot: capture state at step 5 for adaptive success state saving
    EARLY_SNAPSHOT_FRAME = 5

    def __init__(
        self,
        worker_id: int,
        session_id: str,
        server_url: str,
        video_dir: str | None,
        label: str,
        task_queue: "queue.Queue",
        first_task: dict,
        max_steps: int = 600,
        prior_file: str | None = None,
        noise_temperature: float = 1.0,
        explore_noise_prob: float = 0.0,
        explore_noise_scale: float = 0.0,
        episode_done_queue: "queue.Queue | None" = None,
        success_rates_file: str | None = None,
        default_rollout_type: str | None = None,
        storage_width: int | None = None,
        storage_height: int | None = None,
        per_garment_type_inference_config: dict | None = None,
        save_pkl: bool = True,
    ):
        self.worker_id = worker_id
        self.session_id = session_id
        self.server_url = server_url
        self.video_dir = video_dir
        self.label = label
        self.max_steps = max_steps
        # False = metrics-only: skip pkl/physics-state writes, still report success.
        self.save_pkl = save_pkl
        self.port = PROXY_BASE_PORT + worker_id
        self.prior_file = prior_file
        self.noise_temperature = noise_temperature
        # DART-style per-chunk additive correlated noise. Both must be > 0
        # for any noise to fire. Default 0.0 → strictly bit-identical behavior.
        # Submission path NEVER constructs an EvalWorker; this is rollout-only.
        self.explore_noise_prob = float(explore_noise_prob)
        self.explore_noise_scale = float(explore_noise_scale)
        self.default_rollout_type = default_rollout_type or "normal"
        # Trajectory images get resized to (storage_height, storage_width) before
        # being pinned in the per-episode trajectory list. The policy server always
        # sees the original camera resolution.
        # None disables resize (stores at native camera resolution).
        self.storage_width = storage_width
        self.storage_height = storage_height
        # Fixed per-garment-type inference config overrides.  When non-empty,
        # eval_worker uses these instead of Thompson Sampling — each episode's
        # inference_config is built from the garment_type's entry (filling
        # missing fields from inference_optimization.DEFAULT_CONFIG).
        self.per_garment_type_inference_config = per_garment_type_inference_config or None

        # Shared task queue (thread-safe).  Each item is a dict:
        #   {garment_name, garment_type, seed, ep_idx, restore_data}
        # first_task is already popped from the queue and used for Isaac Sim startup.
        # ep_idx is pre-assigned by the orchestrator for global uniqueness.
        self._task_queue = task_queue
        self._first_task = first_task
        self._first_task_consumed = False
        # Local retry queue: failed replay/hard_mining tasks are re-enqueued here
        # (same worker) until success or max_attempts reached. Takes priority over
        # the shared task queue so the retry runs while the garment is still loaded.
        from collections import deque as _deque
        self._retry_queue: "_deque[dict]" = _deque()

        # Current garment info (updated on each task switch)
        self.garment_name = first_task["garment_name"]
        self.garment_type = first_task["garment_type"]
        self.base_seed = first_task["seed"]

        # Optional queue for notifying the orchestrator as soon as each episode
        # PKL is written (streaming dataset writes instead of batch-at-end).
        self._episode_done_queue = episode_done_queue

        self.pkl_paths: list[Path] = []
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._nan_detected = False  # set when Isaac Sim returns NaN state (physics explosion)
        self._crash_detected = False  # set when Isaac Sim crashes mid-episode (WebSocket drops with incomplete episode)
        self._episode_inference_configs: dict[int, dict] = {}  # ep_idx -> config
        # Hard mining: track which NPZ paths have already yielded a success
        # so we only save the first success per failure state.
        self._solved_npz_paths: set[str] = set()
        # Success rates for adaptive success state saving (loaded from success_rates.json).
        # {by_type: {type: sr}, by_garment: {name: sr}}
        self._success_rates: dict = {}
        if success_rates_file:
            try:
                with open(success_rates_file) as _f:
                    self._success_rates = json.load(_f)
                logger.info(
                    f"{label} loaded success rates: "
                    f"{', '.join(f'{k}={v:.1%}' for k, v in sorted(self._success_rates.get('by_type', {}).items()))}"
                )
            except Exception as e:
                logger.warning(f"{label} failed to load success rates from {success_rates_file}: {e}")
        # Signaled from inside the asyncio loop once the WebSocket server has bound the port.
        # Workers must wait_ready() before starting the sim so the thin client never gets ECONNREFUSED.
        self._ready_event = threading.Event()

        # Transient full-resolution step-0 image cache for the frame-1 stale-
        # camera check. Populated on frame 0, cleared after frame 1 runs.
        self._stale_cam_step0: dict[str, np.ndarray] = {}

        # Sim-facing transport: the sim (RemotePolicy) is an HTTP client that
        # POSTs commands here; do_POST hands each request to the async gateway
        # via this queue and blocks on a per-request Event for the response.
        # A None sentinel (pushed by join()) ends the gateway loop.
        self._sim_req_queue: "queue.Queue" = queue.Queue()
        self._http = None

    def _resize_for_storage(self, img: np.ndarray) -> np.ndarray:
        """Resize a decoded camera image to (storage_height, storage_width) so
        the trajectory/pkl don't carry the full 640×480. Returns a contiguous
        copy (so the slice into `decoded` can be released after the policy call).
        No-op when storage dims are not set or already match."""
        if self.storage_width is None or self.storage_height is None:
            return img.copy()
        ih, iw = img.shape[:2]
        if ih == self.storage_height and iw == self.storage_width:
            return img.copy()
        resized = cv2.resize(img, (self.storage_width, self.storage_height),
                             interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized)

    # -- public --

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=False, name=f"proxy-w{self.worker_id}"
        )
        self._thread.start()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Block until the proxy WebSocket server has bound its port (or timeout)."""
        return self._ready_event.wait(timeout=timeout)

    def join(self):
        """Block until the proxy thread has finished all flushes and exited.
        No timeout: we must not leave the proxy alive holding the port."""
        if self._thread:
            self._sim_req_queue.put(None)  # sentinel: end the gateway loop
            self._thread.join()

    def _get_next_task(self) -> dict | None:
        """Pop the next episode, preferring local retries over the shared queue.

        Retry tasks (re-enqueued by _flush_episode on replay/hard_mining failure)
        take priority so the retry runs while the same garment is still loaded
        in Isaac Sim — avoiding the ~5s garment-switch overhead. When the retry
        buffer is empty, pull the next task from the shared queue.

        Returns dict with garment_name, garment_type, seed, ep_idx, restore_data
        or None if both queues are drained.
        """
        import queue as _q
        # Local retries jump the line: same garment already loaded, no switch cost.
        if self._retry_queue:
            return self._retry_queue.popleft()
        if not self._first_task_consumed:
            self._first_task_consumed = True
            return self._first_task
        try:
            return self._task_queue.get_nowait()
        except _q.Empty:
            return None

    def _enqueue_retry(
        self,
        garment_name: str, garment_type: str, seed: int, ep_idx: int,
        restore_data: dict, attempt: int,
    ):
        """Re-enqueue a failed replay/hard_mining task for another attempt.

        The next attempt runs with the same garment already loaded in Isaac Sim
        (no ~5s switch cost). attempt is the 0-indexed number we're about to
        try (so the caller passes attempt_that_failed + 1).
        """
        retry_restore = dict(restore_data)
        retry_restore["_attempt"] = attempt
        retry_task = {
            "garment_name": garment_name,
            "garment_type": garment_type,
            "seed": seed,
            "ep_idx": ep_idx,
            "restore_data": retry_restore,
        }
        self._retry_queue.append(retry_task)
        logger.info(
            f"{self.label} ep{ep_idx}: queued retry attempt {attempt + 1}"
            f"/{int(retry_restore.get('_max_attempts', 1))} for {garment_name}"
        )

    def _emit_phantom_discards(
        self,
        garment_name: str, garment_type: str, seed: int, ep_idx: int,
        rollout_type: str, n: int,
    ):
        """Emit n 'early-stop' discard events so total_expected accounting
        (upper-bounded at max_attempts per task) converges even when a task
        succeeds before exhausting its retry budget.
        """
        if n <= 0 or self._episode_done_queue is None:
            return
        for _ in range(n):
            self._episode_done_queue.put({
                "garment": garment_name,
                "garment_type": garment_type,
                "seed": seed,
                "ep_idx": ep_idx,
                "success": False,
                "discarded": True,
                "discard_reason": "early_stop_on_success",
                "rollout_type": rollout_type,
            })

    # -- internal --

    def _run(self):
        """Proxy thread: serve the sim over HTTP, run the async gateway.

        The sim (RemotePolicy) POSTs commands to a small HTTP server; each
        request is queued and answered by the async _gateway, which also keeps
        a persistent WebSocket to the policy server.
        """
        import asyncio
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        proxy = self

        class _SimHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    body = json.loads(raw)
                except Exception:
                    body = {}
                ev = threading.Event()
                slot: list = [None]
                proxy._sim_req_queue.put((self.path, body, ev, slot))
                # No timeout here: the first inference legitimately blocks for a
                # while (JAX JIT compilation, best-of-N). A truly dead gateway is
                # bounded by the sim's urllib timeout and the worker's
                # NO_OUTPUT_TIMEOUT, which restart the sim cleanly.
                ev.wait()
                payload = json.dumps(slot[0] if slot[0] is not None else {}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass  # silence per-request logging

        ThreadingHTTPServer.allow_reuse_address = True
        try:
            self._http = ThreadingHTTPServer(("localhost", self.port), _SimHandler)
            self._http.daemon_threads = True
        except Exception as e:
            self._ready_event.set()  # unblock wait_ready() even on error
            self._error = e
            logger.error(f"{self.label} Proxy HTTP bind error: {e}", exc_info=True)
            return

        threading.Thread(
            target=self._http.serve_forever, daemon=True,
            name=f"proxy-http-w{self.worker_id}",
        ).start()
        # Signal the port is bound BEFORE the worker starts the sim, so the
        # client never gets ECONNREFUSED.
        self._ready_event.set()
        logger.info(f"{self.label} Proxy listening on http://localhost:{self.port}")
        try:
            asyncio.run(self._gateway())
        except Exception as e:
            self._error = e
            logger.error(f"{self.label} Proxy gateway error: {e}", exc_info=True)
        finally:
            # shutdown() stops serve_forever; server_close() releases the port
            # so the next sim restart can rebind it (else EADDRINUSE).
            try:
                self._http.shutdown()
            except Exception:
                pass
            try:
                self._http.server_close()
            except Exception:
                pass

    async def _gateway(self):
        """Drive one sim session: route obs→policy, action→sim, manage episodes.

        Requests arrive from the sim-facing HTTP server via ``_sim_req_queue``
        as ``(endpoint, msg, event, slot)`` tuples; each is answered exactly
        once via ``respond(...)``. A ``None`` item is the shutdown sentinel.
        """
        import asyncio

        # Persistent connection to policy server (session already created by worker)
        ws_policy = await self._connect_policy()
        if ws_policy is None:
            logger.error(f"{self.label} Could not connect to policy server — aborting proxy")
            return

        # Per-episode state
        trajectory: list[dict] = []
        current_ep_idx: int = -1
        garment_info_for_ep: dict | None = None
        augmentation_info_for_ep: dict | None = None  # texture + tint from PATCH 8
        current_inference_config: dict | None = None  # per-episode inference hyperparams
        # Per-episode garment tracking (for multi-garment mode)
        ep_garment_name: str = self.garment_name
        ep_garment_type: str = self.garment_type
        ep_base_seed: int = self.base_seed
        ep_restore_data: dict | None = self._first_task.get("restore_data")
        ep_global_idx: int | None = self._first_task.get("ep_idx")
        # True when a task has been dequeued but its outcome (success/failure
        # PKL or crash re-queue) has not yet been recorded.  Used in finally
        # to detect "task assigned but never produced any frames" (e.g. sim
        # hung during garment switch) and re-queue it.
        ep_task_pending: bool = True
        # Previous episode's garment info (for flushing on next reset)
        prev_garment_name: str = ep_garment_name
        prev_garment_type: str = ep_garment_type
        prev_seed: int = ep_base_seed
        prev_restore_data: dict | None = ep_restore_data

        # EMA tracking (initialized here; reset per-episode in reset handler)
        ema_value: float = 0.0
        ema_max: float = 0.0
        ema_chunk_count: int = 0
        failure_state_snapshot: dict | None = None
        failure_state_requested: bool = False
        failure_frame: int | None = None

        # Early snapshot at step 5 for adaptive success saving
        early_snapshot: dict | None = None
        early_snapshot_requested: bool = False

        # Proxy-side action chunk cache — stateless server approach.
        # The server returns a full chunk on each infer_chunk call; the proxy
        # manages the cache and inpainting state locally.  This eliminates all
        # server-side session state and reduces server calls by ~execute_in_n_steps×.
        action_chunk: np.ndarray | None = None   # [execute_in_n_steps, 12]
        chunk_idx: int = 0
        next_initial_actions: list | None = None  # for inpainting (passed to next infer_chunk)
        execute_in_n_steps: int = 20              # updated from first server response
        chunk_value: float | None = None          # V(s) from integrated value head (per-chunk)

        # Garment type inference memory: warmup call detects type on first obs,
        # first 5 real chunks refine via majority vote, then fixed permanently.
        gt_memory_predictions: list[int] = []
        gt_memory_fixed: bool = False
        gt_memory_current: int | None = None  # None = needs warmup call

        # Success replay mode: when replay_actions are provided, use saved
        # actions instead of policy actions (but still call policy for values).
        ep_replay_actions: np.ndarray | None = None  # [N, 12] saved actions
        replay_action_idx: int = 0

        # Load inference optimization prior if available
        _inference_prior = None
        if self.prior_file:
            try:
                from lehome_solution.eval.inference_optimization import load_prior
                _inference_prior = load_prior(self.prior_file)
                logger.info(f"{self.label} Loaded inference prior from {self.prior_file}")
            except Exception as e:
                logger.warning(f"{self.label} Failed to load prior: {e}")

        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        # Pending background flush futures (value+pkl); awaited in finally before exit.
        flush_futures: list = []

        # The sim posts to clean HTTP endpoints; map each to the message type
        # the handlers below already dispatch on (kept verbatim).
        _ENDPOINT_TO_TYPE = {
            "/next_task": "next_task",
            "/reset": "reset",
            "/infer": "action",
            "/snapshot": "state_snapshot",
            "/update_check_status": "update_check_status",
        }
        try:
            while True:
                item = await loop.run_in_executor(None, self._sim_req_queue.get)
                if item is None:
                    break  # shutdown sentinel pushed by join()
                endpoint, msg, _ev, _slot = item

                def respond(d, _ev=_ev, _slot=_slot):
                    _slot[0] = d
                    _ev.set()

                if endpoint == "/ping":
                    respond({"status": "ok"})
                    continue
                # The sim POSTs JSON with base64-encoded images, so there is no
                # binary frame to unpack — image_blob is always None.
                image_blob = None
                msg_type = _ENDPOINT_TO_TYPE.get(endpoint)
                if msg_type is None:
                    respond({"error": f"unknown endpoint {endpoint}"})
                    continue

                if msg_type == "next_task":
                    # If NaN was detected, shut down immediately so sim restarts.
                    if self._nan_detected:
                        respond({"shutdown": True})
                        logger.info(f"{self.label} NaN detected — sent shutdown for sim restart")
                        continue

                    # Streaming protocol: thin client asks what to do next.
                    task_info = self._get_next_task()
                    if task_info is None:
                        respond({"shutdown": True})
                        logger.info(f"{self.label} All tasks done, sent shutdown")
                    else:
                        # Extract ep_idx first — re-queued tasks (NaN/crash)
                        # may lack it.  Use current_ep_idx+1 as fallback.
                        task_ep_idx = task_info.get("ep_idx", current_ep_idx + 1)
                        # Eagerly flush the current trajectory before updating
                        # garment state.  If the sim hangs/crashes between
                        # next_task and reset, the finally block would otherwise
                        # flush with the NEW garment name (wrong).  Flushing
                        # here ensures the trajectory is always saved under the
                        # correct garment.  The reset handler will see an empty
                        # trajectory and skip its own flush.
                        if current_ep_idx >= 0 and trajectory:
                            if len(trajectory) >= self.MIN_EPISODE_FRAMES:
                                # Block (in executor, so we don't stall the asyncio
                                # loop) until a global flush slot is available.
                                # See _MAX_CONCURRENT_FLUSHES for the rationale.
                                await loop.run_in_executor(None, _FLUSH_SEMAPHORE.acquire)
                                try:
                                    fut = loop.run_in_executor(
                                        None,
                                        self._flush_episode_release,
                                        trajectory[:], current_ep_idx, garment_info_for_ep,
                                        failure_state_snapshot, failure_frame,
                                        ep_garment_name, ep_garment_type, ep_base_seed,
                                        augmentation_info_for_ep, ep_restore_data,
                                        early_snapshot, near_success_frame,
                                    )
                                except Exception:
                                    _FLUSH_SEMAPHORE.release()
                                    raise
                                # Prune completed futures so the list doesn't
                                # hold onto them across the entire eval session.
                                flush_futures = [f for f in flush_futures if not f.done()]
                                flush_futures.append(fut)
                            else:
                                logger.warning(
                                    f"{self.label} ep{current_ep_idx}: discarding "
                                    f"{len(trajectory)}-frame episode "
                                    f"(min={self.MIN_EPISODE_FRAMES})"
                                )
                            trajectory = []
                        # Previous task is done (flushed or discarded above).
                        ep_task_pending = False
                        # Snapshot the previous episode's garment info before
                        # overwriting, so the reset handler can flush it correctly.
                        prev_garment_name = ep_garment_name
                        prev_garment_type = ep_garment_type
                        prev_seed = ep_base_seed
                        prev_restore_data = ep_restore_data
                        # Update current garment/type for this episode
                        ep_garment_name = task_info["garment_name"]
                        ep_garment_type = task_info["garment_type"]
                        ep_base_seed = task_info["seed"]
                        ep_restore_data = task_info.get("restore_data")
                        ep_global_idx = task_ep_idx
                        # New task assigned but not yet recorded.
                        ep_task_pending = True
                        # Also update self for policy server messages
                        self.garment_name = ep_garment_name
                        self.garment_type = ep_garment_type
                        self.base_seed = ep_base_seed
                        resp = {
                            "garment_name": ep_garment_name,
                            "garment_type": ep_garment_type,
                            "seed": ep_base_seed,
                            "ep_idx": task_ep_idx,
                        }
                        # Include restore_data so thin client can restore failure state
                        if ep_restore_data is not None:
                            resp["restore_data"] = ep_restore_data
                        respond(resp)
                        logger.info(
                            f"{self.label} next_task: {ep_garment_name} "
                            f"seed={ep_base_seed} ep_idx={task_ep_idx}"
                            f"{' (restore)' if ep_restore_data else ''}"
                        )

                elif msg_type == "reset":
                    # If NaN was detected, just ack and skip — sim is shutting down.
                    if self._nan_detected:
                        respond({"status": "ok"})
                        continue

                    ep_idx_from_sim = msg.get("ep_idx", 0)

                    # Snapshot the just-finished episode before clearing state.
                    # Use prev_* values saved in next_task handler — ep_garment_name
                    # has already been overwritten with the NEW episode's garment.
                    if current_ep_idx >= 0 and trajectory:
                        if len(trajectory) >= self.MIN_EPISODE_FRAMES:
                            snap_traj = trajectory[:]
                            snap_ep_idx = current_ep_idx
                            snap_ginfo = garment_info_for_ep
                            snap_failure = failure_state_snapshot
                            snap_failure_frame = failure_frame
                            snap_garment_name = prev_garment_name
                            snap_garment_type = prev_garment_type
                            snap_seed = prev_seed
                            snap_augmentation = augmentation_info_for_ep
                            snap_restore_data = prev_restore_data
                            snap_early = early_snapshot
                            snap_near_success = near_success_frame
                            await loop.run_in_executor(None, _FLUSH_SEMAPHORE.acquire)
                            try:
                                fut = loop.run_in_executor(
                                    None,
                                    self._flush_episode_release,
                                    snap_traj, snap_ep_idx, snap_ginfo,
                                    snap_failure, snap_failure_frame,
                                    snap_garment_name, snap_garment_type, snap_seed,
                                    snap_augmentation, snap_restore_data,
                                    snap_early, snap_near_success,
                                )
                            except Exception:
                                _FLUSH_SEMAPHORE.release()
                                raise
                            flush_futures = [f for f in flush_futures if not f.done()]
                            flush_futures.append(fut)
                        else:
                            logger.warning(
                                f"{self.label} ep{current_ep_idx}: discarding "
                                f"{len(trajectory)}-frame episode "
                                f"(min={self.MIN_EPISODE_FRAMES})"
                            )

                    trajectory = []

                    # Clear proxy-side chunk cache — new episode starts fresh.
                    action_chunk = None
                    chunk_idx = 0
                    next_initial_actions = None
                    chunk_value = None
                    chunk_checkpoint = None
                    chunk_garment_type = None
                    chunk_completion = None
                    chunk_ttc = None
                    # Keypoint + world-modeling head predictions.
                    # One set per chunk, reused for every frame until the next
                    # policy call — same pattern as success_pred / checkpoint_pred.
                    chunk_keypoint = None       # list[float] length 21 (per-slot normalized distances)
                    chunk_wm_flow = {}          # dict: success/completion/keypoint × cond/uncond
                    # Best-of-N diagnostics (only non-None when the server
                    # runs with num_rollout_candidates > 1).
                    chunk_bon = {}              # dict: chosen/mean/min/max/std/spread/n_valid
                    policy_call_count = 0  # track calls for fast-warmup

                    # Garment type inference memory: warmup call detects type,
                    # first 5 real chunks refine via majority vote, then fixed.
                    gt_memory_predictions: list[int] = []
                    gt_memory_fixed = False
                    gt_memory_current: int | None = None  # None = needs warmup

                    # EMA tracking: reset per episode
                    ema_value = 0.0
                    ema_max = 0.0
                    ema_chunk_count = 0
                    failure_state_snapshot = None
                    failure_state_requested = False
                    failure_frame = None
                    # Realized augmentation arrives in the /reset payload (the
                    # sim applies it during episode setup, then reports it back).
                    augmentation_info_for_ep = msg.get("augmentation_info")

                    # Early snapshot at step 5 for adaptive success saving
                    early_snapshot: dict | None = None
                    early_snapshot_requested = False

                    # Near-success tracking: frame where EMA success_pred > 0.9
                    # (saved if episode fails)
                    near_success_frame: int | None = None

                    # Replay mode: load replay_actions from task if present
                    ep_replay_actions = None
                    replay_action_idx = 0
                    ep_semi_success_replay = False
                    rd = ep_restore_data or {}
                    if "_replay_actions" in rd:
                        ep_replay_actions = np.array(rd["_replay_actions"], dtype=np.float32)
                        ep_semi_success_replay = bool(rd.get("_semi_success_replay"))
                        mode_label = "semi_success_replay" if ep_semi_success_replay else "replay"
                        logger.info(
                            f"{self.label} ep{ep_idx_from_sim}: {mode_label} mode "
                            f"({len(ep_replay_actions)} saved actions)"
                        )

                    # Semi-success replay: we know the ground-truth garment
                    # type from saved metadata, and the open-loop replay phase
                    # uses recorded actions anyway — so the warmup machinery
                    # is pure overhead. Lock garment-type to GT and bypass
                    # fast-warmup chunk shrinking entirely.
                    if ep_semi_success_replay:
                        gt_memory_current = GARMENT_TYPE_TO_ID.get(self.garment_type, 0)
                        gt_memory_fixed = True
                        policy_call_count = _FAST_WARMUP_CALLS_BYPASS
                        logger.info(
                            f"{self.label} ep{ep_idx_from_sim}: "
                            f"semi_success_replay — garment type locked to GT "
                            f"({self.garment_type}={gt_memory_current}), "
                            f"warmup + fast-warmup skipped"
                        )

                    current_ep_idx = ep_idx_from_sim
                    garment_info_for_ep = msg.get("garment_info")
                    self._explore_logged_this_episode = False

                    # Sample inference config for this episode.
                    # Priority order:
                    #   1. per_garment_type_inference_config (fixed override)
                    #   2. Thompson Sampling prior (if loaded)
                    # Hard mining and success replay use the current best config (exploit
                    # only) — their outcomes shouldn't influence exploration.
                    current_inference_config = None
                    if self.per_garment_type_inference_config is not None:
                        try:
                            from lehome_solution.eval.inference_optimization import (
                                DEFAULT_CONFIG as _IO_DEFAULT,
                            )
                            from lehome_solution.constants import ACTION_HORIZON as _AH
                            overrides = self.per_garment_type_inference_config.get(
                                ep_garment_type, {})
                            cfg = dict(_IO_DEFAULT)
                            cfg.update(overrides)
                            # Cast types and derive execute_in_n_steps.
                            cfg["actions_to_execute"] = int(cfg["actions_to_execute"])
                            cfg["actions_to_keep"] = int(cfg["actions_to_keep"])
                            cfg["k_execute"] = float(cfg["k_execute"])
                            cfg["num_rollout_candidates"] = max(1, int(cfg.get("num_rollout_candidates", 1)))
                            # When ate < 4, the wrapper's spline cannot resample
                            # the chunk; force k_execute=1 (mirrors sample_config).
                            if cfg["actions_to_execute"] < 4 and cfg["k_execute"] != 1.0:
                                cfg["k_execute"] = 1.0
                            cfg["execute_in_n_steps"] = max(1, int(cfg["k_execute"] * cfg["actions_to_execute"]))
                            cfg["num_steps"] = _IO_DEFAULT["num_steps"]
                            if cfg["actions_to_execute"] + cfg["actions_to_keep"] > _AH:
                                logger.warning(
                                    f"{self.label} ep{current_ep_idx} per-type config "
                                    f"violates ate+atk<={_AH}, falling back to default"
                                )
                                cfg = dict(_IO_DEFAULT)
                                cfg["execute_in_n_steps"] = max(1, int(cfg["k_execute"] * cfg["actions_to_execute"]))
                            current_inference_config = cfg
                            self._episode_inference_configs[current_ep_idx] = current_inference_config
                            logger.info(
                                f"{self.label} ep{current_ep_idx} inference config "
                                f"(fixed per-type, gt={ep_garment_type}): "
                                f"ate={current_inference_config['actions_to_execute']} "
                                f"eins={current_inference_config['execute_in_n_steps']} "
                                f"atk={current_inference_config['actions_to_keep']} "
                                f"tti={current_inference_config['time_threshold_inpaint']:.2f} "
                                f"cfg={current_inference_config['cfg_scale']} "
                                f"nt={current_inference_config.get('noise_temperature', 1.0)} "
                                f"nrc={current_inference_config.get('num_rollout_candidates', 1)}"
                            )
                        except Exception as e:
                            logger.warning(f"{self.label} ep{current_ep_idx} per-type config build failed: {e}")
                    elif _inference_prior is not None:
                        try:
                            _is_exploit_ep = (
                                ep_replay_actions is not None  # success_replay
                                or (ep_restore_data is not None and "_replay_actions" not in (ep_restore_data or {}))  # hard_mining
                            )
                            if _is_exploit_ep:
                                from lehome_solution.eval.inference_optimization import get_best_config
                                current_inference_config = get_best_config(
                                    _inference_prior, garment_type=ep_garment_type)
                            else:
                                from lehome_solution.eval.inference_optimization import sample_config
                                ep_rng = np.random.RandomState((ep_base_seed + current_ep_idx) % (2**32))
                                current_inference_config = sample_config(
                                    _inference_prior, ep_rng, garment_type=ep_garment_type)
                            self._episode_inference_configs[current_ep_idx] = current_inference_config
                            logger.info(
                                f"{self.label} ep{current_ep_idx} inference config "
                                f"({'best' if _is_exploit_ep else 'sampled'}, gt={ep_garment_type}): "
                                f"ate={current_inference_config['actions_to_execute']} "
                                f"eins={current_inference_config['execute_in_n_steps']} "
                                f"atk={current_inference_config['actions_to_keep']} "
                                f"tti={current_inference_config['time_threshold_inpaint']:.2f} "
                                f"nt={current_inference_config.get('noise_temperature', 1.0)} "
                                f"nrc={current_inference_config.get('num_rollout_candidates', 1)}"
                            )
                        except Exception as e:
                            logger.warning(f"{self.label} ep{current_ep_idx} config sampling failed: {e}")

                    respond({"status": "ok"})
                    logger.info(f"{self.label} Reset ep{current_ep_idx} ok")

                elif msg_type == "update_check_status":
                    # Update the last trajectory frame's check_status + distances.
                    # Sent by the thin client after success detection so the
                    # final dense reward checkpoint lands on the last executed
                    # action frame (not a new frame).
                    cs = msg.get("check_status")
                    if trajectory and cs is not None:
                        trajectory[-1]["check_status"] = np.asarray(cs, dtype=np.float32)
                    cd = msg.get("check_distances")
                    if trajectory and cd is not None:
                        trajectory[-1]["check_distances"] = np.asarray(cd, dtype=np.float32)
                    respond({"status": "ok"})

                elif msg_type == "state_snapshot":
                    # Physics state from thin client at the detected failure/success frame.
                    # Contains garment particle positions/velocities and robot joints.
                    snapshot_data = msg.get("state", {})
                    n_pts = len(snapshot_data.get("garment_points", []))
                    if failure_state_requested:
                        failure_state_snapshot = snapshot_data
                        failure_state_requested = False
                        logger.info(
                            f"{self.label} ep{current_ep_idx}: received state snapshot "
                            f"({n_pts} particles) at failure frame {failure_frame}"
                        )
                    if early_snapshot_requested:
                        early_snapshot = snapshot_data
                        early_snapshot_requested = False
                    respond({"status": "ok"})

                elif msg_type == "action":
                    # Decode obs (images are base64 in the JSON body; no blob).
                    decoded = _decode_obs_images(msg, image_blob)

                    # Log observation dtype/shape/range on first frame of each episode.
                    # Helps diagnose camera format changes across resolutions.
                    frame_idx = len(trajectory)
                    if frame_idx == 0:
                        for obs_key in ("observation.images.top_rgb",
                                        "observation.images.left_rgb",
                                        "observation.images.right_rgb"):
                            img = decoded.get(obs_key)
                            if img is not None and isinstance(img, np.ndarray):
                                logger.info(
                                    f"{self.label} ep{current_ep_idx} obs {obs_key}: "
                                    f"shape={img.shape} dtype={img.dtype} "
                                    f"range=[{float(img.min()):.2f},{float(img.max()):.2f}] "
                                    f"mean={float(img.mean()):.2f}"
                                )
                        state = decoded.get("observation.state")
                        if state is not None and isinstance(state, np.ndarray):
                            logger.info(
                                f"{self.label} ep{current_ep_idx} state: "
                                f"shape={state.shape} dtype={state.dtype} "
                                f"range=[{float(state.min()):.3f}, {float(state.max()):.3f}]"
                                f" values={[f'{v:.3f}' for v in state.tolist()]}"
                            )
                    # STALE-CAMERA DETECTION: compare full-resolution decoded images
                    # between step 0 and step 1. Uses `_stale_cam_step0` (captured
                    # on the previous frame from `decoded`) instead of trajectory[0]
                    # because trajectory images are downsized for memory and would
                    # have different bytes even when the camera is fresh.
                    if frame_idx == 1 and self._stale_cam_step0:
                        for cam_key, obs_key in (
                            ("top_rgb", "observation.images.top_rgb"),
                            ("left_rgb", "observation.images.left_rgb"),
                            ("right_rgb", "observation.images.right_rgb"),
                        ):
                            img_curr = decoded.get(obs_key)
                            img_prev = self._stale_cam_step0.get(cam_key)
                            if img_curr is not None and img_prev is not None:
                                same = (img_curr.tobytes() == img_prev.tobytes())
                                mean_diff = abs(float(img_curr.mean()) - float(img_prev.mean()))
                                if same:
                                    logger.warning(
                                        f"{self.label} ep{current_ep_idx} STALE CAMERA: "
                                        f"{cam_key} is IDENTICAL at step 0 and step 1 — "
                                        f"camera NOT updating (Isaac Sim resolution bug?)"
                                    )
                                else:
                                    logger.info(
                                        f"{self.label} ep{current_ep_idx} camera {cam_key}: "
                                        f"updating ok (mean_diff={mean_diff:.2f} from step 0)"
                                    )
                        # Release the full-resolution step-0 snapshot now that the
                        # comparison is done — no other code path reads it.
                        self._stale_cam_step0 = {}
                    state = np.asarray(decoded.get("observation.state", np.zeros(12)), dtype=np.float32).copy()

                    # NaN state detection: Isaac Sim physics explosion.
                    # Once PhysX returns NaN, it never recovers (even after
                    # reset/garment switch).  Stop this episode, re-queue
                    # the current task, and tell the thin client to shutdown
                    # so run_worker_session can restart the whole sim.
                    if np.any(np.isnan(state)):
                        if not self._nan_detected:
                            self._nan_detected = True
                            logger.error(
                                f"{self.label} ep{current_ep_idx} step{frame_idx}: "
                                f"NaN STATE DETECTED — Isaac Sim physics exploded. "
                                f"Discarding episode, re-queuing task, restarting sim."
                            )
                            # Discard corrupted trajectory
                            trajectory.clear()
                            # Re-queue current task so another worker picks it up
                            self._task_queue.put({
                                "garment_name": ep_garment_name,
                                "garment_type": ep_garment_type,
                                "seed": ep_base_seed,
                                "ep_idx": ep_global_idx,
                                "restore_data": ep_restore_data,
                            })
                            # Task is handled (re-queued) — clear pending so
                            # the finally block doesn't re-queue it again.
                            ep_task_pending = False
                        # Send stop + zeros to end this episode quickly
                        respond({"actions": [0.0] * 12, "stop": True})
                        continue

                    top = decoded.get("observation.images.top_rgb")
                    left = decoded.get("observation.images.left_rgb")
                    right = decoded.get("observation.images.right_rgb")

                    frame: dict = {
                        "observation.state": state,
                        "action": np.zeros(12, dtype=np.float32),  # placeholder
                        "task": ep_garment_name or ep_garment_type or "eval",
                        "timestamp": frame_idx / self.FPS,
                    }
                    # Store images downsized to (storage_height, storage_width) so
                    # the trajectory/pkl/video-render closures don't each pin a
                    # ~1.7 GB 640×480 copy. The policy always receives native
                    # camera resolution.
                    if top is not None:
                        frame["observation.images.top_rgb"] = self._resize_for_storage(np.asarray(top)[..., :3])
                    if left is not None:
                        frame["observation.images.left_rgb"] = self._resize_for_storage(np.asarray(left)[..., :3])
                    if right is not None:
                        frame["observation.images.right_rgb"] = self._resize_for_storage(np.asarray(right)[..., :3])
                    # Capture full-resolution step-0 images for the frame-1 stale-
                    # camera check. Held only until frame 1 runs, then cleared.
                    if frame_idx == 0:
                        self._stale_cam_step0 = {}
                        if top is not None:
                            self._stale_cam_step0["top_rgb"] = np.asarray(top)[..., :3]
                        if left is not None:
                            self._stale_cam_step0["left_rgb"] = np.asarray(left)[..., :3]
                        if right is not None:
                            self._stale_cam_step0["right_rgb"] = np.asarray(right)[..., :3]
                    check_status = decoded.get("check_status")
                    if check_status is not None:
                        frame["check_status"] = np.asarray(check_status, dtype=np.float32).copy()
                    check_distances = decoded.get("check_distances")
                    if check_distances is not None:
                        frame["check_distances"] = np.asarray(check_distances, dtype=np.float32).copy()
                    trajectory.append(frame)

                    # Propagate current chunk's value/checkpoint predictions to
                    # every frame (not just keyframes).  One prediction per chunk,
                    # reused until the next policy call.
                    if chunk_value is not None:
                        trajectory[-1]["success_pred"] = chunk_value
                    if chunk_checkpoint is not None:
                        trajectory[-1]["checkpoint_pred"] = chunk_checkpoint
                    if chunk_garment_type is not None:
                        trajectory[-1]["garment_type_pred"] = chunk_garment_type
                    if chunk_completion is not None:
                        trajectory[-1]["completion_pred"] = chunk_completion
                    if chunk_ttc is not None:
                        trajectory[-1]["ttc_pred"] = chunk_ttc
                    # Propagate keypoint + WM-flow chunk predictions to every frame.
                    if chunk_keypoint is not None:
                        trajectory[-1]["keypoint_distances_pred"] = np.asarray(chunk_keypoint, dtype=np.float32).copy()
                    for k, v in chunk_wm_flow.items():
                        if v is None:
                            continue
                        arr = np.asarray(v, dtype=np.float32)
                        trajectory[-1][f"wm_flow_{k}"] = arr.item() if arr.ndim == 0 else arr.copy()
                    # Best-of-N stats: per-chunk scalars, same value across the
                    # chunk's frames (like success_pred).
                    for k, v in chunk_bon.items():
                        if v is None:
                            continue
                        trajectory[-1][k] = v

                    # Chunk-based stateless inference.
                    # Call the policy server only when the current chunk is exhausted
                    # (~every execute_in_n_steps steps).  Between calls, return the
                    # cached chunk[chunk_idx].  This is ~20× fewer server calls and
                    # eliminates all server-side session state.
                    # Garment type warmup: throwaway call to detect garment type
                    # before the first real inference.  Only garment_type_pred is
                    # kept; actions/values/inpainting are all discarded.
                    if gt_memory_current is None:
                        warmup_msg = _encode_obs_for_policy(
                            decoded,
                            self.session_id,
                            garment_type_input_id=0,
                        )
                        await ws_policy.send(json.dumps(warmup_msg))
                        warmup_resp = json.loads(await ws_policy.recv())
                        warmup_gt = warmup_resp.get("garment_type_pred")
                        gt_memory_current = warmup_gt if warmup_gt is not None else 0
                        logger.info(
                            f"{self.label} ep{current_ep_idx}: "
                            f"garment type warmup prediction: {gt_memory_current}"
                        )

                    if action_chunk is None or chunk_idx >= execute_in_n_steps:
                        policy_msg = _encode_obs_for_policy(
                            decoded,
                            self.session_id,
                            garment_type_input_id=gt_memory_current,
                        )
                        if next_initial_actions is not None:
                            policy_msg["initial_actions"] = next_initial_actions
                        # Build inference_config: start with per-episode Thompson Sampling
                        # config (if any), then inject static noise_temperature as fallback
                        # (only when inference optimization is not managing it).
                        effective_inference_config = dict(current_inference_config) if current_inference_config else {}
                        if "noise_temperature" not in effective_inference_config and self.noise_temperature != 1.0:
                            effective_inference_config["noise_temperature"] = self.noise_temperature
                        # DART-style correlated-noise injection. Bernoulli per chunk;
                        # only fires when BOTH prob > 0 AND scale > 0. When neither is
                        # set the key is never added → submission path is bit-identical.
                        if (
                            self.explore_noise_prob > 0.0
                            and self.explore_noise_scale > 0.0
                            and np.random.random() < self.explore_noise_prob
                        ):
                            effective_inference_config["explore_noise_scale"] = self.explore_noise_scale
                            if not getattr(self, "_explore_logged_this_episode", False):
                                logger.info(
                                    f"{self.label} ep{current_ep_idx}: exploration noise fired "
                                    f"(prob={self.explore_noise_prob}, scale={self.explore_noise_scale})"
                                )
                                self._explore_logged_this_episode = True
                        # Fast warmup: first 5 policy calls shorten the chunk
                        # (actions_to_execute=5 → execute_in_n_steps=5 via k_execute=1)
                        # so the proxy cycles back to the server quickly and
                        # establishes the garment type prediction. actions_to_keep
                        # is NOT overridden — it stays at the per-type Thompson-sampled
                        # value so inpainting continuity is preserved. Denoising
                        # num_steps is not reduced either.
                        _FAST_WARMUP_CALLS = 5
                        if policy_call_count < _FAST_WARMUP_CALLS:
                            effective_inference_config["actions_to_execute"] = 5
                            effective_inference_config["execute_in_n_steps"] = 5
                        policy_call_count += 1
                        if effective_inference_config:
                            policy_msg["inference_config"] = effective_inference_config
                        await ws_policy.send(json.dumps(policy_msg))
                        resp = json.loads(await ws_policy.recv())
                        if "error" in resp:
                            logger.warning(f"{self.label} Policy error: {resp['error']}")
                        raw_chunk = resp.get("actions", [[0.0] * 12])
                        action_chunk = np.array(raw_chunk, dtype=np.float32)
                        # Server returns next_initial_actions for next episode's inpainting
                        next_initial_actions = resp.get("next_initial_actions")  # list or None
                        execute_in_n_steps = resp.get("execute_in_n_steps", 20)
                        # Integrated value head: one V(s) per chunk, reused for all frames
                        chunk_value = resp.get("success_pred")  # float or None
                        chunk_checkpoint = resp.get("checkpoint_pred")  # float or None
                        chunk_garment_type = resp.get("garment_type_pred")  # int or None
                        chunk_completion = resp.get("completion_pred")  # float or None
                        chunk_ttc = resp.get("ttc_pred")  # float or None
                        # Keypoint distance head (Head 1): 21-wide vector.
                        chunk_keypoint = resp.get("keypoint_distances")  # list[21] or None
                        # WM-flow head (Head 3): cond + uncond monitoring predictions,
                        # extracted at t=0.1 inside the denoising loop.
                        chunk_wm_flow = {
                            k: resp.get(f"wm_flow_{k}")
                            for k in (
                                "success_cond", "completion_cond", "keypoint_cond",
                                "success_uncond", "completion_uncond", "keypoint_uncond",
                            )
                        }
                        chunk_bon = {
                            k: resp.get(k) for k in (
                                "best_of_n_score_chosen", "best_of_n_score_mean",
                                "best_of_n_score_min", "best_of_n_score_max",
                                "best_of_n_score_std", "best_of_n_score_spread",
                                "best_of_n_n_valid",
                            )
                        }
                        chunk_idx = 0
                        # Mark this frame as a keyframe (policy was queried here).
                        # Used by KeyframeDatasetWriter to build a small image dataset
                        # for fast value re-prediction without video decoding.
                        trajectory[-1]["is_keyframe"] = True

                        # Best-of-N candidate-spread summary: "spread" is max-min
                        # of the score across N candidates (0 means all got the
                        # same score → best-of-N is wasted effort). Only shown
                        # when chunk_bon has finite data (server ran with N>1).
                        _bon_spread = chunk_bon.get("best_of_n_score_spread")
                        _bon_n = chunk_bon.get("best_of_n_n_valid")
                        _bon_chosen = chunk_bon.get("best_of_n_score_chosen")
                        _bon_str = ""
                        if _bon_spread is not None and _bon_n is not None \
                                and np.isfinite(float(_bon_spread)):
                            _bon_str = f" bon[n={int(_bon_n)}]spread={float(_bon_spread):.4f}"
                            if _bon_chosen is not None and np.isfinite(float(_bon_chosen)):
                                _bon_str += f" chosen={float(_bon_chosen):.4f}"

                        logger.info(
                            f"{self.label} ep{current_ep_idx} step{frame_idx}: new chunk "
                            f"len={len(action_chunk)} inpainting={'yes' if next_initial_actions else 'no'}"
                            f"{f' value={chunk_value:.4f}' if chunk_value is not None else ''}"
                            f"{f' completion={chunk_completion:.2%}' if chunk_completion is not None else ''}"
                            f"{f' ttc={chunk_ttc:.4f}' if chunk_ttc is not None else ''}"
                            f"{_bon_str}"
                        )

                        # Garment type memory: refine via majority vote
                        # over the first 5 real chunks, then fix permanently.
                        if chunk_garment_type is not None:
                            if not gt_memory_fixed:
                                gt_memory_predictions.append(chunk_garment_type)
                                counts = Counter(gt_memory_predictions)
                                majority = counts.most_common(1)[0][0]
                                if majority != gt_memory_current:
                                    logger.info(
                                        f"{self.label} ep{current_ep_idx} step{frame_idx}: "
                                        f"garment type updated: {gt_memory_current} -> {majority}"
                                    )
                                    gt_memory_current = majority
                                if len(gt_memory_predictions) >= 5:
                                    gt_memory_fixed = True
                                    logger.info(
                                        f"{self.label} ep{current_ep_idx}: "
                                        f"garment type FIXED: {gt_memory_current}"
                                    )
                            elif chunk_garment_type != gt_memory_current:
                                logger.warning(
                                    f"{self.label} ep{current_ep_idx} step{frame_idx}: "
                                    f"garment type prediction changed to {chunk_garment_type} "
                                    f"but memory is fixed at {gt_memory_current}"
                                )

                        # Failure state capture: update EMA on each new chunk value.
                        # Skip for restore episodes — don't capture new failure states
                        # from recovery attempts (keep original NPZ in persistent dir).
                        if chunk_value is not None and ep_restore_data is None:
                            ema_chunk_count += 1
                            ema_value = self.EMA_ALPHA * chunk_value + (1 - self.EMA_ALPHA) * ema_value
                            if ema_chunk_count > self.EMA_SKIP_CHUNKS:
                                ema_max = max(ema_max, ema_value)
                                drop = ema_max - ema_value
                                if (drop > self.EMA_DROP_THRESH
                                        and ema_max > self.EMA_PEAK_THRESH
                                        and failure_frame is None):
                                    failure_frame = frame_idx
                                    failure_state_requested = True
                                    logger.info(
                                        f"{self.label} ep{current_ep_idx} step{frame_idx}: "
                                        f"FAILURE DETECTED ema={ema_value:.3f} "
                                        f"peak={ema_max:.3f} drop={drop:.3f} "
                                        f"— requesting state snapshot"
                                    )

                        # Near-success detection: first frame where EMA > 0.9.
                        if (near_success_frame is None
                                and ema_chunk_count > self.EMA_SKIP_CHUNKS
                                and ema_value > 0.9):
                            near_success_frame = len(trajectory) - 1

                    # Note: value/checkpoint/garment_type_pred are propagated to
                    # every frame above (after trajectory.append).  The chunk
                    # variables are updated when a new chunk is requested.

                    # Replay mode: use saved actions instead of policy actions.
                    # Policy was still called above for value predictions.
                    if ep_replay_actions is not None:
                        if replay_action_idx < len(ep_replay_actions):
                            action_arr = ep_replay_actions[replay_action_idx].copy()
                            replay_action_idx += 1
                        else:
                            if ep_semi_success_replay:
                                # Semi-success replay: hand off to the policy.
                                # Clear replay actions so subsequent steps use
                                # the normal policy path.
                                logger.info(
                                    f"{self.label} ep{current_ep_idx} step{frame_idx}: "
                                    f"semi-success replay actions exhausted, "
                                    f"handing off to policy"
                                )
                                ep_replay_actions = None
                                # Use the policy action from the current chunk
                                action_arr = action_chunk[chunk_idx] if action_chunk is not None else action_arr
                            else:
                                # Standard replay: repeat last action for 1s
                                # to let the garment settle, then stop.
                                action_arr = ep_replay_actions[-1].copy()
                                settle_frames = replay_action_idx - len(ep_replay_actions)
                                replay_action_idx += 1
                                if settle_frames == 0:
                                    logger.info(
                                        f"{self.label} ep{current_ep_idx} step{frame_idx}: "
                                        f"replay actions exhausted, repeating last action for 1s"
                                    )
                                if settle_frames >= self.FPS:  # 30 frames = 1s
                                    if not stop_episode:
                                        stop_episode = True
                                        logger.info(
                                            f"{self.label} ep{current_ep_idx} step{frame_idx}: "
                                            f"replay settle done, stopping episode"
                                        )
                                    resp_msg = {"actions": action_arr.tolist(), "stop": True}
                                    respond(resp_msg)
                                    continue
                    else:
                        action_arr = action_chunk[chunk_idx]
                    chunk_idx += 1

                    # Validate and log action on first frame + periodically.
                    if frame_idx == 0 or frame_idx % 100 == 0:
                        act_min = float(action_arr.min())
                        act_max = float(action_arr.max())
                        act_norm = float(np.linalg.norm(action_arr))
                        logger.info(
                            f"{self.label} ep{current_ep_idx} step{frame_idx} "
                            f"chunk_pos={chunk_idx-1}/{execute_in_n_steps} action: "
                            f"min={act_min:.3f} max={act_max:.3f} norm={act_norm:.3f} "
                            f"values={[f'{v:.3f}' for v in action_arr.tolist()]}"
                        )
                    if float(np.linalg.norm(action_arr)) < 1e-6:
                        logger.warning(
                            f"{self.label} ep{current_ep_idx} step{frame_idx}: "
                            f"NEAR-ZERO action — policy may be stuck"
                        )

                    # Fill action into trajectory frame
                    trajectory[-1]["action"] = action_arr.copy()

                    # Stuck detection: check at chunk boundaries (every ~20 steps)
                    # to avoid per-frame overhead. When stuck, tell sim to stop
                    # and trim trajectory to det_frame + STUCK_KEEP.
                    stop_episode = False
                    if chunk_idx >= execute_in_n_steps and self._check_stuck(trajectory):
                        det_frame = len(trajectory) - self.STUCK_WINDOW
                        trim_at = det_frame + self.STUCK_KEEP
                        orig_len = len(trajectory)
                        trajectory[:] = trajectory[:trim_at]
                        stop_episode = True
                        # Invalidate near_success_frame if it points beyond trimmed trajectory
                        if near_success_frame is not None and near_success_frame >= trim_at:
                            near_success_frame = None
                        logger.info(
                            f"{self.label} ep{current_ep_idx} STUCK detected at frame "
                            f"{det_frame}, trimmed {orig_len - trim_at} frames "
                            f"(keeping {trim_at}/{orig_len})"
                        )

                    # Request early snapshot at step 5 (for adaptive success saving)
                    if (frame_idx == self.EARLY_SNAPSHOT_FRAME
                            and ep_replay_actions is None
                            and ep_restore_data is None
                            and early_snapshot is None):
                        early_snapshot_requested = True

                    # Send action back to sim
                    resp_msg = {"actions": action_arr.tolist()}
                    needs_snapshot = (
                        (failure_state_requested and failure_state_snapshot is None) or
                        (early_snapshot_requested and early_snapshot is None)
                    )
                    if needs_snapshot:
                        resp_msg["save_state"] = True
                    if stop_episode:
                        resp_msg["stop"] = True
                    respond(resp_msg)

        except Exception as e:
            # "no close frame received or sent" is expected when Isaac Sim exits abruptly.
            msg = str(e)
            if "no close frame" in msg or "keepalive ping timeout" in msg:
                logger.debug(f"{self.label} Proxy gateway closed: {e}")
            else:
                logger.warning(f"{self.label} Proxy gateway error: {e}")
        finally:
            # Flush last episode — but detect crash (too few frames) and re-queue
            if trajectory and len(trajectory) >= self.MIN_EPISODE_FRAMES:
                await loop.run_in_executor(None, _FLUSH_SEMAPHORE.acquire)
                try:
                    fut = loop.run_in_executor(
                        None,
                        self._flush_episode_release,
                        trajectory[:], current_ep_idx, garment_info_for_ep,
                        failure_state_snapshot, failure_frame,
                        ep_garment_name, ep_garment_type, ep_base_seed,
                        augmentation_info_for_ep, ep_restore_data,
                        early_snapshot, near_success_frame,
                    )
                except Exception:
                    _FLUSH_SEMAPHORE.release()
                    raise
                flush_futures.append(fut)
                ep_task_pending = False
            elif trajectory:
                # Isaac Sim crashed mid-episode producing a garbage episode.
                # Discard the trajectory and re-queue the task for retry.
                logger.warning(
                    f"{self.label} ep{current_ep_idx}: discarding {len(trajectory)}-frame "
                    f"episode (crash detected, min={self.MIN_EPISODE_FRAMES}), "
                    f"re-queuing {ep_garment_name}"
                )
                self._crash_detected = True
                self._task_queue.put({
                    "garment_name": ep_garment_name,
                    "garment_type": ep_garment_type,
                    "seed": ep_base_seed,
                    "ep_idx": ep_global_idx,
                    "restore_data": ep_restore_data,
                })
                ep_task_pending = False
            elif ep_task_pending:
                # Task was dequeued (next_task or first_task) but no frames
                # ever arrived — sim hung during garment switch / startup
                # / stabilization.  Re-queue so it isn't silently lost.
                logger.warning(
                    f"{self.label} task pending but no frames recorded "
                    f"(hang during switch/startup): re-queuing {ep_garment_name}"
                )
                self._crash_detected = True
                self._task_queue.put({
                    "garment_name": ep_garment_name,
                    "garment_type": ep_garment_type,
                    "seed": ep_base_seed,
                    "ep_idx": ep_global_idx,
                    "restore_data": ep_restore_data,
                })
                ep_task_pending = False

            # Wait for all background flushes to complete
            if flush_futures:
                await _asyncio.gather(*flush_futures, return_exceptions=True)

            try:
                await ws_policy.close()
            except Exception:
                pass

            logger.info(f"{self.label} Proxy done, {len(self.pkl_paths)} pkls written")

    async def _connect_policy(self):
        """Open a persistent WebSocket to the policy server."""
        import websockets
        for attempt in range(20):
            try:
                ws = await websockets.connect(
                    self.server_url,
                    max_size=100 * 1024 * 1024,
                    open_timeout=30,
                    ping_interval=20,
                    ping_timeout=120,
                )
                logger.info(f"{self.label} Proxy connected to policy server {self.server_url}")
                return ws
            except Exception as e:
                logger.warning(f"{self.label} Policy connect {attempt+1}/20: {e}")
                await self._async_sleep(3)
        return None

    @staticmethod
    async def _async_sleep(seconds: float):
        import asyncio
        await asyncio.sleep(seconds)

    def _check_stuck(self, trajectory: list[dict]) -> bool:
        """Check if the episode is stuck based on recent trajectory frames.

        Uses a rolling window of STUCK_WINDOW frames. Returns True if both
        state_std and action_std are below STUCK_THRESHOLD across the window.
        """
        n = len(trajectory)
        if n < self.STUCK_WINDOW:
            return False

        window = trajectory[-self.STUCK_WINDOW:]
        states = np.array([f["observation.state"] for f in window], dtype=np.float32)
        actions = np.array([f["action"] for f in window], dtype=np.float32)

        s_std = np.mean(np.std(states, axis=0))
        a_std = np.mean(np.std(actions, axis=0))

        if s_std < self.STUCK_THRESHOLD and a_std < self.STUCK_THRESHOLD:
            # Also check value std if available (consistent per-episode)
            if window[0].get("success_pred") is not None:
                v_std = np.std([f["success_pred"] for f in window])
                if v_std >= self.STUCK_THRESHOLD:
                    return False  # Value changing → not stuck
            return True

        return False

    def _flush_episode_release(self, *args, **kwargs):
        """_flush_episode wrapper that releases the global flush semaphore on exit.

        The caller (gateway loop) acquires _FLUSH_SEMAPHORE *before* submitting
        this to the executor; this wrapper guarantees the slot is released
        even if _flush_episode raises.  The acquire-then-submit pattern is
        what bounds the trajectory-snapshot RAM (vs. acquiring inside
        _flush_episode, which would let snapshots queue up unboundedly in
        executor task args).
        """
        try:
            return self._flush_episode(*args, **kwargs)
        finally:
            _FLUSH_SEMAPHORE.release()

    def _flush_episode(
        self,
        trajectory: list[dict],
        ep_idx: int,
        garment_info: dict | None,
        failure_snapshot: dict | None = None,
        failure_frame: int | None = None,
        flush_garment_name: str | None = None,
        flush_garment_type: str | None = None,
        flush_seed: int | None = None,
        augmentation_info: dict | None = None,
        restore_data: dict | None = None,
        early_snapshot: dict | None = None,
        near_success_frame: int | None = None,
    ):
        """Write temp pkl for a completed episode.

        If failure_snapshot is provided, also write a failure state file to
        {video_dir}/physics_states/failure/ with enough data to restore the garment
        in Isaac Sim without replaying.

        Hard mining mode (from restore_data._hard_mining_mode):
        - Only successful episodes are saved (failures are discarded).
        - Only the first success per NPZ is saved (duplicates skipped).
        - "exact": on success, the source NPZ is deleted from the failures pool.
        - "augs": on success, the source NPZ is NOT deleted.

        flush_garment_name/type/seed: explicit per-episode values for multi-garment mode.
        Falls back to self.garment_name/type/base_seed for backward compatibility.
        """
        if not trajectory:
            return

        garment_name = flush_garment_name or self.garment_name
        garment_type = flush_garment_type or self.garment_type
        base_seed = flush_seed if flush_seed is not None else self.base_seed

        # Extract hard mining / replay metadata from restore_data
        hard_mining_mode = None
        restore_npz_path = None
        replay_mode = False
        semi_success_replay_mode = False
        max_attempts = 1
        attempt = 0  # 0-indexed
        if restore_data:
            hard_mining_mode = restore_data.get("_hard_mining_mode")
            restore_npz_path = restore_data.get("_restore_npz_path")
            replay_mode = "_replay_actions" in restore_data
            semi_success_replay_mode = bool(restore_data.get("_semi_success_replay"))
            max_attempts = int(restore_data.get("_max_attempts", 1))
            attempt = int(restore_data.get("_attempt", 0))

        # Determine rollout_type for episode metadata
        if semi_success_replay_mode:
            rollout_type = "semi_success_replay"
        elif replay_mode:
            rollout_type = "success_replay"
        elif hard_mining_mode:
            rollout_type = "hard_mining"
        else:
            rollout_type = self.default_rollout_type

        # Determine success: scan ALL frames (conditions can pass mid-episode
        # then un-pass if garment shifts). All garment types require ALL
        # conditions to pass simultaneously.
        is_success = False
        for frame in trajectory:
            cs = frame.get("check_status")
            if cs is not None and len(cs) > 0:
                if float(cs.min()) > 0.5:
                    is_success = True
                    break

        # Hard mining filtering: only save successful episodes
        if hard_mining_mode:
            if not is_success:
                attempts_done = attempt + 1
                can_retry = attempts_done < max_attempts and restore_data is not None
                logger.info(
                    f"{self.label} ep{ep_idx}: hard mining [{hard_mining_mode}] "
                    f"FAILED — discarding episode "
                    f"(attempt {attempts_done}/{max_attempts})"
                )
                if self._episode_done_queue is not None:
                    self._episode_done_queue.put({
                        "garment": garment_name,
                        "garment_type": garment_type,
                        "seed": base_seed,
                        "ep_idx": ep_idx,
                        "success": False,
                        "discarded": True,
                        "discard_reason": "hard_mining_failure",
                    })
                if can_retry:
                    self._enqueue_retry(
                        garment_name, garment_type, base_seed, ep_idx,
                        restore_data, attempts_done,
                    )
                return
            # First-success-only: skip if this NPZ was already solved
            if restore_npz_path and restore_npz_path in self._solved_npz_paths:
                logger.info(
                    f"{self.label} ep{ep_idx}: hard mining [{hard_mining_mode}] "
                    f"SUCCESS but NPZ already solved — skipping duplicate"
                )
                if self._episode_done_queue is not None:
                    self._episode_done_queue.put({
                        "garment": garment_name,
                        "garment_type": garment_type,
                        "seed": base_seed,
                        "ep_idx": ep_idx,
                        "success": True,
                        "discarded": True,
                        "discard_reason": "duplicate_success",
                    })
                return
            # Mark as solved
            if restore_npz_path:
                self._solved_npz_paths.add(restore_npz_path)
            # "exact" mode: delete the source NPZ on success
            if hard_mining_mode == "exact" and restore_npz_path:
                try:
                    npz_file = Path(restore_npz_path)
                    if npz_file.exists():
                        npz_file.unlink()
                        logger.info(
                            f"{self.label} ep{ep_idx}: hard mining [exact] "
                            f"SUCCESS — deleted NPZ {npz_file.name}"
                        )
                except Exception as e:
                    logger.warning(f"{self.label} ep{ep_idx}: failed to delete NPZ: {e}")
            logger.info(
                f"{self.label} ep{ep_idx}: hard mining [{hard_mining_mode}] "
                f"SUCCESS — saving episode to dataset"
            )

        # Success replay filtering: only save successful replay episodes
        if replay_mode and not semi_success_replay_mode:
            if not is_success:
                attempts_done = attempt + 1
                can_retry = attempts_done < max_attempts and restore_data is not None
                logger.info(
                    f"{self.label} ep{ep_idx}: success_replay "
                    f"FAILED — discarding episode "
                    f"(attempt {attempts_done}/{max_attempts})"
                )
                if self._episode_done_queue is not None:
                    self._episode_done_queue.put({
                        "garment": garment_name,
                        "garment_type": garment_type,
                        "seed": base_seed,
                        "ep_idx": ep_idx,
                        "success": False,
                        "discarded": True,
                        "discard_reason": "success_replay_failure",
                        "rollout_type": "success_replay",
                    })
                if can_retry:
                    self._enqueue_retry(
                        garment_name, garment_type, base_seed, ep_idx,
                        restore_data, attempts_done,
                    )
                return
            logger.info(
                f"{self.label} ep{ep_idx}: success_replay "
                f"SUCCESS — saving replayed episode to dataset"
            )

        # Semi-success replay filtering: save only successful episodes.
        # On success, also save as a success state for future success_replay.
        if semi_success_replay_mode:
            if not is_success:
                attempts_done = attempt + 1
                can_retry = attempts_done < max_attempts and restore_data is not None
                logger.info(
                    f"{self.label} ep{ep_idx}: semi_success_replay "
                    f"FAILED — discarding episode "
                    f"(attempt {attempts_done}/{max_attempts})"
                )
                if self._episode_done_queue is not None:
                    self._episode_done_queue.put({
                        "garment": garment_name,
                        "garment_type": garment_type,
                        "seed": base_seed,
                        "ep_idx": ep_idx,
                        "success": False,
                        "discarded": True,
                        "discard_reason": "semi_success_replay_failure",
                        "rollout_type": "semi_success_replay",
                    })
                if can_retry:
                    self._enqueue_retry(
                        garment_name, garment_type, base_seed, ep_idx,
                        restore_data, attempts_done,
                    )
                return
            # Success! Save full episode as a success state for future success_replay.
            # Use the original restore_data as the snapshot (it's the frame-5 state
            # from the semi-success NPZ — early_snapshot is not captured for replay episodes).
            logger.info(
                f"{self.label} ep{ep_idx}: semi_success_replay "
                f"SUCCESS — saving episode to dataset + saving as success state"
            )
            if restore_data:
                snapshot_for_save = {
                    k: restore_data[k] for k in (
                        "garment_points", "garment_velocities",
                        "garment_pos", "garment_ori",
                        "left_joint_pos", "right_joint_pos",
                    ) if k in restore_data
                }
                if snapshot_for_save:
                    all_actions = np.array(
                        [f["action"] for f in trajectory],
                        dtype=np.float32,
                    )
                    self._save_success_state(
                        snapshot_for_save, 0, all_actions,
                        trajectory, ep_idx, base_seed,
                        sanitize_basename(garment_name), garment_info,
                        garment_name, garment_type,
                        states_subdir="success",
                    )

        nv = sum(1 for f in trajectory if "success_pred" in f)
        if nv > 0:
            logger.info(f"{self.label} ep{ep_idx}: values on {nv}/{len(trajectory)} frames")
        else:
            logger.info(f"{self.label} ep{ep_idx}: no value annotations ({len(trajectory)} frames)")

        seed = base_seed
        base = sanitize_basename(garment_name)
        # Skip the first frame (initial post-reset observation); keep all if very short.
        save_frames = trajectory[1:] if len(trajectory) > 2 else trajectory

        # Write trajectory pkl (consumed by dataset/video writers). Skipped in
        # metrics-only mode; success metadata is still reported below.
        if self.video_dir and self.save_pkl and save_frames:
            out_dir = Path(self.video_dir) / "temp_episodes"
            out_dir.mkdir(parents=True, exist_ok=True)
            retry_suffix = f"_retry{attempt}" if attempt > 0 else ""
            name = f"{base}_seed{seed}_ep{ep_idx}{retry_suffix}.pkl"
            path = out_dir / name
            payload: dict = {"frames": save_frames}
            if garment_info:
                payload["garment_info"] = garment_info
            if ep_idx in self._episode_inference_configs:
                payload["inference_config"] = self._episode_inference_configs[ep_idx]
            if augmentation_info:
                payload["augmentation_info"] = augmentation_info
            try:
                with open(path, "wb") as f:
                    pickle.dump(payload, f)
                self.pkl_paths.append(path)
                logger.info(f"{self.label} Wrote pkl ep{ep_idx}: {path.name} ({len(save_frames)} frames, skipped first)")
            except Exception as e:
                logger.warning(f"{self.label} Failed to write pkl ep{ep_idx}: {e}")

        # Report episode result — always (independent of pkl/dataset writing).
        if self._episode_done_queue is not None:
            ep_result = {
                "garment": garment_name,
                "garment_type": garment_type,
                "seed": seed,
                "ep_idx": ep_idx,
                "return": 0.0,
                "length": len(trajectory),
                "success": is_success,
                "split": "Unseen" if "Unseen" in garment_name else "Seen",
                "rollout_type": rollout_type,
            }
            self._episode_done_queue.put(ep_result)
            # Early-stop-on-success: emit phantom discards so the orchestrator's
            # processed/total_expected accounting still converges.
            if is_success and (hard_mining_mode or replay_mode) and max_attempts > 1:
                unused = max_attempts - (attempt + 1)
                self._emit_phantom_discards(
                    garment_name, garment_type, seed, ep_idx,
                    rollout_type, unused,
                )

        # Physics-state snapshots — only useful with dataset collection.
        if self.video_dir and self.save_pkl:
            # Save failure state snapshot (if detected and received from thin client)
            # Only save if the episode actually failed — value drop detection
            # captures the position early, but recovery is possible.
            if failure_snapshot and failure_frame is not None and not is_success:
                self._save_failure_state(
                    failure_snapshot, failure_frame, trajectory,
                    ep_idx, seed, base, garment_info,
                    garment_name, garment_type,
                    augmentation_info=augmentation_info,
                )

            # Save success state for later replay (if episode succeeded).
            # Two cases:
            # 1. Hard mining success: use the initial restore state (start of episode)
            # 2. Adaptive: early snapshot (step 5) with P(save) = 1 - SR (FR-proportional)
            saved_via_snapshot = False
            if is_success and not replay_mode:
                if hard_mining_mode and restore_data:
                    # Case 1: hard mining success — save initial restore state + all actions
                    initial_snapshot = {
                        k: restore_data[k] for k in (
                            "garment_points", "garment_velocities",
                            "garment_pos", "garment_ori",
                            "left_joint_pos", "right_joint_pos",
                        ) if k in restore_data
                    }
                    if initial_snapshot:
                        all_actions = np.array(
                            [f["action"] for f in trajectory],
                            dtype=np.float32,
                        )
                        self._save_success_state(
                            initial_snapshot, 0, all_actions,
                            trajectory, ep_idx, seed, base, garment_info,
                            garment_name, garment_type,
                            augmentation_info=augmentation_info,
                        )
                        saved_via_snapshot = True

                # Case 2: adaptive saving using early snapshot (step 5)
                # Save with P = 1 - SR, so rare successes are always saved
                if not saved_via_snapshot and early_snapshot:
                    from lehome_solution.eval.dataset_writer import _GTYPE_TO_KEY
                    type_key = _GTYPE_TO_KEY.get(garment_type, garment_type)
                    sr_by_type = self._success_rates.get("by_type", {})
                    sr_by_garment = self._success_rates.get("by_garment", {})
                    # Use garment-specific SR if available, else type-level, else 0.5
                    sr = sr_by_garment.get(garment_name, sr_by_type.get(type_key, 0.5))
                    save_prob = 1.0 if sr < 0.5 else (1.0 - sr)
                    if np.random.random() < save_prob:
                        early_frame = self.EARLY_SNAPSHOT_FRAME
                        actions_from_early = np.array(
                            [f["action"] for f in trajectory[early_frame:]],
                            dtype=np.float32,
                        )
                        self._save_success_state(
                            early_snapshot, early_frame, actions_from_early,
                            trajectory, ep_idx, seed, base, garment_info,
                            garment_name, garment_type,
                            augmentation_info=augmentation_info,
                        )
                        logger.info(
                            f"{self.label} ep{ep_idx}: adaptive save (SR={sr:.1%}, "
                            f"P(save)={save_prob:.1%}) — saved early snapshot"
                        )
                    else:
                        logger.debug(
                            f"{self.label} ep{ep_idx}: adaptive skip (SR={sr:.1%}, "
                            f"P(save)={save_prob:.1%})"
                        )

            # Save semi-success state: ALL failed episodes that reached the
            # first checkpoint (no FR-proportional gating).
            # Track whether this branch fired so the near-success branch below
            # does NOT double-save (both write to physics_states/semi_success/).
            saved_checkpoint_semi_success = False
            if not is_success and not replay_mode and not hard_mining_mode and early_snapshot:
                checkpoint_frame = self._find_first_checkpoint_frame(
                    trajectory, garment_name
                )
                if checkpoint_frame is not None and checkpoint_frame > self.EARLY_SNAPSHOT_FRAME:
                    early_frame = self.EARLY_SNAPSHOT_FRAME
                    # Save actions only up to the checkpoint frame
                    actions_to_cp = np.array(
                        [f["action"] for f in trajectory[early_frame:checkpoint_frame]],
                        dtype=np.float32,
                    )
                    self._save_success_state(
                        early_snapshot, early_frame, actions_to_cp,
                        trajectory, ep_idx, seed, base, garment_info,
                        garment_name, garment_type,
                        augmentation_info=augmentation_info,
                        states_subdir="semi_success",
                    )
                    saved_checkpoint_semi_success = True
                    logger.info(
                        f"{self.label} ep{ep_idx}: semi-success save "
                        f"(cp_frame={checkpoint_frame}) "
                        f"— saved {len(actions_to_cp)} actions to checkpoint"
                    )

            # Save near-success state: failed episodes where at some point
            # success_pred > 0.9 and completion_pred > 0.7. Skip if the
            # checkpoint branch above already saved a semi-success snapshot
            # for this episode (both write to physics_states/semi_success/
            # from the same early_snapshot — double-save skews FR-proportional
            # sampling of the pool).
            if (not is_success and not replay_mode and not hard_mining_mode
                    and not saved_checkpoint_semi_success
                    and early_snapshot and near_success_frame is not None
                    and near_success_frame > self.EARLY_SNAPSHOT_FRAME):
                early_frame = self.EARLY_SNAPSHOT_FRAME
                # Save actions from early snapshot up to the near-success frame
                actions_to_ns = np.array(
                    [f["action"] for f in trajectory[early_frame:near_success_frame]],
                    dtype=np.float32,
                )
                self._save_success_state(
                    early_snapshot, early_frame, actions_to_ns,
                    trajectory, ep_idx, seed, base, garment_info,
                    garment_name, garment_type,
                    augmentation_info=augmentation_info,
                    states_subdir="semi_success",
                )
                ns_sp = "?"
                if near_success_frame < len(trajectory):
                    ns_sp = f"{trajectory[near_success_frame].get('success_pred', 0.0):.3f}"
                logger.info(
                    f"{self.label} ep{ep_idx}: near-success save "
                    f"(ns_frame={near_success_frame}, ema_sp={ns_sp}) "
                    f"— saved {len(actions_to_ns)} actions"
                )

    @staticmethod
    def _find_first_checkpoint_frame(
        trajectory: list[dict], garment_name: str
    ) -> int | None:
        """Find the first frame where the first checkpoint conditions are met.

        Uses GARMENT_CHECKPOINTS to determine which conditions constitute the
        first checkpoint. Returns the frame index, or None if never reached.
        """
        from lehome_solution.eval.dataset_writer import (
            GARMENT_CHECKPOINTS, _garment_type_from_name,
        )
        gtype = _garment_type_from_name(garment_name) if garment_name else None
        groups = GARMENT_CHECKPOINTS.get(gtype) if gtype else None
        if groups is None or len(groups) < 2:
            return None
        first_cp_group = groups[1]  # groups[0] is free/spread
        for i, frame in enumerate(trajectory):
            cs = frame.get("check_status")
            if cs is not None and len(cs) > 0:
                if all(cs[idx] > 0.5 for idx in first_cp_group if idx < len(cs)):
                    return i
        return None

    def _save_failure_state(
        self,
        snapshot: dict,
        failure_frame: int,
        trajectory: list[dict],
        ep_idx: int,
        seed: int,
        base_name: str,
        garment_info: dict | None,
        save_garment_name: str | None = None,
        save_garment_type: str | None = None,
        augmentation_info: dict | None = None,
    ):
        """Write a self-contained failure state file for later restoration."""
        from lehome_solution.eval.physics_states import save_physics_state_npz

        # Extract top_rgb at failure moment for visual reference
        top_rgb = None
        if failure_frame < len(trajectory):
            top_rgb = trajectory[failure_frame].get("observation.images.top_rgb")
            if top_rgb is not None:
                top_rgb = np.asarray(top_rgb)

        save_physics_state_npz(
            save_dir=Path(self.video_dir) / "physics_states" / "failure",
            basename=base_name,
            seed=seed,
            ep_idx=ep_idx,
            snapshot=snapshot,
            garment_name=save_garment_name or self.garment_name,
            garment_type=save_garment_type or self.garment_type,
            failure_frame=failure_frame,
            n_frames=len(trajectory),
            garment_info=garment_info,
            augmentation_info=augmentation_info,
            top_rgb=top_rgb,
            min_actions=0,
            label=self.label,
        )

    # Minimum actions for a success state to be worth saving/replaying (2s at 30Hz)
    _MIN_SUCCESS_STATE_ACTIONS = 60

    def _save_success_state(
        self,
        snapshot: dict,
        snapshot_frame: int,
        actions: np.ndarray,
        trajectory: list[dict],
        ep_idx: int,
        seed: int,
        base_name: str,
        garment_info: dict | None,
        save_garment_name: str | None = None,
        save_garment_type: str | None = None,
        augmentation_info: dict | None = None,
        states_subdir: str = "success",
    ):
        """Write a state file for later replay with different augmentations."""
        from lehome_solution.eval.physics_states import save_physics_state_npz

        save_physics_state_npz(
            save_dir=Path(self.video_dir) / "physics_states" / states_subdir,
            basename=base_name,
            seed=seed,
            ep_idx=ep_idx,
            snapshot=snapshot,
            garment_name=save_garment_name or self.garment_name,
            garment_type=save_garment_type or self.garment_type,
            actions=actions,
            snapshot_frame=snapshot_frame,
            n_frames=len(trajectory),
            garment_info=garment_info,
            augmentation_info=augmentation_info,
            min_actions=self._MIN_SUCCESS_STATE_ACTIONS,
            label=self.label,
        )


# ---------------------------------------------------------------------------
# run_worker_session  (streaming: N workers share one task queue, each runs
#                      one Isaac Sim and pulls tasks until the queue is empty)
# ---------------------------------------------------------------------------

MAX_RESTARTS = 3  # max times to restart a worker session (NaN, crash, or hang)


def run_worker_session(
    task_queue: "queue.Queue",
    port: int,
    max_steps: int = 600,
    worker_id: int = 0,
    server_url: str | None = None,
    video_dir: str | None = None,
    camera_width: int | None = None,
    camera_height: int | None = None,
    top_camera_width: int | None = None,
    top_camera_height: int | None = None,
    enable_depth: bool = False,
    prior_file: str | None = None,
    noise_temperature: float = 1.0,
    explore_noise_prob: float = 0.0,
    explore_noise_scale: float = 0.0,
    aug_config: dict | None = None,
    episode_done_queue: "queue.Queue | None" = None,
    success_rates_file: str | None = None,
    default_rollout_type: str | None = None,
    storage_width: int | None = None,
    storage_height: int | None = None,
    per_garment_type_inference_config: dict | None = None,
    save_pkl: bool = True,
) -> dict:
    """Run episodes from a shared task queue in a single Isaac Sim process.

    Each task in the queue is a dict:
        {garment_name, garment_type, seed, restore_data (optional)}
    The worker starts Isaac Sim with the first task, then switches garments
    in-place for subsequent tasks.  Runs until the queue is empty.

    If Isaac Sim crashes, hangs, or produces NaN state, the session is
    killed and restarted with remaining tasks (up to MAX_RESTARTS times).

    Returns dict with keys:
        episodes: list of episode result dicts
        hung: bool
        server_dead: bool
    """
    all_episodes: list[dict] = []
    label = f"[W{worker_id}]"

    for restart_attempt in range(MAX_RESTARTS + 1):
        result = _run_worker_session_once(
            task_queue=task_queue,
            port=port,
            max_steps=max_steps,
            worker_id=worker_id,
            server_url=server_url,
            video_dir=video_dir,
            camera_width=camera_width,
            camera_height=camera_height,
            top_camera_width=top_camera_width,
            top_camera_height=top_camera_height,
            enable_depth=enable_depth,
            prior_file=prior_file,
            noise_temperature=noise_temperature,
            explore_noise_prob=explore_noise_prob,
            explore_noise_scale=explore_noise_scale,
            aug_config=aug_config,
            episode_done_queue=episode_done_queue,
            success_rates_file=success_rates_file,
            default_rollout_type=default_rollout_type,
            storage_width=storage_width,
            storage_height=storage_height,
            per_garment_type_inference_config=per_garment_type_inference_config,
            save_pkl=save_pkl,
        )
        all_episodes.extend(result["episodes"])

        if not result.get("needs_restart"):
            # Normal exit — done
            return {
                "episodes": all_episodes,
                "hung": False,
                "server_dead": result["server_dead"],
            }

        # NaN, crash, or hang detected — loop to restart if tasks remain
        remaining = task_queue.qsize()
        if remaining == 0:
            logger.info(f"{label} restart: no tasks remaining, done")
            return {"episodes": all_episodes, "hung": False, "server_dead": False}
        logger.warning(
            f"{label} restart {restart_attempt + 1}/{MAX_RESTARTS}: "
            f"{result.get('restart_reason', 'unknown')} — "
            f"{remaining} tasks remaining, restarting Isaac Sim"
        )

    remaining = task_queue.qsize()
    logger.error(
        f"{label} Exhausted {MAX_RESTARTS} restarts, "
        f"{remaining} tasks still remaining"
    )
    return {"episodes": all_episodes, "hung": True, "server_dead": False}


def _run_worker_session_once(
    task_queue: "queue.Queue",
    port: int,
    max_steps: int = 600,
    worker_id: int = 0,
    server_url: str | None = None,
    video_dir: str | None = None,
    camera_width: int | None = None,
    camera_height: int | None = None,
    top_camera_width: int | None = None,
    top_camera_height: int | None = None,
    enable_depth: bool = False,
    prior_file: str | None = None,
    noise_temperature: float = 1.0,
    explore_noise_prob: float = 0.0,
    explore_noise_scale: float = 0.0,
    aug_config: dict | None = None,
    episode_done_queue: "queue.Queue | None" = None,
    success_rates_file: str | None = None,
    default_rollout_type: str | None = None,
    storage_width: int | None = None,
    storage_height: int | None = None,
    per_garment_type_inference_config: dict | None = None,
    save_pkl: bool = True,
) -> dict:
    """Single attempt at running a worker session. Returns dict with extra
    key 'needs_restart' (bool) indicating whether the caller should restart."""
    import queue as _q

    if server_url is None:
        server_url = f"ws://localhost:{port}"
    session_id = f"w{worker_id}"
    proxy_url = f"http://localhost:{PROXY_BASE_PORT + worker_id}"

    # Pop first task — needed for Isaac Sim startup (garment list file)
    try:
        first_task = task_queue.get_nowait()
    except _q.Empty:
        return {"episodes": [], "hung": False, "server_dead": False, "needs_restart": False}

    label = f"[W{worker_id}]"
    logger.info(f"{label} starting — first garment: {first_task['garment_name']}")

    # Per-sim log file
    sim_log_path: Path | None = None
    if video_dir:
        sim_log_path = _log_path(video_dir, logcfg.ISAAC_SIM, f"isaac_sim_w{worker_id}_session.log")

    # Liveness check
    resp = _signal_server(server_url, {"type": "ping"}, recv_timeout=60)
    if resp is None:
        logger.error(f"{label} policy server unresponsive (ping failed)")
        return {"episodes": [], "hung": True, "server_dead": True, "needs_restart": False}

    proxy = _ProxyServer(
        worker_id=worker_id,
        session_id=session_id,
        server_url=server_url,
        video_dir=video_dir,
        label=label,
        task_queue=task_queue,
        first_task=first_task,
        max_steps=max_steps,
        prior_file=prior_file,
        noise_temperature=noise_temperature,
        explore_noise_prob=explore_noise_prob,
        explore_noise_scale=explore_noise_scale,
        episode_done_queue=episode_done_queue,
        success_rates_file=success_rates_file,
        default_rollout_type=default_rollout_type,
        storage_width=storage_width,
        storage_height=storage_height,
        per_garment_type_inference_config=per_garment_type_inference_config,
        save_pkl=save_pkl,
    )
    proxy.start()
    if not proxy.wait_ready(timeout=30):
        logger.error(f"{label} Proxy failed to become ready within 30s")
        return {"episodes": [], "hung": True, "server_dead": False, "needs_restart": False}

    sim_proc = None
    hung = False
    output_lines: list[str] = []

    try:
        first_gt = first_task["garment_type"]
        first_gname = first_task["garment_name"]

        eval_env = {
            **os.environ,
            "LEHOME_DISABLE_KEYBOARD": "1",
            "PYTHONUNBUFFERED": "1",
            "LEHOME_NO_DEPTH": "0" if enable_depth else "1",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "LEHOME_REMOTE_URL": proxy_url,
            "LEHOME_WORKER_LABEL": f"W{worker_id}",
        }
        if aug_config is not None and aug_config.get("enabled", False):
            import json as _json_env
            eval_env["LEHOME_GARMENT_AUGMENTATION"] = "1"
            eval_env["LEHOME_AUG_CONFIG"] = _json_env.dumps(aug_config)
        if camera_width is not None and camera_height is not None:
            eval_env["LEHOME_CAMERA_WIDTH"] = str(camera_width)
            eval_env["LEHOME_CAMERA_HEIGHT"] = str(camera_height)
        if top_camera_width is not None and top_camera_height is not None:
            eval_env["LEHOME_TOP_CAMERA_WIDTH"] = str(top_camera_width)
            eval_env["LEHOME_TOP_CAMERA_HEIGHT"] = str(top_camera_height)

        sim_cmd = [
            str(LEHOME_VENV_PYTHON), "-u", "-m", "scripts.eval",
            "--policy_type", "remote",
            "--remote_url", proxy_url,
            "--garment_type", first_gt,
            "--garment_name", first_gname,
            "--num_episodes", "9999",  # proxy controls shutdown; this is just a hint
            "--max_steps", str(max_steps),
            "--headless",
            "--enable_cameras",
            "--device", "cpu",
            "--seed", str(first_task["seed"]),
            "--session_id", session_id,
        ]
        if camera_width is not None and camera_height is not None:
            sim_cmd += ["--camera_width", str(camera_width), "--camera_height", str(camera_height)]

        sim_log_fh = open(sim_log_path, "a") if sim_log_path else None  # append: survives NaN restarts
        sim_proc = subprocess.Popen(
            sim_cmd,
            cwd=str(LEHOME_CHALLENGE_DIR),
            env=eval_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        start_wait = time.time()
        while time.time() - start_wait < GARMENT_LIST_WAIT_TIMEOUT:
            ready, _, _ = select.select([sim_proc.stdout], [], [], 1.0)
            if ready:
                line = sim_proc.stdout.readline()
                if not line:
                    break
                if sim_log_fh:
                    sim_log_fh.write(line)
                    sim_log_fh.flush()
                if "Loaded" in line and "garments" in line:
                    logger.info(f"{label} Isaac Sim loaded garment")
                    break
                if "Evaluating:" in line:
                    break

        # Wait for simulation initialization
        start_init = time.time()
        while time.time() - start_init < SIM_INIT_TIMEOUT:
            ready, _, _ = select.select([sim_proc.stdout], [], [], SIM_INIT_TIMEOUT)
            if not ready:
                break
            line = sim_proc.stdout.readline()
            if not line:
                break
            if sim_log_fh:
                sim_log_fh.write(line)
                sim_log_fh.flush()
            output_lines.append(line)
            if "Starting evaluation" in line or "Evaluating:" in line:
                logger.info(f"{label} simulation initialized")
                break

        # Monitor Isaac Sim subprocess until it exits (proxy sends shutdown when queue empty)
        abort_loop_count = 0

        try:
            while True:
                ready, _, _ = select.select([sim_proc.stdout], [], [], NO_OUTPUT_TIMEOUT)
                if not ready:
                    logger.warning(f"{label} no output for {NO_OUTPUT_TIMEOUT}s, killing")
                    _kill_pg(sim_proc.pid)
                    hung = True
                    break

                line = sim_proc.stdout.readline()
                if not line:
                    break

                if sim_log_fh:
                    sim_log_fh.write(line)
                    sim_log_fh.flush()
                output_lines.append(line)

                if "_abort_signal_handle_callback" in line or "orchestrator.py" in line:
                    abort_loop_count += 1
                    if abort_loop_count > ABORT_LOOP_THRESHOLD:
                        logger.warning(f"{label} abort loop detected, killing")
                        _kill_pg(sim_proc.pid)
                        hung = True
                        break
                else:
                    abort_loop_count = 0

                if "Switching garment:" in line or "Garment switched" in line:
                    logger.info(f"{label} {line.strip()}")
                if "Episode " in line and "Return=" in line:
                    logger.info(f"{label} {line.strip()}")
                if "All tasks done" in line:
                    logger.info(f"{label} all tasks done, waiting for shutdown")

            if not hung:
                try:
                    sim_proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    _kill_pg(sim_proc.pid)
                    hung = True
        except Exception as e:
            logger.error(f"{label} monitoring error: {e}")
            _kill_pg(sim_proc.pid)
            hung = True
        finally:
            if sim_log_fh:
                sim_log_fh.close()

    except Exception as e:
        logger.error(f"{label} error before/during sim start: {e}", exc_info=True)
        if sim_proc is not None:
            _kill_pg(sim_proc.pid)
        hung = True
    finally:
        proxy.join()
        if proxy._error:
            logger.warning(f"{label} Proxy thread error: {proxy._error}")

    # Parse self-describing episode results from stdout.
    # Format: "Episode N (GarmentName): Return=X, Length=Y, Success=Z"
    output = "".join(output_lines)
    matches = re.findall(
        r"Episode \d+ \(([^)]+)\): Return=([\d.-]+), Length=(\d+), Success=(\w+)",
        output,
    )
    episodes = [
        {
            "return": float(m[1]),
            "length": int(m[2]),
            "success": m[3] == "True",
            "garment": m[0],
            "garment_type": garment_name_to_type(m[0]),
            "seed": 0,  # seed not in stdout; PKL has the real data
            "ep_idx": i,
            "split": "Unseen" if "Unseen" in m[0] else "Seen",
        }
        for i, m in enumerate(matches)
    ]

    nan_restart = proxy._nan_detected
    crash_restart = proxy._crash_detected and not task_queue.empty()
    hung_restart = hung and not task_queue.empty()
    needs_restart = nan_restart or crash_restart or hung_restart
    if nan_restart:
        restart_reason = "NaN physics"
    elif crash_restart:
        restart_reason = "crash (<10 frames)"
    elif hung_restart:
        restart_reason = "hang (no output timeout)"
    else:
        restart_reason = None
    status = restart_reason or ("done" if not hung else "HUNG (no tasks left)")
    logger.info(f"{label} {len(episodes)} episodes {status}")
    for ep in episodes:
        logger.info(f"  {ep['garment']} Return={ep['return']:.1f} Success={ep['success']}")

    return {
        "episodes": episodes, "hung": hung, "server_dead": False,
        "needs_restart": needs_restart, "restart_reason": restart_reason,
    }
