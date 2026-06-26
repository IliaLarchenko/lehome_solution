"""
Policy server for LeHome Challenge.

Serves a single shared model over WebSocket to multiple concurrent workers.
Each worker gets its own session with independent episode state (action chunks,
inpainting). Video recording and saving are done by workers; server only
returns model predictions.

Server-side batching: incoming inference requests are accumulated in a queue.
When the GPU finishes the previous batch, all pending requests are batched
into a single forward pass.  This gives ~Nx throughput with N concurrent workers.

Protocol (policy server, port 8000):
    {"type": "ping"}
    Response: {"status": "ok"}

    {"type": "infer_chunk", ...obs..., "initial_actions": [...]}
    Response: {"actions": [...], "next_initial_actions": [...], ...}
"""

import asyncio
import dataclasses
import json
import logging
import os
import queue
import socket
import threading
import time as _time
from pathlib import Path

import jax
import numpy as np
import tyro
import websockets

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.8')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

from openpi.policies import policy as _policy

from lehome_solution.policies import policy_config as _policy_config
from lehome_solution.shared.eval_wrapper import LeHomePolicyWrapper, LeHomeWrapperConfig


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    # Registry config name (e.g. pi_modified_real_bc) or a training-yaml
    # recipe path (e.g. configs/train_real_bc.yaml) — both rebuild the same
    # model architecture; the registry entry carries the inference contract.
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the LeHome policy server."""

    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # LeHome Wrapper execution parameters.
    # If None, loaded from checkpoint assets/inference_config.json (if exists).
    # Otherwise falls back to DEFAULT_CONFIG from inference_optimization.py.
    actions_to_execute: int | None = None
    actions_to_keep: int | None = None
    execute_in_n_steps: int | None = None
    num_steps: int | None = None
    # Best-of-N rollout sampling. N=1 → no best-of-N (default). N>1 replicates
    # each inference request into N candidates with independent denoising
    # noise, picks the argmax by the WM-flow Δsuccess score averaged over the
    # CFG cond/uncond passes. Adds ~N× action-expert cost; prefix / VLM cost
    # unchanged.
    num_rollout_candidates: int = 1

    # Enable integrated success head on the model config at serve time.
    integrated_value: bool = True

    # Attention backend. None → taken from the train config (inference contract);
    # set on the CLI to override. Must match training or actions are wrong.
    use_flash_attention: bool | None = None
    use_xsa: bool | None = None


def create_policy(args: Args, num_steps: int) -> _policy.Policy:
    from lehome_solution.training import attention_backend as _attention_backend
    from lehome_solution.training.yaml_train_config import resolve_train_config

    train_config = resolve_train_config(args.policy.config)

    # Attention backend from the train config unless the CLI overrides.
    use_flash = (
        train_config.use_flash_attention
        if args.use_flash_attention is None else args.use_flash_attention
    )
    use_xsa = train_config.use_xsa if args.use_xsa is None else args.use_xsa
    backend = _attention_backend.install(
        use_flash_attention=use_flash,
        use_xsa=use_xsa,
    )
    logging.info(
        f"Attention backend: {backend} "
        f"(use_flash_attention={use_flash}, use_xsa={use_xsa})"
    )

    sample_kwargs = {"num_steps": num_steps}
    if args.integrated_value:
        import dataclasses as _dc
        train_config = _dc.replace(
            train_config,
            model=_dc.replace(train_config.model, use_success_head=True),
        )
        logging.info("Integrated success head enabled (use_success_head=True override)")
    return _policy_config.create_trained_policy(
        train_config,
        args.policy.dir,
        sample_kwargs=sample_kwargs
    )


def _numpy_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8, np.uint8)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_serializable(v) for v in obj]
    return obj


def _decode_observation(data: dict) -> dict:
    import base64
    obs = {}
    for k, v in data.items():
        if isinstance(v, dict) and "base64" in v:
            raw = base64.b64decode(v["base64"])
            arr = np.frombuffer(raw, dtype=np.dtype(v["dtype"]))
            arr = arr.reshape(v["shape"])
            obs[k] = arr
        elif isinstance(v, list):
            obs[k] = np.array(v)
        else:
            obs[k] = v
    return obs


# ---------------------------------------------------------------------------
# InferenceBatcher: queue-based batching for GPU inference
# ---------------------------------------------------------------------------

class InferenceBatcher:
    """Accumulates inference requests and batches them for GPU efficiency.

    Architecture: a dedicated inference thread runs a loop:
      1. Block-wait for the first request
      2. Short drain window (5ms) to collect concurrent requests
      3. Batch all pending requests into one GPU forward pass
      4. Dispatch individual results back to waiting async handlers

    While the GPU is busy processing a batch, new requests accumulate
    in the queue and get batched on the next iteration.
    """

    def __init__(self, wrapper: LeHomePolicyWrapper, drain_ms: float = 5.0):
        self._wrapper = wrapper
        self._drain_s = drain_ms / 1000.0
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="inference-batcher"
        )
        self._thread.start()

    def submit(self, obs: dict, initial_actions, inference_config: dict | None,
               loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        """Submit an inference request. Returns an asyncio Future with the result."""
        future = loop.create_future()
        self._queue.put((obs, initial_actions, inference_config, future, loop))
        return future

    def _inference_loop(self):
        while True:
            # Block until first request arrives
            first = self._queue.get()
            if first is None:
                break

            # Short drain window to collect concurrent requests
            batch = [first]
            deadline = _time.monotonic() + self._drain_s
            while _time.monotonic() < deadline:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    if len(batch) >= 6:
                        break  # full batch, don't wait
                    _time.sleep(0.001)

            # Final non-blocking drain
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            batch_size = len(batch)
            try:
                # Always use batched path — JAX caches compilations per batch size,
                # so sizes 1-8 compile once during warmup then hit cache forever.
                requests = [(obs, ia, cfg) for obs, ia, cfg, _, _ in batch]
                results = self._wrapper.infer_chunk_batched(requests)
                for (_, _, _, future, loop), result in zip(batch, results):
                    loop.call_soon_threadsafe(future.set_result, result)
                if batch_size > 1:
                    logging.info(f"Batched inference: {batch_size} requests")
            except Exception as e:
                logging.error(f"Batch inference error (B={batch_size}): {e}", exc_info=True)
                for _, _, _, future, loop in batch:
                    loop.call_soon_threadsafe(future.set_exception, e)


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_client(websocket, batcher: InferenceBatcher):
    """Handle a single WebSocket client connection."""
    logging.info(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.pop("type", None)
                session_id = data.pop("session_id", "default")

                # --- Liveness probe ---
                if msg_type == "ping":
                    await websocket.send(json.dumps({"status": "ok"}))
                    continue

                # --- Stateless chunk inference (primary path) ---
                if msg_type == "infer_chunk":
                    initial_actions_raw = data.pop("initial_actions", None)
                    inference_config = data.pop("inference_config", None)
                    obs = _decode_observation(data)
                    initial_actions = (
                        np.array(initial_actions_raw, dtype=np.float32)
                        if initial_actions_raw is not None else None
                    )

                    # Submit to batcher — awaits until GPU batch completes
                    loop = asyncio.get_event_loop()
                    chunk, next_initial, value_pred, checkpoint_pred, effective_config = await batcher.submit(
                        obs, initial_actions, inference_config, loop
                    )

                    logging.info(
                        f"[{session_id}] infer_chunk: chunk_len={len(chunk)} "
                        f"inpainting={'yes' if initial_actions is not None else 'no'}"
                        f"{f' value_pred={value_pred:.4f}' if value_pred is not None else ''}"
                    )
                    response = {
                        "actions": _numpy_to_serializable(chunk),
                        "next_initial_actions": (
                            _numpy_to_serializable(next_initial) if next_initial is not None else None
                        ),
                        "execute_in_n_steps": effective_config["execute_in_n_steps"],
                        "actions_to_keep": effective_config["actions_to_keep"],
                    }
                    if value_pred is not None:
                        response["success_pred"] = value_pred
                    if checkpoint_pred is not None:
                        response["checkpoint_pred"] = checkpoint_pred
                    gt_pred = effective_config.get("garment_type_pred")
                    if gt_pred is not None:
                        response["garment_type_pred"] = gt_pred
                    comp_pred = effective_config.get("completion_pred")
                    if comp_pred is not None:
                        response["completion_pred"] = comp_pred
                    ttc_pred = effective_config.get("ttc_pred")
                    if ttc_pred is not None:
                        response["ttc_pred"] = ttc_pred
                    # Keypoint-distance head (Head 1) — full 21-wide vector.
                    kpt = effective_config.get("keypoint_distances")
                    if kpt is not None:
                        response["keypoint_distances"] = kpt
                    # WM-flow head (Head 3) — cond + uncond monitoring.
                    for k in (
                        "success_cond", "completion_cond", "keypoint_cond",
                        "success_uncond", "completion_uncond", "keypoint_uncond",
                    ):
                        v = effective_config.get(f"wm_flow_{k}")
                        if v is not None:
                            response[f"wm_flow_{k}"] = v
                    # Best-of-N candidate-spread diagnostics (only present
                    # when num_rollout_candidates > 1).
                    for k in (
                        "best_of_n_score_chosen", "best_of_n_score_mean",
                        "best_of_n_score_min", "best_of_n_score_max",
                        "best_of_n_score_std", "best_of_n_score_spread",
                        "best_of_n_n_valid",
                    ):
                        v = effective_config.get(k)
                        if v is not None:
                            response[k] = v
                    await websocket.send(json.dumps(response))
                    continue

                # Unknown message type
                await websocket.send(json.dumps({"error": f"unknown message type: {msg_type}"}))

            except Exception as e:
                logging.error(f"Error processing request: {e}", exc_info=True)
                await websocket.send(json.dumps({"error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        logging.info(f"Client disconnected: {websocket.remote_address}")


# ---------------------------------------------------------------------------
# Config loading & main
# ---------------------------------------------------------------------------

def _load_inference_config(args: Args) -> dict:
    """Load inference config from checkpoint assets or use defaults.

    Priority: CLI args (if set) > checkpoint assets/inference_config.json > DEFAULT_CONFIG.

    The checkpoint may store per-garment-type configs (new format) or a flat
    config (old format). For the server default we pick the first garment type's
    config — the eval_worker overrides per-request anyway.
    """
    from lehome_solution.eval.inference_optimization import DEFAULT_CONFIG

    # Start with defaults
    config = dict(DEFAULT_CONFIG)
    config["execute_in_n_steps"] = max(1, int(config["k_execute"] * config["actions_to_execute"]))

    # Try loading from checkpoint assets
    if isinstance(args.policy, Checkpoint) and args.policy.dir:
        config_path = Path(args.policy.dir) / "assets" / "inference_config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved = json.load(f)
            # New per-garment-type format: use first garment type as server default
            if "per_garment_type" in saved:
                first_gt = next(iter(saved["per_garment_type"]))
                gt_cfg = saved["per_garment_type"][first_gt]
                config.update({k: gt_cfg[k] for k in config if k in gt_cfg})
                logging.info("Loaded per-garment-type inference config from %s "
                             "(using %s as server default): %s",
                             config_path, first_gt, gt_cfg)
            else:
                # Old flat format (backward compat)
                config.update({k: saved[k] for k in config if k in saved})
                logging.info("Loaded inference config from %s: %s", config_path, saved)
        else:
            logging.warning("No inference_config.json in %s/assets — using defaults: %s",
                            args.policy.dir, config)

    # CLI overrides (non-None values)
    if args.actions_to_execute is not None:
        config["actions_to_execute"] = args.actions_to_execute
    if args.actions_to_keep is not None:
        config["actions_to_keep"] = args.actions_to_keep
    if args.execute_in_n_steps is not None:
        config["execute_in_n_steps"] = args.execute_in_n_steps
    if args.num_steps is not None:
        config["num_steps"] = args.num_steps

    return config


def main(args: Args) -> None:
    inf_cfg = _load_inference_config(args)

    logging.info("Loading policy...")
    policy = create_policy(args, num_steps=inf_cfg["num_steps"])

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    wrapper_config = LeHomeWrapperConfig(
        actions_to_execute=inf_cfg["actions_to_execute"],
        actions_to_keep=inf_cfg["actions_to_keep"],
        execute_in_n_steps=inf_cfg["execute_in_n_steps"],
        num_steps=inf_cfg["num_steps"],
        num_rollout_candidates=max(1, int(args.num_rollout_candidates)),
    )

    wrapper = LeHomePolicyWrapper(
        policy,
        config=wrapper_config,
    )

    # Create batcher (starts inference thread)
    batcher = InferenceBatcher(wrapper)

    logging.info(
        f"Config: execute={wrapper_config.actions_to_execute}, "
        f"keep={wrapper_config.actions_to_keep}, "
        f"steps={wrapper_config.execute_in_n_steps}, "
        f"num_steps={wrapper_config.num_steps}, "
        f"best_of_n={wrapper_config.num_rollout_candidates}"
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Starting server (host={hostname}, ip={local_ip}, port={args.port})")

    async def serve():
        async with websockets.serve(
            lambda ws: handle_client(ws, batcher),
            "0.0.0.0",
            args.port,
            max_size=100 * 1024 * 1024,
            ping_interval=5,
            ping_timeout=5,
        ):
            logging.info(f"Policy server listening on ws://0.0.0.0:{args.port}")
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
