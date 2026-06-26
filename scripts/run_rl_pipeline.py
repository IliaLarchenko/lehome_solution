#!/usr/bin/env python3
"""Automated RL pipeline: warm-up BC training + iterative rollout/train cycles.

This is the main entry point for experiments. It orchestrates:
  - scripts/train.py (BC/RL training with dynamic config)
  - scripts/run_eval.py (rollout collection)
  - scripts/recompute_advantages.py (advantage recomputation)
  - HuggingFace Hub uploads (checkpoints + datasets)

All progress is logged to a single wandb run. Pipeline state is saved to
pipeline_state.json for resumability.

Usage:
    # Full pipeline (sim-round RL config)
    uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml

    # Resume from crash
    uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml --resume

    # Single-phase modes (run one step, update state, exit)
    uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml --train_only
    uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml --rollout_only
    uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml --advantages_only

    # Real-round sim-correction collection (rollout worker against a pinned checkpoint)
    uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim_to_real.yaml \
        --rollout_worker --rollout_checkpoint_path <.../_hf_checkpoints/step_N>

    # Override config values
    uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml \
        --num_iterations 20 --steps_per_iteration 2000
"""

import os

# Pipeline orchestrator only — no GPU needed.
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import dataclasses
import json
import logging
import pickle
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Raise RLIMIT_NOFILE soft limit to the hard cap so child processes (train.py,
# rollout workers) inherit it. The torchcodec/fsspec video decoder cache in
# lerobot keeps every opened mp4 file handle alive for the worker's lifetime,
# which trivially exceeds the distro default soft limit of 1024.
_nofile_soft, _nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _nofile_soft < _nofile_hard:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_nofile_hard, _nofile_hard))


from lehome_solution.training.pipeline_config import RLPipelineConfig
from lehome_solution.utils.logging_config import log_path as _log_path
from lehome_solution.utils import logging_config as logcfg

REPO_ROOT = Path(__file__).parent.parent


# Unbuffer stdout/stderr so logs are visible in tmux/background
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subprocess helpers & state persistence (extracted to lehome_solution.utils)
# ---------------------------------------------------------------------------

from lehome_solution.utils.subprocess_manager import (
    run_subprocess as _run_subprocess_impl,
    run_subprocess_passthrough as _run_subprocess_passthrough_impl,
)
from lehome_solution.utils.state_persistence import (
    STATE_FILENAME,
    default_state as _default_state,
    load_state as _load_state,
    save_state as _save_state,
)


def _run_subprocess(cmd, log_path, *, label="", tail_patterns=None):
    return _run_subprocess_impl(cmd, log_path, cwd=REPO_ROOT, label=label, tail_patterns=tail_patterns)


def _run_subprocess_passthrough(cmd, *, label="", log_path=None):
    return _run_subprocess_passthrough_impl(cmd, cwd=REPO_ROOT, label=label, log_path=log_path)


# ---------------------------------------------------------------------------
# HF sync daemon lifecycle
# ---------------------------------------------------------------------------

def _start_hf_sync_daemon(
    cfg: RLPipelineConfig,
    state: dict,
    *,
    mode: str = "pipeline",
) -> "tuple[subprocess.Popen | None, 'HFSyncClient | None']":
    """Start the HF sync daemon subprocess. Returns (process, client) or (None, None).

    Args:
        mode: 'trainer' (polls for rollouts), 'rollout' (polls for checkpoints),
              or 'pipeline' (queue-only, no polling).
    """
    if not cfg.hf_sync_enabled:
        return None, None
    if not cfg.hf_model_repo and not cfg.hf_dataset_repo:
        return None, None

    from lehome_solution.distributed.hf_sync_client import HFSyncClient

    sync_dir = Path(state["checkpoint_dir"]) / "_hf_sync"
    sync_dir.mkdir(parents=True, exist_ok=True)

    # Kill stale daemon if PID file exists
    _cleanup_stale_daemon(sync_dir)

    cmd = [
        "uv", "run", "python", str(REPO_ROOT / "scripts" / "hf_sync_daemon.py"),
        "--sync_dir", str(sync_dir),
        "--parent_pid", str(os.getpid()),
        "--mode", mode,
    ]
    if cfg.hf_model_repo:
        cmd.extend(["--hf_model_repo", cfg.hf_model_repo])
    if cfg.hf_dataset_repo:
        cmd.extend(["--hf_dataset_repo", cfg.hf_dataset_repo])

    # Mode-specific directories for background polling
    if mode == "trainer":
        local_rollout_dir = str(Path(state["checkpoint_dir"]) / "_hf_rollouts")
        cmd.extend(["--local_rollout_dir", local_rollout_dir])
        # Auto-upload checkpoints as they're saved during training
        cmd.extend(["--checkpoint_dir", state["checkpoint_dir"]])
        cmd.extend(["--keep_period", str(cfg.keep_period)])
    elif mode == "rollout":
        local_ckpt_dir = str(Path(state["checkpoint_dir"]) / "_hf_checkpoints")
        cmd.extend(["--local_checkpoint_dir", local_ckpt_dir])

    # Log daemon stderr to file for debugging
    daemon_log_dir = sync_dir / "logs"
    daemon_log_dir.mkdir(parents=True, exist_ok=True)
    daemon_stderr_path = daemon_log_dir / "daemon_stderr.log"
    daemon_stderr_fh = open(daemon_stderr_path, "w")

    logger.info("Starting HF sync daemon (mode=%s): %s", mode, sync_dir)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=daemon_stderr_fh,
    )

    # Wait for PID file (up to 10s)
    pid_file = sync_dir / "daemon.pid"
    for _ in range(50):
        if pid_file.exists():
            break
        # Check if process already exited (crashed)
        if proc.poll() is not None:
            daemon_stderr_fh.close()
            stderr_text = daemon_stderr_path.read_text().strip()
            logger.error("HF sync daemon crashed on startup (rc=%d). Stderr:\n%s",
                        proc.returncode, stderr_text[-2000:] if stderr_text else "(empty)")
            return None, None
        time.sleep(0.2)

    if not pid_file.exists():
        logger.error("HF sync daemon PID file not created after 10s — daemon may have failed. "
                     "Check %s", daemon_stderr_path)
        if proc.poll() is not None:
            daemon_stderr_fh.close()
            stderr_text = daemon_stderr_path.read_text().strip()
            logger.error("Daemon stderr: %s", stderr_text[-2000:] if stderr_text else "(empty)")
        return None, None

    client = HFSyncClient(sync_dir)
    logger.info("HF sync daemon started (PID=%d, mode=%s)", proc.pid, mode)
    return proc, client


def _stop_hf_sync_daemon(proc: "subprocess.Popen | None", sync_client: "HFSyncClient | None"):
    """Submit shutdown request and wait for daemon to exit."""
    if proc is None or sync_client is None:
        return
    if not sync_client.enabled:
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_SHUTDOWN

    try:
        sync_client.submit(OP_SHUTDOWN, {})
        # Wait up to 120s for clean shutdown
        proc.wait(timeout=120)
        logger.info("HF sync daemon stopped cleanly")
    except Exception as e:
        logger.warning("HF sync daemon shutdown error: %s, killing", e)
        proc.kill()
        proc.wait(timeout=5)


def _cleanup_stale_daemon(sync_dir: Path):
    """Kill old daemon if PID file exists."""
    pid_file = sync_dir / "daemon.pid"
    if not pid_file.exists():
        return

    try:
        old_pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return

    if old_pid == os.getpid():
        return

    try:
        os.kill(old_pid, 0)  # Check alive
        logger.info("Killing stale HF sync daemon PID %d", old_pid)
        import signal
        os.kill(old_pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(old_pid, 0)
            os.kill(old_pid, signal.SIGKILL)
        except OSError:
            pass
    except OSError:
        pass

    pid_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_pipeline_config(config_path: str, overrides: dict) -> RLPipelineConfig:
    return RLPipelineConfig.from_yaml(config_path, overrides)


# ---------------------------------------------------------------------------
# Build training config
# ---------------------------------------------------------------------------

def _find_success_rates_file(state: dict) -> str | None:
    """Find the most recent success_rates.json for FR-proportional sampling."""
    from lehome_solution.eval.rollout_strategies import SUCCESS_RATES_FILE
    for ds in reversed(state.get("rl_datasets", [])):
        sr_path = Path(ds["root"]) / SUCCESS_RATES_FILE
        if sr_path.exists():
            return str(sr_path)
    ckpt_dir = state.get("checkpoint_dir")
    if ckpt_dir:
        sr_path = Path(ckpt_dir) / SUCCESS_RATES_FILE
        if sr_path.exists():
            return str(sr_path)
    return None


def build_train_config(pipeline_cfg: RLPipelineConfig, state: dict):
    """Build a TrainConfig with dynamic data_sources from pipeline state."""
    from lehome_solution.training.config import get_config, DataSourceSpec, TrainConfig

    base = get_config(pipeline_cfg.config_name)

    # Find success rates for FR-proportional BC/dagger sampling
    sr_file = _find_success_rates_file(state)

    data_sources = [
        DataSourceSpec(
            root=pipeline_cfg.bc_dataset.root,
            repo_id=pipeline_cfg.bc_dataset.repo_id,
            sampling_share=state["bc_sampling_share"],
            is_bc=True,
            success_rates_file=sr_file,
            use_for_success_labels=pipeline_cfg.bc_dataset.use_for_success_labels,
        ),
    ]

    for i, ds in enumerate(state["rl_datasets"]):
        # RL rollouts always get true labels.
        data_sources.append(DataSourceSpec(
            root=ds["root"],
            repo_id=ds.get("repo_id", f"lehome_rl_{i}"),
            sampling_share=ds["sampling_share"],
        ))

    # DAgger datasets: always included.  Treated like BC for sampling —
    # fixed advantage (no GAE) and FR-proportional garment sampling so harder
    # garments get oversampled.  (This slot also carries the golden dataset
    # when wired via `initial_dagger_datasets`.)
    for i, ds in enumerate(state.get("dagger_datasets", [])):
        data_sources.append(DataSourceSpec(
            root=ds["root"],
            repo_id=ds.get("repo_id", f"lehome_dagger_{i}"),
            sampling_share=ds["sampling_share"],
            is_dagger=True,
            success_rates_file=sr_file,
        ))

    # Real-robot teleop BC datasets. Native real schema (degree units, 20 Hz,
    # real camera columns) — no unit conversion anywhere; uniform sampling,
    # success NaN-masked (no failure signal).
    for i, ds in enumerate(pipeline_cfg.bc_real_datasets):
        data_sources.append(DataSourceSpec(
            root=ds.root,
            repo_id=ds.repo_id or f"lehome_real_{i}",
            sampling_share=ds.sampling_share,
            is_real=True,
        ))

    # Use checkpoint dir for assets (norm stats, FAST tokenizer)
    # so each run is self-contained and doesn't depend on outputs/assets/
    checkpoint_assets = str(Path(state["checkpoint_dir"]) / "assets")
    from lehome_solution.training.config import AssetsConfig
    new_assets = dataclasses.replace(base.data.assets, assets_dir=checkpoint_assets)
    wc = pipeline_cfg.advantage.weighting_clipping
    new_data = dataclasses.replace(
        base.data,
        data_sources=tuple(data_sources),
        assets=new_assets,
        advantage_weighting_clipping=wc,
        bc_advantage=pipeline_cfg.bc_dataset.bc_advantage,
        dagger_advantage=pipeline_cfg.dagger_dataset.dagger_advantage,
        pct_real=pipeline_cfg.pct_real,
    )

    checkpoint_dir_exists = (
        state["checkpoint_dir"] is not None
        and Path(state["checkpoint_dir"]).exists()
    )
    resume = checkpoint_dir_exists and state["current_train_steps"] > 0

    # Apply model-level overrides from pipeline config
    model = base.model
    model = dataclasses.replace(model, train_aug=pipeline_cfg.train_augmentation)
    model = dataclasses.replace(model, use_advantage_embedding=pipeline_cfg.use_advantage_embedding)
    model = dataclasses.replace(model, freeze_vision_backbone=pipeline_cfg.freeze_vision_backbone)

    # Aux-loss weight overrides (None fields are left at the model's dataclass default).
    aux = pipeline_cfg.aux_losses
    aux_overrides = {
        "fast_loss_weight": aux.fast,
        "success_loss_weight": aux.success,
        "checkpoint_loss_weight": aux.checkpoint,
        "ttc_loss_weight": aux.ttc,
        "completion_loss_weight": aux.completion,
        "garment_type_loss_weight": aux.garment_type,
        "keypoint_distance_loss_weight": aux.keypoint_distance,
        "wm_fast_success_loss_weight": aux.wm_fast_success,
        "wm_fast_completion_loss_weight": aux.wm_fast_completion,
        "wm_fast_keypoint_loss_weight": aux.wm_fast_keypoint,
        "wm_flow_success_loss_weight": aux.wm_flow_success,
        "wm_flow_completion_loss_weight": aux.wm_flow_completion,
        "wm_flow_keypoint_loss_weight": aux.wm_flow_keypoint,
    }
    aux_overrides = {k: v for k, v in aux_overrides.items() if v is not None}
    if aux_overrides:
        model = dataclasses.replace(model, **aux_overrides)

    return dataclasses.replace(
        base,
        data=new_data,
        model=model,
        exp_name=pipeline_cfg.exp_name,
        project_name=pipeline_cfg.project_name,
        num_train_steps=state["current_train_steps"],
        resume=resume,
        overwrite=False,
        batch_size=pipeline_cfg.batch_size or base.batch_size,
        save_interval=pipeline_cfg.save_interval,
        keep_period=pipeline_cfg.keep_period,
        wandb_enabled=pipeline_cfg.wandb_enabled,
        assets_base_dir=checkpoint_assets,
        use_flash_attention=pipeline_cfg.use_flash_attention,
        use_xsa=pipeline_cfg.use_xsa,
    )


# ---------------------------------------------------------------------------
# Find checkpoints
# ---------------------------------------------------------------------------

def find_latest_checkpoint_step(checkpoint_dir: str) -> int | None:
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return None
    steps = []
    for d in ckpt_path.iterdir():
        if not d.is_dir() or not (d / "params").is_dir():
            continue
        # Support both "30500" (trainer-created) and "step_30500" (HF-downloaded) formats
        if d.name.isdigit():
            steps.append(int(d.name))
        elif d.name.startswith("step_") and d.name[5:].isdigit():
            steps.append(int(d.name[5:]))
    return max(steps) if steps else None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(pipeline_cfg: RLPipelineConfig, state: dict) -> bool:
    """Run training subprocess. Returns True on success."""
    
    # Clear HF datasets cache to avoid stale arrow files after parquet rewrites
    import shutil
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
        logger.info("Cleared HF datasets cache (%s)", hf_cache)

    config = build_train_config(pipeline_cfg, state)

    config_file = Path(state["checkpoint_dir"]) / f"_pipeline_config_iter{state['iteration']}.pkl"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "wb") as f:
        pickle.dump(config, f)

    phase = f"iter{state['iteration']}"
    n_dagger = len(state.get("dagger_datasets", []))
    logger.info("Training (%s): target %d steps, %d sources (BC=%.2f, %d RL, %d dagger)",
                phase, config.num_train_steps, len(config.data.data_sources),
                state["bc_sampling_share"], len(state["rl_datasets"]), n_dagger)

    cmd = ["uv", "run", "python", "scripts/train.py", "--config_file", str(config_file)]

    log_path = _log_path(state["checkpoint_dir"], logcfg.TRAINING, f"train_{phase}.log")
    rc = _run_subprocess(cmd, log_path, label=f"train/{phase}", tail_patterns=[
        "Step ", "loss=", "Saved checkpoint", "ERROR", "Traceback",
    ])

    config_file.unlink(missing_ok=True)

    if rc != 0:
        logger.error("Training failed (rc=%d). Log: %s", rc, log_path)
        return False

    step = find_latest_checkpoint_step(state["checkpoint_dir"])
    if step is not None:
        state["latest_checkpoint_step"] = step
    logger.info("Training complete. Latest checkpoint: step %s", state["latest_checkpoint_step"])

    wandb_id_file = Path(state["checkpoint_dir"]) / "wandb_id.txt"
    if wandb_id_file.exists():
        state["wandb_run_id"] = wandb_id_file.read_text().strip()

    return True


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _get_checkpoint_dir_for_rollout(state: dict) -> str | None:
    """Resolve the checkpoint directory for rollout collection."""
    checkpoint_step = state["latest_checkpoint_step"]
    checkpoint_dir = str(Path(state["checkpoint_dir"]) / str(checkpoint_step))

    if not Path(checkpoint_dir).exists():
        checkpoint_step = find_latest_checkpoint_step(state["checkpoint_dir"])
        if checkpoint_step is None:
            logger.error("No valid checkpoint in %s", state["checkpoint_dir"])
            return None
        checkpoint_dir = str(Path(state["checkpoint_dir"]) / str(checkpoint_step))

    return checkpoint_dir


def _make_rollout_id(model_step: int, strategy: str, worker_id: str | None = None) -> str:
    """Generate a unified rollout ID.

    Format: rollout_{step}_{strategy}_{YYYYMMDD_HHMMSS}[_{worker_id}]
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"rollout_{model_step}_{strategy}_{ts}"
    if worker_id:
        base = f"{base}_{worker_id}"
    return base


def _is_rejected_path(result: str | None) -> bool:
    """True if _run_single_rollout returned a `.REJECTED_CORRUPT` path.

    Used to gate HF upload + state collection while still allowing wandb
    logging (SR/metrics come from eval_summary.json in the rollout parent dir,
    independent of the dataset rename).
    """
    return result is not None and str(result).endswith(".REJECTED_CORRUPT")


def _run_single_rollout(
    pipeline_cfg: RLPipelineConfig, state: dict,
    checkpoint_dir: str,
    *,
    garment_list: list[str] | None = None,
    task_overrides: list[dict] | None = None,
    num_garments: int | None = None,
    num_episodes_override: int | None = None,
    label_suffix: str = "",
    rollout_id: str | None = None,
    rollout_type: str | None = None,
    aug_config_override: dict | None = None,
    subset: str = "seen",
) -> str | None:
    """Run one rollout collection. Returns eval_dataset path or None."""
    num_episodes = num_episodes_override or pipeline_cfg.num_episodes
    cmd = [
        "uv", "run", "python", "scripts/run_eval.py",
        "--checkpoint_dir", checkpoint_dir,
        "--config_name", pipeline_cfg.config_name,
        "--num_episodes", str(num_episodes),
        "--num_workers", str(pipeline_cfg.num_workers),
        "--camera_width", str(pipeline_cfg.camera_width),
        "--camera_height", str(pipeline_cfg.camera_height),
        *(["--top_camera_width", str(pipeline_cfg.top_camera_width)] if pipeline_cfg.top_camera_width else []),
        *(["--top_camera_height", str(pipeline_cfg.top_camera_height)] if pipeline_cfg.top_camera_height else []),
        *(["--dataset_width", str(pipeline_cfg.dataset_width)] if pipeline_cfg.dataset_width else []),
        *(["--dataset_height", str(pipeline_cfg.dataset_height)] if pipeline_cfg.dataset_height else []),
        "--dataset_format", pipeline_cfg.dataset_format,
        "--dataset_units", pipeline_cfg.dataset_units,
        "--no_wandb",
    ]
    if not pipeline_cfg.save_debug_video:
        cmd.append("--no_save_debug_video")

    # Attention backend flags (XSA auto-disables flash in pipeline_config).
    if not pipeline_cfg.use_flash_attention:
        cmd.append("--no_flash_attention")
    if pipeline_cfg.use_xsa:
        cmd.append("--use_xsa")

    # Pass eval_run_id for unified naming
    if rollout_id:
        cmd.extend(["--eval_run_id", rollout_id])

    # Pass inference optimization prior if enabled and file exists
    prior_file = state.get("inference_prior_file")
    if prior_file and Path(prior_file).exists():
        cmd.extend(["--prior_file", prior_file])

    # Per-garment-type fixed inference config (overrides Thompson Sampling).
    # When `inference_optimization.per_garment_type_config` is non-empty, eval_worker
    # uses these configs directly per episode based on garment_type.
    pgt_cfg = pipeline_cfg.inference_optimization.per_garment_type_config
    if pgt_cfg:
        import json as _json_pgt
        cmd.extend(["--per_garment_type_inference_config", _json_pgt.dumps(pgt_cfg)])

    # Pass noise temperature for exploration
    noise_temp = pipeline_cfg.noise_temperature
    if noise_temp != 1.0:
        cmd.extend(["--noise_temperature", str(noise_temp)])

    # Best-of-N candidate sampling at the policy server. N=1 disables.
    if pipeline_cfg.num_rollout_candidates > 1:
        cmd.extend(["--num_rollout_candidates", str(pipeline_cfg.num_rollout_candidates)])

    # DART-style exploration noise (rollout-only). Tri-gate: enabled flag,
    # prob > 0, scale > 0. Otherwise we don't pass the flags at all so the
    # CLI defaults (0.0) keep behavior bit-identical. Disabled for replay
    # strategies — those need to faithfully reproduce / extend stored states.
    en = pipeline_cfg.exploration_noise
    _replay_types = {"success_replay", "semi_success_replay", "hard_mining"}
    if en.enabled and en.prob > 0.0 and en.scale > 0.0 and rollout_type not in _replay_types:
        cmd.extend([
            "--explore_noise_prob", str(en.prob),
            "--explore_noise_scale", str(en.scale),
        ])

    # Pass advantage config through so the live debug-video GAE overlay
    # matches what the trainer actually optimises.
    cmd.extend([
        "--gae_lambda", str(pipeline_cfg.advantage.gae_lambda),
        "--success_alpha", str(pipeline_cfg.advantage.success_alpha),
        "--completion_alpha", str(pipeline_cfg.advantage.completion_alpha),
        "--value_tail_k", str(pipeline_cfg.advantage.value_tail_k),
    ])

    # Augmentation config: per-strategy override or pipeline default
    import json as _json_aug
    aug_dict = aug_config_override if aug_config_override is not None else pipeline_cfg.augmentation.to_dict()
    if any(v for k, v in aug_dict.items() if k != "step_color_tint"):
        aug_dict["enabled"] = True
        cmd.extend(["--aug_config", _json_aug.dumps(aug_dict)])

    # Rollout type label for episode metadata
    if rollout_type:
        cmd.extend(["--rollout_type", rollout_type])

    # Pass success rates for adaptive success state saving
    from lehome_solution.eval.rollout_strategies import SUCCESS_RATES_FILE
    _sr_found = False
    for ds in reversed(state.get("rl_datasets", [])):
        sr_path = Path(ds["root"]) / SUCCESS_RATES_FILE
        if sr_path.exists():
            cmd.extend(["--success_rates_file", str(sr_path)])
            _sr_found = True
            break
    if not _sr_found:
        # Fallback: check checkpoint dir (rollout workers don't have rl_datasets)
        sr_path = Path(state["checkpoint_dir"]) / SUCCESS_RATES_FILE
        if sr_path.exists():
            cmd.extend(["--success_rates_file", str(sr_path)])

    if task_overrides:
        # Write task overrides to a temp JSON file for exact (garment, seed) replay.
        # Force num_episodes=1: each override is one (garment, seed) task that runs as
        # ep_idx=0 for reproducible garment pose.  The caller duplicates entries to get
        # the desired total episode count.
        import json as _json, tempfile as _tempfile
        overrides_file = _tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="task_overrides_", delete=False,
        )
        _json.dump(task_overrides, overrides_file)
        overrides_file.close()
        # Override num_episodes to 1 for reproducibility
        for i, arg in enumerate(cmd):
            if arg == "--num_episodes":
                cmd[i + 1] = "1"
                break
        cmd.extend(["--task_overrides_file", overrides_file.name])
    elif garment_list:
        cmd.extend(["--garment_list"] + garment_list)
    else:
        # Randomized garment sampling. Map the strategy subset to run_eval's
        # mutually-exclusive subset flags ("unseen" is run_eval's default → no flag).
        if subset == "seen":
            cmd.append("--seen_only")
        elif subset == "all":
            cmd.append("--all")
        cmd.extend([
            "--randomize_garments",
            "--num_garments", str(num_garments or pipeline_cfg.num_garments),
        ])

    iteration = state['iteration']
    label = f"rollout/iter{iteration}{label_suffix}"
    log_path = _log_path(state["checkpoint_dir"], logcfg.ROLLOUT, f"rollout_iter{iteration}{label_suffix}.log")
    logger.info("[%s] Collecting rollouts (%s garments)", label,
                len(garment_list) if garment_list else (num_garments or pipeline_cfg.num_garments))

    rc = _run_subprocess_passthrough(cmd, label=label, log_path=log_path)
    if rc != 0:
        logger.error("[%s] Rollout collection failed (rc=%d)", label, rc)
        return None

    # If rollout_id was provided, look for that specific directory first.
    # When run_eval rejects a dataset via _assert_dataset_integrity it renames
    # the dir to `eval_dataset.REJECTED_CORRUPT`. Return that path too so the
    # caller can still log SR/metrics to wandb (eval_summary.json sits in the
    # rollout parent dir, independent of the dataset rename); the caller is
    # responsible for gating HF upload + state collection on is_rejected_path().
    eval_videos_dir = REPO_ROOT / "outputs" / "eval_videos"
    if rollout_id:
        specific_dir = eval_videos_dir / rollout_id / "eval_dataset"
        if specific_dir.exists():
            logger.info("[%s] Rollout complete: %s", label, specific_dir)
            return str(specific_dir)
        rejected_specific = eval_videos_dir / rollout_id / "eval_dataset.REJECTED_CORRUPT"
        if rejected_specific.exists():
            logger.warning("[%s] Rollout produced REJECTED dataset: %s", label, rejected_specific)
            return str(rejected_specific)

    # Fallback: find most recently created eval directory
    eval_dirs = sorted(
        [d for d in eval_videos_dir.iterdir()
         if d.is_dir() and (d.name.startswith("rl") or d.name.startswith("step")
                            or d.name.startswith("rollout_"))],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not eval_dirs:
        logger.error("[%s] No eval directories found in %s", label, eval_videos_dir)
        return None

    eval_dataset = eval_dirs[0] / "eval_dataset"
    if eval_dataset.exists():
        logger.info("[%s] Rollout complete: %s", label, eval_dataset)
        return str(eval_dataset)
    rejected = eval_dirs[0] / "eval_dataset.REJECTED_CORRUPT"
    if rejected.exists():
        logger.warning("[%s] Rollout produced REJECTED dataset: %s", label, rejected)
        return str(rejected)
    logger.error("[%s] eval_dataset not found in %s", label, eval_dirs[0])
    return None


def _get_garment_pool(subset: str = "seen") -> list[str]:
    """Get the garment pool for a rollout strategy.

    `subset` is the strategy-level garment subset ("seen" | "unseen" | "all"),
    defaulting to seen-only so RL rollouts never leak held-out unseen garments
    into training data.
    """
    from lehome_solution.eval import get_garments
    pool = []
    for gt in ["top_long", "top_short", "pant_long", "pant_short"]:
        pool.extend(get_garments(REPO_ROOT, gt, subset=subset))
    return pool


def _load_sr_for_strategies(state: dict) -> dict | None:
    """Load success rates for FR-proportional sampling in strategies."""
    from lehome_solution.eval.rollout_strategies import (
        load_success_rates, load_success_rates_from_file,
    )
    # Try RL dataset directories first (most authoritative)
    sr = load_success_rates(state.get("rl_datasets", []))
    if sr:
        return sr
    # Fallback: checkpoint dir (downloaded from HF by _download_model_assets_from_hf)
    checkpoint_dir = state.get("checkpoint_dir")
    if checkpoint_dir:
        from lehome_solution.eval.rollout_strategies import SUCCESS_RATES_FILE
        sr = load_success_rates_from_file(Path(checkpoint_dir) / SUCCESS_RATES_FILE)
        if sr:
            logger.info("Loaded success rates from checkpoint dir: %s", checkpoint_dir)
            return sr
        # Also check HF checkpoints assets
        hf_assets = Path(checkpoint_dir) / "_hf_checkpoints" / "latest" / "assets" / SUCCESS_RATES_FILE
        sr = load_success_rates_from_file(hf_assets)
        if sr:
            logger.info("Loaded success rates from HF assets: %s", hf_assets)
            return sr
    logger.warning("No success_rates.json found — curriculum/FR-proportional sampling will fall back to uniform random")
    return None


def _execute_strategy(
    pipeline_cfg: RLPipelineConfig,
    state: dict,
    checkpoint_dir: str,
    strategy,
    *,
    rollout_id: str,
) -> tuple[str | None, bool, bool]:
    """Execute a single rollout strategy.

    Returns ``(eval_dataset_path, should_skip, is_dagger)``. ``is_dagger`` is
    True when the produced rollout should be routed into the dagger dataset
    pool instead of the RL pool (semi_success_replay with ``dagger_only``).

    Shared by run_rollout_collection (main pipeline) and run_rollout_worker_loop (distributed).
    """
    import numpy as _np
    import time as _time
    name = strategy.name

    # Per-strategy overrides (fall back to pipeline defaults)
    n_garments = strategy.num_garments or pipeline_cfg.num_garments
    n_episodes = strategy.num_episodes or pipeline_cfg.num_episodes
    # Merge strategy augmentation overrides on top of pipeline defaults.
    # strategy.augmentation is a raw dict with only explicitly-set keys.
    if strategy.augmentation is not None:
        strat_aug = pipeline_cfg.augmentation.to_dict()
        strat_aug.update(strategy.augmentation)
    else:
        strat_aug = None

    if name == "random":
        result = _run_single_rollout(
            pipeline_cfg, state, checkpoint_dir,
            num_garments=n_garments, num_episodes_override=n_episodes,
            label_suffix="_random", rollout_id=rollout_id,
            rollout_type="random",
            aug_config_override=strat_aug,
            subset=strategy.subset,
        )
    elif name == "full":
        from lehome_solution.eval.rollout_strategies import full_strategy
        pool = _get_garment_pool(strategy.subset)
        all_garments = full_strategy(pool)
        result = _run_single_rollout(
            pipeline_cfg, state, checkpoint_dir,
            garment_list=all_garments, num_episodes_override=n_episodes,
            label_suffix="_full", rollout_id=rollout_id,
            rollout_type="full",
            aug_config_override=strat_aug,
        )
    elif name == "curriculum":
        from lehome_solution.eval.rollout_strategies import (
            curriculum_strategy,
        )
        pool = list(strategy.garment_list) if strategy.garment_list else _get_garment_pool(strategy.subset)
        sr = _load_sr_for_strategies(state)
        rng = _np.random.RandomState(int(_time.time_ns()) % (2**31))
        n_rollouts = n_garments * n_episodes
        sampled = curriculum_strategy(
            pool, n_rollouts, rng,
            success_rates=sr,
            target_sr=strategy.target_sr or 0.5,
        )
        result = _run_single_rollout(
            pipeline_cfg, state, checkpoint_dir,
            garment_list=sampled, num_episodes_override=1,
            label_suffix="_curriculum", rollout_id=rollout_id,
            rollout_type="curriculum",
            aug_config_override=strat_aug,
        )
    elif name == "hard_mining":
        from lehome_solution.eval.rollout_strategies import (
            hard_mining_strategy, get_failure_states,
        )
        pool = _get_garment_pool(strategy.subset)
        persistent_dir = Path(state["checkpoint_dir"]) / "failure_states"
        fs = get_failure_states(persistent_dir)
        sr = _load_sr_for_strategies(state)
        rng = _np.random.RandomState(int(_time.time_ns()) % (2**31))
        selected, overrides = hard_mining_strategy(
            pool, n_garments, rng,
            failure_states=fs if fs else None,
            replays_per_state=strategy.replays_per_state,
            remove_on_success=strategy.remove_on_success,
            success_rates=sr,
        )
        if overrides:
            result = _run_single_rollout(
                pipeline_cfg, state, checkpoint_dir,
                task_overrides=overrides, label_suffix="_hard",
                rollout_id=rollout_id,
                aug_config_override=strat_aug,
            )
        else:
            logger.info("Hard mining: no failure states, skipping")
            return None, True, False  # skip
    elif name == "success_replay":
        from lehome_solution.eval.rollout_strategies import (
            success_replay_strategy, get_success_states,
            mark_states_in_use, mark_states_consumed,
            cleanup_consumed_states,
        )
        pool = _get_garment_pool(strategy.subset)
        persistent_ss_dir = Path(state["checkpoint_dir"]) / "success_states"
        ss = get_success_states(persistent_ss_dir)
        sr = _load_sr_for_strategies(state)
        rng = _np.random.RandomState(int(_time.time_ns()) % (2**31))
        selected, overrides = success_replay_strategy(
            pool, strategy.num_states, strategy.replays_per_state, rng,
            success_states=ss if ss else None,
            success_rates=sr,
            uniform_garment_sampling=strategy.uniform_garment_sampling,
        )
        if overrides:
            # Mark selected states as in_use before rollout
            used_npz_paths = {ov["restore_npz"] for ov in overrides if "restore_npz" in ov}
            used_states = [s for s in ss if s["npz_path"] in used_npz_paths]
            mark_states_in_use(used_states, rollout_id=rollout_id)

            result = _run_single_rollout(
                pipeline_cfg, state, checkpoint_dir,
                task_overrides=overrides, label_suffix="_replay",
                rollout_id=rollout_id,
                aug_config_override=strat_aug,
            )
            if result:
                # Mark as consumed after successful dataset write
                mark_states_consumed(used_states)
                consumed_names = [Path(s["npz_path"]).name for s in used_states]
                if consumed_names:
                    state.setdefault("_consumed_success_states", []).extend(consumed_names)
                # Clean up consumed states (delete NPZ+JSON pairs)
                cleanup_consumed_states(persistent_ss_dir)
        else:
            logger.info("Success replay: no success states, skipping")
            return None, True, False  # skip
    elif name == "semi_success_replay":
        from lehome_solution.eval.rollout_strategies import (
            semi_success_replay_strategy, get_success_states,
            mark_states_in_use, mark_states_consumed,
            cleanup_consumed_states,
        )
        pool = _get_garment_pool(strategy.subset)
        persistent_semi_dir = Path(state["checkpoint_dir"]) / "semi_success_states"
        ss = get_success_states(persistent_semi_dir)
        sr = _load_sr_for_strategies(state)
        rng = _np.random.RandomState(int(_time.time_ns()) % (2**31))
        dagger_only = bool(getattr(strategy, "dagger_only", False))
        selected, overrides = semi_success_replay_strategy(
            pool, strategy.num_states, strategy.replays_per_state, rng,
            semi_success_states=ss if ss else None,
            success_rates=sr,
            dagger_only=dagger_only,
        )
        if overrides:
            used_npz_paths = {ov["restore_npz"] for ov in overrides if "restore_npz" in ov}
            used_states = [s for s in ss if s["npz_path"] in used_npz_paths]
            mark_states_in_use(used_states, rollout_id=rollout_id)

            suffix = "_semi_replay_dagger" if dagger_only else "_semi_replay"
            result = _run_single_rollout(
                pipeline_cfg, state, checkpoint_dir,
                task_overrides=overrides, label_suffix=suffix,
                rollout_id=rollout_id,
                aug_config_override=strat_aug,
            )
            # Always mark semi-success states as consumed (regardless of outcome)
            mark_states_consumed(used_states)
            consumed_names = [Path(s["npz_path"]).name for s in used_states]
            if consumed_names:
                state.setdefault("_consumed_semi_success_states", []).extend(consumed_names)
            cleanup_consumed_states(persistent_semi_dir)
            # Collect any new success states generated from successful semi-success replays
            if result:
                persistent_ss_dir = Path(state["checkpoint_dir"]) / "success_states"
                _collect_success_states([result], persistent_ss_dir)
            return result, False, dagger_only
        else:
            logger.info("Semi-success replay: no semi-success states, skipping")
            return None, True, False  # skip
    else:
        logger.warning("Unknown strategy: %s, using random", name)
        result = _run_single_rollout(
            pipeline_cfg, state, checkpoint_dir,
            num_garments=n_garments, label_suffix=f"_{name}",
            rollout_id=rollout_id,
            rollout_type=name,
        )

    return result, False, False


def run_rollout_collection(pipeline_cfg: RLPipelineConfig, state: dict) -> str | None:
    """Run rollout collection with configured strategies.

    Supports strategies (configured via rollout_strategies in YAML):
    - random: sample N garments randomly (default)
    - full: all garments with N episodes each (unbiased metrics)
    - curriculum: select garments near 50% success rate
    - hard_mining: re-roll failed garments (FR-proportional sampling)
    - success_replay: replay successful episodes with different augmentations
    - semi_success_replay: replay episodes that reached first checkpoint, then hand off to policy

    Each strategy specifies its own num_garments (or uses pipeline default).
    """
    checkpoint_dir = _get_checkpoint_dir_for_rollout(state)
    if checkpoint_dir is None:
        return None

    strategies_cfg = pipeline_cfg.rollout_strategies

    model_step = state.get("latest_checkpoint_step", 0)
    worker_id = state.get("_worker_id")  # set by rollout_worker mode

    # Default: pure random (backward compatible)
    if not strategies_cfg:
        rid = _make_rollout_id(model_step, "random", worker_id)
        return _run_single_rollout(pipeline_cfg, state, checkpoint_dir,
                                   rollout_id=rid, rollout_type="random")

    all_datasets = []
    dagger_results: list[str] = []

    for strategy in strategies_cfg:
        rid = _make_rollout_id(model_step, strategy.name, worker_id)

        result, skipped, is_dagger = _execute_strategy(
            pipeline_cfg, state, checkpoint_dir, strategy,
            rollout_id=rid,
        )

        if skipped:
            continue

        if result:
            if _is_rejected_path(result):
                logger.warning(
                    "Strategy %s produced a REJECTED dataset (%s) — dropping from "
                    "RL/dagger pool and skipping failure-state collection",
                    strategy.name, result,
                )
                continue
            if is_dagger:
                dagger_results.append(result)
            else:
                all_datasets.append((result, strategy.name))
            # Incrementally collect failure states so later strategies
            # (hard_mining) can see NPZs from earlier rollouts in this iteration.
            persistent_fs_dir = Path(state["checkpoint_dir"]) / "failure_states"
            _collect_failure_states([result], persistent_fs_dir)

    # Route dagger-flagged rollouts into state["dagger_datasets"] so the trainer
    # treats them like BC/DAgger (fixed advantage, FR-proportional sampling).
    if dagger_results:
        state.setdefault("dagger_datasets", [])
        existing = {ds["root"] for ds in state["dagger_datasets"]}
        for root in dagger_results:
            if root in existing:
                continue
            state["dagger_datasets"].append({
                "root": root,
                "sampling_share": 1.0,
                "repo_id": f"lehome_dagger_{len(state['dagger_datasets'])}",
            })
            logger.info("Added dagger dataset (semi_success_replay): %s", root)

    if not all_datasets and not dagger_results:
        return None

    # Store ALL new RL datasets (added together in update_dataset_shares so they
    # skip the decay that applies to datasets from previous iterations).
    state["_new_rollout_datasets"] = [
        {"root": ds, "strategy": strategy, "model_step": model_step}
        for ds, strategy in all_datasets
    ]
    # Return last dataset root (used for advantage recomputation path).
    # Prefer an RL dataset; fall back to a dagger result so the pipeline still
    # has a sentinel root to return.
    if all_datasets:
        return all_datasets[-1][0]
    return dagger_results[-1]



# ---------------------------------------------------------------------------
# Recompute advantages
# ---------------------------------------------------------------------------

def recompute_advantages(pipeline_cfg: RLPipelineConfig, state: dict) -> bool:
    # Collect RL datasets only — dagger gets fixed advantage (treated like BC)
    all_datasets = [(ds["root"], ds["sampling_share"], ds.get("segment_only", False))
                    for ds in state["rl_datasets"] if Path(ds["root"]).exists()]

    if not all_datasets:
        logger.warning("No RL datasets to recompute advantages for")
        return True

    dirs = [d[0] for d in all_datasets]
    weights = [d[1] for d in all_datasets]

    # Determine advantage mode: segment for iteration 0, gae for later
    advantage_mode = pipeline_cfg.advantage.mode
    if advantage_mode == "auto":
        mode = "segment" if state["iteration"] == 0 else "gae"
    else:
        mode = advantage_mode

    # Collect force-segment dirs: explicitly flagged + auto-threshold
    threshold = pipeline_cfg.advantage.segment_only_threshold
    force_segment_dirs = []
    for root, share, seg_only in all_datasets:
        if seg_only or (threshold > 0 and share < threshold):
            force_segment_dirs.append(root)

    cmd = [
        "uv", "run", "python", "scripts/recompute_advantages.py",
        "--eval_dirs", *dirs,
        "--sampling_shares", *[str(w) for w in weights],
        "--mode", mode,
        "--clip_min", str(pipeline_cfg.advantage.weighting_clipping[0]),
        "--clip_max", str(pipeline_cfg.advantage.weighting_clipping[1]),
        "--gae_lambda", str(pipeline_cfg.advantage.gae_lambda),
        "--gamma", str(pipeline_cfg.advantage.gamma),
        "--beta", str(pipeline_cfg.advantage.beta),
        "--success_alpha", str(pipeline_cfg.advantage.success_alpha),
        "--completion_alpha", str(pipeline_cfg.advantage.completion_alpha),
        "--value_tail_k", str(pipeline_cfg.advantage.value_tail_k),
        "--precision_boost", str(pipeline_cfg.precision_boost.boost),
        "--precision_top_k", str(pipeline_cfg.precision_boost.top_k),
        "--precision_min_successes", str(pipeline_cfg.precision_boost.min_successes),
    ]

    if force_segment_dirs and mode == "gae":
        cmd.extend(["--force_segment_dirs", *force_segment_dirs])
        logger.info("Force segment-only for %d datasets (threshold=%.3f)",
                     len(force_segment_dirs), threshold)

    logger.info("Recomputing advantages across %d datasets (mode=%s)", len(dirs), mode)

    log_path = _log_path(state["checkpoint_dir"], logcfg.ADVANTAGES, f"advantages_iter{state['iteration']}.log")
    rc = _run_subprocess(cmd, log_path, label="advantages", tail_patterns=[
        "SEGMENT ADVANTAGE", "OVERALL", "Success rate", "Updated", "ERROR", "Traceback",
    ])
    if rc != 0:
        logger.error("Advantage recomputation failed (rc=%d). Log: %s", rc, log_path)
        return False

    logger.info("Advantage recomputation complete (mode=%s)", mode)
    return True


# ---------------------------------------------------------------------------
# Wandb eval logging
# ---------------------------------------------------------------------------

def _compute_binary_pred_metrics(preds: "np.ndarray", labels: "np.ndarray", prefix: str) -> dict:
    """Compute accuracy, BCE, MSE, ROC AUC for binary predictions."""
    import numpy as np

    if len(preds) < 10:
        return {}

    eps = 1e-7
    preds_c = np.clip(preds, eps, 1 - eps)
    bce = -np.mean(labels * np.log(preds_c) + (1 - labels) * np.log(1 - preds_c))
    mse = np.mean((preds - labels) ** 2)
    accuracy = np.mean((preds > 0.5) == (labels > 0.5))

    metrics = {
        f"{prefix}/bce": float(bce),
        f"{prefix}/mse": float(mse),
        f"{prefix}/accuracy": float(accuracy),
        f"{prefix}/pred_mean": float(np.mean(preds)),
        f"{prefix}/label_mean": float(np.mean(labels)),
        f"{prefix}/n_frames": len(preds),
    }

    if len(np.unique(labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            metrics[f"{prefix}/roc_auc"] = float(roc_auc_score(labels, preds))
        except Exception:
            order = np.argsort(-preds)
            sorted_labels = labels[order]
            n_pos = labels.sum()
            n_neg = len(labels) - n_pos
            if n_pos > 0 and n_neg > 0:
                tp_cumsum = np.cumsum(sorted_labels)
                fp_cumsum = np.cumsum(1 - sorted_labels)
                tpr = tp_cumsum / n_pos
                fpr = fp_cumsum / n_neg
                metrics[f"{prefix}/roc_auc"] = float(np.trapz(tpr, fpr))

    return metrics


def _compute_success_pred_metrics(eval_dataset_root: str) -> dict:
    """Compute prediction quality metrics from eval dataset parquets.

    Only includes episodes from random/full rollout types (unbiased).

    Computes for:
    - Success prediction: P(success) vs binary success label
    - Checkpoint prediction: P(reward>=0.5) vs binary (dense_return >= 0.5)
    - Garment type prediction: predicted class vs true garment_type_id
    - Completion prediction: predicted %completion vs true frame_position / episode_length

    Returns dict with rollout_value/ rollout_checkpoint/ rollout_garment/ rollout_completion/ prefixed keys.
    """
    import glob
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}

    import numpy as np
    from lehome_solution.eval.metadata import EpisodeMetadata
    from pathlib import Path

    _METRICS_INCLUDED_ROLLOUT_TYPES = {"random", "full", "normal"}

    pq_files = sorted(glob.glob(os.path.join(eval_dataset_root, "data/chunk-*/*.parquet")))
    if not pq_files:
        return {}

    # Real-format datasets have none of the aux columns we'd compute metrics
    # against (success_pred / value / success / advantage / ...). Bail out
    # before hitting a KeyError on the first column read.
    try:
        _probe = pq.read_table(pq_files[0])
        _probe_cols = set(_probe.column_names)
    except Exception:
        return {}
    if "success_pred" not in _probe_cols and "value" not in _probe_cols:
        logger.info(
            "Eval dataset at %s has no success_pred/value column (likely real_bc format); "
            "skipping success-prediction metrics.",
            eval_dataset_root,
        )
        return {}

    # Load episode metadata for ground truth garment_type_id and dense_return
    meta_dir = Path(eval_dataset_root) / "meta" / "eval_episode_metadata"
    ep_meta: dict[int, dict] = {}
    if meta_dir.exists():
        import json
        for mf in sorted(meta_dir.glob("*.json")):
            try:
                m = json.load(open(mf))
                ep_idx = int(mf.stem.split("_")[-1])
                ep_meta[ep_idx] = m
            except Exception:
                continue

    # Build set of episodes from random/full rollout types only
    included_episodes: set[int] = set()
    for ep_idx, m in ep_meta.items():
        rt = m.get("rollout_type", "normal")
        if rt in _METRICS_INCLUDED_ROLLOUT_TYPES:
            included_episodes.add(ep_idx)
    # If no metadata available, include all episodes (backward compat)
    has_metadata = bool(ep_meta)

    # Read all frames
    all_value_preds = []
    all_success_labels = []
    all_cp_preds = []
    all_cp_labels = []
    all_gt_preds = []  # per-frame garment type predictions
    all_gt_labels = []  # per-frame garment type labels
    all_gt_preds_first5 = []  # first 5 predictions per episode
    all_gt_labels_first5 = []
    # Completion predictions collected per-episode (in order) so we can compute
    # ground-truth completion_frac = i / len(episode) after all frames are seen.
    comp_preds_by_ep: dict[int, list[float]] = {}
    # TTC predictions collected per-episode for regression metrics.
    ttc_preds_by_ep: dict[int, list[float]] = {}

    # Track per-episode frame counts for "first 5 chunks" metric
    ep_chunk_counts: dict[int, int] = {}  # ep_idx -> chunks seen so far

    # Keypoint + WM-flow predictions. Per-episode arrays so we
    # can shift the ground-truth by WM_FUTURE_HORIZON (=30) when scoring the
    # future-target heads. All stored as list[np.ndarray] or list[float] keyed
    # by ep_idx, preserving frame order.
    kpt_preds_by_ep: dict[int, list[np.ndarray]] = {}
    kpt_gt_by_ep: dict[int, list[np.ndarray]] = {}
    wm_flow_success_cond_by_ep: dict[int, list[float]] = {}
    wm_flow_success_uncond_by_ep: dict[int, list[float]] = {}
    wm_flow_completion_cond_by_ep: dict[int, list[float]] = {}
    wm_flow_completion_uncond_by_ep: dict[int, list[float]] = {}
    wm_flow_keypoint_cond_by_ep: dict[int, list[np.ndarray]] = {}
    wm_flow_keypoint_uncond_by_ep: dict[int, list[np.ndarray]] = {}
    # Per-episode success_pred / completion_pred streams (piecewise-constant
    # within a chunk). Used to compute WM-flow "local advantage" metrics:
    # at each chunk boundary, adv = wm_flow_*[b] - current_pred[b], correlated
    # with (a) episode terminal success and (b) the observed change in the
    # current_pred at the NEXT boundary (delta over N executed steps).
    success_pred_stream_by_ep: dict[int, list[float]] = {}
    completion_pred_stream_by_ep: dict[int, list[float]] = {}
    # Best-of-N diagnostics. Values are propagated per-chunk so the per-frame
    # stream is piecewise constant; we de-dup via chunk boundaries below so
    # each chunk contributes exactly one sample to the aggregated metrics.
    bon_spread_stream_by_ep: dict[int, list[float]] = {}
    bon_std_stream_by_ep: dict[int, list[float]] = {}
    bon_chosen_stream_by_ep: dict[int, list[float]] = {}
    bon_mean_stream_by_ep: dict[int, list[float]] = {}
    bon_min_stream_by_ep: dict[int, list[float]] = {}
    bon_max_stream_by_ep: dict[int, list[float]] = {}
    bon_n_valid_stream_by_ep: dict[int, list[int]] = {}

    columns = ["success_pred", "success", "episode_index"]
    # Try to read optional columns (with backward compat for old column names)
    optional_cols = ["checkpoint_pred", "dense_reward", "garment_type_pred"]

    for pf in pq_files:
        try:
            t = pq.read_table(pf)
        except Exception:
            continue

        col_names = t.column_names
        val_col = "success_pred" if "success_pred" in col_names else "value"
        values = t[val_col].to_pylist()
        successes = t["success"].to_pylist()
        ep_indices = t["episode_index"].to_pylist()
        cp_col = "checkpoint_pred" if "checkpoint_pred" in col_names else "checkpoint_value" if "checkpoint_value" in col_names else None
        cp_values = t[cp_col].to_pylist() if cp_col else None
        gt_preds_col = t["garment_type_pred"].to_pylist() if "garment_type_pred" in col_names else None
        comp_preds_col = t["completion_pred"].to_pylist() if "completion_pred" in col_names else None
        ttc_preds_col = t["ttc_pred"].to_pylist() if "ttc_pred" in col_names else None
        # Keypoint + WM-flow prediction columns. Absent on
        # older rollouts → return None so the per-row loop skips them.
        bon_spread_col = t["best_of_n_score_spread"].to_pylist() if "best_of_n_score_spread" in col_names else None
        bon_std_col = t["best_of_n_score_std"].to_pylist() if "best_of_n_score_std" in col_names else None
        bon_chosen_col = t["best_of_n_score_chosen"].to_pylist() if "best_of_n_score_chosen" in col_names else None
        bon_mean_col = t["best_of_n_score_mean"].to_pylist() if "best_of_n_score_mean" in col_names else None
        bon_min_col = t["best_of_n_score_min"].to_pylist() if "best_of_n_score_min" in col_names else None
        bon_max_col = t["best_of_n_score_max"].to_pylist() if "best_of_n_score_max" in col_names else None
        bon_n_valid_col = t["best_of_n_n_valid"].to_pylist() if "best_of_n_n_valid" in col_names else None
        kpt_pred_col = t["keypoint_distances_pred"].to_pylist() if "keypoint_distances_pred" in col_names else None
        check_dist_col = t["check_distances"].to_pylist() if "check_distances" in col_names else None
        wm_s_cond_col = t["wm_flow_success_cond"].to_pylist() if "wm_flow_success_cond" in col_names else None
        wm_s_unc_col = t["wm_flow_success_uncond"].to_pylist() if "wm_flow_success_uncond" in col_names else None
        wm_c_cond_col = t["wm_flow_completion_cond"].to_pylist() if "wm_flow_completion_cond" in col_names else None
        wm_c_unc_col = t["wm_flow_completion_uncond"].to_pylist() if "wm_flow_completion_uncond" in col_names else None
        wm_k_cond_col = t["wm_flow_keypoint_cond"].to_pylist() if "wm_flow_keypoint_cond" in col_names else None
        wm_k_unc_col = t["wm_flow_keypoint_uncond"].to_pylist() if "wm_flow_keypoint_uncond" in col_names else None

        def _scalar_of(row):
            return row[0] if isinstance(row, list) else row

        for row_idx in range(t.num_rows):
            ep_idx = int(ep_indices[row_idx])

            # Filter: only random/full rollout types for metrics
            if has_metadata and ep_idx not in included_episodes:
                continue

            # Success prediction
            pv = values[row_idx]
            lv = successes[row_idx]
            pv = pv[0] if isinstance(pv, list) else pv
            lv = lv[0] if isinstance(lv, list) else lv
            if pv is not None and lv is not None:
                all_value_preds.append(float(pv))
                all_success_labels.append(float(lv))
            if pv is not None:
                # Per-episode stream for WM-flow "local advantage" computation.
                success_pred_stream_by_ep.setdefault(ep_idx, []).append(float(pv))

            # Checkpoint prediction
            if cp_values is not None:
                cpv = cp_values[row_idx]
                cpv = cpv[0] if isinstance(cpv, list) else cpv
                if cpv is not None and ep_idx in ep_meta:
                    max_r = ep_meta[ep_idx].get("max_reward", ep_meta[ep_idx].get("dense_return", 0.0))
                    cp_label = 1.0 if float(max_r) >= 0.5 else 0.0
                    all_cp_preds.append(float(cpv))
                    all_cp_labels.append(cp_label)

            # Completion prediction — collect per-episode in order for ground-truth
            # computation after the full episode has been seen.
            if comp_preds_col is not None:
                cv = comp_preds_col[row_idx]
                cv = cv[0] if isinstance(cv, list) else cv
                if cv is not None:
                    comp_preds_by_ep.setdefault(ep_idx, []).append(float(cv))
                    # Separate stream for the WM-flow advantage metrics (kept
                    # parallel to success_pred_stream_by_ep for easy alignment).
                    completion_pred_stream_by_ep.setdefault(ep_idx, []).append(float(cv))

            # TTC prediction — collect per-episode for regression metrics.
            if ttc_preds_col is not None:
                tv = ttc_preds_col[row_idx]
                tv = tv[0] if isinstance(tv, list) else tv
                if tv is not None:
                    ttc_preds_by_ep.setdefault(ep_idx, []).append(float(tv))

            # Keypoint distance prediction + ground truth (current frame).
            if kpt_pred_col is not None:
                kp = kpt_pred_col[row_idx]
                if kp is not None:
                    kpt_preds_by_ep.setdefault(ep_idx, []).append(
                        np.asarray(kp, dtype=np.float32)
                    )
                    cd = check_dist_col[row_idx] if check_dist_col is not None else None
                    kpt_gt_by_ep.setdefault(ep_idx, []).append(
                        np.asarray(cd, dtype=np.float32) if cd is not None else np.full(7, np.nan, dtype=np.float32)
                    )
            # WM-flow scalar predictions (success / completion, cond + uncond).
            if wm_s_cond_col is not None:
                v = _scalar_of(wm_s_cond_col[row_idx])
                if v is not None:
                    wm_flow_success_cond_by_ep.setdefault(ep_idx, []).append(float(v))
            if wm_s_unc_col is not None:
                v = _scalar_of(wm_s_unc_col[row_idx])
                if v is not None:
                    wm_flow_success_uncond_by_ep.setdefault(ep_idx, []).append(float(v))
            if wm_c_cond_col is not None:
                v = _scalar_of(wm_c_cond_col[row_idx])
                if v is not None:
                    wm_flow_completion_cond_by_ep.setdefault(ep_idx, []).append(float(v))
            if wm_c_unc_col is not None:
                v = _scalar_of(wm_c_unc_col[row_idx])
                if v is not None:
                    wm_flow_completion_uncond_by_ep.setdefault(ep_idx, []).append(float(v))
            # Best-of-N scalar streams (piecewise constant per chunk).
            if bon_spread_col is not None:
                v = _scalar_of(bon_spread_col[row_idx])
                if v is not None:
                    bon_spread_stream_by_ep.setdefault(ep_idx, []).append(float(v))
            if bon_std_col is not None:
                v = _scalar_of(bon_std_col[row_idx])
                if v is not None:
                    bon_std_stream_by_ep.setdefault(ep_idx, []).append(float(v))
            if bon_chosen_col is not None:
                v = _scalar_of(bon_chosen_col[row_idx])
                if v is not None:
                    bon_chosen_stream_by_ep.setdefault(ep_idx, []).append(float(v))
            if bon_mean_col is not None:
                v = _scalar_of(bon_mean_col[row_idx])
                if v is not None:
                    bon_mean_stream_by_ep.setdefault(ep_idx, []).append(float(v))
            if bon_min_col is not None:
                v = _scalar_of(bon_min_col[row_idx])
                if v is not None:
                    bon_min_stream_by_ep.setdefault(ep_idx, []).append(float(v))
            if bon_max_col is not None:
                v = _scalar_of(bon_max_col[row_idx])
                if v is not None:
                    bon_max_stream_by_ep.setdefault(ep_idx, []).append(float(v))
            if bon_n_valid_col is not None:
                v = _scalar_of(bon_n_valid_col[row_idx])
                if v is not None:
                    bon_n_valid_stream_by_ep.setdefault(ep_idx, []).append(int(v))

            # WM-flow keypoint predictions (21-wide, cond + uncond).
            if wm_k_cond_col is not None:
                v = wm_k_cond_col[row_idx]
                if v is not None:
                    wm_flow_keypoint_cond_by_ep.setdefault(ep_idx, []).append(
                        np.asarray(v, dtype=np.float32)
                    )
            if wm_k_unc_col is not None:
                v = wm_k_unc_col[row_idx]
                if v is not None:
                    wm_flow_keypoint_uncond_by_ep.setdefault(ep_idx, []).append(
                        np.asarray(v, dtype=np.float32)
                    )

            # Garment type prediction
            if gt_preds_col is not None:
                gtv = gt_preds_col[row_idx]
                gtv = gtv[0] if isinstance(gtv, list) else gtv
                if gtv is not None and int(gtv) >= 0 and ep_idx in ep_meta:
                    gt_label = ep_meta[ep_idx].get("garment_type_id", -1)
                    if gt_label >= 0:
                        all_gt_preds.append(int(gtv))
                        all_gt_labels.append(int(gt_label))

                        # Track first 5 chunks per episode
                        count = ep_chunk_counts.get(ep_idx, 0)
                        if count < 5:
                            all_gt_preds_first5.append(int(gtv))
                            all_gt_labels_first5.append(int(gt_label))
                            ep_chunk_counts[ep_idx] = count + 1

    metrics = {}

    # Success prediction metrics
    if all_value_preds:
        preds = np.array(all_value_preds)
        labels = np.array(all_success_labels)
        m = _compute_binary_pred_metrics(preds, labels, "rollout_value")
        if m:
            m["rollout_value/pred_std"] = float(np.std(preds))
        metrics.update(m)
        logger.info("Success pred: acc=%.3f auc=%.3f (%d frames)",
                     m.get("rollout_value/accuracy", -1),
                     m.get("rollout_value/roc_auc", -1), len(preds))

    # Checkpoint prediction metrics
    if all_cp_preds:
        cp_preds = np.array(all_cp_preds)
        cp_labels = np.array(all_cp_labels)
        m = _compute_binary_pred_metrics(cp_preds, cp_labels, "rollout_checkpoint")
        metrics.update(m)
        logger.info("Checkpoint pred: acc=%.3f auc=%.3f (%d frames)",
                     m.get("rollout_checkpoint/accuracy", -1),
                     m.get("rollout_checkpoint/roc_auc", -1), len(cp_preds))

    # Garment type prediction metrics
    if all_gt_preds:
        gt_preds = np.array(all_gt_preds)
        gt_labels = np.array(all_gt_labels)
        gt_acc = np.mean(gt_preds == gt_labels)
        metrics["rollout_garment/accuracy"] = float(gt_acc)
        metrics["rollout_garment/n_frames"] = len(gt_preds)

        # Per-class accuracy
        from lehome_solution.constants import GARMENT_TYPES
        for tid, tname in enumerate(GARMENT_TYPES):
            mask = gt_labels == tid
            if mask.sum() > 0:
                metrics[f"rollout_garment/accuracy_{tname}"] = float(np.mean(gt_preds[mask] == gt_labels[mask]))

        # First 5 chunks accuracy (early prediction quality)
        if all_gt_preds_first5:
            gt5_preds = np.array(all_gt_preds_first5)
            gt5_labels = np.array(all_gt_labels_first5)
            gt5_acc = np.mean(gt5_preds == gt5_labels)
            metrics["rollout_garment/accuracy_first5"] = float(gt5_acc)
            # Per-chunk accuracy (chunk 0 = first prediction, chunk 4 = 5th)
            # We stored them sequentially per episode, so chunks 0-4 for each episode
            # Group by position within episode
            pos_in_ep = []
            for ep_idx in sorted(ep_chunk_counts.keys()):
                n = ep_chunk_counts[ep_idx]
                pos_in_ep.extend(range(n))
            if len(pos_in_ep) == len(gt5_preds):
                pos_arr = np.array(pos_in_ep)
                for chunk_pos in range(5):
                    mask = pos_arr == chunk_pos
                    if mask.sum() > 0:
                        metrics[f"rollout_garment/accuracy_chunk{chunk_pos}"] = float(
                            np.mean(gt5_preds[mask] == gt5_labels[mask])
                        )

        logger.info("Garment type pred: acc=%.3f first5=%.3f (%d frames, %d first5)",
                     gt_acc, metrics.get("rollout_garment/accuracy_first5", -1),
                     len(gt_preds), len(all_gt_preds_first5))

    # Completion prediction metrics: per-frame regression against true
    # completion_frac = i / len(episode). Flattens per-episode lists after the
    # loop so ground-truth is normalised per actual episode length.
    if comp_preds_by_ep:
        all_comp_preds: list[float] = []
        all_comp_labels: list[float] = []
        for ep_idx, preds_list in comp_preds_by_ep.items():
            n = len(preds_list)
            if n < 2:
                continue
            for i, pv in enumerate(preds_list):
                all_comp_preds.append(pv)
                all_comp_labels.append(i / (n - 1))
        if len(all_comp_preds) >= 10:
            cp = np.array(all_comp_preds)
            cl = np.array(all_comp_labels)
            mae = float(np.mean(np.abs(cp - cl)))
            mse = float(np.mean((cp - cl) ** 2))
            metrics["rollout_completion/mae"] = mae
            metrics["rollout_completion/mse"] = mse
            metrics["rollout_completion/pred_mean"] = float(np.mean(cp))
            metrics["rollout_completion/pred_std"] = float(np.std(cp))
            metrics["rollout_completion/label_mean"] = float(np.mean(cl))
            metrics["rollout_completion/n_frames"] = len(cp)
            metrics["rollout_completion/n_episodes"] = len(comp_preds_by_ep)
            # Pearson correlation — tracks whether the prediction grows with true progress.
            if np.std(cp) > 1e-8 and np.std(cl) > 1e-8:
                metrics["rollout_completion/pearson_r"] = float(
                    np.corrcoef(cp, cl)[0, 1]
                )
            logger.info("Completion pred: mae=%.3f mse=%.3f r=%.3f (%d frames, %d eps)",
                         mae, mse, metrics.get("rollout_completion/pearson_r", float("nan")),
                         len(cp), len(comp_preds_by_ep))

    # TTC (time-to-completion) prediction metrics: per-frame regression against
    # ground-truth in [0, 1]. Success: 1 − steps_left/600. Failure: 0.
    # Must match the training target in transforms.py::ComputeAuxTargets and
    # the sigmoid model head in pi_modified.py.
    if ttc_preds_by_ep:
        all_ttc_preds: list[float] = []
        all_ttc_labels: list[float] = []
        for ep_idx, preds_list in ttc_preds_by_ep.items():
            n = len(preds_list)
            if n < 2:
                continue
            ep_m = ep_meta.get(ep_idx, {})
            ep_success = ep_m.get("success", None)
            if ep_success is None:
                continue
            for i, pv in enumerate(preds_list):
                steps_left = n - 1 - i
                if ep_success:
                    label = 1.0 - steps_left / 600.0
                else:
                    label = 0.0
                all_ttc_preds.append(pv)
                all_ttc_labels.append(label)
        if len(all_ttc_preds) >= 10:
            tp = np.array(all_ttc_preds)
            tl = np.array(all_ttc_labels)
            mae = float(np.mean(np.abs(tp - tl)))
            mse = float(np.mean((tp - tl) ** 2))
            metrics["rollout_ttc/mae"] = mae
            metrics["rollout_ttc/mse"] = mse
            metrics["rollout_ttc/pred_mean"] = float(np.mean(tp))
            metrics["rollout_ttc/pred_std"] = float(np.std(tp))
            metrics["rollout_ttc/label_mean"] = float(np.mean(tl))
            metrics["rollout_ttc/n_frames"] = len(tp)
            metrics["rollout_ttc/n_episodes"] = len(ttc_preds_by_ep)
            if np.std(tp) > 1e-8 and np.std(tl) > 1e-8:
                metrics["rollout_ttc/pearson_r"] = float(np.corrcoef(tp, tl)[0, 1])
            logger.info("TTC pred: mae=%.3f mse=%.3f r=%.3f (%d frames, %d eps)",
                         mae, mse, metrics.get("rollout_ttc/pearson_r", float("nan")),
                         len(tp), len(ttc_preds_by_ep))

    # ─── Keypoint-distance head (Head 1) ─────────────────────────────────
    # Per-frame predicted normalized distance vs actual check_distances.
    # Prediction is 21-wide; ground truth is 7-wide. Only the per-garment slice
    # of the prediction is scored, using KEYPOINT_SLICES mapping.
    # Ground truth is capped at KEYPOINT_DISTANCE_CAP to mirror the training
    # target: values far above threshold are all "very far" and MAE should
    # not be dominated by the long tail.
    if kpt_preds_by_ep:
        from lehome_solution.constants import KEYPOINT_SLICES, KEYPOINT_DISTANCE_CAP
        kpt_errs: list[float] = []
        for ep_idx, preds_list in kpt_preds_by_ep.items():
            gt_id = int(ep_meta.get(ep_idx, {}).get("garment_type_id", -1))
            if gt_id not in KEYPOINT_SLICES:
                continue
            start, width = KEYPOINT_SLICES[gt_id]
            gt_list = kpt_gt_by_ep.get(ep_idx, [])
            for pred, gt in zip(preds_list, gt_list):
                pred_slice = pred[start : start + width]
                gt_slice = np.clip(gt[:width], 0.0, KEYPOINT_DISTANCE_CAP)
                valid = np.isfinite(pred_slice) & np.isfinite(gt_slice)
                if valid.any():
                    kpt_errs.extend(np.abs(pred_slice[valid] - gt_slice[valid]).tolist())
        if len(kpt_errs) >= 10:
            arr = np.asarray(kpt_errs, dtype=np.float32)
            metrics["rollout_keypoint/mae"] = float(np.mean(arr))
            metrics["rollout_keypoint/mse"] = float(np.mean(arr ** 2))
            metrics["rollout_keypoint/n_valid_slots"] = int(arr.size)
            logger.info(
                "Keypoint pred: mae=%.3f mse=%.3f (%d valid slot-observations)",
                metrics["rollout_keypoint/mae"], metrics["rollout_keypoint/mse"], arr.size,
            )

    # ─── WM-flow head (Head 3) — monitoring-only cond + uncond ──────────
    # For success: target = episode terminal success label (same as success_pred).
    # For completion: target = future_frac = min(i + 30, N - 1) / (N - 1) per ep
    #   (mirrors data-loader's completion_future definition at t + 30).
    # For keypoint: target = check_distances at frame min(i + 30, N - 1).
    _WM_H = 30

    def _log_wm_scalar(name, preds_by_ep, labels_by_ep):
        """Emit rollout_wm_flow_{name}/{accuracy,mae,...} metrics.

        ``preds_by_ep`` and ``labels_by_ep`` carry same-length per-episode lists.
        Binary success metrics get the _compute_binary_pred_metrics treatment
        to surface ROC-AUC; MSE-style targets get MAE + Pearson.
        """
        if not preds_by_ep:
            return
        flat_p: list[float] = []
        flat_l: list[float] = []
        for ep_idx, plist in preds_by_ep.items():
            llist = labels_by_ep.get(ep_idx, [])
            n = min(len(plist), len(llist))
            for i in range(n):
                pv, lv = plist[i], llist[i]
                if pv is None or lv is None or not np.isfinite(lv):
                    continue
                flat_p.append(float(pv))
                flat_l.append(float(lv))
        if len(flat_p) < 10:
            return
        p_arr = np.asarray(flat_p, dtype=np.float32)
        l_arr = np.asarray(flat_l, dtype=np.float32)
        # Distinguish binary-target (0/1) from continuous targets. Success is
        # binary; completion is continuous in [0, 1].
        is_binary = bool(np.all((l_arr == 0) | (l_arr == 1)))
        if is_binary:
            m = _compute_binary_pred_metrics(p_arr, l_arr, f"rollout_wm_flow_{name}")
            metrics.update(m)
            logger.info(
                "WM-flow %s: acc=%.3f auc=%.3f (%d frames)",
                name,
                m.get(f"rollout_wm_flow_{name}/accuracy", -1),
                m.get(f"rollout_wm_flow_{name}/roc_auc", -1),
                len(p_arr),
            )
        else:
            mae = float(np.mean(np.abs(p_arr - l_arr)))
            metrics[f"rollout_wm_flow_{name}/mae"] = mae
            metrics[f"rollout_wm_flow_{name}/pred_mean"] = float(np.mean(p_arr))
            metrics[f"rollout_wm_flow_{name}/pred_std"] = float(np.std(p_arr))
            metrics[f"rollout_wm_flow_{name}/label_mean"] = float(np.mean(l_arr))
            metrics[f"rollout_wm_flow_{name}/n_frames"] = int(p_arr.size)
            if np.std(p_arr) > 1e-8 and np.std(l_arr) > 1e-8:
                metrics[f"rollout_wm_flow_{name}/pearson_r"] = float(
                    np.corrcoef(p_arr, l_arr)[0, 1]
                )
            logger.info(
                "WM-flow %s: mae=%.3f r=%.3f (%d frames)",
                name, mae,
                metrics.get(f"rollout_wm_flow_{name}/pearson_r", float("nan")),
                len(p_arr),
            )

    # Build labels for WM-flow heads from the per-episode frame counts.
    #
    # Success head now outputs Δsuccess = ep_success − V̂(s_t), so the label
    # is the same per-frame delta (binary episode outcome minus per-frame
    # success_pred). Without per-frame success_pred we fall back to the old
    # constant-per-episode label, which will over-report MAE but still yields
    # a meaningful trend; the proper delta labels are preferred when
    # ``success_pred_stream_by_ep[ep]`` is available.
    wm_success_labels_by_ep: dict[int, list[float]] = {}
    wm_completion_labels_by_ep: dict[int, list[float]] = {}
    for ep_idx in list(wm_flow_success_cond_by_ep.keys()) + list(wm_flow_completion_cond_by_ep.keys()):
        if ep_idx in wm_success_labels_by_ep:
            continue
        ep_m = ep_meta.get(ep_idx, {})
        ep_success = ep_m.get("success", None)
        n = max(
            len(wm_flow_success_cond_by_ep.get(ep_idx, [])),
            len(wm_flow_completion_cond_by_ep.get(ep_idx, [])),
        )
        if n == 0:
            continue
        # Δsuccess target: ep_success − success_pred(t) per frame. Falls back
        # to constant binary if the per-frame success_pred stream is missing.
        if ep_success is not None:
            ep_success_bin = 1.0 if ep_success else 0.0
            sp_list = success_pred_stream_by_ep.get(ep_idx, [])
            if sp_list and len(sp_list) >= n:
                wm_success_labels_by_ep[ep_idx] = [
                    ep_success_bin - float(sp_list[i]) for i in range(n)
                ]
            else:
                wm_success_labels_by_ep[ep_idx] = [ep_success_bin] * n
        # Completion target: future fraction at t + 30 (mirrors data-loader logic).
        wm_completion_labels_by_ep[ep_idx] = [
            min(i + _WM_H, n - 1) / max(n, 1) for i in range(n)
        ]

    _log_wm_scalar("success_cond", wm_flow_success_cond_by_ep, wm_success_labels_by_ep)
    _log_wm_scalar("success_uncond", wm_flow_success_uncond_by_ep, wm_success_labels_by_ep)
    _log_wm_scalar("completion_cond", wm_flow_completion_cond_by_ep, wm_completion_labels_by_ep)
    _log_wm_scalar("completion_uncond", wm_flow_completion_uncond_by_ep, wm_completion_labels_by_ep)

    # WM-flow keypoint future MAE: predicted at frame i vs check_distances at
    # frame min(i + 30, N - 1). Uses per-garment slice like Head 1.
    def _log_wm_keypoint(name, preds_by_ep):
        if not preds_by_ep:
            return
        from lehome_solution.constants import KEYPOINT_SLICES, KEYPOINT_DISTANCE_CAP
        errs: list[float] = []
        for ep_idx, preds_list in preds_by_ep.items():
            gt_id = int(ep_meta.get(ep_idx, {}).get("garment_type_id", -1))
            if gt_id not in KEYPOINT_SLICES:
                continue
            start, width = KEYPOINT_SLICES[gt_id]
            gt_list = kpt_gt_by_ep.get(ep_idx, [])
            if not gt_list:
                continue
            n = len(gt_list)
            for i, pred in enumerate(preds_list):
                fi = min(i + _WM_H, n - 1)
                gt = gt_list[fi] if 0 <= fi < n else None
                if gt is None:
                    continue
                pred_slice = pred[start : start + width]
                # Cap GT to match training-side target clipping.
                gt_slice = np.clip(gt[:width], 0.0, KEYPOINT_DISTANCE_CAP)
                valid = np.isfinite(pred_slice) & np.isfinite(gt_slice)
                if valid.any():
                    errs.extend(np.abs(pred_slice[valid] - gt_slice[valid]).tolist())
        if len(errs) >= 10:
            arr = np.asarray(errs, dtype=np.float32)
            metrics[f"rollout_wm_flow_keypoint_{name}/mae"] = float(np.mean(arr))
            metrics[f"rollout_wm_flow_keypoint_{name}/n_valid_slots"] = int(arr.size)
            logger.info(
                "WM-flow keypoint %s: mae=%.3f (%d slots)",
                name, float(np.mean(arr)), arr.size,
            )

    _log_wm_keypoint("cond", wm_flow_keypoint_cond_by_ep)
    _log_wm_keypoint("uncond", wm_flow_keypoint_uncond_by_ep)

    # ─── Best-of-N candidate-spread diagnostics ─────────────────────────
    # Values are piecewise-constant per chunk (same for every frame in the
    # chunk), so de-dup via chunk boundaries (detected from success_pred
    # changes — same trick used by the advantage block above). One sample
    # per chunk → the per-chunk spread isn't inflated by chunk length.
    if bon_spread_stream_by_ep:
        per_chunk = {
            "spread": [], "std": [], "chosen": [], "mean": [], "min": [], "max": [],
            "n_valid": [],
        }
        for ep_idx, sp_list in success_pred_stream_by_ep.items():
            if has_metadata and ep_idx not in included_episodes:
                continue
            sp = np.asarray(sp_list, dtype=np.float32)
            if sp.size < 1:
                continue
            changes = np.abs(np.diff(sp)) > 1e-6
            boundaries = np.concatenate([[0], np.where(changes)[0] + 1])
            for stream_name, target in (
                ("spread", bon_spread_stream_by_ep),
                ("std", bon_std_stream_by_ep),
                ("chosen", bon_chosen_stream_by_ep),
                ("mean", bon_mean_stream_by_ep),
                ("min", bon_min_stream_by_ep),
                ("max", bon_max_stream_by_ep),
                ("n_valid", bon_n_valid_stream_by_ep),
            ):
                s = target.get(ep_idx, [])
                if not s:
                    continue
                for b in boundaries:
                    if 0 <= b < len(s):
                        per_chunk[stream_name].append(float(s[b]))

        # Only emit metrics if the server actually ran with best-of-N > 1
        # (otherwise every chunk has n_valid = 0 and the spread is NaN).
        n_valid_arr = np.asarray(per_chunk["n_valid"], dtype=np.float32)
        has_bon = n_valid_arr.size > 0 and float(n_valid_arr.max()) > 1
        if has_bon:
            # Filter to chunks that actually had multiple valid candidates.
            valid_mask_np = n_valid_arr > 1

            def _emit(name: str, vals: list[float], with_extras: bool):
                arr = np.asarray(vals, dtype=np.float32)
                if arr.size == 0:
                    return
                # Align to valid_mask_np if sizes match (chunk-aligned streams).
                if arr.size == valid_mask_np.size:
                    arr = arr[valid_mask_np]
                finite = np.isfinite(arr)
                if not finite.any():
                    return
                a = arr[finite]
                prefix = f"rollout_best_of_n/{name}"
                metrics[f"{prefix}/mean"] = float(a.mean())
                if with_extras:
                    metrics[f"{prefix}/std"] = float(a.std())
                    metrics[f"{prefix}/min"] = float(a.min())
                    metrics[f"{prefix}/max"] = float(a.max())
                metrics[f"{prefix}/n_chunks"] = int(a.size)

            # Spread is the primary "is best-of-N useful?" signal. Because
            # every candidate in a group has the same current-time joint
            # subtracted, spread / std are invariant to the delta shift.
            _emit("spread", per_chunk["spread"], with_extras=True)
            # Per-chunk std across candidates — correlated with spread but
            # less tail-sensitive. Also invariant to the delta shift.
            _emit("std", per_chunk["std"], with_extras=True)
            # Score of the chosen candidate (= per-chunk max). Now a DELTA:
            # WM-flow predicted future joint minus current-time joint
            # (`success_pred × completion_pred`). Positive chosen/mean means
            # the model believes the selected chunk improves its own estimate.
            _emit("chosen", per_chunk["chosen"], with_extras=True)
            # Mean / min of the N candidate scores per chunk (same delta
            # semantics as `chosen`).
            _emit("candidates_mean", per_chunk["mean"], with_extras=False)
            _emit("candidates_min", per_chunk["min"], with_extras=False)
            _emit("candidates_max", per_chunk["max"], with_extras=False)

            # Fraction of chunks that had multiple valid candidates. If < 1,
            # some chunks degraded to N=1 (e.g. WM-flow preds were NaN for
            # all but one candidate).
            metrics["rollout_best_of_n/multi_valid_frac"] = float(valid_mask_np.mean())

            logger.info(
                "best-of-N: spread mean=%.4f max=%.4f std_mean=%.4f chosen_mean=%.4f (%d chunks)",
                metrics.get("rollout_best_of_n/spread/mean", float("nan")),
                metrics.get("rollout_best_of_n/spread/max", float("nan")),
                metrics.get("rollout_best_of_n/std/mean", float("nan")),
                metrics.get("rollout_best_of_n/chosen/mean", float("nan")),
                metrics.get("rollout_best_of_n/spread/n_chunks", 0),
            )

    # ─── WM-flow LOCAL ADVANTAGE ────────────────────────────────────────
    # "Does the model think its chosen chunk improves its own estimate?"
    # At every chunk boundary (detected via changes in success_pred — which
    # is piecewise-constant within a chunk because it's propagated per-call),
    # we compute:
    #   adv_X_cond   = wm_flow_X_cond[b] - current_X[b]
    #   adv_X_uncond = wm_flow_X_uncond[b] - current_X[b]
    # for X ∈ {success, completion, joint = success × completion}.
    # Tracked metrics per advantage signal:
    #   /mean                         — is the model generally optimistic?
    #   /positive_frac                — fraction of chunks where adv > 0
    #   /corr_with_episode_success    — does adv predict eventual success?
    #   /corr_with_observed_delta     — does adv correlate with the actual
    #                                    change in current_X over the next
    #                                    chunk (i.e. after executing the
    #                                    predicted-forward N steps)?
    if success_pred_stream_by_ep and wm_flow_success_cond_by_ep:
        # Collect per-chunk-boundary samples across all included episodes.
        # The ``_avg`` variants are the elementwise mean of the paired cond /
        # uncond advantage — a balanced signal that doesn't lean on either the
        # advantage-visible (possibly shortcut-biased) or advantage-hidden
        # (possibly failure-miscalibrated) distribution alone.
        adv_buckets: dict[str, list[float]] = {
            k: [] for k in (
                "success_cond", "success_uncond", "success_avg",
                "completion_cond", "completion_uncond", "completion_avg",
                "joint_cond", "joint_uncond", "joint_avg",
            )
        }
        ep_success_per_sample: list[float] = []
        delta_buckets: dict[str, list[float]] = {
            "success": [], "completion": [], "joint": [],
        }

        for ep_idx in list(success_pred_stream_by_ep.keys()):
            if has_metadata and ep_idx not in included_episodes:
                continue
            sp = np.asarray(success_pred_stream_by_ep.get(ep_idx, []), dtype=np.float32)
            cp = np.asarray(completion_pred_stream_by_ep.get(ep_idx, []), dtype=np.float32)
            wmsc = np.asarray(wm_flow_success_cond_by_ep.get(ep_idx, []), dtype=np.float32)
            wmsu = np.asarray(wm_flow_success_uncond_by_ep.get(ep_idx, []), dtype=np.float32)
            wmcc = np.asarray(wm_flow_completion_cond_by_ep.get(ep_idx, []), dtype=np.float32)
            wmcu = np.asarray(wm_flow_completion_uncond_by_ep.get(ep_idx, []), dtype=np.float32)
            # Must have all 4 WM-flow streams + sp + cp, same length.
            L = len(sp)
            if L < 2 or len(cp) != L or len(wmsc) != L or len(wmsu) != L \
               or len(wmcc) != L or len(wmcu) != L:
                continue
            # Chunk boundaries: where success_pred changes value (1e-6 tol).
            changes = np.abs(np.diff(sp)) > 1e-6
            boundaries = np.concatenate([[0], np.where(changes)[0] + 1])
            ep_success = ep_meta.get(ep_idx, {}).get("success", None)
            ep_success_bin = (
                1.0 if ep_success else 0.0
            ) if ep_success is not None else None

            for i, b in enumerate(boundaries):
                # Require all WM-flow predictions at b to be finite.
                if not (np.isfinite(wmsc[b]) and np.isfinite(wmsu[b])
                        and np.isfinite(wmcc[b]) and np.isfinite(wmcu[b])):
                    continue
                sp_b, cp_b = float(sp[b]), float(cp[b])
                joint_b = sp_b * cp_b
                wmsc_b = float(wmsc[b]); wmsu_b = float(wmsu[b])
                wmcc_b = float(wmcc[b]); wmcu_b = float(wmcu[b])
                # wm_flow_success_* are Δsuccess directly (= true_success −
                # V̂(s) during training), so they ARE the advantage — no
                # extra ``- sp_b`` subtraction here.
                adv_buckets["success_cond"].append(wmsc_b)
                adv_buckets["success_uncond"].append(wmsu_b)
                adv_buckets["success_avg"].append(0.5 * (wmsc_b + wmsu_b))
                adv_buckets["completion_cond"].append(wmcc_b - cp_b)
                adv_buckets["completion_uncond"].append(wmcu_b - cp_b)
                adv_buckets["completion_avg"].append(0.5 * (wmcc_b + wmcu_b) - cp_b)
                # Joint advantage: additive decomposition (Δsuccess +
                # Δcompletion). Success head already outputs the delta;
                # completion head outputs the raw prediction so we subtract
                # the current-time baseline. Purely diagnostic — not used
                # for best-of-N selection anymore.
                adv_buckets["joint_cond"].append(wmsc_b + (wmcc_b - cp_b))
                adv_buckets["joint_uncond"].append(wmsu_b + (wmcu_b - cp_b))
                adv_buckets["joint_avg"].append(
                    0.5 * (wmsc_b + wmsu_b) + 0.5 * (wmcc_b + wmcu_b) - cp_b
                )
                if ep_success_bin is not None:
                    ep_success_per_sample.append(ep_success_bin)
                else:
                    ep_success_per_sample.append(np.nan)
                # Observed delta over the executed chunk (to the NEXT boundary).
                if i + 1 < len(boundaries):
                    b_next = boundaries[i + 1]
                    delta_buckets["success"].append(float(sp[b_next]) - sp_b)
                    delta_buckets["completion"].append(float(cp[b_next]) - cp_b)
                    delta_buckets["joint"].append(
                        float(sp[b_next]) * float(cp[b_next]) - joint_b
                    )
                else:
                    delta_buckets["success"].append(np.nan)
                    delta_buckets["completion"].append(np.nan)
                    delta_buckets["joint"].append(np.nan)

        # Map each advantage signal to the observed-delta stream it should
        # correlate against (success_* ↔ Δsuccess, completion_* ↔ Δcompletion,
        # joint_* ↔ Δjoint).
        _delta_for = {
            "success_cond": "success", "success_uncond": "success", "success_avg": "success",
            "completion_cond": "completion", "completion_uncond": "completion", "completion_avg": "completion",
            "joint_cond": "joint", "joint_uncond": "joint", "joint_avg": "joint",
        }

        ep_s_arr = np.asarray(ep_success_per_sample, dtype=np.float32)
        logged_any = False
        for name, vals in adv_buckets.items():
            arr = np.asarray(vals, dtype=np.float32)
            if arr.size < 10:
                continue
            prefix = f"rollout_wm_flow_advantage/{name}"
            metrics[f"{prefix}/mean"] = float(arr.mean())
            metrics[f"{prefix}/positive_frac"] = float((arr > 0).mean())
            metrics[f"{prefix}/n_samples"] = int(arr.size)

            # Correlation with episode success.
            valid_es = np.isfinite(ep_s_arr) & np.isfinite(arr)
            if valid_es.sum() >= 10:
                a_v = arr[valid_es]
                e_v = ep_s_arr[valid_es]
                if np.std(a_v) > 1e-8 and np.std(e_v) > 1e-8:
                    metrics[f"{prefix}/corr_with_episode_success"] = float(
                        np.corrcoef(a_v, e_v)[0, 1]
                    )

            # Correlation with observed delta over the executed chunk.
            delta_arr = np.asarray(delta_buckets[_delta_for[name]], dtype=np.float32)
            valid_d = np.isfinite(delta_arr) & np.isfinite(arr)
            if valid_d.sum() >= 10:
                a_v = arr[valid_d]
                d_v = delta_arr[valid_d]
                if np.std(a_v) > 1e-8 and np.std(d_v) > 1e-8:
                    metrics[f"{prefix}/corr_with_observed_delta"] = float(
                        np.corrcoef(a_v, d_v)[0, 1]
                    )
            logged_any = True

        if logged_any:
            # Concise line: the 6 /mean values at a glance.
            logger.info(
                "WM-flow advantage means: sc=%.4f su=%.4f cc=%.4f cu=%.4f jc=%.4f ju=%.4f (n=%d)",
                metrics.get("rollout_wm_flow_advantage/success_cond/mean", float("nan")),
                metrics.get("rollout_wm_flow_advantage/success_uncond/mean", float("nan")),
                metrics.get("rollout_wm_flow_advantage/completion_cond/mean", float("nan")),
                metrics.get("rollout_wm_flow_advantage/completion_uncond/mean", float("nan")),
                metrics.get("rollout_wm_flow_advantage/joint_cond/mean", float("nan")),
                metrics.get("rollout_wm_flow_advantage/joint_uncond/mean", float("nan")),
                metrics.get("rollout_wm_flow_advantage/success_cond/n_samples", 0),
            )

    return metrics


def _load_eval_summary(eval_dataset_root: str) -> dict | None:
    """Load eval_summary.json from a dataset's parent directory."""
    summary_path = Path(eval_dataset_root).parent / "eval_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def _summary_to_metrics(summary: dict, prefix: str) -> dict:
    """Convert an eval_summary.json into wandb metrics dict with given prefix."""
    overall = summary.get("overall", {})
    metrics = {
        f"{prefix}/success_rate": overall.get("success_rate", 0),
        f"{prefix}/avg_reward": overall.get("avg_reward", 0),
        f"{prefix}/weighted_sr": overall.get("weighted_sr", 0),
        f"{prefix}/weighted_reward": overall.get("weighted_reward", 0),
    }
    for gtype, info in summary.get("by_type", {}).items():
        metrics[f"{prefix}/sr_{gtype}"] = info.get("success_rate", 0)
        if "avg_reward" in info:
            metrics[f"{prefix}/reward_{gtype}"] = info["avg_reward"]
    return metrics


def _build_inference_prior_log(state: dict) -> dict:
    """Build wandb log dict with inference prior charts and posterior summary metrics.

    Returns a dict suitable for merging into a run.log() call. Empty dict on failure.
    """
    prior_path = state.get("inference_prior_file")
    if not prior_path or not Path(prior_path).exists():
        return {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb
        from lehome_solution.eval.inference_optimization import (
            load_prior, get_garment_type_priors, get_posterior_summary,
        )
        sys.path.insert(0, str(Path(__file__).parent / "analysis"))
        from plot_inference_prior import plot_inference_prior

        prior = load_prior(prior_path)
        log = {}

        # Posterior summary metrics (per-garment-type best configs, entropy, etc.)
        log.update(get_posterior_summary(prior))

        # Combined chart
        fig = plot_inference_prior(prior)
        log["inference_param_opt/prior_combined"] = wandb.Image(fig)
        plt.close(fig)

        # Per-garment-type charts
        gt_priors = get_garment_type_priors(prior)
        for gt, gt_prior in gt_priors.items():
            n_eps = gt_prior.get("n_episodes", 0)
            fig_gt = plot_inference_prior(gt_prior, title_suffix=f"{gt} (cum n={n_eps})")
            log[f"inference_param_opt/prior_{gt}"] = wandb.Image(fig_gt)
            plt.close(fig_gt)

        return log
    except Exception as e:
        logger.warning("Failed to build inference prior log: %s", e, exc_info=True)
        return {}


def _log_single_rollout_to_wandb(
    pipeline_cfg: RLPipelineConfig, state: dict,
    eval_dataset_root: str, strategy: str, model_step: int,
):
    """Log a single rollout's metrics to wandb. Used by rollout worker."""
    if not pipeline_cfg.wandb_enabled or state.get("wandb_run_id") is None:
        return

    try:
        import wandb
    except ImportError:
        return

    summary = _load_eval_summary(eval_dataset_root)
    if summary is None:
        logger.warning("No eval_summary.json for %s, skipping wandb log", eval_dataset_root)
        return

    prefix = "rollout" if strategy == "random" else f"rollout_extra/{strategy}"
    metrics = _summary_to_metrics(summary, prefix)
    metrics["rollout/model_step"] = model_step

    value_metrics = _compute_success_pred_metrics(eval_dataset_root)
    if value_metrics:
        metrics.update(value_metrics)

    try:
        run = wandb.init(
            id=state["wandb_run_id"],
            resume="allow",
            project=pipeline_cfg.project_name,
            settings=wandb.Settings(init_timeout=180),
        )
        # Rollout metrics are charted against rollout/model_step (step_metric).
        # Do NOT pass step= to run.log: wandb's global step must be monotonic
        # per run, and concurrent workers with different checkpoints (and even
        # the same worker across restarts) trip that guard and cause silent,
        # permanent data loss on the run.
        run.define_metric("rollout/*", step_metric="rollout/model_step")
        run.define_metric("rollout_extra/*", step_metric="rollout/model_step")
        run.define_metric("rollout_value/*", step_metric="rollout/model_step")
        run.define_metric("rollout_completion/*", step_metric="rollout/model_step")
        run.define_metric("inference_param_opt/*", step_metric="rollout/model_step")

        # Merge inference prior charts + metrics into the same log call
        # so they share rollout/model_step and don't create duplicate panels.
        if strategy in _INFERENCE_OPT_INCLUDED_ROLLOUT_TYPES:
            metrics.update(_build_inference_prior_log(state))

        run.log(metrics)
        run.finish()
        logger.info("Logged %s rollout metrics to wandb (model_step=%d)", strategy, model_step)
    except Exception as e:
        logger.warning("Failed to log rollout to wandb: %s", e, exc_info=True)


def log_eval_to_wandb(pipeline_cfg: RLPipelineConfig, state: dict, eval_dataset_root: str):
    """Log rollout metrics to the unified wandb run.

    With multi-strategy rollouts:
    - "random" strategy -> main rollout/ metrics
    - Other strategies -> rollout_extra/{strategy}/ metrics
    - Success pred metrics aggregated across all current-iteration datasets
    - Falls back to eval_dataset_root if no strategy info is available
    """
    if not pipeline_cfg.wandb_enabled or state["wandb_run_id"] is None:
        return

    try:
        import wandb
    except ImportError:
        return

    # Group current-iteration datasets by strategy
    # Current-iteration datasets have sampling_share == 1.0
    from collections import defaultdict
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for ds in state["rl_datasets"]:
        if ds["sampling_share"] == 1.0:
            strategy = ds.get("strategy", "random")
            by_strategy[strategy].append(ds)

    # If no share==1.0 datasets found, fall back to the provided eval_dataset_root
    if not by_strategy:
        summary = _load_eval_summary(eval_dataset_root)
        if summary is None:
            logger.warning("No eval_summary.json at %s", Path(eval_dataset_root).parent)
            return
        by_strategy["random"] = [{"root": eval_dataset_root}]

    try:
        run = wandb.init(
            id=state["wandb_run_id"],
            resume="allow",
            project=pipeline_cfg.project_name,
            settings=wandb.Settings(init_timeout=180),
        )

        # Define custom x-axis so rollout metrics use model_step, not the global
        # training step (avoids silent data loss from non-monotonic steps).
        run.define_metric("rollout/*", step_metric="rollout/model_step")
        run.define_metric("rollout_extra/*", step_metric="rollout/model_step")
        run.define_metric("rollout_value/*", step_metric="rollout/model_step")
        run.define_metric("rollout_completion/*", step_metric="rollout/model_step")
        run.define_metric("inference_param_opt/*", step_metric="rollout/model_step")

        step = state["current_train_steps"]
        iteration = state["iteration"]

        metrics = {
            "rollout/iteration": iteration,
            "rollout/model_step": step,
            "pipeline/num_rl_datasets": len(state["rl_datasets"]),
            "pipeline/bc_share": state["bc_sampling_share"],
        }

        # Dataset shares
        for i, ds in enumerate(state["rl_datasets"]):
            metrics[f"pipeline/rl_share_{i}"] = ds["sampling_share"]

        # Log each strategy's metrics
        for strategy, datasets in by_strategy.items():
            # Use the first dataset with a valid summary (usually only one per strategy)
            for ds in datasets:
                summary = _load_eval_summary(ds["root"])
                if summary is not None:
                    break
            else:
                continue

            if strategy == "random":
                # Main metrics under rollout/
                metrics.update(_summary_to_metrics(summary, "rollout"))
            else:
                # Extra strategies under rollout_extra/{strategy}/
                metrics.update(_summary_to_metrics(summary, f"rollout_extra/{strategy}"))

        # Success prediction quality metrics aggregated across all current-iteration datasets
        # Use random strategy dataset for success pred metrics (main one)
        random_roots = [ds["root"] for ds in by_strategy.get("random", [])]
        if random_roots:
            value_metrics = _compute_success_pred_metrics(random_roots[0])
            if value_metrics:
                metrics.update(value_metrics)

        # Merge inference prior charts + summary into the same log call
        metrics.update(_build_inference_prior_log(state))

        run.log(metrics)

        # Upload 1 random video per (strategy, garment_type) combo
        by_key = defaultdict(list)  # (strategy, garment_type) -> [video_paths]
        for strategy, datasets in by_strategy.items():
            for ds in datasets:
                ds_run_dir = Path(ds["root"]).parent
                for vf in ds_run_dir.glob("*.mp4"):
                    parts = vf.stem.split("_")
                    gtype = "_".join(parts[:2]) if len(parts) >= 2 else vf.stem
                    by_key[(strategy, gtype)].append(vf)
        video_log = {}
        for (strategy, gtype), vfs in by_key.items():
            vf = random.choice(vfs)
            try:
                video_log[f"media/rollout_video_{strategy}_{gtype}"] = wandb.Video(
                    str(vf), fps=30, format="mp4",
                )
            except Exception:
                pass
        if video_log:
            run.log(video_log)

        run.finish()
        logger.info("Logged rollout metrics to wandb (step=%d, iter=%d)", step, iteration)
    except Exception as e:
        logger.warning("Failed to log to wandb: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Download missing datasets from HF
# ---------------------------------------------------------------------------

def download_missing_datasets(pipeline_cfg: RLPipelineConfig, state: dict,
                              sync_client: "HFSyncClient | None" = None):
    """Download any RL/dagger datasets missing locally from the HF dataset repo.

    Derives the HF path from the local root path (extracts rollout_id and
    subdirectory). Skips datasets that already have meta/info.json locally.
    """
    hf_repo = pipeline_cfg.hf_dataset_repo
    if not hf_repo:
        return

    all_datasets = list(state.get("rl_datasets", []))
    all_datasets.extend(state.get("dagger_datasets", []))

    missing = []
    for ds in all_datasets:
        root = ds["root"]
        if (Path(root) / "meta" / "info.json").exists():
            continue
        missing.append(ds)

    if not missing:
        return

    if not sync_client or not sync_client.enabled or not sync_client.is_daemon_alive():
        logger.warning("No HF sync daemon — skipping download of %d missing datasets", len(missing))
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_DOWNLOAD_MISSING
    logger.info("Submitting download of %d missing datasets (non-blocking)", len(missing))
    try:
        sync_client.submit(OP_DOWNLOAD_MISSING, {
            "repo_id": hf_repo,
            "datasets": [{"root": ds["root"]} for ds in missing],
        })
    except Exception as e:
        logger.error("Daemon download_missing submit failed: %s", e)


def _wait_for_missing_datasets(state: dict):
    """Block until all RL/dagger datasets exist locally (have meta/info.json).

    Polls every 30s.  Logs progress.  Waits indefinitely.
    """
    all_datasets = list(state.get("rl_datasets", []))
    all_datasets.extend(state.get("dagger_datasets", []))

    total = len(all_datasets)
    if total == 0:
        return

    # Quick check — maybe everything is already present
    missing = [ds for ds in all_datasets
               if not (Path(ds["root"]) / "meta" / "info.json").exists()]
    if not missing:
        return

    logger.info("Waiting for %d / %d datasets to download ...", len(missing), total)

    import time as _time
    while True:
        _time.sleep(30)
        missing = [ds for ds in all_datasets
                   if not (Path(ds["root"]) / "meta" / "info.json").exists()]
        present = total - len(missing)
        logger.info("  datasets ready: %d / %d  (still waiting for %d)",
                    present, total, len(missing))
        if not missing:
            logger.info("All datasets available locally.")
            return


# ---------------------------------------------------------------------------
# HuggingFace uploads
# ---------------------------------------------------------------------------

HIGH_SR_THRESHOLD = 0.8


def _maybe_upload_high_sr_checkpoint(
    eval_dataset_root: str,
    step: int,
    checkpoint_path: str,
    hf_model_repo: str,
    cfg: "RLPipelineConfig",
    sync_client: "HFSyncClient",
):
    """Upload a numbered checkpoint if the full rollout achieved high SR.

    Only uploads params+assets (numbered_only=True), never overwrites 'latest'.
    Skips if this step was already uploaded as a numbered checkpoint.
    """
    summary = _load_eval_summary(eval_dataset_root)
    if summary is None:
        return

    overall_sr = summary.get("overall", {}).get("success_rate", 0)
    if overall_sr <= HIGH_SR_THRESHOLD:
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_CHECKPOINT

    # Rollout worker downloads checkpoints to _hf_checkpoints/step_{N}/
    # but upload_checkpoint_to_hf expects checkpoint_dir/{step}/.
    # Create a symlink {parent}/{step} -> step_{step}/ so the upload function
    # can find it at the expected path.
    ckpt_p = Path(checkpoint_path)
    parent = ckpt_p.parent
    step_link = parent / str(step)
    if not step_link.exists() and ckpt_p.exists():
        try:
            step_link.symlink_to(ckpt_p.name)
        except OSError:
            logger.warning("Could not create symlink for numbered checkpoint upload")
            return

    logger.info(
        "Full rollout SR=%.3f > %.2f at step %d — uploading numbered checkpoint to %s",
        overall_sr, HIGH_SR_THRESHOLD, step, hf_model_repo,
    )
    sync_client.submit(OP_UPLOAD_CHECKPOINT, {
        "checkpoint_dir": str(parent),
        "step": step,
        "repo_id": hf_model_repo,
        "keep_period": cfg.keep_period,
        "numbered_only": True,
    })


def upload_checkpoint(pipeline_cfg: RLPipelineConfig, state: dict,
                      sync_client: "HFSyncClient | None" = None):
    """Upload checkpoint to HuggingFace Hub if configured."""
    repo_id = pipeline_cfg.hf_model_repo
    if not repo_id:
        return

    step = state["latest_checkpoint_step"]
    if step is None:
        return

    if not sync_client or not sync_client.enabled:
        logger.warning("No HF sync daemon — skipping checkpoint upload for step %d", step)
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_CHECKPOINT
    sync_client.submit(OP_UPLOAD_CHECKPOINT, {
        "checkpoint_dir": state["checkpoint_dir"],
        "step": step,
        "repo_id": repo_id,
        "keep_period": pipeline_cfg.keep_period,
    })


def _upload_wandb_id_to_hf(pipeline_cfg: RLPipelineConfig, state: dict,
                           sync_client: "HFSyncClient | None" = None):
    """Upload wandb_id.txt to HF model repo root for distributed workers.

    Synchronous — small file, and a silent daemon-side failure here leaves
    rollout workers without a run ID forever. Log success/failure loudly.
    """
    repo_id = pipeline_cfg.hf_model_repo
    wandb_id_path = str(Path(state["checkpoint_dir"]) / "wandb_id.txt")
    if not repo_id:
        return
    if not Path(wandb_id_path).exists():
        logger.warning("wandb_id.txt not found at %s — cannot upload to HF", wandb_id_path)
        return

    try:
        from lehome_solution.training.hf_upload import upload_wandb_id_to_hf as _direct_upload
        ok = _direct_upload(wandb_id_path, repo_id)
        if ok:
            logger.info("Uploaded wandb_id.txt to HF (%s)", repo_id)
        else:
            logger.warning("Direct upload of wandb_id.txt to %s returned False", repo_id)
    except Exception as e:
        logger.warning("Direct upload of wandb_id.txt failed: %s", e, exc_info=True)


def _watch_and_upload_wandb_id(pipeline_cfg: RLPipelineConfig, state: dict,
                               sync_client: "HFSyncClient | None",
                               stop_event: "threading.Event"):
    """Background poll: upload wandb_id.txt to HF as soon as train.py writes it.

    Prevents rollout workers from waiting an entire warmup before they can
    resume logging to the shared wandb run.
    """
    wandb_id_path = Path(state["checkpoint_dir"]) / "wandb_id.txt"
    repo_id = pipeline_cfg.hf_model_repo
    if not repo_id or not sync_client or not sync_client.enabled:
        return
    # Snapshot mtime so we re-upload if train.py rewrites the file.
    initial_mtime = wandb_id_path.stat().st_mtime if wandb_id_path.exists() else None
    while not stop_event.is_set():
        if wandb_id_path.exists():
            mtime = wandb_id_path.stat().st_mtime
            if mtime != initial_mtime:
                try:
                    _upload_wandb_id_to_hf(pipeline_cfg, state, sync_client=sync_client)
                    logger.info("Uploaded wandb_id.txt to HF early (during training)")
                except Exception as e:
                    logger.warning("Early wandb_id upload failed: %s", e)
                return
        stop_event.wait(5.0)


def _upload_success_rates_to_hf(pipeline_cfg: RLPipelineConfig, state: dict,
                                sync_client: "HFSyncClient | None" = None):
    """Upload success_rates.json to HF model repo root after advantage recomputation."""
    repo_id = pipeline_cfg.hf_model_repo
    if not repo_id:
        return

    from lehome_solution.eval.rollout_strategies import SUCCESS_RATES_FILE
    for ds in reversed(state.get("rl_datasets", [])):
        sr_path = Path(ds["root"]) / SUCCESS_RATES_FILE
        if sr_path.exists():
            if not sync_client or not sync_client.enabled:
                return

            from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_ASSET
            sync_client.submit(OP_UPLOAD_ASSET, {
                "asset_type": "success_rates",
                "path": str(sr_path),
                "repo_id": repo_id,
            })
            return


def _upload_inference_prior_to_hf(pipeline_cfg: RLPipelineConfig, state: dict,
                                  sync_client: "HFSyncClient | None" = None):
    """Upload inference_prior.json to HF model repo root. Only rollout workers call this."""
    repo_id = pipeline_cfg.hf_model_repo
    prior_path = state.get("inference_prior_file")
    if not repo_id or not prior_path or not Path(prior_path).exists():
        return

    if not sync_client or not sync_client.enabled:
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_ASSET
    sync_client.submit(OP_UPLOAD_ASSET, {
        "asset_type": "inference_prior",
        "path": prior_path,
        "repo_id": repo_id,
    })


def _upload_pipeline_state_to_hf(pipeline_cfg: RLPipelineConfig, state: dict,
                                 state_path: Path,
                                 sync_client: "HFSyncClient | None" = None):
    """Upload pipeline_state.json to HF model repo root after training."""
    repo_id = pipeline_cfg.hf_model_repo
    if not repo_id or not state_path.exists():
        return

    if not sync_client or not sync_client.enabled:
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_ASSET
    sync_client.submit(OP_UPLOAD_ASSET, {
        "asset_type": "pipeline_state",
        "path": str(state_path),
        "repo_id": repo_id,
    })


def _download_model_assets_from_hf(pipeline_cfg: RLPipelineConfig, state: dict,
                                   sync_client: "HFSyncClient | None" = None):
    """Download inference prior, success rates, and wandb_id from HF repo root."""
    repo_id = pipeline_cfg.hf_model_repo
    if not repo_id:
        return

    if not sync_client or not sync_client.enabled or not sync_client.is_daemon_alive():
        logger.warning("No HF sync daemon — skipping model asset downloads")
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_DOWNLOAD_ASSET
    prior_path = state.get("inference_prior_file")
    checkpoint_dir = state.get("checkpoint_dir")

    if prior_path and not Path(prior_path).exists():
        try:
            sync_client.submit(OP_DOWNLOAD_ASSET, {
                "repo_id": repo_id,
                "asset_name": "inference_prior.json",
                "local_path": prior_path,
            })
        except Exception as e:
            logger.warning("Daemon download inference_prior failed: %s", e)

    if checkpoint_dir:
        sr_dest = str(Path(checkpoint_dir) / "success_rates.json")
        try:
            # Block until download completes — curriculum sampling needs SR data immediately
            result = sync_client.submit_and_wait(OP_DOWNLOAD_ASSET, {
                "repo_id": repo_id,
                "asset_name": "success_rates.json",
                "local_path": sr_dest,
            }, timeout=30)
            if result and result.result and not result.result.get("success"):
                logger.warning("success_rates.json not available on HF (will use local if exists)")
        except Exception as e:
            logger.warning("Daemon download success_rates failed: %s", e)

        wandb_dest = str(Path(checkpoint_dir) / "wandb_id.txt")
        if not Path(wandb_dest).exists():
            try:
                # Block until download completes — wandb logging needs the run ID
                sync_client.submit_and_wait(OP_DOWNLOAD_ASSET, {
                    "repo_id": repo_id,
                    "asset_name": "wandb_id.txt",
                    "local_path": wandb_dest,
                }, timeout=30)
            except Exception as e:
                logger.warning("Daemon download wandb_id failed: %s", e)


# ---------------------------------------------------------------------------
# Failure state management
# ---------------------------------------------------------------------------

def _collect_physics_states(
    eval_run_dirs: list[str], persistent_dir: Path,
    subdir: str = "success", legacy_subdir: str | None = "success_states",
    label: str = "success",
):
    """Copy new physics state NPZ+JSON pairs from eval runs to a persistent directory.

    Args:
        subdir: subdirectory under physics_states/ to look for states
        legacy_subdir: optional legacy directory name to also check
        label: human-readable label for log messages
    """
    persistent_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    collected = 0
    for eval_dir in eval_run_dirs:
        base = Path(eval_dir).parent if Path(eval_dir).name == "eval_dataset" else Path(eval_dir)
        candidate_dirs = [base / "physics_states" / subdir]
        if legacy_subdir:
            candidate_dirs.append(base / legacy_subdir)
        for ss_dir in candidate_dirs:
            if not ss_dir.exists():
                continue
            for npz_path in ss_dir.glob("*.npz"):
                dest = persistent_dir / npz_path.name
                if not dest.exists():
                    shutil.copy2(str(npz_path), str(dest))
                    json_src = npz_path.with_suffix(".json")
                    if json_src.exists():
                        shutil.copy2(str(json_src), str(dest.with_suffix(".json")))
                    collected += 1
    if collected:
        logger.info("Collected %d new %s states -> %s (%d total)",
                    collected, label, persistent_dir, len(list(persistent_dir.glob("*.npz"))))


def _collect_success_states(eval_run_dirs: list[str], persistent_dir: Path):
    """Copy new success state NPZ+JSON pairs from eval runs to the persistent directory."""
    _collect_physics_states(eval_run_dirs, persistent_dir, subdir="success",
                           legacy_subdir="success_states", label="success")


def _collect_semi_success_states(eval_run_dirs: list[str], persistent_dir: Path):
    """Copy new semi-success state NPZ+JSON pairs from eval runs to the persistent directory."""
    _collect_physics_states(eval_run_dirs, persistent_dir, subdir="semi_success",
                           legacy_subdir=None, label="semi-success")


def _collect_failure_states(eval_run_dirs: list[str], persistent_dir: Path):
    """Copy new failure state NPZ+JSON pairs from eval runs to the persistent directory.

    Looks in both new (physics_states/failure/) and legacy (failure_states/) locations.
    """
    _collect_physics_states(eval_run_dirs, persistent_dir, subdir="failure",
                           legacy_subdir="failure_states", label="failure")


def _remove_solved_failures(eval_run_dirs: list[str], persistent_dir: Path):
    """Mark solved failure states as consumed and delete them.

    Matches NPZ filenames ({garment}_seed{S}_ep{I}.npz) against episode
    metadata from the latest eval runs. If a garment+seed combo succeeded,
    its NPZ+JSON pair is removed from the persistent dir.
    """
    if not persistent_dir.exists():
        return

    from lehome_solution.eval.dataset_writer import EVAL_EPISODE_META_DIR
    from lehome_solution.eval.rollout_strategies import _read_state_metadata

    # Collect all successful (garment, seed) pairs from eval runs
    solved = set()
    for eval_dir in eval_run_dirs:
        ds_path = Path(eval_dir)
        meta_dir = ds_path / EVAL_EPISODE_META_DIR
        if not meta_dir.exists():
            continue
        for mf in meta_dir.glob("episode_*.json"):
            try:
                with open(mf) as f:
                    meta = json.load(f)
                if meta.get("success", False):
                    garment = meta.get("garment", "")
                    seed = meta.get("seed", 0)
                    solved.add((garment, seed))
            except (json.JSONDecodeError, OSError):
                continue

    if not solved:
        return

    # Remove solved NPZ+JSON pairs
    removed = 0
    for npz_path in list(persistent_dir.glob("*.npz")):
        try:
            meta = _read_state_metadata(npz_path)
            if meta is None:
                continue
            garment = meta.get("garment", "")
            seed = meta.get("seed", 0)
            if (garment, seed) in solved:
                npz_path.unlink()
                json_path = npz_path.with_suffix(".json")
                if json_path.exists():
                    json_path.unlink()
                removed += 1
                logger.info("Removed solved failure state: %s", npz_path.name)
        except Exception as e:
            logger.warning("Failed to check/remove failure state %s: %s", npz_path.name, e)

    if removed:
        remaining = len(list(persistent_dir.glob("*.npz")))
        logger.info("Removed %d solved failure states (%d remaining)", removed, remaining)


def _upload_failure_states(pipeline_cfg: RLPipelineConfig, persistent_dir: Path,
                           sync_client: "HFSyncClient | None" = None):
    """Upload persistent failure states to HF if configured."""
    repo_id = pipeline_cfg.hf_dataset_repo
    if not repo_id or not persistent_dir.exists():
        return
    if not sync_client or not sync_client.enabled:
        logger.warning("No HF sync daemon — skipping failure states upload")
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_STATES
    sync_client.submit(OP_UPLOAD_STATES, {
        "kind": "failure",
        "states_dir": str(persistent_dir),
        "repo_id": repo_id,
    })


def _upload_success_states(pipeline_cfg: RLPipelineConfig, persistent_dir: Path,
                            sync_client: "HFSyncClient | None" = None):
    """Upload persistent success states to HF if configured."""
    repo_id = pipeline_cfg.hf_dataset_repo
    if not repo_id or not persistent_dir.exists():
        return
    if not sync_client or not sync_client.enabled:
        logger.warning("No HF sync daemon — skipping success states upload")
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_STATES
    sync_client.submit(OP_UPLOAD_STATES, {
        "kind": "success",
        "states_dir": str(persistent_dir),
        "repo_id": repo_id,
    })


def _upload_semi_success_states(pipeline_cfg: RLPipelineConfig, persistent_dir: Path,
                                sync_client: "HFSyncClient | None" = None):
    """Upload persistent semi-success states to HF if configured."""
    repo_id = pipeline_cfg.hf_dataset_repo
    if not repo_id or not persistent_dir.exists():
        return
    if not sync_client or not sync_client.enabled:
        logger.warning("No HF sync daemon — skipping semi-success states upload")
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_STATES
    sync_client.submit(OP_UPLOAD_STATES, {
        "kind": "semi_success",
        "states_dir": str(persistent_dir),
        "repo_id": repo_id,
    })


def sync_dagger_datasets(pipeline_cfg: RLPipelineConfig, state: dict,
                         sync_client: "HFSyncClient | None" = None) -> list[str]:
    """Download latest dagger datasets from HF and add to state.

    Called after rollout but before advantages, so dagger data gets
    advantage computation alongside RL rollout data.

    Returns list of dagger eval_dataset roots (for advantage computation).
    """
    repo_id = pipeline_cfg.hf_dataset_repo
    if not repo_id:
        return []

    if not sync_client or not sync_client.enabled or not sync_client.is_daemon_alive():
        logger.warning("No HF sync daemon — skipping dagger dataset sync")
        return _scan_local_dagger(state)

    from lehome_solution.distributed.hf_sync_protocol import OP_SYNC_DAGGER
    try:
        sync_client.submit(OP_SYNC_DAGGER, {
            "repo_id": repo_id,
            "checkpoint_dir": state.get("checkpoint_dir"),
        })
    except Exception as e:
        logger.error("Daemon sync_dagger submit failed: %s", e)
    # Scan local dirs for any dagger datasets (daemon downloads arrive asynchronously)
    return _scan_local_dagger(state)


def _scan_local_dagger(state: dict) -> list[str]:
    """Scan checkpoint-local dagger download dirs and add new datasets to state."""
    # Download dagger datasets to checkpoint dir so each run is self-contained
    checkpoint_dir = state.get("checkpoint_dir")
    if not checkpoint_dir:
        return []
    dagger_local_base = Path(checkpoint_dir) / "_hf_dagger"
    if not dagger_local_base.exists():
        return []

    dagger_roots = []
    # Find all eval_dataset_success dirs under dagger/session_*/
    for ds_dir in sorted(dagger_local_base.glob("dagger/session_*/eval_dataset_success")):
        if any(ds_dir.glob("data/chunk-*/*.parquet")):
            dagger_roots.append(str(ds_dir))

    if not dagger_roots:
        return []

    # Add new dagger datasets at share=1.0 (existing ones decay via update_dataset_shares)
    existing_dagger_roots = {
        ds["root"] for ds in state.get("dagger_datasets", [])
    }
    state.setdefault("dagger_datasets", [])
    for root in dagger_roots:
        if root not in existing_dagger_roots:
            state["dagger_datasets"].append({
                "root": root,
                "sampling_share": 1.0,
                "repo_id": f"lehome_dagger_{len(state['dagger_datasets'])}",
            })
            logger.info("Added dagger dataset: %s", root)

    logger.info("Dagger datasets: %d total", len(state["dagger_datasets"]))
    return dagger_roots


def upload_dagger_values(pipeline_cfg: RLPipelineConfig, state: dict,
                         sync_client: "HFSyncClient | None" = None):
    """Upload updated dagger parquets back to HuggingFace Hub after value prediction."""
    repo_id = pipeline_cfg.hf_dataset_repo
    if not repo_id:
        return

    dagger_datasets = state.get("dagger_datasets", [])
    if not dagger_datasets:
        return

    if not sync_client or not sync_client.enabled or not sync_client.is_daemon_alive():
        logger.warning("No HF sync daemon — skipping dagger values upload")
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_DAGGER_VALUES
    sync_client.submit(OP_UPLOAD_DAGGER_VALUES, {
        "repo_id": repo_id,
        "dagger_datasets": [{"root": ds["root"]} for ds in dagger_datasets],
    })


def upload_datasets(pipeline_cfg: RLPipelineConfig, eval_dataset_roots: list[str],
                    sync_client: "HFSyncClient | None" = None):
    """Upload rollout datasets to HuggingFace Hub if configured."""
    repo_id = pipeline_cfg.hf_dataset_repo
    if not repo_id:
        return

    if not sync_client or not sync_client.enabled:
        logger.warning("No HF sync daemon — skipping dataset upload (%d datasets)", len(eval_dataset_roots))
        return

    from lehome_solution.distributed.hf_sync_protocol import OP_UPLOAD_DATASET
    for eval_dataset_root in eval_dataset_roots:
        # Upload the whole eval run folder (eval_dataset + keyframes + videos)
        eval_run_dir = Path(eval_dataset_root).parent
        eval_run_id = eval_run_dir.name
        sync_client.submit(OP_UPLOAD_DATASET, {
            "eval_dataset_dir": str(eval_run_dir),
            "eval_run_id": eval_run_id,
            "repo_id": repo_id,
        })


# ---------------------------------------------------------------------------
# Dataset share management
# ---------------------------------------------------------------------------

def update_dataset_shares(pipeline_cfg: RLPipelineConfig, state: dict, new_dataset_root: str | None):
    """Decay old RL dataset shares, add new datasets, decay BC share.

    New datasets from the current iteration (stored in state['_new_rollout_datasets']
    by run_rollout_collection) are added with share=1.0 AFTER decaying old datasets.
    This ensures multi-strategy datasets from the same iteration all get full weight.
    """
    rl_decay = pipeline_cfg.rl_decay_factor
    rl_min = pipeline_cfg.rl_min_sampling_share

    # Decay (RL + BC + dagger) only when new rollout data arrived; otherwise this
    # is a no-op poll and decay would be double-applied on the next real call.
    new_datasets = state.pop("_new_rollout_datasets", None)
    has_new_data = bool(new_datasets) or (new_dataset_root is not None)

    # Decay existing RL datasets from previous iterations
    if has_new_data:
        kept = []
        for ds in state["rl_datasets"]:
            new_share = ds["sampling_share"] * rl_decay
            if new_share >= rl_min:
                ds["sampling_share"] = round(new_share, 4)
                kept.append(ds)
            else:
                logger.info("Excluding RL dataset (share %.3f < %.3f): %s",
                            new_share, rl_min, ds["root"])
        state["rl_datasets"] = kept

    # Add all new datasets from this iteration with share=1.0
    if new_datasets:
        for nd in new_datasets:
            entry = {
                "root": nd["root"],
                "sampling_share": 1.0,
                "repo_id": f"lehome_rl_{len(state['rl_datasets'])}",
            }
            if nd.get("strategy"):
                entry["strategy"] = nd["strategy"]
            if nd.get("model_step") is not None:
                entry["model_step"] = nd["model_step"]
            state["rl_datasets"].append(entry)
    elif new_dataset_root is not None:
        # Backward compat: single dataset (no multi-strategy)
        state["rl_datasets"].append({
            "root": new_dataset_root,
            "sampling_share": 1.0,
            "repo_id": f"lehome_rl_{len(state['rl_datasets'])}",
        })

    if has_new_data:
        bc_decay = pipeline_cfg.bc_dataset.decay_factor
        bc_min = pipeline_cfg.bc_dataset.min_sampling_share
        state["bc_sampling_share"] = round(max(state["bc_sampling_share"] * bc_decay, bc_min), 4)

        dagger_decay = pipeline_cfg.dagger_dataset.decay_factor
        dagger_min = pipeline_cfg.dagger_dataset.min_sampling_share
        for ds in state.get("dagger_datasets", []):
            ds["sampling_share"] = round(max(ds["sampling_share"] * dagger_decay, dagger_min), 4)

    n_dagger = len(state.get("dagger_datasets", []))
    logger.info("Shares: BC=%.3f, %d RL datasets, %d dagger datasets",
                state["bc_sampling_share"], len(state["rl_datasets"]), n_dagger)
    for ds in state["rl_datasets"]:
        logger.info("  RL: share=%.3f %s", ds["sampling_share"], ds["root"])
    for ds in state.get("dagger_datasets", []):
        logger.info("  DAG: share=%.3f %s", ds["sampling_share"], ds["root"])


# ---------------------------------------------------------------------------
# Inference optimization
# ---------------------------------------------------------------------------

from lehome_solution.eval.dataset_writer import EVAL_EPISODE_META_DIR


_INFERENCE_OPT_INCLUDED_ROLLOUT_TYPES = {"full"}


def _update_known_rollouts(sync_client, consumed: set, state: dict):
    """Write known rollout IDs to the sync dir so daemon only downloads new ones.

    Known = consumed IDs + rollout IDs already in state["rl_datasets"].
    """
    if not sync_client or not sync_client.enabled:
        return
    known = set(consumed)
    # Also add rollout IDs from rl_datasets in state (might not be in consumed)
    for ds in state.get("rl_datasets", []):
        hf_id = ds.get("hf_rollout_id")
        if hf_id:
            known.add(hf_id)
        # Also add the directory name from root path
        root = ds.get("root", "")
        if root:
            known.add(Path(root).parent.name)
    from lehome_solution.distributed.hf_sync_protocol import DIR_READY, atomic_write_json
    ready_dir = sync_client._sync_dir / DIR_READY
    ready_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        ready_dir / "known_rollouts.json",
        json.dumps({"known": sorted(known)}, indent=2),
    )


def _update_inference_prior(pipeline_cfg: RLPipelineConfig, state: dict, eval_dataset_roots: list[str]) -> bool:
    """Read episode results + inference configs from ALL new datasets, update prior once.

    All datasets from the same iteration are processed together in a single
    update_prior call so they share one decay cycle and one normalization baseline.

    Only full rollouts are used for optimization — the bandit objective is total
    SR across all garment types, so partial/curriculum/hard-mining evals would
    bias updates against configs that are fine on easy garments but happened to
    run during a hard-subset pass. Excluded rollout types:
    random/curriculum/hard_mining/success_replay.

    Returns True if the prior was actually updated (eligible episodes found).
    """
    from lehome_solution.eval.inference_optimization import load_prior, update_prior, save_prior
    from lehome_solution.eval.eval_utils import garment_name_to_type
    from lehome_solution.eval.rollout_strategies import load_success_rates_from_file

    prior_path = state["inference_prior_file"]
    prior = load_prior(prior_path)

    # Load per-garment-TYPE success rates — the baseline subtracted from reward.
    # Per-type (not per-garment) because the optimization objective is the total
    # SR across types, and pooling at the type level gives a lower-variance
    # baseline that doesn't flip sign on noisy per-garment SRs.
    garment_type_sr: dict[str, float] = {}
    sr_file = _find_success_rates_file(state)
    if sr_file:
        sr_data = load_success_rates_from_file(sr_file)
        if sr_data:
            garment_type_sr = sr_data.get("by_type", {})
    if not garment_type_sr:
        logger.warning("No per-garment-type success rates found, using 0.5 baseline")

    # Collect episodes from ALL new dataset roots
    episodes = []
    skipped = 0
    for eval_dataset_root in eval_dataset_roots:
        meta_dir = Path(eval_dataset_root) / EVAL_EPISODE_META_DIR
        if not meta_dir.exists():
            logger.warning("No episode metadata dir at %s, skipping", meta_dir)
            continue

        for meta_file in sorted(meta_dir.glob("episode_*.json")):
            with open(meta_file) as f:
                meta = json.load(f)
            inference_config = meta.get("inference_config")
            if inference_config is None:
                continue
            # Only include unbiased rollout types
            rollout_type = meta.get("rollout_type", "normal")
            if rollout_type not in _INFERENCE_OPT_INCLUDED_ROLLOUT_TYPES:
                skipped += 1
                continue
            garment = meta.get("garment", "")
            garment_type = garment_name_to_type(garment) if garment else "unknown"
            # Binary success as reward signal for inference param optimization
            reward = 1.0 if meta.get("success") else 0.0
            episodes.append({
                "inference_config": inference_config,
                "garment": garment,
                "garment_type": garment_type,
                "reward": reward,
            })

    if not episodes:
        logger.info("No eligible episodes for inference prior update (skipped %d)", skipped)
        return False

    prior = update_prior(
        prior, episodes,
        decay_factor=pipeline_cfg.inference_optimization.decay_factor,
        garment_type_success_rates=garment_type_sr,
    )
    save_prior(prior, prior_path)
    logger.info("Updated inference prior (%d episodes, skipped %d, iter %d)",
                len(episodes), skipped, prior["iteration"])
    return True


def _save_inference_config_to_checkpoint(state: dict):
    """Save per-garment-type best inference configs from prior to checkpoint assets dir."""
    prior_path = state.get("inference_prior_file")
    if not prior_path or not Path(prior_path).exists():
        return

    checkpoint_dir = state.get("checkpoint_dir")
    step = state.get("latest_checkpoint_step")
    if not checkpoint_dir or step is None:
        return

    try:
        from lehome_solution.eval.inference_optimization import load_prior, get_best_config
        from lehome_solution.constants import GARMENT_TYPES

        prior = load_prior(prior_path)

        # Save per-garment-type best configs
        per_gt = {}
        for gt in GARMENT_TYPES:
            per_gt[gt] = get_best_config(prior, garment_type=gt)

        config_data = {
            "iteration": prior.get("iteration", 0),
            "per_garment_type": per_gt,
        }

        # Save to each checkpoint step's assets dir
        assets_dir = Path(checkpoint_dir) / str(step) / "assets"
        if assets_dir.exists():
            config_path = assets_dir / "inference_config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.info("Saved per-garment-type inference configs to %s", config_path)
    except Exception as e:
        logger.warning("Failed to save inference config: %s", e)




# ---------------------------------------------------------------------------
# Step-aware dataset share management (for trainer mode)
# ---------------------------------------------------------------------------

def update_dataset_shares_step_aware(
    pipeline_cfg: RLPipelineConfig, state: dict,
    new_datasets: list[dict], current_model_step: int,
):
    """Add new datasets and decay old ones if the model step has advanced.

    Logic:
    - If new datasets come from the same model step as the highest existing
      datasets → just add them at 1.0, no decay (same checkpoint, more data).
    - If new datasets come from a newer model step → add at 1.0, apply one
      round of iteration-based decay to all existing RL datasets (same as
      the default pipeline's update_dataset_shares).
    - Datasets that fall below rl_min_sampling_share are removed.
    - BC share is decayed only when model step advances (not on same-step adds).
    - Dagger dataset shares are decayed (clamped to min) only when model step advances.
    """
    rl_decay = pipeline_cfg.rl_decay_factor
    rl_min = pipeline_cfg.rl_min_sampling_share

    # Determine if model step has advanced
    new_step = max((nd.get("model_step", 0) for nd in new_datasets), default=0) if new_datasets else 0
    highest_existing = max((ds.get("model_step", 0) for ds in state["rl_datasets"]), default=0)
    step_advanced = new_step > highest_existing

    # Decay existing datasets only if model step advanced
    if step_advanced:
        kept = []
        for ds in state["rl_datasets"]:
            new_share = ds["sampling_share"] * rl_decay
            if new_share >= rl_min:
                ds["sampling_share"] = round(new_share, 4)
                kept.append(ds)
            else:
                logger.info("Excluding RL dataset (share %.3f < %.3f): %s",
                            new_share, rl_min, ds["root"])
        state["rl_datasets"] = kept

    # Add new datasets with share=1.0
    for nd in new_datasets:
        entry = {
            "root": nd["root"],
            "sampling_share": 1.0,
            "repo_id": nd.get("repo_id", f"lehome_rl_{len(state['rl_datasets'])}"),
            "model_step": nd.get("model_step", current_model_step),
        }
        if nd.get("strategy"):
            entry["strategy"] = nd["strategy"]
        if nd.get("hf_rollout_id"):
            entry["hf_rollout_id"] = nd["hf_rollout_id"]
        state["rl_datasets"].append(entry)

    # Track consumed rollout IDs
    consumed = state.setdefault("consumed_rollout_ids", [])
    for nd in new_datasets:
        rid = nd.get("hf_rollout_id")
        if rid and rid not in consumed:
            consumed.append(rid)

    # Decay BC and dagger shares only when model step advances
    if step_advanced:
        bc_decay = pipeline_cfg.bc_dataset.decay_factor
        bc_min = pipeline_cfg.bc_dataset.min_sampling_share
        state["bc_sampling_share"] = round(max(state["bc_sampling_share"] * bc_decay, bc_min), 4)

        dagger_decay = pipeline_cfg.dagger_dataset.decay_factor
        dagger_min = pipeline_cfg.dagger_dataset.min_sampling_share
        for ds in state.get("dagger_datasets", []):
            ds["sampling_share"] = round(max(ds["sampling_share"] * dagger_decay, dagger_min), 4)

    n_dagger = len(state.get("dagger_datasets", []))
    logger.info("Dataset shares (new_step=%s, advanced=%s): BC=%.3f, %d RL datasets, %d dagger",
                new_step, step_advanced, state["bc_sampling_share"],
                len(state["rl_datasets"]), n_dagger)
    for ds in state["rl_datasets"]:
        logger.info("  RL: share=%.3f step=%s %s",
                    ds["sampling_share"], ds.get("model_step", "?"), ds["root"])


# ---------------------------------------------------------------------------
# Distributed: rollout worker loop
# ---------------------------------------------------------------------------

def run_rollout_worker_loop(
    cfg: RLPipelineConfig,
    state: dict,
    state_path: Path,
    worker_id: str,
    *,
    pinned_checkpoint_path: str | None = None,
):
    """Continuous rollout loop: download checkpoint from HF, run rollouts, upload, repeat.

    Runs non-stop. All HF operations go through the sync daemon (required):
    - Checkpoint downloads: daemon polls in background, worker reads ready/checkpoint.json
    - Dataset/state uploads: fire-and-forget via daemon
    - State downloads: submit_and_wait via daemon (needed before hard_mining/success_replay)
    If daemon fails to start and HF repos are configured, the pipeline aborts.

    When ``pinned_checkpoint_path`` is provided, the worker uses that exact
    checkpoint directory for every iteration and skips HF checkpoint polling.
    Dataset/state uploads still go through the daemon as usual. The pinned
    path must be of the form ``.../_hf_checkpoints/step_<N>`` so the step can
    be parsed for rollout-id naming.
    """
    from lehome_solution.distributed.hf_sync import make_rollout_id

    hf_model_repo = cfg.hf_model_repo
    hf_dataset_repo = cfg.hf_dataset_repo
    if not hf_model_repo or not hf_dataset_repo:
        logger.error("--rollout_worker requires hf_model_repo and hf_dataset_repo in config")
        sys.exit(1)

    poll_interval = cfg.checkpoint_poll_interval_s
    local_ckpt_dir = str(Path(state["checkpoint_dir"]) / "_hf_checkpoints")
    persistent_fs_dir = Path(state["checkpoint_dir"]) / "failure_states"
    state["_worker_id"] = worker_id

    checkpoint_path = None
    current_step = None

    # --- Optional pin to a specific checkpoint -----------------------------
    # When set, the worker resolves it ONCE up front, never polls HF for a
    # newer step, and reuses the same path on every iteration.
    pinned_step: int | None = None
    if pinned_checkpoint_path:
        pin_path = Path(pinned_checkpoint_path).resolve()
        if not pin_path.exists():
            logger.error("--rollout_checkpoint_path does not exist: %s", pin_path)
            sys.exit(1)
        # Parse "step_<N>" from the directory name (matches HF layout).
        name = pin_path.name
        if name.startswith("step_"):
            try:
                pinned_step = int(name[len("step_"):])
            except ValueError:
                pinned_step = None
        if pinned_step is None:
            try:
                pinned_step = int(name)
            except ValueError:
                logger.error(
                    "--rollout_checkpoint_path must end in 'step_<N>' or '<N>' "
                    "to derive the step number; got: %s", pin_path)
                sys.exit(1)
        checkpoint_path = str(pin_path)
        current_step = pinned_step
        state["latest_checkpoint_step"] = current_step
        logger.info(
            "Rollout worker pinned to checkpoint %s (step=%d) — HF checkpoint polling disabled",
            checkpoint_path, current_step,
        )

    # --- Start HF sync daemon (rollout mode — polls for checkpoints) ---
    daemon_proc, sync_client = _start_hf_sync_daemon(cfg, state, mode="rollout")
    if sync_client is None and (cfg.hf_model_repo or cfg.hf_dataset_repo):
        logger.error("HF sync daemon failed to start — cannot continue without it")
        sys.exit(1)
    _daemon_warned = False

    def _use_daemon():
        """Check if daemon is available for HF operations."""
        nonlocal _daemon_warned
        if sync_client is None:
            if not _daemon_warned:
                logger.warning("HF sync daemon not available — HF operations will be skipped")
                _daemon_warned = True
            return False
        alive = sync_client.is_daemon_alive()
        if not alive and not _daemon_warned:
            logger.warning("HF sync daemon died — HF operations will be skipped")
            _daemon_warned = True
        return alive

    try:
        # Initialize inference optimization prior if enabled
        inf_opt = cfg.inference_optimization
        if inf_opt.enabled:
            prior_path = Path(state["checkpoint_dir"]) / f"inference_prior_{worker_id}.json"
            if not prior_path.exists():
                # Try to seed from shared prior on HF
                shared_prior_path = Path(state["checkpoint_dir"]) / "inference_prior.json"
                state["inference_prior_file"] = str(shared_prior_path)
                _download_model_assets_from_hf(cfg, state, sync_client=sync_client)
                if shared_prior_path.exists():
                    import shutil
                    shutil.copy2(str(shared_prior_path), str(prior_path))
                    logger.info("Seeded worker prior from shared prior at %s", shared_prior_path)
                else:
                    from lehome_solution.eval.inference_optimization import init_prior, save_prior
                    prior = init_prior()
                    save_prior(prior, prior_path)
                    logger.info("Initialized inference prior at %s", prior_path)
            state["inference_prior_file"] = str(prior_path)

        def _sync_wandb_id(_ckpt_path=None):
            """Read wandb_id.txt, downloading from HF if not present locally."""
            wandb_file = Path(state["checkpoint_dir"]) / "wandb_id.txt"
            if not wandb_file.exists() and _use_daemon():
                # Trainer may have uploaded it since our last download attempt
                _download_model_assets_from_hf(cfg, state, sync_client=sync_client)
            if wandb_file.exists():
                wid = wandb_file.read_text().strip()
                if wid:
                    state["wandb_run_id"] = wid
                    logger.info("Synced wandb_run_id from HF: %s", wid)

        # Download success rates from HF for curriculum sampling
        _download_model_assets_from_hf(cfg, state, sync_client=sync_client)
        # Load wandb_run_id into state so rollout logging works before first
        # checkpoint arrives (download above just writes the file to disk).
        _sync_wandb_id()

        def _check_for_checkpoint():
            """Check for new checkpoint via daemon.

            Returns (checkpoint_path, step) or None.
            """
            nonlocal checkpoint_path, current_step

            # Pinned mode: surface the pinned checkpoint exactly once (first
            # call), never advance afterwards. HF polling is bypassed
            # entirely so a newer upstream checkpoint can't override the pin.
            if pinned_step is not None:
                if checkpoint_path is not None and current_step == pinned_step:
                    return checkpoint_path, current_step
                return None

            if not _use_daemon():
                return None

            # Daemon polls in background — just read ready file
            info = sync_client.get_ready_checkpoint()
            if info and info.get("step") is not None:
                new_step = info["step"]
                new_path = info.get("checkpoint_path")
                # Only accept strictly newer checkpoints — never regress.
                # The HF upload flow deletes+recreates latest/, creating a
                # window where get_latest_model_step falls back to old
                # numbered folders and the daemon can write a stale step
                # into ready/checkpoint.json.
                cur = current_step or 0
                if new_step > cur and new_path:
                    logger.info("Daemon has checkpoint step %d ready (was %s)", new_step, current_step)
                    checkpoint_path = new_path
                    current_step = new_step
                    state["latest_checkpoint_step"] = current_step
                    _sync_wandb_id(checkpoint_path)
                    return checkpoint_path, current_step
            return None

        def _download_states_for_strategy(strategy):
            """No-op: states are collected locally after each strategy.

            The rollout worker copies states from the eval run dir to the
            persistent checkpoint dir (via _collect_failure/success_states)
            right after each strategy completes — no HF round-trip needed.
            States are uploaded to HF for other workers, but this worker
            already has them locally.
            """
            pass

        def _upload_after_strategy(result, rollout_id, is_dagger: bool = False):
            """Upload dataset + states after a strategy completes. Fire-and-forget."""
            if not _use_daemon():
                logger.warning("No HF sync daemon — skipping post-strategy uploads for %s", rollout_id)
                return

            eval_run_dir = str(Path(result).parent)

            from lehome_solution.distributed.hf_sync_protocol import (
                OP_UPLOAD_DATASET, OP_UPLOAD_DAGGER_ROLLOUT,
                OP_UPLOAD_STATES, OP_CLEANUP_STATES,
            )
            # Upload rollout dataset. Semi_success_replay with dagger_only gets
            # uploaded under the dagger/ prefix so the trainer picks it up as
            # a dagger session rather than an RL rollout.
            if is_dagger:
                sync_client.submit(OP_UPLOAD_DAGGER_ROLLOUT, {
                    "eval_run_dir": eval_run_dir,
                    "rollout_id": rollout_id,
                    "repo_id": hf_dataset_repo,
                })
            else:
                sync_client.submit(OP_UPLOAD_DATASET, {
                    "eval_run_dir": eval_run_dir,
                    "rollout_id": rollout_id,
                    "repo_id": hf_dataset_repo,
                })

            # Upload failure states
            sync_client.submit(OP_UPLOAD_STATES, {
                "states_dir": str(persistent_fs_dir),
                "repo_id": hf_dataset_repo,
                "worker_id": worker_id,
                "kind": "failure_individual",
            })
            # Upload success states
            persistent_ss_dir = Path(state["checkpoint_dir"]) / "success_states"
            sync_client.submit(OP_UPLOAD_STATES, {
                "states_dir": str(persistent_ss_dir),
                "repo_id": hf_dataset_repo,
                "worker_id": worker_id,
                "kind": "success_individual",
            })
            # Upload semi-success states
            persistent_semi_dir = Path(state["checkpoint_dir"]) / "semi_success_states"
            if persistent_semi_dir.exists():
                sync_client.submit(OP_UPLOAD_STATES, {
                    "states_dir": str(persistent_semi_dir),
                    "repo_id": hf_dataset_repo,
                    "worker_id": worker_id,
                    "kind": "semi_success_individual",
                })
            # Cleanup consumed states
            consumed = state.pop("_consumed_success_states", [])
            if consumed:
                sync_client.submit(OP_CLEANUP_STATES, {
                    "hf_dataset_repo": hf_dataset_repo,
                    "filenames": list(set(consumed)),
                })

        logger.info("ROLLOUT WORKER: id=%s, model repo=%s", worker_id, hf_model_repo)

        # Default to a single random strategy if none configured
        if cfg.rollout_strategies:
            strategies = cfg.rollout_strategies
        else:
            from lehome_solution.training.pipeline_config import RolloutStrategyConfig
            strategies = [RolloutStrategyConfig(name="random", fraction=1.0)]

        while True:
            # Check for checkpoint (daemon polls automatically, or direct HF call)
            _check_for_checkpoint()

            if checkpoint_path is None:
                # No checkpoint at all yet — wait
                logger.info("No checkpoint available yet, waiting %ds...", poll_interval)
                time.sleep(poll_interval)
                continue

            # Run all strategies
            for strategy in strategies:
                # Check for newer checkpoint before each strategy
                _check_for_checkpoint()

                # Re-download success rates from HF before each strategy
                # (trainer may have uploaded new rates since last iteration)
                _download_model_assets_from_hf(cfg, state, sync_client=sync_client)

                # Download states needed by this strategy
                _download_states_for_strategy(strategy)

                rollout_id = make_rollout_id(current_step, strategy.name, worker_id)

                result, skipped, is_dagger = _execute_strategy(
                    cfg, state, checkpoint_path, strategy,
                    rollout_id=rollout_id,
                )

                if skipped or result is None:
                    if not skipped:
                        logger.warning("Rollout failed for strategy %s, continuing", strategy.name)
                    continue

                rejected = _is_rejected_path(result)
                if rejected:
                    logger.warning(
                        "Rollout %s dataset was rejected by integrity check — "
                        "logging SR/metrics to wandb but skipping HF upload, "
                        "state collection, inference-prior update, high-SR pin",
                        rollout_id,
                    )
                else:
                    # Manage local failure/success states
                    _collect_failure_states([result], persistent_fs_dir)
                    _remove_solved_failures([result], persistent_fs_dir)
                    persistent_ss_dir = Path(state["checkpoint_dir"]) / "success_states"
                    _collect_success_states([result], persistent_ss_dir)
                    persistent_semi_dir = Path(state["checkpoint_dir"]) / "semi_success_states"
                    _collect_semi_success_states([result], persistent_semi_dir)

                    # Upload everything — fire-and-forget. When is_dagger, the
                    # rollout is pushed under dagger/session_<rollout_id>/ on HF so
                    # the trainer's sync_dagger_datasets picks it up as dagger data.
                    _upload_after_strategy(result, rollout_id, is_dagger=is_dagger)

                    # Update inference optimization prior from rollout results.
                    # Upload only when the prior actually changed (i.e. a full
                    # rollout fed it) — non-full rollouts are no-ops post-Fix-1
                    # so re-uploading would just churn HF.
                    if inf_opt.enabled and state.get("inference_prior_file"):
                        if _update_inference_prior(cfg, state, [result]):
                            _upload_inference_prior_to_hf(cfg, state, sync_client=sync_client)

                # Log rollout metrics to wandb (runs even on rejection — SR/return
                # come from eval_summary.json, which is unaffected by the rename).
                _log_single_rollout_to_wandb(cfg, state, result, strategy.name, current_step)

                # Pin high-SR checkpoints: if full rollout has > 0.8 overall SR,
                # upload as a numbered checkpoint (params+assets only, no latest overwrite)
                if not rejected and strategy.name == "full" and _use_daemon() and hf_model_repo:
                    _maybe_upload_high_sr_checkpoint(
                        result, current_step, checkpoint_path, hf_model_repo,
                        cfg, sync_client,
                    )

                logger.info("Worker %s: completed rollout %s", worker_id, rollout_id)

            # Increment iteration counter for RNG diversity across loops
            state["_worker_iteration"] = state.get("_worker_iteration", 0) + 1
    finally:
        if daemon_proc:
            _stop_hf_sync_daemon(daemon_proc, sync_client)


# ---------------------------------------------------------------------------
# Distributed: trainer loop
# ---------------------------------------------------------------------------

def run_trainer_loop(cfg: RLPipelineConfig, state: dict, state_path: Path,
                     wait_for_data: bool = False):
    """Continuous training loop that checks HF for new rollouts between iterations.

    All HF operations go through the sync daemon (required):
    - Rollout dataset downloads: daemon polls in background, trainer reads ready/datasets.json
    - Checkpoint uploads: fire-and-forget via daemon
    - Dagger sync, asset downloads: via daemon
    If daemon fails to start and HF repos are configured, the pipeline aborts.

    Trains non-stop. Between iterations:
    1. Read ready/datasets.json — use whatever is downloaded (daemon polls in background)
    2. Repredict values (dagger), recompute advantages (all)
    3. Train next iteration
    4. Submit upload_checkpoint to daemon (fire-and-forget)

    Args:
        wait_for_data: If True, block until all missing datasets are downloaded
            before the first training iteration only.
    """
    print(f"[TRAINER] Starting trainer loop (iter={state['iteration']}, step={state.get('latest_checkpoint_step')})",
          flush=True)

    hf_model_repo = cfg.hf_model_repo
    hf_dataset_repo = cfg.hf_dataset_repo
    if not hf_model_repo or not hf_dataset_repo:
        logger.error("--trainer requires hf_model_repo and hf_dataset_repo in config")
        sys.exit(1)

    local_rollout_dir = str(Path(state["checkpoint_dir"]) / "_hf_rollouts")
    num_iterations = cfg.num_iterations
    consumed = set(state.get("consumed_rollout_ids", []))

    # --- Start HF sync daemon (trainer mode — polls for new rollouts) ---
    daemon_proc, sync_client = _start_hf_sync_daemon(cfg, state, mode="trainer")
    if sync_client is None and (cfg.hf_model_repo or cfg.hf_dataset_repo):
        logger.error("HF sync daemon failed to start — cannot continue without it")
        sys.exit(1)
    _daemon_alive = lambda: sync_client and sync_client.is_daemon_alive()

    try:
        # Initialize inference optimization prior if enabled
        inf_opt = cfg.inference_optimization
        if inf_opt.enabled:
            prior_path = Path(state["checkpoint_dir"]) / "inference_prior.json"
            state["inference_prior_file"] = str(prior_path)
            _download_model_assets_from_hf(cfg, state, sync_client=sync_client)
            if not prior_path.exists():
                from lehome_solution.eval.inference_optimization import init_prior, save_prior
                prior = init_prior()
                save_prior(prior, prior_path)
                logger.info("Initialized inference prior at %s", prior_path)
            _save_state(state, state_path)

        # Write known rollout IDs so daemon only downloads new ones
        _update_known_rollouts(sync_client, consumed, state)

        # --- Check for local checkpoint, download from HF if missing ---
        local_step = find_latest_checkpoint_step(state["checkpoint_dir"])
        if local_step is not None:
            state["latest_checkpoint_step"] = local_step
            state["current_train_steps"] = local_step
        elif state.get("latest_checkpoint_step") is not None and _daemon_alive():
            # State says we should have a checkpoint but it's not local — download it
            expected_step = state["latest_checkpoint_step"]
            logger.info("Checkpoint step %d not found locally — downloading from HF ...",
                        expected_step)
            from lehome_solution.distributed.hf_sync_protocol import OP_DOWNLOAD_CHECKPOINT
            try:
                result = sync_client.submit_and_wait(OP_DOWNLOAD_CHECKPOINT, {
                    "hf_model_repo": hf_model_repo,
                    "local_dir": state["checkpoint_dir"],
                    "include_train_state": True,
                }, timeout=1800)
                if result and result.success:
                    dl_step = (result.result or {}).get("step")
                    if dl_step:
                        state["latest_checkpoint_step"] = dl_step
                        state["current_train_steps"] = dl_step
                        logger.info("Downloaded checkpoint step %d from HF", dl_step)
                    else:
                        logger.warning("Checkpoint download returned no step info")
                else:
                    logger.warning("Checkpoint download failed: %s",
                                   result.error if result else "no result")
            except Exception as e:
                logger.warning("Failed to download checkpoint from HF: %s", e)
            # Rename step_N/ -> N/ so orbax can find it, and copy assets
            ckpt_base = Path(state["checkpoint_dir"])
            for d in ckpt_base.iterdir():
                if d.is_dir() and d.name.startswith("step_") and d.name[5:].isdigit():
                    numeric_dir = ckpt_base / d.name[5:]
                    if not numeric_dir.exists():
                        d.rename(numeric_dir)
                        logger.info("Renamed %s -> %s", d.name, numeric_dir.name)
            # Copy assets (norm_stats, FAST tokenizer) to top-level assets/ dir
            for step_dir in ckpt_base.iterdir():
                if not step_dir.is_dir() or not step_dir.name.isdigit():
                    continue
                step_assets = step_dir / "assets"
                top_assets = ckpt_base / "assets"
                if step_assets.is_dir() and not top_assets.exists():
                    import shutil
                    shutil.copytree(str(step_assets), str(top_assets))
                    logger.info("Copied assets from %s/ to top-level assets/", step_dir.name)
            # Re-check local
            local_step = find_latest_checkpoint_step(state["checkpoint_dir"])
            if local_step is not None:
                state["latest_checkpoint_step"] = local_step
                state["current_train_steps"] = local_step

        is_fresh_start = state["latest_checkpoint_step"] is None

        if is_fresh_start:
            logger.info("=" * 60)
            logger.info("TRAINER: Fresh start — warmup (%d steps)", cfg.warmup_steps)
            logger.info("=" * 60)

            download_missing_datasets(cfg, state, sync_client=sync_client)
            if wait_for_data:
                _wait_for_missing_datasets(state)
            _save_state(state, state_path)

            # Compute segment advantages on initial datasets
            # (auto mode → segment for iteration 0)
            if state["rl_datasets"] or state.get("dagger_datasets"):
                if not recompute_advantages(cfg, state):
                    logger.error("Advantage computation failed during warmup")
                    sys.exit(1)
                _upload_success_rates_to_hf(cfg, state, sync_client=sync_client)

            state["current_train_steps"] = cfg.warmup_steps + 1
            _save_state(state, state_path)

            _stop_wandb_watch = threading.Event()
            _wandb_watch_thread = threading.Thread(
                target=_watch_and_upload_wandb_id,
                args=(cfg, state, sync_client, _stop_wandb_watch),
                daemon=True,
            )
            _wandb_watch_thread.start()
            try:
                if not run_training(cfg, state):
                    logger.error("Warmup training failed")
                    sys.exit(1)
            finally:
                _stop_wandb_watch.set()

            upload_checkpoint(cfg, state, sync_client=sync_client)
            _upload_wandb_id_to_hf(cfg, state, sync_client=sync_client)
            _upload_pipeline_state_to_hf(cfg, state, state_path, sync_client=sync_client)

            if inf_opt.enabled and state.get("inference_prior_file"):
                _save_inference_config_to_checkpoint(state)

            state["iteration"] = 1
            state["consumed_rollout_ids"] = list(consumed)
            _save_state(state, state_path)
            logger.info("Warmup complete. Step: %s", state["latest_checkpoint_step"])

        logger.info("TRAINER: continuous training, checking %s for new rollouts between iterations",
                    hf_dataset_repo)

        while state["iteration"] < num_iterations:
            iteration = state["iteration"]
            logger.info("")
            logger.info("=" * 60)
            logger.info("TRAINER ITERATION %d / %d", iteration + 1, num_iterations)
            logger.info("=" * 60)

            # --- Check for ready datasets (daemon downloads in background) ---
            # Dedup key is the rollout dir name (hf_rollout_id), not the local path,
            # so `consumed` stays consistent across restarts and matches what
            # update_dataset_shares_step_aware writes into state.
            new_datasets = []
            if _daemon_alive():
                ready_ds = sync_client.get_ready_datasets()
                from lehome_solution.distributed.hf_sync import parse_rollout_id
                for r_path in ready_ds:
                    rollout_dir_name = Path(r_path).parent.name
                    if rollout_dir_name in consumed:
                        continue
                    parsed = parse_rollout_id(rollout_dir_name)
                    new_datasets.append({
                        "root": r_path,
                        "model_step": parsed["model_step"] if parsed else 0,
                        "strategy": parsed["strategy"] if parsed else "unknown",
                        "hf_rollout_id": rollout_dir_name,
                    })
                    consumed.add(rollout_dir_name)
                if new_datasets:
                    logger.info("Daemon has %d new datasets ready", len(new_datasets))
            else:
                logger.warning("HF sync daemon not alive — cannot check for new rollouts")

            if new_datasets:
                current_step = state.get("latest_checkpoint_step", 0)
                update_dataset_shares_step_aware(cfg, state, new_datasets, current_step)
                # update_dataset_shares_step_aware already appends hf_rollout_ids
                # into state["consumed_rollout_ids"]; no need to overwrite here.
                _save_state(state, state_path)
                _update_known_rollouts(sync_client, consumed, state)

                # Update inference prior from new rollout results, and upload
                # to HF whenever a full rollout actually changed it.
                if inf_opt.enabled and state.get("inference_prior_file"):
                    new_roots = [nd["root"] for nd in new_datasets]
                    if _update_inference_prior(cfg, state, new_roots):
                        _upload_inference_prior_to_hf(cfg, state, sync_client=sync_client)
            else:
                logger.info("No new rollouts (training with existing %d datasets)",
                            len(state["rl_datasets"]))

            # --- Standard pipeline: download missing -> dagger sync -> advantages -> train -> upload ---
            # All HF ops go through daemon when available

            download_missing_datasets(cfg, state, sync_client=sync_client)
            if wait_for_data:
                _wait_for_missing_datasets(state)
                wait_for_data = False  # Only wait on first iteration

            sync_dagger_datasets(cfg, state, sync_client=sync_client)
            _save_state(state, state_path)

            if new_datasets:
                if not recompute_advantages(cfg, state):
                    # Training on stale/missing advantages poisons the RL signal.
                    # Skip this iteration and retry next poll instead.
                    logger.error("Advantage recomputation failed — skipping training this iteration")
                    _save_state(state, state_path)
                    time.sleep(30)
                    continue
                _upload_success_rates_to_hf(cfg, state, sync_client=sync_client)
            else:
                logger.info("Skipping advantage recomputation (no new datasets)")

            # Training — complete warmup if iteration 0 (crash recovery),
            # otherwise train for steps_per_iteration
            latest = state["latest_checkpoint_step"] or 0
            if state["iteration"] == 0:
                state["current_train_steps"] = cfg.warmup_steps + 1
            else:
                state["current_train_steps"] = latest + cfg.steps_per_iteration + 1
            _save_state(state, state_path)

            _stop_wandb_watch = threading.Event()
            _wandb_watch_thread = threading.Thread(
                target=_watch_and_upload_wandb_id,
                args=(cfg, state, sync_client, _stop_wandb_watch),
                daemon=True,
            )
            _wandb_watch_thread.start()
            try:
                if not run_training(cfg, state):
                    logger.error("Training failed at iteration %d", iteration)
                    sys.exit(1)
            finally:
                _stop_wandb_watch.set()

            # Upload checkpoint + wandb_id + pipeline_state — fire-and-forget via daemon
            upload_checkpoint(cfg, state, sync_client=sync_client)
            _upload_wandb_id_to_hf(cfg, state, sync_client=sync_client)
            _upload_pipeline_state_to_hf(cfg, state, state_path, sync_client=sync_client)

            # Save inference config if enabled
            if cfg.inference_optimization.enabled and state.get("inference_prior_file"):
                _save_inference_config_to_checkpoint(state)

            state["iteration"] += 1
            state["consumed_rollout_ids"] = list(consumed)
            _save_state(state, state_path)

            logger.info("Trainer iteration %d complete. Step: %s", iteration + 1, state["latest_checkpoint_step"])

        logger.info("=" * 60)
        logger.info("TRAINER COMPLETE: %d iterations", state["iteration"])
        logger.info("=" * 60)
    finally:
        if daemon_proc:
            _stop_hf_sync_daemon(daemon_proc, sync_client)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Pipeline YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")

    # Single-phase modes
    parser.add_argument("--train_only", action="store_true",
                        help="Run one training phase and exit")
    parser.add_argument("--rollout_only", action="store_true",
                        help="Run one rollout phase and exit")
    parser.add_argument("--advantages_only", action="store_true",
                        help="Recompute advantages and exit")

    # Distributed modes
    parser.add_argument("--rollout_worker", action="store_true",
                        help="Run as rollout worker: poll HF for checkpoints, run rollouts, upload")
    parser.add_argument("--trainer", action="store_true",
                        help="Run as trainer: poll HF for rollout datasets, train, upload")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for all missing datasets to download before the first training iteration")
    parser.add_argument("--worker_id", type=str, default=None,
                        help="Worker ID for distributed mode (default: hostname)")
    parser.add_argument("--rollout_checkpoint_path", type=str, default=None,
                        help="If set, --rollout_worker uses this exact checkpoint "
                             "directory (e.g. <run>/_hf_checkpoints/step_75500) "
                             "instead of polling HF for the latest. The worker "
                             "stays pinned to this checkpoint for its entire run.")

    # Config overrides
    parser.add_argument("--start_iteration", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--steps_per_iteration", type=int, default=None)
    parser.add_argument("--num_garments", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # Load config
    skip_keys = {
        "config", "resume", "start_iteration",
        "train_only", "rollout_only", "advantages_only",
        "rollout_worker", "trainer", "wait", "worker_id",
        "rollout_checkpoint_path",
    }
    overrides = {k: v for k, v in vars(args).items() if k not in skip_keys and v is not None}
    cfg = load_pipeline_config(args.config, overrides)

    # Import triggers JAX init on CPU
    from lehome_solution.training.config import get_config

    # Remove JAX_PLATFORMS so children inherit clean env
    os.environ.pop("JAX_PLATFORMS", None)

    base_config = get_config(cfg.config_name)
    checkpoint_base = base_config.checkpoint_base_dir
    checkpoint_dir = str(
        (Path(checkpoint_base) / cfg.config_name / cfg.exp_name).resolve()
    )

    state_path = Path(checkpoint_dir) / STATE_FILENAME

    # Load or create state
    if args.resume or args.train_only or args.rollout_only or args.advantages_only or args.rollout_worker or args.trainer:
        state = _load_state(state_path)
        if state["checkpoint_dir"] is None:
            # No saved state — initialize fresh with defaults
            state = _default_state()
            state["checkpoint_dir"] = checkpoint_dir
            state["bc_sampling_share"] = cfg.bc_dataset.initial_sampling_share
            for i, ds in enumerate(cfg.initial_rl_datasets):
                from lehome_solution.training.pipeline_config import _infer_strategy
                strategy = ds.strategy or _infer_strategy(ds.root)
                entry = {
                    "root": ds.root,
                    "sampling_share": ds.sampling_share,
                    "repo_id": ds.repo_id or f"lehome_rl_init_{i}",
                    "strategy": strategy,
                }
                if ds.segment_only:
                    entry["segment_only"] = True
                state["rl_datasets"].append(entry)
            for i, ds in enumerate(cfg.initial_dagger_datasets):
                state["dagger_datasets"].append({
                    "root": ds.root,
                    "sampling_share": ds.sampling_share,
                    "repo_id": ds.repo_id or f"lehome_dagger_init_{i}",
                    "strategy": "dagger",
                })
        if args.resume:
            logger.info("Resuming: iter=%d phase=%s steps=%d",
                        state["iteration"], state["phase"], state["current_train_steps"])
    else:
        # Fresh start — check if checkpoint dir already has state
        if state_path.exists():
            logger.error(
                "Pipeline state already exists at %s. "
                "Use --resume to continue, or delete the checkpoint dir to start fresh.",
                state_path,
            )
            sys.exit(1)

        state = _default_state()
        state["checkpoint_dir"] = checkpoint_dir
        state["bc_sampling_share"] = cfg.bc_dataset.initial_sampling_share

        for i, ds in enumerate(cfg.initial_rl_datasets):
            entry = {
                "root": ds.root,
                "sampling_share": ds.sampling_share,
                "repo_id": ds.repo_id or f"lehome_rl_init_{i}",
            }
            if ds.segment_only:
                entry["segment_only"] = True
            state["rl_datasets"].append(entry)
        for i, ds in enumerate(cfg.initial_dagger_datasets):
            state["dagger_datasets"].append({
                "root": ds.root,
                "sampling_share": ds.sampling_share,
                "repo_id": ds.repo_id or f"lehome_dagger_init_{i}",
            })

        if args.start_iteration is not None:
            state["iteration"] = args.start_iteration

    # --- Distributed modes ---

    if args.rollout_worker:
        worker_id = args.worker_id or platform.node()
        return run_rollout_worker_loop(
            cfg, state, state_path, worker_id,
            pinned_checkpoint_path=args.rollout_checkpoint_path,
        )

    if args.trainer:
        return run_trainer_loop(cfg, state, state_path, wait_for_data=args.wait)

    # --- Single-phase modes ---

    if args.train_only:
        return _run_train_only(cfg, state, state_path)
    if args.rollout_only:
        return _run_rollout_only(cfg, state, state_path)
    if args.advantages_only:
        return _run_advantages_only(cfg, state, state_path)

    # --- No mode specified ---
    logger.error(
        "No mode specified. Use --trainer or --rollout_worker "
        "(or --train_only, --rollout_only, --advantages_only for single-phase)."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Single-phase runners
# ---------------------------------------------------------------------------

def _run_train_only(cfg: RLPipelineConfig, state: dict, state_path: Path):
    """Run one training phase as if it's the next pipeline step."""
    daemon_proc, sync_client = _start_hf_sync_daemon(cfg, state)
    try:
        steps_per_iter = cfg.steps_per_iteration

        step = find_latest_checkpoint_step(state["checkpoint_dir"])
        if step is not None:
            state["latest_checkpoint_step"] = step
        latest = state["latest_checkpoint_step"] or 0
        state["current_train_steps"] = latest + steps_per_iter + 1
        _save_state(state, state_path)

        logger.info("TRAIN ONLY: target %d steps", state["current_train_steps"])
        if not run_training(cfg, state):
            sys.exit(1)

        upload_checkpoint(cfg, state, sync_client=sync_client)
        _upload_wandb_id_to_hf(cfg, state, sync_client=sync_client)
        _upload_pipeline_state_to_hf(cfg, state, state_path, sync_client=sync_client)

        # Save best inference config to checkpoint assets
        if cfg.inference_optimization.enabled and state.get("inference_prior_file"):
            _save_inference_config_to_checkpoint(state)

        state["phase"] = "eval_log"
        _save_state(state, state_path)
        logger.info("Training complete. Latest: step %s", state["latest_checkpoint_step"])
    finally:
        if daemon_proc:
            _stop_hf_sync_daemon(daemon_proc, sync_client)


def _ensure_checkpoint_step(state: dict):
    """Resolve latest checkpoint step if not yet known. Exits on failure."""
    if state["latest_checkpoint_step"] is None:
        step = find_latest_checkpoint_step(state["checkpoint_dir"])
        if step is None:
            logger.error("No checkpoint found at %s", state["checkpoint_dir"])
            sys.exit(1)
        state["latest_checkpoint_step"] = step
        state["current_train_steps"] = step


def _run_rollout_only(cfg: RLPipelineConfig, state: dict, state_path: Path):
    """Run one rollout as if it's the next pipeline step."""
    daemon_proc, sync_client = _start_hf_sync_daemon(cfg, state)
    try:
        _ensure_checkpoint_step(state)

        # Initialize inference prior if needed
        inf_opt = cfg.inference_optimization
        if inf_opt.enabled:
            prior_path = Path(state["checkpoint_dir"]) / "inference_prior.json"
            state["inference_prior_file"] = str(prior_path)
            _download_model_assets_from_hf(cfg, state, sync_client=sync_client)
            if not prior_path.exists():
                from lehome_solution.eval.inference_optimization import init_prior, save_prior
                save_prior(init_prior(), prior_path)
        else:
            _download_model_assets_from_hf(cfg, state, sync_client=sync_client)

        logger.info("ROLLOUT ONLY: from step %s", state["latest_checkpoint_step"])
        eval_dataset_root = run_rollout_collection(cfg, state)
        if eval_dataset_root is None:
            sys.exit(1)

        new_ds_roots = [ds["root"] for ds in state.get("_new_rollout_datasets", [])]

        upload_datasets(cfg, new_ds_roots, sync_client=sync_client)
        update_dataset_shares(cfg, state, eval_dataset_root)

        # Manage persistent failure states
        persistent_fs_dir = Path(state["checkpoint_dir"]) / "failure_states"
        _collect_failure_states(new_ds_roots, persistent_fs_dir)
        _remove_solved_failures(new_ds_roots, persistent_fs_dir)
        _upload_failure_states(cfg, persistent_fs_dir, sync_client=sync_client)

        # Collect and upload success/semi-success states from rollouts
        persistent_ss_dir = Path(state["checkpoint_dir"]) / "success_states"
        _collect_success_states(new_ds_roots, persistent_ss_dir)
        persistent_semi_dir = Path(state["checkpoint_dir"]) / "semi_success_states"
        _collect_semi_success_states(new_ds_roots, persistent_semi_dir)
        _upload_success_states(cfg, persistent_ss_dir, sync_client=sync_client)
        _upload_semi_success_states(cfg, persistent_semi_dir, sync_client=sync_client)

        # Update and upload inference prior — only upload when the prior
        # actually changed (i.e. a full rollout fed it).
        if inf_opt.enabled and state.get("inference_prior_file") and new_ds_roots:
            if _update_inference_prior(cfg, state, new_ds_roots):
                _upload_inference_prior_to_hf(cfg, state, sync_client=sync_client)

        # Log eval metrics + inference prior to wandb
        if state["rl_datasets"]:
            log_eval_to_wandb(cfg, state, state["rl_datasets"][-1]["root"])

        state["phase"] = "advantages"
        _save_state(state, state_path)
        logger.info("Rollout complete: %s", eval_dataset_root)
    finally:
        if daemon_proc:
            _stop_hf_sync_daemon(daemon_proc, sync_client)


def _run_advantages_only(cfg: RLPipelineConfig, state: dict, state_path: Path):
    """Recompute advantages across all RL datasets."""
    logger.info("ADVANTAGES ONLY: %d datasets", len(state["rl_datasets"]))
    if not recompute_advantages(cfg, state):
        sys.exit(1)

    # Upload success_rates directly (no daemon in standalone mode)
    if cfg.hf_model_repo:
        from lehome_solution.eval.rollout_strategies import SUCCESS_RATES_FILE
        from lehome_solution.training.hf_upload import upload_success_rates_to_hf
        for ds in reversed(state.get("rl_datasets", [])):
            sr_path = Path(ds["root"]) / SUCCESS_RATES_FILE
            if sr_path.exists():
                upload_success_rates_to_hf(str(sr_path), cfg.hf_model_repo)
                break

    state["phase"] = "train"
    _save_state(state, state_path)
    logger.info("Advantages recomputed")


if __name__ == "__main__":
    main()
