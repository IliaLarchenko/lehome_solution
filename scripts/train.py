"""
Training script for LeHome Challenge solution.

Based on https://github.com/PhysicalIntelligence/openpi/blob/behavior/openpi/scripts/train.py with custom modifications.
"""

import concurrent.futures
import dataclasses
import functools
import logging
import os
import platform
import sys
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

# JAX memory: let it use up to 95% of GPU memory.
# 'platform' allocator (CUDA malloc) doesn't preallocate but is slightly slower
# than JAX's BFC allocator for small allocs.  Fine for training.
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.95')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

# Configure OpenBLAS to prevent thread creation errors
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')

# Disable HF datasets caching to avoid stale arrow files after parquet rewrites.
# recompute_advantages.py modifies parquets in-place between training iterations;
# stale cached arrow files cause None values in dataset reads.
import tempfile
os.environ["HF_DATASETS_CACHE"] = tempfile.mkdtemp(prefix="hf_datasets_")
import datasets
datasets.disable_caching()

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

# Import LeHome-specific modules
from lehome_solution.training import checkpoints as _checkpoints
from lehome_solution.training import config as _config
from lehome_solution.training import data_loader as _data_loader
from lehome_solution.training import weight_loaders as _weight_loaders
from lehome_solution.training import attention_backend as _attention_backend

from lehome_solution.models.pi_modified import PiModified
from lehome_solution.models.observation import Observation
from lehome_solution.training.loss_weights import type_balanced_weighted_mean


# ── HF upload background machinery ────────────────────────────────────────
# Orbax `save_state` is async (queues a write, returns immediately). We mirror
# that for HF uploads via a single-worker ThreadPoolExecutor: each save_state
# can fire-and-forget a follow-up `upload_checkpoint_to_hf`, training never
# blocks. If a previous upload is still running when the next save fires
# (slow upload vs. fast saves), we SKIP the new one rather than queue — the
# upload routine always pushes the LATEST checkpoint on each call, so missing
# an intermediate one is harmless. `_drain_hf_uploads` at the end of training
# waits for the in-flight upload so the final step lands.
_hf_upload_executor: concurrent.futures.ThreadPoolExecutor | None = None
_hf_upload_future: concurrent.futures.Future | None = None


def _maybe_submit_hf_upload(config, step: int, checkpoint_manager=None) -> None:
    """Fire-and-forget background HF upload after a save_state call.

    ``save_state`` returns quickly but Orbax finishes the actual write +
    atomic rename + max_to_keep delete on its own threads — so the
    ``{checkpoint_dir}/{step}/`` directory typically does NOT exist by the
    time we get here. We:

    1. Submit the upload to a single-worker executor (so concurrent uploads
       can't race on the same repo).
    2. Inside the worker, call ``checkpoint_manager.wait_until_finished()``
       so Orbax's async write completes before we read files.
    3. If by the time the wait returns the requested step has been pruned
       (e.g. ``max_to_keep=1`` collapsed an intermediate one), fall back to
       Orbax's ``latest_step()`` — we'd rather push the newest available
       checkpoint than skip the cycle.
    """
    global _hf_upload_executor, _hf_upload_future
    if _hf_upload_executor is None:
        _hf_upload_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hf_upload",
        )
    if _hf_upload_future is not None and not _hf_upload_future.done():
        logging.info("HF upload still running, skipping step %d", step)
        return

    from pathlib import Path

    from lehome_solution.training.hf_upload import upload_checkpoint_to_hf

    def _run() -> None:
        try:
            if checkpoint_manager is not None:
                checkpoint_manager.wait_until_finished()
            target_step = step
            ckpt_path = Path(str(config.checkpoint_dir)) / str(target_step)
            if not ckpt_path.exists() and checkpoint_manager is not None:
                latest = checkpoint_manager.latest_step()
                if latest is not None and latest != target_step:
                    logging.info(
                        "HF upload: step %d already pruned by Orbax, "
                        "falling back to latest_step=%d", target_step, latest,
                    )
                    target_step = int(latest)
            upload_checkpoint_to_hf(
                str(config.checkpoint_dir),
                target_step,
                config.hf_model_repo,
                keep_period=config.keep_period or 2000,
            )
        except Exception as e:  # noqa: BLE001 — never crash the training loop
            logging.warning("HF upload failed at step %d: %r", step, e)

    _hf_upload_future = _hf_upload_executor.submit(_run)


def _drain_hf_uploads() -> None:
    global _hf_upload_executor, _hf_upload_future
    if _hf_upload_future is not None:
        try:
            _hf_upload_future.result()
        except Exception as e:  # noqa: BLE001
            logging.warning("HF upload (final) failed: %r", e)
    if _hf_upload_executor is not None:
        _hf_upload_executor.shutdown(wait=True)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)

    # Filter out fields that won't exist in the pretrained checkpoint:
    # - nnx.Intermediate (cached state, not saved)
    # - Newly added trainable params (success head, garment type embedding, etc.)
    #   These are initialized fresh and don't need to match the pretrained weights.
    def filter_intermediate_fields(params_dict):
        flat = traverse_util.flatten_dict(params_dict)
        skip_field_names = [
            'action_correlation_cholesky',
            'L_spatial', 'L_temporal',
            'cached_num_inpaint_actions', 'cached_input_action_dim',
            'cached_Sigma_uo_Sigma_oo_inv', 'cached_L_cond_free',
            'cached_Sigma_ou_Sigma_uu_inv', 'cached_L_cond_inp',
            'advantage_embeddings',
            'advantage_adarms_vec',
            'success_query_token',
            'success_head',
            'garment_type_input_embedding',
            'garment_type_adarms_embedding',
        ]
        filtered = {k: v for k, v in flat.items()
                   if not any(field in str(k) for field in skip_field_names)}
        return traverse_util.unflatten_dict(filtered)

    params_shape_filtered = filter_intermediate_fields(params_shape)
    loaded_params_filtered = filter_intermediate_fields(loaded_params)
    at.check_pytree_equality(expected=params_shape_filtered, got=loaded_params_filtered, check_shapes=True, check_dtypes=True)

    def should_exclude(k, v):
        if isinstance(v, jax.ShapeDtypeStruct):
            return True
        intermediate_field_names = [
            'action_correlation_cholesky', 'L_spatial', 'L_temporal',
            'cached_num_inpaint_actions', 'cached_input_action_dim',
            'cached_Sigma_uo_Sigma_oo_inv', 'cached_L_cond_free',
            'cached_Sigma_ou_Sigma_uu_inv', 'cached_L_cond_inp',
        ]
        return any(field in str(k) for field in intermediate_field_names)

    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items()
         if not should_exclude(k, v)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    # Aux-head weight decay replaces the old explicit `head_l2` loss term:
    # the mask is True on success/checkpoint/garment_type/completion/ttc head
    # kernels + biases, and `AdamWWithAuxDecay` applies `aux_head_weight_decay`
    # on top of the baseline decay for those params only.
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule,
        weight_decay_mask=_config.aux_head_weight_decay_mask,
    )

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# -- Structured metric keys for wandb logging --

# These are the metrics we log under train/ section
_TRAIN_LOSS_KEYS = {
    "loss", "action_loss", "fast_loss", "success_loss", "checkpoint_loss",
    "garment_type_loss", "completion_loss", "ttc_loss",
    # Keypoint + world-modeling heads.
    "keypoint_distance_loss",
    "wm_fast_success_loss", "wm_fast_completion_loss", "wm_fast_keypoint_loss",
    "wm_flow_success_loss", "wm_flow_completion_loss", "wm_flow_keypoint_loss",
}
_TRAIN_METRIC_KEYS = {
    "fast_perplexity", "success_pred_mean", "success_pred_std",
    "success_target_mean", "success_mae", "success_valid_frac",
    "success_pred_mean_weighted", "success_target_mean_weighted",
    "checkpoint_pred_mean",
    "checkpoint_pred_mean_weighted", "checkpoint_target_mean_weighted",
    "garment_type_accuracy",
    "completion_pred_mean",
    "completion_pred_mean_weighted", "completion_target_mean_weighted",
    "ttc_pred_mean", "ttc_valid_frac", "ttc_target_mean",
    "ttc_pred_mean_weighted", "ttc_target_mean_weighted",
    # TTC bootstrap diagnostics: fraction of valid TTC samples that used the
    # bootstrap target vs. analytical fallback, and the per-batch mean of
    # the per-sample TTC loss-weight scale (drops to ~0.8–0.99 as older RL
    # samples enter the batch; stays at ~1.0 when training on recent data).
    "ttc_bootstrap_frac", "ttc_weight_scale_mean",
    "grad_norm", "param_norm",
    "weight_mean", "weight_min", "weight_max",
    # Keypoint + world-modeling head metrics. keypoint_distance_valid_frac is
    # a SINGLE shared metric across all 3 heads (Head 1 current + Heads 2/3
    # future). wm_flow_t_valid_frac shows the share of batch samples with at
    # least one low-noise flow sample (the training budget for WM-flow).
    "keypoint_distance_mae", "keypoint_distance_valid_frac", "wm_flow_t_valid_frac",
    # wm_flow_success is now Δsuccess = true − V̂. Pred mean/std confirms the
    # head's operating range (~[-1, 1]); target mean confirms the label
    # distribution feeding the MSE loss.
    "wm_flow_success_pred_mean", "wm_flow_success_pred_std",
    "wm_flow_success_target_mean",
    "action_pad_frac",
}


def _inactive_metric_prefixes(config: "_config.TrainConfig") -> tuple[str, ...]:
    """Return name prefixes whose metrics should be hidden from logs / wandb.

    A head is "inactive" when its loss weight is exactly 0 (so no gradient
    flows through it) OR — for the keypoint/WM heads — when its dedicated
    ``use_*_head`` flag is False (the head isn't even built).

    Inactive metrics still get computed inside the JITted train_step (those
    forward passes happen anyway), but we drop them from the console log
    line and the wandb payload — otherwise we end up plotting flat-zero (or
    pretrained-init constant, like ``checkpoint_pred_mean=0.68``) curves
    that aren't even being learned.
    """
    m = config.model
    inactive: list[str] = []
    if getattr(m, "success_loss_weight", 1.0) == 0.0:
        inactive.append("success_")
    if getattr(m, "checkpoint_loss_weight", 1.0) == 0.0:
        inactive.append("checkpoint_")
    if getattr(m, "ttc_loss_weight", 1.0) == 0.0:
        inactive.append("ttc_")
    if getattr(m, "garment_type_loss_weight", 1.0) == 0.0:
        inactive.append("garment_type_")
    if getattr(m, "completion_loss_weight", 1.0) == 0.0:
        inactive.append("completion_")
    if not getattr(m, "use_keypoint_distance_head", False):
        inactive.append("keypoint_distance_")
    if not getattr(m, "use_wm_fast_head", False):
        inactive.append("wm_fast_")
    if not getattr(m, "use_wm_flow_head", False):
        inactive.append("wm_flow_")
    if not getattr(m, "use_fast_auxiliary", False):
        inactive.append("fast_")
    return tuple(inactive)


def _filter_inactive_metrics(raw_info: dict, inactive_prefixes: tuple[str, ...]) -> dict:
    """Drop entries whose key starts with any of ``inactive_prefixes``."""
    if not inactive_prefixes:
        return raw_info
    return {k: v for k, v in raw_info.items() if not k.startswith(inactive_prefixes)}


def _structure_metrics(raw_info: dict) -> dict:
    """Convert flat metrics dict into structured wandb sections.

    Returns dict with train/loss_name and train/metric_name keys.
    Drops per-joint losses and other noisy metrics.
    """
    structured = {}
    for key, val in raw_info.items():
        if key in _TRAIN_LOSS_KEYS:
            structured[f"train/{key}"] = val
        elif key in _TRAIN_METRIC_KEYS:
            structured[f"train/{key}"] = val
    return structured


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    observation, actions = batch

    @at.typecheck
    def loss_fn(
        model: PiModified, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions
    ):
        losses_dict = model.compute_detailed_loss(rng, observation, actions, train=True, num_flow_samples=config.num_flow_samples)
        per_sample_loss = losses_dict["total_loss"]  # [B]
        B = per_sample_loss.shape[0]
        ones_b = jnp.ones_like(per_sample_loss)
        # action + fast losses: unweighted batch mean, no type balancing.
        # Sampling distribution (prioritised / FR-proportional) is the training signal.
        total_loss = jnp.mean(per_sample_loss)

        # ── Auxiliary losses: unified IS-weighted × per-garment debias × type-balanced ──
        # is_weight = (1 / (N * p_i)) / ep_len — set uniformly for every source in
        # the data loader. Debias multipliers are per-garment (and absorb the
        # success-tail boost for the success multiplier only).
        gt_id = observation.garment_type_id if observation.garment_type_id is not None else jnp.zeros_like(per_sample_loss, dtype=jnp.int32)
        is_w = observation.is_weight if observation.is_weight is not None else ones_b
        sdw = observation.success_debias_weight if observation.success_debias_weight is not None else ones_b
        cdw = observation.checkpoint_debias_weight if observation.checkpoint_debias_weight is not None else ones_b

        def _aux(loss_key, valid, weights, scale):
            if loss_key not in losses_dict:
                return jnp.float32(0.0)
            return scale * type_balanced_weighted_mean(losses_dict[loss_key], weights, valid, gt_id)

        def _record_weighted(prefix, pred_key, target, valid, weights):
            """Log type-balanced weighted mean of predictions and targets under
            the *same* weights used for the loss. Lets you compare what the
            gradient sees vs. the unweighted `{prefix}_pred_mean`.
            """
            if pred_key in losses_dict:
                pred = losses_dict[pred_key]
                losses_dict[f"{prefix}_pred_mean_weighted"] = (
                    type_balanced_weighted_mean(pred, weights, valid, gt_id) * ones_b
                )
            if target is not None:
                losses_dict[f"{prefix}_target_mean_weighted"] = (
                    type_balanced_weighted_mean(target, weights, valid, gt_id) * ones_b
                )

        # success head: BCE on P(success). All datasets (BC/DAgger have success=1).
        if "success_loss" in losses_dict and observation.success_target is not None:
            valid = jnp.isfinite(observation.success_target)
            w = is_w * sdw
            total_loss = total_loss + _aux(
                "success_loss", valid, w, config.model.success_loss_weight,
            )
            _record_weighted("success", "_success_pred_per_sample", observation.success_target, valid, w)

        # checkpoint head: BCE on P(reward >= 0.5).
        if "checkpoint_loss" in losses_dict and observation.checkpoint_target is not None:
            valid = jnp.isfinite(observation.checkpoint_target)
            w = is_w * cdw
            total_loss = total_loss + _aux(
                "checkpoint_loss", valid, w, config.model.checkpoint_loss_weight,
            )
            _record_weighted("checkpoint", "_checkpoint_pred_per_sample", observation.checkpoint_target, valid, w)

        # ttc head: same weighting as success (debias + tail boost), further
        # scaled per-sample by ``ttc_weight_scale`` so older RL datasets
        # contribute at 0.01x (last-N RL samples pass through at 1.0).
        if "ttc_loss" in losses_dict and observation.ttc_target is not None:
            valid = jnp.isfinite(observation.ttc_target)
            ttc_scale = (
                observation.ttc_weight_scale
                if observation.ttc_weight_scale is not None
                else ones_b
            )
            w = is_w * sdw * ttc_scale
            total_loss = total_loss + _aux(
                "ttc_loss", valid, w, config.model.ttc_loss_weight,
            )
            _record_weighted("ttc", "_ttc_pred_per_sample", observation.ttc_target, valid, w)

            # Diagnostic: what fraction of TTC-valid samples used the
            # bootstrap target vs. the analytical fallback. Also track the
            # per-batch mean of the per-sample weight multiplier so you can
            # eyeball "how many older RL samples landed in this batch".
            if observation.ttc_used_bootstrap is not None:
                valid_f = valid.astype(jnp.float32)
                used_f = observation.ttc_used_bootstrap.astype(jnp.float32)
                n_valid = jnp.maximum(valid_f.sum(), 1.0)
                losses_dict["ttc_bootstrap_frac"] = (
                    (used_f * valid_f).sum() / n_valid
                ) * ones_b
            if observation.ttc_weight_scale is not None:
                losses_dict["ttc_weight_scale_mean"] = (
                    jnp.mean(observation.ttc_weight_scale) * ones_b
                )

        # completion head: successful episodes only (success>0.5), no debias, no tail boost.
        if "completion_loss" in losses_dict and observation.completion_target is not None:
            valid = jnp.isfinite(observation.completion_target)
            if observation.success_target is not None:
                valid = valid & (observation.success_target > 0.5)
            total_loss = total_loss + _aux(
                "completion_loss", valid, is_w, config.model.completion_loss_weight,
            )
            _record_weighted("completion", "_completion_pred_per_sample", observation.completion_target, valid, is_w)

        # garment type head: all data, no debias.
        if "garment_type_loss" in losses_dict:
            valid = (gt_id >= 0)
            total_loss = total_loss + _aux(
                "garment_type_loss", valid, is_w, config.model.garment_type_loss_weight,
            )

        # Keypoint-distance head (Head 1): per-sample already masked + averaged
        # over valid slots inside compute_detailed_loss. Weight by is_weight
        # (no debias — slot-level valid mask filters BC/DAgger cleanly). Valid
        # when the per-sample mask had at least one True slot.
        if "keypoint_distance_loss" in losses_dict and observation.keypoint_distance_mask is not None:
            valid = observation.keypoint_distance_mask.any(axis=-1)
            total_loss = total_loss + _aux(
                "keypoint_distance_loss", valid, is_w, config.model.keypoint_distance_loss_weight,
            )

        # WM-FAST head (Head 2, training-only). Targets are future values; masks
        # match current-frame semantics (BC/DAgger future success/completion
        # carry the same BC forcing as the current-step labels). success uses
        # sdw (same as main success head), completion uses plain is_w,
        # keypoint uses is_w as Head 1.
        if "wm_fast_success_loss" in losses_dict and observation.success_future_target is not None:
            valid = jnp.isfinite(observation.success_future_target)
            w = is_w * sdw
            total_loss = total_loss + _aux(
                "wm_fast_success_loss", valid, w, config.model.wm_fast_success_loss_weight,
            )
        if "wm_fast_completion_loss" in losses_dict and observation.completion_future_target is not None:
            # Mirror completion_loss semantics: success-only filter + is_w weight.
            valid = jnp.isfinite(observation.completion_future_target)
            if observation.success_target is not None:
                valid = valid & (observation.success_target > 0.5)
            total_loss = total_loss + _aux(
                "wm_fast_completion_loss", valid, is_w, config.model.wm_fast_completion_loss_weight,
            )
        if "wm_fast_keypoint_loss" in losses_dict and observation.keypoint_distance_future_mask is not None:
            valid = observation.keypoint_distance_future_mask.any(axis=-1)
            total_loss = total_loss + _aux(
                "wm_fast_keypoint_loss", valid, is_w, config.model.wm_fast_keypoint_loss_weight,
            )

        # WM-flow head (Head 3). Per-sample loss is already zero when t > 0.5
        # inside the model (flow samples with too-noisy actions contribute
        # nothing). We ALSO gate the type-balanced weighted-mean denominator
        # by the same t-mask so the logged metric and the gradient see the
        # same effective sample set.
        wm_flow_t_valid = losses_dict.get("_wm_flow_t_valid_per_sample")
        if "wm_flow_success_loss" in losses_dict and observation.success_future_target is not None:
            valid = jnp.isfinite(observation.success_future_target)
            if wm_flow_t_valid is not None:
                valid = valid & wm_flow_t_valid
            w = is_w * sdw
            total_loss = total_loss + _aux(
                "wm_flow_success_loss", valid, w, config.model.wm_flow_success_loss_weight,
            )
        if "wm_flow_completion_loss" in losses_dict and observation.completion_future_target is not None:
            # Mirror completion_loss semantics: success-only filter + is_w weight.
            valid = jnp.isfinite(observation.completion_future_target)
            if observation.success_target is not None:
                valid = valid & (observation.success_target > 0.5)
            if wm_flow_t_valid is not None:
                valid = valid & wm_flow_t_valid
            total_loss = total_loss + _aux(
                "wm_flow_completion_loss", valid, is_w, config.model.wm_flow_completion_loss_weight,
            )
        if "wm_flow_keypoint_loss" in losses_dict and observation.keypoint_distance_future_mask is not None:
            valid = observation.keypoint_distance_future_mask.any(axis=-1)
            if wm_flow_t_valid is not None:
                valid = valid & wm_flow_t_valid
            total_loss = total_loss + _aux(
                "wm_flow_keypoint_loss", valid, is_w, config.model.wm_flow_keypoint_loss_weight,
            )

        # ── Valid-only logging for the new keypoint + WM heads ─────────────
        # The per-sample losses above include zeros for invalid samples (BC /
        # DAgger / samples without check_distances). A plain batch mean would
        # dilute the reported metric by those zeros. Here we overwrite the
        # logged values with the mean computed over VALID samples only, so
        # the metric reflects what the gradient actually sees.
        def _valid_only_mean_replace(loss_key, valid_bool):
            if loss_key not in losses_dict or valid_bool is None:
                return
            per_sample = losses_dict[loss_key]
            v = valid_bool.astype(per_sample.dtype)
            v_sum = v.sum()
            scalar = jnp.where(
                v_sum > 0,
                (per_sample * v).sum() / jnp.maximum(v_sum, 1.0),
                0.0,
            )
            # Broadcast to [B] so the downstream jnp.mean(val) in the info
            # loop is a no-op (mean of constant array = that constant).
            losses_dict[loss_key] = scalar * jnp.ones_like(per_sample)

        # Head 1 current-frame validity
        kpt_valid_now = (
            observation.keypoint_distance_mask.any(axis=-1)
            if observation.keypoint_distance_mask is not None else None
        )
        # Heads 2 / 3 future validity
        kpt_valid_fut = (
            observation.keypoint_distance_future_mask.any(axis=-1)
            if observation.keypoint_distance_future_mask is not None else None
        )
        # Mirror success_loss / completion_loss validity semantics. Future heads
        # now have the SAME coverage as their current-time counterparts (user
        # requirement) — so the logged valid-means are computed over the same
        # sample sets, not gated by the sparse keypoint_mask.
        s_fut_valid = (
            jnp.isfinite(observation.success_future_target)
            if observation.success_future_target is not None else None
        )
        c_fut_valid = (
            jnp.isfinite(observation.completion_future_target)
            if observation.completion_future_target is not None else None
        )
        # completion uses the same success-only filter as completion_loss.
        if c_fut_valid is not None and observation.success_target is not None:
            c_fut_valid = c_fut_valid & (observation.success_target > 0.5)

        _valid_only_mean_replace("keypoint_distance_loss", kpt_valid_now)
        _valid_only_mean_replace("keypoint_distance_mae_per_sample", kpt_valid_now)
        # WM-FAST heads: the post-FAST token is always present at training
        # (FAST always provided), so t-validity doesn't apply.
        _valid_only_mean_replace("wm_fast_success_loss", s_fut_valid)
        _valid_only_mean_replace("wm_fast_completion_loss", c_fut_valid)
        _valid_only_mean_replace("wm_fast_keypoint_loss", kpt_valid_fut)
        # WM-flow heads: intersect the semantic valid mask with the t ≤ 0.5
        # filter so the logged metric denominator matches exactly which
        # samples contributed to the gradient.
        def _with_t(vb):
            return vb & wm_flow_t_valid if (vb is not None and wm_flow_t_valid is not None) else vb
        _valid_only_mean_replace("wm_flow_success_loss", _with_t(s_fut_valid))
        _valid_only_mean_replace("wm_flow_completion_loss", _with_t(c_fut_valid))
        _valid_only_mean_replace("wm_flow_keypoint_loss", _with_t(kpt_valid_fut))

        # Also surface keypoint_distance_mae (rename from per_sample key to the
        # name consumed by the info loop).
        if "keypoint_distance_mae_per_sample" in losses_dict:
            losses_dict["keypoint_distance_mae"] = losses_dict["keypoint_distance_mae_per_sample"]

        # Shared "valid share" metric — one number for all 3 heads, representing
        # the batch fraction that has keypoint-distance supervision (i.e. the
        # share of recent check_distances rollouts in training).
        if kpt_valid_now is not None:
            losses_dict["keypoint_distance_valid_frac"] = (
                kpt_valid_now.astype(jnp.float32).mean() * jnp.ones(B)
            )
        # WM-flow t-validity metric — true fraction of flow samples (out of
        # N_flow_samples × B total) with t ≤ 0.5. The model already emits this
        # as a broadcasted scalar; just plumb it through. Expected value for
        # Beta(1.5, 1) time sampler is P(t ≤ 0.5) ≈ 0.354, independent of
        # num_flow_samples (each flow sample is iid).
        if "_wm_flow_t_valid_frac_scalar" in losses_dict:
            losses_dict["wm_flow_t_valid_frac"] = losses_dict["_wm_flow_t_valid_frac_scalar"]

        return total_loss, losses_dict

    train_rng = jax.random.fold_in(rng, state.step)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, losses_dict), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Compute param norm (kernel weights only)
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    _AUX_KEYS = (
        "action_loss", "success_loss", "fast_loss", "success_pred_mean", "success_pred_std",
        "fast_accuracy", "success_target_mean", "success_mae", "success_valid_frac",
        "success_pred_mean_weighted", "success_target_mean_weighted",
        "checkpoint_loss", "checkpoint_pred_mean",
        "checkpoint_pred_mean_weighted", "checkpoint_target_mean_weighted",
        "garment_type_loss", "garment_type_accuracy",
        "completion_loss", "completion_pred_mean",
        "completion_pred_mean_weighted", "completion_target_mean_weighted",
        "ttc_loss", "ttc_pred_mean", "ttc_valid_frac", "ttc_target_mean",
        "ttc_pred_mean_weighted", "ttc_target_mean_weighted",
        "ttc_bootstrap_frac", "ttc_weight_scale_mean",
        # Keypoint + WM heads (current and future).
        # keypoint_distance_valid_frac is ONE shared metric across all 3 heads.
        # wm_flow_t_valid_frac is the fraction of batch where t ≤ 0.5 (the
        # cutoff for valid WM-flow supervision). All "*_loss" values have been
        # replaced with valid-only means in loss_fn so these reported numbers
        # reflect the real training signal.
        "keypoint_distance_loss", "keypoint_distance_mae", "keypoint_distance_valid_frac",
        "wm_fast_success_loss", "wm_fast_completion_loss", "wm_fast_keypoint_loss",
        "wm_flow_success_loss", "wm_flow_completion_loss", "wm_flow_keypoint_loss",
        "wm_flow_t_valid_frac",
        "wm_flow_success_pred_mean", "wm_flow_success_pred_std",
        "wm_flow_success_target_mean",
        # Diagnostic: fraction of action timesteps that were LeRobot-padded
        # (horizon overshooting the episode end). Masked out of action_loss.
        "action_pad_frac",
    )
    for key in _AUX_KEYS:
        if key in losses_dict:
            val = losses_dict[key]
            info[key] = val if (isinstance(val, (float, int)) or (hasattr(val, 'ndim') and val.ndim == 0)) else jnp.mean(val)
    if "fast_loss" in info:
        info["fast_perplexity"] = jnp.exp(info["fast_loss"])

    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    backend = _attention_backend.install(
        use_flash_attention=config.use_flash_attention,
        use_xsa=config.use_xsa,
    )
    logging.info(f"Attention backend: {backend}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for JAX RNG: {seed}")

    rng = jax.random.key(seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_lehome_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Verify aux target fields are present in Observation (required for JIT trace)
    obs_check = batch[0]
    for field in ("checkpoint_target", "garment_type_id", "completion_target", "ttc_target"):
        val = getattr(obs_check, field, None)
        if val is None:
            logging.error("MISSING field '%s' in Observation batch — aux losses will be zero!", field)
        else:
            logging.info("Observation.%s: shape=%s dtype=%s", field, val.shape, val.dtype)

    # Log camera views from first batch (batch is always (Observation, actions))
    obs_for_log = batch[0]
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in obs_for_log.images.values()], axis=1))
        for i in range(min(5, len(next(iter(obs_for_log.images.values())))))
    ]
    wandb.log({"media/camera_views": images_to_log}, step=0)

    # Get norm_stats for correlation matrix
    data_config = data_loader.data_config()
    if data_config.norm_stats is None:
        raise ValueError("norm_stats not found. Run compute_norm_stats.py first.")
    norm_stats = data_config.norm_stats

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # Load/reload correlation matrix (uses NumPy, must be outside JIT/eval_shape).
    # _all_correction_matrices is stored as a plain attribute (not NNX-tracked)
    # so it doesn't affect params/ema_params/sharding trees.
    model = nnx.merge(train_state.model_def, train_state.params)
    if isinstance(model, PiModified) and norm_stats is not None:
        model.load_correlation_matrix(norm_stats)
        logging.info("Loaded correlation matrix")
        train_state = dataclasses.replace(
            train_state, model_def=nnx.graphdef(model), params=nnx.state(model)
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    # Timing accumulators
    _t_gpu_window = 0.0
    _t_data_window = 0.0
    _t_window_steps = 0

    # One-time computation of which metric prefixes to hide from logs.
    inactive_prefixes = _inactive_metric_prefixes(config)
    if inactive_prefixes:
        logging.info(
            "Suppressing inactive-head metric prefixes from console + wandb: %s",
            inactive_prefixes,
        )

    for step in pbar:
        _t0 = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        jax.block_until_ready(info)
        _t1 = time.perf_counter()
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            # Drop metrics for heads whose loss weight is 0 or whose
            # use_*_head flag is False — they're either flat-zero or stuck
            # at pretrained-init constants and would just clutter the dash.
            reduced_info = _filter_inactive_metrics(reduced_info, inactive_prefixes)

            main_metrics = {k: v for k, v in reduced_info.items()
                          if k in _TRAIN_LOSS_KEYS | _TRAIN_METRIC_KEYS | {"loss"}}
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in main_metrics.items())
            logging.info(f"Step {step}: {info_str}")

            wandb.log(_structure_metrics(reduced_info), step=step)
            sys.stderr.flush()
            infos = []

        _t2 = time.perf_counter()
        batch = next(data_iter)
        _t3 = time.perf_counter()

        _t_gpu_window += (_t1 - _t0)
        _t_data_window += (_t3 - _t2)
        _t_window_steps += 1

        if _t_window_steps == 10:
            logging.info(
                f"TIMING [last 10 steps]: "
                f"gpu={_t_gpu_window/10:.3f}s, "
                f"data={_t_data_window/10:.3f}s, "
                f"total={(_t_gpu_window+_t_data_window)/10:.3f}s/step"
            )
            _t_gpu_window = 0.0
            _t_data_window = 0.0
            _t_window_steps = 0

        if step % 500 == 3:
            profile_path = str(config.checkpoint_dir / "memory_profile.prof")
            jax.profiler.save_device_memory_profile(profile_path)
            logging.info(f"Saved device memory profile to {profile_path}")
            from lehome_solution.training import memory_report
            memory_report.print_memory_report()

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            if getattr(config, "hf_model_repo", None):
                _maybe_submit_hf_upload(config, step, checkpoint_manager)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    # Drain any in-flight HF uploads before exit so the latest checkpoint
    # actually lands on the hub.
    _drain_hf_uploads()

    if config.wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    # Support loading config from a pickle file (used by RL pipeline)
    if "--config_file" in sys.argv:
        idx = sys.argv.index("--config_file")
        config_path = sys.argv[idx + 1]
        import pickle
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        main(config)
    # YAML-driven training: --yaml configs/train_real_bc.yaml [--resume|--overwrite]
    # [--exp_name NAME]. The YAML names a registered base config and overrides
    # fields on top (see lehome_solution/training/yaml_train_config.py).
    elif "--yaml" in sys.argv:
        from lehome_solution.training.yaml_train_config import train_config_from_yaml

        idx = sys.argv.index("--yaml")
        yaml_path = sys.argv[idx + 1]
        rest = sys.argv[1:idx] + sys.argv[idx + 2:]
        overrides: dict = {}
        i = 0
        while i < len(rest):
            arg = rest[i]
            if arg == "--resume":
                overrides["resume"] = True
            elif arg == "--overwrite":
                overrides["overwrite"] = True
            elif arg == "--exp_name":
                overrides["exp_name"] = rest[i + 1]
                i += 1
            else:
                raise SystemExit(
                    f"Unknown argument in --yaml mode: {arg} "
                    "(supported: --resume, --overwrite, --exp_name NAME)"
                )
            i += 1
        main(train_config_from_yaml(yaml_path, overrides))
    else:
        main(_config.cli())
