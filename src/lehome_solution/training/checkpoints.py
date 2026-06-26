"""Checkpoint management with FAST tokenizer saving.

Adapted from openpi.training.checkpoints, which cannot be imported here: its
module imports openpi.training.data_loader -> lerobot.common (the pre-0.4
lerobot API absent from this venv). The unchanged parts (CallbackHandler /
CallbackSave, load_norm_stats, _split_params) are therefore
copies; the LeHome additions are FAST-tokenizer asset saving and partial
params restore for resuming after adding new heads.

Reference: https://github.com/Physical-Intelligence/openpi
"""

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import os
import shutil
from typing import Protocol

from etils import epath
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.training.utils as training_utils

from lehome_solution.training import data_loader as _data_loader

# Use our custom normalize (has per_timestamp fields and can save/load them)
from lehome_solution.shared import normalize as _normalize


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    assets_base_dir: str | None = None,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

        # Save FAST tokenizer if it exists
        model = nnx.merge(state.model_def, state.params)
        # Check if model has FAST enabled (indicated by having fast_token_embedding)
        if hasattr(model, 'fast_token_embedding'):
            if assets_base_dir is not None:
                resolved_base = epath.Path(assets_base_dir)
            else:
                # Infer assets_base_dir from checkpoint structure
                # checkpoint_dir is like: ./outputs/checkpoints/config_name/exp_name
                # assets_base_dir is like: ./outputs/assets/config_name
                checkpoint_dir = checkpoint_manager.directory
                parts = checkpoint_dir.parts
                if 'checkpoints' in parts:
                    idx = parts.index('checkpoints')
                    assets_base_parts = parts[:idx] + ('assets',) + (parts[idx + 1],)
                    resolved_base = epath.Path(*assets_base_parts)
                else:
                    resolved_base = checkpoint_dir.parent.parent / 'assets' / checkpoint_dir.parent.name

            fast_tokenizer_source = resolved_base / data_config.asset_id / "fast_tokenizer"
            fast_tokenizer_dest = directory / data_config.asset_id / "fast_tokenizer"

            if fast_tokenizer_source.exists():
                fast_tokenizer_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(fast_tokenizer_source, fast_tokenizer_dest, dirs_exist_ok=True)
                logging.info(f"Saved FAST tokenizer to checkpoint: {fast_tokenizer_dest}")
            else:
                logging.warning(f"FAST tokenizer source not found: {fast_tokenizer_source}")

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


#
# ── LEGACY OPT-STATE MIGRATION SHIM ─────────────────────────────────────────
# One-shot support for checkpoints saved by the old ``AdamWBF16`` optimizer.
# Old chain: ``optax.chain(clip_by_global_norm, optax.adamw(..., mu_dtype=bf16))``
#   → opt_state: ``(EmptyState, (ScaleByAdamState_mu_bf16, EmptyState, EmptyState))``
# New chain (AdamWWithAuxDecay): flat 5-tuple with ``mu`` in fp32 and an extra
# stateless ``add_decayed_weights`` for the aux-head mask.
# Set ``LEHOME_LEGACY_OPT_STATE=1`` on the first resume after the cleanup; the
# next save will rewrite opt_state in the new structure and the flag can be
# removed. Delete this whole block once no legacy checkpoints remain.
#

def _legacy_bf16_opt_state_target(new_opt_state):
    """Rebuild an abstract opt_state matching the *old* AdamWBF16 chain, with
    ``mu`` dtype flipped from fp32 → bf16 and the tree nested to match the old
    ``(clip, adamw(...))`` layout. Takes the abstract new-style opt_state tree
    (ShapeDtypeStruct leaves) and returns the equivalent abstract old-style tree.
    """
    clip_s, adam_s, base_decay_s, _aux_decay_s, lr_s = new_opt_state

    def _to_bf16(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(x.shape, jnp.bfloat16)
        if hasattr(x, "dtype") and hasattr(x, "astype"):
            return x.astype(jnp.bfloat16)
        return x

    adam_s_bf16 = adam_s._replace(mu=jax.tree.map(_to_bf16, adam_s.mu))
    return (clip_s, (adam_s_bf16, base_decay_s, lr_s))


def _convert_legacy_opt_state_to_new(legacy_opt_state):
    """Restored old-style opt_state → new-style flat tree. Casts ``mu`` bf16→fp32
    and inserts a fresh ``MaskedState(EmptyState())`` at position 3 for the new
    aux-head ``add_decayed_weights(..., mask=fn)`` transform (which optax wraps
    in ``optax.masked`` when a mask is provided).
    """
    from optax.transforms._masking import MaskedState

    clip_s, inner = legacy_opt_state
    adam_s, base_decay_s, lr_s = inner

    def _to_fp32(x):
        if hasattr(x, "dtype") and hasattr(x, "astype") and x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
        return x

    adam_s_fp32 = adam_s._replace(mu=jax.tree.map(_to_fp32, adam_s.mu))
    aux_decay_s = MaskedState(inner_state=optax.EmptyState())
    return (clip_s, adam_s_fp32, base_decay_s, aux_decay_s, lr_s)


def _legacy_opt_state_requested() -> bool:
    return os.environ.get("LEHOME_LEGACY_OPT_STATE", "0").strip().lower() in ("1", "true", "yes")
#
# ────────────────────────────────────────────────────────────────────────────
#


def _restore_params_partial(
    checkpoint_manager: ocp.CheckpointManager,
    step: int | None,
    live_params,
) -> dict:
    """Restore params from disk, tolerating new live params not on disk.

    Strategy:
      1. Resolve ``step`` (``None`` → latest).
      2. Use ``ocp.PyTreeCheckpointer().restore(abs_params_dir)`` on
         ``<ckpt>/<step>/params`` — the low-level restore returns the on-disk
         tree as a plain dict of arrays and does NOT do strict tree-structure
         matching against a template. This sidesteps the mismatch that
         ``CheckpointManager.restore`` raises when the live model has new
         params or the on-disk ckpt has retired params.
      3. Merge key-by-key into the live tree: on-disk keys overwrite live
         values, and live-only keys keep their fresh random init.

    Note on key conventions: nnx.Param leaves land on disk under
    ``.../value`` (e.g. ``.../bias/value``). We preserve that wrapping in
    the returned tree because downstream code in ``restore_state`` feeds the
    tree straight into ``state.params`` which is an ``nnx.State``.
    """
    import flax
    from pathlib import Path

    ckpt_step = step if step is not None else checkpoint_manager.latest_step()
    if ckpt_step is None:
        raise RuntimeError(
            "Checkpoint manager has no steps to restore — step=None and "
            "latest_step() returned None."
        )

    ckpt_params_dir = (
        Path(str(checkpoint_manager.directory)).resolve() / str(ckpt_step) / "params"
    )
    try:
        with ocp.PyTreeCheckpointer() as ckptr:
            raw_disk = ckptr.restore(str(ckpt_params_dir))
    except Exception as restore_err:
        logging.warning(
            "Low-level params restore at %s failed (%s). Falling back to the "
            "legacy strict restore — will error if the live tree has new keys.",
            ckpt_params_dir, restore_err,
        )
        return checkpoint_manager.restore(
            ckpt_step,
            items={"params": {"params": live_params}},
        )

    # ``raw_disk`` has the shape {"params": {actual_tree}} because that's how we
    # saved it. Inner tree already carries ``.../value`` wrappers at leaves.
    if not (isinstance(raw_disk, dict) and "params" in raw_disk):
        raise RuntimeError(
            f"Unexpected on-disk params layout at {ckpt_params_dir}: "
            f"top-level keys = {list(raw_disk.keys()) if isinstance(raw_disk, dict) else type(raw_disk)}"
        )
    disk_inner = raw_disk["params"]

    # Flatten both trees with the same key convention (``/value`` suffix kept).
    # ``live_params`` is an nnx.State → convert to pure dict, then add ``value``
    # wrappers so it lines up with the on-disk layout.
    to_pure = getattr(live_params, "to_pure_dict", None)
    live_plain = to_pure() if to_pure is not None else live_params
    live_flat_plain = flax.traverse_util.flatten_dict(live_plain, sep="/")
    live_flat = {f"{k}/value": v for k, v in live_flat_plain.items()}

    disk_flat = flax.traverse_util.flatten_dict(disk_inner, sep="/")

    overlapping = [k for k in live_flat if k in disk_flat]
    missing_on_disk = [k for k in live_flat if k not in disk_flat]
    extra_on_disk = [k for k in disk_flat if k not in live_flat]

    if not missing_on_disk and not extra_on_disk:
        logging.info(
            "Params tree structures match at step %s — using direct restore.",
            ckpt_step,
        )
    else:
        logging.warning(
            "Partial params restore at step %s: %d keys loaded from disk, %d "
            "new live-only keys keep random init, %d extra on-disk keys "
            "ignored. Example new keys: %s  Example ignored: %s",
            ckpt_step, len(overlapping), len(missing_on_disk), len(extra_on_disk),
            [k.replace("/value", "") for k in missing_on_disk[:3]],
            [k.replace("/value", "") for k in extra_on_disk[:3]],
        )

    # Merge: live baseline (random init for new heads), overwritten by disk
    # values for every shared key.
    merged_flat = dict(live_flat)
    for k in overlapping:
        merged_flat[k] = disk_flat[k]

    # Materialize any remaining ShapeDtypeStruct leaves (new-only keys) into
    # real random-init arrays. The live tree was built via ``jax.eval_shape``
    # so new-head leaves are abstract; downstream code (``jnp.array``,
    # ``TrainState``) requires concrete arrays.
    # Convention: N(0, 0.02) for kernels (matches ``success_query_token`` style),
    # zero init for biases / 1-D leaves. Seeded by the param path for
    # reproducibility across resumes.
    import numpy as _np
    import hashlib as _hashlib
    n_materialized = 0
    for k, v in merged_flat.items():
        if isinstance(v, jax.ShapeDtypeStruct):
            shape = v.shape
            dtype = v.dtype
            is_bias = k.endswith("/bias/value") or len(shape) <= 1
            if is_bias:
                merged_flat[k] = _np.zeros(shape, dtype=dtype)
            else:
                seed = int(_hashlib.sha256(k.encode()).hexdigest()[:8], 16)
                rng = _np.random.default_rng(seed)
                merged_flat[k] = rng.normal(0.0, 0.02, shape).astype(dtype)
            n_materialized += 1
    if n_materialized:
        logging.info(
            "Materialized %d new-head leaves with random init "
            "(kernels: N(0, 0.02); biases: zeros).", n_materialized,
        )

    # Downstream (``restore_state`` no-train-state branch) assigns the result
    # directly to ``TrainState.params`` which is typed as ``nnx.State``. Convert
    # the plain merged dict back to an ``nnx.State`` by using the live template
    # (strip ``/value`` for the pure-dict form, then ``replace_by_pure_dict``).
    merged_plain_flat = {
        (k[: -len("/value")] if k.endswith("/value") else k): v
        for k, v in merged_flat.items()
    }
    merged_plain_tree = flax.traverse_util.unflatten_dict(merged_plain_flat, sep="/")

    # live_params is an nnx.State (abstract). replace_by_pure_dict mutates in
    # place, filling in concrete values while preserving the State's graph
    # structure (nnx.VariableState wrappers + types).
    live_params.replace_by_pure_dict(merged_plain_tree)
    return {"params": {"params": live_params}}


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    legacy_mode = _legacy_opt_state_requested()

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)

        # Check which items actually exist in the checkpoint directory
        ckpt_step = step if step is not None else checkpoint_manager.latest_step()
        ckpt_dir = checkpoint_manager.directory / str(ckpt_step)
        has_train_state = (ckpt_dir / "train_state").exists()

        if has_train_state:
            if legacy_mode:
                logging.warning(
                    "LEHOME_LEGACY_OPT_STATE=1: restoring opt_state with the old "
                    "AdamWBF16 tree shape; will cast mu bf16→fp32 and reshape "
                    "into the new AdamWWithAuxDecay layout after restore. "
                    "Remove the env var after the first save."
                )
                restore_train_state = dataclasses.replace(
                    train_state,
                    opt_state=_legacy_bf16_opt_state_target(train_state.opt_state),
                )
            else:
                restore_train_state = train_state

            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": restore_train_state,
                    "params": {"params": params},
                },
            )

            restored_ts = restored["train_state"]
            if legacy_mode:
                restored_ts = dataclasses.replace(
                    restored_ts,
                    opt_state=_convert_legacy_opt_state_to_new(restored_ts.opt_state),
                )
            result = _merge_params(restored_ts, restored["params"])
        else:
            # No train_state in checkpoint (e.g. downloaded from HF without optimizer state,
            # OR the user deleted train_state/ to force a fresh optimizer after adding new
            # params). Restore params only, and tolerate new live params that have no
            # counterpart on disk (they keep their random init; optimizer moments start
            # at zero via the fresh opt-state build below).
            logging.warning(
                "Checkpoint %s has no train_state. "
                "Restoring params only — optimizer starts fresh.",
                ckpt_dir,
            )
            restored = _restore_params_partial(checkpoint_manager, step, params)
            # Build a real TrainState with restored params and fresh optimizer.
            # The input `state` may be abstract (ShapeDtypeStruct from jax.eval_shape)
            # so we must create concrete values for step and opt_state.
            restored_params = restored["params"]["params"]
            real_step = jax.numpy.int32(ckpt_step)

            # In EMA mode, restored_params are EMA params. Use them as training
            # params too (the separate training params aren't in the checkpoint).
            # Copy for EMA so JAX doesn't try to donate the same buffer twice.
            training_params = restored_params
            ema_params_out = (
                jax.tree.map(lambda x: jax.numpy.array(x) if hasattr(x, 'dtype') else x, restored_params)
                if state.ema_decay is not None else None
            )

            # Materialize zero optimizer state matching the expected abstract shape.
            # state.opt_state is abstract (ShapeDtypeStruct) from jax.eval_shape —
            # we create real zero arrays with the same structure.
            fresh_opt_state = jax.tree.map(
                lambda x: jax.numpy.zeros(x.shape, x.dtype)
                if isinstance(x, jax.ShapeDtypeStruct) else x,
                state.opt_state,
            )
            result = training_utils.TrainState(
                step=real_step,
                params=training_params,
                model_def=state.model_def,
                tx=state.tx,
                opt_state=fresh_opt_state,
                ema_decay=state.ema_decay,
                ema_params=ema_params_out,
            )

    return result


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`.
    # If the restored train_state has an empty `params` dict, it means we're in the non-EMA case
    # and we should restore the main training params.
    if not train_state.params:
        # Non-EMA case: The saved 'params' are the training weights.
        return dataclasses.replace(train_state, params=params["params"])
    else:
        # EMA case: The saved 'params' are the ema_params, and train_state already has the training params.
        return dataclasses.replace(train_state, ema_params=params["params"])
