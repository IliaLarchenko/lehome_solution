"""Per-loss weight composition helpers.

Every auxiliary loss in `train_step` is built from two atomic per-sample fields
set by the data loader:

    is_weight       — (1 / (N * p_i)) / ep_len  — fully unbiases sampling AND
                      normalises contribution per episode in one step. Non-zero
                      for every source, including BC/DAgger.
    garment_type_id — int garment type in [0, NUM_GARMENT_TYPES).

Plus per-sample debias multipliers supplied via the observation:

    success_debias_weight    — per-garment success debias × optional tail boost.
    checkpoint_debias_weight — per-garment checkpoint debias.

Aggregators:
    type_balanced_weighted_mean — used for aux losses; equalises the four garment
        types, then takes a weighted mean within each type. Preserves the natural
        weight distribution between samples of the same type.
"""

from __future__ import annotations

import jax.numpy as jnp

from lehome_solution.constants import NUM_GARMENT_TYPES


def type_balanced_weighted_mean(loss, weights, valid, garment_type_id):
    """Weighted mean within each garment type, then unweighted mean across types.

    For each garment type t present in the batch:
        ℓ_t = Σ_{i∈t, valid} w_i · loss_i / Σ_{i∈t, valid} w_i
    Final loss = mean_{t present} ℓ_t.
    Types absent from the batch contribute nothing (and are not counted).

    NaN-safe: ``loss`` is zeroed at invalid positions before the weighted sum,
    so callers can pass raw targets containing NaN.
    """
    safe_loss = jnp.where(valid, loss, 0.0)
    type_losses = []
    type_present = []
    for t in range(NUM_GARMENT_TYPES):
        mask_t = (garment_type_id == t) & valid
        w_t = jnp.where(mask_t, weights, 0.0)
        w_t_sum = jnp.maximum(w_t.sum(), 1e-8)
        type_loss = (safe_loss * w_t).sum() / w_t_sum
        type_losses.append(type_loss)
        type_present.append(mask_t.any())

    type_stack = jnp.stack(type_losses)
    present = jnp.stack(type_present)
    return jnp.where(present, type_stack, 0.0).sum() / jnp.maximum(
        present.sum().astype(jnp.float32), 1.0
    )
