"""Monkey-patch openpi's Gemma attention with Exclusive Self Attention (XSA).

After standard attention, subtract the projection of each output onto the
token's own value vector:

    z = y - (y . v_self / ||v_self||^2) * v_self

Reference: https://arxiv.org/abs/2603.09078

The patch replaces ``gemma.Attention`` and therefore applies to BOTH the
prefix (vision/state) expert and the suffix (action) expert.

``install(use_flash_attention=...)`` chooses between:
  - ``True``  → flash attention (``jax.nn.dot_product_attention``) + XSA post-step
  - ``False`` → manual attention einsum + XSA post-step
Either way v is computed locally (before the attention call), so XSA composes
cleanly with flash — the flash kernel returns y and we still have v in scope.
"""

import logging

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

_INSTALLED_VARIANT: str | None = None  # "flash" | "manual" | None


def _apply_xsa(encoded_bt_kgh: jnp.ndarray, self_v_btkh: jnp.ndarray, dtype) -> jnp.ndarray:
    """Subtract projection of ``encoded`` onto the self-value.

    encoded_bt_kgh: [B, T, K, G, H]
    self_v_btkh:    [B, T, K, H]
    Returns tensor of the same shape as ``encoded_bt_kgh`` in ``dtype``.
    """
    self_v_g = self_v_btkh[:, :, :, None, :]  # [B, T, K, 1, H] — broadcast over G
    enc_f32 = encoded_bt_kgh.astype(jnp.float32)
    sv_f32 = self_v_g.astype(jnp.float32)
    dot = jnp.sum(enc_f32 * sv_f32, axis=-1, keepdims=True)         # [B, T, K, G, 1]
    norm_sq = jnp.sum(sv_f32 * sv_f32, axis=-1, keepdims=True)      # [B, T, K, 1, 1]
    coef = dot / (norm_sq + 1e-6)
    return (enc_f32 - coef * sv_f32).astype(dtype)


def install(*, use_flash_attention: bool = False):
    """Install the XSA monkey-patch on gemma.Attention.

    ``use_flash_attention=True`` uses ``jax.nn.dot_product_attention`` for the
    softmax(QK^T/√d)V step and applies XSA as a post-process on the output.
    """
    global _INSTALLED_VARIANT
    variant = "flash" if use_flash_attention else "manual"
    if _INSTALLED_VARIANT == variant:
        return
    if _INSTALLED_VARIANT is not None:
        raise RuntimeError(
            f"xsa_attention_patch already installed as {_INSTALLED_VARIANT!r}; "
            f"cannot re-install as {variant!r}"
        )

    from openpi.models import gemma as _gemma
    from openpi.models.gemma import _name, _apply_rope, lora

    class XSAAttention(_gemma.Attention):
        """Drop-in replacement that applies XSA to the attention output."""

        @nn.compact
        def __call__(self, xs, positions, attn_mask, kv_cache):
            assert all(c.head_dim == self.configs[0].head_dim for c in self.configs)
            assert all(c.num_heads == self.configs[0].num_heads for c in self.configs)
            assert all(c.num_kv_heads == self.configs[0].num_kv_heads for c in self.configs)

            dtype = next(x.dtype for x in xs if x is not None)

            # --- Q, K, V projections (unchanged from original) ---
            qkvs = []
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is None:
                    continue
                if config.num_kv_heads == config.num_heads:
                    qkv_einsum = lora.Einsum(
                        shape=(3, config.num_heads, config.width, config.head_dim),
                        name=_name("qkv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_config=config.lora_configs.get("attn"),
                    )
                    qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
                else:
                    q_einsum = lora.Einsum(
                        shape=(config.num_heads, config.width, config.head_dim),
                        name=_name("q_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                        lora_config=config.lora_configs.get("attn"),
                    )
                    q = q_einsum("BTD,NDH->BTNH", x)
                    kv_einsum = lora.Einsum(
                        shape=(2, config.num_kv_heads, config.width, config.head_dim),
                        name=_name("kv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_config=config.lora_configs.get("attn"),
                    )
                    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                    qkvs.append((q, k, v))

            q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

            q = _apply_rope(q, positions=positions)
            if not use_flash_attention:
                # Manual path: pre-scale q (original gemma convention).
                q *= self.configs[0].head_dim ** -0.5
            # Flash path: pass scale= to dot_product_attention instead.
            k = _apply_rope(k, positions=positions)

            assert q.dtype == k.dtype == v.dtype == dtype

            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                k = jnp.concatenate([cache_k, k], axis=1)
                v = jnp.concatenate([cache_v, v], axis=1)

            num_kv_heads = self.configs[0].num_kv_heads

            if use_flash_attention:
                # Flash attention: q [B,T,N,H], k/v [B,S,K,H], mask [B,1,T,S]
                encoded_btnh = jax.nn.dot_product_attention(
                    q, k, v,
                    mask=attn_mask,
                    scale=self.configs[0].head_dim ** -0.5,
                    implementation="xla",
                )
                # Reshape to [B, T, K, G, H] for XSA's self-v broadcast.
                encoded = einops.rearrange(
                    encoded_btnh, "B T (K G) H -> B T K G H", K=num_kv_heads
                )
            else:
                # Manual attention (matches original gemma.Attention).
                q_grp = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)
                logits = jnp.einsum(
                    "BTKGH,BSKH->BKGTS", q_grp, k, preferred_element_type=jnp.float32
                )
                if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
                    raise ValueError(
                        f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
                    )
                big_neg = -2.3819763e38
                masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
                probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
                encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)

            # --- XSA: subtract projection onto self value ---
            # Self value for query position t is v at position (S - T) + t
            # in the concatenated K/V sequence — i.e. the last T rows of v.
            T = encoded.shape[1]
            self_v = v[:, -T:, :, :]  # [B, T, K, H]
            encoded = _apply_xsa(encoded, self_v, dtype)

            encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

            # --- Output projections (unchanged from original) ---
            out = []
            start = 0
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is not None:
                    end = start + x.shape[1]
                    out_einsum = lora.Einsum(
                        shape=(config.num_heads, config.head_dim, config.width),
                        name=_name("attn_vec_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                        lora_config=config.lora_configs.get("attn"),
                    )
                    out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                    start = end
                else:
                    out.append(None)

            return out, (k, v)

    _gemma.Attention = XSAAttention
    _INSTALLED_VARIANT = variant
    logger.info(f"Installed XSA attention patch (variant={variant})")
