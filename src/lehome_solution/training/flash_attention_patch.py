"""Monkey-patch openpi's Gemma attention to use JAX flash attention.

The original implementation materializes the full [B, N, T, S] attention score
matrix in float32, which is ~25 GB per layer for batch=160, seq=2200.
jax.nn.dot_product_attention computes attention in tiles (flash attention),
never materializing the N×N score matrix, drastically reducing memory.

Usage: import this module before creating the model.

    from lehome_solution.training import flash_attention_patch
    flash_attention_patch.install()
"""

import logging

import flax.linen as nn
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

_INSTALLED = False


def install():
    """Install the flash attention monkey-patch on gemma.Attention."""
    global _INSTALLED
    if _INSTALLED:
        return

    from openpi.models import gemma as _gemma
    from openpi.models.gemma import _name, _apply_rope, lora

    class FlashAttention(_gemma.Attention):
        """Drop-in replacement using jax.nn.dot_product_attention."""

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
            # NOTE: do NOT pre-scale q — pass scale to dot_product_attention instead
            k = _apply_rope(k, positions=positions)

            assert q.dtype == k.dtype == v.dtype == dtype

            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                k = jnp.concatenate([cache_k, k], axis=1)
                v = jnp.concatenate([cache_v, v], axis=1)

            # --- Flash attention (replaces manual einsum + softmax) ---
            # q: [B, T, N, H], k: [B, S, K, H], v: [B, S, K, H]
            # attn_mask: [B, 1, T, S] — broadcasts over N heads
            # Note: cuDNN has stride requirements that JAX can't satisfy for
            # GQA with KV cache concat, so we use XLA implementation.
            encoded = jax.nn.dot_product_attention(
                q, k, v,
                mask=attn_mask,
                scale=self.configs[0].head_dim ** -0.5,
                implementation="xla",
            )
            # encoded: [B, T, N, H]

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

    # Replace the Attention class in the gemma module.
    # Block.__call__ references `Attention` by name at call time,
    # so this takes effect for all subsequent model construction.
    _gemma.Attention = FlashAttention
    _INSTALLED = True
    logger.info("Installed flash attention patch (replaced gemma.Attention)")
