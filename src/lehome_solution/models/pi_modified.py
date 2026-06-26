"""The main model for LeHome Challenge garment manipulation.

Based on Pi0.5 implementation from PhysicalIntelligence/openpi
"""

import logging
import pathlib

import numpy as np
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.models import siglip as _siglip
from openpi.models.pi0 import make_attn_mask, posemb_sincos
from openpi.shared import array_typing as at

# Import from our custom modules
from lehome_solution.models import pi_modified_config
from lehome_solution.models.observation import Observation, preprocess_observation
from lehome_solution.constants import NUM_GARMENT_TYPES

logger = logging.getLogger("lehome_solution")



# Module-level cache for precomputed correction matrices.
# Stored outside NNX Module to avoid polluting state/graphdef trees.
# Keyed by id(model) but since models are recreated, we just use a single slot.
_correction_matrices_cache: jnp.ndarray | None = None


def _bce_with_logits(logits, target):
    """Numerically stable BCE from logits, computed in float32.

    Equivalent to ``-(y * log(sigmoid(z)) + (1-y) * log(1-sigmoid(z)))`` but
    avoids the log(0) cliff and the bf16 precision loss around saturated
    predictions (see analysis of "S-curve jitter" on confident heads). Keeps
    the heads themselves in bf16 — only the loss evaluation is upcast.
    """
    z = logits.astype(jnp.float32)
    y = target.astype(jnp.float32)
    # -y*log_sigmoid(z) - (1-y)*log_sigmoid(-z) is numerically stable for all z.
    return -(y * jax.nn.log_sigmoid(z) + (1.0 - y) * jax.nn.log_sigmoid(-z))


class KVCacheTransform(nnx.Module):
    """Transforms prefix KV cache by mixing across layers.
    
    Each destination layer's K and V become learnable linear combinations
    of all source layers' K and V, plus a bias term. This allows the action
    expert to attend to learned combinations of VLM layers rather than being
    forced to attend layer-by-layer.
    
    Initialized as identity transform (k_coeffs = I, bias = 0) so the model
    starts with the same behavior as without transformation.
    """
    
    def __init__(self, num_layers: int, head_dim: int, num_kv_heads: int, rngs: nnx.Rngs):
        # K transformation: [dest_layer, src_layer]
        # Initialize as identity so transformation is initially a no-op
        self.k_coeffs = nnx.Param(jnp.eye(num_layers, dtype=jnp.float32))
        
        # K bias: [layer, num_kv_heads, head_dim]
        # Initialize as zeros
        self.k_bias = nnx.Param(jnp.zeros((num_layers, num_kv_heads, head_dim), dtype=jnp.float32))
        
        # V transformation (independent from K)
        self.v_coeffs = nnx.Param(jnp.eye(num_layers, dtype=jnp.float32))
        self.v_bias = nnx.Param(jnp.zeros((num_layers, num_kv_heads, head_dim), dtype=jnp.float32))
    
    def __call__(self, kv_cache: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Transform KV cache by mixing across layers.
        
        Args:
            kv_cache: Tuple of (cache_k, cache_v) where each has shape
                     [num_layers, batch, seq_len, num_kv_heads, head_dim]
        
        Returns:
            Transformed (k_new, v_new) with same shape and dtype as input
        """
        cache_k, cache_v = kv_cache
        # Shape: [layers, batch, seq_len, num_kv_heads, head_dim]
        
        # Preserve original dtype (important for bfloat16 training)
        original_dtype = cache_k.dtype
        
        # Transform K: each destination layer is a weighted combination of all source layers
        # k_new[dest] = sum_src(k_coeffs[dest, src] * cache_k[src]) + k_bias[dest]
        # Einsum: [dest, src] @ [src, batch, seq, heads, dim] -> [dest, batch, seq, heads, dim]
        k_new = jnp.einsum('ds,sbtkh->dbtkh', self.k_coeffs.value, cache_k)
        k_new = k_new + self.k_bias.value[:, None, None, :, :]  # Add bias
        
        # Transform V (same operation, independent parameters)
        v_new = jnp.einsum('ds,sbtkh->dbtkh', self.v_coeffs.value, cache_v)
        v_new = v_new + self.v_bias.value[:, None, None, :, :]
        
        # Cast back to original dtype
        k_new = k_new.astype(original_dtype)
        v_new = v_new.astype(original_dtype)
        
        return (k_new, v_new)


class PiModified(_model.BaseModel):
    def __init__(self, config: pi_modified_config.PiModifiedConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        
        # Store config for later use
        self.config = config
        
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Override depth for smaller model variants
        if config.num_llm_layers is not None:
            import dataclasses as _dc
            paligemma_config = _dc.replace(paligemma_config, depth=config.num_llm_layers)
            action_expert_config = _dc.replace(action_expert_config, depth=config.num_llm_layers)
            logger.info(f"Using reduced LLM depth: {config.num_llm_layers} layers (default: 18)")

        # Initialize Gemma models with AdaRMS (Pi05 style)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        # Replace the massive Gemma vocab embedding (257152 × width = ~527M params)
        # with a tiny dummy when using dedicated state embedding.
        # The Gemma embedder is never called when use_state_embedding=True,
        # but the submodule allocates it unconditionally.  This hack reclaims ~2 GB.
        if config.use_state_embedding:
            dummy = jnp.zeros((1, paligemma_config.width), dtype=jnp.float32)
            llm.embedder['input_embedding'].value = dummy
            logger.info(
                "Replaced Gemma embedding table (257152 x %d) with dummy (1 x %d)",
                paligemma_config.width, paligemma_config.width,
            )

        # Initialize vision model
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant=config.siglip_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # KV cache transformation for cross-layer attention
        # Allows action expert to attend to learned combinations of VLM layers
        if config.use_kv_transform:
            self.kv_transform = KVCacheTransform(
                num_layers=paligemma_config.depth,
                head_dim=paligemma_config.head_dim,
                num_kv_heads=paligemma_config.num_kv_heads,
                rngs=rngs
            )
        else:
            self.kv_transform = None
        
        # Small state embedding (replaces massive Gemma vocab embedder for state tokens)
        if config.use_state_embedding:
            self.state_embedding = nnx.Embed(
                num_embeddings=256,
                features=paligemma_config.width,
                rngs=rngs,
            )
            logger.info("Using dedicated state embedding (256 x %d) instead of Gemma embedder", paligemma_config.width)

        # Pi05 style layers
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Correlated noise generation
        # Initialize as NNX Intermediate (excluded from checkpoints, loaded from norm_stats)
        flat_dim = config.action_horizon * config.action_dim
        self.action_correlation_cholesky = nnx.Intermediate(
            jnp.eye(flat_dim),  # Identity matrix as placeholder (360x360)
        )
        self.use_correlated_noise = config.use_correlated_noise
        self.correlation_beta = config.correlation_beta  # Shrinkage parameter for regularization

        # FAST auxiliary training components
        if config.use_fast_auxiliary:
            # FAST embedding layer (vocab_size → paligemma_width)
            # Use paligemma width (2048) to match other prefix tokens
            self.fast_token_embedding = nnx.Embed(
                num_embeddings=config.fast_vocab_size,
                features=paligemma_config.width,
                rngs=rngs
            )
            
            # FAST projection head (paligemma_width → vocab_size)
            self.fast_token_proj = nnx.Linear(
                paligemma_config.width,
                config.fast_vocab_size,
                rngs=rngs
            )
            
            logger.info(f"FAST auxiliary enabled, vocab_size={config.fast_vocab_size}")

        # Advantage embedding (pi0.6* style RL conditioning).
        # Two learnable vectors indexed 0=positive, 1=negative.
        # A single token from this table is prepended to the action-expert prefix
        # so the model can condition on the quality of the current observation.
        # When neutral (no clear signal), the token is masked out via attention mask.
        if config.use_advantage_embedding:
            self.advantage_embeddings = nnx.Embed(
                num_embeddings=2,
                features=paligemma_config.width,
                rngs=rngs,
            )
            logger.info("Advantage embeddings enabled (2 vectors: positive/negative, neutral=masked)")

            # AdaRMS channel for advantage conditioning.
            # A single learnable vector at action-expert width; added to `adarms_cond`
            # when the per-sample advantage mask is active, so every RMSNorm in the
            # action expert gets an advantage-dependent (scale, shift, gate).
            # Zero-init -> identity at step 0 -> bit-identical resume from existing
            # checkpoints.  CFG uncond: subtract this vector from adarms_cond.
            self.advantage_adarms_vec = nnx.Param(
                jnp.zeros((1, action_expert_config.width), dtype=jnp.float32)
            )
            logger.info(
                f"Advantage AdaRMS vector enabled (1 x {action_expert_config.width}, zero-init)"
            )

        # Garment type input embedding: 4 learnable vectors (one per garment type).
        # Placed in the state attention group (group 1), invisible to group 0
        # (images, success_query, task tokens) so the garment_type_head still
        # predicts from images alone.
        if config.use_garment_type_input:
            self.garment_type_input_embedding = nnx.Embed(
                num_embeddings=NUM_GARMENT_TYPES,
                features=paligemma_config.width,
                rngs=rngs,
            )
            logger.info(f"Garment type input embedding enabled ({NUM_GARMENT_TYPES} vectors)")

            # AdaRMS channel for garment type conditioning.
            # Separate embedding at action-expert width (not shared with the prefix
            # token because widths differ: 1024 vs 2048, and the two paths serve
            # different roles — prefix is attention content, adarms is modulation).
            # Zero-init so training starts bit-identical to the pre-AdaRMS model.
            self.garment_type_adarms_embedding = nnx.Embed(
                num_embeddings=NUM_GARMENT_TYPES,
                features=action_expert_config.width,
                embedding_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
            logger.info(
                f"Garment type AdaRMS embedding enabled "
                f"({NUM_GARMENT_TYPES} x {action_expert_config.width}, zero-init)"
            )

        # Integrated success head: learned query token + linear head → P(success) ∈ (0,1)
        if config.use_success_head:
            self.success_query_token = nnx.Param(jax.random.normal(rngs(), (1, paligemma_config.width)) * 0.02)
            self.success_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
            logger.info("Success head enabled: learned query token + Linear(2048, 1) → sigmoid → P(success)")
            # Auxiliary heads: all share the same query token and hidden representation
            self.checkpoint_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
            self.garment_type_head = nnx.Linear(paligemma_config.width, NUM_GARMENT_TYPES, rngs=rngs)
            self.completion_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
            self.ttc_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
            logger.info("Auxiliary heads enabled: checkpoint (P(r>=0.5)), garment_type (4-class), completion (%%), ttc")
            # Keypoint-distance head (Head 1): predicts 21 normalized per-condition
            # distances from the same success_query hidden state. Per-garment-type
            # slices (see constants.KEYPOINT_SLICES). Raw linear output (no
            # activation): values can exceed 1 to express "still far above threshold".
            from lehome_solution.constants import KEYPOINT_HEAD_WIDTH as _KP_W
            self.keypoint_distance_head = nnx.Linear(paligemma_config.width, _KP_W, rngs=rngs)
            logger.info(f"Keypoint-distance head enabled: Linear({paligemma_config.width}, {_KP_W}) — per-garment slices")

            # World-modeling / Q head 2 (post-FAST, training-only). Separate
            # learned query token appended after FAST tokens in the prefix
            # (higher cumsum group → invisible to all prior tokens; visible to
            # itself only). Predicts future success, completion, and keypoint
            # distances at t + WM_FUTURE_HORIZON. Zero-gradient-to-VLM is NOT
            # enforced — gradients flow back to shape the VLM representation.
            if config.use_wm_fast_head:
                self.wm_fast_query_token = nnx.Param(
                    jax.random.normal(rngs(), (1, paligemma_config.width)) * 0.02
                )
                self.wm_fast_success_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
                self.wm_fast_completion_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
                self.wm_fast_keypoint_head = nnx.Linear(paligemma_config.width, _KP_W, rngs=rngs)
                logger.info("WM-FAST head enabled: learned query after FAST → success/completion/keypoint future targets")

            # World-modeling / Q head 3 (post-flow). Separate learned query
            # token appended to the tail of the action-expert suffix. Fed by
            # denoised actions (noisy during training → loss weight (1-t)).
            # Width matches action_expert_config.
            if config.use_wm_flow_head:
                self.wm_flow_query_token = nnx.Param(
                    jax.random.normal(rngs(), (1, action_expert_config.width)) * 0.02
                )
                self.wm_flow_success_head = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
                self.wm_flow_completion_head = nnx.Linear(action_expert_config.width, 1, rngs=rngs)
                self.wm_flow_keypoint_head = nnx.Linear(action_expert_config.width, _KP_W, rngs=rngs)
                logger.info("WM-flow head enabled: learned query at end of suffix → future targets, loss weight = (1 - t)")

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _compute_advantage_token(
        self,
        advantage: at.Float[at.Array, " b"],
        rng: at.KeyArrayLike | None,
        *,
        inference: bool = False,
    ) -> tuple[at.Float[at.Array, "b d"], at.Bool[at.Array, " b"]]:
        """Select advantage embedding per batch sample.

        Always uses the positive embedding (index 0). Negative embedding
        (index 1) is kept in params for checkpoint compat but never selected.

        Returns:
            (token, mask) where mask is True for active tokens, False for masked.

        During inference: always positive embedding, mask=True.
        During training:
          - advantage < 0 → always masked out (neutral).
          - advantage >= 0 → stochastic masking: P(neutral) ramps from
            ``neutral_prob_at_zero`` (adv=0) to ``neutral_prob_at_max``
            (adv >= ``max_adv_for_neutral``).
        """
        batch_size = advantage.shape[0]
        d = self.advantage_embeddings.embedding.value.shape[-1]

        if inference:
            idx = jnp.zeros((batch_size,), dtype=jnp.int32)  # always positive
            token = self.advantage_embeddings(idx).astype(self.config.dtype)
            mask = jnp.ones((batch_size,), dtype=jnp.bool_)
            return token, mask

        # Always use positive embedding (index 0). Negative embedding kept in
        # params for checkpoint compat but never selected.
        token = self.advantage_embeddings(
            jnp.zeros((batch_size,), dtype=jnp.int32)
        ).astype(self.config.dtype)

        # Mask out (neutral) when advantage < 0; stochastic masking when >= 0.
        is_negative = advantage < self.config.positive_advantage_threshold
        # For non-negative: P(neutral) ramps from neutral_prob_at_zero (adv=0)
        # down to neutral_prob_at_max (adv >= max_adv_for_neutral).
        t = jnp.minimum(advantage / self.config.max_adv_for_neutral, 1.0)
        t = jnp.maximum(t, 0.0)
        p_neutral = (
            (1.0 - t) * self.config.neutral_prob_at_zero
            + t * self.config.neutral_prob_at_max
        )
        uniform = jax.random.uniform(rng, (batch_size,))
        use_neutral = is_negative | (uniform < p_neutral)

        token = jnp.where(use_neutral[:, None], 0.0, token)
        mask = ~use_neutral

        return token, mask

    def load_correlation_matrix(self, norm_stats: dict):
        """Load full correlation matrix from normalization statistics and apply shrinkage.
        
        This should be called after model initialization when norm_stats are available.
        Applies shrinkage regularization: S_reg = beta * S + (1-beta) * I for robustness.
        
        Args:
            norm_stats: Dictionary containing normalization statistics (from normalize.load()),
                       with 'actions' key containing NormStats with action_correlation_cholesky field.
                       
        Raises:
            ValueError: If use_correlated_noise=True but correlation matrix is missing.
            TypeError: If norm_stats structure is incorrect.
        """
        if not self.use_correlated_noise:
            logger.info("Correlated noise disabled in config, skipping correlation matrix loading")
            return
        
        if not isinstance(norm_stats, dict):
            raise TypeError(
                f"norm_stats must be a dict, got {type(norm_stats).__name__}. "
                "Ensure norm_stats are loaded using openpi.shared.normalize.load()."
            )

        if 'actions' not in norm_stats:
            raise ValueError(
                "use_correlated_noise=True but 'actions' key not found in norm_stats. "
                f"Found keys: {list(norm_stats.keys())}. "
                "Run compute_norm_stats.py with --correlation flag to generate correlation matrix."
            )
        
        actions_stats = norm_stats['actions']

        # Support both dict and attribute access for the cholesky field.
        if isinstance(actions_stats, dict):
            chol_matrix = actions_stats.get('action_correlation_cholesky')
        elif hasattr(actions_stats, 'action_correlation_cholesky'):
            chol_matrix = actions_stats.action_correlation_cholesky
        else:
            raise TypeError(
                f"norm_stats['actions'] has unexpected type {type(actions_stats).__name__} "
                f"and cannot access 'action_correlation_cholesky'. "
                "Ensure norm_stats are loaded using openpi.shared.normalize.load()."
            )

        if chol_matrix is None:
            raise ValueError(
                "use_correlated_noise=True but 'action_correlation_cholesky' is None in norm_stats['actions']. "
                "This means the correlation matrix was not computed during norm_stats generation. "
                "Run compute_norm_stats.py with --correlation flag to generate correlation matrix."
            )
        
        expected_dim = self.action_horizon * self.action_dim
        try:
            L = jnp.array(chol_matrix)
        except Exception as e:
            raise ValueError(
                f"Failed to convert action_correlation_cholesky to array: {e}. "
                "The correlation matrix may be corrupted or in an invalid format."
            )
        
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(
                f"action_correlation_cholesky must be a square 2D matrix, got shape {L.shape}. "
                f"Expected shape: ({expected_dim}, {expected_dim})"
            )
        
        if L.shape[0] != expected_dim:
            raise ValueError(
                f"action_correlation_cholesky has wrong dimensions: {L.shape[0]}x{L.shape[0]}. "
                f"Expected {expected_dim}x{expected_dim} (action_horizon={self.action_horizon} * action_dim={self.action_dim}). "
                "This indicates the correlation matrix was computed for a different action space configuration."
            )
        
        # Σ_reg = beta * Σ + (1-beta) * I  (shrinkage regularization)
        Sigma = L @ L.T
        beta = self.correlation_beta
        logger.info(f"Applying shrinkage regularization with beta={beta:.2f}")

        Sigma_reg = beta * Sigma + (1 - beta) * jnp.eye(Sigma.shape[0])

        try:
            L_reg = jnp.linalg.cholesky(Sigma_reg)
        except Exception as e:
            raise RuntimeError(
                f"Cholesky decomposition failed on regularized covariance: {e}. "
                "This indicates the regularized correlation matrix is not positive definite. "
                f"Current beta={beta:.2f}. Try decreasing correlation_beta closer to 0.0 for more shrinkage/regularization."
            )
        
        self.action_correlation_cholesky.value = L_reg

        logger.info(
            f"✓ Loaded correlation matrix with shape {L_reg.shape} "
            f"(beta={beta:.2f} shrinkage applied)"
        )
        logger.info(
            f"  Memory usage: {L_reg.nbytes / 1024 / 1024:.2f} MB"
        )

        # Precompute full-space correction matrices for batched inpainting.
        # Stored in module-level global to avoid NNX state/graphdef tracking.
        # NB: max_initial MUST be >= the largest ``actions_to_keep`` value any
        # caller uses. Real-robot submission uses atk=15 (one full second of
        # 30 Hz sim-frames overlap). Sim eval defaults to atk=0 (no inpaint),
        # so the previous default of 10 was sufficient for sim but silently
        # corrupted real: indexing ``cache[15]`` clipped to ``cache[10]`` (JAX
        # OOB clip), so the inpaint correction used the wrong propagation
        # matrix for 5 of the 15 inpainted timesteps — injecting noise across
        # consecutive chunks on the real robot.
        global _correction_matrices_cache
        logger.info("Precomputing correction matrices for batched inpainting...")
        _correction_matrices_cache = self._precompute_all_correction_matrices(max_initial=20)

    def generate_correlated_noise(
        self, 
        rng: at.KeyArrayLike, 
        batch_size: int,
    ) -> at.Float[at.Array, "b {self.action_horizon} {self.action_dim}"]:
        """Generate correlated noise matching action covariance structure.
        
        Uses full correlation matrix with optional beta shrinkage for robustness.
        
        Args:
            rng: Random key for noise generation
            batch_size: Number of noise samples to generate
            
        Returns:
            Correlated noise with shape [batch_size, action_horizon, action_dim]
            
        Raises:
            RuntimeError: If use_correlated_noise=True but correlation matrix not loaded.
        """
        if not self.use_correlated_noise:
            # Independent Gaussian noise when correlated noise is disabled
            return jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        if _correction_matrices_cache is None:
            raise RuntimeError(
                "use_correlated_noise=True but correlation matrix is not loaded. "
                "Ensure load_correlation_matrix() was called during model initialization. "
                "Run compute_norm_stats.py with --correlation flag to generate correlation matrix."
            )
        
        flat_dim = self.action_horizon * self.action_dim
        standard_normal = jax.random.normal(rng, (batch_size, flat_dim))
        correlated_flat = standard_normal @ self.action_correlation_cholesky.value.T
        return correlated_flat.reshape(batch_size, self.action_horizon, self.action_dim)

    def _precompute_all_correction_matrices(self, max_initial: int = 10) -> jnp.ndarray:
        """Precompute full-space correction matrices for all possible inpaint lengths.

        Returns [max_initial+1, flat_dim, flat_dim] array where index n gives the
        360×360 correction matrix for n initial actions.  Entry [n, j, i] maps a
        delta at observed dim j to a correction at unobserved dim i.
        Index 0 = no inpainting (all zeros).

        Must be called after load_correlation_matrix().
        """
        flat_dim = self.action_horizon * self.action_dim  # 360
        all_C = np.zeros((max_initial + 1, flat_dim, flat_dim), dtype=np.float32)

        L_value = self.action_correlation_cholesky.value
        if L_value is None:
            logger.warning("Correlation not loaded — correction matrices will be zeros")
            return jnp.array(all_C)

        L = np.asarray(L_value)
        Sigma = L @ L.T

        for n in range(1, max_initial + 1):
            inpaint_dims = self.action_dim  # always 12
            O_list = [t * self.action_dim + d for t in range(n) for d in range(inpaint_dims)]
            O_set = set(O_list)
            U_list = [i for i in range(flat_dim) if i not in O_set]
            O_idx = np.array(O_list, dtype=np.int32)
            U_idx = np.array(U_list, dtype=np.int32)

            Sigma_OO = Sigma[np.ix_(O_idx, O_idx)]
            Sigma_UO = Sigma[np.ix_(U_idx, O_idx)]
            eps = 1e-6 * max(np.mean(np.diag(Sigma_OO)), 1.0)
            Sigma_OO_reg = Sigma_OO + eps * np.eye(len(O_idx))

            # correction_small: [|U|, |O|] = Sigma_UO @ Sigma_OO^{-1}
            correction_small = np.linalg.solve(Sigma_OO_reg, Sigma_UO.T).T

            # Scatter into full-space [360, 360]:
            # C_full[o, u] = correction_small[u_local, o_local]
            # so that delta @ C_full gives corrections at U positions
            for o_local, o_global in enumerate(O_list):
                for u_local, u_global in enumerate(U_list):
                    all_C[n, o_global, u_global] = correction_small[u_local, o_local]

            logger.info(f"  Precomputed correction matrix for {n} initial actions ({len(O_list)} observed, {len(U_list)} free)")

        result = jnp.array(all_C)
        logger.info(f"All correction matrices precomputed: {result.shape} ({result.nbytes / 1024 / 1024:.1f} MB)")
        return result

    @at.typecheck
    def embed_prefix(
        self,
        obs: Observation,
        advantage_token=None,  # [B, D] pre-selected advantage embedding, or None
        advantage_mask=None,   # [B] bool, True=active, False=masked (neutral)
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        int | None,
        int | None,  # advantage_token_idx: position of advantage token in prefix (or None)
        int | None,  # wm_fast_token_idx: position of WM-FAST token in prefix (or None)
        tuple[int, int] | None,  # state_token_range: (start, end) indices of state tokens (or None)
    ]:
        """
        Embed prefix: images + [success_query] + state + [garment_type_input] + [advantage] + [FAST_tokens] + [WM_FAST].

        Attention hierarchy (via cumsum ar_mask trick):
          - Group 0 (cumsum=0): images + success — bidirectional among themselves
          - Group 1 (cumsum=1): state — sees group 0 + self, invisible to group 0
          - Group 2 (cumsum=2): advantage — sees groups 0-1 + self, invisible to groups 0-1
          - Group 3+ (cumsum=3,4,...): FAST — causal, sees everything, invisible to prefix
          - Group 3+fast_len (when WM-FAST added): WM-FAST token — sees everything
            before (incl. FAST), invisible to earlier tokens; training-only.

        Args:
            obs: Observation (may include fast_tokens and fast_token_mask)
            advantage_token: [B, D] pre-selected advantage embedding, or None
            advantage_mask: [B] bool per-sample mask (True=active, False=masked/neutral)

        Returns:
            tokens, input_mask, ar_mask, num_image_tokens, advantage_token_idx, wm_fast_token_idx, state_token_range
        """
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Embed images
        # Respect freeze_vision_backbone config: if frozen, always use train=False
        # If not frozen, use the model's training state (self.deterministic)
        vision_train_mode = (not self.deterministic) and (not self.config.freeze_vision_backbone)
        num_image_tokens = 0

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=vision_train_mode)
            num_image_tokens += image_tokens.shape[1]

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # Track number of image tokens for value head attention masking
        if not self.config.use_success_head:
            num_image_tokens = None

        # Value query token: placed right after images, before state.
        if self.config.use_success_head:
            batch_size_local = obs.state.shape[0]
            vq = jnp.broadcast_to(self.success_query_token.value, (batch_size_local, 1, self.success_query_token.value.shape[-1]))
            tokens.append(vq)
            input_mask.append(jnp.ones((batch_size_local, 1), dtype=jnp.bool_))
            ar_mask += [False]

        # Add state as discrete tokens (Pi05 style)
        # Discretize state into bins
        discretized_state = jnp.digitize(obs.state, bins=jnp.linspace(-1, 1, 256 + 1)[:-1]) - 1
        discretized_state = jnp.clip(discretized_state, 0, 255)  # Ensure valid range

        # Embed each dimension of the discretized state
        state_tokens = []
        if self.config.use_state_embedding:
            # Use dedicated small embedding (256 entries) with sqrt scaling to match Gemma convention
            scale = jnp.sqrt(float(self.state_embedding.features))
            for i in range(obs.state.shape[-1]):
                state_dim_tokens = self.state_embedding(discretized_state[:, i:i+1]) * scale
                state_tokens.append(state_dim_tokens)
        else:
            for i in range(obs.state.shape[-1]):
                state_dim_tokens = self.PaliGemma.llm(discretized_state[:, i:i+1], method="embed")
                state_tokens.append(state_dim_tokens)

        state_token_range = None
        if state_tokens:
            state_tokens = jnp.concatenate(state_tokens, axis=1)  # shape: [batch_size, state_dim, embed_dim]
            state_token_start = sum(t.shape[1] for t in tokens)
            state_token_range = (state_token_start, state_token_start + state_tokens.shape[1])
            tokens.append(state_tokens)
            input_mask.append(jnp.ones((obs.state.shape[0], obs.state.shape[-1]), dtype=jnp.bool_))
            # State = cumsum group 1: first token True (bumps cumsum), rest False.
            # State sees group 0 (images+success) + self, but group 0 can't see state.
            ar_mask += [True] + [False] * (state_tokens.shape[1] - 1)

        # Garment type input token: same attention group as state (group 1).
        # ar_mask=False keeps it in group 1 (no cumsum bump).
        # Bidirectional with state tokens, visible to advantage/FAST/suffix,
        # but INVISIBLE to group 0 (images, success_query_token).
        if self.config.use_garment_type_input and obs.garment_type_id is not None:
            batch_size_local = obs.state.shape[0]
            gt_input_ids = jnp.clip(obs.garment_type_id, 0, NUM_GARMENT_TYPES - 1)  # [B]
            gt_input_emb = self.garment_type_input_embedding(gt_input_ids)  # [B, D]
            tokens.append(gt_input_emb[:, None, :])  # [B, 1, D]
            input_mask.append(jnp.ones((batch_size_local, 1), dtype=jnp.bool_))
            ar_mask += [False]  # Same group as state (group 1)

        # Advantage embedding token (pi0.6* style RL conditioning)
        # cumsum group 2: True bumps cumsum again. Advantage sees groups 0-1 + self,
        # but groups 0-1 can't see advantage. Per-sample neutral masking applied after make_attn_mask().
        advantage_token_idx = None
        if self.config.use_advantage_embedding and advantage_token is not None:
            batch_size_local = obs.state.shape[0]
            advantage_token_idx = sum(t.shape[1] for t in tokens)  # position in full prefix
            tokens.append(advantage_token[:, None, :])  # [B, 1, D]
            input_mask.append(jnp.ones((batch_size_local, 1), dtype=jnp.bool_))
            ar_mask += [True]  # cumsum group 2: sees groups 0-1+self, invisible to them

        # FAST tokens (from observation if provided)
        wm_fast_token_idx = None
        fast_present = self.config.use_fast_auxiliary and obs.fast_tokens is not None
        if fast_present:
            fast_tokens = obs.fast_tokens  # [B, T]
            fast_token_mask = obs.fast_token_mask  # [B, T]

            # Teacher forcing: shift right [BOS, tok0, tok1, ..., tok_{T-1}]
            bos_token = jnp.zeros((fast_tokens.shape[0], 1), dtype=jnp.int32)
            shifted_tokens = jnp.concatenate([bos_token, fast_tokens[:, :-1]], axis=1)

            # Shift mask too: [True, mask_0, mask_1, ..., mask_{T-1}]
            bos_mask = jnp.ones((fast_tokens.shape[0], 1), dtype=jnp.bool_)
            shifted_mask = jnp.concatenate([bos_mask, fast_token_mask[:, :-1]], axis=1)

            # Embed using FAST embedding layer (NOT Paligemma!)
            fast_token_emb = self.fast_token_embedding(shifted_tokens)  # [B, T, D]

            tokens.append(fast_token_emb)
            input_mask.append(shifted_mask)  # Use the actual token mask
            # Causal for FAST: ALL tokens are causal (pure autoregressive)
            ar_mask += [True] * shifted_tokens.shape[1]

        # WM-FAST query token (Head 2): appended AFTER FAST, training-only.
        # ar_mask=True bumps cumsum → new group → invisible to all earlier tokens
        # (including FAST), but itself can attend to everything before.
        # Only added when FAST is present (i.e. training); action expert removes
        # both FAST and this token from the KV cache downstream.
        if fast_present and self.config.use_wm_fast_head and getattr(self, "wm_fast_query_token", None) is not None:
            batch_size_local = obs.state.shape[0]
            wm_fast_token_idx = sum(t.shape[1] for t in tokens)  # position before append
            wm_tok = jnp.broadcast_to(
                self.wm_fast_query_token.value,
                (batch_size_local, 1, self.wm_fast_query_token.value.shape[-1]),
            )
            tokens.append(wm_tok)
            input_mask.append(jnp.ones((batch_size_local, 1), dtype=jnp.bool_))
            ar_mask += [True]  # new cumsum group

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, num_image_tokens, advantage_token_idx, wm_fast_token_idx, state_token_range

    @at.typecheck
    def embed_suffix(
        self,
        obs: Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        advantage_mask: at.Bool[at.Array, " b"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        """Embed the action-expert suffix tokens and build the AdaRMS conditioning vector.

        AdaRMS conditioning is a per-sample vector that modulates every RMSNorm
        (pre-attention and pre-FFW) in every action-expert layer via (scale, shift, gate).
        We fuse three signals into it (all at action-expert width):

          c = time_emb + garment_contrib + masked_advantage_contrib

        - time_emb: flow-matching timestep conditioning (existing behavior).
        - garment_contrib: per-sample garment-type vector from
          ``garment_type_adarms_embedding`` (zero-init).
        - masked_advantage_contrib: per-sample advantage vector from
          ``advantage_adarms_vec`` (zero-init), gated by ``advantage_mask``
          (True = active, False = neutral -> contribute zero).

        Args:
            obs: Observation (``garment_type_id`` used when enabled).
            noisy_actions: The flow-matching noisy action chunk.
            timestep: Flow-matching timestep per sample.
            advantage_mask: Per-sample advantage activation mask from
                ``_compute_advantage_token``. None means advantage disabled
                OR the caller wants no advantage contribution (e.g. CFG uncond).
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Pi05 style: no explicit state token in suffix (it's in prefix as discrete tokens)

        action_tokens = self.action_in_proj(noisy_actions)
        # Embed timestep using sine-cosine positional encoding
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        # Pi05 style: time MLP for adaRMS
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens

        # Base adaRMS conditioning: timestep only.
        adarms_cond = time_emb  # [B, action_expert_width]

        # Garment-type AdaRMS channel: per-sample additive contribution.
        # Zero-init means no effect at step 0; gradients then specialize per type.
        if self.config.use_garment_type_input and obs.garment_type_id is not None:
            gt_ids = jnp.clip(obs.garment_type_id, 0, NUM_GARMENT_TYPES - 1)  # [B]
            gt_contrib = self.garment_type_adarms_embedding(gt_ids).astype(adarms_cond.dtype)  # [B, D]
            adarms_cond = adarms_cond + gt_contrib

        # Advantage AdaRMS channel: single learnable vector, masked per sample.
        # Uses the SAME mask already computed by ``_compute_advantage_token`` so
        # training-time stochastic neutral masking (and CFG-style inference
        # toggling) apply transparently to this channel.
        if (
            self.config.use_advantage_embedding
            and advantage_mask is not None
        ):
            adv_vec = self.advantage_adarms_vec.value.astype(adarms_cond.dtype)  # [1, D]
            adv_contrib = jnp.where(
                advantage_mask[:, None],  # [B, 1]
                adv_vec,                  # [1, D]
                jnp.zeros_like(adv_vec),
            )  # broadcasts to [B, D]
            adarms_cond = adarms_cond + adv_contrib

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))

        # image/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # WM-flow query token (Head 3): appended at the TAIL of the suffix.
        # ar_mask=False keeps it in the same cumsum group as action tokens →
        # bidirectional with actions (no secrets leaked, and action tokens
        # are free to attend to it per the user spec).
        # Action extraction downstream uses suffix_out[:, :action_horizon]
        # to avoid grabbing this WM token as an action.
        if self.config.use_wm_flow_head and getattr(self, "wm_flow_query_token", None) is not None:
            batch_size_local = noisy_actions.shape[0]
            wm_tok = jnp.broadcast_to(
                self.wm_flow_query_token.value,
                (batch_size_local, 1, self.wm_flow_query_token.value.shape[-1]),
            )
            tokens.append(wm_tok)
            input_mask.append(jnp.ones((batch_size_local, 1), dtype=jnp.bool_))
            ar_mask += [False]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Not used - we only use compute_detailed_loss() for training."""
        raise NotImplementedError("Use compute_detailed_loss() instead")

    @override
    def compute_detailed_loss(
        self, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions, *, train: bool = False, num_flow_samples: int = 1
    ) -> dict[str, at.Float[at.Array, "*b"]]:
        """
        Compute detailed loss with multiple flow matching samples.
        
        Simplified approach using KV cache:
        - Compute prefix KV cache once (with FAST tokens)
        - Remove FAST tokens from cache (action expert doesn't attend to FAST)
        - Process N flow samples independently, each reusing the same cached prefix
        - Each sample has different noise and different time
        - Average losses across samples
        """
        losses = {}

        preprocess_rng, rng = jax.random.split(rng)
        observation = preprocess_observation(
            preprocess_rng, observation, train=train,
            rotate_top_image=self.config.rotate_top_image,
            train_aug=self.config.train_aug,
        )

        batch_size = actions.shape[0]

        if train and self.config.train_aug is not None and self.config.train_aug.calibration_scale_noise > 0:
            cal_rng, rng = jax.random.split(rng)
            v = self.config.train_aug.calibration_scale_noise
            scale = 1.0 + jax.random.uniform(cal_rng, (batch_size, 12), minval=-v, maxval=v)
            if observation.state is not None:
                observation = observation.replace(state=observation.state * scale)
            actions = actions * scale[:, None, :]

        # Compute per-sample advantage token (stochastic selection during training).
        # Falls back to masked (zeros advantage) when no advantage data is present,
        # which can happen during BC pre-training with use_advantage_embedding=True.
        advantage_token = None
        advantage_mask = None
        if self.config.use_advantage_embedding:
            adv_rng, rng = jax.random.split(rng)
            adv_values = (
                observation.advantage
                if observation.advantage is not None
                else jnp.zeros(batch_size, dtype=jnp.float32)
            )
            advantage_token, advantage_mask = self._compute_advantage_token(adv_values, adv_rng, inference=False)

        # 1. Embed prefix once (includes FAST tokens + optional WM-FAST query
        # when FAST tokens are provided in the observation).
        prefix_tokens, prefix_mask, prefix_ar_mask, num_image_tokens, adv_token_idx, wm_fast_token_idx, state_token_range = self.embed_prefix(
            observation, advantage_token, advantage_mask
        )

        # 2. Compute prefix KV cache
        # ar_mask cumsum trick gives correct group visibility:
        #   group 0 (images+success) ↔ group 0 only
        #   group 1 (state) → sees group 0+self
        #   group 2 (advantage) → sees groups 0-1+self
        #   group 3+ (FAST) → causal, sees everything
        #   group 3+fast_len (WM-FAST) → sees everything prior, invisible to it
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # Per-sample state dropout: zero attention rows+columns for state tokens
        if train and state_token_range is not None and self.config.train_aug is not None and self.config.train_aug.state_dropout_prob > 0:
            sdrop_rng, rng = jax.random.split(rng)
            state_drop = jax.random.bernoulli(sdrop_rng, self.config.train_aug.state_dropout_prob, (batch_size,))
            s_start, s_end = state_token_range
            keep = jnp.logical_not(state_drop)  # [B]
            prefix_attn_mask = prefix_attn_mask.at[:, :, s_start:s_end].set(
                prefix_attn_mask[:, :, s_start:s_end] & keep[:, None, None]
            )
            # Zero rows: state tokens cannot attend to anything
            prefix_attn_mask = prefix_attn_mask.at[:, s_start:s_end, :].set(
                prefix_attn_mask[:, s_start:s_end, :] & keep[:, None, None]
            )

        # Per-sample advantage masking for neutral (masked) samples
        if adv_token_idx is not None and advantage_mask is not None:
            prefix_attn_mask = prefix_attn_mask.at[:, :, adv_token_idx].set(
                prefix_attn_mask[:, :, adv_token_idx] & advantage_mask[:, None]
            )
            prefix_attn_mask = prefix_attn_mask.at[:, adv_token_idx, :].set(
                prefix_attn_mask[:, adv_token_idx, :] & advantage_mask[:, None]
            )

        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache_full = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions_prefix
        )
        
        # 3. Success head: extract P(success) prediction and compute BCE loss
        # Note: success_loss is returned separately (not added to total_loss here).
        # train.py adds it with IS-weighting + type-balancing in the train_step.
        if self.config.use_success_head and num_image_tokens is not None:
            success_idx = num_image_tokens  # success token position
            success_hidden = prefix_out[:, success_idx, :]  # [B, D]
            sp_logit = self.success_head(success_hidden).squeeze(-1)  # [B] (bf16 logit)
            sp_pred = jax.nn.sigmoid(sp_logit)  # [B]

            # Expose raw per-sample sigmoid output so train.py can compute
            # loss-matching weighted mean metrics. Underscore prefix keeps it
            # out of the AUX_KEYS auto-mean loop.
            losses["_success_pred_per_sample"] = sp_pred  # [B]

            if observation.success_target is not None:
                success_target = observation.success_target  # [B]
                # Mask out NaN targets (BC samples with no value supervision)
                valid = jnp.isfinite(success_target)  # [B]
                safe_target = jnp.where(valid, success_target, 0.0)
                # Stable BCE from logits, fp32 upcast (avoids bf16 log-cliff
                # jitter at saturated predictions).
                bce = _bce_with_logits(sp_logit, safe_target)
                per_sample = jnp.where(valid, bce, 0.0)  # [B]
                losses["success_loss"] = per_sample  # [B]
                n_valid = jnp.maximum(valid.sum(), 1.0)
                valid_pred = jnp.where(valid, sp_pred, 0.0)
                pred_mean = valid_pred.sum() / n_valid
                losses["success_pred_mean"] = pred_mean * jnp.ones(batch_size)
                # Compute std only over valid samples (Bessel correction for unbiased estimate)
                pred_var = jnp.where(valid, (sp_pred - pred_mean) ** 2, 0.0).sum() / jnp.maximum(n_valid - 1, 1.0)
                losses["success_pred_std"] = jnp.where(n_valid > 1, jnp.sqrt(pred_var), 0.0) * jnp.ones(batch_size)
                losses["success_target_mean"] = safe_target.sum() / n_valid * jnp.ones(batch_size)
                losses["success_mae"] = jnp.where(valid, jnp.abs(sp_pred - safe_target), 0.0)  # [B]
                losses["success_valid_frac"] = (valid.sum() / batch_size) * jnp.ones(batch_size)

            # 3c. Auxiliary heads: checkpoint, garment_type, completion
            # All use the same success_hidden representation from the query token.
            # Forward pass always runs (for consistent JIT trace); losses zeroed when targets absent.
            cp_logit = self.checkpoint_head(success_hidden).squeeze(-1)  # [B] (bf16 logit)
            cp_pred = jax.nn.sigmoid(cp_logit)  # [B]
            gt_logits = self.garment_type_head(success_hidden)  # [B, 4]
            comp_pred = jax.nn.sigmoid(self.completion_head(success_hidden).squeeze(-1))  # [B]
            losses["_checkpoint_pred_per_sample"] = cp_pred  # [B]
            losses["_completion_pred_per_sample"] = comp_pred  # [B]

            # Checkpoint loss: P(reward >= 0.5) — stable BCE from logits, fp32.
            if observation.checkpoint_target is not None:
                cp_target = observation.checkpoint_target  # [B]
                cp_valid = jnp.isfinite(cp_target)
                cp_safe = jnp.where(cp_valid, cp_target, 0.0)
                cp_bce = _bce_with_logits(cp_logit, cp_safe)
                losses["checkpoint_loss"] = jnp.where(cp_valid, cp_bce, 0.0)  # [B]
            else:
                losses["checkpoint_loss"] = jnp.zeros(batch_size)  # [B]
            losses["checkpoint_pred_mean"] = jnp.mean(cp_pred) * jnp.ones(batch_size)

            # Garment type loss: 4-class cross-entropy
            if observation.garment_type_id is not None:
                gt_target = observation.garment_type_id  # [B] int
                gt_valid = gt_target >= 0
                gt_safe = jnp.where(gt_valid, gt_target, 0)
                gt_log_probs = jax.nn.log_softmax(gt_logits, axis=-1)  # [B, 4]
                gt_ce = -gt_log_probs[jnp.arange(batch_size), gt_safe]  # [B]
                losses["garment_type_loss"] = jnp.where(gt_valid, gt_ce, 0.0)  # [B]
                gt_pred_class = jnp.argmax(gt_logits, axis=-1)
                gt_n_valid = jnp.maximum(gt_valid.sum(), 1.0)
                # Accuracy averaged over valid samples only (not diluted by BC zeros)
                losses["garment_type_accuracy"] = jnp.where(gt_valid, (gt_pred_class == gt_safe).astype(jnp.float32), 0.0).sum() / gt_n_valid * jnp.ones(batch_size)
            else:
                losses["garment_type_loss"] = jnp.zeros(batch_size)
                losses["garment_type_accuracy"] = jnp.zeros(batch_size)

            # Completion loss: step/total_steps, success-only — MSE
            if observation.completion_target is not None:
                comp_target = observation.completion_target  # [B]
                comp_valid = jnp.isfinite(comp_target)
                comp_safe = jnp.where(comp_valid, comp_target, 0.0)
                comp_mse = (comp_pred - comp_safe) ** 2
                losses["completion_loss"] = jnp.where(comp_valid, comp_mse, 0.0)  # [B]
            else:
                losses["completion_loss"] = jnp.zeros(batch_size)
            losses["completion_pred_mean"] = jnp.mean(comp_pred) * jnp.ones(batch_size)

            # TTC (time-to-completion) loss: 1 - steps_left/600 (success) or 0 (failure)
            # Output via sigmoid → range [0, 1]
            ttc_raw = self.ttc_head(success_hidden).squeeze(-1)  # [B]
            ttc_pred = jax.nn.sigmoid(ttc_raw)  # [B], range [0, 1]
            losses["_ttc_pred_per_sample"] = ttc_pred  # [B]
            if observation.ttc_target is not None:
                ttc_target = observation.ttc_target  # [B]
                ttc_valid = jnp.isfinite(ttc_target)
                ttc_safe = jnp.where(ttc_valid, ttc_target, 0.0)
                ttc_mse = (ttc_pred - ttc_safe) ** 2
                losses["ttc_loss"] = jnp.where(ttc_valid, ttc_mse, 0.0)  # [B]
                losses["ttc_valid_frac"] = (ttc_valid.sum() / batch_size) * jnp.ones(batch_size)
                n_valid = jnp.maximum(ttc_valid.sum(), 1.0)
                valid_target = jnp.where(ttc_valid, ttc_safe, 0.0)
                losses["ttc_target_mean"] = (valid_target.sum() / n_valid) * jnp.ones(batch_size)
            else:
                losses["ttc_loss"] = jnp.zeros(batch_size)
                losses["ttc_valid_frac"] = jnp.zeros(batch_size)
                losses["ttc_target_mean"] = jnp.zeros(batch_size)
            losses["ttc_pred_mean"] = jnp.mean(ttc_pred) * jnp.ones(batch_size)

            # Keypoint-distance head (Head 1): per-condition distance regression.
            # Raw linear output — distances > 1.0 are informative ("still far above
            # threshold") and should not be clipped.
            kpt_pred = self.keypoint_distance_head(success_hidden)  # [B, 21]
            losses["_keypoint_distance_pred_per_sample"] = kpt_pred  # [B, 21]
            kpt_target = observation.keypoint_distance_target  # [B, 21] or None
            kpt_mask = observation.keypoint_distance_mask       # [B, 21] bool or None
            if kpt_target is not None and kpt_mask is not None:
                kpt_mask_f = kpt_mask.astype(kpt_pred.dtype)
                safe_target = jnp.where(kpt_mask, kpt_target, 0.0)
                slot_sq = (kpt_pred - safe_target) ** 2 * kpt_mask_f  # [B, 21]
                # Per-sample mean over valid slots (avoid div by zero).
                n_valid_per_sample = jnp.maximum(kpt_mask_f.sum(axis=-1), 1.0)
                losses["keypoint_distance_loss"] = slot_sq.sum(axis=-1) / n_valid_per_sample  # [B]
                # MAE metric is also per-sample (mean over valid slots); train.py
                # aggregates to the valid-sample mean for logging.
                losses["keypoint_distance_mae_per_sample"] = (
                    jnp.abs(kpt_pred - safe_target) * kpt_mask_f
                ).sum(axis=-1) / n_valid_per_sample  # [B]
            else:
                losses["keypoint_distance_loss"] = jnp.zeros(batch_size)
                losses["keypoint_distance_mae_per_sample"] = jnp.zeros(batch_size)

        # 4. Extract FAST loss from prefix output (before removing from cache).
        # Account for an optional WM-FAST token appended after FAST: when present,
        # FAST outputs occupy [prefix_len - fast_len - 1 : prefix_len - 1].
        fast_loss_value = 0.0
        fast_len = 0
        wm_fast_tail = 1 if wm_fast_token_idx is not None else 0
        fast_targets = observation.fast_tokens
        fast_token_mask = observation.fast_token_mask

        if self.config.use_fast_auxiliary and fast_targets is not None:
            fast_len = fast_targets.shape[1]
            fast_start_idx = prefix_tokens.shape[1] - fast_len - wm_fast_tail
            fast_outputs = prefix_out[:, fast_start_idx:fast_start_idx + fast_len, :]  # [B, T, D]
            
            # Project to FAST vocab
            fast_logits = self.fast_token_proj(fast_outputs)  # [B, T, vocab_size]
            
            # Cross-entropy loss with teacher forcing
            pred_logits = fast_logits  # [B, T, vocab]
            target_tokens = fast_targets  # [B, T]
            loss_mask = fast_token_mask  # [B, T]
            
            log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
            target_log_probs = jnp.take_along_axis(
                log_probs,
                target_tokens[:, :, None],
                axis=-1
            ).squeeze(-1)  # [B, T]
            
            fast_token_loss = -target_log_probs  # [B, T]
            
            # Apply mask and normalize by number of valid tokens
            masked_loss = fast_token_loss * loss_mask  # [B, T]
            num_valid_tokens = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1)  # [B]
            losses["fast_loss"] = jnp.sum(masked_loss, axis=-1) / num_valid_tokens  # [B]
            
            # Accuracy (only on valid tokens)
            pred_tokens = jnp.argmax(pred_logits, axis=-1)
            correct = (pred_tokens == target_tokens) * loss_mask
            losses["fast_accuracy"] = jnp.sum(correct, axis=-1) / num_valid_tokens
            
            fast_loss_value = self.config.fast_loss_weight * jnp.mean(losses["fast_loss"])
        elif fast_targets is not None:
            # FAST auxiliary is disabled but data contains FAST tokens
            raise ValueError(
                "use_fast_auxiliary=False but observation contains fast_tokens. "
                "Either enable use_fast_auxiliary in config or ensure data doesn't contain fast_tokens."
            )

        # 4b. WM-FAST head (Head 2): predicts future success / completion / keypoint
        # distances at t + WM_FUTURE_HORIZON from the token placed after FAST.
        # Training-only: wm_fast_token_idx is None at inference (FAST absent).
        if wm_fast_token_idx is not None and self.config.use_wm_fast_head:
            wm_fast_hidden = prefix_out[:, wm_fast_token_idx, :]  # [B, D]
            wm_fast_s_logit = self.wm_fast_success_head(wm_fast_hidden).squeeze(-1)  # [B] (bf16)
            wm_fast_s_pred = jax.nn.sigmoid(wm_fast_s_logit)  # [B]
            wm_fast_c_pred = jax.nn.sigmoid(self.wm_fast_completion_head(wm_fast_hidden).squeeze(-1))  # [B]
            wm_fast_k_pred = self.wm_fast_keypoint_head(wm_fast_hidden)  # [B, 21]
            losses["_wm_fast_success_pred_per_sample"] = wm_fast_s_pred
            losses["_wm_fast_completion_pred_per_sample"] = wm_fast_c_pred
            losses["_wm_fast_keypoint_pred_per_sample"] = wm_fast_k_pred

            # Future success (stable BCE from logits, NaN-masked)
            s_target = observation.success_future_target
            if s_target is not None:
                s_valid = jnp.isfinite(s_target)
                s_safe = jnp.where(s_valid, s_target, 0.0)
                s_bce = _bce_with_logits(wm_fast_s_logit, s_safe)
                losses["wm_fast_success_loss"] = jnp.where(s_valid, s_bce, 0.0)
            else:
                losses["wm_fast_success_loss"] = jnp.zeros(batch_size)

            # Future completion (MSE, NaN-masked)
            c_target = observation.completion_future_target
            if c_target is not None:
                c_valid = jnp.isfinite(c_target)
                c_safe = jnp.where(c_valid, c_target, 0.0)
                c_mse = (wm_fast_c_pred - c_safe) ** 2
                losses["wm_fast_completion_loss"] = jnp.where(c_valid, c_mse, 0.0)
            else:
                losses["wm_fast_completion_loss"] = jnp.zeros(batch_size)

            # Future keypoint distances (masked MSE)
            k_target = observation.keypoint_distance_future_target
            k_mask = observation.keypoint_distance_future_mask
            if k_target is not None and k_mask is not None:
                k_mask_f = k_mask.astype(wm_fast_k_pred.dtype)
                k_safe = jnp.where(k_mask, k_target, 0.0)
                k_sq = (wm_fast_k_pred - k_safe) ** 2 * k_mask_f
                k_denom = jnp.maximum(k_mask_f.sum(axis=-1), 1.0)
                losses["wm_fast_keypoint_loss"] = k_sq.sum(axis=-1) / k_denom
            else:
                losses["wm_fast_keypoint_loss"] = jnp.zeros(batch_size)
        else:
            losses["wm_fast_success_loss"] = jnp.zeros(batch_size)
            losses["wm_fast_completion_loss"] = jnp.zeros(batch_size)
            losses["wm_fast_keypoint_loss"] = jnp.zeros(batch_size)

        # 5. Remove FAST + WM-FAST tokens from KV cache (action expert doesn't
        # attend to either). Both sit at the end of the prefix.
        # KV cache shape: [layers, batch, seq_len, num_kv_heads, head_dim]
        tail_drop = fast_len + wm_fast_tail  # total tokens to remove from cache
        if tail_drop > 0:
            cache_k, cache_v = kv_cache_full
            cache_k = cache_k[:, :, :-tail_drop, :, :]
            cache_v = cache_v[:, :, :-tail_drop, :, :]
            kv_cache_for_actions = (cache_k, cache_v)
            prefix_len_for_actions = prefix_tokens.shape[1] - tail_drop
            # Truncate prefix mask and ar_mask for action expert
            prefix_mask_for_actions = prefix_mask[:, :-tail_drop]
            prefix_ar_mask_for_actions = prefix_ar_mask[:-tail_drop]
        else:
            kv_cache_for_actions = kv_cache_full
            prefix_len_for_actions = prefix_tokens.shape[1]
            prefix_mask_for_actions = prefix_mask
            prefix_ar_mask_for_actions = prefix_ar_mask
        
        # 5b. Mask advantage token in action expert's prefix mask (per-sample)
        if adv_token_idx is not None and advantage_mask is not None:
            # adv_token_idx is relative to full prefix; adjust for FAST removal
            adv_idx_for_actions = adv_token_idx  # advantage is before FAST, so no adjustment needed
            prefix_mask_for_actions = prefix_mask_for_actions.at[:, adv_idx_for_actions].set(
                prefix_mask_for_actions[:, adv_idx_for_actions] & advantage_mask
            )

        # 6. Knowledge insulation: stop gradients from action expert to VLM
        # This must happen BEFORE kv_transform so transform still receives gradients
        if self.config.use_knowledge_insulation:
            kv_cache_for_actions = jax.tree.map(jax.lax.stop_gradient, kv_cache_for_actions)
        
        # 7. Transform KV cache (after stop_gradient, so it receives action expert gradients)
        if self.kv_transform is not None:
            kv_cache_for_actions = self.kv_transform(kv_cache_for_actions)
        
        # 8. Define single flow sample processing.
        # When use_wm_flow_head is True, the suffix has action_horizon + 1 tokens:
        # [a_0, a_1, ..., a_{H-1}, wm_flow_query]. Action extraction takes the
        # FIRST action_horizon positions; WM extraction reads the tail position.
        use_wm_flow = self.config.use_wm_flow_head and getattr(self, "wm_flow_query_token", None) is not None
        def process_one_flow_sample(sample_rng):
            """Process one flow sample using the original cached prefix.

            Returns (action_loss [B,H,D], wm_s_loss [B], wm_c_loss [B], wm_k_loss [B])
            — WM losses are zero-filled when use_wm_flow_head is False.
            """
            noise_rng, time_rng = jax.random.split(sample_rng)

            # Generate different noise and time for this sample
            noise = self.generate_correlated_noise(noise_rng, batch_size)
            time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001

            # Compute noisy actions and target velocity
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            # Embed suffix for this sample.
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time, advantage_mask=advantage_mask
            )

            # Build attention mask: suffix attends to prefix (without FAST/WM-FAST) + itself
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(
                prefix_mask_for_actions, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            suffix_positions = prefix_len_for_actions + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache_for_actions,
                adarms_cond=[None, adarms_cond]
            )

            # Compute velocity and loss from the FIRST action_horizon tokens
            # (never the WM-flow token, which sits at the tail).
            v_t = self.action_out_proj(suffix_out[:, :self.action_horizon])
            action_loss_i = jnp.square(v_t - u_t)  # [B, H, D]

            # WM-flow head: per-sample loss weight = (1 - t), and samples with
            # t > 0.5 are EXCLUDED entirely (pure-noise inputs carry no signal
            # for the future-prediction heads). The t-mask is propagated via
            # ``t_valid_i`` so train.py's valid-only metric denominator only
            # counts the samples that actually contributed to the gradient.
            # t in [0.001, 1.0]: t=1 -> pure noise, t=0 -> clean.
            _WM_FLOW_T_MAX = 0.5
            t_valid_i = time <= _WM_FLOW_T_MAX  # [B] bool
            if use_wm_flow:
                wm_flow_hidden = suffix_out[:, -1, :]  # [B, D_action]
                # wm_flow_success now predicts Δsuccess = true_success − V̂(s_t)
                # (action-conditional residual over the V-head baseline). Raw
                # linear output in ~[-1, 1] (no sigmoid). Completion head keeps
                # sigmoid since its target is a fraction in [0, 1].
                wm_s_pred = self.wm_flow_success_head(wm_flow_hidden).squeeze(-1)  # [B] raw Δ
                wm_c_pred = jax.nn.sigmoid(self.wm_flow_completion_head(wm_flow_hidden).squeeze(-1))  # [B]
                wm_k_pred = self.wm_flow_keypoint_head(wm_flow_hidden)  # [B, 21]
                # weight: (1 - t) gated to zero for t > 0.5.
                w = jnp.where(t_valid_i, 1.0 - time, 0.0).astype(wm_s_pred.dtype)  # [B]

                # Δsuccess target: true episode outcome minus the V-head's own
                # prediction for s_t, stop-gradient on V so the Q head doesn't
                # drag the V head. Requires the success head to be enabled,
                # which is already enforced by the wm_flow head itself
                # (wm_flow_head only exists when use_success_head=True).
                s_target = observation.success_future_target
                if s_target is not None:
                    sp_baseline = jax.lax.stop_gradient(sp_pred).astype(jnp.float32)
                    delta_target = s_target.astype(jnp.float32) - sp_baseline  # [B], ~[-1, 1]
                    s_valid = jnp.isfinite(s_target) & jnp.isfinite(sp_baseline) & t_valid_i
                    s_safe = jnp.where(s_valid, delta_target, 0.0)
                    s_mse = (wm_s_pred.astype(jnp.float32) - s_safe) ** 2
                    wm_s_loss_i = jnp.where(s_valid, s_mse, 0.0) * w
                else:
                    wm_s_loss_i = jnp.zeros(batch_size)

                c_target = observation.completion_future_target
                if c_target is not None:
                    c_valid = jnp.isfinite(c_target) & t_valid_i
                    c_safe = jnp.where(c_valid, c_target, 0.0)
                    c_mse = (wm_c_pred - c_safe) ** 2
                    wm_c_loss_i = jnp.where(c_valid, c_mse, 0.0) * w
                else:
                    wm_c_loss_i = jnp.zeros(batch_size)

                k_target = observation.keypoint_distance_future_target
                k_mask = observation.keypoint_distance_future_mask
                if k_target is not None and k_mask is not None:
                    k_mask_f = (k_mask & t_valid_i[:, None]).astype(wm_k_pred.dtype)
                    k_safe = jnp.where(k_mask, k_target, 0.0)
                    k_sq = (wm_k_pred - k_safe) ** 2 * k_mask_f
                    k_denom = jnp.maximum(k_mask_f.sum(axis=-1), 1.0)
                    wm_k_loss_i = (k_sq.sum(axis=-1) / k_denom) * w
                else:
                    wm_k_loss_i = jnp.zeros(batch_size)
            else:
                wm_s_loss_i = jnp.zeros(batch_size)
                wm_c_loss_i = jnp.zeros(batch_size)
                wm_k_loss_i = jnp.zeros(batch_size)
                wm_s_pred = jnp.zeros(batch_size)

            # Always return wm_s_pred (fp32) so the outer aggregation can
            # compute prediction mean / std across valid flow samples. When
            # ``use_wm_flow`` is False the placeholder zeros are never read
            # (t_valid_i mask gates the aggregation).
            return action_loss_i, wm_s_loss_i, wm_c_loss_i, wm_k_loss_i, t_valid_i, wm_s_pred.astype(jnp.float32)

        # 9. Vectorize over N flow samples
        flow_rngs = jax.random.split(rng, num_flow_samples)
        with at.disable_typechecking():
            (
                all_action_losses, all_wm_s, all_wm_c, all_wm_k, all_t_valid,
                all_wm_s_pred,
            ) = jax.vmap(process_one_flow_sample)(flow_rngs)

        # 10. Average over flow samples
        action_loss = jnp.mean(all_action_losses, axis=0)  # [B, H, D]
        # Mask out PADDED timesteps (action horizon overshoots the episode
        # and LeRobot repeats the last real action). Padded targets don't
        # reflect actual future actions; training against them would teach
        # the model to predict the "stand still" frame at episode end as if
        # it were signal. When ``action_is_pad`` is absent (older datasets,
        # inference-time compute), default to no padding.
        action_pad_mask = observation.action_is_pad  # [B, H] bool or None
        if action_pad_mask is not None:
            valid = (~action_pad_mask).astype(action_loss.dtype)  # [B, H]
            masked_loss = action_loss * valid[..., None]  # [B, H, D]
            # Per-sample mean over (valid H) × D. Avoid div-by-zero for the
            # edge case where all timesteps are padded (shouldn't happen in
            # practice but be safe).
            valid_count = valid.sum(axis=-1)  # [B]
            per_h_mean_D = masked_loss.mean(axis=-1)  # [B, H] — mean over D is safe (D constant)
            action_loss_per_sample = per_h_mean_D.sum(axis=-1) / jnp.maximum(valid_count, 1.0)
            losses["action_loss"] = action_loss_per_sample
            # Diagnostic: what fraction of action timesteps in this batch are
            # padded? Expect low (~action_horizon / mean_ep_len), but grows
            # with short episodes and near-boundary frames.
            losses["action_pad_frac"] = (
                action_pad_mask.astype(jnp.float32).mean() * jnp.ones(batch_size)
            )
        else:
            losses["action_loss"] = jnp.mean(action_loss, axis=(-2, -1))

        # WM-flow: each flow sample with t > 0.5 contributes 0, so plain
        # ``mean / N`` dilutes the per-sample loss by the fraction of invalid
        # flow samples. Instead: normalize by the count of VALID flow samples
        # per batch element, so the gradient magnitude reflects real signal.
        # When no flow sample is valid for a given batch element, the loss is
        # 0 AND the sample is marked invalid via ``_wm_flow_t_valid_per_sample``.
        all_t_valid_f = all_t_valid.astype(jnp.float32)
        valid_count_per_sample = all_t_valid_f.sum(axis=0)  # [B]
        has_any_valid = valid_count_per_sample > 0              # [B] bool
        valid_count_safe = jnp.maximum(valid_count_per_sample, 1.0)
        losses["wm_flow_success_loss"] = jnp.where(
            has_any_valid, jnp.sum(all_wm_s, axis=0) / valid_count_safe, 0.0
        )
        losses["wm_flow_completion_loss"] = jnp.where(
            has_any_valid, jnp.sum(all_wm_c, axis=0) / valid_count_safe, 0.0
        )
        losses["wm_flow_keypoint_loss"] = jnp.where(
            has_any_valid, jnp.sum(all_wm_k, axis=0) / valid_count_safe, 0.0
        )
        # Per-batch-sample valid flag (at least one flow sample was low-noise);
        # used by train.py as a validity predicate when forming the weighted-
        # mean metric denominator.
        losses["_wm_flow_t_valid_per_sample"] = has_any_valid  # [B] bool
        # True per-flow-sample fraction (over ALL N × B flow samples). This is
        # what you'd expect ~P(t ≤ 0.5) ≈ 0.35 for Beta(1.5, 1).
        _wm_flow_t_valid_frac_scalar = all_t_valid_f.mean()
        losses["_wm_flow_t_valid_frac_scalar"] = _wm_flow_t_valid_frac_scalar * jnp.ones(batch_size)

        # ── wm_flow_success prediction + target diagnostics ─────────────
        # The success head now outputs Δ = true_success − V̂(s_t). Logging
        # pred mean/std lets us check the head's operating range at a
        # glance (expect ~[-1, 1], mean near 0 once trained). Target mean
        # confirms the label distribution entering the loss.
        # Masked over low-noise flow samples (all_t_valid_f) and across the
        # full batch — one scalar broadcast to [B].
        total_t_valid_scalar = all_t_valid_f.sum()
        wm_s_pred_sum = (all_wm_s_pred * all_t_valid_f).sum()
        wm_s_pred_mean_scalar = jnp.where(
            total_t_valid_scalar > 0,
            wm_s_pred_sum / jnp.maximum(total_t_valid_scalar, 1.0),
            0.0,
        )
        losses["wm_flow_success_pred_mean"] = wm_s_pred_mean_scalar * jnp.ones(batch_size)
        wm_s_pred_var_sum = ((all_wm_s_pred - wm_s_pred_mean_scalar) ** 2 * all_t_valid_f).sum()
        wm_s_pred_var = jnp.where(
            total_t_valid_scalar > 1,
            wm_s_pred_var_sum / jnp.maximum(total_t_valid_scalar - 1, 1.0),
            0.0,
        )
        losses["wm_flow_success_pred_std"] = jnp.where(
            total_t_valid_scalar > 1, jnp.sqrt(wm_s_pred_var), 0.0
        ) * jnp.ones(batch_size)

        # Target Δ mean: success_future_target − sg(V̂). Same across flow
        # samples (no noise dependence), so compute once over batch. Use the
        # ``has_any_valid`` mask so only samples that contributed to the
        # gradient count.
        s_fut_target = observation.success_future_target
        if s_fut_target is not None and self.config.use_success_head:
            s_fut_f = s_fut_target.astype(jnp.float32)
            sp_base = jax.lax.stop_gradient(sp_pred).astype(jnp.float32)
            delta_tgt = s_fut_f - sp_base  # [B]
            tgt_valid = jnp.isfinite(s_fut_f) & jnp.isfinite(sp_base) & has_any_valid
            tgt_valid_f = tgt_valid.astype(jnp.float32)
            n_tgt_valid = tgt_valid_f.sum()
            tgt_sum = (delta_tgt * tgt_valid_f).sum()
            tgt_mean = jnp.where(n_tgt_valid > 0, tgt_sum / jnp.maximum(n_tgt_valid, 1.0), 0.0)
            losses["wm_flow_success_target_mean"] = tgt_mean * jnp.ones(batch_size)
        else:
            losses["wm_flow_success_target_mean"] = jnp.zeros(batch_size)

        # 12. Total loss (value loss excluded — added uniformly in train_step)
        losses["total_loss"] = losses["action_loss"] + fast_loss_value
        
        return losses

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        # Mask-based inpainting (supports batching with different inpaint lengths per element):
        inpaint_targets: at.Float[at.Array, "b ah ad"] | None = None,  # [B, 30, 12] padded target values
        inpaint_mask: at.Float[at.Array, "b flat"] | None = None,      # [B, 360] 1=observed
        inpaint_lengths: at.Int[at.Array, " b"] | None = None,         # [B] num initial actions per element
        time_threshold_inpaint: float | at.Float[at.Array, "..."] | None = None,  # scalar or [B]
        noise_temperature: float | at.Float[at.Array, "..."] = 1.0,  # scalar or [B]
        cfg_scale: float | at.Float[at.Array, "..."] | None = None,  # Classifier-Free Guidance scale
        explore_noise_scale: float | at.Float[at.Array, "..."] = 0.0,  # DART-style additive correlated noise
        num_candidates: int = 1,  # best-of-N: prefix VLM runs once on B unique requests
                                  # then kv_cache + observation are tiled to B*N for the
                                  # flow loop. Each candidate gets independent noise so
                                  # trajectories diverge. Must be a Python int (static
                                  # under JIT — see static_argnames in the policy).
        cfg_disabled: bool = False,  # Force-skip the unconditional CFG pass. When True
                                     # the model only runs the conditional (advantage>0)
                                     # branch — ~2× faster denoising, but loses CFG
                                     # guidance. Used by the real-robot server for
                                     # realtime inference (cfg_scale is implicitly 1.0).
                                     # Static under JIT.
    ) -> _model.Actions:
        observation = preprocess_observation(
            None, observation, train=False,
            rotate_top_image=self.config.rotate_top_image,
            train_aug=self.config.train_aug,
        )
        # t=1 is noise, t=0 is target (opposite of pi0 paper convention)
        dt = -1.0 / num_steps
        B_unique = observation.state.shape[0]  # rows of unique requests
        flat_dim = self.action_horizon * self.action_dim  # 360

        has_inpainting = inpaint_mask is not None

        # Ensure FAST tokens are never used during inference
        if observation.fast_tokens is not None:
            raise ValueError(
                "FAST tokens must not be provided during inference (sample_actions). "
                "FAST tokens are only used during training for auxiliary loss. "
                "Set observation.fast_tokens=None before calling sample_actions."
            )

        # ===================================================================
        # Phase 1: prefix VLM at B_unique (computed ONCE per unique request).
        # ===================================================================
        # Advantage token for inference: always positive (index 0), mask=True.
        adv_token_inference = None
        adv_mask_inference = None
        if self.config.use_advantage_embedding:
            adv_token_inference, adv_mask_inference = self._compute_advantage_token(
                jnp.zeros(B_unique, dtype=jnp.float32), rng=None, inference=True
            )

        # Fill KV cache with a forward pass of the prefix (no FAST tokens during inference,
        # so no WM-FAST token either — wm_fast_token_idx is always None here).
        prefix_tokens, prefix_mask, prefix_ar_mask, num_image_tokens, adv_token_idx, _, _ = self.embed_prefix(
            observation, adv_token_inference, adv_mask_inference
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # Per-sample advantage masking (during inference mask is always True, kept for consistency)
        if adv_token_idx is not None and adv_mask_inference is not None:
            prefix_attn_mask = prefix_attn_mask.at[:, :, adv_token_idx].set(
                prefix_attn_mask[:, :, adv_token_idx] & adv_mask_inference[:, None]
            )
            prefix_attn_mask = prefix_attn_mask.at[:, adv_token_idx, :].set(
                prefix_attn_mask[:, adv_token_idx, :] & adv_mask_inference[:, None]
            )

        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        # Prefix-derived prediction heads (still at B_unique).
        success_pred = None
        checkpoint_pred = None
        garment_type_pred = None
        completion_pred = None
        ttc_pred = None
        keypoint_distances_pred = None
        if self.config.use_success_head and num_image_tokens is not None:
            success_idx = num_image_tokens
            success_hidden = prefix_out[:, success_idx, :]  # [B_unique, D]
            success_pred = jax.nn.sigmoid(self.success_head(success_hidden).squeeze(-1))
            checkpoint_pred = jax.nn.sigmoid(self.checkpoint_head(success_hidden).squeeze(-1))
            garment_type_pred = jnp.argmax(self.garment_type_head(success_hidden), axis=-1)
            completion_pred = jax.nn.sigmoid(self.completion_head(success_hidden).squeeze(-1))
            ttc_pred = jax.nn.sigmoid(self.ttc_head(success_hidden).squeeze(-1))
            if getattr(self, "keypoint_distance_head", None) is not None:
                keypoint_distances_pred = self.keypoint_distance_head(success_hidden)  # [B_unique, 21]

        # Transform KV cache for cross-layer attention
        if self.kv_transform is not None:
            kv_cache = self.kv_transform(kv_cache)

        # ===================================================================
        # Phase 2: tile prefix outputs to B_unique * num_candidates.
        # ===================================================================
        # The prefix is bit-identical for all N candidates of the same
        # request, so we tile the kv_cache + masks + observation rather than
        # rerun the VLM N times. Divergence comes from independent noise
        # drawn at the (B*N) batch dim below. Guarded so num_candidates=1
        # is a no-op (bit-identical to the pre-best-of-N path).
        if num_candidates > 1:
            def _rep(x):
                if x is None:
                    return x
                return jnp.repeat(x, num_candidates, axis=0)

            observation = jax.tree.map(
                lambda x: jnp.repeat(x, num_candidates, axis=0)
                    if (hasattr(x, "shape") and getattr(x, "ndim", 0) > 0
                        and x.shape[0] == B_unique)
                    else x,
                observation,
            )
            # KV cache layout is [layers, batch, t, kv_heads, head_dim] — the
            # batch axis is 1, NOT 0. Repeating along axis 0 would replicate
            # layers and the gemma typecheck rejects the resulting (l*N, B,
            # t, ...) shape (caught by jaxtyping).
            kv_cache = jax.tree.map(
                lambda x: jnp.repeat(x, num_candidates, axis=1), kv_cache,
            )
            prefix_mask = _rep(prefix_mask)
            if adv_mask_inference is not None:
                adv_mask_inference = _rep(adv_mask_inference)
            success_pred = _rep(success_pred)
            checkpoint_pred = _rep(checkpoint_pred)
            garment_type_pred = _rep(garment_type_pred)
            completion_pred = _rep(completion_pred)
            ttc_pred = _rep(ttc_pred)
            keypoint_distances_pred = _rep(keypoint_distances_pred)

            def _tile_per_sample(p):
                if p is None:
                    return None
                arr = jnp.asarray(p)
                if arr.ndim == 0:
                    return arr  # scalar — broadcast later, no-op
                if arr.shape[0] == B_unique:
                    return jnp.repeat(arr, num_candidates, axis=0)
                return arr  # already at the post-tile shape (or unexpected)
            noise_temperature = _tile_per_sample(noise_temperature)
            time_threshold_inpaint = _tile_per_sample(time_threshold_inpaint)
            cfg_scale = _tile_per_sample(cfg_scale)
            explore_noise_scale = _tile_per_sample(explore_noise_scale)
            if has_inpainting:
                inpaint_targets = _rep(inpaint_targets)
                inpaint_mask = _rep(inpaint_mask)
                inpaint_lengths = _rep(inpaint_lengths)
            if noise is not None and noise.shape[0] == B_unique:
                # Caller-supplied noise is replicated across candidates — they
                # will denoise identically, defeating divergence. Production
                # paths pass noise=None so we draw fresh below.
                noise = _rep(noise)

        batch_size = B_unique * num_candidates  # post-tile leading axis

        # ===================================================================
        # Phase 3: noise + inpaint preprocessing at batch_size (= B*N).
        # ===================================================================
        if noise is None:
            rng, noise_rng = jax.random.split(rng)
            if _correction_matrices_cache is not None:
                noise = self.generate_correlated_noise(noise_rng, batch_size)
            else:
                noise = jax.random.normal(
                    noise_rng, (batch_size, self.action_horizon, self.action_dim)
                )

        # Scale noise — support per-element [B*N] or scalar temperature.
        noise_temp = jnp.asarray(noise_temperature)
        if noise_temp.ndim == 0:
            noise = noise * jnp.sqrt(noise_temp)
        else:
            noise = noise * jnp.sqrt(noise_temp[:, None, None])

        if has_inpainting:
            noise_flat = noise.reshape(batch_size, flat_dim)
            targets_flat = inpaint_targets.reshape(batch_size, flat_dim)
            fixed_z = noise_flat * inpaint_mask
            x0 = targets_flat * inpaint_mask
            if _correction_matrices_cache is not None:
                C_batch = _correction_matrices_cache[inpaint_lengths]
            else:
                C_batch = jnp.zeros((batch_size, flat_dim, flat_dim))
        else:
            fixed_z = None
            x0 = None
            C_batch = None

        rng, step_rng = jax.random.split(rng)

        # Classifier-Free Guidance: build unconditional prefix mask
        # (hides advantage token from suffix cross-attention)
        cfg_scale = jnp.asarray(
            cfg_scale if cfg_scale is not None else self.config.cfg_scale
        )
        use_cfg = (
            self.config.use_advantage_embedding
            and adv_token_idx is not None
            and not cfg_disabled
        )
        if use_cfg:
            # Unconditional mask: same as prefix_mask but advantage token = False
            prefix_mask_uncond = prefix_mask.at[:, adv_token_idx].set(False)

        # Resolve time_threshold_inpaint: scalar or [B*N]
        tti = (
            jnp.asarray(time_threshold_inpaint) if time_threshold_inpaint is not None
            else jnp.asarray(self.config.time_threshold_inpaint)
        )
        # Broadcast scalar to [batch_size] for per-element threshold comparison
        if tti.ndim == 0:
            tti = jnp.broadcast_to(tti, (batch_size,))

        use_wm_flow = self.config.use_wm_flow_head and getattr(self, "wm_flow_query_token", None) is not None
        _KP_W = self.wm_flow_keypoint_head.kernel.value.shape[-1] if use_wm_flow else 1
        def step(carry):
            x_t, time, step_rng, wm_s_c, wm_c_c, wm_k_c, wm_s_u, wm_c_u, wm_k_u = carry

            # Model forward pass (conditional).
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size),
                advantage_mask=adv_mask_inference,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_step = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            # Action tokens are the FIRST ``action_horizon`` suffix positions
            # (the WM-flow token, if present, lives at the tail).
            v_cond = self.action_out_proj(suffix_out[:, :self.action_horizon])

            # WM-flow predictions (conditional pass). Reading suffix_out[:, -1]
            # gives the WM-flow token output when enabled — or the last action
            # token otherwise (unused because use_wm_flow=False).
            if use_wm_flow:
                wm_hid_c = suffix_out[:, -1, :]
                # Cast heads' outputs to float32: the while_loop requires carry
                # input/output dtypes to match, and the init placeholders are
                # float32 (to preserve precision when returned to callers).
                # success head now returns Δsuccess (raw, no sigmoid).
                wm_s_c_new = self.wm_flow_success_head(wm_hid_c).squeeze(-1).astype(jnp.float32)
                wm_c_c_new = jax.nn.sigmoid(self.wm_flow_completion_head(wm_hid_c).squeeze(-1)).astype(jnp.float32)
                wm_k_c_new = self.wm_flow_keypoint_head(wm_hid_c).astype(jnp.float32)
            else:
                wm_s_c_new, wm_c_c_new, wm_k_c_new = wm_s_c, wm_c_c, wm_k_c

            # Classifier-Free Guidance: unconditional pass when cfg_scale != 1.0
            if use_cfg:
                prefix_attn_mask_uncond = einops.repeat(prefix_mask_uncond, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask_uncond = jnp.concatenate([prefix_attn_mask_uncond, suffix_attn_mask], axis=-1)

                adv_vec = self.advantage_adarms_vec.value.astype(adarms_cond.dtype)
                if adv_mask_inference is not None:
                    adv_sub = jnp.where(
                        adv_mask_inference[:, None],
                        adv_vec,
                        jnp.zeros_like(adv_vec),
                    )
                else:
                    adv_sub = adv_vec
                adarms_cond_uncond = adarms_cond - adv_sub

                (_, suffix_out_uncond), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask_uncond,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond_uncond],
                )
                v_uncond = self.action_out_proj(suffix_out_uncond[:, :self.action_horizon])

                if use_wm_flow:
                    wm_hid_u = suffix_out_uncond[:, -1, :]
                    # success head returns Δsuccess (raw, no sigmoid).
                    wm_s_u_new = self.wm_flow_success_head(wm_hid_u).squeeze(-1).astype(jnp.float32)
                    wm_c_u_new = jax.nn.sigmoid(self.wm_flow_completion_head(wm_hid_u).squeeze(-1)).astype(jnp.float32)
                    wm_k_u_new = self.wm_flow_keypoint_head(wm_hid_u).astype(jnp.float32)
                else:
                    wm_s_u_new, wm_c_u_new, wm_k_u_new = wm_s_u, wm_c_u, wm_k_u

                cs = cfg_scale
                if cs.ndim == 1:
                    cs = cs[:, None, None]
                v_t = v_uncond + cs * (v_cond - v_uncond)
            else:
                v_t = v_cond
                # Without CFG, unconditional mirrors conditional (advantage
                # contribution is irrelevant — there is no mask divergence).
                wm_s_u_new, wm_c_u_new, wm_k_u_new = wm_s_c_new, wm_c_c_new, wm_k_c_new

            x_t_new = x_t + dt * v_t

            # Mask-based inpainting correction (vectorized, supports different inpaint lengths per element)
            if has_inpainting:
                time_new = time + dt

                # Compute desired values at observed positions: x_t[O] = (1-t)*x0[O] + t*z[O]
                x_desired = (1.0 - time_new) * x0 + time_new * fixed_z  # [B, 360]
                x_flat = x_t_new.reshape(batch_size, flat_dim)

                # Delta at observed positions (zero at free positions due to mask)
                delta = (x_desired - x_flat) * inpaint_mask  # [B, 360]

                # Hard constraint at observed positions
                x_flat = jnp.where(inpaint_mask > 0.5, x_desired, x_flat)

                # Correlated correction: propagate O errors to U positions
                # correction[b, i] = sum_j delta[b, j] * C_batch[b, j, i]
                correction = jnp.einsum('bj,bji->bi', delta, C_batch)  # [B, 360]
                # Only apply at unobserved positions
                correction = correction * (1.0 - inpaint_mask)

                # Per-element stability check
                max_corr = jnp.max(jnp.abs(correction), axis=-1)  # [B]
                stable = max_corr <= 1.0  # [B]
                x_flat = x_flat + jnp.where(stable[:, None], correction, 0.0)

                x_corrected = x_flat.reshape(batch_size, self.action_horizon, self.action_dim)

                # Per-element threshold: only apply correction when time_new > tti[b]
                should_correct = (time_new > tti)  # [B]
                x_t_new = jnp.where(should_correct[:, None, None], x_corrected, x_t_new)

            return (x_t_new, time + dt, step_rng,
                    wm_s_c_new, wm_c_c_new, wm_k_c_new,
                    wm_s_u_new, wm_c_u_new, wm_k_u_new)

        def cond(carry):
            _x_t, time, *_ = carry
            # Robust to floating-point error
            return time >= -dt / 2

        # Initial WM carries — zero placeholders. Overwritten in every step once
        # ``use_wm_flow`` is True; preserved as-is otherwise (harmlessly unused).
        wm_init_s = jnp.zeros((batch_size,), dtype=jnp.float32)
        wm_init_k = jnp.zeros((batch_size, _KP_W), dtype=jnp.float32)
        init_carry = (
            noise, 1.0, step_rng,
            wm_init_s, wm_init_s, wm_init_k,  # cond
            wm_init_s, wm_init_s, wm_init_k,  # uncond
        )
        (x_0, _t_final, _rng_final,
         wm_flow_success_cond, wm_flow_completion_cond, wm_flow_keypoint_cond,
         wm_flow_success_uncond, wm_flow_completion_uncond, wm_flow_keypoint_uncond
         ) = jax.lax.while_loop(cond, step, init_carry)

        # DART-style exploratory perturbation in normalized action space.
        # Drawn from the same correlation structure as the seed noise so the
        # perturbation looks like a plausible action rather than white kicks.
        # Hard-gated: only fires when scale > 0 AND the correlation cache is
        # loaded (correlation cache load is a runtime gate against accidental
        # i.i.d. fallback). Inference servers never set this — rollout-only.
        ens = jnp.asarray(explore_noise_scale, dtype=jnp.float32)
        if ens.ndim == 0:
            ens_b = jnp.broadcast_to(ens, (batch_size,))
        else:
            ens_b = ens
        if _correction_matrices_cache is not None:
            rng, explore_rng = jax.random.split(rng)
            explore_noise = self.generate_correlated_noise(explore_rng, batch_size)
            x_0 = x_0 + ens_b[:, None, None] * explore_noise

        # Keypoint-distance head prediction (Head 1) was computed during Phase 1
        # at B_unique then tiled to batch_size=B*num_candidates in Phase 2.
        # No denoising-loop dependency.

        # WM-flow predictions correspond to the FINAL loop iteration — which
        # evaluates the suffix at t = 0.1 (last step before x_0). When
        # ``use_wm_flow`` is False all WM fields stay at their zero placeholders.
        wm_flow_preds = None
        if use_wm_flow:
            wm_flow_preds = {
                "success_cond": wm_flow_success_cond,
                "completion_cond": wm_flow_completion_cond,
                "keypoint_cond": wm_flow_keypoint_cond,
                "success_uncond": wm_flow_success_uncond,
                "completion_uncond": wm_flow_completion_uncond,
                "keypoint_uncond": wm_flow_keypoint_uncond,
            }

        return (
            x_0,
            success_pred, checkpoint_pred, garment_type_pred, completion_pred, ttc_pred,
            keypoint_distances_pred, wm_flow_preds,
        )
