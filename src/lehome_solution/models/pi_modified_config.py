"""PiModified Model Configuration

Configuration for PiModified model for LeHome Challenge garment manipulation.
Adapted from the Pi0.5 architecture for garment manipulation.
"""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at

from lehome_solution.models.observation import Observation
from lehome_solution.models.train_aug_config import TrainAugmentationConfig

if TYPE_CHECKING:
    from lehome_solution.models.pi_modified import PiModified


@dataclasses.dataclass(frozen=True)
class PiModifiedConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Action dimensions: 12D dual-arm (6 joints per arm x 2 arms)
    action_dim: int = 12
    # Action horizon: 30 steps at 30 FPS = 1 second
    # NOTE: Adjust based on actual dataset FPS (check dataset meta/info.json)
    action_horizon: int = 30
    max_token_len: int = 200  # Only used for compatibility, not for actual tokenization

    # Whether to use correlated noise matching action covariance structure
    use_correlated_noise: bool = True

    # Shrinkage parameter for correlation regularization
    correlation_beta: float = 0.5

    # FAST auxiliary training configuration
    use_fast_auxiliary: bool = False
    fast_loss_weight: float = 0.1

    # Action dimensions to encode with FAST (all 12 dims for dual-arm)
    fast_encoded_dims: str | list[tuple[int, int]] = "0:12"

    # FAST tokenizer vocab size
    fast_vocab_size: int = 1024
    max_fast_tokens: int = 32

    # FAST tokenizer path
    fast_tokenizer_path: str | None = None

    # KV cache transformation for cross-layer attention
    use_kv_transform: bool = True

    # Knowledge insulation
    use_knowledge_insulation: bool = True

    # Time threshold for inpainting during inference
    time_threshold_inpaint: float = 0.3

    # Advantage embedding (pi0.6* style RL conditioning).
    # Always uses the positive vector (idx 0); the negative vector (idx 1) is
    # kept in params for checkpoint compat but never selected. During training
    # negative-advantage samples are always masked out (neutral) via the
    # attention mask; non-negative samples are masked stochastically with
    # P(neutral) decaying with the advantage. During inference the positive
    # vector is always used (unmasked).
    use_advantage_embedding: bool = False
    # P(masked/neutral) when |A| = 0  (fully ambiguous advantage → heavy masking)
    neutral_prob_at_zero: float = 0.5
    # P(masked/neutral) when |A| >= max_adv_for_neutral  (strong signal → little masking)
    neutral_prob_at_max: float = 0.1
    # |A| value at which neutral/masking probability saturates to neutral_prob_at_max
    max_adv_for_neutral: float = 2.0
    # Threshold: advantage >= this value can get positive embedding (with stochastic
    # masking); advantage < this value is always masked out (neutral).
    positive_advantage_threshold: float = 0.0

    # Classifier-Free Guidance scale for advantage conditioning during inference.
    # 1.0 = standard conditional (no guidance overhead).
    # 0.0 = unconditional (ignores advantage token).
    # >1.0 = amplified conditioning (sharpens advantage-conditioned prediction).
    # V_guided = V_uncond + cfg_scale * (V_cond - V_uncond)
    cfg_scale: float = 1.0

    # Integrated success head: learned query token in VLM prefix → P(success) ∈ (0,1).
    use_success_head: bool = False
    success_loss_weight: float = 0.1

    # Auxiliary prediction heads (all share the success query token).
    # P(reward >= 0.5) — checkpoint probability head.
    checkpoint_loss_weight: float = 0.1
    # Garment type classification (4-class softmax).
    garment_type_loss_weight: float = 0.1
    # Garment type input embedding: feed garment type as a conditioning token in
    # the state attention group (group 1).  Invisible to the success_query_token
    # (group 0), so the garment_type_head still learns to predict from images.
    use_garment_type_input: bool = True
    # Completion percentage prediction (step/total_steps, success-only).
    completion_loss_weight: float = 0.1
    # Time-to-completion head (sigmoid output in [0, 1]). Analytical target:
    # 1 - steps_left/600 (success) or 0 (failure); newer RL data uses a TD
    # bootstrap target instead (see transforms.ComputeAuxTargets).
    ttc_loss_weight: float = 0.1

    # -----------------------------------------------------------------------
    # World-modeling / Q-function auxiliary heads.
    #
    # Three additional signals on top of the existing aux heads:
    #
    #   (1) keypoint_distance_head — predicts per-condition normalized distances
    #       used by the success checker. 21 outputs with per-garment slices
    #       (see lehome_solution.constants.KEYPOINT_SLICES). Runs from the
    #       success_query hidden state. Enabled by default when use_success_head.
    #
    #   (2) post-FAST Q/WM head — a new learned query token appended after the
    #       FAST block in the prefix. Predicts future (t + 30 steps) success,
    #       completion, and keypoint distances. Training-only: absent at
    #       inference because FAST itself is absent there.
    #
    #   (3) post-flow Q/WM head — a new learned query token appended at the
    #       tail of the action-expert suffix. Same three future predictions.
    #       Inference reads this at t=0.1 (last denoising step before t→0).
    #
    # Loss weights default to 0.05 each. Predictions are raw linear outputs for
    # the keypoint part (distances can exceed 1). success/completion use sigmoid.
    # -----------------------------------------------------------------------
    use_keypoint_distance_head: bool = True
    keypoint_distance_loss_weight: float = 0.05

    use_wm_fast_head: bool = True
    wm_fast_success_loss_weight: float = 0.05
    wm_fast_completion_loss_weight: float = 0.05
    wm_fast_keypoint_loss_weight: float = 0.05

    use_wm_flow_head: bool = True
    wm_flow_success_loss_weight: float = 0.05
    wm_flow_completion_loss_weight: float = 0.05
    wm_flow_keypoint_loss_weight: float = 0.05

    # Rotate top camera image by 180° for proper egocentric view (dataset-specific).
    rotate_top_image: bool = False

    # Override number of LLM layers for smaller model variants.
    # When set, both paligemma and action expert use this depth instead of the
    # default from their variant configs (e.g. 9 instead of 18).
    # None means use the default depth from the variant.
    num_llm_layers: int | None = None

    # Vision backbone finetuning control
    freeze_vision_backbone: bool = True
    # Whether to freeze the SigLIP head (projection layer from SigLIP width to VLM width).
    # Independent of freeze_vision_backbone — allows training just the head while
    # keeping the rest of the vision encoder frozen.
    freeze_vision_head: bool = True

    # SigLIP variant for vision backbone.
    # Default "So400m/14" is the original Pi0.5 vision encoder (~400M params).
    # Smaller variants: "S/14" (~22M), "B/16" (~86M, SigLIP-2), etc.
    siglip_variant: str = "So400m/14"

    # Use a small 256-entry state embedding instead of the Gemma vocab embedder.
    # The Gemma embedder has 257K entries (527M params) but only indices 0-255
    # are used for discretized state.  When True, a dedicated 256×width embedding
    # replaces it and the Gemma embedding table is frozen (saving optimizer state).
    use_state_embedding: bool = False


    # Training-time image + state augmentation config.  All defaults preserve
    # the pre-refactor behaviour (brightness/contrast/saturation/blur/top-cam
    # geo at their historical magnitudes; everything else off).  See
    # `lehome_solution.models.train_aug_config.TrainAugmentationConfig`.
    # NOTE: `state_noise_std` previously lived on this dataclass; it is now
    # `train_aug.state_noise_std`.
    train_aug: TrainAugmentationConfig = dataclasses.field(default_factory=TrainAugmentationConfig)

    # Input image resolution (H, W).  Must be divisible by the SigLIP patch size.
    # Default (224, 224) works for all /14 and /16 variants.
    # Some SigLIP-2 models may require different resolutions (e.g. 256, 384).
    image_resolution: tuple[int, int] = (224, 224)

    def get_fast_dim_ranges(self) -> list[tuple[int, int]]:
        """Parse fast_encoded_dims into list of ranges."""
        if isinstance(self.fast_encoded_dims, str):
            ranges = []
            for range_str in self.fast_encoded_dims.split(','):
                start, end = map(int, range_str.strip().split(':'))
                ranges.append((start, end))
            return ranges
        return self.fast_encoded_dims

    def get_total_fast_dims(self) -> int:
        """Get total number of dimensions encoded by FAST."""
        return sum(end - start for start, end in self.get_fast_dim_ranges())

    @property
    @override
    def model_type(self):
        return "pi_modified"

    @override
    def create(self, rng: at.KeyArrayLike) -> "PiModified":
        from lehome_solution.models.pi_modified import PiModified

        return PiModified(self, rngs=nnx.Rngs(rng))

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "PiModified":
        """Load model, filling missing params (e.g. new aux heads) from random init."""
        import orbax.checkpoint as ocp
        import logging

        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        expected = state.to_pure_dict()

        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(expected, params)

        # Fill missing keys from randomly-initialized model (for new aux heads)
        missing = set(expected.keys()) - set(params.keys())
        if missing:
            logging.getLogger("lehome_solution").info(
                "Filling %d missing param groups from random init: %s", len(missing), sorted(missing)
            )
            # Create a real model to get initialized values
            real_model = self.create(jax.random.key(42))
            _, real_state = nnx.split(real_model)
            real_dict = real_state.to_pure_dict()
            for key in missing:
                params[key] = real_dict[key]

        at.check_pytree_equality(expected=expected, got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple["Observation", _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *self.image_resolution, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            obs_kwargs = {
                "images": {
                    "top_rgb": image_spec,
                    "left_rgb": image_spec,
                    "right_rgb": image_spec,
                },
                "image_masks": {
                    "top_rgb": image_mask_spec,
                    "left_rgb": image_mask_spec,
                    "right_rgb": image_mask_spec,
                },
                "state": jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
            }

            if self.use_fast_auxiliary:
                obs_kwargs["fast_tokens"] = jax.ShapeDtypeStruct([batch_size, self.max_fast_tokens], jnp.int32)
                obs_kwargs["fast_token_mask"] = jax.ShapeDtypeStruct([batch_size, self.max_fast_tokens], bool)

            if self.use_advantage_embedding:
                obs_kwargs["advantage"] = jax.ShapeDtypeStruct([batch_size], jnp.float32)

            if self.use_garment_type_input:
                obs_kwargs["garment_type_id"] = jax.ShapeDtypeStruct([batch_size], jnp.int32)

            if self.use_success_head:
                obs_kwargs["success_target"] = jax.ShapeDtypeStruct([batch_size], jnp.float32)
                obs_kwargs["is_weight"] = jax.ShapeDtypeStruct([batch_size], jnp.float32)
                obs_kwargs["checkpoint_target"] = jax.ShapeDtypeStruct([batch_size], jnp.float32)
                obs_kwargs.setdefault("garment_type_id", jax.ShapeDtypeStruct([batch_size], jnp.int32))
                obs_kwargs["completion_target"] = jax.ShapeDtypeStruct([batch_size], jnp.float32)
                obs_kwargs["ttc_target"] = jax.ShapeDtypeStruct([batch_size], jnp.float32)
                obs_kwargs["ttc_weight_scale"] = jax.ShapeDtypeStruct([batch_size], jnp.float32)
                obs_kwargs["ttc_used_bootstrap"] = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

            observation_spec = Observation(**obs_kwargs)

        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
