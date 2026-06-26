"""Weight loaders for PiModified model initialization.

Handles dimension mismatches (e.g. Pi0.5 uses action_dim=32, we use 12)
by loading the intersection of dimensions and keeping the rest from the
randomly initialized model.

Supports loading SigLIP-2 pretrained vision weights from HuggingFace.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import pathlib
import re

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)


def _merge_params_flexible(
    loaded_params: at.Params,
    params: at.Params,
    *,
    missing_regex: str,
) -> at.Params:
    """Merge loaded parameters into reference parameters, handling dimension mismatches.

    When a parameter exists in both loaded and reference but shapes differ:
    - If ranks match, load the intersection (min of each axis) and keep the
      reference values (random init) for the remaining elements.
    - If ranks differ, skip the parameter (keep reference).

    This allows loading e.g. Pi0.5 (action_dim=32) into PiModified (action_dim=12)
    by taking the first 12 rows/columns of projection layers.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    result = {}
    for k, v_loaded in flat_loaded.items():
        if k not in flat_ref:
            continue

        v_ref = flat_ref[k]

        # Reference may be a jax.ShapeDtypeStruct (shape spec) or a real array.
        # Extract shape/dtype without converting to numpy.
        ref_shape = v_ref.shape
        ref_dtype = v_ref.dtype
        v_loaded = np.asarray(v_loaded)

        if v_loaded.shape == ref_shape:
            # Exact match — load directly
            result[k] = v_loaded.astype(ref_dtype) if v_loaded.dtype != ref_dtype else v_loaded
        elif v_loaded.ndim == len(ref_shape) and v_loaded.ndim > 0:
            # Same rank (non-scalar), different shape — load intersection, zero-fill rest
            slices = tuple(slice(0, min(s_l, s_r)) for s_l, s_r in zip(v_loaded.shape, ref_shape))
            merged = np.zeros(ref_shape, dtype=ref_dtype)
            merged[slices] = np.array(v_loaded[slices]).astype(ref_dtype)
            result[k] = merged
            logger.warning(
                f"Dimension mismatch at '{k}': loaded {v_loaded.shape} vs model {ref_shape}. "
                f"Loaded intersection {tuple(s.stop for s in slices)}, rest zero-initialized."
            )
        else:
            # Different rank or scalar mismatch — skip, keep reference
            logger.warning(
                f"Shape mismatch at '{k}': loaded {v_loaded.shape} vs model {ref_shape}. "
                f"Keeping random init."
            )

    flat_loaded.clear()

    # Merge missing weights from reference
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _scale_kv_coeffs(merged_params: at.Params, loaded_params: at.Params) -> at.Params:
    """Scale KV transform coefficients when loading from a larger checkpoint.

    When loading e.g. 18-layer KV coefficients into a 9-layer model,
    ``_merge_params_flexible`` takes the top-left [N, N] submatrix.
    This function rescales each row so the total coefficient sum is preserved:
        scale[d] = sum(full_row[d]) / sum(truncated_row[d])

    If a truncated row sums to ~0 (all kept coefficients are negligible),
    it is re-initialized randomly instead of scaling, since scaling would
    amplify noise.

    Only modifies ``k_coeffs`` and ``v_coeffs`` under ``kv_transform/``.
    """
    flat_merged = flax.traverse_util.flatten_dict(merged_params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    rng = np.random.RandomState(42)

    for suffix in ("k_coeffs", "v_coeffs"):
        key = f"kv_transform/{suffix}"
        if key not in flat_merged or key not in flat_loaded:
            continue

        loaded_arr = np.asarray(flat_loaded[key])
        merged_arr = np.asarray(flat_merged[key])

        if loaded_arr.shape == merged_arr.shape:
            continue  # same depth, nothing to scale

        n_dst = merged_arr.shape[0]
        n_src = merged_arr.shape[1]
        n_reinit = 0
        for d in range(n_dst):
            full_row_sum = float(loaded_arr[d, :].sum())
            trunc_row_sum = float(merged_arr[d, :].sum())
            if abs(trunc_row_sum) > 1e-8:
                scale = full_row_sum / trunc_row_sum
                merged_arr[d, :] *= scale
            else:
                # Truncated row is ~zero — random init (uniform, sums to ~1)
                merged_arr[d, :] = rng.randn(n_src).astype(merged_arr.dtype)
                merged_arr[d, :] /= np.abs(merged_arr[d, :]).sum()
                merged_arr[d, :] *= abs(full_row_sum) if abs(full_row_sum) > 1e-8 else 1.0
                n_reinit += 1

        flat_merged[key] = merged_arr
        logger.info(
            f"Scaled {key}: {loaded_arr.shape} -> {merged_arr.shape} "
            f"(row sums preserved, {n_reinit} rows re-initialized)"
        )

    return flax.traverse_util.unflatten_dict(flat_merged, sep="/")


@dataclasses.dataclass(frozen=True)
class PiModifiedWeightLoader(WeightLoader):
    """Loads checkpoints for PiModified model.

    Automatically detects:
    - Pi05 checkpoint: Loads weights with dimension-aware merge (handles action_dim mismatch)
    - PiModified checkpoint: Loads all weights directly

    When loading into a model with fewer LLM layers (``num_llm_layers``):
    - Scanned layer params are truncated to the first N layers automatically
    - KV transform coefficients are scaled so per-row sums are preserved
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint
        params_path = download.maybe_download(self.params_path)

        # Load directly with PyTreeCheckpointer (handles both old and new checkpoint formats)
        with ocp.PyTreeCheckpointer() as ckptr:
            restored = ckptr.restore(params_path)

        # Handle nested 'params' key (from some checkpoint formats)
        if isinstance(restored, dict) and "params" in restored:
            loaded_params = restored["params"]
        else:
            loaded_params = restored

        # Remove 'value' suffixes (from nnx.State format)
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) >= 2):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items() if len(kp) >= 2}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)

        # Drop the Gemma embedding table from loaded checkpoint if the model
        # uses a dedicated state embedding (the table is replaced with a dummy).
        # Note: Pi0.5 tokenizes state as text ("0", "1", ...) via SentencePiece,
        # so the relevant embeddings are scattered across the 257K vocab table.
        # PiModified's state_embedding uses direct indices 0-255, which is a
        # different approach — random init is correct here.
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        ref_embed_key = "PaliGemma/llm/embedder/input_embedding"
        if ref_embed_key in flat_ref and flat_ref[ref_embed_key].shape[0] == 1:
            flat_loaded_check = flax.traverse_util.flatten_dict(loaded_params, sep="/")
            if ref_embed_key in flat_loaded_check:
                logger.info(
                    "Replacing Gemma embedding table with dummy (%s -> %s)",
                    flat_loaded_check[ref_embed_key].shape, flat_ref[ref_embed_key].shape,
                )
                # Replace with dummy (1, width) instead of deleting, so the
                # embedder dict survives unflatten and validation passes.
                flat_loaded_check[ref_embed_key] = np.zeros(
                    flat_ref[ref_embed_key].shape, dtype=flat_ref[ref_embed_key].dtype
                )
                loaded_params = flax.traverse_util.unflatten_dict(flat_loaded_check, sep="/")

        # Detect checkpoint type: PiModified has kv_transform, success_head, etc.
        is_pi_modified = any(
            k in loaded_params
            for k in ('kv_transform', 'success_head', 'success_query_token')
        )

        if is_pi_modified:
            # Loading PiModified checkpoint — use flexible merge to handle
            # checkpoints from older configs (e.g. action_dim=32 → 12).
            # New model params not present in the checkpoint keep random init.
            logging.info("Loading PiModified checkpoint")
            new_param_regex = (
                ".*advantage_embeddings.*|"
                ".*advantage_adarms_vec.*|"
                ".*garment_type_input_embedding.*|"
                ".*garment_type_adarms_embedding.*|"
                ".*checkpoint_head.*|"
                ".*garment_type_head.*|"
                ".*completion_head.*|"
                ".*ttc_head.*|"
                # World-modeling / Q heads.
                ".*keypoint_distance_head.*|"
                ".*wm_fast_query_token.*|"
                ".*wm_fast_success_head.*|"
                ".*wm_fast_completion_head.*|"
                ".*wm_fast_keypoint_head.*|"
                ".*wm_flow_query_token.*|"
                ".*wm_flow_success_head.*|"
                ".*wm_flow_completion_head.*|"
                ".*wm_flow_keypoint_head.*"
            )
            merged = _merge_params_flexible(loaded_params, params, missing_regex=new_param_regex)
        else:
            # Loading Pi05 checkpoint — preserve new PiModified-specific parameters
            logging.info("Loading Pi05 checkpoint (new PiModified parameters will use random init)")

            # These parameters are NEW in PiModified (not in Pi05), so keep them from params (random init)
            missing_regex = (
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*|"
                ".*advantage_embeddings.*|"
                ".*advantage_adarms_vec.*|"
                ".*state_embedding.*|"
                ".*success_head.*|"
                ".*success_query_token.*|"
                ".*checkpoint_head.*|"
                ".*garment_type_head.*|"
                ".*garment_type_input_embedding.*|"
                ".*garment_type_adarms_embedding.*|"
                ".*completion_head.*|"
                ".*ttc_head.*|"
                # World-modeling / Q heads.
                ".*keypoint_distance_head.*|"
                ".*wm_fast_query_token.*|"
                ".*wm_fast_success_head.*|"
                ".*wm_fast_completion_head.*|"
                ".*wm_fast_keypoint_head.*|"
                ".*wm_flow_query_token.*|"
                ".*wm_flow_success_head.*|"
                ".*wm_flow_completion_head.*|"
                ".*wm_flow_keypoint_head.*"
            )
            merged = _merge_params_flexible(loaded_params, params, missing_regex=missing_regex)

        # Scale KV coefficients when loading from a larger checkpoint
        merged = _scale_kv_coeffs(merged, loaded_params)

        return merged
