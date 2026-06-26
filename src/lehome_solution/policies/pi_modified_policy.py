"""Minimal Policy subclass for PiModified models that handles tuple unpacking.

Based on the PiBehavior policy but adapted for LeHome.
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.policies.policy import Policy
from lehome_solution.models.observation import Observation  # Use our custom Observation with FAST fields
from lehome_solution.constants import ACTION_HORIZON, ACTION_DIM


class PiModifiedPolicy(Policy):
    """Policy for PiModified models.

    Handles tuple unpacking from PiModified.sample_actions() and supports
    initial_actions for rolling inpainting (converted to mask format here).
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Re-jit ``sample_actions`` with ``num_candidates`` marked static.
        # ``num_candidates`` controls a Python-level tiling factor inside the
        # model (jnp.repeat with a static repeats count) — it cannot be a
        # traced int because the post-tile shapes must be known at trace time.
        # We compile a separate trace per (B_unique, num_candidates) pair,
        # which is bounded ({1..max_workers} × {1, 2, 3} = ~18 entries).
        if not getattr(self, "_is_pytorch_model", False):
            from openpi.shared import nnx_utils
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions,
                static_argnames=("num_candidates", "cfg_disabled"),
            )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, initial_actions: np.ndarray | None = None, inference_overrides: dict | None = None, cfg_disabled: bool = False) -> dict:
        # Reuse all parent logic for input processing
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        # Prepare sample_kwargs
        sample_kwargs = dict(self._sample_kwargs)

        # Apply per-request overrides (from inference optimization)
        if inference_overrides:
            if "num_steps" in inference_overrides:
                sample_kwargs["num_steps"] = int(inference_overrides["num_steps"])
            if "time_threshold_inpaint" in inference_overrides:
                sample_kwargs["time_threshold_inpaint"] = jnp.array(
                    inference_overrides["time_threshold_inpaint"], dtype=jnp.float32
                )
            if "noise_temperature" in inference_overrides:
                sample_kwargs["noise_temperature"] = float(inference_overrides["noise_temperature"])
            if "cfg_scale" in inference_overrides:
                sample_kwargs["cfg_scale"] = float(inference_overrides["cfg_scale"])
            if "explore_noise_scale" in inference_overrides:
                sample_kwargs["explore_noise_scale"] = float(inference_overrides["explore_noise_scale"])
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        if initial_actions is not None:
            # Create training-format observation batch for proper transform processing
            training_obs = {}

            # Map evaluation keys to training keys if needed
            if "observation/state" in obs:
                training_obs["observation/state"] = obs["observation/state"]
            elif "state" in obs:
                training_obs["observation/state"] = obs["state"]

            if "observation/top_rgb" in obs:
                training_obs["observation/top_rgb"] = obs["observation/top_rgb"]
            elif "image" in obs and "top_rgb" in obs["image"]:
                training_obs["observation/top_rgb"] = obs["image"]["top_rgb"]

            if "observation/left_rgb" in obs:
                training_obs["observation/left_rgb"] = obs["observation/left_rgb"]
            elif "image" in obs and "left_rgb" in obs["image"]:
                training_obs["observation/left_rgb"] = obs["image"]["left_rgb"]

            if "observation/right_rgb" in obs:
                training_obs["observation/right_rgb"] = obs["observation/right_rgb"]
            elif "image" in obs and "right_rgb" in obs["image"]:
                training_obs["observation/right_rgb"] = obs["image"]["right_rgb"]

            initial_batch = {
                **training_obs,
                "actions": np.array(initial_actions, copy=True),  # Copy: DeltaActions mutates in-place
            }

            # Apply the full input transform pipeline (delta transforms + normalization)
            transformed_batch = self._input_transform(initial_batch)
            normalized_initial_actions = transformed_batch["actions"]

            # Convert to JAX and add batch dim
            norm_ia = jnp.asarray(normalized_initial_actions)
            if norm_ia.ndim == 2:
                norm_ia = norm_ia[None, ...]  # [1, n, D]

            # Convert to mask-based inpainting format
            action_horizon = ACTION_HORIZON
            action_dim = ACTION_DIM
            flat_dim = action_horizon * action_dim
            n = norm_ia.shape[1]
            # Pad to [1, action_horizon, action_dim]
            padded = jnp.zeros((1, action_horizon, action_dim), dtype=jnp.float32)
            padded = padded.at[:, :n, :action_dim].set(norm_ia[:, :n, :action_dim])
            sample_kwargs["inpaint_targets"] = padded
            # Build mask [1, flat_dim]
            mask = np.zeros(flat_dim, dtype=np.float32)
            for t in range(n):
                mask[t * action_dim : (t + 1) * action_dim] = 1.0
            sample_kwargs["inpaint_mask"] = jnp.array(mask[None, :])
            sample_kwargs["inpaint_lengths"] = jnp.array([n], dtype=jnp.int32)

        observation = Observation.from_dict(inputs)
        start_time = time.monotonic()

        # cfg_disabled is a Python-level static — passed positionally as a
        # static_argname so JIT compiles a separate kernel for True (uncond
        # branch erased) vs False (default CFG path).
        if cfg_disabled:
            sample_kwargs["cfg_disabled"] = True

        # Unpack tuple return from PiModified.sample_actions
        (
            actions,
            value_pred, checkpoint_pred, garment_type_pred, completion_pred, ttc_pred,
            keypoint_distances_pred, wm_flow_preds,
        ) = self._sample_actions(sample_rng, observation, **sample_kwargs)

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        if value_pred is not None:
            outputs["success_pred"] = value_pred
        if checkpoint_pred is not None:
            outputs["checkpoint_pred"] = checkpoint_pred
        if garment_type_pred is not None:
            outputs["garment_type_pred"] = garment_type_pred
        if completion_pred is not None:
            outputs["completion_pred"] = completion_pred
        if ttc_pred is not None:
            outputs["ttc_pred"] = ttc_pred
        if keypoint_distances_pred is not None:
            outputs["keypoint_distances"] = keypoint_distances_pred  # [B, 21]
        if wm_flow_preds is not None:
            # Cond / uncond are kept separate for monitoring. The keypoint
            # entries are full-width [B, 21]; the scalar success/completion
            # are [B]. All float32.
            for k, v in wm_flow_preds.items():
                outputs[f"wm_flow_{k}"] = v

        model_time = time.monotonic() - start_time

        # Convert to numpy (same as parent)
        outputs = {
            k: np.asarray(v[0, ...]) if isinstance(v, (jnp.ndarray, np.ndarray)) else v
            for k, v in outputs.items()
        }

        # Apply output transforms
        outputs = self._output_transform(outputs)

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_batched(
        self,
        obs_list: list[dict],
        initial_actions_list: list[np.ndarray | None],
        inference_overrides_list: list[dict | None],
        num_candidates: int = 1,
        cfg_disabled: bool = False,
    ) -> list[dict]:
        """Batched inference: process multiple requests in one GPU forward pass.

        Each element is transformed independently (delta + normalization), then
        stacked for a single model call.  Results are split and output-transformed
        per element.

        Args:
            obs_list: List of observation dicts (one per UNIQUE request — best-of-N
                replication is no longer the caller's responsibility).
            initial_actions_list: List of initial_actions arrays (or None).
            inference_overrides_list: List of per-request override dicts (or None).
            num_candidates: Best-of-N. When > 1 the model runs the VLM prefix once
                on B unique requests and tiles its kv_cache to B*num_candidates
                for the flow loop. The returned list has length B*num_candidates,
                with rows [i*num_candidates : (i+1)*num_candidates] corresponding
                to request i's N candidates.

        Returns:
            List of output dicts of length B * num_candidates.
        """
        B = len(obs_list)
        N = max(1, int(num_candidates))
        action_horizon = ACTION_HORIZON
        action_dim = ACTION_DIM
        flat_dim = action_horizon * action_dim

        # 1. Transform each element independently
        all_inputs = []
        all_inpaint_targets = []
        all_inpaint_masks = []
        all_inpaint_lengths = []
        all_time_threshold = []
        all_noise_temp = []
        all_cfg_scale = []
        all_explore_noise = []

        default_tti = self._sample_kwargs.get("time_threshold_inpaint", 0.3)

        for i in range(B):
            obs = obs_list[i]
            ia = initial_actions_list[i]
            overrides = inference_overrides_list[i] or {}

            # Transform observation
            inp = jax.tree.map(lambda x: x, obs)
            inp = self._input_transform(inp)
            all_inputs.append(inp)

            # Per-element overrides
            tti_val = overrides.get("time_threshold_inpaint", default_tti)
            if isinstance(tti_val, jnp.ndarray):
                tti_val = float(tti_val)
            all_time_threshold.append(float(tti_val))
            all_noise_temp.append(float(overrides.get("noise_temperature", 1.0)))
            all_cfg_scale.append(float(overrides.get("cfg_scale", self._sample_kwargs.get("cfg_scale", 1.0))))
            all_explore_noise.append(float(overrides.get("explore_noise_scale", 0.0)))

            # Transform initial_actions through the full pipeline (delta + normalization)
            if ia is not None:
                training_obs = {}
                for key in ("observation/state", "observation/top_rgb",
                            "observation/left_rgb", "observation/right_rgb"):
                    if key in obs:
                        training_obs[key] = obs[key]
                initial_batch = {**training_obs, "actions": np.array(ia, copy=True)}
                transformed = self._input_transform(initial_batch)
                norm_ia = np.asarray(transformed["actions"])  # [n, 12]

                n = norm_ia.shape[0]
                # Pad to [action_horizon, action_dim]
                padded = np.zeros((action_horizon, action_dim), dtype=np.float32)
                padded[:n, :action_dim] = norm_ia[:, :action_dim]
                all_inpaint_targets.append(padded)

                # Build mask
                mask = np.zeros(flat_dim, dtype=np.float32)
                for t in range(n):
                    mask[t * action_dim : (t + 1) * action_dim] = 1.0
                all_inpaint_masks.append(mask)
                all_inpaint_lengths.append(n)
            else:
                all_inpaint_targets.append(np.zeros((action_horizon, action_dim), dtype=np.float32))
                all_inpaint_masks.append(np.zeros(flat_dim, dtype=np.float32))
                all_inpaint_lengths.append(0)

        # 2. Stack into batched arrays
        batched_inputs = jax.tree.map(
            lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
            *all_inputs,
        )
        inpaint_targets = jnp.array(np.stack(all_inpaint_targets))  # [B, 30, 12]
        inpaint_mask = jnp.array(np.stack(all_inpaint_masks))       # [B, 360]
        inpaint_lengths = jnp.array(all_inpaint_lengths, dtype=jnp.int32)  # [B]
        time_threshold = jnp.array(all_time_threshold, dtype=jnp.float32)  # [B]
        noise_temp = jnp.array(all_noise_temp, dtype=jnp.float32)  # [B]
        cfg_scale_arr = jnp.array(all_cfg_scale, dtype=jnp.float32)  # [B]
        explore_noise_arr = jnp.array(all_explore_noise, dtype=jnp.float32)  # [B]

        # 3. Single RNG split
        self._rng, sample_rng = jax.random.split(self._rng)

        # 4. Build sample_kwargs (fixed for batch)
        sample_kwargs = dict(self._sample_kwargs)
        # Remove per-request overrides that are now batched
        sample_kwargs.pop("time_threshold_inpaint", None)
        sample_kwargs.pop("noise_temperature", None)
        sample_kwargs.pop("cfg_scale", None)
        sample_kwargs.pop("explore_noise_scale", None)
        sample_kwargs.pop("initial_actions", None)

        # Check if any element has inpainting
        has_any_inpaint = any(l > 0 for l in all_inpaint_lengths)
        if has_any_inpaint:
            sample_kwargs["inpaint_targets"] = inpaint_targets
            sample_kwargs["inpaint_mask"] = inpaint_mask
            sample_kwargs["inpaint_lengths"] = inpaint_lengths

        sample_kwargs["time_threshold_inpaint"] = time_threshold
        sample_kwargs["noise_temperature"] = noise_temp
        sample_kwargs["cfg_scale"] = cfg_scale_arr
        sample_kwargs["explore_noise_scale"] = explore_noise_arr
        sample_kwargs["num_candidates"] = N
        # cfg_disabled is static — separate JIT trace per True/False (only 2 entries).
        if cfg_disabled:
            sample_kwargs["cfg_disabled"] = True

        # 5. Single model call. Returns arrays at leading dim B*N (row k →
        # request k//N, candidate k%N).
        observation = Observation.from_dict(batched_inputs)
        start_time = time.monotonic()
        (
            actions,
            value_pred, checkpoint_pred, garment_type_pred, completion_pred, ttc_pred,
            keypoint_distances_pred, wm_flow_preds,
        ) = self._sample_actions(sample_rng, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        # 6. Split and output-transform per (request, candidate). State is at
        # leading dim B (input shape) — tile for output indexing alignment so
        # each candidate row carries the same per-request state.
        BN = B * N
        if N > 1:
            state_tiled = jnp.repeat(batched_inputs["state"], N, axis=0)
        else:
            state_tiled = batched_inputs["state"]
        results = []
        for i in range(BN):
            outputs_i = {
                "state": state_tiled[i:i+1],
                "actions": actions[i:i+1],
            }
            if value_pred is not None:
                outputs_i["success_pred"] = value_pred[i]
            if checkpoint_pred is not None:
                outputs_i["checkpoint_pred"] = checkpoint_pred[i]
            if garment_type_pred is not None:
                outputs_i["garment_type_pred"] = garment_type_pred[i]
            if completion_pred is not None:
                outputs_i["completion_pred"] = completion_pred[i]
            if ttc_pred is not None:
                outputs_i["ttc_pred"] = ttc_pred[i]
            # NOTE: downstream post-processing (line below) strips a leading
            # batch dim via ``v[0, ...]`` for every ndim > 0 array. To survive
            # that, array-valued predictions (keypoint vectors) must keep a
            # leading batch dim; scalars are left alone.
            if keypoint_distances_pred is not None:
                outputs_i["keypoint_distances"] = keypoint_distances_pred[i:i+1]  # [1, 21]
            if wm_flow_preds is not None:
                for k, v in wm_flow_preds.items():
                    if v.ndim >= 2:  # [B, 21] keypoint predictions
                        outputs_i[f"wm_flow_{k}"] = v[i:i+1]
                    else:  # [B] scalar predictions
                        outputs_i[f"wm_flow_{k}"] = v[i]  # 0-d

            # Remove batch dim → numpy
            outputs_i = {
                k: np.asarray(v[0, ...]) if isinstance(v, (jnp.ndarray, np.ndarray)) and hasattr(v, 'shape') and v.ndim > 0 else v
                for k, v in outputs_i.items()
            }
            outputs_i = self._output_transform(outputs_i)
            outputs_i["policy_timing"] = {"infer_ms": model_time * 1000 / BN}
            results.append(outputs_i)

        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
