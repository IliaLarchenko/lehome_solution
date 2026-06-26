"""LeHome Policy Transforms

Transforms LeHome observations to model format.

Based on B1K policy transforms but adapted for LeHome's 12D state/action space
and top/left/right camera configuration.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LeHomeInputs(transforms.DataTransformFn):
    """Transforms LeHome observations into the model's expected input format.

    LeHome state is already a 12D vector (no proprioception extraction needed).
    Images come from three cameras: top_rgb, left_rgb, right_rgb.
    """

    # Determines which model will be used (kept for compatibility)
    model_type: _model.ModelType | str = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        # State is already 12D — read directly
        state = np.asarray(data["observation/state"], dtype=np.float32)

        if "actions" in data:
            action = data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        top_image = _parse_image(data["observation/top_rgb"])
        left_image = _parse_image(data["observation/left_rgb"])
        right_image = _parse_image(data["observation/right_rgb"])

        # LeHome uses 3 cameras: top, left, right
        names = ("top_rgb", "left_rgb", "right_rgb")
        images = (top_image, left_image, right_image)
        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action
        # Per-timestep pad flag: True where the action horizon overshoots the
        # episode and LeRobot repeated the last real action. The training loss
        # masks these out so padded timesteps don't contaminate the flow
        # matching gradient.
        if "action_is_pad" in data:
            inputs["action_is_pad"] = data["action_is_pad"]

        # Pass through normalised advantage for advantage-embedding conditioning
        if "advantage" in data:
            inputs["advantage"] = data["advantage"]

        # Pass through success_pred (P(success) predictions) and success (binary label)
        if "success_pred" in data:
            inputs["success_pred"] = data["success_pred"]
        if "success" in data:
            inputs["success"] = data["success"]
        if "success_target" in data:
            inputs["success_target"] = data["success_target"]

        # IS weight for de-biasing aux losses under prioritized sampling
        if "is_weight" in data:
            inputs["is_weight"] = data["is_weight"]

        # Per-garment average rates for label smoothing baseline
        if "garment_avg_success" in data:
            inputs["garment_avg_success"] = data["garment_avg_success"]
        if "garment_avg_checkpoint" in data:
            inputs["garment_avg_checkpoint"] = data["garment_avg_checkpoint"]
        # Backward compat: legacy type_avg_reward -> garment_avg_success
        if "type_avg_reward" in data and "garment_avg_success" not in inputs:
            inputs["garment_avg_success"] = data["type_avg_reward"]

        # Debiasing weights for prediction head losses
        if "success_debias_weight" in data:
            inputs["success_debias_weight"] = data["success_debias_weight"]
        if "checkpoint_debias_weight" in data:
            inputs["checkpoint_debias_weight"] = data["checkpoint_debias_weight"]

        # Auxiliary head targets and intermediate fields. NOTE: this is an
        # allowlist — any key not listed here is silently dropped. When adding
        # new transforms downstream (e.g. ``ComputeKeypointDistanceTargets``,
        # ``ComputeFutureAuxTargets``), remember to pass their source inputs
        # and outputs through here too.
        for key in (
            "checkpoint_target", "garment_type_id",
            "completion_target", "completion_frac", "ep_len",
            "dense_return", "max_reward",
            # Keypoint + world-modeling raw inputs (consumed by
            # ComputeKeypointDistanceTargets / ComputeFutureAuxTargets).
            "check_distances", "check_distances_future",
            "success_future", "completion_future",
            # Past ttc_pred(t + 30) for the TTC bootstrap target in
            # ComputeAuxTargets. NaN when absent → analytical fallback.
            "ttc_pred_future",
            # Per-sample multiplier on the TTC loss weight (1.0 for last-N
            # RL datasets; 0.01 for older RL so they regularise but don't
            # dominate the head).
            "ttc_weight_scale",
        ):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class LeHomeOutputs(transforms.DataTransformFn):
    """Transforms model outputs back to LeHome action format (12D).

    NOTE: allowlist of passed-through keys. Any new model output that should
    reach the policy server response MUST be listed here — otherwise it is
    silently dropped. Same pattern as ``LeHomeInputs`` (the inbound
    allowlist): any new model output / observation field must be added here.
    """

    def __call__(self, data: dict) -> dict:
        result = {"actions": np.asarray(data["actions"])}

        # Preserve prediction head outputs.
        for key in (
            "success_pred", "checkpoint_pred", "garment_type_pred",
            "completion_pred", "ttc_pred",
            # Keypoint + world-modeling predictions. MUST be
            # listed here or they get dropped before reaching the rollout worker.
            "keypoint_distances",
            "wm_flow_success_cond", "wm_flow_completion_cond", "wm_flow_keypoint_cond",
            "wm_flow_success_uncond", "wm_flow_completion_uncond", "wm_flow_keypoint_uncond",
        ):
            if key in data:
                result[key] = data[key]

        return result
