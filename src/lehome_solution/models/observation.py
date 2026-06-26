"""Observation class and preprocessing with FAST auxiliary fields support.

Based on openpi with FAST fields added for PiModified model.
Adapted for LeHome Challenge garment manipulation.
"""

from collections.abc import Sequence
from typing import Generic, TypeVar
import dataclasses

import augmax
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.shared import image_tools
from openpi.shared import array_typing as at

from lehome_solution.models.train_aug_config import TrainAugmentationConfig

ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)

# LeHome camera keys
IMAGE_KEYS = (
    "top_rgb",
    "left_rgb",
    "right_rgb",
)
IMAGE_RESOLUTION = (224, 224)


def _is_top_key(key: str) -> bool:
    return ("left" not in key and "right" not in key and "wrist" not in key)


def _squeeze_scalar(x):
    """Squeeze trailing size-1 dimension if present (handles LeRobot [B,1] shape)."""
    if x is None:
        return None
    import numpy as _np
    x = _np.asarray(x)
    if x.ndim >= 2 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def _squeeze_scalar_int(x):
    """Squeeze trailing size-1 dimension and cast to int32."""
    if x is None:
        return None
    import numpy as _np
    x = _np.asarray(x)
    if x.ndim >= 2 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x.astype(_np.int32)




@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Observation with FAST auxiliary fields."""

    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]

    fast_tokens: at.Int[ArrayT, "*b t"] | None = None
    fast_token_mask: at.Bool[ArrayT, "*b t"] | None = None

    # RMS-normalised advantage for RL training (clipped to configured range).
    # None during pure BC training; consumed by PiModified when
    # use_advantage_embedding=True to select the per-sample embedding token.
    advantage: at.Float[ArrayT, "*b"] | None = None

    # Success target for integrated success head training: binary success (0/1).
    # None when success head is disabled.
    success_target: at.Float[ArrayT, "*b"] | None = None

    # Per-sample unbiased weight: (1 / (N * p_i)) / ep_len.
    # Same definition for every source (RL prioritised / BC-DAgger FR-proportional /
    # uniform). Never zeroed: BC/DAgger samples contribute normally to every aux loss.
    is_weight: at.Float[ArrayT, "*b"] | None = None

    # Per-garment debias multipliers (injected by data_loader from
    # garment_debias.json written by recompute_advantages.py). The success
    # multiplier also absorbs the x10 success-tail boost on the last 30 frames
    # of successful episodes. Default 1.0 when no table is available.
    success_debias_weight: at.Float[ArrayT, "*b"] | None = None
    checkpoint_debias_weight: at.Float[ArrayT, "*b"] | None = None

    # Auxiliary head targets (all share the success query token).
    # P(reward >= 0.5) target: 1.0 if reward >= 0.5, else 0.0. NaN = masked.
    checkpoint_target: at.Float[ArrayT, "*b"] | None = None
    # Garment type ID (0-3). -1 = masked (shouldn't happen for valid data).
    garment_type_id: at.Int[ArrayT, "*b"] | None = None
    # Completion target: step/total_steps for successful episodes. NaN = masked.
    completion_target: at.Float[ArrayT, "*b"] | None = None
    # Time-to-completion target: -steps_left/600 (success) or -1-steps_left/600 (failure).
    # Range [-2, 0]. NaN = masked (no success label available).
    ttc_target: at.Float[ArrayT, "*b"] | None = None
    # Per-sample multiplier applied to the TTC loss weight (1.0 for recent
    # RL datasets and BC/DAgger, 0.01 for older RL). Default None behaves
    # as 1.0 in train.py — preserves compat with tests and older code paths
    # that don't plumb this field through.
    ttc_weight_scale: at.Float[ArrayT, "*b"] | None = None
    # Diagnostic flag: True when ``ComputeAuxTargets`` used the TD-bootstrap
    # target (ttc_pred(t+30) − 30/600 for success / = ttc_pred for failure).
    # False when it fell back to the analytical formula or the sample is
    # NaN-masked. Logged as ``ttc_bootstrap_frac`` in train.py.
    ttc_used_bootstrap: at.Bool[ArrayT, "*b"] | None = None

    # Per-condition keypoint-distance targets (normalized by threshold, so ~1.0 = at boundary).
    # Width 21 = 5 (top_short) + 5 (top_long) + 4 (short_pant) + 7 (long_pant).
    # For a given sample only the slots for its garment type are valid; the mask is True
    # at those slots and False elsewhere. BC/DAgger/older samples have all-False masks.
    keypoint_distance_target: at.Float[ArrayT, "*b k"] | None = None
    keypoint_distance_mask: at.Bool[ArrayT, "*b k"] | None = None

    # Future targets at t + WM_FUTURE_HORIZON (chunk_len = 30 steps):
    # Q-function / world-modeling supervision for the post-FAST and post-flow heads.
    # Semantics identical to the current-step targets (mask logic, slot layout).
    success_future_target: at.Float[ArrayT, "*b"] | None = None
    completion_future_target: at.Float[ArrayT, "*b"] | None = None
    keypoint_distance_future_target: at.Float[ArrayT, "*b k"] | None = None
    keypoint_distance_future_mask: at.Bool[ArrayT, "*b k"] | None = None

    # Per-timestep action pad flag [action_horizon]: True where the action
    # horizon overshoots the episode and the action at that timestep is
    # LeRobot's padding (repeated last real action within the same episode,
    # never a leak from the next episode). Used by the flow-matching loss in
    # compute_detailed_loss to mask out padded timesteps so they don't
    # contribute gradient.
    action_is_pad: at.Bool[ArrayT, "*b ah"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """Convert dict to Observation."""
        # Convert uint8 images to float32 [-1, 1]. Build a new dict so we don't
        # mutate the caller's batch (reused/cached batches would skip conversion
        # on the second pass and silently feed already-normalized floats back in).
        images = {}
        for key, img in data["image"].items():
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            images[key] = img

        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            fast_tokens=data.get("fast_tokens"),
            fast_token_mask=data.get("fast_token_mask"),
            advantage=_squeeze_scalar(data.get("advantage")),
            success_target=_squeeze_scalar(data.get("success_target")),
            is_weight=_squeeze_scalar(data.get("is_weight")),
            success_debias_weight=_squeeze_scalar(data.get("success_debias_weight")),
            checkpoint_debias_weight=_squeeze_scalar(data.get("checkpoint_debias_weight")),
            checkpoint_target=_squeeze_scalar(data.get("checkpoint_target")),
            garment_type_id=_squeeze_scalar_int(data.get("garment_type_id")),
            completion_target=_squeeze_scalar(data.get("completion_target")),
            ttc_target=_squeeze_scalar(data.get("ttc_target")),
            ttc_weight_scale=_squeeze_scalar(data.get("ttc_weight_scale")),
            ttc_used_bootstrap=_squeeze_scalar(data.get("ttc_used_bootstrap")),
            keypoint_distance_target=data.get("keypoint_distance_target"),
            keypoint_distance_mask=data.get("keypoint_distance_mask"),
            success_future_target=_squeeze_scalar(data.get("success_future_target")),
            completion_future_target=_squeeze_scalar(data.get("completion_future_target")),
            keypoint_distance_future_target=data.get("keypoint_distance_future_target"),
            keypoint_distance_future_mask=data.get("keypoint_distance_future_mask"),
            action_is_pad=data.get("action_is_pad"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert Observation to dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


def _build_color_chain(cfg: TrainAugmentationConfig) -> augmax.Chain | None:
    """Build the per-camera augmax color chain from config.  Returns None if
    every component is disabled (callers skip the vmap entirely)."""
    ops = []
    if cfg.brightness > 0 or cfg.contrast > 0 or cfg.saturation > 0 or cfg.hue > 0:
        ops.append(augmax.ColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            saturation=cfg.saturation,
            hue=cfg.hue,
        ))
    if cfg.gauss_blur_prob > 0:
        ops.append(augmax.GaussianBlur(sigma=cfg.gauss_blur_sigma, p=cfg.gauss_blur_prob))
    if not ops:
        return None
    return augmax.Chain(*ops)


def _build_top_geo_chain(cfg: TrainAugmentationConfig, h: int, w: int) -> augmax.Chain | None:
    """Top-camera-only geometric chain (crop + rotate).  Returns None if both
    components are disabled."""
    ops = []
    if cfg.top_crop_frac < 1.0:
        ops.append(augmax.RandomCrop(int(w * cfg.top_crop_frac), int(h * cfg.top_crop_frac)))
        ops.append(augmax.Resize(w, h))
    if cfg.top_rotate_deg > 0:
        ops.append(augmax.Rotate((-cfg.top_rotate_deg, cfg.top_rotate_deg)))
    if not ops:
        return None
    return augmax.Chain(*ops)


def _zoom_translate_one(rng, image, zoom_range: float, translate_px: float):
    """Per-image zoom + translate around centre.  Static shapes; bilinear."""
    rng_s, rng_tx, rng_ty = jax.random.split(rng, 3)
    if zoom_range > 0:
        scale = jax.random.uniform(rng_s, (), minval=1.0 - zoom_range, maxval=1.0 + zoom_range)
    else:
        scale = jnp.array(1.0)
    if translate_px > 0:
        tx = jax.random.uniform(rng_tx, (), minval=-translate_px, maxval=translate_px)
        ty = jax.random.uniform(rng_ty, (), minval=-translate_px, maxval=translate_px)
    else:
        tx = jnp.array(0.0)
        ty = jnp.array(0.0)
    H, W, _ = image.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    # scale_and_translate convention: out_pos = scale * in_pos + translation
    # To zoom about centre by factor s and additionally shift by (ty, tx),
    # translation = (cy * (1 - s) + ty, cx * (1 - s) + tx).
    translation = jnp.stack([cy * (1.0 - scale) + ty, cx * (1.0 - scale) + tx])
    out = jax.image.scale_and_translate(
        image,
        image.shape,
        spatial_dims=(0, 1),
        scale=jnp.stack([scale, scale]),
        translation=translation,
        method="bilinear",
        antialias=True,
    )
    return out


def _line_kernel(K: int, angle, length):
    """JIT-friendly line-segment kernel of fixed window size K (static).

    `angle` (rad) and `length` (px) are traced scalars.  Returns a (K, K)
    float32 kernel, normalised to sum 1.
    """
    c = (K - 1) / 2.0
    ys, xs = jnp.mgrid[0:K, 0:K].astype(jnp.float32)
    dx = xs - c
    dy = ys - c
    u_x = jnp.cos(angle)
    u_y = jnp.sin(angle)
    perp = jnp.abs(-u_y * dx + u_x * dy)
    proj = jnp.abs(u_x * dx + u_y * dy)
    mask = (perp < 0.5) & (proj <= length / 2.0)
    kernel = mask.astype(jnp.float32)
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def _motion_blur_one(rng, image, max_len: int, prob: float):
    """Per-image motion blur: random angle, random length in [1, max_len]."""
    rng_len, rng_angle, rng_apply = jax.random.split(rng, 3)
    angle = jax.random.uniform(rng_angle, (), minval=0.0, maxval=jnp.pi)
    length = jax.random.uniform(rng_len, (), minval=1.0, maxval=float(max_len))
    kernel = _line_kernel(max_len, angle, length)
    # Depthwise conv: lhs (N=1, C=H*W*3? no — use NHWC layout)
    # Use jax.lax.conv_general_dilated with feature_group_count=C.
    img = image[None]                            # (1, H, W, C)
    C = image.shape[-1]
    # kernel needs shape (kH, kW, in_channels/groups, out_channels) for HWIO.
    # With groups=C and depthwise, we want (kH, kW, 1, C).
    k = jnp.broadcast_to(kernel[..., None, None], (max_len, max_len, 1, C))
    out = jax.lax.conv_general_dilated(
        img,
        k,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=C,
    )[0]
    apply = jax.random.bernoulli(rng_apply, prob)
    return jnp.where(apply, out, image)


def _jpeg_proxy(image_batch, prob: float, low_h: int, low_w: int, rng):
    """Batched bilinear downsample → upsample roundtrip with per-image apply mask.

    `low_h`, `low_w` are static.  Single `jax.image.resize` call each way over
    the whole batch.
    """
    B, H, W, C = image_batch.shape
    low = jax.image.resize(image_batch, (B, low_h, low_w, C), method="bilinear")
    hi = jax.image.resize(low, (B, H, W, C), method="bilinear")
    apply = jax.random.bernoulli(rng, prob, (B,))
    return jnp.where(apply[:, None, None, None], hi, image_batch)


def _cutout_one(rng, image, prob: float, max_area: float):
    """Single rectangular cutout filled with 0 (grey in [-1,1])."""
    rng_apply, rng_area, rng_aspect, rng_cy, rng_cx = jax.random.split(rng, 5)
    H, W, _ = image.shape
    apply = jax.random.bernoulli(rng_apply, prob)
    area_frac = jax.random.uniform(rng_area, (), minval=0.0, maxval=max_area)
    aspect = jax.random.uniform(rng_aspect, (), minval=0.3, maxval=3.0)
    area_px = area_frac * H * W
    h = jnp.sqrt(area_px * aspect)
    w = jnp.sqrt(area_px / aspect)
    cy = jax.random.uniform(rng_cy, (), minval=0.0, maxval=float(H))
    cx = jax.random.uniform(rng_cx, (), minval=0.0, maxval=float(W))
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    ys = jnp.arange(H)[:, None].astype(jnp.float32)
    xs = jnp.arange(W)[None, :].astype(jnp.float32)
    inside = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)  # (H, W)
    mask = inside[..., None] & apply
    # 0.5 in [0,1] space = grey; caller is operating in [0,1].
    return jnp.where(mask, jnp.array(0.5, image.dtype), image)


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    rotate_top_image: bool = False,
    train_aug: TrainAugmentationConfig | None = None,
) -> Observation:
    """Preprocess observations with config-driven image + state augmentation.

    When ``train=False`` no augmentation runs; only resize-on-load and the
    optional dataset-specific top-camera 180° flip apply (inference path).

    When ``train=True``, every component is gated by ``train_aug`` — flag-off
    components are erased at JIT trace time (Python ``if`` on a frozen
    dataclass field).  Each component pre-samples its randomness OUTSIDE
    ``jax.vmap`` and the vmap'd op uses only ``jnp.where`` for apply / skip.
    No ``lax.cond`` / ``lax.switch`` is allowed inside the vmap (it compiles
    to a radix-sort kernel that is catastrophically slow at training time).

    Color augs share their RNG across cameras unless
    ``train_aug.per_camera_color=True``.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    if train_aug is None:
        # Backward-compat fallback: scripts/tests calling without a config get
        # historical defaults (which include color jitter + top-cam geo + blur).
        train_aug = TrainAugmentationConfig()

    batch_shape = observation.state.shape[:-1]
    out_state = observation.state

    if train:
        # One top-level split per stream — keep this list aligned with the
        # consumers below.  Always split the same number of keys so the trace
        # is stable across flag combinations (cheap, ignored when unused).
        rngs = jax.random.split(rng, 16)
        (rng_color_shared, rng_color_percam,
         rng_gamma, rng_gain,
         rng_zoomtrans, rng_top_geo,
         rng_noise, rng_blur_extra,
         rng_motion, rng_jpeg,
         rng_cutout, rng_camdrop,
         rng_state, _r1, _r2, _r3) = tuple(rngs)
        batch_size = observation.state.shape[0]

        # State noise
        if train_aug.state_noise_std > 0:
            out_state = out_state + jax.random.normal(rng_state, out_state.shape) * train_aug.state_noise_std

        # Pre-build static chains (config fields are JIT-time constants).
        color_chain = _build_color_chain(train_aug)
        top_geo_chain = _build_top_geo_chain(train_aug, *image_resolution)
        do_zoom_trans = (train_aug.zoom_range > 0 or train_aug.translate_px > 0)
        do_motion_blur = (train_aug.motion_blur_max_len > 0 and train_aug.motion_blur_prob > 0)
        do_jpeg = (train_aug.jpeg_proxy_scale < 1.0 and train_aug.jpeg_proxy_prob > 0)
        do_cutout = (train_aug.cutout_prob > 0 and train_aug.cutout_max_area > 0)
        do_gamma = train_aug.gamma_range > 0
        do_gain = train_aug.per_channel_gain > 0
        do_noise = train_aug.additive_noise_std > 0
        do_camdrop = train_aug.cam_dropout_prob > 0

        # Per-camera color keys.  Shared mode: each camera gets the SAME base
        # per-sample key (identical color across cameras).  Per-camera mode:
        # each camera gets independent per-sample keys.
        num_cams = len(image_keys)
        if train_aug.per_camera_color:
            color_root_keys = jax.random.split(rng_color_percam, num_cams)
        else:
            color_root_keys = [rng_color_shared] * num_cams

        # Per-camera independent keys for everything else (intrinsics, noise,
        # blur, jpeg, cutout, motion, camdrop).
        zoomtrans_root_keys = jax.random.split(rng_zoomtrans, num_cams) if do_zoom_trans else [None] * num_cams
        # Per-camera independent RNGs for the top-geo (crop + rotate) chain
        # when it applies to all cameras. Top-only mode reuses the single
        # ``rng_top_geo`` for the top cam only (same as before).
        top_geo_all_cams = train_aug.apply_top_geo_to_all_cameras
        top_geo_root_keys = (
            jax.random.split(rng_top_geo, num_cams)
            if (top_geo_chain is not None and top_geo_all_cams)
            else [rng_top_geo] * num_cams
        )
        gamma_root_keys = jax.random.split(rng_gamma, num_cams) if do_gamma else [None] * num_cams
        gain_root_keys = jax.random.split(rng_gain, num_cams) if do_gain else [None] * num_cams
        noise_root_keys = jax.random.split(rng_noise, num_cams) if do_noise else [None] * num_cams
        motion_root_keys = jax.random.split(rng_motion, num_cams) if do_motion_blur else [None] * num_cams
        jpeg_root_keys = jax.random.split(rng_jpeg, num_cams) if do_jpeg else [None] * num_cams
        cutout_root_keys = jax.random.split(rng_cutout, num_cams) if do_cutout else [None] * num_cams
        camdrop_root_keys = jax.random.split(rng_camdrop, num_cams) if do_camdrop else [None] * num_cams

        # JPEG-proxy low-res shape — static.
        low_h = max(1, int(image_resolution[0] * train_aug.jpeg_proxy_scale))
        low_w = max(1, int(image_resolution[1] * train_aug.jpeg_proxy_scale))

    out_images = {}
    out_masks = {}

    for cam_idx, key in enumerate(image_keys):
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            image = image_tools.resize_with_pad(image, *image_resolution)

        # Dataset-specific: rotate top view by 180° for proper egocentric view.
        is_top = _is_top_key(key)
        if rotate_top_image and is_top:
            image = jnp.flip(image, axis=(1, 2))

        if train:
            # Convert [-1, 1] → [0, 1] for augs.
            image = image / 2.0 + 0.5

            # Top-camera geometric (crop + rotate). Fires on the top cam by
            # default; on all cameras when ``apply_top_geo_to_all_cameras=True``,
            # with per-camera independent RNG.
            if top_geo_chain is not None and (is_top or top_geo_all_cams):
                geo_keys = jax.random.split(top_geo_root_keys[cam_idx], batch_size)
                image = jax.vmap(top_geo_chain)(geo_keys, image)

            # Random zoom + translate on ALL cameras.
            if do_zoom_trans:
                keys_zt = jax.random.split(zoomtrans_root_keys[cam_idx], batch_size)
                image = jax.vmap(_zoom_translate_one, in_axes=(0, 0, None, None))(
                    keys_zt, image, train_aug.zoom_range, train_aug.translate_px,
                )

            # Color jitter / Gaussian blur (augmax).
            if color_chain is not None:
                ckeys = jax.random.split(color_root_keys[cam_idx], batch_size)
                image = jax.vmap(color_chain)(ckeys, image)

            # Gamma.
            if do_gamma:
                gamma = jax.random.uniform(
                    gamma_root_keys[cam_idx], (batch_size,),
                    minval=1.0 - train_aug.gamma_range,
                    maxval=1.0 + train_aug.gamma_range,
                )
                # pow over [0,1] is safe; clamp just in case augs pushed slightly negative.
                image = jnp.power(jnp.clip(image, 0.0, 1.0), gamma[:, None, None, None])

            # Per-channel gain.
            if do_gain:
                gains = 1.0 + jax.random.uniform(
                    gain_root_keys[cam_idx], (batch_size, 1, 1, 3),
                    minval=-train_aug.per_channel_gain,
                    maxval=train_aug.per_channel_gain,
                )
                image = image * gains

            # Additive Gaussian noise.
            if do_noise:
                noise = jax.random.normal(noise_root_keys[cam_idx], image.shape) * train_aug.additive_noise_std
                image = image + noise

            # Motion blur (top-cam only).
            if do_motion_blur and is_top:
                mkeys = jax.random.split(motion_root_keys[cam_idx], batch_size)
                image = jax.vmap(_motion_blur_one, in_axes=(0, 0, None, None))(
                    mkeys, image, train_aug.motion_blur_max_len, train_aug.motion_blur_prob,
                )

            # JPEG-proxy roundtrip.
            if do_jpeg:
                image = _jpeg_proxy(image, train_aug.jpeg_proxy_prob, low_h, low_w, jpeg_root_keys[cam_idx])

            # Cutout (random erasing).
            if do_cutout:
                ckeys = jax.random.split(cutout_root_keys[cam_idx], batch_size)
                image = jax.vmap(_cutout_one, in_axes=(0, 0, None, None))(
                    ckeys, image, train_aug.cutout_prob, train_aug.cutout_max_area,
                )

            # Safety clip then back to [-1, 1].
            image = jnp.clip(image, 0.0, 1.0)
            image = image * 2.0 - 1.0

        out_images[key] = image

    # Build masks (default ones), applying camera-dropout AFTER the per-image
    # ops so the model sees a zero image with mask=False.
    for cam_idx, key in enumerate(image_keys):
        if key not in observation.image_masks:
            mask = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            mask = jnp.asarray(observation.image_masks[key])
        out_masks[key] = mask

    if train and do_camdrop:
        for cam_idx, key in enumerate(image_keys):
            if _is_top_key(key):
                continue  # never drop the top camera
            drop = jax.random.bernoulli(camdrop_root_keys[cam_idx], train_aug.cam_dropout_prob, (batch_size,))
            # Zero in [-1,1] space → grey image.
            zero_img = jnp.zeros_like(out_images[key])
            out_images[key] = jnp.where(drop[:, None, None, None], zero_img, out_images[key])
            out_masks[key] = jnp.logical_and(out_masks[key], jnp.logical_not(drop))

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=out_state,
        fast_tokens=getattr(observation, 'fast_tokens', None),
        fast_token_mask=getattr(observation, 'fast_token_mask', None),
        advantage=getattr(observation, 'advantage', None),
        success_target=getattr(observation, 'success_target', None),
        is_weight=getattr(observation, 'is_weight', None),
        checkpoint_target=getattr(observation, 'checkpoint_target', None),
        garment_type_id=getattr(observation, 'garment_type_id', None),
        completion_target=getattr(observation, 'completion_target', None),
        ttc_target=getattr(observation, 'ttc_target', None),
        ttc_weight_scale=getattr(observation, 'ttc_weight_scale', None),
        ttc_used_bootstrap=getattr(observation, 'ttc_used_bootstrap', None),
        success_debias_weight=getattr(observation, 'success_debias_weight', None),
        checkpoint_debias_weight=getattr(observation, 'checkpoint_debias_weight', None),
        keypoint_distance_target=getattr(observation, 'keypoint_distance_target', None),
        keypoint_distance_mask=getattr(observation, 'keypoint_distance_mask', None),
        success_future_target=getattr(observation, 'success_future_target', None),
        completion_future_target=getattr(observation, 'completion_future_target', None),
        keypoint_distance_future_target=getattr(observation, 'keypoint_distance_future_target', None),
        keypoint_distance_future_mask=getattr(observation, 'keypoint_distance_future_mask', None),
        action_is_pad=getattr(observation, 'action_is_pad', None),
    )
