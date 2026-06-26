"""Training-time image-augmentation config.

Lives in its own module (no JAX / torch imports) so it can be imported by the
YAML loader (`lehome_solution.training.pipeline_config`) without triggering
JAX init.  Consumed by `preprocess_observation` in
`lehome_solution.models.observation` and stored on `PiModifiedConfig`.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TrainAugmentationConfig:
    """Image augmentations applied inside `compute_detailed_loss`.

    All ranges are symmetric (±value) unless otherwise noted.  Each block has
    its own enable knob — set the magnitude to 0 (or prob to 0) to disable.

    JIT contract: every field is read at trace time as a Python constant via
    `self.config.train_aug.<field>`.  All randomness is sampled OUTSIDE
    `jax.vmap`.  No `jax.lax.cond/switch` is used inside any vmap'd op
    (avoids the catastrophically slow radix-sort kernel that results otherwise).
    """

    # --- ColorJitter (augmax) -------------------------------------------------
    # All cameras share the same color RNG when `per_camera_color=False`
    # (matches the historical behaviour).  Set True for independent draws per
    # camera — real cameras have different sensors / white-balance.
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.0          # 0 = disabled (augmax range is [0, 0.5])
    per_camera_color: bool = False

    # --- Gamma (pointwise pow) ------------------------------------------------
    # gamma ~ U(1 - gamma_range, 1 + gamma_range); 0 disables.
    gamma_range: float = 0.0

    # --- Per-channel multiplicative gain (color-temperature surrogate) --------
    # gain_c ~ U(1 - r, 1 + r) independently per RGB channel; 0 disables.
    per_channel_gain: float = 0.0

    # --- Gaussian blur (augmax.GaussianBlur) ----------------------------------
    gauss_blur_sigma: float = 1.0   # sigma when applied
    gauss_blur_prob: float = 0.5    # 0 disables (always sharp)

    # --- Additive Gaussian noise (sensor noise) -------------------------------
    # std in [0,1] image units; e.g. 0.012 ≈ 3/255.  0 disables.
    additive_noise_std: float = 0.0

    # --- Random zoom + translation on ALL cameras (FOV / mount drift) ---------
    # scale ~ U(1 - zoom_range, 1 + zoom_range); 0 disables.
    # translate ~ U(-translate_px, +translate_px) on each spatial axis.
    zoom_range: float = 0.0
    translate_px: float = 0.0

    # --- Top-camera geometric (crop + rotate) ---------------------------------
    # `top_crop_frac=1.0` disables crop; `top_rotate_deg=0` disables rotate.
    # When ``apply_top_geo_to_all_cameras=True`` the same chain (per-camera
    # independent RNG) also fires on the wrist cameras — handy in setups
    # where the wrist cam mount drifts run-to-run and we want a small
    # spatial perturbation on all views without separate wrist knobs.
    top_crop_frac: float = 0.95
    top_rotate_deg: float = 5.0
    apply_top_geo_to_all_cameras: bool = False

    # --- Motion blur on top camera --------------------------------------------
    # Single fixed kernel size (must be static); 0 disables the whole block.
    # When >0, length is sampled per image in [1, motion_blur_max_len] at
    # random angle.  Applied with `motion_blur_prob` per image.
    motion_blur_max_len: int = 0
    motion_blur_prob: float = 0.0

    # --- JPEG-proxy: bilinear resize roundtrip --------------------------------
    # On hit, image is downsampled to int(H*scale) × int(W*scale) and resized
    # back to (H, W).  Captures the high-frequency loss component of JPEG.
    # `jpeg_proxy_scale=1.0` OR `jpeg_proxy_prob=0` disables.
    jpeg_proxy_scale: float = 1.0
    jpeg_proxy_prob: float = 0.0

    # --- Cutout / random erasing ----------------------------------------------
    # Single rectangle of random area, filled with 0 (in [-1,1] = grey).
    # area ~ U(0, cutout_max_area * H*W); aspect ~ U(0.3, 3).
    cutout_prob: float = 0.0
    cutout_max_area: float = 0.0

    # --- Whole-camera dropout (wrist cams only) -------------------------------
    # With per-sample probability `cam_dropout_prob`, zero the image AND set
    # `image_mask[cam] = False` so the model sees the existing dropout path.
    # Never applied to the top camera.
    cam_dropout_prob: float = 0.0

    # --- State dropout --------------------------------------------------------
    # With per-sample probability `state_dropout_prob`, mask state tokens out of
    # the attention matrix (both rows and columns zeroed) so the model cannot
    # attend to or from them.  Positional encoding / ar_mask structure preserved.
    state_dropout_prob: float = 0.0

    # --- State noise ----------------------------------------------------------
    # Additive Gaussian on robot state (in normalised units).  Was a top-level
    # PiModifiedConfig field; moved here for visibility.
    state_noise_std: float = 0.0

    # --- Calibration scale noise ----------------------------------------------
    # Per-sample, per-joint multiplicative noise simulating calibration
    # differences across robot setups.  scale ~ Uniform(1-v, 1+v) where
    # v = calibration_scale_noise.  Applied to BOTH state and actions
    # (same scale per joint, shared across time steps in the action chunk).
    calibration_scale_noise: float = 0.0

    @property
    def enabled(self) -> bool:
        """True if anything in this config will run at train time."""
        return (
            self.brightness > 0
            or self.contrast > 0
            or self.saturation > 0
            or self.hue > 0
            or self.gamma_range > 0
            or self.per_channel_gain > 0
            or self.gauss_blur_prob > 0
            or self.additive_noise_std > 0
            or self.zoom_range > 0
            or self.translate_px > 0
            or self.top_crop_frac < 1.0
            or self.top_rotate_deg > 0
            or (self.motion_blur_max_len > 0 and self.motion_blur_prob > 0)
            or (self.jpeg_proxy_scale < 1.0 and self.jpeg_proxy_prob > 0)
            or self.cutout_prob > 0
            or self.cam_dropout_prob > 0
            or self.state_noise_std > 0
            or self.calibration_scale_noise > 0
        )
