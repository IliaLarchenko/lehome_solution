"""Typed dataclasses for RL pipeline configuration.

Replaces raw dict access with typed, validated config objects.
Loads from YAML (configs/rl_pipeline_sim.yaml or configs/rl_pipeline_sim_to_real.yaml) with optional CLI overrides.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any

import yaml

from lehome_solution.models.train_aug_config import TrainAugmentationConfig


@dataclasses.dataclass
class AdvantageConfig:
    mode: str = "auto"  # auto | segment | gae
    gamma: float = 0.999
    gae_lambda: float = 0.99
    # Clipping range for advantage weighting: exp(clip(advantage, min, max)).
    # Controls prioritized sampling probabilities and IS weights.
    # (0, 0) = uniform sampling with IS weight 1.0 (advantage has no effect on weighting).
    weighting_clipping: tuple[float, float] = (-2.0, 2.0)
    # AWR temperature: global_std is multiplied by beta before normalizing advantages.
    # Higher beta → more uniform sampling (less aggressive prioritization).
    # beta=1 is standard normalization; beta=5 matches typical AWR literature values.
    beta: float = 1.0
    # When a dataset's sampling_share drops below this threshold, force segment-only
    # advantages (no GAE). 0.0 = never auto-convert. Init datasets with segment_only=True
    # always use segment regardless of this threshold.
    segment_only_threshold: float = 0.0
    # Dampening coefficient on the P_success transition term in the GAE_success delta:
    #   δ = r[t] + (γ·P_success[t+1] − P_success[t]) · success_alpha
    # Reduces how strongly the (noisy) value signal influences GAE while leaving the
    # reward r[t] at full weight. 0.0 => GAE becomes a pure reward-only signal.
    success_alpha: float = 0.5
    # Weight on GAE_completion, a reward-free GAE over the completion head:
    #   δ_c = (γ·C[t+1] − C[t]) · completion_alpha
    # Added to the blended advantage at full strength regardless of w (decoupled from
    # the GAE/segment blend). 0.0 disables completion shaping.
    completion_alpha: float = 0.5
    # Tail-frame value correction: linearly interpolate the EMA of P(success) over
    # the last K frames from its T-K-1 value to the true success label (1 or 0).
    # Fixes the "almost-success looks like failure" bias near episode end. 0 disables.
    value_tail_k: int = 30


@dataclasses.dataclass
class ExplorationNoiseConfig:
    """DART-style additive correlated noise for rollout collection.

    Per-chunk Bernoulli(prob) draw; on hit, the policy server adds
    ``scale * generate_correlated_noise(...)`` to the sampled action chunk in
    normalized action space. Noise is drawn from the same covariance that
    seeds the flow-matching loop, so perturbations look like plausible
    actions rather than i.i.d. white kicks.

    HARD INVARIANT: this is a ROLLOUT-ONLY mechanism. ``enabled=False`` is
    the default; even with ``enabled=True``, noise only fires when both
    ``prob > 0`` AND ``scale > 0``. Inference servers never read this config,
    so the noise cannot leak into evaluation runs.
    """
    enabled: bool = False
    prob: float = 0.0    # per-chunk probability of injecting noise (0..1)
    scale: float = 0.0   # multiplier on correlated-noise draw (normalized units)


@dataclasses.dataclass
class PrecisionBoostConfig:
    """Per-garment precision boost applied during advantage recomputation.

    Picks the top ``top_k`` fraction of successful episodes PER GARMENT
    (ranked by binding-condition margin on the final ``check_distances``)
    and adds ``boost`` to every frame's advantage. Garments with fewer than
    ``min_successes`` successful episodes in the recompute window are skipped.

    Set ``boost=0`` to disable.
    """
    boost: float = 0.0
    top_k: float = 0.20
    min_successes: int = 5


@dataclasses.dataclass
class AugmentationConfig:
    """Rollout augmentation config.  All values are deltas from defaults — set 0 to disable.

    Episode-level augmentations (applied once after env.reset):
        pattern_swap_p      — probability of swapping garment texture from pool
        color_remap_p       — probability of LAB-space color remap
        pos_offset_range    — meters, extra uniform(−v, +v) per xyz added to reset range
        rot_offset_range    — degrees, extra uniform(−v, +v) per euler angle
        scale_range         — e.g. 0.1 → scale ∈ [0.9, 1.1]; success thresholds auto-adjusted
        roughness_range     — delta from default roughness, uniform(−v, +v), clamped [0,1]
        camera_pos_jitter   — meters, uniform(−v, +v) per xyz on each camera
        camera_rot_jitter   — degrees, uniform(−v, +v) per euler angle on each camera
        arm_xy_shift        — meters, per-arm independent uniform(−v, +v) on base XY (physics-affecting,
                              skipped in visual_only replays; saved value reused for hard-mining replay)
        arm_rot_z_deg       — degrees, per-arm independent uniform(−v, +v) on base Z-axis rotation
                              (physics-affecting; same replay semantics as arm_xy_shift)
        table_uv_shift      — UV units, uniform(−v, +v) per s/t axis applied to table-cover material
                              via UsdTransform2d (visual only, no physics impact)
        table_uv_rot_deg    — degrees, UV-space rotation on the same UsdTransform2d
        camera_focal_jitter — fractional, e.g. 0.05 → focal ∈ [0.95×, 1.05×] base, per camera
        light_dome_rot_deg  — degrees, uniform(−v, +v) per Euler axis on /World/Light dome orientation

    Per-step augmentations (applied every render step):
        step_color_tint        — existing per-step scale+bias color randomization
        light_intensity_range  — delta from default 1200, uniform(−v, +v)
        light_color_range      — delta per channel from default (0.75, 0.75, 0.75)
        arm_color_range        — shades-of-orange jitter on both arms (single shared draw per tick).
                                 0 disables; >0 enables; range box is hard-coded in the fork's augmentor.
        light_color_temp_range — Kelvin half-width around 6500K, uniform(−v, +v); 0 disables.
    """
    # Episode-level
    pattern_swap_p: float = 0.0
    color_remap_p: float = 0.0
    pos_offset_range: float = 0.0
    rot_offset_range: float = 0.0
    scale_range: float = 0.0
    roughness_range: float = 0.0
    camera_pos_jitter: float = 0.0      # wrist cameras position jitter (meters)
    camera_rot_jitter: float = 0.0      # wrist cameras rotation jitter (degrees)
    top_camera_pos_jitter: float = 0.0  # top camera position jitter (meters), 0 = same as camera_pos_jitter
    top_camera_rot_jitter: float = 0.0  # top camera rotation jitter (degrees), 0 = same as camera_rot_jitter
    # Constant (non-random) offset added to the top camera pose on top of any
    # jitter. Both default to all zeros = no change (use upstream offset from
    # garment_bi_cfg_v2.py). Values are in the camera's BASE-LOCAL frame (the
    # right-robot base, rotated 180° around Z relative to world — so local +Y
    # points toward the robots in world, and local +X points opposite to world
    # +X). For the real-aligned framing (workspace centered, 10° tilt away from
    # robots), use approximately:
    #     top_camera_pos_offset:    [0.0, 0.25, 0.15]   # +Y_local → +25 cm toward robots, +Z_local → +15 cm up
    #     top_camera_rot_offset_deg: [-29.0, 0.0, 0.0]  # -29° around local +X = +29° around world +X
    #                                                   # → removes 19° forward tilt + adds 10° tilt away
    # NB: a non-trivial rotation offset can flip the rendered image orientation
    # relative to the legacy "raw is upside-down" assumption used by
    # dataset_writer / model `rotate_top_image=True`. Small rotations (≤ a few
    # tens of degrees) keep that convention; a 180° flip would not.
    top_camera_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    top_camera_rot_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Constant focal-length multiplier applied to the top camera only (1.0 =
    # no change). Widens / narrows the sim's HFOV without touching the wrist
    # cams. Consumed by the lehome-challenge fork's visual_augmentation engine.
    top_camera_focal_scale: float = 1.0
    arm_xy_shift: float = 0.0           # per-arm XY base shift (meters), physics-affecting → not visual_only
    arm_rot_z_deg: float = 0.0          # per-arm Z-axis base rotation (degrees), physics-affecting
    table_uv_shift: float = 0.0         # UV-space translation on table-cover texture (no physics)
    table_uv_rot_deg: float = 0.0       # UV-space rotation on table-cover texture (no physics)
    camera_focal_jitter: float = 0.0    # fractional focal length jitter per camera, e.g. 0.05 → ±5%
    light_dome_rot_deg: float = 0.0     # dome light Euler rotation jitter (degrees)
    # Per-step
    step_color_tint: bool = True
    light_intensity_range: float = 0.0
    light_color_range: float = 0.0
    arm_color_range: float = 0.0          # >0 enables per-step shades-of-orange arm color
    light_color_temp_range: float = 0.0   # Kelvin half-width around 6500K

    @property
    def enabled(self) -> bool:
        """True if any augmentation is active."""
        return (
            self.pattern_swap_p > 0
            or self.color_remap_p > 0
            or self.pos_offset_range > 0
            or self.rot_offset_range > 0
            or self.scale_range > 0
            or self.roughness_range > 0
            or self.camera_pos_jitter > 0
            or self.camera_rot_jitter > 0
            or self.top_camera_pos_jitter > 0
            or self.top_camera_rot_jitter > 0
            or any(abs(v) > 0 for v in self.top_camera_pos_offset)
            or any(abs(v) > 0 for v in self.top_camera_rot_offset_deg)
            or abs(self.top_camera_focal_scale - 1.0) > 1e-6
            or self.arm_xy_shift > 0
            or self.arm_rot_z_deg > 0
            or self.table_uv_shift > 0
            or self.table_uv_rot_deg > 0
            or self.camera_focal_jitter > 0
            or self.light_dome_rot_deg > 0
            or self.step_color_tint
            or self.light_intensity_range > 0
            or self.light_color_range > 0
            or self.arm_color_range > 0
            or self.light_color_temp_range > 0
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RolloutStrategyConfig:
    name: str = "random"
    # Per-strategy overrides (None = use pipeline-level defaults)
    num_garments: int | None = None
    num_episodes: int | None = None
    # Per-strategy augmentation overrides (raw dict, merged on top of pipeline defaults)
    # Only keys present in this dict override pipeline values; rest inherited.
    augmentation: dict | None = None
    # Curriculum params
    target_sr: float | None = None
    # Garment subset this strategy draws from: "seen" (default), "unseen", or
    # "all". Controls both the explicit-list pool (full / curriculum / replay /
    # hard_mining) and the randomized `random` strategy. Default is seen-only
    # so RL rollouts never leak held-out unseen garments into training data.
    subset: str = "seen"
    # Optional explicit garment pool for this strategy. When set, overrides
    # the `subset`-derived garment pool (used by random / full / curriculum).
    garment_list: list[str] | None = None
    # Hard mining params
    remove_on_success: bool = True  # remove NPZ from pool on success
    # Success replay params
    num_states: int = 10        # N: number of success states to sample
    replays_per_state: int = 5  # M: number of replay rollouts per sampled state
    # When True, sample success states uniformly per garment instance
    # (each garment gets equal probability regardless of pool size or SR).
    # When False (default), use FR-proportional sampling (harder garments
    # get more replays). Use True for real-format dataset collection where
    # balanced garment coverage matters more than SR-weighted curriculum.
    uniform_garment_sampling: bool = False
    # semi_success_replay: if True, restrict to dagger-origin states and route
    # the resulting rollout into the dagger dataset pool (trained like BC).
    dagger_only: bool = False

    def __post_init__(self):
        if self.subset not in ("seen", "unseen", "all"):
            raise ValueError(
                f"rollout strategy {self.name!r}: subset must be one of "
                f"'seen' | 'unseen' | 'all', got {self.subset!r}"
            )


@dataclasses.dataclass
class BCDatasetConfig:
    root: str = "data/lehome_challenge_merged/four_types_merged"
    repo_id: str = "lehome_bc"
    initial_sampling_share: float = 1.0
    decay_factor: float = 0.9
    min_sampling_share: float = 0.3
    # Fixed advantage value injected for BC samples (post-clipping scale).
    bc_advantage: float = 0.3
    # When False, BC samples write NaN for success/max_reward/dense_return so
    # they are masked out of the success/checkpoint/ttc losses (action + FAST +
    # completion + garment_type only).  Default True forces success=1 on BC.
    use_for_success_labels: bool = True


@dataclasses.dataclass
class DaggerDatasetConfig:
    decay_factor: float = 0.9
    min_sampling_share: float = 0.3
    # Fixed advantage value injected for dagger samples (treated like BC — uniform sampling).
    dagger_advantage: float = 0.3


@dataclasses.dataclass
class InferenceOptConfig:
    enabled: bool = False
    decay_factor: float = 0.99
    # When non-empty, eval_worker uses these fixed inference configs per garment
    # type instead of Thompson Sampling.  Takes precedence over the prior file
    # whether `enabled` is true or false.  Map: garment_type -> dict of
    # {actions_to_execute, k_execute, actions_to_keep, time_threshold_inpaint,
    # cfg_scale, noise_temperature, num_rollout_candidates}.  Missing fields
    # fall back to inference_optimization.DEFAULT_CONFIG.  num_steps is fixed
    # at 10 and not overridable here.
    per_garment_type_config: dict[str, dict] | None = None


@dataclasses.dataclass
class AuxLossesConfig:
    """Per-loss weights on the model's auxiliary training heads.

    The main flow-matching `action` loss is always at weight 1.0 and is not
    configurable here.  Each field below multiplies its corresponding per-sample
    loss before it is added to the total training loss.  None = keep the model's
    dataclass default (do not override).
    """
    fast: float | None = None            # FAST auxiliary action-token loss
    success: float | None = None         # BCE on P(success)
    checkpoint: float | None = None      # BCE on P(reward >= 0.5)
    ttc: float | None = None             # MSE on time-to-completion
    completion: float | None = None      # MSE on completion fraction (success-only)
    garment_type: float | None = None    # CE on 4-class garment type
    # Keypoint-distance head (Head 1): MSE on per-condition normalized distances.
    keypoint_distance: float | None = None
    # WM-FAST head (Head 2, training-only): three future-horizon losses.
    wm_fast_success: float | None = None
    wm_fast_completion: float | None = None
    wm_fast_keypoint: float | None = None
    # WM-flow head (Head 3): three future-horizon losses, weighted by (1 - t).
    wm_flow_success: float | None = None
    wm_flow_completion: float | None = None
    wm_flow_keypoint: float | None = None


def _validate_keys(raw: dict, dataclass_cls: type, context: str) -> None:
    """Raise ValueError if raw dict contains keys not in the dataclass."""
    known = set(dataclass_cls.__dataclass_fields__.keys())
    unknown = set(raw.keys()) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in {context}: {sorted(unknown)}. "
            f"Check for typos. Known keys: {sorted(known)}"
        )


_STRATEGY_KEYWORDS = ["semi_success_replay", "success_replay", "hard_mining", "curriculum", "random", "full"]


def _infer_strategy(root: str) -> str:
    """Infer rollout strategy from dataset root path name."""
    name = root.rstrip("/").split("/")[-1].lower()
    for kw in _STRATEGY_KEYWORDS:
        if kw in name:
            return kw
    return ""


@dataclasses.dataclass
class InitialRLDataset:
    root: str = ""
    sampling_share: float = 1.0
    repo_id: str = ""
    strategy: str = ""  # Rollout strategy; inferred from root name if empty
    segment_only: bool = False  # Force segment-only advantages (no GAE)


@dataclasses.dataclass
class InitialDaggerDataset:
    root: str = ""
    sampling_share: float = 1.0
    repo_id: str = ""


@dataclasses.dataclass
class BCRealDataset:
    """One real-robot teleop BC dataset.

    Goes through the real->sim per-sample rewrite path in MultiDataset:
    camera rename, 16:9->4:3 top letterbox, 180° pre-rotation, cubic resample
    of the 20-Hz action chunk to 30-Hz. See `real_data_transforms.py`.
    """
    root: str = ""
    repo_id: str = "lehome_real_bc"
    sampling_share: float = 1.0


@dataclasses.dataclass
class RLPipelineConfig:
    # Base training config name (from config.py)
    config_name: str = "pi_modified_bc_rl"
    exp_name: str = "bc_rl_v1"
    project_name: str = "LeHome"

    # Warm-up phase (first training iteration when starting fresh)
    warmup_steps: int = 3000

    # RL cycle
    num_iterations: int = 10
    steps_per_iteration: int = 1000

    # Rollout collection
    num_garments: int = 30
    num_episodes: int = 3
    num_workers: int = 6
    camera_width: int = 640
    camera_height: int = 480
    # TOP-cam render override (real BC dataset uses 720x1280 for right_front).
    # When set, the launcher applies these to top_camera only; wrists keep
    # the shared camera_width/camera_height.
    top_camera_width: int | None = None
    top_camera_height: int | None = None
    # Dataset image resolution. None = same as camera. Set smaller to downscale
    # at dataset-write time (cheaper dataset writes, smaller parquets). Sim still
    # renders at camera_width/height so policy quality is unaffected. Ignored
    # when dataset_format=real_bc (each camera keeps its native render shape).
    dataset_width: int | None = None
    dataset_height: int | None = None
    # Output dataset schema. "sim" (default) = EvalDatasetWriter (all aux fields,
    # 30 fps, single shape, radians). "real_bc" = RealFormatDatasetWriter
    # matching data/lehome_real/four_types_merged: 20 fps, per-camera shapes
    # (top 720x1280 real-canonical, wrist 480x640), garment via task_index,
    # no aux fields. Use real_bc when collecting sim success-replays for a
    # real-first training mix.
    dataset_format: str = "sim"
    # State/action units for dataset_format=real_bc: "radians" (default — sim
    # USD radians, unchanged) or "degrees" (real-robot degree-mode convention:
    # arm joints in degrees, gripper 0-100 — required when the dataset will be
    # mixed with real recordings). Motor-pct is not supported anywhere.
    dataset_units: str = "radians"
    max_retries: int = 0
    noise_temperature: float = 1.0
    # Best-of-N rollout sampling — policy-server default N (fallback for any
    # request without a per-garment/per-request override). When > 1, the policy
    # server expands each inference request into N candidates with independent
    # denoising noise and returns the one that maximizes the avg wm_flow joint
    # score ``0.5 * (s_cond*c_cond + s_uncond*c_uncond)``. N=1 disables.
    # Cost is ~N× action-expert forward per chunk; the VLM prefix is shared.
    # Default 3 matches the global default in
    # inference_optimization.DEFAULT_CONFIG (the mode across the four
    # per-garment-type configs).
    num_rollout_candidates: int = 3
    # Debug video: per-episode 3-camera grid + S/A/R/C overlay, for human inspection.
    # Not used in training. When False, all related work is skipped (no compose,
    # no overlay, no ffmpeg, no thread pool).  Rendered at a fixed downscaled
    # resolution (320x240 per camera) regardless of dataset image size.
    save_debug_video: bool = True

    # Rollout-time simulator augmentation (nested config).
    augmentation: AugmentationConfig = dataclasses.field(default_factory=AugmentationConfig)

    # Training-time image + state augmentation (nested config; defaults match
    # the pre-refactor hard-coded values in observation.preprocess_observation).
    # Propagated onto `model.train_aug` by `run_rl_pipeline._build_train_config`.
    train_augmentation: TrainAugmentationConfig = dataclasses.field(default_factory=TrainAugmentationConfig)

    # Dataset config
    bc_dataset: BCDatasetConfig = dataclasses.field(default_factory=BCDatasetConfig)
    dagger_dataset: DaggerDatasetConfig = dataclasses.field(default_factory=DaggerDatasetConfig)
    rl_decay_factor: float = 0.8
    rl_min_sampling_share: float = 0.3
    initial_rl_datasets: list[InitialRLDataset] = dataclasses.field(default_factory=list)
    initial_dagger_datasets: list[InitialDaggerDataset] = dataclasses.field(default_factory=list)

    # Real-robot teleop BC datasets.  Native real schema (degree units, 20 Hz,
    # real camera columns) — no unit conversion anywhere on the load path.
    # Fraction of every training batch drawn from these sources together is
    # controlled by `pct_real`.
    bc_real_datasets: list[BCRealDataset] = dataclasses.field(default_factory=list)
    # Fraction of each training batch drawn from real-robot sources.
    # 0 disables rebalancing (real sources mix in proportional to their
    # sampling_share).  Ignored when `bc_real_datasets` is empty.
    pct_real: float = 0.0

    # Rollout strategies
    rollout_strategies: list[RolloutStrategyConfig] | None = None

    # Advantage computation
    advantage: AdvantageConfig = dataclasses.field(default_factory=AdvantageConfig)

    # Per-garment precision boost (runs after advantage computation in
    # recompute_advantages.py; no-op when boost=0).
    precision_boost: PrecisionBoostConfig = dataclasses.field(default_factory=PrecisionBoostConfig)

    # Rollout-only DART-style exploration noise (no-op by default; never
    # propagated to the submission path).
    exploration_noise: ExplorationNoiseConfig = dataclasses.field(default_factory=ExplorationNoiseConfig)

    # Wandb
    wandb_enabled: bool = True

    # Model overrides
    use_advantage_embedding: bool = True  # Enable advantage embedding conditioning in model
    freeze_vision_backbone: bool = True  # Freeze SigLIP vision encoder weights

    # Attention backend. use_xsa=True auto-disables flash attention.
    use_flash_attention: bool = True
    use_xsa: bool = False

    # Training overrides
    batch_size: int | None = None
    save_interval: int = 500
    keep_period: int = 5000

    # HuggingFace Hub
    hf_model_repo: str | None = None
    hf_dataset_repo: str | None = None

    # Inference optimization
    inference_optimization: InferenceOptConfig = dataclasses.field(default_factory=InferenceOptConfig)

    # Auxiliary loss weights (None = inherit model default)
    aux_losses: AuxLossesConfig = dataclasses.field(default_factory=AuxLossesConfig)

    # Distributed settings (rollout_worker / trainer modes)
    checkpoint_poll_interval_s: int = 120  # how often worker checks for new checkpoint on HF

    # HF sync daemon
    hf_sync_enabled: bool = True  # enable background HF sync daemon

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "RLPipelineConfig":
        """Load config from YAML file with optional overrides."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        if overrides:
            for key, val in overrides.items():
                if val is not None:
                    raw[key] = val

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "RLPipelineConfig":
        # Keys that existed in older configs but have been migrated — silently ignore.
        _DEPRECATED_KEYS = {
            "garment_pattern_augmentation_p", "garment_color_augmentation_p",
        }

        # Parse nested configs with unknown-key validation
        bc_raw = raw.get("bc_dataset", {})
        _validate_keys(bc_raw, BCDatasetConfig, "bc_dataset")
        bc = BCDatasetConfig(**{k: v for k, v in bc_raw.items() if k in BCDatasetConfig.__dataclass_fields__})

        dagger_raw = raw.get("dagger_dataset", {})
        _validate_keys(dagger_raw, DaggerDatasetConfig, "dagger_dataset")
        dagger = DaggerDatasetConfig(**{k: v for k, v in dagger_raw.items() if k in DaggerDatasetConfig.__dataclass_fields__})

        # Advantage config (nested `advantage:` key)
        adv_raw = raw.get("advantage", {})
        _validate_keys(adv_raw, AdvantageConfig, "advantage")
        # Convert YAML list [min, max] to tuple for weighting_clipping
        if "weighting_clipping" in adv_raw and isinstance(adv_raw["weighting_clipping"], list):
            adv_raw["weighting_clipping"] = tuple(adv_raw["weighting_clipping"])
        advantage = AdvantageConfig(**{k: v for k, v in adv_raw.items() if k in AdvantageConfig.__dataclass_fields__})

        # Precision boost (nested `precision_boost:` key)
        pb_raw = raw.get("precision_boost", {})
        _validate_keys(pb_raw, PrecisionBoostConfig, "precision_boost")
        precision_boost = PrecisionBoostConfig(**{
            k: v for k, v in pb_raw.items() if k in PrecisionBoostConfig.__dataclass_fields__
        })

        # Exploration noise (nested `exploration_noise:` key)
        en_raw = raw.get("exploration_noise", {}) or {}
        _validate_keys(en_raw, ExplorationNoiseConfig, "exploration_noise")
        exploration_noise = ExplorationNoiseConfig(**{
            k: v for k, v in en_raw.items() if k in ExplorationNoiseConfig.__dataclass_fields__
        })

        # Augmentation config
        aug_raw = raw.get("augmentation", {})
        # Backward compat: old flat keys -> new nested config
        if "garment_pattern_augmentation_p" in raw and "pattern_swap_p" not in aug_raw:
            aug_raw.setdefault("pattern_swap_p", raw["garment_pattern_augmentation_p"])
        if "garment_color_augmentation_p" in raw and "color_remap_p" not in aug_raw:
            aug_raw.setdefault("color_remap_p", raw["garment_color_augmentation_p"])
        _validate_keys(aug_raw, AugmentationConfig, "augmentation")
        aug = AugmentationConfig(**{k: v for k, v in aug_raw.items() if k in AugmentationConfig.__dataclass_fields__})

        # Training-time augmentation config.
        train_aug_raw = raw.get("train_augmentation", {}) or {}
        # Backward compat: top-level `state_noise_std` used to live on RLPipelineConfig.
        # Migrate it into the nested config when present and not already overridden.
        if "state_noise_std" in raw and "state_noise_std" not in train_aug_raw:
            train_aug_raw["state_noise_std"] = raw["state_noise_std"]
        _validate_keys(train_aug_raw, TrainAugmentationConfig, "train_augmentation")
        train_aug = TrainAugmentationConfig(
            **{k: v for k, v in train_aug_raw.items() if k in TrainAugmentationConfig.__dataclass_fields__}
        )

        # Rollout strategies
        strats_raw = raw.get("rollout_strategies")
        strategies = None
        if strats_raw:
            parsed_strats = []
            for i, s in enumerate(strats_raw):
                s_aug_raw = s.pop("augmentation", None)
                s_aug = None
                if isinstance(s_aug_raw, dict):
                    # Store raw dict — only explicitly-set keys, merged with pipeline defaults at use time
                    _validate_keys(s_aug_raw, AugmentationConfig, f"rollout_strategies[{i}].augmentation")
                    s_aug = {k: v for k, v in s_aug_raw.items() if k in AugmentationConfig.__dataclass_fields__}
                # Backward compat: old flat keys on strategy
                elif "garment_pattern_augmentation_p" in s or "garment_color_augmentation_p" in s:
                    s_aug = {}
                    if "garment_pattern_augmentation_p" in s:
                        s_aug["pattern_swap_p"] = s.pop("garment_pattern_augmentation_p")
                    if "garment_color_augmentation_p" in s:
                        s_aug["color_remap_p"] = s.pop("garment_color_augmentation_p")
                _validate_keys(s, RolloutStrategyConfig, f"rollout_strategies[{i}]")
                strat_kwargs = {k: v for k, v in s.items() if k in RolloutStrategyConfig.__dataclass_fields__}
                strat_kwargs["augmentation"] = s_aug
                parsed_strats.append(RolloutStrategyConfig(**strat_kwargs))
            strategies = parsed_strats

        # Inference optimization
        infer_raw = raw.get("inference_optimization", {})
        _validate_keys(infer_raw, InferenceOptConfig, "inference_optimization")
        infer_opt = InferenceOptConfig(**{k: v for k, v in infer_raw.items() if k in InferenceOptConfig.__dataclass_fields__})

        # Aux-loss weights
        aux_raw = raw.get("aux_losses", {}) or {}
        _validate_keys(aux_raw, AuxLossesConfig, "aux_losses")
        aux_losses = AuxLossesConfig(**{k: v for k, v in aux_raw.items() if k in AuxLossesConfig.__dataclass_fields__})

        # Initial RL datasets
        init_rl_raw = raw.get("initial_rl_datasets", [])
        init_rl = [InitialRLDataset(**d) for d in init_rl_raw]

        # Initial dagger datasets
        init_dagger_raw = raw.get("initial_dagger_datasets", [])
        init_dagger = [InitialDaggerDataset(**d) for d in init_dagger_raw]

        # Real-robot BC datasets
        bc_real_raw = raw.get("bc_real_datasets", []) or []
        for d in bc_real_raw:
            _validate_keys(d, BCRealDataset, "bc_real_datasets[*]")
        bc_real = [BCRealDataset(**d) for d in bc_real_raw]

        # Top-level fields: validate that all raw keys are recognized
        nested_keys = {
            "bc_dataset", "dagger_dataset", "advantage", "precision_boost",
            "exploration_noise",
            "augmentation", "train_augmentation", "rollout_strategies",
            "inference_optimization", "initial_rl_datasets", "initial_dagger_datasets",
            "bc_real_datasets",
            "aux_losses",
        }
        # `state_noise_std` is migrated into `train_augmentation.state_noise_std`
        # above; allow it as a top-level key for backward compat only.
        _LEGACY_TOP_KEYS = {"state_noise_std"}
        top_fields = {f.name for f in dataclasses.fields(cls)} - nested_keys
        all_known_keys = top_fields | nested_keys | _DEPRECATED_KEYS | _LEGACY_TOP_KEYS
        unknown_top = set(raw.keys()) - all_known_keys
        if unknown_top:
            raise ValueError(
                f"Unknown keys in pipeline config: {sorted(unknown_top)}. "
                f"Check for typos. Known top-level keys: {sorted(top_fields | nested_keys)}"
            )

        top_kwargs = {}
        for key in top_fields:
            if key in raw:
                top_kwargs[key] = raw[key]

        return cls(
            **top_kwargs,
            bc_dataset=bc,
            dagger_dataset=dagger,
            advantage=advantage,
            precision_boost=precision_boost,
            exploration_noise=exploration_noise,
            augmentation=aug,
            train_augmentation=train_aug,
            rollout_strategies=strategies,
            inference_optimization=infer_opt,
            initial_rl_datasets=init_rl,
            initial_dagger_datasets=init_dagger,
            bc_real_datasets=bc_real,
            aux_losses=aux_losses,
        )

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if self.num_iterations < 1:
            errors.append("num_iterations must be >= 1")
        if self.steps_per_iteration < 1:
            errors.append("steps_per_iteration must be >= 1")
        if self.num_workers < 1:
            errors.append("num_workers must be >= 1")
        if self.bc_dataset.decay_factor <= 0 or self.bc_dataset.decay_factor > 1:
            errors.append("bc_dataset.decay_factor must be in (0, 1]")
        if self.dagger_dataset.decay_factor <= 0 or self.dagger_dataset.decay_factor > 1:
            errors.append("dagger_dataset.decay_factor must be in (0, 1]")
        if self.advantage.mode not in ("auto", "segment", "gae"):
            errors.append(f"advantage.mode must be auto|segment|gae, got {self.advantage.mode}")
        if self.dataset_format not in ("sim", "real_bc"):
            errors.append(f"dataset_format must be sim|real_bc, got {self.dataset_format}")
        if self.dataset_units not in ("radians", "degrees"):
            errors.append(f"dataset_units must be radians|degrees, got {self.dataset_units}")
        if (self.top_camera_width is None) != (self.top_camera_height is None):
            errors.append("top_camera_width and top_camera_height must be set together (or both None)")
        if self.advantage.gamma <= 0 or self.advantage.gamma > 1:
            errors.append("advantage.gamma must be in (0, 1]")
        if self.precision_boost.top_k < 0 or self.precision_boost.top_k > 1:
            errors.append("precision_boost.top_k must be in [0, 1]")
        if self.precision_boost.min_successes < 1:
            errors.append("precision_boost.min_successes must be >= 1")

        # Validate per_garment_type_config keys against PARAM_SPACES + GARMENT_TYPES.
        if self.inference_optimization.per_garment_type_config:
            try:
                from lehome_solution.constants import GARMENT_TYPES
                from lehome_solution.eval.inference_optimization import PARAM_SPACES
            except Exception as e:  # pragma: no cover - import guard
                errors.append(f"per_garment_type_config validation skipped (import failed: {e})")
            else:
                allowed_params = set(PARAM_SPACES.keys())
                for gt, cfg in self.inference_optimization.per_garment_type_config.items():
                    if gt not in GARMENT_TYPES:
                        errors.append(
                            f"per_garment_type_config: unknown garment_type '{gt}'. "
                            f"Allowed: {sorted(GARMENT_TYPES)}"
                        )
                        continue
                    if not isinstance(cfg, dict):
                        errors.append(f"per_garment_type_config[{gt}] must be a dict")
                        continue
                    unknown = set(cfg.keys()) - allowed_params
                    if unknown:
                        errors.append(
                            f"per_garment_type_config[{gt}]: unknown params {sorted(unknown)}. "
                            f"Allowed: {sorted(allowed_params)}"
                        )
        return errors


    def to_dict(self) -> dict:
        """Convert to dict."""
        return dataclasses.asdict(self)
