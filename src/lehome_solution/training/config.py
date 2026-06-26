"""Training configuration for LeHome challenge.

Reference: https://github.com/Physical-Intelligence/openpi
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, List, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import optax
from typing_extensions import override
import tyro

# Import from OpenPI
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
from openpi.training.config import AssetsConfig
import openpi.transforms as _transforms

# Import from LeHome custom modules
from lehome_solution.models import pi_modified_config
from lehome_solution.models.train_aug_config import TrainAugmentationConfig
from lehome_solution.training.data_loader import DaggerSamplingConfig, InitialStuckConfig, SimGarmentSkipConfig
from lehome_solution.policies import lehome_policy
from lehome_solution.shared import normalize as _normalize
from lehome_solution.training import weight_loaders
from lehome_solution import transforms as lehome_transforms
from lehome_solution.constants import ACTION_HORIZON, ACTION_DIM

ModelType: TypeAlias = _model.ModelType


_AUX_HEAD_NAMES: tuple[str, ...] = (
    "success_head",
    "checkpoint_head",
    "garment_type_head",
    "completion_head",
    "ttc_head",
    # New aux heads (keypoint + world-modeling / Q).
    "keypoint_distance_head",
    "wm_fast_success_head",
    "wm_fast_completion_head",
    "wm_fast_keypoint_head",
    "wm_flow_success_head",
    "wm_flow_completion_head",
    "wm_flow_keypoint_head",
)


def aux_head_weight_decay_mask(params: Any) -> Any:
    """Boolean PyTree matching ``params``: True on aux-head kernel/bias leaves.

    Used as ``weight_decay_mask`` for ``AdamWWithAuxDecay`` so the extra aux-head
    decay applies only to the aux prediction heads listed in ``_AUX_HEAD_NAMES``
    (success / checkpoint / garment_type / completion / ttc plus the keypoint and
    world-modeling wm_fast_* / wm_flow_* heads) — replacing the previous explicit
    ``head_l2`` loss term.
    """
    import jax

    def _path_parts(path):
        parts = []
        for p in path:
            key = getattr(p, "key", None)
            if key is None:
                key = getattr(p, "name", None)
            if key is None:
                key = getattr(p, "idx", None)
            parts.append(str(key) if key is not None else str(p))
        return parts

    def classify(path, _leaf):
        parts = _path_parts(path)
        return any(name in parts for name in _AUX_HEAD_NAMES)

    return jax.tree_util.tree_map_with_path(classify, params)


@dataclasses.dataclass(frozen=True)
class AdamWWithAuxDecay(_optimizer.OptimizerConfig):
    """AdamW with an extra weight-decay channel applied only to aux-head params.

    ``weight_decay`` is the uniform baseline (matches openpi's default 1e-10 —
    effectively zero). ``aux_head_weight_decay`` is added on top of that baseline
    for params where ``weight_decay_mask`` is True, which is produced by
    ``aux_head_weight_decay_mask`` and covers every aux head in
    ``_AUX_HEAD_NAMES`` (success/checkpoint/garment_type/completion/ttc plus the
    keypoint + wm_fast_*/wm_flow_* heads) — kernels + biases. This replaces the
    previous ``head_l2`` loss term and prevents aux-head sigmoid saturation under
    class-imbalanced BCE signals.
    """

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10          # openpi default (avoids 0.0 OOM edge case)
    aux_head_weight_decay: float = 0.001  # extra decay on aux-head params (replaces head_l2)
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr,
        weight_decay_mask=None,
    ) -> optax.GradientTransformation:
        chain: list[optax.GradientTransformation] = [
            optax.clip_by_global_norm(self.clip_gradient_norm),
            optax.scale_by_adam(b1=self.b1, b2=self.b2, eps=self.eps),
            optax.add_decayed_weights(self.weight_decay),
        ]
        if weight_decay_mask is not None and self.aux_head_weight_decay > 0:
            chain.append(
                optax.add_decayed_weights(
                    self.aux_head_weight_decay, mask=weight_decay_mask,
                )
            )
        chain.append(optax.scale_by_learning_rate(lr))
        return optax.chain(*chain)


Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class DataSourceSpec:
    """CLI-friendly spec for one data source in a multi-dataset config."""
    root: str = ""
    repo_id: str = "lehome_data"
    sampling_share: float = 1.0
    is_bc: bool = False
    is_dagger: bool = False  # DAgger recovery data: fixed advantage, uniform sampling
    # Real-robot source: real schema (degree units, native 20-Hz cadence, real
    # camera column names). Data flows through with NO unit conversion — only
    # the per-source ``speed_factor`` time-axis resample. Skips the BC/DAgger
    # field-forcing block (no advantage, no forced success) and receives
    # `use_for_success_labels=False` automatically (real data has no failure
    # signal). Paired with a real-camera RepackTransform mapping in the
    # factory (see `raw_real_camera_mapping`).
    is_real: bool = False
    # Bucket label for batch-mix control via ``bucket_shares`` (declared on
    # ``LeRobotLeHomeDataConfig``). Sources within the same bucket split that
    # bucket's allocation proportional to their natural length. Empty string
    # disables bucket-rebalance for this source.
    bucket: str = ""
    # Canonical garment-name override. Used when the parquet ``task`` column
    # doesn't carry one of ``Pant_Long`` / ``Pant_Short`` / ``Top_Long`` /
    # ``Top_Short`` (e.g. home-real teleop where every row says "Fold the
    # Garment" and the garment type lives in the folder name).
    garment_type_name: str = ""
    # Time-axis speedup for real sources. >1.0 speeds up the trajectory
    # (larger per-step Δaction); <1.0 slows it down.
    speed_factor: float = 1.0
    # DAgger-only: separate speed factors for human-leader vs auto-policy
    # frames within a DAgger source. ``is_dagger=True`` + ``is_real=True``
    # triggers per-chunk selection based on the t=0 frame's task_is_policy.
    dagger_human_speed_factor: float = 2.0
    dagger_auto_speed_factor: float = 2.0
    success_rates_file: str | None = None  # FR-proportional sampling for BC/dagger
    # When False, samples from this source are NaN-masked out of the success,
    # checkpoint, and ttc losses (they still contribute to action + FAST +
    # completion + garment_type).  Typically set False for BC/DAgger/golden
    # sources when forcing `success=1` on them creates a sharpness shortcut.
    # Always False for real sources (forced inside the factory).
    use_for_success_labels: bool = True


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False
    # If true, will use per-timestamp normalization for actions
    use_per_timestamp_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence.
    action_sequence_keys: Sequence[str] = ("actions",)

    # Root directory for LeHome dataset.
    lehome_dataset_root: str | None = None

    # Episodes index to use for training
    episodes_index: List[int] | None = None

    # Multi-source dataset configs (set by factory from DataSourceSpec list).
    # When set, create_lehome_dataset() builds a MultiDataset instead of a single LeRobotDataset.
    multi_sources: list | None = None

    # Fraction of each training batch drawn from real-robot teleop sources
    # (those with is_real=True in multi_sources).  0 = no rebalance (real
    # sources contribute proportional to their `sampling_share`).  Applied
    # inside MultiDataset by scaling the per-source effective length.
    pct_real: float = 0.0
    # Optional batch-mix control: ``{bucket_name: target_share}``. When set,
    # every multi-source must carry a non-empty ``bucket`` label and shares
    # must sum to 1.0. Sources within a bucket split that bucket's allocation
    # proportional to their natural length. Takes precedence over ``pct_real``.
    bucket_shares: dict[str, float] | None = None
    # Per-frame sampling weights for DAgger sources (``is_dagger=True`` +
    # ``is_real=True``). See DaggerSamplingConfig.
    dagger_sampling: Any = None
    # Frame-level skip rule for sim sources of a specific garment type
    # (e.g. sim Pant_Long first-half — execution differs from real). See
    # SimGarmentSkipConfig.
    sim_garment_skip: Any = None
    initial_stuck: Any = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for LeHome."""

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        h, w = getattr(model_config, "image_resolution", (224, 224))
        inputs = [_transforms.ResizeImages(h, w)]
        return _transforms.Group(inputs=inputs)


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=False,  # Always use z-score normalization for LeHome
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LeRobotLeHomeDataConfig(DataConfigFactory):
    """Data configuration for LeHome dataset."""

    action_sequence_keys: Sequence[str] = ("action",)
    use_delta_joint_actions: bool = False

    # FAST auxiliary tokenization (only for PiModified with use_fast_auxiliary)
    use_fast_tokenization: bool = False

    # Override the dataset root on the CLI without touching the suppressed base_config.
    # When set, takes precedence over base_config.lehome_dataset_root.
    lehome_dataset_root: str | None = None

    # Enable advantage processing (clipping + embedding conditioning + prioritized sampling).
    # Set to True when RL datasets with computed advantages are present.
    use_advantages: bool = False

    # Clipping range (min, max) for advantage weighting: exp(clip(advantage, min, max)).
    # Controls prioritized sampling and IS weights. (0, 0) = uniform sampling.
    advantage_weighting_clipping: tuple[float, float] = (-2.0, 2.0)

    # Multi-dataset sources. When set, a MultiDataset is built at training time
    # instead of a single LeRobotDataset, eliminating the merge step.
    data_sources: tuple[DataSourceSpec, ...] | None = None
    # Advantage value injected for BC samples (post-clipping scale).
    bc_advantage: float = 0.3
    # Advantage value injected for dagger samples (treated like BC — uniform sampling).
    dagger_advantage: float = 0.3
    # Fraction of each training batch drawn from real-robot sources. 0 disables
    # rebalancing. Propagated onto DataConfig.pct_real.
    pct_real: float = 0.0
    # Real-data mode: change the RepackTransform mapping to consume the
    # native real LeRobot column names (`right_front` / `left_wrist` /
    # `right_wrist`) instead of the sim ones (`top_rgb` / `left_rgb` /
    # `right_rgb`). Model-side keys (`top_rgb` / `left_rgb` / `right_rgb`)
    # are unchanged, so this is a data-side rename only.
    raw_real_camera_mapping: bool = False
    # Optional batch-mix control: ``{bucket_name: target_share}``. Propagated
    # onto ``DataConfig.bucket_shares`` which ``MultiDataset`` consumes.
    bucket_shares: dict[str, float] | None = None
    # DAgger sampling config. Propagated onto ``DataConfig.dagger_sampling``
    # which ``MultiDataset`` uses to derive per-frame weights for DAgger
    # sources.
    dagger_sampling: Any = None
    # Sim-source per-garment first-half skip rule. Propagated onto
    # ``DataConfig.sim_garment_skip``.
    sim_garment_skip: Any = None
    initial_stuck: Any = None
    # Label smoothing for success and checkpoint head targets (0 = disabled).
    # success_target = success * (1 - alpha) + garment_avg_success * alpha
    # checkpoint_target = checkpoint * (1 - alpha) + garment_avg_checkpoint * alpha
    success_label_smoothing: float = 0.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transforms for LeHome observations.
        # Model-side keys (`top_rgb`/`left_rgb`/`right_rgb`) are unchanged;
        # the SOURCE columns differ for raw-real vs sim/translated-real.
        if self.raw_real_camera_mapping:
            top_src = "observation.images.right_front"
            left_src = "observation.images.left_wrist"
            right_src = "observation.images.right_wrist"
        else:
            top_src = "observation.images.top_rgb"
            left_src = "observation.images.left_rgb"
            right_src = "observation.images.right_rgb"
        repack_mapping = {
            "observation/top_rgb": top_src,
            "observation/left_rgb": left_src,
            "observation/right_rgb": right_src,
            "observation/state": "observation.state",
            "actions": "action",
            # Per-timestep pad flag emitted by LeRobotDataset's delta_timestamps
            # handler: True where the action horizon overshoots the episode and
            # the action was padded (repeated last real action). Used by the
            # training loss to exclude padded timesteps.
            "action_is_pad": "action_is_pad",
            "episode_index": "episode_index",
            "index": "index",
        }
        # Pass raw advantage through the repack stage when advantage processing is enabled
        if self.use_advantages:
            repack_mapping["advantage"] = "advantage"

        # Pass value + success columns through for success head training
        use_success_head = getattr(model_config, 'use_success_head', False)
        if use_success_head:
            repack_mapping["success_pred"] = "success_pred"
            repack_mapping["success"] = "success"
            repack_mapping["is_weight"] = "is_weight"
            # Per-garment debias multipliers (success_debias also carries the
            # success-tail boost for the last 30 frames of successful episodes).
            repack_mapping["success_debias_weight"] = "success_debias_weight"
            repack_mapping["checkpoint_debias_weight"] = "checkpoint_debias_weight"
            # Aux target inputs (injected by data loader, consumed by ComputeAuxTargets)
            repack_mapping["garment_type_id"] = "garment_type_id"
            repack_mapping["completion_frac"] = "completion_frac"
            repack_mapping["ep_len"] = "ep_len"
            repack_mapping["dense_return"] = "dense_return"
            # max_reward is consumed by ComputeAuxTargets to derive checkpoint_target.
            # Without it the target collapses to the binary success label and the
            # checkpoint head becomes a clone of the success head.
            repack_mapping["max_reward"] = "max_reward"
            if self.success_label_smoothing > 0:
                repack_mapping["garment_avg_success"] = "garment_avg_success"
                repack_mapping["garment_avg_checkpoint"] = "garment_avg_checkpoint"
            # Keypoint-distance + future-horizon inputs (consumed by
            # ComputeKeypointDistanceTargets + ComputeFutureAuxTargets).
            # Safe to pass through always: NaN / missing columns propagate
            # cleanly to all-masked targets downstream.
            repack_mapping["check_distances"] = "check_distances"
            repack_mapping["check_distances_future"] = "check_distances_future"
            repack_mapping["success_future"] = "success_future"
            repack_mapping["completion_future"] = "completion_future"
            # Past ttc_pred at t + 30 (bootstrap anchor for ComputeAuxTargets).
            # NaN when the column is absent from the source parquet or the
            # source is older than the last-N RL tail → analytical fallback.
            repack_mapping["ttc_pred_future"] = "ttc_pred_future"
            # Per-sample multiplier on the TTC loss weight (1.0 for last-N
            # RL + BC/DAgger, 0.01 for older RL samples). Applied in train.py.
            repack_mapping["ttc_weight_scale"] = "ttc_weight_scale"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        # Prepare data for policy training
        data_transforms = _transforms.Group(
            inputs=[lehome_policy.LeHomeInputs(model_type=model_config.model_type)],
            outputs=[lehome_policy.LeHomeOutputs()],
        )

        # Delta action transforms — all 12 dims are delta
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(12)
        else:
            delta_action_mask = _transforms.make_bool_mask(-12)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (subtask state, task ID, padding)
        model_transforms = ModelTransformFactory()(model_config)
        
        # ComputeSuccessTarget and ComputeAuxTargets must run BEFORE
        # AdvantageToWeight (uses raw advantage). Gated on use_success_head
        # ONLY — without use_advantages, ComputeSuccessTarget short-circuits
        # on missing advantage (success_target stays None) and
        # ComputeAuxTargets still emits garment_type_id + completion_target
        # (the fields that DON'T depend on advantage). This lets the
        # raw-real config train the completion + garment_type heads without
        # any advantage machinery.
        if use_success_head:
            model_transforms = model_transforms.push(
                inputs=[
                    lehome_transforms.ComputeSuccessTarget(
                        label_smoothing=self.success_label_smoothing,
                    ),
                    lehome_transforms.ComputeAuxTargets(
                        label_smoothing=self.success_label_smoothing,
                    ),
                    # Keypoint-distance targets + future scalar Q-targets. Safe to
                    # run always: transforms emit NaN/all-False masks when the
                    # raw inputs are absent (BC / DAgger / older datasets), so
                    # downstream losses are cleanly zeroed via isfinite masks.
                    lehome_transforms.ComputeKeypointDistanceTargets(),
                    lehome_transforms.ComputeFutureAuxTargets(),
                ],
            )
            logging.info(
                "ComputeSuccessTarget + ComputeAuxTargets + keypoint/future targets enabled"
                + (f" (label_smoothing={self.success_label_smoothing})" if self.success_label_smoothing > 0 else "")
            )

        # Advantage clipping for embedding conditioning
        if self.use_advantages:
            wc = self.advantage_weighting_clipping
            model_transforms = model_transforms.push(
                inputs=[lehome_transforms.AdvantageToWeight(
                    clip_min=wc[0],
                    clip_max=wc[1],
                )],
            )
            logging.info(f"AdvantageToWeight enabled: clipping=({wc[0]}, {wc[1]})")

        # FAST tokenization (if enabled for PiModified)
        if self.use_fast_tokenization and hasattr(model_config, 'use_fast_auxiliary') and model_config.use_fast_auxiliary:
            asset_id = self.assets.asset_id or self.repo_id
            # Respect AssetsConfig.assets_dir override (same as norm_stats lookup)
            effective_assets_dir = epath.Path(self.assets.assets_dir or assets_dirs)
            tokenizer_path = effective_assets_dir / asset_id / "fast_tokenizer"

            # Get base config to access norm_stats
            base_config = self.create_base_config(assets_dirs, model_config)

            # Only add transform if tokenizer directory exists
            if tokenizer_path.exists():
                model_transforms = model_transforms.push(
                    inputs=[lehome_transforms.TokenizeFASTActions(
                        tokenizer_path=str(tokenizer_path),
                        encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                        max_fast_tokens=model_config.max_fast_tokens,
                        norm_stats=base_config.norm_stats,
                        use_per_timestamp=base_config.use_per_timestamp_norm,
                    )],
                )
            else:
                logging.warning(
                    f"FAST tokenizer not found at {tokenizer_path}. "
                    "FAST auxiliary training will be disabled (inference mode)."
                )

        base = self.create_base_config(assets_dirs, model_config)
        extra = {}
        if self.lehome_dataset_root is not None:
            extra["lehome_dataset_root"] = self.lehome_dataset_root

        # Build multi_sources from data_sources specs
        if self.data_sources is not None:
            from lehome_solution.training.data_loader import DataSourceConfig

            multi_sources = []
            for spec in self.data_sources:
                bc_advantage_raw = None
                use_prioritized = True
                if spec.is_bc and self.use_advantages:
                    bc_advantage_raw = self.bc_advantage
                    use_prioritized = False  # BC uses FR-proportional or uniform
                if spec.is_dagger and self.use_advantages:
                    bc_advantage_raw = self.dagger_advantage
                    use_prioritized = False  # Dagger treated like BC — fixed advantage, uniform sampling
                if spec.is_real:
                    use_prioritized = False  # Real: no advantage column, no FR data — uniform sampling
                multi_sources.append(DataSourceConfig(
                    repo_id=spec.repo_id,
                    root=spec.root,
                    sampling_share=spec.sampling_share,
                    is_bc=spec.is_bc,
                    is_dagger=spec.is_dagger,
                    is_real=spec.is_real,
                    bucket=spec.bucket,
                    garment_type_name=spec.garment_type_name,
                    speed_factor=spec.speed_factor,
                    dagger_human_speed_factor=spec.dagger_human_speed_factor,
                    dagger_auto_speed_factor=spec.dagger_auto_speed_factor,
                    bc_advantage_raw=bc_advantage_raw,
                    use_prioritized_sampling=use_prioritized,
                    success_rates_file=spec.success_rates_file,
                    advantage_weighting_clipping=self.advantage_weighting_clipping,
                    # Real data has no failure signal — always NaN-mask out of
                    # the success/checkpoint/ttc losses regardless of spec value.
                    use_for_success_labels=(False if spec.is_real else spec.use_for_success_labels),
                ))
            extra["multi_sources"] = multi_sources
            extra["pct_real"] = float(self.pct_real)
            if self.bucket_shares:
                extra["bucket_shares"] = dict(self.bucket_shares)
            if self.dagger_sampling is not None:
                extra["dagger_sampling"] = self.dagger_sampling
            if self.sim_garment_skip is not None:
                extra["sim_garment_skip"] = self.sim_garment_skip
            if self.initial_stuck is not None:
                extra["initial_stuck"] = self.initial_stuck
            logging.info(
                f"MultiDataset: {len(multi_sources)} sources, "
                f"bc_advantage={self.bc_advantage}, dagger_advantage={self.dagger_advantage}, "
                f"pct_real={self.pct_real}"
            )

        return dataclasses.replace(
            base,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            **extra,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "LeHome"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config (PiModified only for LeHome).
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi_modified_config.PiModifiedConfig)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Note: PyTorch support removed - JAX only

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=LeRobotLeHomeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int | None = None
    # Global batch size.
    batch_size: int = 512
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # FSDP configuration for model sharding across devices.
    fsdp_devices: int = 1

    # Number of flow matching samples per training step
    num_flow_samples: int = 1

    # Attention backend selection. When both are False the vanilla gemma
    # attention is used. ``use_xsa=True`` auto-disables flash attention because
    # the two patches replace the same class.
    use_flash_attention: bool = True
    use_xsa: bool = False

    # Optional HuggingFace model repo for periodic auto-upload from `train.py`.
    # When set, each `save_state` triggers `upload_checkpoint_to_hf` for the
    # just-saved step. Numbered uploads gated by `keep_period`; `latest/` is
    # updated every cycle. Set to None (default) to disable auto-upload.
    hf_model_repo: str | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        freeze = self.freeze_filter
        # Also freeze vision backbone weights when model config says so
        if getattr(self.model, 'freeze_vision_backbone', False):
            freeze_head = getattr(self.model, 'freeze_vision_head', True)
            if not freeze_head:
                # Freeze backbone but keep head (projection) trainable
                vision_filter = nnx.All(
                    nnx_utils.PathRegex(".*PaliGemma/img.*"),
                    nnx.Not(nnx_utils.PathRegex(".*PaliGemma/img/head.*")),
                )
            else:
                vision_filter = nnx_utils.PathRegex(".*PaliGemma/img.*")
            freeze = nnx.Any(freeze, vision_filter)
        # Freeze Gemma embedding table when using dedicated state embedding
        # (the 257K-entry table is unused but can't be removed from the submodule)
        if getattr(self.model, 'use_state_embedding', False):
            freeze = nnx.Any(freeze, nnx_utils.PathRegex(".*embedder/input_embedding.*"))
        return nnx.All(nnx.Param, nnx.Not(freeze))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Shared base model config — all LeHome configs derive from this via dataclasses.replace()
_BASE_MODEL_CONFIG = pi_modified_config.PiModifiedConfig(
    action_horizon=ACTION_HORIZON,
    action_dim=ACTION_DIM,
    use_correlated_noise=True,
    correlation_beta=0.5,
    use_fast_auxiliary=True,
    fast_loss_weight=0.05,
    fast_encoded_dims="0:12",
    fast_vocab_size=1024,
    max_fast_tokens=50,
    use_kv_transform=True,
    use_knowledge_insulation=False,
    freeze_vision_backbone=True,
    rotate_top_image=True,
    use_advantage_embedding=True,
    neutral_prob_at_zero=0.5,
    neutral_prob_at_max=0.1,
    max_adv_for_neutral=2.0,
    use_success_head=True,
    success_loss_weight=0.1,
    freeze_vision_head=False,
    use_state_embedding=True,
)

_BASE_LR_SCHEDULE = _optimizer.CosineDecaySchedule(
    warmup_steps=1000,
    peak_lr=1e-4,
    decay_steps=100_000,
    decay_lr=1e-5,
)


# LeHome Training Configurations.
#
# Registry entries are deliberately THIN: they carry the inference contract —
# the model architecture / preprocessing flags + asset_id that
# `create_trained_policy` / `serve.py` need to rebuild a policy by name —
# plus the bare minimum to anchor asset/checkpoint paths. The full training
# recipes live in YAML:
#   * real: configs/train_real_bc.yaml  → `scripts/train.py --yaml configs/train_real_bc.yaml`
#   * sim:  configs/rl_pipeline_sim.yaml → RL pipeline (overrides train_aug,
#     aux loss weights, advantage embedding, batch size, ... on top)

# Shared TrainConfig template — every registry entry derives from this via
# dataclasses.replace().
_BASE_TRAIN_CONFIG = TrainConfig(
    name="_base",  # always overridden by registry entries
    project_name="LeHome",
    model=_BASE_MODEL_CONFIG,
    optimizer=AdamWWithAuxDecay(),
    lr_schedule=_BASE_LR_SCHEDULE,
    num_flow_samples=5,
    batch_size=192,
    assets_base_dir="./outputs/assets",
    checkpoint_base_dir="./outputs/checkpoints",
    num_workers=8,
    save_interval=500,
    keep_period=5000,
)

_CONFIGS = [
    # ── Sim track (BC + RL via the RL pipeline) ────────────────────────────
    # data_sources are built dynamically by the RL pipeline from pipeline
    # state; train_aug / aux loss weights / advantage embedding / attention
    # flags / batch size are overridden from configs/rl_pipeline_sim.yaml at
    # launch (run_rl_pipeline.py builds the final TrainConfig and pickles it
    # to train.py).
    dataclasses.replace(
        _BASE_TRAIN_CONFIG,
        name="pi_modified_bc_rl",
        exp_name="sim_rl",
        # Inference contract: trained with XSA — serving must match (serve.py reads this).
        use_xsa=True,
        use_flash_attention=True,
        data=LeRobotLeHomeDataConfig(
            repo_id="multi_bc_rl",
            assets=AssetsConfig(
                asset_id="sim_rl",
            ),
            base_config=DataConfig(use_per_timestamp_norm=True),
            data_sources=None,  # Set dynamically by RL pipeline
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            use_advantages=True,
            bc_advantage=0.3,
            success_label_smoothing=0.05,
        ),
        weight_loader=weight_loaders.PiModifiedWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100_000,
    ),

    # ── Real track (BC, native real schema) ─────────────────────────────────
    # INFERENCE CONTRACT ONLY. The model flags below define the architecture
    # + preprocessing baked into every real checkpoint, and serve.py rebuilds
    # the policy from this entry by name (--policy.config pi_modified_real_bc)
    # — so they must stay in the registry. Everything else — data sources,
    # bucket mix, train_aug, LR schedule, init checkpoint, loss weights,
    # num_train_steps — lives in configs/train_real_bc.yaml. Train with
    #   uv run python scripts/train.py --yaml configs/train_real_bc.yaml
    # Training this entry directly (without the yaml) is unsupported.
    dataclasses.replace(
        _BASE_TRAIN_CONFIG,
        name="pi_modified_real_bc",
        exp_name="real_bc_v1",
        model=dataclasses.replace(
            _BASE_MODEL_CONFIG,
            # Success-query token + linear heads exist in real checkpoints
            # (which of them actually trains is a loss-weight question — see
            # the yaml). Keypoint + world-modeling heads are NOT built
            # (params absent from real checkpoints).
            use_success_head=True,
            use_keypoint_distance_head=False,
            use_wm_fast_head=False,
            use_wm_flow_head=False,
            # Garment-type input embedding on: the inference bootstrap
            # latches the predicted type into this slot from chunk 2 onward
            # (PolicyClient in scripts/record_real_dagger.py).
            use_garment_type_input=True,
            # No advantage machinery in real checkpoints.
            use_advantage_embedding=False,
            # Real data is native — no 180° top-cam flip at inference.
            rotate_top_image=False,
        ),
        data=LeRobotLeHomeDataConfig(
            repo_id="real_bc",
            assets=AssetsConfig(
                asset_id="real_bc",  # keys norm_stats / FAST tokenizer lookup
            ),
            base_config=DataConfig(use_per_timestamp_norm=True),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            # Real-camera source columns: right_front / left_wrist / right_wrist.
            raw_real_camera_mapping=True,
        ),
    ),

]
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
