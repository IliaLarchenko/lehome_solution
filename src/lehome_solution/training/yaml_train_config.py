"""Build a ``TrainConfig`` from a YAML file.

Lets training be controlled entirely from YAML (e.g.
``configs/train_real_bc.yaml``) instead of editing ``config.py``:

    uv run python scripts/train.py --yaml configs/train_real_bc.yaml --resume

The YAML starts from a registered base config (``base_config:`` — used for
the config ``name``, which keys the assets/checkpoint directories, and for
any field the YAML doesn't mention) and overrides fields on top:

* top-level keys -> ``TrainConfig`` fields (``exp_name``, ``batch_size``,
  ``num_train_steps``, ``hf_model_repo``, ...)
* ``init_checkpoint`` -> ``PiModifiedWeightLoader(path)``; explicit ``null``
  -> ``NoOpWeightLoader`` (fresh init)
* ``lr_schedule:`` / ``optimizer:`` -> fields of the base config's schedule /
  optimizer dataclass
* ``model:`` -> ``PiModifiedConfig`` fields; nested ``train_aug:`` ->
  ``TrainAugmentationConfig`` fields
* ``data:`` -> ``LeRobotLeHomeDataConfig`` fields, with conveniences:
  ``asset_id`` (-> ``AssetsConfig``), ``use_per_timestamp_norm`` (-> the
  suppressed ``DataConfig``), ``data_sources`` (list of ``DataSourceSpec``
  dicts), ``dagger_sampling`` / ``sim_garment_skip`` / ``initial_stuck``
  (nested dataclasses; explicit ``null`` disables)

Every section validates key names against the target dataclass and raises on
unknown keys, so typos fail loudly instead of silently training with defaults.
"""

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from lehome_solution.models.train_aug_config import TrainAugmentationConfig
from lehome_solution.training import config as _config
from lehome_solution.training import weight_loaders
from lehome_solution.training.data_loader import (
    DaggerSamplingConfig,
    InitialStuckConfig,
    SimGarmentSkipConfig,
)

from openpi.training.config import AssetsConfig


def _validate_keys(raw: dict, cls: type, section: str) -> None:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in '{section}': {sorted(unknown)}. "
            f"Known keys for {cls.__name__}: {sorted(known)}"
        )


def _coerce_lists(cls: type, raw: dict) -> dict:
    """YAML lists -> tuples (frozen dataclasses use tuple fields)."""
    out = dict(raw)
    for key, val in out.items():
        if isinstance(val, list):
            out[key] = tuple(val)
    return out


def _replace_section(obj: Any, raw: dict, section: str) -> Any:
    cls = type(obj)
    _validate_keys(raw, cls, section)
    return dataclasses.replace(obj, **_coerce_lists(cls, raw))


def _build_nested(base: Any, raw: Any, cls: type, section: str) -> Any:
    """Nested optional dataclass: dict overrides, explicit null disables."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"'{section}' must be a mapping or null, got {type(raw).__name__}")
    start = base if isinstance(base, cls) else cls()
    return _replace_section(start, raw, section)


def _build_data(base_data: Any, raw: dict) -> Any:
    """Apply a ``data:`` section on top of the base LeRobotLeHomeDataConfig."""
    raw = dict(raw)

    if "asset_id" in raw:
        asset_id = raw.pop("asset_id")
        base_data = dataclasses.replace(base_data, assets=AssetsConfig(asset_id=asset_id))

    if "use_per_timestamp_norm" in raw:
        per_ts = raw.pop("use_per_timestamp_norm")
        inner = base_data.base_config or _config.DataConfig()
        base_data = dataclasses.replace(
            base_data, base_config=dataclasses.replace(inner, use_per_timestamp_norm=per_ts)
        )

    if "data_sources" in raw:
        specs = []
        for i, src in enumerate(raw.pop("data_sources") or []):
            _validate_keys(src, _config.DataSourceSpec, f"data.data_sources[{i}]")
            specs.append(_config.DataSourceSpec(**_coerce_lists(_config.DataSourceSpec, src)))
        base_data = dataclasses.replace(base_data, data_sources=tuple(specs) or None)

    for key, cls in (
        ("dagger_sampling", DaggerSamplingConfig),
        ("sim_garment_skip", SimGarmentSkipConfig),
        ("initial_stuck", InitialStuckConfig),
    ):
        if key in raw:
            built = _build_nested(getattr(base_data, key), raw.pop(key), cls, f"data.{key}")
            base_data = dataclasses.replace(base_data, **{key: built})

    return _replace_section(base_data, raw, "data")


# Top-level YAML keys handled outside the generic TrainConfig field pass.
_SPECIAL_KEYS = {"base_config", "init_checkpoint", "model", "data", "lr_schedule", "optimizer"}
# TrainConfig fields that cannot be set from YAML.
_BLOCKED_FIELDS = {"name", "weight_loader", "freeze_filter", "model", "data", "lr_schedule", "optimizer"}


def train_config_from_yaml(path: str | Path, overrides: dict[str, Any] | None = None) -> "_config.TrainConfig":
    """Load a TrainConfig from YAML. ``overrides`` are top-level field overrides
    (e.g. ``{"resume": True}`` from the CLI) applied last."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    base_name = raw.get("base_config")
    if not base_name:
        raise ValueError(f"{path}: 'base_config' is required (a registered config name, e.g. pi_modified_real_bc)")
    cfg = _config.get_config(base_name)

    # Top-level TrainConfig fields.
    top_fields = {f.name for f in dataclasses.fields(_config.TrainConfig)} - _BLOCKED_FIELDS
    unknown = set(raw) - top_fields - _SPECIAL_KEYS
    if unknown:
        raise ValueError(
            f"{path}: unknown top-level keys: {sorted(unknown)}. "
            f"Known: {sorted(top_fields | _SPECIAL_KEYS)}"
        )
    top_kwargs = {k: raw[k] for k in raw if k in top_fields}

    # Nested sections (each replaces on top of the base config's value).
    if isinstance(raw.get("model"), dict):
        model_raw = dict(raw["model"])
        if "train_aug" in model_raw:
            model_raw["train_aug"] = _build_nested(
                getattr(cfg.model, "train_aug", None), model_raw.pop("train_aug"),
                TrainAugmentationConfig, "model.train_aug",
            )
        top_kwargs["model"] = _replace_section(cfg.model, model_raw, "model")

    if isinstance(raw.get("data"), dict):
        top_kwargs["data"] = _build_data(cfg.data, raw["data"])

    if isinstance(raw.get("lr_schedule"), dict):
        top_kwargs["lr_schedule"] = _replace_section(cfg.lr_schedule, raw["lr_schedule"], "lr_schedule")

    if isinstance(raw.get("optimizer"), dict):
        top_kwargs["optimizer"] = _replace_section(cfg.optimizer, raw["optimizer"], "optimizer")

    if "init_checkpoint" in raw:
        ckpt = raw["init_checkpoint"]
        top_kwargs["weight_loader"] = (
            weight_loaders.PiModifiedWeightLoader(ckpt) if ckpt else weight_loaders.NoOpWeightLoader()
        )

    if overrides:
        for key, val in overrides.items():
            if key not in top_fields:
                raise ValueError(f"Unknown override '{key}' (must be a TrainConfig field)")
            top_kwargs[key] = val

    return dataclasses.replace(cfg, **top_kwargs)


def resolve_train_config(name_or_yaml: str) -> "_config.TrainConfig":
    """Resolve a registry config name OR a .yaml/.yml path to a TrainConfig.

    Lets scripts that take a config name (compute_norm_stats, FAST tokenizer,
    serve) accept the full YAML recipe instead — needed because the registry
    entries only carry the inference contract, not e.g. ``data_sources``.
    """
    s = str(name_or_yaml)
    if s.endswith((".yaml", ".yml")):
        return train_config_from_yaml(s)
    return _config.get_config(s)
