"""Test training configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
from lehome_solution.training.config import (
    get_config, AdamWWithAuxDecay, aux_head_weight_decay_mask,
)
from lehome_solution.models.pi_modified_config import PiModifiedConfig
from lehome_solution.constants import GARMENT_TYPES, GARMENT_TYPE_TO_ID


def test_load_config():
    config = get_config("pi_modified_bc_rl")
    assert config.name == "pi_modified_bc_rl"
    assert config.project_name == "LeHome"


def test_config_model_params():
    config = get_config("pi_modified_bc_rl")
    assert config.model.action_dim == 12
    assert config.model.action_horizon == 30
    assert config.model.model_type == "pi_modified"


def test_config_data_params():
    config = get_config("pi_modified_bc_rl")
    assert config.data.repo_id == "multi_bc_rl"
    assert config.data.use_delta_joint_actions is True


def test_garment_types():
    assert len(GARMENT_TYPES) == 4
    assert "top_long" in GARMENT_TYPES
    assert "top_short" in GARMENT_TYPES
    assert "pant_long" in GARMENT_TYPES
    assert "pant_short" in GARMENT_TYPES


def test_garment_type_to_id_mapping():
    for i, name in enumerate(GARMENT_TYPES):
        assert GARMENT_TYPE_TO_ID[name] == i



def test_config_not_found():
    with pytest.raises(ValueError, match="not found"):
        get_config("nonexistent_config")


def test_config_optimizer():
    config = get_config("pi_modified_bc_rl")
    assert isinstance(config.optimizer, AdamWWithAuxDecay)
    # Baseline decay matches openpi (1e-10); aux heads get meaningful decay.
    assert config.optimizer.weight_decay == pytest.approx(1e-10)
    assert config.optimizer.aux_head_weight_decay > 0


def test_aux_head_weight_decay_mask():
    """Mask must be True at aux-head kernel/bias leaves, False elsewhere."""
    import jax
    import flax.nnx as nnx

    class ToyModel(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.backbone = nnx.Linear(8, 8, rngs=rngs)
            self.success_head = nnx.Linear(8, 1, rngs=rngs)
            self.checkpoint_head = nnx.Linear(8, 1, rngs=rngs)
            self.garment_type_head = nnx.Linear(8, 4, rngs=rngs)
            self.completion_head = nnx.Linear(8, 1, rngs=rngs)
            self.ttc_head = nnx.Linear(8, 1, rngs=rngs)
            self.learned_token = nnx.Param(jax.numpy.zeros((1, 8)))

    model = ToyModel(nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    mask = aux_head_weight_decay_mask(params)

    # Flatten and check: aux-head entries True, backbone/learned_token entries False.
    for path, leaf in jax.tree_util.tree_flatten_with_path(mask)[0]:
        path_str = "/".join(
            str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p)))) for p in path
        )
        is_aux = any(
            h in path_str for h in (
                "success_head", "checkpoint_head", "garment_type_head",
                "completion_head", "ttc_head",
            )
        )
        assert bool(leaf) == is_aux, f"mask mismatch at {path_str}: got {leaf}, expected {is_aux}"


def test_config_num_flow_samples():
    config = get_config("pi_modified_bc_rl")
    assert config.num_flow_samples == 5


def test_fast_dim_ranges():
    config = get_config("pi_modified_bc_rl")
    ranges = config.model.get_fast_dim_ranges()
    assert ranges == [(0, 12)]
    assert config.model.get_total_fast_dims() == 12


def test_num_llm_layers_in_config():
    config = get_config("pi_modified_bc_rl")
    assert config.model.num_llm_layers is None  # None = use all layers
    assert config.model.paligemma_variant == "gemma_2b"
    assert config.model.action_expert_variant == "gemma_300m"
    assert config.model.use_kv_transform is True


def test_small_model_num_layers_override():
    cfg = PiModifiedConfig(num_llm_layers=9)
    assert cfg.num_llm_layers == 9


def test_kv_coeffs_scaling():
    """Test that KV coefficient scaling preserves row sums."""
    import numpy as np
    from lehome_solution.training.weight_loaders import _scale_kv_coeffs
    import flax.traverse_util

    # Simulate loading 18-layer coefficients into 9-layer model
    full_coeffs = np.random.randn(18, 18).astype(np.float32)
    truncated_coeffs = full_coeffs[:9, :9].copy()

    loaded_params = {"kv_transform": {"k_coeffs": full_coeffs, "v_coeffs": full_coeffs.copy()}}
    merged_params = {"kv_transform": {"k_coeffs": truncated_coeffs, "v_coeffs": truncated_coeffs.copy()}}

    result = _scale_kv_coeffs(merged_params, loaded_params)
    result_flat = flax.traverse_util.flatten_dict(result, sep="/")

    for key in ("kv_transform/k_coeffs", "kv_transform/v_coeffs"):
        scaled = result_flat[key]
        for d in range(9):
            full_sum = full_coeffs[d, :].sum()
            scaled_sum = scaled[d, :].sum()
            np.testing.assert_allclose(scaled_sum, full_sum, rtol=1e-5,
                err_msg=f"Row {d} sum mismatch: {scaled_sum} vs {full_sum}")


def test_kv_coeffs_scaling_zero_row():
    """Test that rows with ~zero truncated sum get re-initialized randomly."""
    import numpy as np
    from lehome_solution.training.weight_loaders import _scale_kv_coeffs
    import flax.traverse_util

    full_coeffs = np.zeros((18, 18), dtype=np.float32)
    # Row 0: all weight on columns >= 9 (will be zero after truncation)
    full_coeffs[0, 10] = 1.0
    # Row 1: weight on column 0 (will survive truncation)
    full_coeffs[1, 0] = 2.0

    truncated_coeffs = full_coeffs[:9, :9].copy()
    assert abs(truncated_coeffs[0, :].sum()) < 1e-8  # row 0 is zero

    loaded_params = {"kv_transform": {"k_coeffs": full_coeffs}}
    merged_params = {"kv_transform": {"k_coeffs": truncated_coeffs}}

    result = _scale_kv_coeffs(merged_params, loaded_params)
    result_flat = flax.traverse_util.flatten_dict(result, sep="/")
    scaled = result_flat["kv_transform/k_coeffs"]

    # Row 0: was zero, should be re-initialized (non-zero)
    assert np.any(scaled[0, :] != 0), "Zero row should be re-initialized"
    # Row 1: should preserve sum
    np.testing.assert_allclose(scaled[1, :].sum(), 2.0, rtol=1e-5)


def test_constants_consistency():
    """constants.py values match pi_modified_config re-exports."""
    from lehome_solution.constants import (
        GARMENT_TYPES as C_GT,
        GARMENT_TYPE_TO_ID as C_GTID,
        ACTION_DIM,
        ACTION_HORIZON,
    )
    assert C_GT == GARMENT_TYPES
    assert C_GTID == GARMENT_TYPE_TO_ID
    assert ACTION_DIM == 12
    assert ACTION_HORIZON == 30

    # GARMENT_TYPES now lives in constants (removed from pi_modified_config)
    assert len(C_GT) == 4


def test_config_no_absolute_paths():
    """No /home/ paths in any training config."""
    import dataclasses

    for name in ("pi_modified_bc_rl", "pi_modified_real_bc"):
        config = get_config(name)
        flat = str(dataclasses.asdict(config))
        assert "/home/" not in flat, f"Config {name} contains absolute /home/ path"


# ---------------------------------------------------------------------------
# Pipeline config validation error tests
# ---------------------------------------------------------------------------

def test_pipeline_config_validation_errors():
    """RLPipelineConfig.validate() catches various invalid configurations."""
    from lehome_solution.training.pipeline_config import (
        RLPipelineConfig, AdvantageConfig, BCDatasetConfig,
    )

    # num_iterations < 1
    cfg = RLPipelineConfig(num_iterations=0)
    errors = cfg.validate()
    assert any("num_iterations" in e for e in errors)

    # Invalid advantage mode
    cfg = RLPipelineConfig(advantage=AdvantageConfig(mode="invalid_mode"))
    errors = cfg.validate()
    assert any("advantage.mode" in e for e in errors)

    # gamma out of range (must be in (0, 1])
    cfg = RLPipelineConfig(advantage=AdvantageConfig(gamma=0.0))
    errors = cfg.validate()
    assert any("gamma" in e for e in errors)

    cfg = RLPipelineConfig(advantage=AdvantageConfig(gamma=1.5))
    errors = cfg.validate()
    assert any("gamma" in e for e in errors)

    # decay_factor out of range (must be in (0, 1])
    cfg = RLPipelineConfig(bc_dataset=BCDatasetConfig(decay_factor=0.0))
    errors = cfg.validate()
    assert any("decay_factor" in e for e in errors)

    cfg = RLPipelineConfig(bc_dataset=BCDatasetConfig(decay_factor=1.5))
    errors = cfg.validate()
    assert any("decay_factor" in e for e in errors)

    # Valid config should have no errors
    cfg = RLPipelineConfig()
    errors = cfg.validate()
    assert len(errors) == 0


# Model fields that define the architecture + preprocessing baked into a
# checkpoint. serve.py rebuilds policies from the registry entry by name, so
# the yaml recipe must never diverge from the registry on these.
_INFERENCE_CONTRACT_MODEL_FIELDS = (
    "action_dim", "action_horizon",
    "use_success_head", "use_keypoint_distance_head",
    "use_wm_fast_head", "use_wm_flow_head",
    "use_garment_type_input", "use_advantage_embedding",
    "use_state_embedding", "use_kv_transform",
    "rotate_top_image",
)


def test_yaml_train_config_inference_contract():
    """configs/train_real_bc.yaml (full training recipe) must agree with the
    thin registry alias pi_modified_real_bc (inference contract) on every
    architecture/preprocessing field — otherwise a checkpoint trained from
    the yaml could not be served via --policy.config pi_modified_real_bc."""
    from lehome_solution.training import config as _config
    from lehome_solution.training.yaml_train_config import train_config_from_yaml

    y = train_config_from_yaml("configs/train_real_bc.yaml")
    r = _config.get_config("pi_modified_real_bc")

    # Name comes from base_config — keys assets/checkpoint dirs.
    assert y.name == "pi_modified_real_bc"

    diffs = [
        f for f in _INFERENCE_CONTRACT_MODEL_FIELDS
        if getattr(y.model, f) != getattr(r.model, f)
    ]
    assert not diffs, f"yaml model contract drifted from registry alias: {diffs}"

    # Data-side inference contract (input pipeline at serve time).
    for f in ("raw_real_camera_mapping", "use_delta_joint_actions", "use_fast_tokenization"):
        assert getattr(y.data, f) == getattr(r.data, f), f
    assert y.data.assets.asset_id == r.data.assets.asset_id == "real_bc"
    assert y.data.base_config.use_per_timestamp_norm == r.data.base_config.use_per_timestamp_norm is True


def test_yaml_train_config_overrides_and_validation():
    """CLI overrides apply on top; unknown keys raise loudly."""
    import pytest as _pytest
    from lehome_solution.training.yaml_train_config import train_config_from_yaml

    cfg = train_config_from_yaml(
        "configs/train_real_bc.yaml", {"resume": True, "exp_name": "exp_x"}
    )
    assert cfg.resume is True
    assert cfg.exp_name == "exp_x"

    bad = Path(tempfile.mkdtemp()) / "bad.yaml"
    bad.write_text("base_config: pi_modified_real_bc\nmodel:\n  garment_type_los_weight: 0.1\n")
    with _pytest.raises(ValueError, match="garment_type_los_weight"):
        train_config_from_yaml(bad)


def test_pipeline_config_to_dict_roundtrip():
    """Load from YAML, convert to_dict(), verify nested keys are present."""
    from lehome_solution.training.pipeline_config import RLPipelineConfig

    cfg = RLPipelineConfig.from_yaml("configs/rl_pipeline_sim.yaml")
    d = cfg.to_dict()

    # Advantage should be nested
    assert "advantage" in d
    assert d["advantage"]["mode"] == "auto"
    assert d["advantage"]["gamma"] == pytest.approx(0.999)
    assert d["advantage"]["gae_lambda"] == pytest.approx(0.99)

    # Top-level fields should be present
    assert "config_name" in d
    assert d["config_name"] == "pi_modified_bc_rl"
    assert "exp_name" in d
    assert "num_iterations" in d
    assert "steps_per_iteration" in d
    assert "warmup_steps" in d
    assert "bc_dataset" in d
    assert "hf_model_repo" in d
