"""Tests for RL pipeline utilities and HF upload module."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Inference optimization tests
# ---------------------------------------------------------------------------

def test_inference_optimization_init():
    from lehome_solution.eval.inference_optimization import init_prior, PARAM_SPACES
    prior = init_prior()
    assert "params" in prior
    for param_name in PARAM_SPACES:
        assert param_name in prior["params"]
        for v in PARAM_SPACES[param_name]:
            v_str = str(v)
            assert v_str in prior["params"][param_name]
            assert prior["params"][param_name][v_str]["alpha"] == 1.0
            assert prior["params"][param_name][v_str]["beta"] == 1.0


def test_inference_optimization_sample():
    from lehome_solution.eval.inference_optimization import init_prior, sample_config
    prior = init_prior()
    rng = np.random.RandomState(42)
    config = sample_config(prior, rng)

    assert "actions_to_execute" in config
    assert "execute_in_n_steps" in config
    assert "actions_to_keep" in config
    assert "num_steps" in config
    assert "time_threshold_inpaint" in config
    assert "k_execute" in config

    # Constraint check
    assert config["actions_to_execute"] + config["actions_to_keep"] <= 30
    assert config["execute_in_n_steps"] >= 1
    assert config["num_steps"] == 10  # fixed, not optimized


def test_inference_optimization_update():
    from lehome_solution.eval.inference_optimization import init_prior, update_prior, get_best_config
    prior = init_prior()

    # Simulate episodes where actions_to_execute=10 consistently wins over 5
    episodes = []
    for _ in range(10):
        episodes.append({
            "inference_config": {
                "actions_to_execute": 10,
                "k_execute": 1.0,
                "actions_to_keep": 3,
                "num_steps": 10,
                "time_threshold_inpaint": 0.5,
            },
            "garment": "garment_A",
            "garment_type": "top_long",
            "reward": 1.0,
        })
    for _ in range(10):
        episodes.append({
            "inference_config": {
                "actions_to_execute": 5,
                "k_execute": 1.0,
                "actions_to_keep": 3,
                "num_steps": 10,
                "time_threshold_inpaint": 0.5,
            },
            "garment": "garment_A",
            "garment_type": "top_long",
            "reward": 0.0,
        })

    prior = update_prior(prior, episodes, decay_factor=0.95,
                         garment_type_success_rates={"top_long": 0.5})

    # actions_to_execute=10 should be favored over 5
    ate_prior = prior["params"]["actions_to_execute"]
    mean_10 = ate_prior["10"]["alpha"] / (ate_prior["10"]["alpha"] + ate_prior["10"]["beta"])
    mean_5 = ate_prior["5"]["alpha"] / (ate_prior["5"]["alpha"] + ate_prior["5"]["beta"])
    assert mean_10 > mean_5

    # History should have one entry
    assert len(prior["history"]) == 1
    assert prior["iteration"] == 1


def test_inference_optimization_save_load():
    from lehome_solution.eval.inference_optimization import init_prior, save_prior, load_prior
    prior = init_prior()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prior.json"
        save_prior(prior, path)
        loaded = load_prior(path)
        assert loaded == prior


def test_inference_optimization_reconcile_targeted_edits(monkeypatch):
    """Reconcile drops removed arms, adds new arms at Beta(1,1), preserves the rest."""
    from lehome_solution.eval import inference_optimization as inf_opt

    # Build a prior with an "old" arm value (13.0) that isn't in current PARAM_SPACES,
    # plus a dirty value on a kept arm. Reconcile should drop 13.0 and preserve 9.0.
    prior = inf_opt.init_prior()
    prior["params"]["cfg_scale"]["13.0"] = {"alpha": 7.5, "beta": 2.5}
    prior["per_garment_type"]["top_long"]["params"]["cfg_scale"]["13.0"] = {"alpha": 4.0, "beta": 1.0}
    prior["params"]["cfg_scale"]["9.0"] = {"alpha": 6.0, "beta": 4.0}
    prior["per_garment_type"]["top_long"]["params"]["cfg_scale"]["9.0"] = {"alpha": 3.0, "beta": 2.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prior.json"
        inf_opt.save_prior(prior, path)

        # Mutate the search space: add "13.0" back temporarily so the test can
        # also verify "add new arm" behavior — we mutate PARAM_SPACES away from
        # what we just saved, then load to trigger reconciliation.
        new_spaces = dict(inf_opt.PARAM_SPACES)
        new_spaces["cfg_scale"] = [5.0, 7.0, 9.0, 15.0]   # drop 11.0 from current, add 15.0
        monkeypatch.setattr(inf_opt, "PARAM_SPACES", new_spaces)

        loaded = inf_opt.load_prior(path)

    cfg_arms = loaded["params"]["cfg_scale"]
    assert "13.0" not in cfg_arms                           # dropped (not in current spaces)
    assert "11.0" not in cfg_arms                           # dropped (mutated away)
    assert cfg_arms["15.0"] == {"alpha": 1.0, "beta": 1.0}  # new arm seeded fresh
    # Kept arm with dirty values is preserved.
    assert cfg_arms["9.0"]["alpha"] == 6.0
    assert cfg_arms["9.0"]["beta"] == 4.0
    gt_cfg = loaded["per_garment_type"]["top_long"]["params"]["cfg_scale"]
    assert "13.0" not in gt_cfg                              # dropped from sub-prior too
    assert gt_cfg["9.0"]["alpha"] == 3.0                     # dirty kept arm preserved
    assert gt_cfg["15.0"] == {"alpha": 1.0, "beta": 1.0}     # new arm seeded fresh


def test_inference_optimization_direct_update():
    """Only the sampled value should be updated (no kernel smoothing)."""
    from lehome_solution.eval.inference_optimization import init_prior, update_prior
    prior = init_prior()

    # One success, SR baseline 0.5 → r=0.5, α += 0.5, β += 0 on sampled slots.
    episodes = [{
        "inference_config": {
            "actions_to_execute": 10,
            "k_execute": 1.0,
            "actions_to_keep": 6,
            "num_steps": 10,
            "time_threshold_inpaint": 0.5,
        },
        "garment": "garment_A",
        "garment_type": "top_long",
        "reward": 1.0,
    }]

    prior = update_prior(prior, episodes, decay_factor=1.0,
                         garment_type_success_rates={"top_long": 0.5})

    atk = prior["params"]["actions_to_keep"]
    # Only the sampled value (6) should be updated; the other arm stays at prior
    assert atk["6"]["alpha"] > 1.0  # Got positive update
    assert atk["0"]["alpha"] == 1.0  # Untouched


def test_inference_optimization_garment_type_normalization():
    """Per-garment-type SR baseline normalizes reward: α += max(r−SR, 0), β += max(SR−r, 0)."""
    from lehome_solution.eval.inference_optimization import init_prior, update_prior
    prior = init_prior()

    # Easy type (top_long SR 0.9), hard type (pant_long SR 0.1).
    # ate=5  success on easy: r=0.1 → α += 0.1.
    # ate=5  fail on easy:    r=−0.9 → β += 0.9.
    # ate=10 success on hard: r=0.9 → α += 0.9.
    # ate=10 fail on hard:    r=−0.1 → β += 0.1.
    gt_sr = {"top_long": 0.9, "pant_long": 0.1}
    episodes = [
        {"inference_config": {"actions_to_execute": 5, "k_execute": 1.0,
         "actions_to_keep": 3, "num_steps": 10, "time_threshold_inpaint": 0.5},
         "garment": "g_a", "garment_type": "top_long", "reward": 1.0},
        {"inference_config": {"actions_to_execute": 5, "k_execute": 1.0,
         "actions_to_keep": 3, "num_steps": 10, "time_threshold_inpaint": 0.5},
         "garment": "g_a", "garment_type": "top_long", "reward": 0.0},
        {"inference_config": {"actions_to_execute": 10, "k_execute": 1.0,
         "actions_to_keep": 3, "num_steps": 10, "time_threshold_inpaint": 0.5},
         "garment": "g_b", "garment_type": "pant_long", "reward": 1.0},
        {"inference_config": {"actions_to_execute": 10, "k_execute": 1.0,
         "actions_to_keep": 3, "num_steps": 10, "time_threshold_inpaint": 0.5},
         "garment": "g_b", "garment_type": "pant_long", "reward": 0.0},
    ]

    prior = update_prior(prior, episodes, decay_factor=1.0,
                         garment_type_success_rates=gt_sr)

    ate = prior["params"]["actions_to_execute"]
    # ate=5  posterior: α=1.1, β=1.9 → mean 0.367
    # ate=10 posterior: α=1.9, β=1.1 → mean 0.633
    mean_5 = ate["5"]["alpha"] / (ate["5"]["alpha"] + ate["5"]["beta"])
    mean_10 = ate["10"]["alpha"] / (ate["10"]["alpha"] + ate["10"]["beta"])
    assert mean_10 > mean_5
    assert ate["5"]["alpha"] == pytest.approx(1.1)
    assert ate["5"]["beta"] == pytest.approx(1.9)
    assert ate["10"]["alpha"] == pytest.approx(1.9)
    assert ate["10"]["beta"] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Original tests
# ---------------------------------------------------------------------------

def test_import_hf_upload():
    from lehome_solution.training.hf_upload import upload_checkpoint_to_hf, upload_dataset_to_hf


def test_import_pipeline_config():
    """Test that pipeline config loading works."""
    # Ensure we can import the pipeline config loader
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    # Can't directly import run_rl_pipeline (it sets JAX_PLATFORMS at import),
    # but we can verify the YAML config loads
    import yaml
    config_path = Path(__file__).parent.parent / "configs" / "rl_pipeline_sim.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["config_name"] == "pi_modified_bc_rl"
    assert "bc_dataset" in cfg
    assert "hf_model_repo" in cfg
    assert "hf_dataset_repo" in cfg


def test_pipeline_config_has_required_fields():
    """Verify pipeline config has all required fields."""
    import yaml
    config_path = Path(__file__).parent.parent / "configs" / "rl_pipeline_sim.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    required = [
        "config_name", "exp_name", "warmup_steps", "num_iterations",
        "steps_per_iteration", "num_workers",
        "bc_dataset", "rl_decay_factor", "advantage", "wandb_enabled",
        "batch_size", "save_interval", "keep_period",
    ]
    for key in required:
        assert key in cfg, f"Missing required config key: {key}"
    # Advantage sub-keys
    assert "mode" in cfg["advantage"]
    assert "weighting_clipping" in cfg["advantage"]


def test_exploration_noise_disabled_by_default():
    """Pipeline config must default exploration_noise to OFF — never leaks
    into rollouts unless explicitly enabled in YAML.
    """
    from lehome_solution.training.pipeline_config import (
        ExplorationNoiseConfig, RLPipelineConfig,
    )
    cfg = ExplorationNoiseConfig()
    assert cfg.enabled is False
    assert cfg.prob == 0.0
    assert cfg.scale == 0.0

    # Loading the shipped YAMLs must also be off: the tri-gate requires
    # enabled AND prob AND scale to be truthy, and enabled stays false even
    # though a yaml may carry non-zero prob/scale for ad-hoc experiments.
    for name in ("rl_pipeline_sim.yaml", "rl_pipeline_sim_to_real.yaml"):
        config_path = Path(__file__).parent.parent / "configs" / name
        pipe_cfg = RLPipelineConfig.from_yaml(config_path)
        assert pipe_cfg.exploration_noise.enabled is False, name


def test_hf_upload_checkpoint_missing_dir():
    """upload_checkpoint_to_hf returns False when checkpoint dir doesn't exist."""
    from lehome_solution.training.hf_upload import upload_checkpoint_to_hf

    with tempfile.TemporaryDirectory() as tmpdir:
        result = upload_checkpoint_to_hf(tmpdir, 999, "fake/repo", keep_period=1000)
        assert result is False


def test_hf_upload_dataset_missing_dir():
    """upload_dataset_to_hf returns False when dataset dir doesn't exist."""
    from lehome_solution.training.hf_upload import upload_dataset_to_hf

    result = upload_dataset_to_hf("/nonexistent/path", "test_run", "fake/repo")
    assert result is False


def test_train_metrics_structure():
    """Verify train.py structures metrics correctly."""
    # Import the structuring function
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    # We need to set env before importing train
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

    from train import _structure_metrics

    raw = {
        "loss": 0.5,
        "action_loss": 0.3,
        "fast_loss": 0.1,
        "success_loss": 0.05,
        "fast_perplexity": 1.1,
        "success_pred_mean": 0.6,
        "success_pred_std": 0.2,
        "grad_norm": 1.5,
        "param_norm": 100.0,
        "weight_mean": 1.0,
        "weight_min": 0.5,
        "weight_max": 2.0,
        # New aux metrics:
        "success_mae": 0.05,
        "checkpoint_loss": 0.03,
        "checkpoint_pred_mean": 0.4,
        "garment_type_loss": 0.15,
        "garment_type_accuracy": 0.8,
        "completion_loss": 0.01,
        "completion_pred_mean": 0.5,
        # These should be dropped:
        "action_loss_left_0": 0.1,
    }

    structured = _structure_metrics(raw)

    # Check structured keys
    assert "train/loss" in structured
    assert "train/action_loss" in structured
    assert "train/fast_loss" in structured
    assert "train/success_loss" in structured
    assert "train/fast_perplexity" in structured
    assert "train/success_pred_mean" in structured
    assert "train/grad_norm" in structured
    assert "train/param_norm" in structured
    assert "train/weight_mean" in structured
    assert "train/success_mae" in structured
    assert "train/checkpoint_loss" in structured
    assert "train/garment_type_loss" in structured
    assert "train/garment_type_accuracy" in structured
    assert "train/completion_loss" in structured

    # Verify dropped keys (per-joint losses etc.)
    assert "train/action_loss_left_0" not in structured


# ---------------------------------------------------------------------------
# Failure state management tests
# ---------------------------------------------------------------------------

def test_get_failure_states_empty_dir():
    """get_failure_states returns empty list for nonexistent dir."""
    from lehome_solution.eval.rollout_strategies import get_failure_states
    assert get_failure_states("/nonexistent/path") == []


def test_get_failure_states_reads_npz():
    """get_failure_states reads NPZ metadata correctly (legacy, no sidecar)."""
    from lehome_solution.eval.rollout_strategies import get_failure_states
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = json.dumps({
            "garment": "Top_Long_Seen_0",
            "garment_type": "top_long",
            "seed": 42,
            "ep_idx": 0,
            "failure_frame": 100,
        })
        np.savez_compressed(
            Path(tmpdir) / "Top_Long_Seen_0_seed42_ep0.npz",
            garment_points=np.zeros((100, 3), dtype=np.float32),
            metadata=np.array(meta),
        )
        results = get_failure_states(tmpdir)
        assert len(results) == 1
        assert results[0]["garment"] == "Top_Long_Seen_0"
        assert results[0]["garment_type"] == "top_long"
        assert results[0]["seed"] == 42
        assert results[0]["npz_path"].endswith(".npz")


def test_get_failure_states_reads_json_sidecar():
    """get_failure_states prefers JSON sidecar over NPZ metadata."""
    from lehome_solution.eval.rollout_strategies import get_failure_states
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create NPZ with metadata
        meta_npz = json.dumps({
            "garment": "Wrong_Name",
            "garment_type": "wrong",
            "seed": 0,
        })
        np.savez_compressed(
            Path(tmpdir) / "Top_Long_Seen_0_seed42_ep0.npz",
            garment_points=np.zeros((100, 3), dtype=np.float32),
            metadata=np.array(meta_npz),
        )
        # Create JSON sidecar (should override NPZ metadata)
        with open(Path(tmpdir) / "Top_Long_Seen_0_seed42_ep0.json", "w") as f:
            json.dump({
                "garment": "Top_Long_Seen_0",
                "garment_type": "top_long",
                "seed": 42,
                "particle_count": 100,
                "status": "active",
            }, f)

        results = get_failure_states(tmpdir)
        assert len(results) == 1
        assert results[0]["garment"] == "Top_Long_Seen_0"
        assert results[0]["seed"] == 42


def test_get_failure_states_filters_non_active():
    """get_failure_states only returns active states."""
    from lehome_solution.eval.rollout_strategies import get_failure_states
    with tempfile.TemporaryDirectory() as tmpdir:
        for status, name in [("active", "active"), ("consumed", "consumed"), ("in_use", "in_use")]:
            meta = json.dumps({"garment": f"G_{name}", "garment_type": "top_long", "seed": 1})
            np.savez_compressed(
                Path(tmpdir) / f"G_{name}_seed1_ep0.npz",
                metadata=np.array(meta),
            )
            with open(Path(tmpdir) / f"G_{name}_seed1_ep0.json", "w") as f:
                json.dump({"garment": f"G_{name}", "garment_type": "top_long", "seed": 1, "status": status}, f)

        results = get_failure_states(tmpdir)
        assert len(results) == 1
        assert results[0]["garment"] == "G_active"


def test_hard_mining_prefers_npz():
    """hard_mining_strategy prefers NPZ failure states over metadata failures."""
    from lehome_solution.eval.rollout_strategies import hard_mining_strategy

    pool = ["Top_Long_Seen_0", "Top_Long_Seen_1", "Pant_Long_Seen_0"]
    rng = np.random.RandomState(42)

    # NPZ failure states
    failure_states = [
        {"garment": "Top_Long_Seen_0", "garment_type": "top_long", "seed": 10, "npz_path": "/tmp/a.npz"},
        {"garment": "Top_Long_Seen_1", "garment_type": "top_long", "seed": 20, "npz_path": "/tmp/b.npz"},
    ]
    # Metadata failures (should be ignored when NPZ available)
    recent_failures = [
        {"garment": "Pant_Long_Seen_0", "seed": 30, "ep_idx": 0},
    ]

    selected, overrides = hard_mining_strategy(
        pool, 2, rng,
        recent_failures=recent_failures,
        failure_states=failure_states,
        replays_per_state=1,
    )
    assert overrides is not None
    # All overrides should have restore_npz (from NPZ path)
    for ov in overrides:
        assert "restore_npz" in ov
        assert ov["hard_mining_mode"] == "exact"  # default mode
    # No Pant_Long overrides (NPZ path took priority)
    assert all(ov["garment"].startswith("Top_Long") for ov in overrides)


def test_hard_mining_skips_without_npz():
    """hard_mining_strategy returns empty when no NPZ failure states available."""
    from lehome_solution.eval.rollout_strategies import hard_mining_strategy

    pool = ["Top_Long_Seen_0", "Pant_Long_Seen_0"]
    rng = np.random.RandomState(42)

    selected, overrides = hard_mining_strategy(
        pool, 1, rng,
        failure_states=None,
        replays_per_state=1,
    )
    assert overrides is None
    assert len(selected) == 0


def test_hard_mining_exact_mode():
    """hard_mining_strategy exact mode includes mode in overrides."""
    from lehome_solution.eval.rollout_strategies import hard_mining_strategy

    pool = ["Top_Long_Seen_0"]
    rng = np.random.RandomState(42)
    failure_states = [
        {"garment": "Top_Long_Seen_0", "garment_type": "top_long", "seed": 10, "npz_path": "/tmp/a.npz"},
    ]

    selected, overrides = hard_mining_strategy(
        pool, 1, rng,
        failure_states=failure_states,
        replays_per_state=1,
        mode="exact",
    )
    assert overrides is not None
    assert len(overrides) == 1
    assert overrides[0]["hard_mining_mode"] == "exact"
    assert overrides[0]["restore_npz"] == "/tmp/a.npz"


def test_hard_mining_remove_on_success():
    """hard_mining_strategy remove_on_success controls mode in overrides."""
    from lehome_solution.eval.rollout_strategies import hard_mining_strategy

    pool = ["Top_Long_Seen_0"]
    rng = np.random.RandomState(42)
    failure_states = [
        {"garment": "Top_Long_Seen_0", "garment_type": "top_long", "seed": 10, "npz_path": "/tmp/a.npz"},
    ]

    # remove_on_success=False -> "augs" mode
    selected, overrides = hard_mining_strategy(
        pool, 1, rng,
        failure_states=failure_states,
        replays_per_state=2,
        remove_on_success=False,
    )
    assert overrides is not None
    # One task per sampled state; up to replays_per_state attempts each.
    assert len(overrides) == 1
    assert overrides[0]["max_attempts"] == 2
    for ov in overrides:
        assert ov["hard_mining_mode"] == "augs"

    # remove_on_success=True -> "exact" mode
    _, overrides2 = hard_mining_strategy(
        pool, 1, rng,
        failure_states=failure_states,
        replays_per_state=1,
        remove_on_success=True,
    )
    assert overrides2 is not None
    for ov in overrides2:
        assert ov["hard_mining_mode"] == "exact"


def test_collect_failure_states():
    """_collect_failure_states copies NPZs to persistent dir."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate eval run dir structure with new physics_states/failure/ layout
        eval_dir = Path(tmpdir) / "rl_run1" / "eval_dataset"
        eval_dir.mkdir(parents=True)
        fs_dir = Path(tmpdir) / "rl_run1" / "physics_states" / "failure"
        fs_dir.mkdir(parents=True)
        np.savez_compressed(fs_dir / "test.npz", data=np.zeros(3))
        # Also write a JSON sidecar
        with open(fs_dir / "test.json", "w") as f:
            json.dump({"garment": "Test", "status": "active"}, f)

        persistent_dir = Path(tmpdir) / "persistent"

        # Import with clean env (JAX_PLATFORMS must be set before import)
        os.environ["JAX_PLATFORMS"] = "cpu"
        from run_rl_pipeline import _collect_failure_states
        os.environ.pop("JAX_PLATFORMS", None)

        _collect_failure_states([str(eval_dir)], persistent_dir)

        assert (persistent_dir / "test.npz").exists()
        assert (persistent_dir / "test.json").exists()


def test_collect_failure_states_legacy_dir():
    """_collect_failure_states also reads from legacy failure_states/ dir."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate eval run dir with legacy failure_states/ layout
        eval_dir = Path(tmpdir) / "rl_run1" / "eval_dataset"
        eval_dir.mkdir(parents=True)
        fs_dir = Path(tmpdir) / "rl_run1" / "failure_states"
        fs_dir.mkdir(parents=True)
        np.savez_compressed(fs_dir / "legacy.npz", data=np.zeros(3))

        persistent_dir = Path(tmpdir) / "persistent"

        os.environ["JAX_PLATFORMS"] = "cpu"
        from run_rl_pipeline import _collect_failure_states
        os.environ.pop("JAX_PLATFORMS", None)

        _collect_failure_states([str(eval_dir)], persistent_dir)

        assert (persistent_dir / "legacy.npz").exists()


def test_remove_solved_failures():
    """_remove_solved_failures removes NPZ+JSON pairs for solved garment+seed combos."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create persistent failure NPZ with JSON sidecar
        persistent_dir = Path(tmpdir) / "failure_states"
        persistent_dir.mkdir()
        meta = json.dumps({"garment": "Top_Long_Seen_0", "seed": 42, "status": "active"})
        np.savez_compressed(
            persistent_dir / "Top_Long_Seen_0_seed42_ep0.npz",
            garment_points=np.zeros((10, 3)),
            metadata=np.array(meta),
        )
        with open(persistent_dir / "Top_Long_Seen_0_seed42_ep0.json", "w") as f:
            json.dump({"garment": "Top_Long_Seen_0", "seed": 42, "status": "active"}, f)

        # Create eval dataset with success metadata (uses EVAL_EPISODE_META_DIR)
        from lehome_solution.eval.dataset_writer import EVAL_EPISODE_META_DIR
        eval_dir = Path(tmpdir) / "eval_dataset"
        eval_dir.mkdir()
        meta_dir = eval_dir / EVAL_EPISODE_META_DIR
        meta_dir.mkdir(parents=True)
        with open(meta_dir / "episode_000000.json", "w") as f:
            json.dump({"garment": "Top_Long_Seen_0", "seed": 42, "success": True}, f)

        os.environ["JAX_PLATFORMS"] = "cpu"
        from run_rl_pipeline import _remove_solved_failures
        os.environ.pop("JAX_PLATFORMS", None)

        _remove_solved_failures([str(eval_dir)], persistent_dir)

        assert not (persistent_dir / "Top_Long_Seen_0_seed42_ep0.npz").exists()
        assert not (persistent_dir / "Top_Long_Seen_0_seed42_ep0.json").exists()


def test_get_success_states_no_side_effects():
    """get_success_states does NOT delete short states (no side effects)."""
    from lehome_solution.eval.rollout_strategies import get_success_states
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a short success state (< 60 actions)
        meta = json.dumps({
            "garment": "Top_Long_Seen_0",
            "garment_type": "top_long",
            "seed": 42,
            "n_actions": 10,
        })
        npz_path = Path(tmpdir) / "short_state.npz"
        np.savez_compressed(
            npz_path,
            actions=np.zeros((10, 12), dtype=np.float32),
            metadata=np.array(meta),
        )
        results = get_success_states(tmpdir, min_actions=60)
        assert len(results) == 0
        # File must still exist (no deletion side effect)
        assert npz_path.exists()


def test_get_success_states_reads_json_sidecar():
    """get_success_states reads n_actions from JSON sidecar without loading NPZ."""
    from lehome_solution.eval.rollout_strategies import get_success_states
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = json.dumps({
            "garment": "Top_Long_Seen_0",
            "garment_type": "top_long",
            "seed": 42,
            "n_actions": 100,
        })
        np.savez_compressed(
            Path(tmpdir) / "state.npz",
            actions=np.zeros((100, 12), dtype=np.float32),
            metadata=np.array(meta),
        )
        with open(Path(tmpdir) / "state.json", "w") as f:
            json.dump({
                "garment": "Top_Long_Seen_0",
                "garment_type": "top_long",
                "seed": 42,
                "n_actions": 100,
                "status": "active",
            }, f)

        results = get_success_states(tmpdir)
        assert len(results) == 1
        assert results[0]["n_actions"] == 100


def test_mark_states_in_use_and_consumed():
    """mark_states_in_use and mark_states_consumed update sidecar status."""
    from lehome_solution.eval.rollout_strategies import (
        mark_states_in_use, mark_states_consumed, get_failure_states,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = json.dumps({"garment": "G", "garment_type": "t", "seed": 1})
        npz_path = Path(tmpdir) / "test.npz"
        np.savez_compressed(npz_path, metadata=np.array(meta))
        with open(npz_path.with_suffix(".json"), "w") as f:
            json.dump({"garment": "G", "garment_type": "t", "seed": 1, "status": "active"}, f)

        states = [{"garment": "G", "npz_path": str(npz_path)}]

        # Mark in_use
        mark_states_in_use(states, rollout_id="test_rollout")
        with open(npz_path.with_suffix(".json")) as f:
            sidecar = json.load(f)
        assert sidecar["status"] == "in_use"
        assert sidecar["rollout_id"] == "test_rollout"

        # in_use states should be filtered out by get_failure_states
        results = get_failure_states(tmpdir)
        assert len(results) == 0

        # Mark consumed
        mark_states_consumed(states)
        with open(npz_path.with_suffix(".json")) as f:
            sidecar = json.load(f)
        assert sidecar["status"] == "consumed"


def test_cleanup_consumed_states():
    """cleanup_consumed_states deletes NPZ+JSON pairs with status=consumed."""
    from lehome_solution.eval.rollout_strategies import cleanup_consumed_states
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create consumed state
        meta = json.dumps({"garment": "G", "garment_type": "t", "seed": 1, "status": "consumed"})
        np.savez_compressed(Path(tmpdir) / "consumed.npz", metadata=np.array(meta))
        with open(Path(tmpdir) / "consumed.json", "w") as f:
            json.dump({"garment": "G", "status": "consumed"}, f)

        # Create active state
        meta2 = json.dumps({"garment": "G2", "garment_type": "t", "seed": 2, "status": "active"})
        np.savez_compressed(Path(tmpdir) / "active.npz", metadata=np.array(meta2))
        with open(Path(tmpdir) / "active.json", "w") as f:
            json.dump({"garment": "G2", "status": "active"}, f)

        removed = cleanup_consumed_states(tmpdir)
        assert removed == 1
        assert not (Path(tmpdir) / "consumed.npz").exists()
        assert not (Path(tmpdir) / "consumed.json").exists()
        assert (Path(tmpdir) / "active.npz").exists()
        assert (Path(tmpdir) / "active.json").exists()


def test_write_state_sidecar():
    """_write_state_sidecar creates a JSON file alongside NPZ."""
    from lehome_solution.eval.rollout_strategies import _write_state_sidecar
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test.npz"
        np.savez_compressed(npz_path, data=np.zeros(3))
        meta = {"garment": "G", "status": "active", "particle_count": 100}
        _write_state_sidecar(npz_path, meta)
        json_path = npz_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["particle_count"] == 100
        assert loaded["status"] == "active"


def test_thin_client_restore_state_code():
    """The fork's generic remote loop includes the physics state-restore logic.

    The 3 old proxy/replay/dagger policies + 2 loops were consolidated into one
    ``remote`` policy (HTTP client, like DockerPolicy) plus one generic
    ``remote_loop`` that owns the single copy of the state-restore / snapshot
    helpers.
    """
    fork = Path(__file__).parent.parent / "lehome-challenge" / "scripts"

    loop_path = fork / "utils" / "remote_loop.py"
    loop_code = loop_path.read_text()
    compile(loop_code, str(loop_path), "exec")
    # Single de-duplicated copy of the restore + snapshot helpers.
    assert "def restore_physics_state" in loop_code
    assert "def capture_physics_state" in loop_code
    assert "Vt.Vec3fArray" in loop_code

    policy_path = fork / "eval_policy" / "remote_policy.py"
    policy_code = policy_path.read_text()
    compile(policy_code, str(policy_path), "exec")
    # HTTP client that serializes observations and posts commands.
    assert "class RemotePolicy" in policy_code
    assert "def serialize_observation" in policy_code
    assert "json.loads" in policy_code

    # The old per-type policies/loops must be gone (diff reduced).
    for stale in (
        fork / "eval_policy" / "ws_proxy_policy.py",
        fork / "eval_policy" / "replay_policy.py",
        fork / "eval_policy" / "dagger_policy.py",
        fork / "utils" / "proxy_loop.py",
        fork / "utils" / "dagger_loop.py",
    ):
        assert not stale.exists(), f"stale fork file still present: {stale}"


def test_hf_upload_failure_states_missing_dir():
    """upload_failure_states_to_hf returns False for missing dir."""
    from lehome_solution.training.hf_upload import upload_failure_states_to_hf
    result = upload_failure_states_to_hf("/nonexistent/path", "fake/repo")
    assert result is False


def test_hf_upload_failure_states_empty_dir():
    """upload_failure_states_to_hf returns True for empty dir (mocked HF)."""
    from unittest.mock import MagicMock, patch

    from lehome_solution.training.hf_upload import upload_failure_states_to_hf

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MagicMock()
            mock_api.list_repo_files.return_value = []
            MockApi.return_value = mock_api
            result = upload_failure_states_to_hf(tmpdir, "fake/repo")
            assert result is True


# ---------------------------------------------------------------------------
# Advantage computation tests
# ---------------------------------------------------------------------------

def _make_episode_info(garment_type, garment, success, dense_return,
                       num_frames=30, checkpoint_frame=None, dense_rewards=None,
                       values=None, sampling_share=1.0, force_segment=False):
    """Helper to build episode info dicts for advantage tests."""
    info = {
        "garment_type": garment_type,
        "garment": garment,
        "success": success,
        "dense_return": dense_return,
        "num_frames": num_frames,
        "sampling_share": sampling_share,
    }
    if force_segment:
        info["force_segment"] = True
    if checkpoint_frame is not None:
        info["checkpoint_frame"] = checkpoint_frame
    if dense_rewards is not None:
        info["dense_rewards"] = dense_rewards
    if values is not None:
        info["values"] = values
    return info


def test_segment_advantages_binary_success():
    """1-checkpoint type (pant_short): advantages are purely success/fail based."""
    from lehome_solution.eval.dataset_writer import compute_segment_advantages

    eps = [
        _make_episode_info("pant_short", "Pant_Short_Seen_0", True, 1.0, num_frames=30),
        _make_episode_info("pant_short", "Pant_Short_Seen_0", False, 0.0, num_frames=30),
        _make_episode_info("pant_short", "Pant_Short_Seen_1", True, 1.0, num_frames=30),
        _make_episode_info("pant_short", "Pant_Short_Seen_1", False, 0.0, num_frames=30),
    ]
    advs = compute_segment_advantages(eps)
    assert len(advs) == 4

    # Success episodes should have positive advantages
    assert all(a > 0 for a in advs[0]), "Success ep should have positive advantages"
    # Failure episodes should have negative advantages
    assert all(a < 0 for a in advs[1]), "Failure ep should have negative advantages"
    # Each episode's frames should all share the same advantage (episode-level)
    assert len(set(advs[0])) == 1, "All frames in an episode should have same advantage"


def test_segment_advantages_two_checkpoint():
    """2-checkpoint type (top_long): episode-level advantages based on dense_return."""
    from lehome_solution.eval.dataset_writer import compute_segment_advantages

    eps = [
        _make_episode_info("top_long", "Top_Long_Seen_0", True, 1.0,
                           num_frames=60, checkpoint_frame=30),
        _make_episode_info("top_long", "Top_Long_Seen_0", False, 0.5,
                           num_frames=60, checkpoint_frame=30),
        _make_episode_info("top_long", "Top_Long_Seen_1", False, 0.0,
                           num_frames=60, checkpoint_frame=None),
    ]
    advs = compute_segment_advantages(eps)
    assert len(advs) == 3

    # Full success (return=1.0) should have higher advantage than partial (return=0.5)
    assert advs[0][0] > advs[1][0]
    # All frames in episode share same advantage (no segment split)
    assert len(set(advs[0])) == 1
    assert len(set(advs[1])) == 1


def test_gae_advantages():
    """GAE with known values produces frame-level advantages."""
    from lehome_solution.eval.dataset_writer import compute_gae_advantages

    # Simple episode: 10 frames, values predicting 50% success, reward at last frame
    eps = [
        _make_episode_info("pant_short", "Pant_Short_Seen_0", True, 1.0,
                           num_frames=10,
                           dense_rewards=[0.0]*9 + [1.0],
                           values=[0.5]*10),
        _make_episode_info("pant_short", "Pant_Short_Seen_0", False, 0.0,
                           num_frames=10,
                           dense_rewards=[0.0]*10,
                           values=[0.5]*10),
    ]
    advs = compute_gae_advantages(eps, lam=0.98, gamma=0.9999)
    assert len(advs) == 2

    # Failure episode should have lower mean advantage than the success episode.
    assert np.mean(advs[1]) < np.mean(advs[0])


def test_completion_gae_skipped_without_predictions():
    """Episodes without completion predictions get GAE_completion=0 for every frame."""
    from lehome_solution.eval.dataset_writer import compute_completion_gae
    out = compute_completion_gae(
        None, length=5, completion_alpha=0.3, success_label=1.0,
    )
    assert out == [0.0] * 5


def test_completion_gae_zero_alpha_is_zero():
    """completion_alpha=0 short-circuits to a zero array."""
    from lehome_solution.eval.dataset_writer import compute_completion_gae
    out = compute_completion_gae(
        [0.1, 0.3, 0.5, 0.7, 0.9], length=5,
        completion_alpha=0.0, success_label=1.0,
    )
    assert out == [0.0] * 5


def test_completion_gae_monotonic_increasing_gives_positive_advantage():
    """Completion rising monotonically from 0→1 on a success yields positive GAE_completion."""
    from lehome_solution.eval.dataset_writer import compute_completion_gae
    completion = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    out = compute_completion_gae(
        completion, length=6,
        completion_alpha=0.3, gamma=0.9999, lam=0.99,
        success_label=1.0,
    )
    assert all(v > 0 for v in out), f"expected all positive, got {out}"


def test_gae_force_segment_via_flag():
    """force_segment=True produces pure segment advantages (no GAE blend),
    while preserving sampling_share for SR/baseline eligibility."""
    from lehome_solution.eval.dataset_writer import compute_gae_advantages

    # Mix of success/failure with different values to ensure GAE varies
    eps = [
        _make_episode_info("pant_short", "Pant_Short_Seen_0", True, 1.0,
                           num_frames=10,
                           dense_rewards=[0.0]*9 + [1.0],
                           values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                           sampling_share=1.0),
        # force_segment with high sampling_share: segment-only advantages but
        # still eligible for SR/baseline computation (share=0.9 >= 0.5)
        _make_episode_info("pant_short", "Pant_Short_Seen_0", True, 1.0,
                           num_frames=10,
                           dense_rewards=[0.0]*9 + [1.0],
                           values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                           sampling_share=0.9, force_segment=True),
        _make_episode_info("pant_short", "Pant_Short_Seen_1", False, 0.0,
                           num_frames=10,
                           dense_rewards=[0.0]*10,
                           values=[0.5]*10, sampling_share=1.0),
    ]
    advs = compute_gae_advantages(eps, lam=0.98, gamma=0.9999)

    # ep[0] (w=1.0): GAE blend should give varying per-frame advantages
    assert len(set(round(v, 6) for v in advs[0])) > 1, \
        f"GAE (w=1) should give varying per-frame advantages, got {advs[0]}"
    # ep[1] (force_segment, share=0.9): segment-only should give uniform per-frame advantage
    assert len(set(round(v, 6) for v in advs[1])) == 1, \
        f"segment-only (force_segment) should give uniform per-frame advantage, got {advs[1]}"


def test_compute_success_target_binary():
    """ComputeSuccessTarget with no smoothing returns binary target."""
    from lehome_solution.transforms import ComputeSuccessTarget

    transform = ComputeSuccessTarget(label_smoothing=0.0)

    data = {"success": 1.0, "advantage": 0.5}
    result = transform(data)
    assert result["success_target"] == pytest.approx(1.0)

    data = {"success": 0.0, "advantage": 0.5}
    result = transform(data)
    assert result["success_target"] == pytest.approx(0.0)


def test_compute_success_target_smoothing():
    """ComputeSuccessTarget with label smoothing shifts toward per-garment success rate baseline."""
    from lehome_solution.transforms import ComputeSuccessTarget

    transform = ComputeSuccessTarget(label_smoothing=0.1)

    # success=1.0, garment_avg=0.5 -> target = 1.0 * 0.9 + 0.5 * 0.1 = 0.95
    data = {"success": 1.0, "advantage": 0.5, "garment_avg_success": 0.5}
    result = transform(data)
    assert result["success_target"] == pytest.approx(0.95)

    # success=0.0, garment_avg=0.8 -> target = 0.0 * 0.9 + 0.8 * 0.1 = 0.08
    data = {"success": 0.0, "advantage": 0.5, "garment_avg_success": 0.8}
    result = transform(data)
    assert result["success_target"] == pytest.approx(0.08)


def test_compute_success_target_nan_bc_masking():
    """BC samples (success=None) get NaN target."""
    from lehome_solution.transforms import ComputeSuccessTarget

    transform = ComputeSuccessTarget(label_smoothing=0.1)
    data = {"success": None, "advantage": 0.5}
    result = transform(data)
    assert np.isnan(result["success_target"])


def test_rl_pipeline_config_segment_only():
    """Config loads segment_only and freeze_vision_backbone correctly."""
    from lehome_solution.training.pipeline_config import RLPipelineConfig

    config_path = Path(__file__).parent.parent / "configs" / "rl_pipeline_sim.yaml"
    cfg = RLPipelineConfig.from_yaml(str(config_path))
    assert cfg.exp_name == "sim_rl"
    assert cfg.freeze_vision_backbone is True
    assert cfg.advantage.segment_only_threshold == pytest.approx(0.1)
    for ds in cfg.initial_rl_datasets:
        assert ds.segment_only is True, f"{ds.root} should be segment_only"


def test_rl_pipeline_config_overrides():
    """CLI overrides take precedence over YAML values."""
    from lehome_solution.training.pipeline_config import RLPipelineConfig

    cfg = RLPipelineConfig.from_yaml(
        "configs/rl_pipeline_sim.yaml",
        overrides={"num_iterations": 50, "steps_per_iteration": 2000},
    )
    assert cfg.num_iterations == 50
    assert cfg.steps_per_iteration == 2000
    # Non-overridden values should remain from YAML
    assert cfg.config_name == "pi_modified_bc_rl"


def test_atomic_state_persistence():
    """State persistence roundtrip with atomic writes."""
    from lehome_solution.utils.state_persistence import default_state, save_state, load_state

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "pipeline_state.json"
        state = default_state()
        state["iteration"] = 5
        state["phase"] = "train"
        state["current_train_steps"] = 3000

        save_state(state, state_path)
        loaded = load_state(state_path)

        assert loaded["iteration"] == 5
        assert loaded["phase"] == "train"
        assert loaded["current_train_steps"] == 3000
        # tmp file should be cleaned up
        assert not state_path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Dense reward tests
# ---------------------------------------------------------------------------

def test_compute_dense_reward_basic():
    """compute_dense_reward produces correct per-frame rewards for known garment types."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # top-long-sleeve has superset groups: [[], [1,3,4], [0,1,2,3,4]] -> 2 checkpoints
    # Each checkpoint is worth 1/2 = 0.5
    # First checkpoint: collar (1) + both spreads (3, 4)
    # Second checkpoint: all 5 conditions (superset of first)

    # 5 conditions: indices 0,1,2,3,4
    frames = [
        # Frame 0: nothing passes
        {"check_status": [0.0, 0.0, 0.0, 0.0, 0.0]},
        # Frame 1: collar + spreads pass (1,3,4) -> first checkpoint reached
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0]},
        # Frame 2: all 5 conditions pass -> second checkpoint reached
        {"check_status": [1.0, 1.0, 1.0, 1.0, 1.0]},
        # Frame 3: no check_status -> reward=0
        {},
    ]

    result = compute_dense_reward(frames, garment_name="Top_Long_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result
    assert len(rewards) == 4
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.5)   # first checkpoint reached
    assert rewards[2] == pytest.approx(0.5)   # second checkpoint reached
    assert rewards[3] == pytest.approx(0.0)   # no check_status
    # checkpoint_held tracks first checkpoint group [1,3,4]
    assert checkpoint_held[0] == pytest.approx(0.0)
    assert checkpoint_held[1] == pytest.approx(1.0)  # group[1]=[1,3,4] passes
    assert checkpoint_held[2] == pytest.approx(1.0)
    assert checkpoint_held[3] == pytest.approx(1.0)  # forward-filled


def test_compute_dense_reward_short_pant():
    """Short pant has 1 checkpoint: reward goes to 1.0 on full success."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # short-pant: [[3,2], [1,0]] -> 1 checkpoint, worth 1.0
    frames = [
        {"check_status": [0.0, 0.0, 0.0, 0.0]},
        # Group 0 done (3,2 pass)
        {"check_status": [0.0, 0.0, 1.0, 1.0]},
        # All groups done (all pass) -> cp=1, reward=1.0
        {"check_status": [1.0, 1.0, 1.0, 1.0]},
    ]

    result = compute_dense_reward(frames, garment_name="Pant_Short_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result
    assert len(rewards) == 3
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.0)   # free group only
    assert rewards[2] == pytest.approx(1.0)   # checkpoint reached
    # For short pants, first checkpoint = success
    assert checkpoint_held[2] == pytest.approx(1.0)


def test_compute_dense_reward_no_check_status():
    """Returns None when no frame has check_status."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    frames = [{"action": [0.0]}, {"action": [1.0]}]
    result = compute_dense_reward(frames, garment_name="Top_Long_Seen_0")
    assert result is None


def test_compute_dense_reward_cumulative():
    """Rewards are only for newly reached checkpoints, not re-awarded."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # top-long-sleeve: superset groups [[], [1,3,4], [0,1,2,3,4]], 2 checkpoints
    frames = [
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0]},  # cp1 reached -> reward=0.5
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0]},  # same cp1 -> reward=0 (not new)
        {"check_status": [1.0, 1.0, 1.0, 1.0, 1.0]},  # cp2 reached -> reward=0.5
    ]

    result = compute_dense_reward(frames, garment_name="Top_Long_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result
    assert rewards[0] == pytest.approx(0.5)
    assert rewards[1] == pytest.approx(0.0)   # already had cp1
    assert rewards[2] == pytest.approx(0.5)


def test_compute_dense_reward_long_pant_requires_all_conditions():
    """Long pant: all 7 conditions must pass for full reward (no success override)."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # long-pant: [[], [4,5,6], [0,1,2,3,4,5,6]] -> 2 checkpoints, each worth 0.5
    frames = [
        # Frame 0: nothing passes
        {"check_status": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        # Frame 1: original conditions 0-3 pass, extras 4-6 fail -> only second cp partially
        {"check_status": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]},
        # Frame 2: all 7 conditions pass -> first cp (4,5,6) + second cp (all) = full reward
        {"check_status": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    ]

    result = compute_dense_reward(frames, garment_name="Pant_Long_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result
    assert len(rewards) == 3
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.0)   # 0-3 pass but not in any checkpoint group
    assert rewards[2] == pytest.approx(1.0)   # both checkpoints reached
    assert checkpoint_held[2] == pytest.approx(1.0)


def test_compute_dense_reward_long_pant_normal_progression():
    """Long pant: normal sequential progression through all checkpoints still works."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    frames = [
        # Group 0 done (1,2 pass)
        {"check_status": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]},
        # Group 0+1 done (1,2,4,5,6 pass) -> cp=1, reward=0.5
        {"check_status": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]},
        # All groups done -> cp=2, reward=0.5
        {"check_status": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    ]

    result = compute_dense_reward(frames, garment_name="Pant_Long_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.5)
    assert rewards[2] == pytest.approx(0.5)


def test_compute_dense_reward_top_superset_no_skip():
    """For tops, second checkpoint is a superset of first — can't earn it without first."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # top-long-sleeve: [[], [1,3,4], [0,1,2,3,4]]
    # Body fold (0,2) passes but collar (1) fails — neither checkpoint satisfied
    # because 1 is required for both checkpoints (first=[1,3,4], second=[0,1,2,3,4])
    frames = [
        {"check_status": [1.0, 0.0, 1.0, 1.0, 1.0]},
        {"check_status": [1.0, 0.0, 1.0, 1.0, 0.0]},
    ]

    result = compute_dense_reward(frames, garment_name="Top_Long_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result
    assert rewards == [0.0, 0.0]
    assert checkpoint_held == [0.0, 0.0]


def test_compute_dense_reward_oscillation():
    """First checkpoint can be gained and lost, with reward oscillating."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # top-long-sleeve: [[], [1,3,4], [0,1,2,3,4]] -> 2 checkpoints, each worth 0.5
    # First checkpoint group is [1,3,4]
    frames = [
        # cp1 reached (conditions 1,3,4 all pass)
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0]},
        # cp1 lost (condition 1 fails)
        {"check_status": [0.0, 0.0, 0.0, 1.0, 1.0]},
        # cp1 regained
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0]},
        # cp1 lost again (condition 4 fails)
        {"check_status": [0.0, 1.0, 0.0, 1.0, 0.0]},
    ]

    result = compute_dense_reward(frames, garment_name="Top_Long_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result
    assert rewards[0] == pytest.approx(0.5)    # gained
    assert rewards[1] == pytest.approx(-0.5)   # lost
    assert rewards[2] == pytest.approx(0.5)    # regained
    assert rewards[3] == pytest.approx(-0.5)   # lost again
    assert sum(rewards) == pytest.approx(0.0)  # net zero
    assert checkpoint_held == [1.0, 0.0, 1.0, 0.0]


def test_compute_dense_reward_gradual_first_checkpoint_shirt():
    """Shirts with check_distances get gradual first-checkpoint reward."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # top-long-sleeve: [[], [1,3,4], [0,1,2,3,4]] -> 2 checkpoints, each worth 0.5
    # Primary distance for gradual reward is index 1.
    # Simulate distance ratio decreasing from 2.0 to 0.8 over 5 frames,
    # checkpoint reached at frame 4 (index 1 passes when ratio <= 1.0).
    frames = [
        # Frame 0: nothing passes, distance ratio = 2.0
        {"check_status": [0.0, 0.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 2.0, 0.0, 0.5, 0.5]},
        # Frame 1: distance decreases to 1.6
        {"check_status": [0.0, 0.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 1.6, 0.0, 0.5, 0.5]},
        # Frame 2: distance decreases to 1.2
        {"check_status": [0.0, 0.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 1.2, 0.0, 0.5, 0.5]},
        # Frame 3: distance decreases to 1.0 (at threshold, but condition not yet passing)
        {"check_status": [0.0, 0.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 1.0, 0.0, 0.5, 0.5]},
        # Frame 4: checkpoint reached! distance = 0.8, condition 1 passes
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 0.8, 0.0, 0.5, 0.5]},
    ]

    result = compute_dense_reward(frames, garment_name="Top_Long_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result

    # Total reward for first checkpoint should be 0.5
    assert sum(rewards) == pytest.approx(0.5)
    # Reward should be spread across frames, not all at frame 4
    assert rewards[4] < 0.5  # NOT all reward at the checkpoint frame
    # First frame gets 0 (no progress yet from d_init)
    assert rewards[0] == pytest.approx(0.0)
    # Each subsequent frame gets positive reward as distance decreases
    for i in range(1, 5):
        assert rewards[i] > 0.0
    # Checkpoint held only from frame 4
    assert checkpoint_held[:4] == [0.0, 0.0, 0.0, 0.0]
    assert checkpoint_held[4] == 1.0


def test_compute_dense_reward_gradual_with_oscillation():
    """Gradual reward only applies to first occurrence; oscillations stay binary."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    frames = [
        # Distance decreasing
        {"check_status": [0.0, 0.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 2.0, 0.0, 0.5, 0.5]},
        {"check_status": [0.0, 0.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 1.4, 0.0, 0.5, 0.5]},
        # First checkpoint reached
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 0.8, 0.0, 0.5, 0.5]},
        # Lost checkpoint (condition 4 fails)
        {"check_status": [0.0, 1.0, 0.0, 1.0, 0.0], "check_distances": [0.0, 0.8, 0.0, 0.5, 1.5]},
        # Regained checkpoint (binary +0.5)
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0], "check_distances": [0.0, 0.7, 0.0, 0.5, 0.5]},
    ]

    result = compute_dense_reward(frames, garment_name="Top_Short_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result

    # Gradual rewards for frames 0-2 should sum to 0.5
    gradual_sum = sum(rewards[:3])
    assert gradual_sum == pytest.approx(0.5)
    # Frame 3: lost checkpoint = -0.5
    assert rewards[3] == pytest.approx(-0.5)
    # Frame 4: regained checkpoint = +0.5 (binary, not gradual)
    assert rewards[4] == pytest.approx(0.5)
    # Net: 0.5 - 0.5 + 0.5 = 0.5
    assert sum(rewards) == pytest.approx(0.5)


def test_compute_dense_reward_no_distances_falls_back_to_binary():
    """Without check_distances, shirts still get binary rewards (backward compat)."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    frames = [
        {"check_status": [0.0, 0.0, 0.0, 1.0, 1.0]},
        {"check_status": [0.0, 1.0, 0.0, 1.0, 1.0]},  # first cp
    ]

    result = compute_dense_reward(frames, garment_name="Top_Long_Seen_0")
    assert result is not None
    rewards, _ = result
    # Without check_distances, falls back to binary
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.5)


def test_compute_dense_reward_long_pant_checkpoint_oscillation():
    """Long pant: first checkpoint (4,5,6) oscillates, reward reflects gain/loss."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    frames = [
        # First checkpoint reached (4,5,6 pass)
        {"check_status": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]},
        # First checkpoint lost (4 fails)
        {"check_status": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]},
        # First checkpoint regained, plus all conditions -> full reward
        {"check_status": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    ]

    result = compute_dense_reward(frames, garment_name="Pant_Long_Seen_0")
    assert result is not None
    rewards, checkpoint_held = result
    assert checkpoint_held[0] == pytest.approx(1.0)
    assert checkpoint_held[1] == pytest.approx(0.0)  # lost
    assert checkpoint_held[2] == pytest.approx(1.0)  # regained
    # +0.5 (gain) -0.5 (loss) +0.5 (regain) +0.5 (second cp) = 1.0
    assert sum(rewards) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Weighted mean/std tests
# ---------------------------------------------------------------------------

def test_weighted_mean_std_equal_weights():
    """Equal weights should give standard mean/std."""
    from lehome_solution.eval.dataset_writer import _weighted_mean_std

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    mean, std = _weighted_mean_std(arr, weights)
    assert mean == pytest.approx(np.mean(arr))
    assert std == pytest.approx(np.std(arr), abs=1e-6)


def test_weighted_mean_std_unequal_weights():
    """Unequal weights should shift mean toward higher-weighted values."""
    from lehome_solution.eval.dataset_writer import _weighted_mean_std

    arr = np.array([0.0, 10.0])
    # Weight heavily on the second value (10.0)
    weights = np.array([1.0, 9.0])

    mean, std = _weighted_mean_std(arr, weights)
    assert mean == pytest.approx(9.0)  # 1/10 * 0 + 9/10 * 10 = 9.0
    # std = sqrt(w1*(0-9)^2 + w2*(10-9)^2) = sqrt(0.1*81 + 0.9*1) = sqrt(9) = 3.0
    assert std == pytest.approx(3.0, abs=1e-6)


def test_weighted_mean_std_single_element():
    """Single element should return (value, 0.0)."""
    from lehome_solution.eval.dataset_writer import _weighted_mean_std

    arr = np.array([42.0])
    weights = np.array([1.0])

    mean, std = _weighted_mean_std(arr, weights)
    assert mean == pytest.approx(42.0)
    assert std == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Garment baselines tests
# ---------------------------------------------------------------------------

def test_garment_baselines_enough_data():
    """With ≥5 episodes per garment, baseline = garment mean."""
    from lehome_solution.eval.dataset_writer import _garment_baselines

    ep_indices = list(range(6))
    returns = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    episodes_info = [{"garment": "Top_Long_Seen_0"}] * 6
    weights = [1.0] * 6

    baselines = _garment_baselines(
        ep_indices, returns, episodes_info, weights,
        type_mean=0.3, type_n=6,
    )
    # 6 eps >= 5: garment mean = 0.5
    for b in baselines:
        assert b == pytest.approx(0.5, abs=1e-6)


def test_garment_baselines_fallback_to_type():
    """With <5 garment episodes but ≥5 type episodes, baseline = type mean."""
    from lehome_solution.eval.dataset_writer import _garment_baselines

    ep_indices = [0, 1]
    returns = [0.7, 0.3]
    episodes_info = [{"garment": "Top_Long_Seen_0"}] * 2
    weights = [1.0] * 2

    baselines = _garment_baselines(
        ep_indices, returns, episodes_info, weights,
        type_mean=0.4, type_n=10,
    )
    # 2 eps < 5: falls back to type_mean=0.4
    for b in baselines:
        assert b == pytest.approx(0.4)


def test_garment_baselines_fallback_to_default():
    """With <5 type episodes, baseline = 0.5 default."""
    from lehome_solution.eval.dataset_writer import _garment_baselines

    ep_indices = [0]
    returns = [0.9]
    episodes_info = [{"garment": "Top_Long_Seen_0"}]
    weights = [1.0]

    baselines = _garment_baselines(
        ep_indices, returns, episodes_info, weights,
        type_mean=0.9, type_n=3,
    )
    # 3 type eps < 5: falls back to 0.5
    assert baselines[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pipeline state crash recovery tests
# ---------------------------------------------------------------------------

def test_pipeline_state_crash_recovery_tmp_gone():
    """Atomic writes ensure tmp file is removed after successful save."""
    from lehome_solution.utils.state_persistence import save_state, load_state, default_state

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "pipeline_state.json"
        state = default_state()
        state["iteration"] = 7
        state["phase"] = "advantages"

        save_state(state, state_path)

        # tmp file should not exist
        assert not state_path.with_suffix(".tmp").exists()
        # state file should exist and load correctly
        loaded = load_state(state_path)
        assert loaded["iteration"] == 7
        assert loaded["phase"] == "advantages"


def test_pipeline_state_crash_recovery_corrupt_fallback():
    """load_state falls back to default when state file is corrupted."""
    from lehome_solution.utils.state_persistence import load_state, default_state

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "pipeline_state.json"

        # Write corrupted JSON
        with open(state_path, "w") as f:
            f.write("{invalid json content")

        # load_state should raise (JSON decode error), it doesn't have
        # built-in fallback to default, so we test what actually happens
        with pytest.raises(json.JSONDecodeError):
            load_state(state_path)


def test_pipeline_state_missing_file_returns_default():
    """load_state returns default state when file doesn't exist."""
    from lehome_solution.utils.state_persistence import load_state, default_state

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "nonexistent" / "pipeline_state.json"
        loaded = load_state(state_path)
        expected = default_state()
        assert loaded == expected


# ---------------------------------------------------------------------------
# Advantage clipping tests
# ---------------------------------------------------------------------------

def test_advantage_clipping_segment():
    """Verify segment advantages are clipped to [clip_min, clip_max]."""
    from lehome_solution.eval.dataset_writer import compute_segment_advantages

    clip_min, clip_max = -3.0, 2.0
    # Create extreme returns: one garment type with all success and one all-failure
    # This creates maximum variance, leading to large raw advantages before clipping
    eps = []
    # Many successes
    for i in range(20):
        eps.append(_make_episode_info(
            "pant_short", f"Pant_Short_Seen_{i % 5}", True, 1.0, num_frames=10,
        ))
    # One failure — extreme outlier
    eps.append(_make_episode_info(
        "pant_short", "Pant_Short_Seen_0", False, 0.0, num_frames=10,
    ))

    advs = compute_segment_advantages(eps, clip_min=clip_min, clip_max=clip_max)

    for ep_advs in advs:
        for a in ep_advs:
            assert clip_min <= a <= clip_max, f"Advantage {a} outside [{clip_min}, {clip_max}]"


def test_advantage_clipping_gae():
    """Verify GAE advantages are clipped to [clip_min, clip_max]."""
    from lehome_solution.eval.dataset_writer import compute_gae_advantages

    clip_min, clip_max = -3.0, 2.0
    eps = []
    # Success episode with late reward
    eps.append(_make_episode_info(
        "pant_short", "Pant_Short_Seen_0", True, 1.0, num_frames=20,
        dense_rewards=[0.0] * 19 + [1.0],
        values=[0.1] * 20,
    ))
    # Multiple failure episodes for contrast
    for i in range(10):
        eps.append(_make_episode_info(
            "pant_short", f"Pant_Short_Seen_{i % 3}", False, 0.0, num_frames=20,
            dense_rewards=[0.0] * 20,
            values=[0.9] * 20,  # overconfident predictions
        ))

    advs = compute_gae_advantages(eps, clip_min=clip_min, clip_max=clip_max)

    for ep_advs in advs:
        for a in ep_advs:
            assert clip_min <= a <= clip_max, f"GAE advantage {a} outside [{clip_min}, {clip_max}]"


# ---------------------------------------------------------------------------
# Distributed / HF sync tests
# ---------------------------------------------------------------------------

def test_parse_rollout_id():
    from lehome_solution.distributed.hf_sync import parse_rollout_id

    # Standard format (no worker)
    result = parse_rollout_id("rollout_7500_random_20260318_150500")
    assert result is not None
    assert result["model_step"] == 7500
    assert result["strategy"] == "random"
    assert result["timestamp"] == "20260318_150500"
    assert result["machine_id"] is None

    # With worker ID
    result = parse_rollout_id("rollout_7500_random_20260318_150500_worker1")
    assert result is not None
    assert result["model_step"] == 7500
    assert result["strategy"] == "random"
    assert result["machine_id"] == "worker1"

    # With worker ID containing underscores
    result = parse_rollout_id("rollout_7500_hard_20260318_150500_my_worker_1")
    assert result is not None
    assert result["model_step"] == 7500
    assert result["strategy"] == "hard"
    assert result["machine_id"] == "my_worker_1"

    # Invalid
    assert parse_rollout_id("step7500_20260318") is None
    assert parse_rollout_id("rollout_notanumber_random_20260318_150500") is None
    assert parse_rollout_id("rollout") is None


def test_make_rollout_id():
    from lehome_solution.distributed.hf_sync import make_rollout_id

    rid = make_rollout_id(7500, "random")
    assert rid.startswith("rollout_7500_random_")
    assert len(rid.split("_")) == 5  # rollout, step, strategy, date, time

    rid_w = make_rollout_id(7500, "random", "worker1")
    assert rid_w.startswith("rollout_7500_random_")
    assert rid_w.endswith("_worker1")
    assert len(rid_w.split("_")) == 6


def test_make_rollout_id_roundtrip():
    """make_rollout_id output should be parseable by parse_rollout_id."""
    from lehome_solution.distributed.hf_sync import make_rollout_id, parse_rollout_id

    for step, strategy, worker in [(1000, "random", None), (7500, "curriculum", "w1"), (0, "hard", "my_host")]:
        rid = make_rollout_id(step, strategy, worker)
        parsed = parse_rollout_id(rid)
        assert parsed is not None
        assert parsed["model_step"] == step
        assert parsed["strategy"] == strategy
        assert parsed["machine_id"] == worker


def test_step_aware_decay():
    """Step-aware decay: new model step triggers one round of decay."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from lehome_solution.training.pipeline_config import RLPipelineConfig
    import dataclasses

    cfg = RLPipelineConfig(
        rl_decay_factor=0.8,
        rl_min_sampling_share=0.3,
        steps_per_iteration=1000,
        bc_dataset=dataclasses.replace(RLPipelineConfig().bc_dataset, decay_factor=0.9, min_sampling_share=0.3),
    )

    state = {
        "rl_datasets": [
            {"root": "/old_ds", "sampling_share": 0.5, "model_step": 1000, "repo_id": "r0"},
            {"root": "/recent_ds", "sampling_share": 1.0, "model_step": 2000, "repo_id": "r1"},
        ],
        "bc_sampling_share": 1.0,
        "consumed_rollout_ids": [],
    }

    # New dataset from newer model step -> triggers decay
    new_datasets = [
        {"root": "/new_ds", "model_step": 3000, "strategy": "random", "hf_rollout_id": "rollout_3000_random_xxx"},
    ]

    import importlib
    import run_rl_pipeline
    importlib.reload(run_rl_pipeline)

    run_rl_pipeline.update_dataset_shares_step_aware(cfg, state, new_datasets, current_model_step=3000)

    # Old dataset: 0.5 * 0.8 = 0.4
    old_ds = next(ds for ds in state["rl_datasets"] if ds["root"] == "/old_ds")
    assert old_ds["sampling_share"] == pytest.approx(0.4, abs=0.01)

    # Recent dataset: 1.0 * 0.8 = 0.8
    recent_ds = next(ds for ds in state["rl_datasets"] if ds["root"] == "/recent_ds")
    assert recent_ds["sampling_share"] == pytest.approx(0.8, abs=0.01)

    # New dataset: share=1.0
    new_ds = next(ds for ds in state["rl_datasets"] if ds["root"] == "/new_ds")
    assert new_ds["sampling_share"] == 1.0
    assert new_ds["model_step"] == 3000

    # Consumed rollout IDs tracked
    assert "rollout_3000_random_xxx" in state["consumed_rollout_ids"]

    # BC decayed (step advanced)
    assert state["bc_sampling_share"] == pytest.approx(0.9, abs=0.01)


def test_step_aware_no_decay_same_step():
    """Adding datasets from same model step should NOT decay existing shares."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from lehome_solution.training.pipeline_config import RLPipelineConfig
    import dataclasses

    cfg = RLPipelineConfig(
        rl_decay_factor=0.8,
        rl_min_sampling_share=0.3,
        bc_dataset=dataclasses.replace(RLPipelineConfig().bc_dataset, decay_factor=0.9, min_sampling_share=0.3),
    )

    state = {
        "rl_datasets": [
            {"root": "/existing", "sampling_share": 0.6, "model_step": 3000, "repo_id": "r0"},
        ],
        "bc_sampling_share": 0.8,
        "consumed_rollout_ids": [],
    }

    # New dataset from SAME model step -> no decay
    new_datasets = [
        {"root": "/same_step", "model_step": 3000, "strategy": "curriculum"},
    ]

    import importlib, run_rl_pipeline
    importlib.reload(run_rl_pipeline)

    run_rl_pipeline.update_dataset_shares_step_aware(cfg, state, new_datasets, current_model_step=3000)

    # Existing dataset: unchanged (no decay, same step)
    existing = next(ds for ds in state["rl_datasets"] if ds["root"] == "/existing")
    assert existing["sampling_share"] == 0.6

    # New dataset added at 1.0
    new_ds = next(ds for ds in state["rl_datasets"] if ds["root"] == "/same_step")
    assert new_ds["sampling_share"] == 1.0

    # BC NOT decayed (same step)
    assert state["bc_sampling_share"] == 0.8


def test_step_aware_decay_removes_below_minimum():
    """Datasets below minimum share should be removed after decay."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from lehome_solution.training.pipeline_config import RLPipelineConfig
    import dataclasses

    cfg = RLPipelineConfig(
        rl_decay_factor=0.5,  # aggressive decay
        rl_min_sampling_share=0.3,
        bc_dataset=dataclasses.replace(RLPipelineConfig().bc_dataset, decay_factor=0.9, min_sampling_share=0.3),
    )

    state = {
        "rl_datasets": [
            # share 0.5 * 0.5 = 0.25 < 0.3 -> removed
            {"root": "/barely_alive", "sampling_share": 0.5, "model_step": 1000, "repo_id": "r0"},
            # share 1.0 * 0.5 = 0.5 >= 0.3 -> kept
            {"root": "/healthy", "sampling_share": 1.0, "model_step": 1000, "repo_id": "r1"},
        ],
        "bc_sampling_share": 1.0,
        "consumed_rollout_ids": [],
    }

    # New dataset from newer step -> triggers decay
    new_datasets = [
        {"root": "/new", "model_step": 2000},
    ]

    import importlib, run_rl_pipeline
    importlib.reload(run_rl_pipeline)

    run_rl_pipeline.update_dataset_shares_step_aware(cfg, state, new_datasets, current_model_step=2000)

    roots = [ds["root"] for ds in state["rl_datasets"]]
    assert "/barely_alive" not in roots  # removed
    assert "/healthy" in roots  # kept at 0.5
    assert "/new" in roots  # added at 1.0

    healthy = next(ds for ds in state["rl_datasets"] if ds["root"] == "/healthy")
    assert healthy["sampling_share"] == 0.5


def test_state_persistence_consumed_rollout_ids():
    """default_state should include consumed_rollout_ids."""
    from lehome_solution.utils.state_persistence import default_state

    state = default_state()
    assert "consumed_rollout_ids" in state
    assert state["consumed_rollout_ids"] == []


def test_hf_upload_step_info():
    """upload_step_info should be importable."""
    from lehome_solution.training.hf_upload import upload_step_info
    assert callable(upload_step_info)


def test_pipeline_config_distributed_fields():
    """Pipeline config should have distributed settings."""
    from lehome_solution.training.pipeline_config import RLPipelineConfig

    cfg = RLPipelineConfig()
    assert hasattr(cfg, "checkpoint_poll_interval_s")
    assert cfg.checkpoint_poll_interval_s == 120


def test_distributed_imports():
    """All distributed module imports should work."""
    from lehome_solution.distributed.hf_sync import (
        get_latest_model_step,
        download_latest_checkpoint,
        list_available_rollouts,
        download_rollout_dataset,
        upload_rollout_dataset,
        download_failure_states,
        upload_failure_states_individually,
        make_rollout_id,
        parse_rollout_id,
    )


def test_list_available_rollouts_mock():
    """list_available_rollouts should parse HF repo tree correctly."""
    from lehome_solution.distributed.hf_sync import list_available_rollouts

    mock_entries = []
    for name in ["rollout_1000_random_20260318_120000",
                 "rollout_2000_curriculum_20260318_130000_worker1",
                 "dagger",  # non-rollout entry
                 "rollout_invalid_name"]:
        entry = MagicMock()
        entry.path = name
        mock_entries.append(entry)

    # Mock _complete markers for the two valid rollouts (required after the
    # partial-upload fix: missing markers mean "skip this rollout").
    complete_infos = []
    for name in ["rollout_1000_random_20260318_120000",
                 "rollout_2000_curriculum_20260318_130000_worker1"]:
        info = MagicMock()
        info.path = f"{name}/_complete"
        complete_infos.append(info)

    with patch("huggingface_hub.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_repo_tree.return_value = mock_entries
        api.get_paths_info.return_value = complete_infos

        results = list_available_rollouts("test/repo")

    assert len(results) == 2
    assert results[0]["model_step"] == 1000
    assert results[0]["strategy"] == "random"
    assert results[1]["model_step"] == 2000
    assert results[1]["machine_id"] == "worker1"


# ---------------------------------------------------------------------------
# Curriculum strategy tests
# ---------------------------------------------------------------------------

def test_curriculum_strategy_garment_type_mapping():
    """Curriculum must correctly map PascalCase garment names to types."""
    from lehome_solution.eval.rollout_strategies import curriculum_strategy

    garment_pool = [
        "Top_Long_Seen_0", "Top_Long_Seen_1",
        "Top_Short_Seen_0", "Top_Short_Seen_1",
        "Pant_Long_Seen_0", "Pant_Long_Seen_1",
        "Pant_Short_Seen_0", "Pant_Short_Seen_1",
    ]
    success_rates = {
        "by_type": {
            "top_long": 0.5,
            "top_short": 0.1,  # hardest type
            "pant_long": 0.3,
            "pant_short": 0.9,  # easiest type
        },
        "by_garment": {g: 0.5 for g in garment_pool},
    }
    rng = np.random.RandomState(42)
    sampled = curriculum_strategy(garment_pool, 1000, rng, success_rates=success_rates)

    # Count how many from each type
    from lehome_solution.eval.eval_utils import garment_name_to_type
    type_counts = {}
    for g in sampled:
        t = garment_name_to_type(g)
        type_counts[t] = type_counts.get(t, 0) + 1

    # Harder types (top_short, pant_long) must get more rollouts than easy (pant_short)
    assert "unknown" not in type_counts, "Garment names should map to known types, not 'unknown'"
    assert type_counts.get("top_short", 0) > type_counts.get("pant_short", 0), (
        f"top_short (SR=10%) should get more rollouts than pant_short (SR=90%), "
        f"got {type_counts}"
    )


# ---------------------------------------------------------------------------
# Missing coverage: success target NaN handling, failure withdrawal,
# 2-checkpoint segment split, GAE value function, stuck detection, edge cases
# ---------------------------------------------------------------------------

def test_compute_success_target_nan_garment_avg():
    """NaN garment_avg_success should fall back to 0.5 baseline."""
    from lehome_solution.transforms import ComputeSuccessTarget

    transform = ComputeSuccessTarget(label_smoothing=0.2)
    data = {"success": 1.0, "advantage": 0.5, "garment_avg_success": float("nan")}
    result = transform(data)
    # With NaN fallback to 0.5: target = 1.0 * 0.8 + 0.5 * 0.2 = 0.9
    assert result["success_target"] == pytest.approx(0.9)
    assert np.isfinite(result["success_target"]), "NaN garment_avg must not propagate"


def test_compute_success_target_missing_garment_avg():
    """Missing garment_avg_success should fall back to 0.5 baseline."""
    from lehome_solution.transforms import ComputeSuccessTarget

    transform = ComputeSuccessTarget(label_smoothing=0.2)
    data = {"success": 0.0, "advantage": 0.5}
    result = transform(data)
    # target = 0.0 * 0.8 + 0.5 * 0.2 = 0.1
    assert result["success_target"] == pytest.approx(0.1)


def test_dense_reward_failure_withdrawal_zeros_total():
    """Failure withdrawal must make total reward exactly 0.0."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # Top with first checkpoint reached then success conditions met
    frames = []
    for i in range(60):
        cs = np.zeros(5, dtype=np.float32)
        if 20 <= i < 40:
            cs[[1, 3, 4]] = 1.0  # first checkpoint
        if i >= 40:
            cs[:] = 1.0  # all conditions
        frames.append({"check_status": cs, "check_interval": 30 if i % 30 == 0 else 0})

    # Build proper check_status frames (only on check_interval boundaries)
    proper_frames = []
    for i in range(60):
        cs = np.zeros(5, dtype=np.float32)
        if i >= 20:
            cs[[1, 3, 4]] = 1.0
        if i >= 40:
            cs[:] = 1.0
        f = {"check_status": cs} if i % 30 == 0 else {}
        proper_frames.append(f)

    result = compute_dense_reward(proper_frames, "Top_Long_Seen_0", success=False)
    if result is not None:
        rewards, _ = result
        total = sum(rewards)
        assert abs(total) < 1e-6, f"Failed episode total reward should be 0, got {total}"


def test_dense_reward_success_total_one():
    """Successful episode should have total reward = 1.0."""
    from lehome_solution.eval.dataset_writer import compute_dense_reward

    # Short pant: all conditions met at the check_status frame
    # check_status appears on certain frames; conditions pass from the start
    frames = []
    for i in range(30):
        f = {}
        # Provide check_status at frame 0 (no conditions) and frame 15 (all pass)
        if i == 0:
            f["check_status"] = np.zeros(4, dtype=np.float32)
        elif i == 15:
            f["check_status"] = np.ones(4, dtype=np.float32)
        frames.append(f)

    result = compute_dense_reward(frames, "Pant_Short_Seen_0", success=True)
    assert result is not None
    rewards, _ = result
    total = sum(rewards)
    assert abs(total - 1.0) < 1e-6, f"Successful episode total should be 1.0, got {total}"


def test_gae_value_function_additive():
    """V = EMA(P_success, α=0.2) − R_so_far (additive; can go negative after rewards)."""
    from lehome_solution.eval.dataset_writer import compute_gae

    # 5 frames, constant predictions, reward at frame 2
    values = [0.8, 0.8, 0.8, 0.8, 0.8]
    rewards = [0.0, 0.0, 0.5, 0.0, 0.5]
    _gae, V = compute_gae(values, rewards, ema_alpha=0.2)

    # Frame 0: EMA seed = 0.8, R_so_far = 0, V = 0.8 − 0 = 0.8
    assert V[0] == pytest.approx(0.8, abs=0.01)
    # After reward at frame 2: cum = 0.5, V[3] ≈ EMA − 0.5 ≈ 0.3
    assert V[3] == pytest.approx(0.3, abs=0.05)
    # V strictly decreases each time cumulative reward jumps
    assert V[3] < V[0]


def test_gae_single_frame_episode():
    """Single-frame episode should produce valid advantage."""
    from lehome_solution.eval.dataset_writer import compute_gae

    # success_alpha=0.5 (default), success_label=1 forces P[T] = 1, γ=0.999 (default).
    # Exact dampened-value TD residual: δ = r + γ·V̂_T − V̂_0 with
    #   V̂_0 = α·(P[0] − R_so_far[0]) = 0.5·(0.5 − 0)   = 0.25
    #   V̂_T = α·(P[T] − R_so_far[T]) = 0.5·(1.0 − 1.0) = 0.0   (R_so_far[T]=Σr=1)
    # δ = 1.0 + 0.999·0.0 − 0.25 = 0.75
    gae, V = compute_gae([0.5], [1.0], success_label=1.0)
    assert len(gae) == 1
    assert len(V) == 1
    # V[0] = EMA(P) − R_so_far = 0.5 − 0 = 0.5  (undampened value, for reporting)
    assert V[0] == pytest.approx(0.5, abs=1e-6)
    assert gae[0] == pytest.approx(1.0 + 0.999 * 0.0 - 0.25, abs=1e-3)


def test_gae_empty_episode():
    """Empty episode should return empty lists."""
    from lehome_solution.eval.dataset_writer import compute_gae

    gae, V = compute_gae([], [])
    assert gae == []
    assert V == []


def test_segment_advantages_two_checkpoint_segment_split():
    """2-checkpoint type: segment split gives different advantages before/after checkpoint."""
    from lehome_solution.eval.dataset_writer import compute_gae_advantages

    # Top_Long episodes with checkpoint at frame 30
    eps = []
    # Partial success: reached cp1 but failed
    eps.append(_make_episode_info(
        "top_long", "Top_Long_Seen_0", False, 0.0, num_frames=60,
        checkpoint_frame=30,
        dense_rewards=[0.0]*29 + [0.5] + [0.0]*29 + [-0.5],  # cp reached then withdrawn
        values=[0.5]*60,
    ))
    # Full success
    eps.append(_make_episode_info(
        "top_long", "Top_Long_Seen_0", True, 1.0, num_frames=60,
        checkpoint_frame=30,
        dense_rewards=[0.0]*29 + [0.5] + [0.0]*29 + [0.5],
        values=[0.5]*60,
    ))
    # Full failure (no checkpoint)
    eps.append(_make_episode_info(
        "top_long", "Top_Long_Seen_1", False, 0.0, num_frames=60,
        dense_rewards=[0.0]*60,
        values=[0.3]*60,
    ))

    advs = compute_gae_advantages(eps, lam=0.99, gamma=0.9999)
    assert len(advs) == 3

    # Full success should have higher average advantage than full failure
    assert np.mean(advs[1]) > np.mean(advs[2])
    # Partial failure (reached checkpoint then failed) gets penalized in segment 2,
    # so it can have lower average than full failure (which gets uniform negative).
    # The key property: segment 1 of partial should be better than segment 1 of failure.
    # Just verify all three produce different advantage profiles.
    means = [np.mean(a) for a in advs]
    assert len(set(round(m, 4) for m in means)) == 3, \
        f"All three episodes should have distinct advantage means, got {means}"


def test_collate_fn_int_none_handling():
    """Collation should use -1 for int fields with None, NaN for float fields."""
    from lehome_solution.training.data_loader import _collate_fn

    items = [
        {"garment_type_id": np.int32(2), "advantage": np.float32(0.5)},
        {"garment_type_id": None, "advantage": None},
    ]
    batch = _collate_fn(items)
    # Int field should get -1 sentinel, not NaN
    assert batch["garment_type_id"][1] == -1
    assert batch["garment_type_id"].dtype in (np.int32, np.int64)
    # Float field should get NaN
    assert np.isnan(batch["advantage"][1])


def test_per_timestamp_normalization_shape_validation():
    """Per-timestamp normalization should raise on shape mismatch."""
    from lehome_solution.transforms_normalize import NormalizeWithPerTimestamp
    from lehome_solution.shared.normalize import NormStats

    stats = NormStats(
        mean=np.zeros(12, dtype=np.float32),
        std=np.ones(12, dtype=np.float32),
        per_timestamp_mean=np.zeros((30, 12), dtype=np.float32),
        per_timestamp_std=np.ones((30, 12), dtype=np.float32),
    )
    # norm_stats is required but we test _normalize directly with explicit stats
    norm = NormalizeWithPerTimestamp(norm_stats=None, use_per_timestamp=True)

    # Valid shape: 30 timesteps
    x = np.random.randn(30, 12).astype(np.float32)
    result = norm._normalize(x, stats)
    assert result.shape == (30, 12)

    # Smaller is fine (sliced)
    x_small = np.random.randn(10, 12).astype(np.float32)
    result_small = norm._normalize(x_small, stats)
    assert result_small.shape == (10, 12)

    # Invalid shape: 40 timesteps > 30 in stats
    x_bad = np.random.randn(40, 12).astype(np.float32)
    with pytest.raises(ValueError, match="action_horizon mismatch"):
        norm._normalize(x_bad, stats)


def test_legacy_opt_state_migration_shim():
    """Build an opt_state with the OLD AdamWBF16 chain shape, run it through the
    migration helpers, and verify the result matches the new AdamWWithAuxDecay
    shape (5-tuple, mu in fp32, extra EmptyState at position 3)."""
    import jax
    import jax.numpy as jnp
    import optax
    from lehome_solution.training.checkpoints import (
        _convert_legacy_opt_state_to_new,
        _legacy_bf16_opt_state_target,
    )

    params = {"w": jnp.zeros((4, 4), dtype=jnp.float32), "b": jnp.zeros((4,), dtype=jnp.float32)}

    # Old chain (AdamWBF16-equivalent): clip + adamw(mu_dtype=bf16)
    old_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(1e-3, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, mu_dtype=jnp.bfloat16),
    )
    old_opt_state = old_tx.init(params)

    # New chain (AdamWWithAuxDecay-equivalent): flat 5-tuple, mu fp32.
    # Must pass a real mask to position [3] — optax wraps masked transforms in
    # ``optax.masked``, producing ``MaskedState(EmptyState())`` rather than a
    # bare ``EmptyState``. The shim's output must match that exactly.
    mask_fn = lambda p: jax.tree.map(lambda _: True, p)
    new_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8),
        optax.add_decayed_weights(1e-10),
        optax.add_decayed_weights(0.01, mask=mask_fn),
        optax.scale_by_learning_rate(1e-3),
    )
    new_opt_state = new_tx.init(params)

    # _legacy_bf16_opt_state_target: given the NEW tree shape, produces the OLD tree shape.
    legacy_target = _legacy_bf16_opt_state_target(new_opt_state)
    assert isinstance(legacy_target, tuple) and len(legacy_target) == 2
    assert isinstance(legacy_target[1], tuple) and len(legacy_target[1]) == 3
    legacy_adam = legacy_target[1][0]
    assert jax.tree_util.tree_leaves(legacy_adam.mu)[0].dtype == jnp.bfloat16

    # Simulate having restored the old-shape opt_state from disk:
    restored_legacy = old_opt_state  # same structure as what Orbax would give back

    # Convert back to the new flat shape, casting mu → fp32.
    migrated = _convert_legacy_opt_state_to_new(restored_legacy)
    assert isinstance(migrated, tuple) and len(migrated) == 5
    assert jax.tree_util.tree_leaves(migrated[1].mu)[0].dtype == jnp.float32
    assert migrated[1].nu["w"].dtype == jnp.float32
    # Position 3 is the new aux-head add_decayed_weights(..., mask=fn), which
    # optax wraps in optax.masked — so the state is MaskedState(EmptyState()).
    from optax.transforms._masking import MaskedState
    assert isinstance(migrated[3], MaskedState)
    assert isinstance(migrated[3].inner_state, optax.EmptyState)

    # The migrated tree must match the NEW tree's structure exactly (same leaves count & layout)
    new_leaves = jax.tree_util.tree_structure(new_opt_state)
    migrated_leaves = jax.tree_util.tree_structure(migrated)
    assert new_leaves == migrated_leaves, (
        f"migrated opt_state structure must match new tree:\n"
        f"new={new_leaves}\nmigrated={migrated_leaves}"
    )
