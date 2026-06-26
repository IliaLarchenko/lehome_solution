"""Test that all core modules import without errors."""

import pytest


def test_import_lehome_solution():
    import lehome_solution


def test_import_openpi():
    import openpi


def test_import_lerobot():
    import lerobot
    assert lerobot.__version__ >= "0.4.0"


def test_import_training_config():
    from lehome_solution.training.config import TrainConfig, get_config, AdamWWithAuxDecay


def test_import_models():
    from lehome_solution.models.pi_modified_config import PiModifiedConfig
    from lehome_solution.models.pi_modified import PiModified
    from lehome_solution.models.observation import Observation


def test_import_policies():
    from lehome_solution.policies.pi_modified_policy import PiModifiedPolicy
    from lehome_solution.policies.lehome_policy import LeHomeInputs, LeHomeOutputs


def test_import_shared():
    from lehome_solution.shared.eval_wrapper import LeHomePolicyWrapper, resize_with_pad
    from lehome_solution.shared.normalize import NormStats


def test_import_transforms():
    from lehome_solution.transforms import TokenizeFASTActions


def test_import_data_loader():
    from lehome_solution.training.data_loader import (
        create_lehome_dataset,
        create_lehome_data_loader,
        transform_dataset,
        extract_episode_lengths_from_dataset,
        TransformedDataset,
        TorchDataLoader,
        FakeDataset,
    )


def test_import_flash_attention_patch():
    from lehome_solution.training.flash_attention_patch import install


def test_import_memory_report():
    from lehome_solution.training.memory_report import print_memory_report


def test_no_omnigibson_imports():
    """Ensure no omnigibson dependency leaks into our code."""
    import sys
    import lehome_solution.training.config
    import lehome_solution.training.data_loader
    assert "omnigibson" not in sys.modules
