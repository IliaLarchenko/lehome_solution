"""Shared fixtures and configuration for tests."""

import os
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (use -m 'not slow' to skip)")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


@pytest.fixture(scope="session", autouse=True)
def set_env():
    """Set environment variables for testing."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
