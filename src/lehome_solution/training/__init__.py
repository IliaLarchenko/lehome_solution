# Eager re-export of training submodules keeps the
# `from lehome_solution.training import config as _config` pattern working for
# main-venv callers (policies.policy_config, etc.). Wrapped in try/except so
# importing leaf modules like `real_data_transforms` (which is pure numpy/scipy
# and used from the lerobot/challenge venv) doesn't fail when the heavier JAX
# stack (`config` → etils → jax) isn't installed.
try:
    from lehome_solution.training import config
    from lehome_solution.training import data_loader
    from lehome_solution.training import weight_loaders
    __all__ = ["config", "data_loader", "weight_loaders"]
except ImportError:
    __all__ = []
