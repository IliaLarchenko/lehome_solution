# Eager re-export of `normalize` keeps the legacy
# `from lehome_solution.shared import normalize as _normalize` callers (config,
# checkpoints, compute_norm_stats) working with a single import. Wrapped in
# try/except so the package stays importable from venvs that lack the
# main-venv-only `numpydantic` dep (the challenge venv, used by the real-robot
# scripts, only pulls in lerobot-flavoured helpers from this subpackage and
# never reaches `normalize`).
try:
    from lehome_solution.shared import normalize
    __all__ = ["normalize"]
except ImportError:
    __all__ = []
