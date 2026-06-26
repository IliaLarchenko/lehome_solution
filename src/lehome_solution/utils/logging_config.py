"""Centralized log path construction for all pipeline components.

All logs go under ``{base_dir}/logs/{component}/{filename}``.
"""

from pathlib import Path

# Component subdirectory names
TRAINING = "training"
ROLLOUT = "rollout"
ADVANTAGES = "advantages"
SERVER = "server"
ORCHESTRATOR = "orchestrator"
ISAAC_SIM = "isaac_sim"
HF_SYNC = "hf_sync"


def log_path(base_dir: str | Path, component: str, filename: str) -> Path:
    """Build a log path under ``{base_dir}/logs/{component}/{filename}``.

    Creates the parent directory if it doesn't exist.
    """
    p = Path(base_dir) / "logs" / component / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
