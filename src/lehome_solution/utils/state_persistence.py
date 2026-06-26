"""Pipeline state persistence with atomic writes."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILENAME = "pipeline_state.json"


def default_state() -> dict:
    return {
        "iteration": 0,
        "phase": "warmup",  # warmup | train | done
        "current_train_steps": 0,
        "bc_sampling_share": 1.0,
        "rl_datasets": [],  # [{root, sampling_share, repo_id}]
        "dagger_datasets": [],  # [{root, sampling_share, repo_id}] — fixed share, never decayed
        "wandb_run_id": None,
        "checkpoint_dir": None,
        "latest_checkpoint_step": None,
        "consumed_rollout_ids": [],  # HF rollout IDs already downloaded by trainer
    }


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return default_state()


def save_state(state: dict, state_path: Path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(state_path)
    logger.info("State saved: iter=%d phase=%s steps=%d",
                state["iteration"], state["phase"], state["current_train_steps"])
