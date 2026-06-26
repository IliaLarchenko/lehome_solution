"""Structured metadata for eval runs and episodes.

Used by the dataset writer and for logging; reusable for RL and analysis.

Reward: We save \"return\" (cumulative episode reward = sum of step rewards) per episode.
The challenge env computes detailed reward components (primary/secondary condition
scores, per-condition value/threshold/passed) in garment_bi_v2._get_rewards() and
success_checker_chanllege; only the scalar step reward is returned and the eval
script only logs Return= and Success=. Detailed breakdown is not currently
exposed in the eval output.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

# Import from constants (lightweight, no JAX dependency).
from lehome_solution.constants import GARMENT_TYPE_TO_ID


def garment_name_to_type_id(garment_name: str) -> int:
    """Map garment name (e.g. Top_Long_Unseen_0) to garment_type_id (0–3). Returns 0 for unknown."""
    from lehome_solution.eval.eval_utils import garment_name_to_type
    return GARMENT_TYPE_TO_ID.get(garment_name_to_type(garment_name), 0)


@dataclasses.dataclass
class EpisodeMetadata:
    """Per-episode result and env data (from eval)."""

    # Identity and env result (no defaults — required)
    garment: str
    seed: int
    split: str  # "Seen" | "Unseen"
    success: bool
    return_: float  # Cumulative episode reward (sum of step rewards)
    length: int
    episode: int
    total: int

    # Optional / with defaults (must follow required fields in dataclass)
    # Garment-type id (0=top_long, 1=top_short, 2=pant_long, 3=pant_short). Stored only in
    # eval_episode_metadata/*.json to avoid confusion with LeRobot task_index.
    garment_type_id: int = 0
    # Extra env fields (e.g. from parser; future: reward_details if challenge logs them)
    extra: dict[str, Any] | None = None
    # Rollout type: "normal", "hard_mining", or "success_replay"
    rollout_type: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "garment": self.garment,
            "seed": self.seed,
            "split": self.split,
            "garment_type_id": self.garment_type_id,
            "success": self.success,
            "return": self.return_,
            "reward": self.return_,  # alias: total episode reward
            "length": self.length,
            "episode": self.episode,
            "total": self.total,
            "rollout_type": self.rollout_type,
        }
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_episode_result(cls, ep: dict[str, Any]) -> EpisodeMetadata:
        garment = ep.get("garment", "")
        known_keys = {"garment", "seed", "split", "garment_type_id", "success", "return", "length", "episode", "total", "rollout_type"}
        return cls(
            garment=garment,
            seed=ep.get("seed", 0),
            split=ep.get("split", ""),
            garment_type_id=ep.get("garment_type_id", garment_name_to_type_id(garment)),
            success=ep.get("success", False),
            return_=ep.get("return", 0.0),
            length=ep.get("length", 0),
            episode=ep.get("episode", 0),
            total=ep.get("total", 1),
            rollout_type=ep.get("rollout_type", "normal"),
            extra={k: v for k, v in ep.items() if k not in known_keys},
        )


@dataclasses.dataclass
class EvalRunMetadata:
    """Run-level parameters and info (checkpoint, time, args)."""

    # Run id and time
    eval_run_id: str
    started_at: str  # ISO format

    # Model
    checkpoint_dir: str
    config_name: str

    # Eval config
    garment_types: list[str]
    garment_subset: str  # "unseen" | "seen" | "all"
    num_episodes: int
    max_steps: int
    num_workers: int
    port: int

    # Paths
    video_dir: str
    dataset_dir: str | None = None

    # Optional
    wandb_project: str | None = None
    no_wandb: bool = False
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "eval_run_id": self.eval_run_id,
            "started_at": self.started_at,
            "checkpoint_dir": self.checkpoint_dir,
            "config_name": self.config_name,
            "garment_types": self.garment_types,
            "garment_subset": self.garment_subset,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "num_workers": self.num_workers,
            "port": self.port,
            "video_dir": self.video_dir,
            "dataset_dir": self.dataset_dir,
            "wandb_project": self.wandb_project,
            "no_wandb": self.no_wandb,
        }
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_args(cls, eval_run_id: str, video_dir: str, dataset_dir: str | None, args: Any) -> EvalRunMetadata:
        return cls(
            eval_run_id=eval_run_id,
            started_at=datetime.utcnow().isoformat() + "Z",
            checkpoint_dir=getattr(args, "checkpoint_dir", ""),
            config_name=getattr(args, "config_name", ""),
            garment_types=list(getattr(args, "garment_types", [])),
            garment_subset=getattr(args, "garment_subset", "unseen"),
            num_episodes=getattr(args, "num_episodes", 0),
            max_steps=getattr(args, "max_steps", 0),
            num_workers=getattr(args, "num_workers", 0),
            port=getattr(args, "port", 0),
            video_dir=video_dir,
            dataset_dir=dataset_dir,
            wandb_project=getattr(args, "wandb_project", None),
            no_wandb=getattr(args, "no_wandb", False),
        )
