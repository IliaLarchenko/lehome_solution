"""HF sync daemon protocol: shared constants and data structures."""

import dataclasses
import json
import os
import time
from pathlib import Path


# Operation types
OP_UPLOAD_CHECKPOINT = "upload_checkpoint"
OP_DOWNLOAD_CHECKPOINT = "download_checkpoint"
OP_UPLOAD_DATASET = "upload_dataset"
OP_UPLOAD_DAGGER_ROLLOUT = "upload_dagger_rollout"
OP_DOWNLOAD_DATASET = "download_dataset"
OP_UPLOAD_STATES = "upload_states"
OP_CLEANUP_STATES = "cleanup_states"
OP_UPLOAD_ASSET = "upload_asset"
OP_DOWNLOAD_ASSET = "download_asset"
OP_UPLOAD_DAGGER_VALUES = "upload_dagger_values"
OP_SYNC_DAGGER = "sync_dagger"
OP_DOWNLOAD_MISSING = "download_missing"
OP_SHUTDOWN = "shutdown"

# Priorities (lower = higher priority)
PRIORITY_HIGH = 1      # checkpoint operations
PRIORITY_NORMAL = 2    # dataset operations
PRIORITY_LOW = 3       # state/asset operations

# Default priorities per operation type
DEFAULT_PRIORITIES = {
    OP_UPLOAD_CHECKPOINT: PRIORITY_HIGH,
    OP_DOWNLOAD_CHECKPOINT: PRIORITY_HIGH,
    OP_UPLOAD_DATASET: PRIORITY_NORMAL,
    OP_UPLOAD_DAGGER_ROLLOUT: PRIORITY_NORMAL,
    OP_DOWNLOAD_DATASET: PRIORITY_NORMAL,
    OP_UPLOAD_STATES: PRIORITY_LOW,
    OP_CLEANUP_STATES: PRIORITY_LOW,
    OP_UPLOAD_ASSET: PRIORITY_LOW,
    OP_DOWNLOAD_ASSET: PRIORITY_LOW,
    OP_UPLOAD_DAGGER_VALUES: PRIORITY_LOW,
    OP_SYNC_DAGGER: PRIORITY_NORMAL,
    OP_DOWNLOAD_MISSING: PRIORITY_NORMAL,
    OP_SHUTDOWN: 99,
}

# Subdirectory names within sync_dir
DIR_QUEUE = "queue"
DIR_ACTIVE = "active"
DIR_DONE = "done"
DIR_FAILED = "failed"
DIR_RESULTS = "results"
DIR_READY = "ready"


@dataclasses.dataclass
class SyncRequest:
    request_id: str
    op_type: str
    priority: int
    params: dict
    created_at: float  # time.time()
    retry_count: int = 0
    max_retries: int = 3
    blocking: bool = False

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "SyncRequest":
        return cls(**json.loads(text))


@dataclasses.dataclass
class SyncResult:
    request_id: str
    success: bool
    result: dict | None = None
    error: str | None = None
    completed_at: float = 0.0
    duration_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "SyncResult":
        return cls(**json.loads(text))


def atomic_write_json(path: Path, content: str):
    """Write JSON atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def make_request_id(op_type: str) -> str:
    """Generate a unique request ID."""
    return f"{int(time.time() * 1000)}_{op_type}_{os.getpid()}"
