"""HF sync daemon client: submit requests and poll results.

Used by the main pipeline process to communicate with the background
HF sync daemon via file-based protocol (JSON marker files).
"""

import json
import logging
import time
from pathlib import Path

from lehome_solution.distributed.hf_sync_protocol import (
    DEFAULT_PRIORITIES,
    DIR_QUEUE,
    DIR_READY,
    DIR_RESULTS,
    SyncRequest,
    SyncResult,
    atomic_write_json,
    make_request_id,
)
from lehome_solution.exceptions import HFSyncError

logger = logging.getLogger(__name__)

# How often to poll when waiting for a blocking result
_POLL_INTERVAL_S = 2.0


class HFSyncClient:
    """Client for submitting requests to the HF sync daemon.

    If sync_dir is None, all operations are no-ops (no daemon configured).
    """

    def __init__(self, sync_dir: str | Path | None):
        if sync_dir is None:
            self._sync_dir = None
            return
        self._sync_dir = Path(sync_dir)
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create sync directory structure if needed."""
        if self._sync_dir is None:
            return
        for subdir in [DIR_QUEUE, DIR_RESULTS, DIR_READY]:
            (self._sync_dir / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._sync_dir is not None

    def submit(
        self,
        op_type: str,
        params: dict,
        *,
        priority: int | None = None,
        blocking: bool = False,
        max_retries: int = 3,
    ) -> str:
        """Submit a request to the daemon. Returns request_id.

        Writes a JSON file to queue/ that the daemon picks up.
        """
        if self._sync_dir is None:
            return ""

        if priority is None:
            priority = DEFAULT_PRIORITIES.get(op_type, 3)

        request_id = make_request_id(op_type)
        req = SyncRequest(
            request_id=request_id,
            op_type=op_type,
            priority=priority,
            params=params,
            created_at=time.time(),
            retry_count=0,
            max_retries=max_retries,
            blocking=blocking,
        )

        queue_dir = self._sync_dir / DIR_QUEUE
        queue_dir.mkdir(parents=True, exist_ok=True)
        req_path = queue_dir / f"{request_id}.json"
        atomic_write_json(req_path, req.to_json())

        logger.debug("Submitted %s request: %s", op_type, request_id)
        return request_id

    def get_result(self, request_id: str) -> SyncResult | None:
        """Non-blocking check for a result. Returns None if not ready."""
        if self._sync_dir is None:
            return None

        result_path = self._sync_dir / DIR_RESULTS / f"{request_id}.json"
        if not result_path.exists():
            return None

        try:
            text = result_path.read_text()
            return SyncResult.from_json(text)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("Corrupt result file %s: %s", result_path, e)
            return None

    def wait_for(self, request_id: str, timeout: float = 3600) -> SyncResult:
        """Block until result is available or timeout.

        Raises HFSyncError on timeout.
        """
        if self._sync_dir is None:
            raise HFSyncError("No sync daemon configured")

        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get_result(request_id)
            if result is not None:
                if not result.success:
                    raise HFSyncError(
                        f"Request {request_id} failed: {result.error}"
                    )
                return result
            time.sleep(_POLL_INTERVAL_S)

        raise HFSyncError(
            f"Timeout waiting for request {request_id} after {timeout}s"
        )

    def submit_and_wait(
        self,
        op_type: str,
        params: dict,
        *,
        priority: int | None = None,
        max_retries: int = 3,
        timeout: float = 3600,
    ) -> SyncResult:
        """Submit a request and wait for the result."""
        request_id = self.submit(
            op_type,
            params,
            priority=priority,
            blocking=True,
            max_retries=max_retries,
        )
        return self.wait_for(request_id, timeout=timeout)

    def is_daemon_alive(self) -> bool:
        """Check if the daemon process is still running."""
        if self._sync_dir is None:
            return False

        pid_file = self._sync_dir / "daemon.pid"
        if not pid_file.exists():
            return False

        try:
            pid = int(pid_file.read_text().strip())
            # Check if process exists (signal 0 = no signal, just check)
            import os
            os.kill(pid, 0)
            return True
        except (ValueError, OSError):
            return False

    def get_ready_datasets(self) -> list[str]:
        """Read list of downloaded dataset paths from ready/datasets.json."""
        if self._sync_dir is None:
            return []

        ready_file = self._sync_dir / DIR_READY / "datasets.json"
        if not ready_file.exists():
            return []

        try:
            data = json.loads(ready_file.read_text())
            return data.get("datasets", [])
        except (json.JSONDecodeError, TypeError):
            return []

    def get_ready_checkpoint(self) -> dict | None:
        """Read latest checkpoint info from ready/checkpoint.json."""
        if self._sync_dir is None:
            return None

        ready_file = self._sync_dir / DIR_READY / "checkpoint.json"
        if not ready_file.exists():
            return None

        try:
            return json.loads(ready_file.read_text())
        except (json.JSONDecodeError, TypeError):
            return None

    def get_ready_states(self, kind: str) -> dict | None:
        """Read state sync status from ready/{kind}_states.json."""
        if self._sync_dir is None:
            return None

        ready_file = self._sync_dir / DIR_READY / f"{kind}_states.json"
        if not ready_file.exists():
            return None

        try:
            return json.loads(ready_file.read_text())
        except (json.JSONDecodeError, TypeError):
            return None

    def get_ready_assets(self) -> list[str]:
        """Read list of downloaded asset names from ready/assets.json."""
        if self._sync_dir is None:
            return []

        ready_file = self._sync_dir / DIR_READY / "assets.json"
        if not ready_file.exists():
            return []

        try:
            data = json.loads(ready_file.read_text())
            return data.get("assets", [])
        except (json.JSONDecodeError, TypeError):
            return []

    def submit_fire_and_forget(
        self,
        op_type: str,
        params: dict,
        *,
        priority: int | None = None,
        max_retries: int = 3,
    ) -> str:
        """Submit a request without waiting for the result. Returns request_id."""
        return self.submit(
            op_type, params,
            priority=priority, blocking=False, max_retries=max_retries,
        )
