"""Tests for HF sync daemon protocol, client, and daemon logic."""

import json
import os
import signal
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------

class TestSyncRequest:
    def test_roundtrip(self):
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest
        req = SyncRequest(
            request_id="123_upload_checkpoint_456",
            op_type="upload_checkpoint",
            priority=1,
            params={"checkpoint_dir": "/tmp/ckpt", "step": 100},
            created_at=1234567890.0,
            retry_count=0,
            max_retries=3,
            blocking=False,
        )
        text = req.to_json()
        loaded = SyncRequest.from_json(text)
        assert loaded.request_id == req.request_id
        assert loaded.op_type == req.op_type
        assert loaded.priority == req.priority
        assert loaded.params == req.params
        assert loaded.created_at == req.created_at
        assert loaded.retry_count == req.retry_count
        assert loaded.max_retries == req.max_retries
        assert loaded.blocking == req.blocking

    def test_json_is_valid(self):
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest
        req = SyncRequest(
            request_id="test",
            op_type="shutdown",
            priority=99,
            params={},
            created_at=0.0,
        )
        data = json.loads(req.to_json())
        assert data["request_id"] == "test"
        assert data["op_type"] == "shutdown"


class TestSyncResult:
    def test_roundtrip(self):
        from lehome_solution.distributed.hf_sync_protocol import SyncResult
        result = SyncResult(
            request_id="abc",
            success=True,
            result={"step": 42},
            error=None,
            completed_at=1234567890.0,
            duration_s=3.14,
        )
        text = result.to_json()
        loaded = SyncResult.from_json(text)
        assert loaded.request_id == result.request_id
        assert loaded.success == result.success
        assert loaded.result == result.result
        assert loaded.error is None
        assert loaded.duration_s == pytest.approx(3.14)

    def test_failure_result(self):
        from lehome_solution.distributed.hf_sync_protocol import SyncResult
        result = SyncResult(
            request_id="fail",
            success=False,
            error="Network timeout",
            completed_at=0.0,
            duration_s=0.0,
        )
        loaded = SyncResult.from_json(result.to_json())
        assert not loaded.success
        assert loaded.error == "Network timeout"


class TestAtomicWriteJson:
    def test_write_creates_file(self):
        from lehome_solution.distributed.hf_sync_protocol import atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            atomic_write_json(path, '{"key": "value"}')
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["key"] == "value"

    def test_write_creates_parent_dirs(self):
        from lehome_solution.distributed.hf_sync_protocol import atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "test.json"
            atomic_write_json(path, '{"nested": true}')
            assert path.exists()

    def test_write_no_tmp_file_left(self):
        from lehome_solution.distributed.hf_sync_protocol import atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            atomic_write_json(path, '{}')
            tmp_path = path.with_suffix(".tmp")
            assert not tmp_path.exists()


class TestMakeRequestId:
    def test_contains_op_type(self):
        from lehome_solution.distributed.hf_sync_protocol import make_request_id
        rid = make_request_id("upload_checkpoint")
        assert "upload_checkpoint" in rid

    def test_contains_pid(self):
        from lehome_solution.distributed.hf_sync_protocol import make_request_id
        rid = make_request_id("test")
        assert str(os.getpid()) in rid

    def test_unique(self):
        from lehome_solution.distributed.hf_sync_protocol import make_request_id
        ids = {make_request_id("test") for _ in range(100)}
        # Within 1ms they might collide, but most should be unique
        assert len(ids) >= 1


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------

class TestHFSyncClient:
    def test_disabled_client(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        client = HFSyncClient(None)
        assert not client.enabled
        assert client.submit("test", {}) == ""
        assert client.get_result("test") is None
        assert client.get_ready_datasets() == []
        assert client.get_ready_checkpoint() is None

    def test_submit_creates_queue_file(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            rid = client.submit("upload_checkpoint", {"step": 42})
            assert rid != ""
            queue_dir = Path(tmpdir) / "queue"
            files = list(queue_dir.glob("*.json"))
            assert len(files) == 1
            data = json.loads(files[0].read_text())
            assert data["op_type"] == "upload_checkpoint"
            assert data["params"]["step"] == 42

    def test_get_result_returns_none_when_missing(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            assert client.get_result("nonexistent") is None

    def test_get_result_reads_result(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.distributed.hf_sync_protocol import SyncResult, atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            result = SyncResult(
                request_id="test_req",
                success=True,
                result={"step": 100},
                completed_at=time.time(),
                duration_s=1.0,
            )
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir(exist_ok=True)
            atomic_write_json(results_dir / "test_req.json", result.to_json())

            loaded = client.get_result("test_req")
            assert loaded is not None
            assert loaded.success
            assert loaded.result["step"] == 100

    def test_is_daemon_alive_no_pid_file(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            assert not client.is_daemon_alive()

    def test_is_daemon_alive_with_current_pid(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            pid_file = Path(tmpdir) / "daemon.pid"
            pid_file.write_text(str(os.getpid()))
            assert client.is_daemon_alive()

    def test_is_daemon_alive_with_dead_pid(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            pid_file = Path(tmpdir) / "daemon.pid"
            # Use a PID that's extremely unlikely to exist
            pid_file.write_text("999999999")
            assert not client.is_daemon_alive()

    def test_get_ready_datasets(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.distributed.hf_sync_protocol import atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            ready_dir = Path(tmpdir) / "ready"
            ready_dir.mkdir(exist_ok=True)
            atomic_write_json(
                ready_dir / "datasets.json",
                json.dumps({"datasets": ["/path/to/ds1", "/path/to/ds2"]}),
            )
            datasets = client.get_ready_datasets()
            assert len(datasets) == 2
            assert "/path/to/ds1" in datasets

    def test_get_ready_checkpoint(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.distributed.hf_sync_protocol import atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            ready_dir = Path(tmpdir) / "ready"
            ready_dir.mkdir(exist_ok=True)
            atomic_write_json(
                ready_dir / "checkpoint.json",
                json.dumps({"checkpoint_path": "/tmp/ckpt", "step": 1000}),
            )
            info = client.get_ready_checkpoint()
            assert info is not None
            assert info["step"] == 1000

    def test_wait_for_raises_on_timeout(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.exceptions import HFSyncError
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            with pytest.raises(HFSyncError, match="Timeout"):
                client.wait_for("nonexistent", timeout=0.1)

    def test_submit_uses_default_priority(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.distributed.hf_sync_protocol import (
            PRIORITY_HIGH, SyncRequest,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            rid = client.submit("upload_checkpoint", {"step": 1})
            queue_dir = Path(tmpdir) / "queue"
            files = list(queue_dir.glob("*.json"))
            req = SyncRequest.from_json(files[0].read_text())
            assert req.priority == PRIORITY_HIGH

    def test_submit_with_custom_priority(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            rid = client.submit("upload_checkpoint", {"step": 1}, priority=0)
            queue_dir = Path(tmpdir) / "queue"
            files = list(queue_dir.glob("*.json"))
            req = SyncRequest.from_json(files[0].read_text())
            assert req.priority == 0


# ---------------------------------------------------------------------------
# Daemon logic tests
# ---------------------------------------------------------------------------

class TestDaemonHelpers:
    def test_scan_queue_empty(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _scan_queue
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_dir = Path(tmpdir)
            (sync_dir / "queue").mkdir()
            requests = _scan_queue(sync_dir)
            assert requests == []

    def test_scan_queue_sorted_by_priority(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _scan_queue
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest, atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_dir = Path(tmpdir)
            queue_dir = sync_dir / "queue"
            queue_dir.mkdir()

            # Low priority
            req1 = SyncRequest("req1", "upload_states", 3, {}, time.time())
            atomic_write_json(queue_dir / "req1.json", req1.to_json())

            # High priority
            req2 = SyncRequest("req2", "upload_checkpoint", 1, {}, time.time())
            atomic_write_json(queue_dir / "req2.json", req2.to_json())

            requests = _scan_queue(sync_dir)
            assert len(requests) == 2
            assert requests[0].request_id == "req2"  # Higher priority first
            assert requests[1].request_id == "req1"

    def test_cleanup_done(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _cleanup_done
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_dir = Path(tmpdir)
            done_dir = sync_dir / "done"
            done_dir.mkdir()

            # Create an old file
            old_file = done_dir / "old.json"
            old_file.write_text("{}")
            # Set modification time to 25 hours ago
            old_mtime = time.time() - 25 * 3600
            os.utime(str(old_file), (old_mtime, old_mtime))

            # Create a recent file
            new_file = done_dir / "new.json"
            new_file.write_text("{}")

            _cleanup_done(sync_dir)

            assert not old_file.exists()
            assert new_file.exists()

    def test_kill_stale_daemon_no_pid_file(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _kill_stale_daemon
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            _kill_stale_daemon(Path(tmpdir))

    def test_kill_stale_daemon_dead_pid(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _kill_stale_daemon
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_file = Path(tmpdir) / "daemon.pid"
            pid_file.write_text("999999999")
            _kill_stale_daemon(Path(tmpdir))
            assert not pid_file.exists()


# ---------------------------------------------------------------------------
# Pipeline config tests
# ---------------------------------------------------------------------------

class TestPipelineConfigNewFields:
    def test_hf_sync_enabled_default(self):
        from lehome_solution.training.pipeline_config import RLPipelineConfig
        cfg = RLPipelineConfig()
        assert cfg.hf_sync_enabled is True

    def test_hf_sync_enabled_false(self):
        from lehome_solution.training.pipeline_config import RLPipelineConfig
        cfg = RLPipelineConfig(hf_sync_enabled=False)
        assert cfg.hf_sync_enabled is False

    def test_config_from_yaml_with_new_fields(self):
        from lehome_solution.training.pipeline_config import RLPipelineConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"
            import yaml
            config_path.write_text(yaml.dump({
                "config_name": "test",
                "exp_name": "test_exp",
                "hf_sync_enabled": False,
            }))
            cfg = RLPipelineConfig.from_yaml(config_path)
            assert cfg.hf_sync_enabled is False


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------

class TestHFSyncError:
    def test_exception_hierarchy(self):
        from lehome_solution.exceptions import HFSyncError, LeHomeError
        assert issubclass(HFSyncError, LeHomeError)

    def test_exception_message(self):
        from lehome_solution.exceptions import HFSyncError
        err = HFSyncError("test error")
        assert str(err) == "test error"


# ---------------------------------------------------------------------------
# Integration-level tests
# ---------------------------------------------------------------------------

class TestDaemonReadyFiles:
    def test_update_ready_datasets(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _update_ready_datasets
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_dir = Path(tmpdir)
            _update_ready_datasets(sync_dir, "/path/to/ds1")
            _update_ready_datasets(sync_dir, "/path/to/ds2")
            # Adding duplicate should not create duplicates
            _update_ready_datasets(sync_dir, "/path/to/ds1")

            ready_file = sync_dir / "ready" / "datasets.json"
            data = json.loads(ready_file.read_text())
            assert len(data["datasets"]) == 2
            assert "/path/to/ds1" in data["datasets"]
            assert "/path/to/ds2" in data["datasets"]

    def test_update_ready_checkpoint(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _update_ready_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_dir = Path(tmpdir)
            _update_ready_checkpoint(sync_dir, {"checkpoint_path": "/tmp/ckpt", "step": 100})

            ready_file = sync_dir / "ready" / "checkpoint.json"
            data = json.loads(ready_file.read_text())
            assert data["step"] == 100
            assert "updated_at" in data


class TestClientServerRoundtrip:
    """Test the full client -> queue -> result cycle (without real daemon)."""

    def test_submit_and_manual_result(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.distributed.hf_sync_protocol import SyncResult, atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            rid = client.submit("upload_checkpoint", {"step": 42})

            # Simulate daemon writing a result
            result = SyncResult(
                request_id=rid,
                success=True,
                result={"step": 42},
                completed_at=time.time(),
                duration_s=0.5,
            )
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir(exist_ok=True)
            atomic_write_json(results_dir / f"{rid}.json", result.to_json())

            # Client should find it
            loaded = client.get_result(rid)
            assert loaded is not None
            assert loaded.success
            assert loaded.result["step"] == 42

    def test_multiple_submits_creates_multiple_files(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            rid1 = client.submit("upload_checkpoint", {"step": 1})
            rid2 = client.submit("upload_dataset", {"path": "/tmp"})
            rid3 = client.submit("upload_states", {"kind": "failure"})

            queue_dir = Path(tmpdir) / "queue"
            files = list(queue_dir.glob("*.json"))
            assert len(files) == 3


# ---------------------------------------------------------------------------
# New protocol operation types
# ---------------------------------------------------------------------------

class TestNewProtocolOps:
    def test_new_op_types_exist(self):
        from lehome_solution.distributed.hf_sync_protocol import (
            OP_UPLOAD_DAGGER_VALUES,
            OP_SYNC_DAGGER,
            OP_DOWNLOAD_MISSING,
        )
        assert OP_UPLOAD_DAGGER_VALUES == "upload_dagger_values"
        assert OP_SYNC_DAGGER == "sync_dagger"
        assert OP_DOWNLOAD_MISSING == "download_missing"

    def test_new_ops_have_default_priorities(self):
        from lehome_solution.distributed.hf_sync_protocol import (
            DEFAULT_PRIORITIES,
            OP_UPLOAD_DAGGER_VALUES,
            OP_SYNC_DAGGER,
            OP_DOWNLOAD_MISSING,
        )
        assert OP_UPLOAD_DAGGER_VALUES in DEFAULT_PRIORITIES
        assert OP_SYNC_DAGGER in DEFAULT_PRIORITIES
        assert OP_DOWNLOAD_MISSING in DEFAULT_PRIORITIES


# ---------------------------------------------------------------------------
# Client new methods
# ---------------------------------------------------------------------------

class TestClientNewMethods:
    def test_get_ready_assets_empty(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            assert client.get_ready_assets() == []

    def test_get_ready_assets_with_data(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.distributed.hf_sync_protocol import atomic_write_json
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            ready_dir = Path(tmpdir) / "ready"
            ready_dir.mkdir(exist_ok=True)
            atomic_write_json(
                ready_dir / "assets.json",
                json.dumps({"assets": ["inference_prior.json", "success_rates.json"]}),
            )
            assets = client.get_ready_assets()
            assert len(assets) == 2
            assert "inference_prior.json" in assets

    def test_get_ready_assets_disabled_client(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        client = HFSyncClient(None)
        assert client.get_ready_assets() == []

    def test_submit_fire_and_forget(self):
        from lehome_solution.distributed.hf_sync_client import HFSyncClient
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest
        with tempfile.TemporaryDirectory() as tmpdir:
            client = HFSyncClient(tmpdir)
            rid = client.submit_fire_and_forget("upload_checkpoint", {"step": 99})
            assert rid != ""
            queue_dir = Path(tmpdir) / "queue"
            files = list(queue_dir.glob("*.json"))
            assert len(files) == 1
            req = SyncRequest.from_json(files[0].read_text())
            assert not req.blocking


# ---------------------------------------------------------------------------
# Daemon new handlers
# ---------------------------------------------------------------------------

class TestDaemonNewHandlers:
    def test_execute_request_shutdown(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _execute_request
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_dir = Path(tmpdir)
            req = SyncRequest(
                "test_shutdown",
                "shutdown",
                99,
                {},
                time.time(),
            )
            result = _execute_request(req, sync_dir)
            assert result["shutdown"] is True

    def test_execute_request_upload_dagger_no_repo(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _execute_request
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest
        with tempfile.TemporaryDirectory() as tmpdir:
            req = SyncRequest(
                "test_dagger",
                "upload_dagger_values",
                3,
                {"repo_id": None, "dagger_datasets": []},
                time.time(),
            )
            result = _execute_request(req, Path(tmpdir))
            assert result["uploaded"] == 0

    def test_execute_request_download_missing_no_datasets(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _execute_request
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest
        with tempfile.TemporaryDirectory() as tmpdir:
            req = SyncRequest(
                "test_missing",
                "download_missing",
                2,
                {"repo_id": None, "datasets": []},
                time.time(),
            )
            result = _execute_request(req, Path(tmpdir))
            assert result["downloaded"] == 0

    def test_execute_request_sync_dagger_no_repo(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _execute_request
        from lehome_solution.distributed.hf_sync_protocol import SyncRequest
        with tempfile.TemporaryDirectory() as tmpdir:
            req = SyncRequest(
                "test_sync_dag",
                "sync_dagger",
                2,
                {"repo_id": None},
                time.time(),
            )
            result = _execute_request(req, Path(tmpdir))
            assert result["sessions_downloaded"] == 0

    def test_update_ready_assets(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _update_ready_assets
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_dir = Path(tmpdir)
            _update_ready_assets(sync_dir, "inference_prior.json")
            _update_ready_assets(sync_dir, "success_rates.json")
            # Duplicates should not create duplicates
            _update_ready_assets(sync_dir, "inference_prior.json")

            ready_file = sync_dir / "ready" / "assets.json"
            data = json.loads(ready_file.read_text())
            assert len(data["assets"]) == 2
            assert "inference_prior.json" in data["assets"]
            assert "success_rates.json" in data["assets"]


# ---------------------------------------------------------------------------
# Daemon mode and polling
# ---------------------------------------------------------------------------

class TestDaemonPolling:
    def test_poll_trainer_no_repo(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _poll_trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise with None repos
            _poll_trainer(Path(tmpdir), None, None)

    def test_poll_rollout_no_repo(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from hf_sync_daemon import _poll_rollout
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise with None repos
            _poll_rollout(Path(tmpdir), None, None)
