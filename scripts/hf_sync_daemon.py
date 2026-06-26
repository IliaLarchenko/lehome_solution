#!/usr/bin/env python3
"""HF sync daemon: background process for all HuggingFace uploads/downloads.

Processes requests from a file-based queue, one at a time, in priority order.
Communicates with the main pipeline via JSON marker files.

Supports three modes:
  - trainer: polls HF for new rollout datasets, downloads automatically
  - rollout: polls HF for new checkpoints, downloads automatically
  - pipeline: no automatic polling (queue-only)

Usage:
    uv run python scripts/hf_sync_daemon.py \
        --sync_dir /path/to/sync \
        --parent_pid 12345 \
        --hf_model_repo user/model \
        --hf_dataset_repo user/dataset \
        --mode trainer \
        --local_rollout_dir /path/to/rollouts \
        --local_checkpoint_dir /path/to/checkpoints
"""

import os

# Daemon is CPU-only — prevent GPU memory allocation by JAX/CUDA
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Bound HF download reads so dead CLOSE-WAIT sockets raise ReadTimeoutError
# instead of hanging forever — snapshot_download's internal retries then recover.
# Observed: CloudFront half-closed connections leave urllib3 readers parked on
# dead sockets; default HF_HUB_DOWNLOAD_TIMEOUT is 10s only for small HEAD/GET,
# large file body reads have no timeout unless this is set.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import argparse
import concurrent.futures
import json
import logging
import signal
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lehome_solution.distributed.hf_sync_protocol import (
    DIR_ACTIVE,
    DIR_DONE,
    DIR_FAILED,
    DIR_QUEUE,
    DIR_READY,
    DIR_RESULTS,
    OP_CLEANUP_STATES,
    OP_DOWNLOAD_ASSET,
    OP_DOWNLOAD_CHECKPOINT,
    OP_DOWNLOAD_DATASET,
    OP_DOWNLOAD_MISSING,
    OP_SHUTDOWN,
    OP_SYNC_DAGGER,
    OP_UPLOAD_ASSET,
    OP_UPLOAD_CHECKPOINT,
    OP_UPLOAD_DAGGER_ROLLOUT,
    OP_UPLOAD_DAGGER_VALUES,
    OP_UPLOAD_DATASET,
    OP_UPLOAD_STATES,
    SyncRequest,
    SyncResult,
    atomic_write_json,
)

logger = logging.getLogger("hf_sync_daemon")

# Retry backoff: 30s, 60s, 120s
_RETRY_BACKOFFS = [30, 60, 120]
_POLL_INTERVAL_S = 2.0
_DONE_CLEANUP_AGE_S = 24 * 3600  # 24 hours
_CLEANUP_INTERVAL_S = 3600  # run cleanup every hour
_BG_POLL_INTERVAL_S = 60.0  # background polling for new data
# Per-request timeout: 15 min for downloads/uploads (checkpoint ~6min, datasets ~2min)
_REQUEST_TIMEOUT_S = 15 * 60
_MAX_CONCURRENT_DOWNLOADS = 4  # parallel dataset downloads


def _pid_alive(pid: int) -> bool:
    """Check if a process with given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_stale_daemon(sync_dir: Path):
    """Kill any stale daemon from a previous run."""
    pid_file = sync_dir / "daemon.pid"
    if not pid_file.exists():
        return

    try:
        old_pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return

    if old_pid == os.getpid():
        return

    if _pid_alive(old_pid):
        logger.info("Killing stale daemon PID %d", old_pid)
        try:
            os.kill(old_pid, signal.SIGTERM)
            # Wait up to 5s for clean exit
            for _ in range(50):
                if not _pid_alive(old_pid):
                    break
                time.sleep(0.1)
            if _pid_alive(old_pid):
                os.kill(old_pid, signal.SIGKILL)
        except OSError:
            pass

    pid_file.unlink(missing_ok=True)


def _scan_queue(sync_dir: Path) -> list[SyncRequest]:
    """Scan queue/ for pending requests, sorted by (priority, created_at)."""
    queue_dir = sync_dir / DIR_QUEUE
    if not queue_dir.exists():
        return []

    requests = []
    for f in queue_dir.glob("*.json"):
        try:
            text = f.read_text()
            req = SyncRequest.from_json(text)
            requests.append(req)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("Corrupt queue file %s: %s", f.name, e)
            # Move corrupt file to failed
            failed_dir = sync_dir / DIR_FAILED
            failed_dir.mkdir(parents=True, exist_ok=True)
            f.rename(failed_dir / f.name)

    # Sort by priority (lower = higher), then by creation time
    requests.sort(key=lambda r: (r.priority, r.created_at))
    return requests


def _move_request(sync_dir: Path, request_id: str, from_dir: str, to_dir: str):
    """Move a request file between directories."""
    src = sync_dir / from_dir / f"{request_id}.json"
    dest_dir = sync_dir / to_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{request_id}.json"
    if src.exists():
        src.rename(dest)


def _write_result(sync_dir: Path, result: SyncResult):
    """Write a result JSON to results/."""
    results_dir = sync_dir / DIR_RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{result.request_id}.json"
    atomic_write_json(result_path, result.to_json())


def _cleanup_done(sync_dir: Path):
    """Remove done/ entries older than 24 hours."""
    done_dir = sync_dir / DIR_DONE
    if not done_dir.exists():
        return

    cutoff = time.time() - _DONE_CLEANUP_AGE_S
    removed = 0
    for f in done_dir.glob("*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        logger.debug("Cleaned up %d old done entries", removed)


def _update_ready_datasets(sync_dir: Path, dataset_path: str):
    """Append a newly downloaded dataset path to ready/datasets.json."""
    ready_dir = sync_dir / DIR_READY
    ready_dir.mkdir(parents=True, exist_ok=True)
    ready_file = ready_dir / "datasets.json"

    existing = {"datasets": []}
    if ready_file.exists():
        try:
            existing = json.loads(ready_file.read_text())
        except (json.JSONDecodeError, TypeError):
            pass

    datasets = existing.get("datasets", [])
    if dataset_path not in datasets:
        datasets.append(dataset_path)
    existing["datasets"] = datasets
    existing["updated_at"] = time.time()
    atomic_write_json(ready_file, json.dumps(existing, indent=2))


def _update_ready_checkpoint(sync_dir: Path, checkpoint_info: dict):
    """Update ready/checkpoint.json with latest checkpoint info."""
    ready_dir = sync_dir / DIR_READY
    ready_dir.mkdir(parents=True, exist_ok=True)
    ready_file = ready_dir / "checkpoint.json"
    checkpoint_info["updated_at"] = time.time()
    atomic_write_json(ready_file, json.dumps(checkpoint_info, indent=2))


def _update_ready_assets(sync_dir: Path, asset_name: str):
    """Mark an asset as downloaded in ready/assets.json."""
    ready_dir = sync_dir / DIR_READY
    ready_dir.mkdir(parents=True, exist_ok=True)
    ready_file = ready_dir / "assets.json"

    existing = {"assets": []}
    if ready_file.exists():
        try:
            existing = json.loads(ready_file.read_text())
        except (json.JSONDecodeError, TypeError):
            pass

    assets = existing.get("assets", [])
    if asset_name not in assets:
        assets.append(asset_name)
    existing["assets"] = assets
    existing["updated_at"] = time.time()
    atomic_write_json(ready_file, json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# Operation handlers
# ---------------------------------------------------------------------------

def _handle_upload_checkpoint(params: dict) -> dict:
    """Upload checkpoint to HF."""
    from lehome_solution.training.hf_upload import upload_checkpoint_to_hf

    success = upload_checkpoint_to_hf(
        params["checkpoint_dir"],
        params["step"],
        params["repo_id"],
        keep_period=params.get("keep_period", 2000),
        numbered_only=params.get("numbered_only", False),
    )
    return {"success": success, "step": params["step"]}


def _handle_download_checkpoint(params: dict, sync_dir: Path) -> dict:
    """Download latest checkpoint from HF."""
    from lehome_solution.distributed.hf_sync import download_latest_checkpoint

    result = download_latest_checkpoint(
        params["hf_model_repo"],
        params["local_dir"],
        include_train_state=params.get("include_train_state", False),
    )
    if result is None:
        raise RuntimeError("No checkpoint available on HF")

    checkpoint_path, step = result
    info = {"checkpoint_path": checkpoint_path, "step": step}
    _update_ready_checkpoint(sync_dir, info)
    return info


def _handle_upload_dataset(params: dict) -> dict:
    """Upload dataset to HF."""
    # Support both upload_dataset_to_hf (main pipeline) and
    # upload_rollout_dataset (worker) via the params
    if "rollout_id" in params:
        from lehome_solution.distributed.hf_sync import upload_rollout_dataset
        success = upload_rollout_dataset(
            params["eval_run_dir"],
            params["rollout_id"],
            params["repo_id"],
        )
    else:
        from lehome_solution.training.hf_upload import upload_dataset_to_hf
        success = upload_dataset_to_hf(
            params["eval_dataset_dir"],
            params["eval_run_id"],
            params["repo_id"],
        )
    return {"success": success}


def _handle_upload_dagger_rollout(params: dict) -> dict:
    """Upload a rollout's eval_dataset as a dagger session to HF.

    Pushes the eval_dataset directory under ``dagger/session_<rollout_id>/
    eval_dataset_success/`` so the trainer's ``sync_dagger_datasets`` picks it
    up as a dagger dataset (trained with fixed advantage, FR-proportional
    sampling).
    """
    from lehome_solution.distributed.hf_sync import upload_rollout_as_dagger
    success = upload_rollout_as_dagger(
        params["eval_run_dir"],
        params["rollout_id"],
        params["repo_id"],
    )
    return {"success": success}


def _handle_download_dataset(params: dict, sync_dir: Path) -> dict:
    """Download rollout dataset from HF."""
    from lehome_solution.distributed.hf_sync import download_rollout_dataset

    local_path = download_rollout_dataset(
        params["hf_dataset_repo"],
        params["rollout_id"],
        params["local_dir"],
        skip_complete_check=True,  # poll already verified _complete marker
    )
    if local_path is None:
        logger.info("Skipping rollout %s (incomplete or unavailable)", params["rollout_id"])
        return {"local_path": None, "rollout_id": params["rollout_id"], "skipped": True}

    _update_ready_datasets(sync_dir, local_path)
    return {"local_path": local_path, "rollout_id": params["rollout_id"]}


def _handle_upload_states(params: dict) -> dict:
    """Upload failure or success states to HF."""
    kind = params.get("kind", "failure")
    if kind == "failure":
        from lehome_solution.training.hf_upload import upload_failure_states_to_hf
        success = upload_failure_states_to_hf(params["states_dir"], params["repo_id"])
    elif kind == "success":
        from lehome_solution.training.hf_upload import upload_success_states_to_hf
        success = upload_success_states_to_hf(params["states_dir"], params["repo_id"])
    elif kind == "failure_individual":
        from lehome_solution.distributed.hf_sync import upload_failure_states_individually
        success = upload_failure_states_individually(
            params["states_dir"], params["repo_id"],
            worker_id=params.get("worker_id"),
        )
    elif kind == "success_individual":
        from lehome_solution.distributed.hf_sync import upload_success_states_individually
        success = upload_success_states_individually(
            params["states_dir"], params["repo_id"],
            worker_id=params.get("worker_id"),
        )
    elif kind == "semi_success":
        from lehome_solution.training.hf_upload import upload_semi_success_states_to_hf
        success = upload_semi_success_states_to_hf(params["states_dir"], params["repo_id"])
    elif kind == "semi_success_individual":
        from lehome_solution.distributed.hf_sync import upload_semi_success_states_individually
        success = upload_semi_success_states_individually(
            params["states_dir"], params["repo_id"],
            worker_id=params.get("worker_id"),
        )
    else:
        raise ValueError(f"Unknown state kind: {kind}")
    return {"success": success, "kind": kind}


def _handle_cleanup_states(params: dict) -> dict:
    """Delete consumed success states from HF."""
    from lehome_solution.distributed.hf_sync import delete_success_states_from_hf

    filenames = params.get("filenames", [])
    if not filenames:
        return {"deleted": 0}
    count = delete_success_states_from_hf(params["hf_dataset_repo"], filenames)
    return {"deleted": count}


def _handle_upload_asset(params: dict) -> dict:
    """Upload a single asset file (inference_prior, success_rates, etc.)."""
    asset_type = params.get("asset_type", "")

    if asset_type == "inference_prior":
        from lehome_solution.training.hf_upload import upload_inference_prior_to_hf
        success = upload_inference_prior_to_hf(
            params["path"], params["repo_id"],
            path_in_repo=params.get("path_in_repo"),
        )
    elif asset_type == "success_rates":
        from lehome_solution.training.hf_upload import upload_success_rates_to_hf
        success = upload_success_rates_to_hf(params["path"], params["repo_id"])
    elif asset_type == "wandb_id":
        from lehome_solution.training.hf_upload import upload_wandb_id_to_hf
        success = upload_wandb_id_to_hf(params["path"], params["repo_id"])
    elif asset_type == "pipeline_state":
        from lehome_solution.training.hf_upload import upload_pipeline_state_to_hf
        success = upload_pipeline_state_to_hf(params["path"], params["repo_id"])
    elif asset_type == "step_info":
        from lehome_solution.training.hf_upload import upload_step_info
        success = upload_step_info(
            params["repo_id"], params["step"],
            iteration=params.get("iteration"),
        )
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")
    return {"success": success, "asset_type": asset_type}


def _handle_download_asset(params: dict, sync_dir: Path) -> dict:
    """Download a single asset file from HF."""
    from lehome_solution.training.hf_upload import download_model_asset

    success = download_model_asset(
        params["repo_id"],
        params["asset_name"],
        params["local_path"],
    )
    if success:
        _update_ready_assets(sync_dir, params["asset_name"])
    return {"success": success, "asset_name": params["asset_name"]}


def _handle_upload_dagger_values(params: dict) -> dict:
    """Upload updated dagger parquets back to HF after value prediction."""
    repo_id = params.get("repo_id")
    dagger_datasets = params.get("dagger_datasets", [])
    if not repo_id or not dagger_datasets:
        return {"uploaded": 0}

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping dagger upload")
        return {"uploaded": 0}

    api = HfApi()
    uploaded = 0

    for ds_info in dagger_datasets:
        ds_path = Path(ds_info["root"])
        if not ds_path.exists():
            continue

        parquet_files = list(ds_path.glob("data/chunk-*/*.parquet"))
        if not parquet_files:
            continue

        # Extract session name from path
        parts = ds_path.parts
        try:
            session_idx = next(i for i, p in enumerate(parts) if p.startswith("session_"))
            session_name = parts[session_idx]
        except StopIteration:
            logger.warning("Cannot determine session name from %s, skipping", ds_path)
            continue

        for pq_file in parquet_files:
            rel = pq_file.relative_to(ds_path)
            hf_path = f"dagger/{session_name}/eval_dataset_success/{rel}"
            try:
                api.upload_file(
                    path_or_fileobj=str(pq_file),
                    path_in_repo=hf_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                uploaded += 1
            except Exception as e:
                logger.warning("Failed to upload %s: %s", hf_path, e)

    return {"uploaded": uploaded}


def _handle_sync_dagger(params: dict, sync_dir: Path) -> dict:
    """Download dagger datasets from HF into {checkpoint_dir}/_hf_dagger/.

    The trainer's ``_scan_local_dagger`` reads from the same directory; both
    must agree or dagger data never reaches training.
    """
    repo_id = params.get("repo_id")
    checkpoint_dir = params.get("checkpoint_dir")
    if not repo_id or not checkpoint_dir:
        return {"sessions_downloaded": 0}

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping dagger sync")
        return {"sessions_downloaded": 0}

    api = HfApi()

    try:
        entries = list(api.list_repo_tree(repo_id, repo_type="dataset", recursive=False))
        dagger_entry = [e for e in entries if getattr(e, "path", "") == "dagger"]
        if not dagger_entry:
            return {"sessions_downloaded": 0}

        sub_entries = list(api.list_repo_tree(
            repo_id, path_in_repo="dagger", repo_type="dataset", recursive=False,
        ))
        session_dirs = [
            e.path for e in sub_entries
            if getattr(e, "path", "").startswith("dagger/session_")
        ]
    except Exception as e:
        logger.warning("Failed to list dagger sessions on HF: %s", e)
        return {"sessions_downloaded": 0}

    if not session_dirs:
        return {"sessions_downloaded": 0}

    dagger_local_base = Path(checkpoint_dir) / "_hf_dagger"
    downloaded = 0

    for session_path in sorted(session_dirs):
        local_ds_dir = dagger_local_base / session_path / "eval_dataset_success"

        if local_ds_dir.exists() and any(local_ds_dir.glob("data/chunk-*/*.parquet")):
            continue

        try:
            snapshot_download(
                repo_id,
                allow_patterns=[f"{session_path}/eval_dataset_success/**"],
                local_dir=str(dagger_local_base),
                repo_type="dataset",
            )
            if local_ds_dir.exists() and any(local_ds_dir.glob("data/chunk-*/*.parquet")):
                downloaded += 1
        except Exception as e:
            logger.warning("Failed to download dagger session %s: %s",
                           session_path.split("/")[-1], e)

    return {"sessions_downloaded": downloaded}


def _handle_download_missing(params: dict) -> dict:
    """Download missing datasets from HF (by path pattern)."""
    repo_id = params.get("repo_id")
    datasets = params.get("datasets", [])
    if not repo_id or not datasets:
        return {"downloaded": 0}

    try:
        from huggingface_hub import snapshot_download
        import shutil
    except ImportError:
        logger.warning("huggingface_hub not installed, cannot download missing datasets")
        return {"downloaded": 0}

    downloaded = 0
    for ds_info in datasets:
        root = Path(ds_info["root"])
        if (root / "meta" / "info.json").exists():
            continue

        # Determine HF path
        parts = root.parts
        hf_path = None
        for i, part in enumerate(parts):
            if part.startswith("rollout_"):
                hf_path = "/".join(parts[i:])
                break
            if part == "dagger":
                hf_path = "/".join(parts[i:])
                break
            if part.endswith("_init"):
                # Init datasets: full_init, curriculum_init, etc.
                hf_path = part
                break

        if not hf_path:
            logger.warning("Cannot determine HF path for %s, skipping", root)
            continue

        try:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="hf_dataset_download_") as tmp_dir:
                snapshot_download(
                    repo_id,
                    allow_patterns=[f"{hf_path}/**"],
                    local_dir=tmp_dir,
                    repo_type="dataset",
                )
                src = Path(tmp_dir) / hf_path
                if src.exists() and (src / "meta" / "info.json").exists():
                    root.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(str(src), str(root), dirs_exist_ok=True)
                    downloaded += 1
        except Exception as e:
            logger.warning("Failed to download %s: %s", hf_path, e)

    return {"downloaded": downloaded}


# ---------------------------------------------------------------------------
# Background polling
# ---------------------------------------------------------------------------

def _poll_trainer_checkpoints(sync_dir: Path, hf_model_repo: str | None,
                              checkpoint_dir: str | None, keep_period: int):
    """Trainer mode: detect new local checkpoints and auto-upload to HF."""
    if not hf_model_repo or not checkpoint_dir:
        return

    try:
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            return

        # Find latest local checkpoint step
        steps = [
            int(d.name) for d in ckpt_path.iterdir()
            if d.is_dir() and d.name.isdigit() and (d / "params").is_dir()
        ]
        if not steps:
            return
        latest_step = max(steps)

        # Track last uploaded step in ready/ dir
        uploaded_file = sync_dir / DIR_READY / "last_uploaded_step.json"
        last_uploaded = 0
        if uploaded_file.exists():
            try:
                data = json.loads(uploaded_file.read_text())
                last_uploaded = data.get("step", 0)
            except (json.JSONDecodeError, TypeError):
                pass

        if latest_step <= last_uploaded:
            return

        # Check if an upload for this step is already queued/active
        for subdir in [DIR_QUEUE, DIR_ACTIVE]:
            d = sync_dir / subdir
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                try:
                    req = SyncRequest.from_json(f.read_text())
                    if req.op_type == OP_UPLOAD_CHECKPOINT and req.params.get("step") == latest_step:
                        return  # already queued
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

        # Queue upload
        from lehome_solution.distributed.hf_sync_protocol import PRIORITY_HIGH, make_request_id
        request_id = make_request_id(OP_UPLOAD_CHECKPOINT)
        req = SyncRequest(
            request_id=request_id,
            op_type=OP_UPLOAD_CHECKPOINT,
            priority=PRIORITY_HIGH,
            params={
                "checkpoint_dir": checkpoint_dir,
                "step": latest_step,
                "repo_id": hf_model_repo,
                "keep_period": keep_period,
            },
            created_at=time.time(),
        )
        queue_path = sync_dir / DIR_QUEUE / f"{request_id}.json"
        atomic_write_json(queue_path, req.to_json())

        # Update tracker
        atomic_write_json(uploaded_file, json.dumps({"step": latest_step}))
        logger.info("Poll: auto-queued checkpoint upload for step %d", latest_step)

    except Exception as e:
        logger.warning("Background poll (trainer checkpoints) failed: %s", e)


def _poll_trainer(sync_dir: Path, hf_dataset_repo: str | None,
                  local_rollout_dir: str | None):
    """Trainer mode: check HF for new rollout datasets and auto-download."""
    if not hf_dataset_repo or not local_rollout_dir:
        return

    try:
        from lehome_solution.distributed.hf_sync import (
            list_available_rollouts,
            download_rollout_dataset,
        )

        # Get list of rollouts the trainer already knows about (don't re-download)
        known = set()
        # 1. Trainer writes known_rollouts.json with consumed + in-use rollout IDs
        known_file = sync_dir / DIR_READY / "known_rollouts.json"
        if known_file.exists():
            try:
                data = json.loads(known_file.read_text())
                known = set(data.get("known", []))
            except (json.JSONDecodeError, TypeError):
                pass
        # 2. Also check ready/datasets.json (what daemon already downloaded)
        ready_file = sync_dir / DIR_READY / "datasets.json"
        if ready_file.exists():
            try:
                data = json.loads(ready_file.read_text())
                known.update(data.get("datasets", []))
            except (json.JSONDecodeError, TypeError):
                pass
        # 3. Also scan local rollout dir for already-downloaded datasets
        local_dir = Path(local_rollout_dir)
        if local_dir.exists():
            for d in local_dir.iterdir():
                if d.is_dir() and (d / "eval_dataset").is_dir():
                    known.add(d.name)  # rollout_id is the directory name

        # Check pending/active queues to avoid duplicates, plus recently-failed
        # (failed requests are retried after 10 min in case the upload was still in progress)
        skip_ids = set()
        for subdir in [DIR_QUEUE, DIR_ACTIVE]:
            d = sync_dir / subdir
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                try:
                    req = SyncRequest.from_json(f.read_text())
                    if req.op_type == OP_DOWNLOAD_DATASET:
                        skip_ids.add(req.params.get("rollout_id", ""))
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        # Skip recently-failed downloads (retry after 10 min cooldown)
        failed_dir = sync_dir / DIR_FAILED
        if failed_dir.exists():
            now = time.time()
            for f in failed_dir.glob("*.json"):
                try:
                    req = SyncRequest.from_json(f.read_text())
                    if req.op_type == OP_DOWNLOAD_DATASET:
                        age_s = now - req.created_at
                        if age_s < 600:  # 10 min cooldown
                            skip_ids.add(req.params.get("rollout_id", ""))
                        else:
                            # Cooldown expired — remove failed entry so it can be retried
                            f.unlink(missing_ok=True)
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

        available = list_available_rollouts(hf_dataset_repo)
        new_count = 0
        for r in available:
            rollout_id = r["rollout_id"]
            if rollout_id in skip_ids:
                continue
            # Check if already downloaded (exact match or substring of known paths)
            if rollout_id in known or any(rollout_id in p for p in known):
                continue

            # Submit download request
            from lehome_solution.distributed.hf_sync_protocol import (
                PRIORITY_NORMAL, make_request_id,
            )
            request_id = make_request_id(OP_DOWNLOAD_DATASET)
            req = SyncRequest(
                request_id=request_id,
                op_type=OP_DOWNLOAD_DATASET,
                priority=PRIORITY_NORMAL,
                params={
                    "hf_dataset_repo": hf_dataset_repo,
                    "rollout_id": rollout_id,
                    "local_dir": local_rollout_dir,
                },
                created_at=time.time(),
            )
            queue_path = sync_dir / DIR_QUEUE / f"{request_id}.json"
            atomic_write_json(queue_path, req.to_json())
            new_count += 1

        if new_count:
            logger.info("Poll: queued %d new rollout downloads", new_count)

    except Exception as e:
        logger.warning("Background poll (trainer) failed: %s", e)


def _poll_rollout(sync_dir: Path, hf_model_repo: str | None,
                  local_checkpoint_dir: str | None):
    """Rollout mode: check HF for new checkpoint and auto-download."""
    if not hf_model_repo or not local_checkpoint_dir:
        return

    try:
        from lehome_solution.distributed.hf_sync import get_latest_model_step

        latest_step = get_latest_model_step(hf_model_repo)
        if latest_step is None:
            return

        # Check what we already have
        ready_file = sync_dir / DIR_READY / "checkpoint.json"
        current_step = None
        if ready_file.exists():
            try:
                data = json.loads(ready_file.read_text())
                current_step = data.get("step")
            except (json.JSONDecodeError, TypeError):
                pass

        if current_step is not None and latest_step <= current_step:
            return  # Already have the latest (or HF returned stale step)

        # Check if a download is already pending
        for subdir in [DIR_QUEUE, DIR_ACTIVE]:
            d = sync_dir / subdir
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                try:
                    req = SyncRequest.from_json(f.read_text())
                    if req.op_type == OP_DOWNLOAD_CHECKPOINT:
                        return  # Already pending
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

        # Submit download request
        from lehome_solution.distributed.hf_sync_protocol import (
            PRIORITY_HIGH, make_request_id,
        )
        request_id = make_request_id(OP_DOWNLOAD_CHECKPOINT)
        req = SyncRequest(
            request_id=request_id,
            op_type=OP_DOWNLOAD_CHECKPOINT,
            priority=PRIORITY_HIGH,
            params={
                "hf_model_repo": hf_model_repo,
                "local_dir": local_checkpoint_dir,
            },
            created_at=time.time(),
        )
        queue_path = sync_dir / DIR_QUEUE / f"{request_id}.json"
        atomic_write_json(queue_path, req.to_json())
        logger.info("Poll: queued checkpoint download (step %d, was %s)",
                     latest_step, current_step)

    except Exception as e:
        logger.warning("Background poll (rollout) failed: %s", e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _execute_request(req: SyncRequest, sync_dir: Path) -> dict:
    """Execute a single request. Returns result dict. Raises on failure."""
    op = req.op_type
    params = req.params

    if op == OP_UPLOAD_CHECKPOINT:
        return _handle_upload_checkpoint(params)
    elif op == OP_DOWNLOAD_CHECKPOINT:
        return _handle_download_checkpoint(params, sync_dir)
    elif op == OP_UPLOAD_DATASET:
        return _handle_upload_dataset(params)
    elif op == OP_UPLOAD_DAGGER_ROLLOUT:
        return _handle_upload_dagger_rollout(params)
    elif op == OP_DOWNLOAD_DATASET:
        return _handle_download_dataset(params, sync_dir)
    elif op == OP_UPLOAD_STATES:
        return _handle_upload_states(params)
    elif op == OP_CLEANUP_STATES:
        return _handle_cleanup_states(params)
    elif op == OP_UPLOAD_ASSET:
        return _handle_upload_asset(params)
    elif op == OP_DOWNLOAD_ASSET:
        return _handle_download_asset(params, sync_dir)
    elif op == OP_UPLOAD_DAGGER_VALUES:
        return _handle_upload_dagger_values(params)
    elif op == OP_SYNC_DAGGER:
        return _handle_sync_dagger(params, sync_dir)
    elif op == OP_DOWNLOAD_MISSING:
        return _handle_download_missing(params)
    elif op == OP_SHUTDOWN:
        return {"shutdown": True}
    else:
        raise ValueError(f"Unknown operation type: {op}")


def _process_one(sync_dir: Path, req: SyncRequest) -> bool:
    """Process a single request. Returns True if shutdown requested."""
    request_id = req.request_id
    logger.info("Processing %s: %s (priority=%d, retry=%d/%d)",
                req.op_type, request_id, req.priority,
                req.retry_count, req.max_retries)

    # Move to active
    _move_request(sync_dir, request_id, DIR_QUEUE, DIR_ACTIVE)

    start_time = time.time()
    error = None
    # Don't use `with ThreadPoolExecutor() as ...`: __exit__ calls
    # shutdown(wait=True) which blocks forever if the worker thread is hung
    # on a Python lock (e.g. snapshot_download deadlock inside hf_hub).
    # Manual shutdown(wait=False) lets the daemon move on; the abandoned
    # thread leaks, but it's bounded by retry/backoff and parent_pid death.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_execute_request, req, sync_dir)
        try:
            result_data = future.result(timeout=_REQUEST_TIMEOUT_S)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        duration = time.time() - start_time

        result = SyncResult(
            request_id=request_id,
            success=True,
            result=result_data,
            completed_at=time.time(),
            duration_s=round(duration, 2),
        )
        _write_result(sync_dir, result)
        _move_request(sync_dir, request_id, DIR_ACTIVE, DIR_DONE)
        logger.info("Completed %s: %s (%.1fs)", req.op_type, request_id, duration)

        if req.op_type == OP_SHUTDOWN:
            return True

    except concurrent.futures.TimeoutError:
        duration = time.time() - start_time
        error = TimeoutError(f"Request timed out after {_REQUEST_TIMEOUT_S}s")
        logger.error("TIMEOUT %s: %s (%.1fs) — request hung, will retry",
                      req.op_type, request_id, duration)
    except Exception as e:
        duration = time.time() - start_time
        error = e
        logger.warning("Failed %s: %s (%.1fs): %s",
                       req.op_type, request_id, duration, e)

    if error is not None:
        req.retry_count += 1
        if req.retry_count >= req.max_retries:
            # Max retries exhausted
            result = SyncResult(
                request_id=request_id,
                success=False,
                error=str(error),
                completed_at=time.time(),
                duration_s=round(duration, 2),
            )
            _write_result(sync_dir, result)
            _move_request(sync_dir, request_id, DIR_ACTIVE, DIR_FAILED)
            logger.error("Request %s failed after %d retries: %s",
                        request_id, req.max_retries, error)
        else:
            # Schedule retry with backoff
            backoff_idx = min(req.retry_count - 1, len(_RETRY_BACKOFFS) - 1)
            backoff = _RETRY_BACKOFFS[backoff_idx]
            req.created_at = time.time() + backoff  # delay next processing
            # Write updated request back to queue
            queue_path = sync_dir / DIR_QUEUE / f"{request_id}.json"
            atomic_write_json(queue_path, req.to_json())
            # Remove from active
            active_path = sync_dir / DIR_ACTIVE / f"{request_id}.json"
            active_path.unlink(missing_ok=True)
            logger.info("Retry %d/%d for %s in %ds",
                       req.retry_count, req.max_retries, request_id, backoff)

    return False


def _process_download_batch(sync_dir: Path, downloads: list[SyncRequest]) -> bool:
    """Process multiple download requests concurrently. Returns True if shutdown requested."""
    if not downloads:
        return False

    batch = downloads[:_MAX_CONCURRENT_DOWNLOADS]
    logger.info("Processing %d downloads concurrently", len(batch))

    # Move all to active
    for req in batch:
        _move_request(sync_dir, req.request_id, DIR_QUEUE, DIR_ACTIVE)

    # Execute concurrently. See _process_one for why we don't use `with`:
    # __exit__'s shutdown(wait=True) blocks forever on a hung worker.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(batch))
    try:
        future_to_req = {
            executor.submit(_execute_request, req, sync_dir): req
            for req in batch
        }

        # Bound the as_completed wait by _REQUEST_TIMEOUT_S so a single
        # hung future can't block the others' completion handling forever.
        deadline = time.time() + _REQUEST_TIMEOUT_S
        completed: set = set()
        try:
            for future in concurrent.futures.as_completed(
                future_to_req, timeout=_REQUEST_TIMEOUT_S
            ):
                completed.add(future)
                req = future_to_req[future]
                start_time = req.created_at  # approximate
                try:
                    result_data = future.result(timeout=max(0, deadline - time.time()))
                    duration = time.time() - start_time

                    result = SyncResult(
                        request_id=req.request_id,
                        success=True,
                        result=result_data,
                        completed_at=time.time(),
                        duration_s=round(duration, 2),
                    )
                    _write_result(sync_dir, result)
                    _move_request(sync_dir, req.request_id, DIR_ACTIVE, DIR_DONE)
                    logger.info("Completed %s: %s (%.1fs)", req.op_type, req.request_id, duration)

                except Exception as e:
                    _record_batch_failure(sync_dir, req, e, start_time)
        except concurrent.futures.TimeoutError:
            pass

        # Anything still pending after the deadline is treated as a timeout.
        for future, req in future_to_req.items():
            if future in completed:
                continue
            _record_batch_failure(
                sync_dir, req,
                concurrent.futures.TimeoutError(
                    f"Request timed out after {_REQUEST_TIMEOUT_S}s"
                ),
                req.created_at,
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return False


def _record_batch_failure(sync_dir: Path, req: SyncRequest, e: BaseException, start_time: float):
    duration = time.time() - start_time
    if isinstance(e, concurrent.futures.TimeoutError):
        logger.error("TIMEOUT %s: %s (%.1fs)", req.op_type, req.request_id, duration)
    else:
        logger.warning("Failed %s: %s (%.1fs): %s", req.op_type, req.request_id, duration, e)

    req.retry_count += 1
    if req.retry_count >= req.max_retries:
        result = SyncResult(
            request_id=req.request_id,
            success=False,
            error=str(e),
            completed_at=time.time(),
            duration_s=round(duration, 2),
        )
        _write_result(sync_dir, result)
        _move_request(sync_dir, req.request_id, DIR_ACTIVE, DIR_FAILED)
        logger.error("Request %s failed after %d retries: %s",
                    req.request_id, req.max_retries, e)
    else:
        backoff_idx = min(req.retry_count - 1, len(_RETRY_BACKOFFS) - 1)
        backoff = _RETRY_BACKOFFS[backoff_idx]
        req.created_at = time.time() + backoff
        queue_path = sync_dir / DIR_QUEUE / f"{req.request_id}.json"
        atomic_write_json(queue_path, req.to_json())
        active_path = sync_dir / DIR_ACTIVE / f"{req.request_id}.json"
        active_path.unlink(missing_ok=True)
        logger.info("Retry %d/%d for %s in %ds",
                   req.retry_count, req.max_retries, req.request_id, backoff)


def _drain_and_exit(sync_dir: Path):
    """Process all remaining queue items, then exit."""
    logger.info("Parent process died, draining queue before exit...")
    while True:
        requests = _scan_queue(sync_dir)
        # Filter out requests with future retry_after
        actionable = [r for r in requests if r.created_at <= time.time()]
        if not actionable:
            break
        req = actionable[0]
        _process_one(sync_dir, req)

    logger.info("Queue drained, exiting")


def daemon_main(sync_dir: Path, parent_pid: int | None,
                hf_model_repo: str | None, hf_dataset_repo: str | None,
                mode: str = "pipeline",
                local_rollout_dir: str | None = None,
                local_checkpoint_dir: str | None = None,
                checkpoint_dir: str | None = None,
                keep_period: int = 2000):
    """Main daemon loop."""
    # Kill stale daemon if any
    _kill_stale_daemon(sync_dir)

    # Create directory structure
    for subdir in [DIR_QUEUE, DIR_ACTIVE, DIR_DONE, DIR_FAILED, DIR_RESULTS, DIR_READY]:
        (sync_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Write PID file
    pid_file = sync_dir / "daemon.pid"
    atomic_write_json(pid_file, str(os.getpid()))

    # Clean up stale items from previous daemon runs:
    # - Uploads (queue + active): PRESERVE — they're irreplaceable, main process already moved on
    # - Downloads (queue + active): CLEAR — daemon polling will re-queue them
    # - Done/shutdown: CLEAR — already completed
    _upload_ops = {"upload_checkpoint", "upload_dataset", "upload_dagger_rollout",
                   "upload_states", "upload_asset", "upload_dagger_values"}
    for subdir in [DIR_QUEUE, DIR_ACTIVE]:
        d = sync_dir / subdir
        if not d.exists():
            continue
        preserved = 0
        cleared = 0
        for f in list(d.glob("*.json")):
            try:
                req = SyncRequest.from_json(f.read_text())
                if req.op_type in _upload_ops:
                    # Move active uploads back to queue for retry
                    if subdir == DIR_ACTIVE:
                        dest = sync_dir / DIR_QUEUE / f.name
                        f.rename(dest)
                    preserved += 1
                elif req.op_type == OP_SHUTDOWN:
                    f.unlink(missing_ok=True)
                    cleared += 1
                else:
                    # Download/sync ops — clear (will be re-queued by polling)
                    f.unlink(missing_ok=True)
                    cleared += 1
            except (json.JSONDecodeError, TypeError, KeyError):
                f.unlink(missing_ok=True)
                cleared += 1
        if preserved or cleared:
            logger.info("Startup: %s/ — preserved %d uploads, cleared %d downloads/other",
                        subdir, preserved, cleared)

    # Always clear done/
    done_dir = sync_dir / DIR_DONE
    if done_dir.exists():
        stale = list(done_dir.glob("*.json"))
        if stale:
            for f in stale:
                f.unlink(missing_ok=True)
            logger.info("Cleared %d stale done/ items", len(stale))

    # In rollout mode, clear stale ready/checkpoint.json and re-derive from
    # local disk. If checkpoints already exist locally, write a ready file
    # immediately so the worker doesn't wait for an HF round-trip.
    if mode == "rollout":
        ready_ckpt = sync_dir / DIR_READY / "checkpoint.json"
        if ready_ckpt.exists():
            logger.info("Clearing stale ready/checkpoint.json from previous daemon run")
            ready_ckpt.unlink(missing_ok=True)

        # Check for already-downloaded checkpoints
        if local_checkpoint_dir:
            ckpt_base = Path(local_checkpoint_dir)
            if ckpt_base.exists():
                step_dirs = sorted(
                    [d for d in ckpt_base.iterdir() if d.is_dir() and d.name.startswith("step_")],
                    key=lambda d: int(d.name.split("_", 1)[1]),
                )
                if step_dirs:
                    latest = step_dirs[-1]
                    latest_step = int(latest.name.split("_", 1)[1])
                    info = {
                        "checkpoint_path": str(latest),
                        "step": latest_step,
                        "updated_at": time.time(),
                    }
                    _update_ready_checkpoint(sync_dir, info)
                    logger.info("Startup: found local checkpoint step_%d, ready immediately", latest_step)

    # Write status
    status_file = sync_dir / "daemon.status"
    atomic_write_json(status_file, "ready")

    logger.info("HF sync daemon started (PID=%d, parent=%s, mode=%s, model=%s, dataset=%s)",
                os.getpid(), parent_pid, mode, hf_model_repo, hf_dataset_repo)

    last_cleanup = time.time()
    last_poll = 0.0  # poll immediately on first idle

    while True:
        # 1. Check parent PID
        if parent_pid is not None and not _pid_alive(parent_pid):
            _drain_and_exit(sync_dir)
            break

        # 2. Periodic cleanup of old done/ entries
        now = time.time()
        if now - last_cleanup > _CLEANUP_INTERVAL_S:
            _cleanup_done(sync_dir)
            last_cleanup = now

        # 3. Scan queue
        requests = _scan_queue(sync_dir)

        # 4. Filter out requests with future created_at (retry backoff)
        actionable = [r for r in requests if r.created_at <= time.time()]

        if not actionable:
            # No work in queue -- do background polling if enough time has passed
            if now - last_poll >= _BG_POLL_INTERVAL_S:
                if mode == "trainer":
                    _poll_trainer(sync_dir, hf_dataset_repo, local_rollout_dir)
                    _poll_trainer_checkpoints(sync_dir, hf_model_repo, checkpoint_dir, keep_period)
                elif mode == "rollout":
                    _poll_rollout(sync_dir, hf_model_repo, local_checkpoint_dir)
                last_poll = now
                # Re-scan immediately after poll (may have queued new items)
                continue

            time.sleep(_POLL_INTERVAL_S)
            continue

        # 5. Process requests — batch downloads concurrently, others serially
        top_req = actionable[0]

        # If highest-priority item is a download, batch all same-priority downloads
        if top_req.op_type in (OP_DOWNLOAD_DATASET,
                               OP_DOWNLOAD_MISSING, OP_SYNC_DAGGER):
            downloads = [r for r in actionable
                         if r.op_type == top_req.op_type
                         and r.priority == top_req.priority]
            if len(downloads) > 1:
                _process_download_batch(sync_dir, downloads)
                continue

        shutdown = _process_one(sync_dir, top_req)
        if shutdown:
            # Process remaining queue before exit
            _drain_and_exit(sync_dir)
            break

    # Cleanup
    pid_file.unlink(missing_ok=True)
    status_file = sync_dir / "daemon.status"
    atomic_write_json(status_file, "stopped")
    logger.info("HF sync daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="HF sync daemon")
    parser.add_argument("--sync_dir", type=str, required=True,
                       help="Directory for sync protocol files")
    parser.add_argument("--parent_pid", type=int, default=None,
                       help="Parent process PID (daemon exits when parent dies)")
    parser.add_argument("--hf_model_repo", type=str, default=None,
                       help="HuggingFace model repo ID")
    parser.add_argument("--hf_dataset_repo", type=str, default=None,
                       help="HuggingFace dataset repo ID")
    parser.add_argument("--mode", type=str, default="pipeline",
                       choices=["trainer", "rollout", "pipeline"],
                       help="Daemon mode: trainer (polls for rollouts), "
                            "rollout (polls for checkpoints), pipeline (queue-only)")
    parser.add_argument("--local_rollout_dir", type=str, default=None,
                       help="Local directory for downloaded rollout datasets (trainer mode)")
    parser.add_argument("--local_checkpoint_dir", type=str, default=None,
                       help="Local directory for downloaded checkpoints (rollout mode)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Checkpoint directory to watch for new saves (trainer mode auto-upload)")
    parser.add_argument("--keep_period", type=int, default=2000,
                       help="Keep period for numbered checkpoint uploads")
    args = parser.parse_args()

    sync_dir = Path(args.sync_dir)

    # Set up logging to daemon.log
    log_file = sync_dir / "daemon.log"
    sync_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
        force=True,
    )

    daemon_main(
        sync_dir=sync_dir,
        parent_pid=args.parent_pid,
        hf_model_repo=args.hf_model_repo,
        hf_dataset_repo=args.hf_dataset_repo,
        mode=args.mode,
        local_rollout_dir=args.local_rollout_dir,
        local_checkpoint_dir=args.local_checkpoint_dir,
        checkpoint_dir=args.checkpoint_dir,
        keep_period=args.keep_period,
    )


if __name__ == "__main__":
    main()
