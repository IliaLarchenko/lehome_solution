"""HuggingFace Hub coordination for distributed rollout collection.

Provides primitives for:
- Checkpoint polling (workers download latest model)
- Rollout dataset upload/download (workers upload, trainer downloads)
- Failure state sync (merge-download, individual file uploads)
- Rollout ID naming convention
"""

import io
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def get_latest_model_step(hf_model_repo: str) -> int | None:
    """Get current model step from HF repo.

    Tries step_info.json first (fast). Falls back to finding the highest
    numbered checkpoint folder if step_info.json doesn't exist yet
    (backward compat with checkpoints uploaded before this feature).
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        logger.warning("huggingface_hub not installed")
        return None

    # Try step_info.json first (fast, single file download)
    try:
        path = hf_hub_download(
            hf_model_repo,
            filename="latest/assets/step_info.json",
            repo_type="model",
            force_download=True,
        )
        with open(path) as f:
            info = json.load(f)
        return info.get("step")
    except Exception:
        pass

    # Fallback: check if latest/ exists and find highest numbered folder
    api = HfApi()
    try:
        entries = list(api.list_repo_tree(hf_model_repo, repo_type="model"))
        # Check for latest/ folder
        has_latest = any(getattr(e, "path", "") == "latest" for e in entries)
        if not has_latest:
            return None

        # Find highest numbered checkpoint folder for the step number
        steps = []
        for entry in entries:
            path = getattr(entry, "path", "")
            if path.isdigit():
                steps.append(int(path))

        if steps:
            return max(steps)

        # latest/ exists but no numbered folders — return 0 to signal "checkpoint available"
        return 0
    except Exception:
        logger.debug("Could not list repo tree for %s", hf_model_repo, exc_info=True)
        return None


def download_latest_checkpoint(hf_model_repo: str, local_dir: str,
                               include_train_state: bool = False) -> tuple[str, int] | None:
    """Download latest checkpoint from HF.

    By default downloads params + assets only (for rollout workers).
    Set ``include_train_state=True`` to also download optimizer state
    (needed for resuming training on a new machine).

    Downloads to a versioned directory ``step_{N}/`` so that an in-use
    checkpoint is never overwritten mid-read. Old step directories are
    cleaned up after a successful download.

    Returns (checkpoint_path, step) or None if not available.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed")
        return None

    try:
        import shutil

        local_base = Path(local_dir)
        local_base.mkdir(parents=True, exist_ok=True)

        # Check remote step
        remote_step = get_latest_model_step(hf_model_repo)
        if remote_step is None:
            return None

        # Check if we already have this step locally
        step_dir = local_base / f"step_{remote_step}"
        step_info = step_dir / "assets" / "step_info.json"
        if step_info.exists():
            logger.info("Local checkpoint already at step %d, skipping download", remote_step)
            return str(step_dir), remote_step

        # Also check legacy latest/ directory
        legacy_latest = local_base / "latest"
        legacy_step_info = legacy_latest / "assets" / "step_info.json"
        if legacy_step_info.exists():
            with open(legacy_step_info) as f:
                local_step = json.load(f).get("step", -1)
            if local_step == remote_step:
                logger.info("Local checkpoint (legacy latest/) already at step %d", local_step)
                return str(legacy_latest), local_step

        # Download to temp dir, then rename to step_{N}/
        tmp_dir = local_base / f"_downloading_step_{remote_step}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        patterns = ["latest/params/**", "latest/assets/**"]
        if include_train_state:
            patterns.append("latest/train_state/**")
        logger.info("Downloading checkpoint step %d (train_state=%s)", remote_step, include_train_state)
        snapshot_download(
            hf_model_repo,
            allow_patterns=patterns,
            local_dir=str(tmp_dir),
            repo_type="model",
        )

        # HF downloads to tmp_dir/latest/ — rename to step_{N}/
        tmp_latest = tmp_dir / "latest"
        if not tmp_latest.exists():
            logger.warning("Download succeeded but latest/ not found in temp dir")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        tmp_latest.rename(step_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Clean up old step directories (keep latest 3)
        step_dirs = sorted(
            [d for d in local_base.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
            reverse=True,
        )
        for d in step_dirs[3:]:
            logger.info("Removing old checkpoint: %s", d.name)
            shutil.rmtree(d, ignore_errors=True)

        # Remove legacy latest/ once we have at least one versioned dir
        if step_dirs and legacy_latest.exists():
            logger.info("Removing legacy latest/ (migrated to step_%d)", remote_step)
            shutil.rmtree(legacy_latest, ignore_errors=True)

        return str(step_dir), remote_step
    except Exception as e:
        logger.warning("Failed to download checkpoint from %s: %s", hf_model_repo, e)
        return None


def list_available_rollouts(hf_dataset_repo: str) -> list[dict]:
    """List rollout_* dirs in HF dataset repo.

    Parses naming convention: rollout_{step}_{strategy}_{datetime}_{machine}
    Returns [{rollout_id, model_step, strategy, timestamp, machine_id}]
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    api = HfApi()
    try:
        entries = list(api.list_repo_tree(hf_dataset_repo, repo_type="dataset", recursive=False))
    except Exception as e:
        logger.warning("Failed to list repo tree for %s: %s", hf_dataset_repo, e)
        return []

    # Collect rollout directory names
    rollout_dirs = [
        getattr(entry, "path", "")
        for entry in entries
        if getattr(entry, "path", "").startswith("rollout_")
    ]
    if not rollout_dirs:
        return []

    # Check _complete markers in one batch call (not list_repo_files which scans everything).
    # If the lookup fails we must treat every rollout as incomplete — silently
    # falling back to "accept all" would ingest partial uploads.
    complete_paths = [f"{d}/_complete" for d in rollout_dirs]
    try:
        path_infos = api.get_paths_info(hf_dataset_repo, complete_paths, repo_type="dataset")
    except Exception as e:
        logger.warning("Failed to check _complete markers for %s: %s — skipping this poll", hf_dataset_repo, e)
        return []
    complete_set = {info.path.rsplit("/", 1)[0] for info in path_infos}

    rollouts = []
    for path in rollout_dirs:
        if path not in complete_set:
            continue
        info = parse_rollout_id(path)
        if info:
            rollouts.append(info)

    return rollouts


def parse_rollout_id(rollout_id: str) -> dict | None:
    """Parse a rollout ID into its components.

    Format: rollout_{step}_{strategy}_{YYYYMMDD_HHMMSS}[_{worker_id}]
    Strategy can be multi-word (e.g. success_replay, hard_mining, semi_success_replay).
    The date field (8 digits) is used to find the boundary between strategy and timestamp.
    """
    import re
    # Match: rollout_{digits}_{strategy}_{8-digit-date}_{time}[_{worker}]
    m = re.match(r"^rollout_(\d+)_(.+?)_(\d{8}_\d{6})(?:_(.+))?$", rollout_id)
    if not m:
        return None

    try:
        model_step = int(m.group(1))
    except ValueError:
        return None

    strategy = m.group(2)
    timestamp = m.group(3)
    machine_id = m.group(4)  # None if no worker suffix

    return {
        "rollout_id": rollout_id,
        "model_step": model_step,
        "strategy": strategy,
        "timestamp": timestamp,
        "machine_id": machine_id,
    }


def download_rollout_dataset(hf_dataset_repo: str, rollout_id: str, local_dir: str,
                             skip_complete_check: bool = False) -> str | None:
    """Download a rollout dataset from HF.

    Uses targeted per-file downloads scoped to the rollout directory,
    avoiding expensive full-repo listings.

    Args:
        skip_complete_check: If True, skip the _complete marker check
            (caller already verified completeness, e.g. during polling).
    Returns local eval_dataset path or None.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return None

    api = HfApi()

    # List only this rollout's files (scoped to subdirectory)
    try:
        entries = list(api.list_repo_tree(
            hf_dataset_repo, path_in_repo=rollout_id,
            recursive=True, repo_type="dataset",
        ))
    except Exception as e:
        logger.warning("Failed to list files for %s/%s: %s", hf_dataset_repo, rollout_id, e)
        return None

    file_paths = [e.path for e in entries if hasattr(e, 'size')]

    if not skip_complete_check:
        complete_marker = f"{rollout_id}/_complete"
        if complete_marker not in file_paths:
            logger.info("Rollout %s has no _complete marker, skipping (partial upload?)", rollout_id)
            return None

    # Download each file individually (skips already-cached files automatically)
    downloadable = [p for p in file_paths if not p.endswith("/_complete")]
    try:
        for fpath in downloadable:
            hf_hub_download(
                hf_dataset_repo,
                filename=fpath,
                local_dir=local_dir,
                repo_type="dataset",
            )
        eval_dataset = Path(local_dir) / rollout_id / "eval_dataset"
        if eval_dataset.exists():
            return str(eval_dataset)
        # Maybe the rollout dir IS the dataset (no eval_dataset subdir)
        rollout_dir = Path(local_dir) / rollout_id
        if any(rollout_dir.glob("data/chunk-*/*.parquet")):
            return str(rollout_dir)
        logger.warning("Downloaded %s but no eval_dataset found", rollout_id)
        return None
    except Exception as e:
        logger.warning("Failed to download rollout %s: %s", rollout_id, e)
        return None


def upload_rollout_as_dagger(eval_run_dir: str, rollout_id: str, hf_dataset_repo: str) -> bool:
    """Upload a rollout's eval_dataset as a dagger session.

    Pushes ``{eval_run_dir}/eval_dataset/`` to
    ``dagger/session_<rollout_id>/eval_dataset_success/`` on HF. This matches
    the layout produced by ``dagger_collect.py`` so the trainer's
    ``sync_dagger_datasets`` / ``_scan_local_dagger`` pick it up as a dagger
    session (fixed advantage, FR-proportional sampling — treated like BC).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed")
        return False

    api = HfApi()
    run_path = Path(eval_run_dir)
    eval_ds = run_path / "eval_dataset"
    if not eval_ds.exists():
        logger.warning("Eval dataset %s does not exist — cannot upload as dagger", eval_ds)
        return False

    session_name = f"session_{rollout_id}"
    hf_prefix = f"dagger/{session_name}/eval_dataset_success"

    try:
        api.create_repo(hf_dataset_repo, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/access repo %s: %s", hf_dataset_repo, e)
        return False

    try:
        api.upload_folder(
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            folder_path=str(eval_ds),
            path_in_repo=hf_prefix,
            commit_message=f"Semi-success replay (dagger): {rollout_id}",
        )
        logger.info("Uploaded rollout %s as dagger session to %s/%s",
                    rollout_id, hf_dataset_repo, hf_prefix)
        return True
    except Exception as e:
        logger.warning("Failed to upload rollout %s as dagger: %s", rollout_id, e)
        return False


def upload_rollout_dataset(eval_run_dir: str, rollout_id: str, hf_dataset_repo: str) -> bool:
    """Upload eval run folder to HF with _complete marker.

    Uploads the full eval run directory, then adds _complete marker as last operation.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed")
        return False

    api = HfApi()
    run_path = Path(eval_run_dir)
    if not run_path.exists():
        logger.warning("Eval run dir %s does not exist", eval_run_dir)
        return False

    try:
        api.create_repo(hf_dataset_repo, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/access repo %s: %s", hf_dataset_repo, e)
        return False

    try:
        api.upload_folder(
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            folder_path=str(run_path),
            path_in_repo=rollout_id,
            commit_message=f"Rollout dataset: {rollout_id}",
        )
        # Upload _complete marker as last file
        api.upload_file(
            path_or_fileobj=io.BytesIO(b"ok"),
            path_in_repo=f"{rollout_id}/_complete",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            commit_message=f"Complete marker for {rollout_id}",
        )
        logger.info("Uploaded rollout %s to %s (with _complete marker)", rollout_id, hf_dataset_repo)
        return True
    except Exception as e:
        logger.warning("Failed to upload rollout %s: %s", rollout_id, e)
        return False


def download_failure_states(hf_dataset_repo: str, local_dir: str) -> int:
    """Download failure_states/*.npz from HF, MERGE with local (don't overwrite existing).

    Returns count of new files downloaded.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return 0

    api = HfApi()
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        remote_files = api.list_repo_files(hf_dataset_repo, repo_type="dataset")
    except Exception as e:
        logger.warning("Failed to list files in %s: %s", hf_dataset_repo, e)
        return 0

    # Find all failure state files (NPZ + JSON sidecars, including per-worker subdirs)
    fs_files = [f for f in remote_files
                if f.startswith("failure_states/") and (f.endswith(".npz") or f.endswith(".json"))]

    existing = {f.name for f in local_path.glob("*.npz")}
    new_count = 0

    for remote_path in fs_files:
        # Extract just the filename (ignore subdirectory structure)
        filename = Path(remote_path).name
        # For JSON sidecars, check if matching NPZ already exists locally
        if filename.endswith(".npz") and filename in existing:
            continue
        if filename.endswith(".json") and filename.replace(".json", ".npz") in existing:
            continue

        try:
            downloaded = hf_hub_download(
                hf_dataset_repo,
                filename=remote_path,
                repo_type="dataset",
            )
            import shutil
            shutil.copy2(downloaded, str(local_path / filename))
            if filename.endswith(".npz"):
                new_count += 1
        except Exception as e:
            logger.warning("Failed to download failure state %s: %s", remote_path, e)

    if new_count:
        logger.info("Downloaded %d new failure states from HF (%d total local)",
                    new_count, len(list(local_path.glob("*.npz"))))
    return new_count


def upload_failure_states_individually(failure_states_dir: str, hf_dataset_repo: str, worker_id: str | None = None) -> bool:
    """Upload failure states using individual file uploads (concurrent-safe).

    Unlike upload_failure_states_to_hf which uses delete_patterns=["*.npz"],
    this adds files individually without deleting others, safe for concurrent workers.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False

    fs_path = Path(failure_states_dir)
    if not fs_path.exists():
        return False

    npz_files = list(fs_path.glob("*.npz"))
    if not npz_files:
        return True

    api = HfApi()
    try:
        api.create_repo(hf_dataset_repo, repo_type="dataset", exist_ok=True)
    except Exception:
        return False

    try:
        api.upload_folder(
            folder_path=str(fs_path),
            path_in_repo="failure_states",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            commit_message=f"Failure states: {len(npz_files)} files",
            allow_patterns=["*.npz", "*.json"],
        )
        logger.info("Uploaded %d failure states to %s", len(npz_files), hf_dataset_repo)
        return True
    except Exception as e:
        logger.warning("Failed to upload failure states: %s", e)
        return False


def download_success_states(hf_dataset_repo: str, local_dir: str) -> int:
    """Download success_states/*.npz from HF, MERGE with local (don't overwrite existing).

    Uses snapshot_download for batch download instead of per-file requests.
    Returns count of new files downloaded.
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        return 0

    api = HfApi()
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    existing_before = {f.name for f in local_path.glob("*.npz")}

    try:
        remote_files = api.list_repo_files(hf_dataset_repo, repo_type="dataset")
    except Exception as e:
        logger.warning("Failed to list files in %s: %s", hf_dataset_repo, e)
        return 0

    ss_files = [f for f in remote_files
                if f.startswith("success_states/") and f.endswith(".npz")]
    new_remote = [f for f in ss_files if Path(f).name not in existing_before]

    if not new_remote:
        return 0

    try:
        import shutil
        cache_dir = snapshot_download(
            hf_dataset_repo,
            repo_type="dataset",
            allow_patterns=["success_states/*.npz", "success_states/*.json"],
        )
        cached_ss_dir = Path(cache_dir) / "success_states"
        new_count = 0
        if cached_ss_dir.exists():
            for npz in cached_ss_dir.glob("*.npz"):
                if npz.name not in existing_before:
                    shutil.copy2(str(npz), str(local_path / npz.name))
                    # Also copy JSON sidecar if it exists
                    json_src = npz.with_suffix(".json")
                    if json_src.exists():
                        shutil.copy2(str(json_src), str(local_path / json_src.name))
                    new_count += 1
        if new_count:
            logger.info("Downloaded %d new success states from HF (%d total local)",
                        new_count, len(list(local_path.glob("*.npz"))))
        return new_count
    except Exception as e:
        logger.warning("Failed to download success states: %s", e)
        return 0


def upload_success_states_individually(success_states_dir: str, hf_dataset_repo: str, worker_id: str | None = None) -> bool:
    """Upload success states using individual file uploads (concurrent-safe)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False

    ss_path = Path(success_states_dir)
    if not ss_path.exists():
        return False

    npz_files = list(ss_path.glob("*.npz"))
    if not npz_files:
        return True

    api = HfApi()
    try:
        api.create_repo(hf_dataset_repo, repo_type="dataset", exist_ok=True)
    except Exception:
        return False

    try:
        api.upload_folder(
            folder_path=str(ss_path),
            path_in_repo="success_states",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            commit_message=f"Success states: {len(npz_files)} files",
            allow_patterns=["*.npz", "*.json"],
        )
        logger.info("Uploaded %d success states to %s", len(npz_files), hf_dataset_repo)
    except Exception as e:
        logger.warning("Failed to upload success states: %s", e)
        return False

    return True


def upload_semi_success_states_individually(semi_success_states_dir: str, hf_dataset_repo: str, worker_id: str | None = None) -> bool:
    """Upload semi-success states using folder upload (concurrent-safe)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False

    ss_path = Path(semi_success_states_dir)
    if not ss_path.exists():
        return False

    npz_files = list(ss_path.glob("*.npz"))
    if not npz_files:
        return True

    api = HfApi()
    try:
        api.create_repo(hf_dataset_repo, repo_type="dataset", exist_ok=True)
    except Exception:
        return False

    try:
        api.upload_folder(
            folder_path=str(ss_path),
            path_in_repo="semi_success_states",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            commit_message=f"Semi-success states: {len(npz_files)} files",
            allow_patterns=["*.npz", "*.json"],
        )
        logger.info("Uploaded %d semi-success states to %s", len(npz_files), hf_dataset_repo)
    except Exception as e:
        logger.warning("Failed to upload semi-success states: %s", e)
        return False

    return True


def delete_success_states_from_hf(
    hf_dataset_repo: str, filenames: list[str],
) -> int:
    """Delete specific success state NPZs from HF dataset repo.

    Args:
        filenames: list of NPZ filenames (not full paths), e.g. ["garment_seed42_ep0.npz"]

    Returns count of files deleted.
    """
    if not filenames:
        return 0
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return 0

    api = HfApi()
    deleted = 0
    # Batch delete via a single commit
    operations = []
    try:
        from huggingface_hub import CommitOperationDelete
        for fname in filenames:
            operations.append(CommitOperationDelete(path_in_repo=f"success_states/{fname}"))
        api.create_commit(
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Archive {len(filenames)} used success states",
        )
        deleted = len(filenames)
        logger.info("Deleted %d archived success states from HF", deleted)
    except Exception as e:
        logger.warning("Failed to delete success states from HF: %s", e)
    return deleted


def make_rollout_id(model_step: int, strategy: str, worker_id: str | None = None) -> str:
    """Generate a rollout ID.

    Format: rollout_{step}_{strategy}_{YYYYMMDD_HHMMSS}[_{worker_id}]
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"rollout_{model_step}_{strategy}_{ts}"
    if worker_id:
        base = f"{base}_{worker_id}"
    return base
