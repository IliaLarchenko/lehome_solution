"""HuggingFace Hub upload utilities for checkpoints and datasets."""

import io
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_last_numbered_step(api, repo_id: str) -> int | None:
    """Find the last numbered checkpoint folder in the HF repo."""
    try:
        steps = []
        for entry in api.list_repo_tree(repo_id, repo_type="model"):
            path = getattr(entry, "path", "")
            if path.isdigit():
                steps.append(int(path))
        return max(steps) if steps else None
    except Exception:
        logger.debug("Failed to list repo tree for %s", repo_id, exc_info=True)
        return None


def upload_checkpoint_to_hf(
    checkpoint_dir: str,
    step: int,
    repo_id: str,
    *,
    keep_period: int = 2000,
    numbered_only: bool = False,
) -> bool:
    """Upload a checkpoint to HuggingFace Hub.

    Structure on HF:
      - {step}/params/...     (inference weights)
      - {step}/assets/...     (norm stats, tokenizer)
      - latest/               (full checkpoint with train_state, updated each cycle)

    Uploads a numbered folder when gap from last numbered upload >= keep_period.
    Always updates 'latest' unless numbered_only=True.

    Args:
        numbered_only: If True, upload only the numbered checkpoint (params + assets)
            and skip the 'latest' update and history squash. Used by rollout workers
            to pin high-SR checkpoints without interfering with trainer uploads.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping upload")
        return False

    api = HfApi()
    ckpt_path = Path(checkpoint_dir) / str(step)

    if not ckpt_path.exists():
        logger.warning("Checkpoint %s does not exist, skipping upload", ckpt_path)
        return False

    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/access repo %s: %s", repo_id, e)
        return False

    success = True

    # Upload numbered folder when gap from last upload >= keep_period
    last_uploaded = _get_last_numbered_step(api, repo_id)
    if numbered_only:
        # Force numbered upload, but skip if this exact step already exists
        should_upload_numbered = last_uploaded is None or last_uploaded != step
    else:
        should_upload_numbered = (last_uploaded is None) or (step - last_uploaded >= keep_period)

    if should_upload_numbered:
        for subdir in ["params", "assets"]:
            local = ckpt_path / subdir
            if local.exists():
                try:
                    api.upload_folder(
                        repo_id=repo_id,
                        repo_type="model",
                        folder_path=str(local),
                        path_in_repo=f"{step}/{subdir}",
                        commit_message=f"Checkpoint step {step} - {subdir}",
                    )
                    logger.info("Uploaded %s/%s to %s", step, subdir, repo_id)
                except Exception as e:
                    logger.warning("Failed to upload %s/%s: %s", step, subdir, e)
                    success = False
        if success:
            logger.info("Numbered checkpoint %d uploaded (last was %s)", step, last_uploaded)

    if not numbered_only:
        # Always update 'latest' with full checkpoint (including train_state)
        try:
            # Delete old latest first
            try:
                api.delete_folder(
                    repo_id=repo_id,
                    repo_type="model",
                    path_in_repo="latest",
                    commit_message=f"Remove old latest before uploading step {step}",
                )
                # Squash immediately after delete so old latest data is purged
                # from git history before uploading the new latest
                try:
                    api.super_squash_history(repo_id=repo_id, repo_type="model")
                    logger.info("Squashed after latest/ deletion for %s", repo_id)
                except Exception as e:
                    logger.debug("Squash after delete failed (non-critical): %s", e)
            except Exception:
                logger.debug("Could not delete old latest/ folder (may not exist yet)", exc_info=True)

            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(ckpt_path),
                path_in_repo="latest",
                commit_message=f"Latest checkpoint (step {step})",
            )
            logger.info("Uploaded latest checkpoint (step %d) to %s", step, repo_id)
        except Exception as e:
            logger.warning("Failed to upload latest: %s", e)
            success = False

        # Upload step_info.json so workers can poll current step cheaply
        if success:
            upload_step_info(repo_id, step)

        # Squash commit history to avoid accumulating old "latest" snapshots
        if success:
            try:
                api.super_squash_history(repo_id=repo_id, repo_type="model")
                logger.info("Squashed commit history for %s", repo_id)
            except Exception as e:
                logger.warning("Failed to squash commit history for %s: %s", repo_id, e)

    return success


def upload_step_info(repo_id: str, step: int, iteration: int | None = None) -> bool:
    """Upload step_info.json to latest/assets/ in the model repo.

    This small file lets workers poll the current model step without
    downloading the full checkpoint.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False

    info = {"step": step}
    if iteration is not None:
        info["iteration"] = iteration

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(info).encode()),
            path_in_repo="latest/assets/step_info.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Step info: step={step}",
        )
        logger.info("Uploaded step_info.json (step=%d) to %s", step, repo_id)
        return True
    except Exception as e:
        logger.warning("Failed to upload step_info.json: %s", e)
        return False


def upload_inference_prior_to_hf(
    prior_path: str,
    repo_id: str,
    path_in_repo: str | None = None,
) -> bool:
    """Upload an inference prior JSON to the model repo root.

    Stored at repo root so it's not wiped by the latest/ delete+reupload cycle.
    `path_in_repo` defaults to the local file's basename — workers therefore
    upload to per-worker HF paths (`inference_prior_worker1.json`, etc.) and
    don't race-overwrite each other. Pass a custom name for the trainer's
    shared seed file.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping prior upload")
        return False

    prior_file = Path(prior_path)
    if not prior_file.exists():
        logger.debug("No inference prior at %s, skipping upload", prior_path)
        return False

    if path_in_repo is None:
        path_in_repo = prior_file.name

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=str(prior_file),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Inference prior update ({path_in_repo})",
        )
        logger.info("Uploaded %s -> %s/%s", prior_file.name, repo_id, path_in_repo)
        return True
    except Exception as e:
        logger.warning("Failed to upload %s: %s", path_in_repo, e)
        return False


def upload_success_rates_to_hf(
    success_rates_path: str,
    repo_id: str,
) -> bool:
    """Upload aggregated success_rates.json to the model repo root.

    Stored at repo root so it's not wiped by the latest/ delete+reupload cycle.
    Updated after every advantage recomputation.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping SR upload")
        return False

    sr_file = Path(success_rates_path)
    if not sr_file.exists():
        logger.debug("No success_rates.json at %s, skipping upload", success_rates_path)
        return False

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=str(sr_file),
            path_in_repo="success_rates.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Success rates for curriculum sampling",
        )
        logger.info("Uploaded success_rates.json to %s", repo_id)
        return True
    except Exception as e:
        logger.warning("Failed to upload success_rates.json: %s", e)
        return False


def upload_wandb_id_to_hf(
    wandb_id_path: str,
    repo_id: str,
) -> bool:
    """Upload wandb_id.txt to the model repo root.

    Stored at repo root — same wandb run for the entire distributed RL pipeline.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping wandb_id upload")
        return False

    wid_file = Path(wandb_id_path)
    if not wid_file.exists():
        logger.debug("No wandb_id.txt at %s, skipping upload", wandb_id_path)
        return False

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=str(wid_file),
            path_in_repo="wandb_id.txt",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Wandb run ID for distributed RL",
        )
        logger.info("Uploaded wandb_id.txt to %s", repo_id)
        return True
    except Exception as e:
        logger.warning("Failed to upload wandb_id.txt: %s", e)
        return False


def upload_pipeline_state_to_hf(
    pipeline_state_path: str,
    repo_id: str,
) -> bool:
    """Upload pipeline_state.json to the model repo root."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping pipeline_state upload")
        return False

    ps_file = Path(pipeline_state_path)
    if not ps_file.exists():
        logger.debug("No pipeline_state.json at %s, skipping upload", pipeline_state_path)
        return False

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=str(ps_file),
            path_in_repo="pipeline_state.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Pipeline state update",
        )
        logger.info("Uploaded pipeline_state.json to %s", repo_id)
        return True
    except Exception as e:
        logger.warning("Failed to upload pipeline_state.json: %s", e)
        return False


def download_model_asset(
    repo_id: str,
    asset_name: str,
    local_path: str,
) -> bool:
    """Download a single asset from the model repo root.

    Returns True if downloaded, False if not found or error.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    try:
        # Use default HF cache (no local_dir) — the shared /tmp/hf_model_assets
        # symlink dir caused silent failures after super_squash_history rewrote
        # the commit history: local_dir metadata pinned to a squashed commit,
        # hf_hub_download then failed with EntryNotFoundError even though the
        # file was present on main. Cache-based path has no such pinning.
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=asset_name,
            repo_type="model",
        )
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local_path)
        logger.info("Downloaded %s from %s -> %s", asset_name, repo_id, local_path)
        return True
    except Exception as e:
        logger.warning("Could not download %s from %s: %s", asset_name, repo_id, e)
        return False


def upload_dataset_to_hf(
    eval_dataset_dir: str,
    eval_run_id: str,
    repo_id: str,
) -> bool:
    """Upload an eval dataset folder to HuggingFace Hub.

    Structure on HF: {eval_run_id}/...  (the eval_dataset contents as-is)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping upload")
        return False

    api = HfApi()
    dataset_path = Path(eval_dataset_dir)

    if not dataset_path.exists():
        logger.warning("Dataset %s does not exist, skipping upload", dataset_path)
        return False

    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/access repo %s: %s", repo_id, e)
        return False

    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(dataset_path),
            path_in_repo=eval_run_id,
            commit_message=f"Rollout dataset: {eval_run_id}",
        )
        logger.info("Uploaded dataset %s to %s/%s", dataset_path, repo_id, eval_run_id)
        return True
    except Exception as e:
        logger.warning("Failed to upload dataset: %s", e)
        return False


def upload_failure_states_to_hf(
    failure_states_dir: str,
    repo_id: str,
) -> bool:
    """Upload failure states (NPZ files) to HuggingFace Hub dataset repo.

    Structure on HF:
      - failure_states/*.npz       (current unsolved failures)
      - failure_states_solved/*.npz (previously failed, now solved)

    Files that exist on HF in failure_states/ but not locally are moved
    to failure_states_solved/ (they were removed locally by _remove_solved_failures).
    """
    try:
        from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping upload")
        return False

    fs_path = Path(failure_states_dir)
    if not fs_path.exists():
        logger.warning("Failure states dir %s does not exist, skipping upload", fs_path)
        return False

    npz_files = list(fs_path.glob("*.npz"))
    local_names = {f.name for f in npz_files}

    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/access repo %s: %s", repo_id, e)
        return False

    # Find remote files to move to solved
    try:
        remote_files = api.list_repo_files(repo_id, repo_type="dataset")
        remote_failure_names = {
            f.split("/", 1)[1]
            for f in remote_files
            if f.startswith("failure_states/") and f.endswith(".npz")
        }
    except Exception:
        logger.debug("Failed to list remote failure states", exc_info=True)
        remote_failure_names = set()

    solved_names = remote_failure_names - local_names
    if solved_names:
        try:
            operations = []
            for name in sorted(solved_names):
                operations.append(
                    CommitOperationCopy(
                        src_path_in_repo=f"failure_states/{name}",
                        path_in_repo=f"failure_states_solved/{name}",
                    )
                )
                operations.append(
                    CommitOperationDelete(path_in_repo=f"failure_states/{name}")
                )
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Move {len(solved_names)} solved failures to failure_states_solved",
            )
            logger.info(
                "Moved %d solved failure states to failure_states_solved/", len(solved_names)
            )
        except Exception as e:
            logger.warning("Failed to move solved failure states: %s", e)

    # Upload current failure states
    if not npz_files:
        logger.info("No failure state NPZ files to upload")
        return True

    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(fs_path),
            path_in_repo="failure_states",
            commit_message=f"Failure states ({len(npz_files)} files)",
            delete_patterns=["*.npz", "*.json"],
        )
        logger.info("Uploaded %d failure states to %s/failure_states", len(npz_files), repo_id)
        return True
    except Exception as e:
        logger.warning("Failed to upload failure states: %s", e)
        return False


def upload_success_states_to_hf(
    success_states_dir: str,
    repo_id: str,
) -> bool:
    """Upload success states (NPZ files) to HuggingFace Hub dataset repo.

    Structure on HF:
      - success_states/*.npz       (current available states)

    Files that exist on HF in success_states/ but not locally are deleted
    (they were consumed locally).
    """
    try:
        from huggingface_hub import CommitOperationDelete, HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping upload")
        return False

    ss_path = Path(success_states_dir)
    if not ss_path.exists():
        logger.warning("Success states dir %s does not exist, skipping upload", ss_path)
        return False

    npz_files = list(ss_path.glob("*.npz"))
    local_names = {f.name for f in npz_files}

    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/access repo %s: %s", repo_id, e)
        return False

    # Find remote files to delete (consumed locally)
    try:
        remote_files = api.list_repo_files(repo_id, repo_type="dataset")
        remote_success_names = {
            f.split("/", 1)[1]
            for f in remote_files
            if f.startswith("success_states/") and f.endswith(".npz")
        }
    except Exception:
        logger.debug("Failed to list remote success states", exc_info=True)
        remote_success_names = set()

    consumed_names = remote_success_names - local_names
    if consumed_names:
        try:
            operations = [
                CommitOperationDelete(path_in_repo=f"success_states/{name}")
                for name in sorted(consumed_names)
            ]
            # Also delete JSON sidecars for consumed states
            for name in sorted(consumed_names):
                json_name = name.replace(".npz", ".json")
                operations.append(
                    CommitOperationDelete(path_in_repo=f"success_states/{json_name}")
                )
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Remove {len(consumed_names)} consumed success states",
            )
            logger.info(
                "Removed %d consumed success states from HF", len(consumed_names)
            )
        except Exception as e:
            logger.warning("Failed to remove consumed success states from HF: %s", e)

    # Upload current success states
    if not npz_files:
        logger.info("No success state NPZ files to upload")
        return True

    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(ss_path),
            path_in_repo="success_states",
            commit_message=f"Success states ({len(npz_files)} files)",
            delete_patterns=["*.npz", "*.json"],
        )
        logger.info("Uploaded %d success states to %s/success_states", len(npz_files), repo_id)
    except Exception as e:
        logger.warning("Failed to upload success states: %s", e)
        return False

    return True


def upload_semi_success_states_to_hf(
    semi_success_states_dir: str,
    repo_id: str,
) -> bool:
    """Upload semi-success states (NPZ + JSON files) to HuggingFace Hub dataset repo.

    Structure on HF:
      - semi_success_states/*.npz + *.json
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping upload")
        return False

    ss_path = Path(semi_success_states_dir)
    if not ss_path.exists():
        logger.warning("Semi-success states dir %s does not exist, skipping upload", ss_path)
        return False

    npz_files = list(ss_path.glob("*.npz"))
    if not npz_files:
        logger.info("No semi-success state NPZ files to upload")
        return True

    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/access repo %s: %s", repo_id, e)
        return False

    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(ss_path),
            path_in_repo="semi_success_states",
            commit_message=f"Semi-success states ({len(npz_files)} files)",
            delete_patterns=["*.npz", "*.json"],
        )
        logger.info("Uploaded %d semi-success states to %s/semi_success_states", len(npz_files), repo_id)
    except Exception as e:
        logger.warning("Failed to upload semi-success states: %s", e)
        return False

    return True
