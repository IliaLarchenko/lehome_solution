#!/usr/bin/env python3
"""Parallel evaluation: single shared policy server, N workers with independent sessions.

Architecture:
  Isaac Sim (thin client)  →  Worker proxy  →  Policy server (serve.py :8000)
                                            →  Value server  (serve.py :8001, optional)

Each worker runs one Isaac Sim process for a batch of garments, switching between
them via env.switch_garment() (no restart).  The worker's proxy thread handles
all policy/value calls and writes temp pkls (video_dir/temp_episodes/).
The orchestrator's background dataset thread appends pkls to the LeRobot dataset
one at a time (safe for LeRobotDataset).  No heavy work blocks the main loop.

Camera resolution defaults to 640×480.  Lower resolutions (e.g. 320×240) are
supported for debugging experiments, but behavior differences are currently under
active investigation.

Usage:
    uv run python scripts/run_eval.py                      # unseen garments, 4 workers
    uv run python scripts/run_eval.py --all                # all garments
    uv run python scripts/run_eval.py --num_workers 4
    uv run python scripts/run_eval.py --num_episodes 1 --garment_types top_long --no_wandb
    uv run python scripts/run_eval.py --no_save_dataset    # skip writing eval dataset
"""

import argparse
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta
from pathlib import Path

from lehome_solution.eval import (
    ALL_GARMENT_TYPES,
    ensure_isaacsim_env,
    garment_name_to_type,
    get_garments,
)
from lehome_solution.utils.logging_config import log_path as _log_path
from lehome_solution.utils import logging_config as logcfg

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
ensure_isaacsim_env(REPO_ROOT)

# Unbuffer stdout/stderr so progress is visible when run under uv or in background
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_BASE = REPO_ROOT / "outputs" / "checkpoints"
DEFAULT_CHECKPOINT_STEP = 7000


def find_latest_checkpoint() -> str | None:
    if not CHECKPOINT_BASE.exists():
        return None
    candidates = []
    for params_dir in CHECKPOINT_BASE.rglob("params"):
        if not params_dir.is_dir():
            continue
        ckpt_dir = params_dir.parent
        rel = ckpt_dir.relative_to(CHECKPOINT_BASE)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if not (ckpt_dir / "assets").is_dir():
            continue
        try:
            mtime = max(f.stat().st_mtime for f in params_dir.rglob("*") if f.is_file())
        except ValueError:
            continue
        candidates.append((mtime, ckpt_dir))
    if not candidates:
        return None
    # Prefer checkpoint at default step (7000) when present
    default_step_candidates = [(m, c) for m, c in candidates if c.name == str(DEFAULT_CHECKPOINT_STEP)]
    if default_step_candidates:
        default_step_candidates.sort(reverse=True)
        logger.info(f"Auto-detected checkpoint (step {DEFAULT_CHECKPOINT_STEP}): {default_step_candidates[0][1]}")
        return str(default_step_candidates[0][1])
    candidates.sort(reverse=True)
    logger.info(f"Auto-detected checkpoint: {candidates[0][1]}")
    return str(candidates[0][1])


def make_eval_run_id(checkpoint_dir: str, prefix: str = "step") -> str:
    from datetime import datetime
    m = re.search(r"/(\d+)/?$", checkpoint_dir.rstrip("/"))
    step = m.group(1) if m else None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{step}_{ts}" if step else f"ckpt_{ts}"



def _bernoulli_ci90(n: int, s: int) -> list[float]:
    """90% CI for Bernoulli success rate using normal approximation."""
    if n == 0:
        return [0.0, 0.0]
    p = s / n
    z = 1.645  # 90% CI
    se = (p * (1 - p) / n) ** 0.5
    return [max(0.0, p - z * se), min(1.0, p + z * se)]


def _avg_reward(eps: list[dict]) -> float:
    """Simple average of dense_return (or binary success as fallback)."""
    if not eps:
        return 0.0
    total = 0.0
    for ep in eps:
        if "dense_return" in ep:
            total += ep["dense_return"]
        else:
            total += 1.0 if ep.get("success") else 0.0
    return total / len(eps)


def _build_eval_summary(episodes: list[dict]) -> dict:
    """Build summary with success rate, avg reward, and 90% CIs for SR."""
    by_type: dict[str, list[dict]] = {}
    by_garment: dict[str, list[dict]] = {}
    for ep in episodes:
        g = ep.get("garment", "")
        gt = garment_name_to_type(g)
        by_type.setdefault(gt, []).append(ep)
        if g:
            by_garment.setdefault(g, []).append(ep)

    def _base_stats(eps: list[dict]) -> dict:
        n = len(eps)
        s = sum(1 for e in eps if e.get("success"))
        return {"total": n, "successes": s, "success_rate": s / n if n else 0.0}

    # --- Per garment type ---
    summary_by_type: dict[str, dict] = {}
    for gt, eps in sorted(by_type.items()):
        d = _base_stats(eps)
        d["sr_ci90"] = _bernoulli_ci90(d["total"], d["successes"])
        d["avg_reward"] = _avg_reward(eps)
        summary_by_type[gt] = d

    # --- Per garment (no CI, just counts) ---
    summary_by_garment: dict[str, dict] = {}
    for gname, eps in sorted(by_garment.items()):
        d = _base_stats(eps)
        d["avg_reward"] = _avg_reward(eps)
        summary_by_garment[gname] = d

    # --- Overall ---
    overall = _base_stats(episodes)
    overall["sr_ci90"] = _bernoulli_ci90(overall["total"], overall["successes"])
    overall["avg_reward"] = _avg_reward(episodes)

    # Weighted overall (equal weight per garment type)
    if summary_by_type:
        nt = len(summary_by_type)
        overall["weighted_sr"] = sum(d["success_rate"] for d in summary_by_type.values()) / nt
        # CI for weighted SR: average per-type CIs (matches the weighted_sr definition)
        type_cis = [d.get("sr_ci90", [d["success_rate"], d["success_rate"]]) for d in summary_by_type.values()]
        overall["weighted_sr_ci90"] = [
            sum(ci[0] for ci in type_cis) / nt,
            sum(ci[1] for ci in type_cis) / nt,
        ]
        overall["weighted_reward"] = sum(d["avg_reward"] for d in summary_by_type.values()) / nt

    return {"by_type": summary_by_type, "by_garment": summary_by_garment, "overall": overall}


def _format_eval_summary(
    summary: dict,
    *,
    elapsed_s: float = 0.0,
    episodes_done: int = 0,
    total_expected: int = 0,
    total_processed: int | None = None,
    run_info: dict | None = None,
) -> str:
    """Format eval summary as human-readable text, including timing info."""
    lines = []

    # --- Run info header ---
    if run_info:
        for k, v in run_info.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    # --- Timing block ---
    h, rem = divmod(int(elapsed_s), 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    _processed = total_processed if total_processed is not None else episodes_done
    _discarded = _processed - episodes_done
    pct = f"{_processed / total_expected:.1%}" if total_expected else "?"
    lines.append(f"Run time:          {elapsed_str}")
    if _discarded > 0:
        lines.append(f"Episodes:          {_processed} / {total_expected} processed ({pct}), "
                      f"{episodes_done} saved, {_discarded} discarded")
    else:
        lines.append(f"Episodes done:     {episodes_done} / {total_expected}  ({pct})")

    if _processed > 0 and elapsed_s > 0:
        spe = elapsed_s / _processed
        remaining = max(0, total_expected - _processed)
        eta_s = remaining * spe
        em, es = divmod(int(eta_s), 60)
        eh, em = divmod(em, 60)
        eta_str = (f"{eh}h {em}m {es}s" if eh else f"{em}m {es}s") if remaining > 0 else "0s (completed)"
        finish_wall = (datetime.now() + timedelta(seconds=eta_s)).strftime("%H:%M") if remaining > 0 else "now"
        lines.append(f"Time per episode:  {spe:.1f}s avg")
        lines.append(f"ETA:               {eta_str}  (~{finish_wall} local)")
    else:
        lines.append("Time per episode:  -- (no data yet)")
        lines.append("ETA:               --")

    lines.append("-" * 50)
    lines.append("EVAL SUMMARY")
    lines.append("=" * 50)
    o = summary.get("overall", {})

    # Helper: format value with optional 90% CI bracket
    def _fmt_pct(val: float, ci: list | None) -> str:
        s = f"{val:.1%}"
        if ci:
            s += f" [{ci[0]:.1%}-{ci[1]:.1%}]"
        return s

    # Overall line
    overall_line = (
        f"OVERALL: {o.get('successes', 0)}/{o.get('total', 0)}"
        f"  SR={_fmt_pct(o.get('success_rate', 0), o.get('sr_ci90'))}"
    )
    if "avg_reward" in o:
        overall_line += f"  reward={o['avg_reward']:.3f}"
    lines.append(overall_line)

    # Weighted overall (equal weight per garment type)
    if "weighted_sr" in o:
        wline = (
            f"WEIGHTED:"
            f"  SR={_fmt_pct(o['weighted_sr'], o.get('weighted_sr_ci90'))}"
        )
        if "weighted_reward" in o:
            wline += f"  reward={o['weighted_reward']:.3f}"
        lines.append(wline)

    lines.append("")
    lines.append("By garment type:")
    for gt, d in summary.get("by_type", {}).items():
        line = (
            f"  {gt}: {d.get('successes', 0)}/{d.get('total', 0)}"
            f"  SR={_fmt_pct(d.get('success_rate', 0), d.get('sr_ci90'))}"
        )
        if "avg_reward" in d:
            line += f"  reward={d['avg_reward']:.3f}"
        lines.append(line)
    lines.append("")
    lines.append("By garment:")
    for gname, d in summary.get("by_garment", {}).items():
        r = d.get("success_rate", 0)
        line = f"  {gname}: {d.get('successes', 0)}/{d.get('total', 0)} ({r:.1%})"
        if "avg_reward" in d:
            line += f"  avg_reward={d['avg_reward']:.3f}"
        lines.append(line)

    # Prediction metrics (if available)
    pm = summary.get("prediction_metrics", {})
    if pm:
        lines.append("")
        lines.append("Prediction metrics:")
        for prefix, label in [("rollout_value", "Success"), ("rollout_checkpoint", "Checkpoint")]:
            acc = pm.get(f"{prefix}/accuracy")
            auc = pm.get(f"{prefix}/roc_auc")
            if acc is not None:
                line = f"  {label}: acc={acc:.3f}"
                if auc is not None:
                    line += f" AUC={auc:.3f}"
                lines.append(line)
        gt_acc = pm.get("rollout_garment/accuracy")
        if gt_acc is not None:
            line = f"  Garment type: acc={gt_acc:.3f}"
            gt5 = pm.get("rollout_garment/accuracy_first5")
            if gt5 is not None:
                line += f" first5={gt5:.3f}"
            lines.append(line)

    return "\n".join(lines)


def _compute_prediction_metrics(eval_dataset_root: str) -> dict:
    """Compute value/checkpoint/garment prediction metrics from eval dataset.

    Only includes episodes from random/full rollout types (unbiased).
    Lightweight version that doesn't import run_rl_pipeline.
    """
    import glob
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}

    _METRICS_INCLUDED_ROLLOUT_TYPES = {"random", "full", "normal"}

    pq_files = sorted(glob.glob(os.path.join(eval_dataset_root, "data/chunk-*/*.parquet")))
    if not pq_files:
        return {}

    # Load episode metadata for ground truth
    meta_dir = Path(eval_dataset_root) / "meta" / "eval_episode_metadata"
    ep_meta: dict[int, dict] = {}
    if meta_dir.exists():
        for mf in sorted(meta_dir.glob("*.json")):
            try:
                m = json.load(open(mf))
                ep_idx = int(mf.stem.split("_")[-1])
                ep_meta[ep_idx] = m
            except Exception:
                continue

    # Build set of episodes from random/full rollout types only
    included_episodes: set[int] = set()
    for ep_idx, m in ep_meta.items():
        rt = m.get("rollout_type", "normal")
        if rt in _METRICS_INCLUDED_ROLLOUT_TYPES:
            included_episodes.add(ep_idx)
    has_metadata = bool(ep_meta)

    all_value_preds, all_success_labels = [], []
    all_cp_preds, all_cp_labels = [], []
    all_gt_preds, all_gt_labels = [], []

    for pf in pq_files:
        try:
            t = pq.read_table(pf)
        except Exception:
            continue
        col_names = t.column_names
        val_col = "success_pred" if "success_pred" in col_names else "value"
        values = t[val_col].to_pylist()
        successes = t["success"].to_pylist()
        ep_indices = t["episode_index"].to_pylist()
        cp_col = "checkpoint_pred" if "checkpoint_pred" in col_names else "checkpoint_value" if "checkpoint_value" in col_names else None
        cp_values = t[cp_col].to_pylist() if cp_col else None
        gt_preds_col = t["garment_type_pred"].to_pylist() if "garment_type_pred" in col_names else None

        for row_idx in range(t.num_rows):
            ep_idx = int(ep_indices[row_idx])

            # Filter: only random/full rollout types for metrics
            if has_metadata and ep_idx not in included_episodes:
                continue

            pv = values[row_idx]
            lv = successes[row_idx]
            pv = pv[0] if isinstance(pv, list) else pv
            lv = lv[0] if isinstance(lv, list) else lv
            if pv is not None and lv is not None:
                all_value_preds.append(float(pv))
                all_success_labels.append(float(lv))

            if cp_values is not None and ep_idx in ep_meta:
                cpv = cp_values[row_idx]
                cpv = cpv[0] if isinstance(cpv, list) else cpv
                if cpv is not None:
                    max_r = ep_meta[ep_idx].get("max_reward", ep_meta[ep_idx].get("dense_return", 0.0))
                    all_cp_preds.append(float(cpv))
                    all_cp_labels.append(1.0 if float(max_r) >= 0.5 else 0.0)

            if gt_preds_col is not None and ep_idx in ep_meta:
                gtv = gt_preds_col[row_idx]
                gtv = gtv[0] if isinstance(gtv, list) else gtv
                if gtv is not None and int(gtv) >= 0:
                    gt_label = ep_meta[ep_idx].get("garment_type_id", -1)
                    if gt_label >= 0:
                        all_gt_preds.append(int(gtv))
                        all_gt_labels.append(int(gt_label))

    metrics = {}

    def _binary_metrics(preds_list, labels_list, prefix):
        if len(preds_list) < 10:
            return {}
        preds = np.array(preds_list)
        labels = np.array(labels_list)
        eps = 1e-7
        pc = np.clip(preds, eps, 1 - eps)
        m = {
            f"{prefix}/accuracy": float(np.mean((preds > 0.5) == (labels > 0.5))),
            f"{prefix}/bce": float(-np.mean(labels * np.log(pc) + (1 - labels) * np.log(1 - pc))),
        }
        if len(np.unique(labels)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                m[f"{prefix}/roc_auc"] = float(roc_auc_score(labels, preds))
            except Exception:
                pass
        return m

    metrics.update(_binary_metrics(all_value_preds, all_success_labels, "rollout_value"))
    metrics.update(_binary_metrics(all_cp_preds, all_cp_labels, "rollout_checkpoint"))

    if all_gt_preds:
        gt_preds = np.array(all_gt_preds)
        gt_labels = np.array(all_gt_labels)
        metrics["rollout_garment/accuracy"] = float(np.mean(gt_preds == gt_labels))
        metrics["rollout_garment/n_frames"] = len(gt_preds)

    return metrics


def _write_eval_summary(
    video_path: Path,
    all_episodes: list[dict],
    *,
    start_time: float | None = None,
    total_expected: int = 0,
    total_processed: int | None = None,
    run_info: dict | None = None,
    eval_dataset_root: Path | None = None,
) -> None:
    """Write eval_summary.json and eval_summary.txt. Call whenever eval_results.json is updated."""
    summary = _build_eval_summary(all_episodes)

    # Compute prediction metrics from eval dataset if available
    if eval_dataset_root is not None and eval_dataset_root.exists():
        try:
            pred_metrics = _compute_prediction_metrics(str(eval_dataset_root))
            if pred_metrics:
                summary["prediction_metrics"] = pred_metrics
        except Exception as e:
            logger.warning("Could not compute prediction metrics: %s", e)
    summary_path = video_path / "eval_summary.json"
    elapsed_s = time.time() - start_time if start_time is not None else 0.0
    _processed = total_processed if total_processed is not None else len(all_episodes)
    _discarded = _processed - len(all_episodes)
    timing = {
        "elapsed_s": round(elapsed_s, 1),
        "episodes_done": len(all_episodes),
        "episodes_discarded": _discarded,
        "total_processed": _processed,
        "total_expected": total_expected,
        "time_per_episode_s": round(elapsed_s / _processed, 1) if _processed else None,
    }
    with open(summary_path, "w") as f:
        json.dump({**summary, "timing": timing, "run_info": run_info or {}}, f, indent=2)
    summary_txt = video_path / "eval_summary.txt"
    with open(summary_txt, "w") as f:
        f.write(_format_eval_summary(
            summary,
            elapsed_s=elapsed_s,
            episodes_done=len(all_episodes),
            total_expected=total_expected,
            total_processed=_processed,
            run_info=run_info,
        ))


def wait_for_server(port: int, timeout: int = 120, proc: subprocess.Popen | None = None) -> bool:
    import socket
    start = time.time()
    while time.time() - start < timeout:
        # If the server process already exited, don't keep waiting
        if proc is not None and proc.poll() is not None:
            logger.error("Server process exited with code %d — check server.log", proc.returncode)
            return False
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    return False


def start_server(
    checkpoint_dir: str,
    config_name: str,
    port: int,
    log_dir: str,
    integrated_value: bool = True,
    use_flash_attention: bool = True,
    use_xsa: bool = False,
    num_rollout_candidates: int = 1,
) -> subprocess.Popen:
    """Start policy server. log_dir is used only for server.log path."""
    cmd = [sys.executable, "scripts/serve.py"]
    cmd += [f"--port={port}"]
    if not integrated_value:
        cmd += ["--no-integrated-value"]
    if not use_flash_attention:
        cmd += ["--use-flash-attention=False"]
    if use_xsa:
        cmd += ["--use-xsa=True"]
    if num_rollout_candidates > 1:
        cmd += [f"--num-rollout-candidates={num_rollout_candidates}"]
    cmd += [
        "policy:checkpoint",
        f"--policy.config={config_name}",
        f"--policy.dir={checkpoint_dir}",
    ]
    logger.info(f"Starting server: {' '.join(cmd)}")
    # Log server output to file for debugging
    server_log = _log_path(log_dir, logcfg.SERVER, "server.log")
    log_fh = open(server_log, "w")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    except Exception:
        log_fh.close()
        raise
    # Keep reference so GC doesn't close it while server runs.
    # Closed explicitly in _cleanup_server().
    proc._log_fh = log_fh
    logger.info(f"Server log: {server_log}")
    return proc


# --- W&B ---

def init_wandb(checkpoint_dir, garment_types, project, eval_run_id, garment_subset):
    try:
        import wandb
    except ImportError:
        return None, None
    m = re.search(r"/(\d+)/?$", checkpoint_dir.rstrip("/"))
    step = int(m.group(1)) if m else None
    run = wandb.init(
        project=project, name=f"eval-{eval_run_id}", job_type="eval",
        config={"checkpoint_dir": checkpoint_dir, "step": step,
                "eval_run_id": eval_run_id, "garment_types": garment_types,
                "subset": garment_subset},
    )
    return run, step


def finish_wandb(run, all_episodes, video_dir, step=None):
    if run is None:
        return
    import wandb

    video_path = Path(video_dir)
    columns = ["garment_type", "garment", "split", "seed", "return", "length", "success", "video"]
    table = wandb.Table(columns=columns)

    for ep in all_episodes:
        gt = ep.get("garment", "").split("_Seen")[0].split("_Unseen")[0].lower()
        video_cell = None
        if video_path.exists():
            matches = list(video_path.glob(f"*_seed{ep.get('seed', '')}.mp4"))
            if matches:
                try:
                    video_cell = wandb.Video(str(matches[0]), fps=30, format="mp4")
                except Exception:
                    pass
        table.add_data(gt, ep.get("garment", ""), ep.get("split", ""),
                       ep.get("seed", ""), ep["return"], ep["length"],
                       ep["success"], video_cell)
    run.log({"eval/results": table})

    # Per garment-type tables
    by_gt = {}
    for ep in all_episodes:
        gt = ep.get("garment", "").split("_Seen")[0].split("_Unseen")[0].lower()
        by_gt.setdefault(gt, []).append(ep)
    for gt, eps in by_gt.items():
        gt_table = wandb.Table(columns=["garment", "split", "seed", "success", "return", "video"])
        for ep in eps:
            vc = None
            if video_path.exists():
                matches = list(video_path.glob(f"*_seed{ep.get('seed', '')}.mp4"))
                if matches:
                    try:
                        vc = wandb.Video(str(matches[0]), fps=30, format="mp4")
                    except Exception:
                        pass
            gt_table.add_data(ep.get("garment", ""), ep.get("split", ""),
                              ep.get("seed", ""), ep["success"], ep["return"], vc)
        run.log({f"eval/{gt}/videos": gt_table})

    # Use _build_eval_summary for consistent stats (includes CIs and weighted)
    summary = _build_eval_summary(all_episodes)
    o = summary.get("overall", {})
    run.summary["eval/overall/success_rate"] = o.get("success_rate", 0)
    run.summary["eval/overall/avg_return"] = (
        sum(e["return"] for e in all_episodes) / len(all_episodes) if all_episodes else 0
    )
    run.summary["eval/overall/num_episodes"] = o.get("total", 0)
    if "avg_reward" in o:
        run.summary["eval/overall/avg_reward"] = o["avg_reward"]
    if "sr_ci90" in o:
        run.summary["eval/overall/sr_ci90_lo"] = o["sr_ci90"][0]
        run.summary["eval/overall/sr_ci90_hi"] = o["sr_ci90"][1]
    if "weighted_sr" in o:
        run.summary["eval/overall/weighted_sr"] = o["weighted_sr"]
    if "weighted_reward" in o:
        run.summary["eval/overall/weighted_reward"] = o["weighted_reward"]
    if step is not None:
        run.summary["train/step"] = step

    artifact = wandb.Artifact(f"eval-results-step{step}" if step else "eval-results", type="eval")
    results_path = video_path / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"episodes": all_episodes}, f, indent=2)
    artifact.add_file(str(results_path))
    run.log_artifact(artifact)
    run.finish()
    logger.info(f"W&B run: {run.url}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Parallel evaluation with shared policy server")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--config_name", type=str, default="pi_modified_bc_rl")
    parser.add_argument("--garment_types", type=str, nargs="+", default=ALL_GARMENT_TYPES,
                        choices=ALL_GARMENT_TYPES)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wandb_project", type=str, default="LeHome")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--video_dir", type=str, default="outputs/eval_videos")
    parser.add_argument("--no_save_dataset", action="store_true", help="Do not write LeRobot eval dataset")
    parser.add_argument("--no_save_debug_video", action="store_true",
                        help="Skip per-episode debug video (3-cam grid + overlay). When set, all related work — frame composition, GAE-for-overlay, ffmpeg, thread pool — is bypassed.")
    parser.add_argument("--no_save_pkl", action="store_true",
                        help="Skip per-episode trajectory pkls + physics-state snapshots "
                             "(saves disk + time; success metrics still reported). "
                             "Implies --no_save_dataset and --no_save_debug_video.")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Success rates only: shorthand for --no_save_pkl "
                             "--no_save_dataset --no_save_debug_video.")
    parser.add_argument("--camera_width", type=int, default=640,
                        help="Isaac Sim camera width (default 640)")
    parser.add_argument("--camera_height", type=int, default=480,
                        help="Isaac Sim camera height (default 480)")
    parser.add_argument("--dataset_width", type=int, default=None,
                        help="Dataset image width (default: same as --camera_width). "
                             "Set smaller than camera to downscale frames at dataset-write time "
                             "(faster writes, smaller parquets). Policy still receives full-res sim frames.")
    parser.add_argument("--dataset_height", type=int, default=None,
                        help="Dataset image height (default: same as --camera_height).")
    parser.add_argument("--top_camera_width", type=int, default=None,
                        help="Override TOP camera render width (real-aligned: 1280). "
                             "Wrist cameras keep --camera_width.")
    parser.add_argument("--top_camera_height", type=int, default=None,
                        help="Override TOP camera render height (real-aligned: 720).")
    parser.add_argument("--dataset_format", choices=("sim", "real_bc"), default="sim",
                        help="'sim' (default): standard EvalDatasetWriter schema with all aux "
                             "fields. 'real_bc': RealFormatDatasetWriter mirroring "
                             "data/lehome_real/four_types_merged schema (20 Hz, top 720x1280 "
                             "real-canonical, wrist 480x640, garment via task_index, "
                             "no aux fields). Cube-spline resample 30Hz->20Hz with 2x wall-clock "
                             "stretch to match real teleop cadence.")
    parser.add_argument("--dataset_units", choices=("radians", "degrees"), default="radians",
                        help="State/action units for dataset_format=real_bc. 'radians' "
                             "(default): sim USD radians unchanged. 'degrees': real-robot "
                             "degree-mode convention (arm joints in degrees, gripper 0-100) — "
                             "use when the dataset will be mixed with real recordings. "
                             "Motor-pct is not supported.")
    parser.add_argument("--depth", action="store_true", default=False,
                        help="Enable depth camera rendering (default: off — depth is unused by "
                             "the policy and adds GPU + network overhead)")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--all", action="store_const", const="all", dest="garment_subset")
    subset.add_argument("--seen_only", action="store_const", const="seen", dest="garment_subset")
    parser.set_defaults(garment_subset="unseen")

    # RL data collection / randomized eval params
    parser.add_argument("--randomize_garments", action="store_true",
                        help="Randomly sample garments instead of using the full fixed subset. "
                             "Pairs with --seen_only for RL data collection.")
    parser.add_argument("--num_garments", type=int, default=None,
                        help="How many garments to sample when --randomize_garments (default: all available)")
    parser.add_argument("--same_seed", action="store_true",
                        help="All episodes of each garment use the same base seed (test policy variance "
                             "at identical initial conditions). Default: seed increments per episode.")
    parser.add_argument("--rl_seed", type=int, default=None,
                        help="RNG seed for garment sampling (--randomize_garments). None = random each run.")
    parser.add_argument("--integrated_value", action="store_true", default=True,
                        help="Use integrated success head from policy model. Enabled by default. "
                             "P(success) predictions come from infer_chunk responses.")
    parser.add_argument("--no_value", action="store_true", default=False,
                        help="Disable success head (no P(success) predictions, no advantage computation).")
    parser.add_argument("--garment_list", type=str, nargs="+", default=None,
                        help="Explicit list of garment names to evaluate (overrides --randomize_garments). "
                             "Used by pipeline for curriculum/hard mining strategies.")
    parser.add_argument("--task_overrides_file", type=str, default=None,
                        help="JSON file with [{garment, seed}, ...] for exact replay. "
                             "Each entry becomes a single-episode task with the exact seed. "
                             "Used by hard_mining strategy.")
    parser.add_argument("--prior_file", type=str, default=None,
                        help="JSON file with inference optimization prior state. "
                             "When provided, each episode samples hyperparameters from the prior.")
    parser.add_argument("--per_garment_type_inference_config", type=str, default=None,
                        help="JSON dict mapping garment_type -> inference param overrides "
                             "(actions_to_execute, k_execute, actions_to_keep, "
                             "time_threshold_inpaint, cfg_scale, noise_temperature, "
                             "num_rollout_candidates).  When set, takes precedence over "
                             "Thompson Sampling: each episode uses the matching garment-type "
                             "config as a fixed inference_config.")
    parser.add_argument("--success_rates_file", type=str, default=None,
                        help="JSON file with success rates (by_type/by_garment). "
                             "Used for adaptive success state saving probability.")
    parser.add_argument("--noise_temperature", type=float, default=1.0,
                        help="Noise temperature for exploration (scales initial noise covariance). "
                             "1.0 = no change, >1.0 = more exploration.")
    # DART-style exploration noise (rollout-only, never used at submission). Both
    # must be > 0 for noise to fire — defaults are bit-identical no-op. Eval/
    # submission paths intentionally do not surface these flags.
    parser.add_argument("--explore_noise_prob", type=float, default=0.0,
                        help="Per-chunk Bernoulli probability of injecting DART-style "
                             "correlated noise into the sampled actions. 0 = disabled. "
                             "ROLLOUT-ONLY; never set on the submission path.")
    parser.add_argument("--explore_noise_scale", type=float, default=0.0,
                        help="Magnitude (in normalized-action units) of the additive "
                             "correlated noise sampled from the action covariance. "
                             "0 = disabled. ROLLOUT-ONLY.")
    parser.add_argument("--num_rollout_candidates", type=int, default=1,
                        help="Best-of-N rollout sampling. When >1, each inference request "
                             "is expanded into N candidates with independent denoising noise; "
                             "the argmax by avg wm_flow joint score is returned.")
    # GAE overlay params — must mirror pipeline advantage config so the
    # debug video matches what training actually uses.
    parser.add_argument("--gae_lambda", type=float, default=0.99)
    parser.add_argument("--success_alpha", type=float, default=0.5)
    parser.add_argument("--completion_alpha", type=float, default=0.5)
    parser.add_argument("--value_tail_k", type=int, default=30)
    parser.add_argument("--garment_pattern_augmentation_p", type=float, default=0.0,
                        help="(Legacy) Probability to swap garment texture pattern.")
    parser.add_argument("--garment_color_augmentation_p", type=float, default=0.0,
                        help="(Legacy) Probability to remap garment colors in LAB space.")
    parser.add_argument("--aug_config", type=str, default=None,
                        help="JSON string or path to augmentation config file.")
    parser.add_argument("--eval_run_id", type=str, default=None,
                        help="Override eval run ID (used by pipeline for unified naming).")
    parser.add_argument("--rollout_type", type=str, default=None,
                        help="Rollout type label for episode metadata (e.g. random, full, curriculum).")
    parser.add_argument("--no_flash_attention", action="store_true", default=False,
                        help="Disable the XLA flash-attention monkey-patch on gemma "
                             "(falls back to vanilla attention or XSA-only if --use_xsa).")
    parser.add_argument("--use_xsa", action="store_true", default=False,
                        help="Use Exclusive Self Attention (XSA) on gemma. Composes "
                             "with flash attention unless --no_flash_attention is set.")
    args = parser.parse_args()

    # An explicit --garment_list (used by the pipeline's full/curriculum/replay
    # strategies) bypasses the seen/unseen subset selection entirely, so the
    # default "unseen" label would be misleading in the printout + metadata.
    if args.garment_list:
        args.garment_subset = "explicit"

    if args.no_value:
        args.integrated_value = False

    # Dataset + debug videos are rendered from the pkls, so skipping pkls skips both.
    if args.metrics_only:
        args.no_save_pkl = True
    if args.no_save_pkl:
        args.no_save_dataset = True
        args.no_save_debug_video = True

    # Build augmentation config dict
    if args.aug_config:
        import json as _json_aug_parse
        try:
            if os.path.isfile(args.aug_config):
                with open(args.aug_config) as f:
                    args._aug_config_dict = _json_aug_parse.load(f)
            else:
                args._aug_config_dict = _json_aug_parse.loads(args.aug_config)
        except Exception as e:
            logger.error(f"Failed to parse --aug_config: {e}")
            sys.exit(1)
    else:
        # Backward compat: build from legacy flags
        args._aug_config_dict = {}
        if args.garment_pattern_augmentation_p > 0:
            args._aug_config_dict["pattern_swap_p"] = args.garment_pattern_augmentation_p
        if args.garment_color_augmentation_p > 0:
            args._aug_config_dict["color_remap_p"] = args.garment_color_augmentation_p
    args._aug_config_dict["enabled"] = bool(args._aug_config_dict)
    args.garment_augmentation = args._aug_config_dict.get("enabled", False)

    # Parse per-garment-type inference config (JSON string).
    args._per_garment_type_inference_config = None
    if args.per_garment_type_inference_config:
        import json as _json_pgt
        try:
            args._per_garment_type_inference_config = _json_pgt.loads(args.per_garment_type_inference_config)
        except Exception as e:
            logger.error(f"Failed to parse --per_garment_type_inference_config: {e}")
            sys.exit(1)
        if not isinstance(args._per_garment_type_inference_config, dict):
            logger.error("--per_garment_type_inference_config must decode to a dict")
            sys.exit(1)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = find_latest_checkpoint()
        if args.checkpoint_dir is None:
            logger.error(f"No checkpoints found under {CHECKPOINT_BASE}")
            sys.exit(1)

    if not args.checkpoint_dir or not Path(args.checkpoint_dir).exists():
        logger.error(f"Checkpoint dir does not exist: {args.checkpoint_dir!r}")
        sys.exit(1)

    # Auto-load per-garment inference config from the checkpoint when not given
    # on the CLI (cfg_scale/noise_temperature/time_threshold_inpaint are applied
    # per request) — matches the RL pipeline instead of weak model defaults.
    if args._per_garment_type_inference_config is None:
        _inf_cfg_path = Path(args.checkpoint_dir) / "assets" / "inference_config.json"
        if _inf_cfg_path.exists():
            try:
                with open(_inf_cfg_path) as _f:
                    _saved_inf_cfg = json.load(_f)
                if isinstance(_saved_inf_cfg, dict) and isinstance(
                    _saved_inf_cfg.get("per_garment_type"), dict
                ):
                    args._per_garment_type_inference_config = _saved_inf_cfg["per_garment_type"]
                    logger.info(
                        "Loaded per-garment-type inference config from %s "
                        "(cfg_scale/noise_temperature/time_threshold_inpaint applied per request)",
                        _inf_cfg_path,
                    )
            except Exception as e:
                logger.warning("Could not load %s: %s", _inf_cfg_path, e)

    eval_run_id = args.eval_run_id or make_eval_run_id(
        args.checkpoint_dir,
        prefix="rl" if args.randomize_garments else "step",
    )
    run_info = {
        "Config": args.config_name,
        "Checkpoint": args.checkpoint_dir,
        "Resolution": f"{args.camera_width}x{args.camera_height}",
        "Workers": args.num_workers,
        "Success head": "integrated" if args.integrated_value else "disabled",
        "Garment subset": args.garment_subset or "unseen",
        "Randomized": args.randomize_garments,
    }
    video_path = (Path(args.video_dir) / eval_run_id).resolve()
    video_path.mkdir(parents=True, exist_ok=True)
    video_dir = str(video_path)  # absolute so worker (cwd=lehome-challenge) writes to repo, not submodule

    # Add a file handler so all orchestrator + worker logs are saved to the run folder.
    # This captures everything: session resets, proxy dtype warnings, episode results, etc.
    _orch_log_path = _log_path(video_path, logcfg.ORCHESTRATOR, "orchestrator.log")
    _file_handler = logging.FileHandler(_orch_log_path, mode="w")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_file_handler)
    logger.info(f"Orchestrator log: {_orch_log_path}")

    # Eval dataset (LeRobot format): main process only, one episode at a time
    dataset_writer = None
    keyframe_writer = None
    if not args.no_save_dataset:
        try:
            from lehome_solution.eval import EvalDatasetWriter, KeyframeDatasetWriter, EvalRunMetadata
            from lehome_solution.eval.real_format_writer import RealFormatDatasetWriter
            dataset_dir = video_path / "eval_dataset"
            use_value = args.integrated_value
            dataset_h = args.dataset_height or args.camera_height
            dataset_w = args.dataset_width or args.camera_width
            image_shape = (dataset_h, dataset_w, 3)
            if args.dataset_format == "real_bc":
                dataset_writer = RealFormatDatasetWriter(
                    dataset_dir, repo_id=eval_run_id,
                    units=args.dataset_units,
                    save_debug_video=not args.no_save_debug_video,
                    gae_lambda=args.gae_lambda,
                    success_alpha=args.success_alpha,
                    completion_alpha=args.completion_alpha,
                    value_tail_k=args.value_tail_k,
                ).create()
                logger.info(
                    f"Real-format dataset: {dataset_dir} "
                    f"(top {args.top_camera_width or args.camera_width}x"
                    f"{args.top_camera_height or args.camera_height}, "
                    f"wrist {args.camera_width}x{args.camera_height})"
                )
            else:
                dataset_writer = EvalDatasetWriter(
                    dataset_dir, repo_id=eval_run_id, use_value=use_value,
                    image_shape=image_shape,
                    save_debug_video=not args.no_save_debug_video,
                    gae_lambda=args.gae_lambda,
                    success_alpha=args.success_alpha,
                    completion_alpha=args.completion_alpha,
                    value_tail_k=args.value_tail_k,
                ).create()
                logger.info(f"Eval dataset: {dataset_dir} (images {dataset_w}x{dataset_h}, sim {args.camera_width}x{args.camera_height})"
                            + (" (with value)" if use_value else ""))
            # Keyframe dataset: ~20x fewer frames, images as PNG, for fast value re-prediction.
            # Real-format runs don't need it (no aux fields → no value re-prediction).
            if use_value and args.dataset_format != "real_bc":
                try:
                    keyframe_dir = video_path / "eval_dataset_keyframes"
                    keyframe_writer = KeyframeDatasetWriter(
                        keyframe_dir, repo_id=f"{eval_run_id}_keyframes",
                        image_shape=image_shape,
                    ).create()
                    logger.info(f"Keyframe dataset: {keyframe_dir}")
                except Exception as e:
                    logger.warning("Could not create keyframe dataset writer: %s", e, exc_info=True)
                    keyframe_writer = None
        except Exception as e:
            logger.warning("Could not create eval dataset writer: %s", e, exc_info=True)
            dataset_writer = None

    # Build task queue: one task per garment (batched across workers, all run in one Isaac Sim per worker)
    tasks = []
    task_num_episodes = args.num_episodes
    if args.task_overrides_file:
        # Exact (garment, seed) replay — same seed for all episodes
        import json as _json
        with open(args.task_overrides_file) as f:
            overrides = _json.load(f)
        for ov in overrides:
            gname = ov["garment"]
            gt = garment_name_to_type(gname)
            seed = ov["seed"]
            # Load NPZ restore data if present
            restore_data = None
            npz_path = ov.get("restore_npz")
            if npz_path and Path(npz_path).exists():
                try:
                    with np.load(npz_path, allow_pickle=True) as npz:
                        restore_data = {}
                        for key in ("garment_points", "garment_velocities",
                                    "garment_pos", "garment_ori",
                                    "left_joint_pos", "right_joint_pos"):
                            if key in npz:
                                restore_data[key] = npz[key].tolist()
                        # Load augmentation info (texture + tint) if saved
                        if "metadata" in npz:
                            try:
                                _meta = json.loads(str(npz["metadata"]))
                                if "augmentation" in _meta:
                                    restore_data["augmentation"] = _meta["augmentation"]
                            except Exception:
                                pass
                    logger.info(f"Loaded restore state from {npz_path} for {gname} "
                                f"(keys: {list(restore_data.keys())})")
                except Exception as e:
                    logger.warning(f"Failed to load NPZ {npz_path}: {e}")
                    restore_data = None
            elif npz_path:
                logger.warning(f"NPZ file not found: {npz_path} — episode will run without state restore!")
            # Pass hard mining metadata through restore_data for the proxy
            hm_mode = ov.get("hard_mining_mode")
            if restore_data is not None and hm_mode:
                restore_data["_hard_mining_mode"] = hm_mode
                restore_data["_restore_npz_path"] = npz_path
            # Attempt budget for early-stop-on-success retry logic (proxy-side).
            if restore_data is not None and ov.get("max_attempts"):
                restore_data["_max_attempts"] = int(ov["max_attempts"])
            # Success / semi-success replay: load saved actions from NPZ
            is_any_replay = ov.get("success_replay") or ov.get("semi_success_replay")
            if is_any_replay and restore_data is not None:
                restore_data["_restore_npz_path"] = npz_path
                try:
                    with np.load(npz_path, allow_pickle=True) as npz:
                        if "actions" in npz:
                            restore_data["_replay_actions"] = npz["actions"].tolist()
                            logger.info(f"Loaded {len(restore_data['_replay_actions'])} replay actions from {npz_path}")
                except Exception as e:
                    logger.warning(f"Failed to load replay actions from {npz_path}: {e}")
                    restore_data = None  # skip this task
                # Don't pass visual augmentation from NPZ — let thin client randomize new augs.
                # But preserve scale_factor: particles in NPZ are already scaled, so we need
                # to update init_scale on restore for correct reward threshold computation.
                if restore_data and "augmentation" in restore_data:
                    _saved_scale = restore_data["augmentation"].get("scale_factor")
                    del restore_data["augmentation"]
                    if _saved_scale is not None and _saved_scale != 1.0:
                        restore_data["_saved_scale_factor"] = _saved_scale
                # Tag semi_success_replay so proxy knows to hand off to policy
                if ov.get("semi_success_replay") and restore_data is not None:
                    restore_data["_semi_success_replay"] = True
            tasks.append((gt, gname, seed, restore_data))
        args.same_seed = True  # force same_seed for exact replay
        # Default to 100% garment augmentation for replay tasks; keys given
        # explicitly via --aug_config win over the forced defaults.
        has_replay = any(ov.get("success_replay") or ov.get("semi_success_replay") for ov in overrides)
        if has_replay:
            args._aug_config_dict.setdefault("pattern_swap_p", 1.0)
            args._aug_config_dict.setdefault("color_remap_p", 1.0)
            args._aug_config_dict["enabled"] = True
            args.garment_augmentation = True
        replay_label = ""
        if any(ov.get("semi_success_replay") for ov in overrides):
            replay_label = " [semi_success_replay mode]"
        elif any(ov.get("success_replay") for ov in overrides):
            replay_label = " [success_replay mode]"
        logger.info(f"Task overrides: {len(tasks)} garments × {task_num_episodes} ep (exact seed replay)"
                     + replay_label)
    elif args.garment_list:
        # Explicit garment list (from pipeline strategies)
        import random as _random
        rng = _random.Random(args.rl_seed)
        for gname in args.garment_list:
            gt = garment_name_to_type(gname)
            base_seed = rng.randint(1, 2**31 - 1)
            tasks.append((gt, gname, base_seed))
        logger.info(f"Garment list: {len(tasks)} garments (explicit list)")
    elif args.randomize_garments:
        import random as _random
        rng = _random.Random(args.rl_seed)
        # Stratified sampling: equal episodes per garment type, then random within type
        by_type: dict[str, list[tuple[str, str]]] = {}
        for gt in args.garment_types:
            for gname in get_garments(REPO_ROOT, gt, subset=args.garment_subset):
                by_type.setdefault(gt, []).append((gt, gname))
        n = args.num_garments or sum(len(v) for v in by_type.values())
        n_types = len(by_type)
        selected = []
        if n_types > 0:
            per_type = n // n_types
            remainder = n % n_types
            for i, (gt, garments) in enumerate(sorted(by_type.items())):
                type_n = per_type + (1 if i < remainder else 0)
                selected.extend(rng.choices(garments, k=type_n))
        rng.shuffle(selected)
        for gt, gname in selected:
            base_seed = rng.randint(1, 2**31 - 1)
            tasks.append((gt, gname, base_seed))
        type_counts = {}
        for gt, _ in selected:
            type_counts[gt] = type_counts.get(gt, 0) + 1
        logger.info(f"Randomized (stratified): {n} tasks from {sum(len(v) for v in by_type.values())} garments "
                    f"(per-type: {type_counts}, subset={args.garment_subset}, rl_seed={args.rl_seed})")
    else:
        seed = 42
        for gt in args.garment_types:
            for gname in get_garments(REPO_ROOT, gt, subset=args.garment_subset):
                tasks.append((gt, gname, seed))
                seed += args.num_episodes  # reserve num_episodes seeds for this garment

    logger.info(f"Eval run: {eval_run_id}")
    logger.info(f"Checkpoint: {args.checkpoint_dir}")
    logger.info(f"Videos: {video_dir}")
    logger.info(f"Tasks: {len(tasks)} garments × {task_num_episodes} ep = {len(tasks) * task_num_episodes} total, {args.num_workers} workers, port {args.port}")
    if args.integrated_value:
        logger.info("Success head: integrated (from policy model)")
    logger.info("=" * 60)

    # Start single shared server
    server_proc = start_server(
        args.checkpoint_dir, args.config_name, args.port, video_dir,
        integrated_value=args.integrated_value,
        use_flash_attention=not args.no_flash_attention,
        use_xsa=args.use_xsa,
        num_rollout_candidates=args.num_rollout_candidates,
    )
    server_timeout = 120
    if not wait_for_server(args.port, timeout=server_timeout, proc=server_proc):
        logger.error("Server failed to start — check %s/server.log", video_dir)
        try:
            os.killpg(os.getpgid(server_proc.pid), 9)
        except Exception:
            pass
        sys.exit(1)
    logger.info("Server ready")

    # W&B
    wandb_run, wandb_step = None, None
    if not args.no_wandb:
        wandb_run, wandb_step = init_wandb(
            args.checkpoint_dir, args.garment_types, args.wandb_project,
            eval_run_id, args.garment_subset)

    # Import worker
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_worker import run_worker_session

    all_episodes = []
    completed = 0
    failed = 0
    server_url = f"ws://localhost:{args.port}"

    # Build a shared task queue: each item is one episode (garment + seed).
    # All workers pull from the same queue, so when one worker finishes early
    # it immediately picks up the next task instead of idling.
    import queue as _q
    episode_tasks = []
    for task in tasks:
        if len(task) == 4:
            gt, gname, base_seed, restore_data = task
        else:
            gt, gname, base_seed = task
            restore_data = None
        for ep in range(task_num_episodes):
            seed = base_seed if args.same_seed else base_seed + ep
            episode_tasks.append({
                "garment_name": gname,
                "garment_type": gt,
                "seed": seed,
                "restore_data": restore_data,
            })
    # Sort by garment type so workers process same-type garments consecutively.
    # Skip sorting for success replay — preserve the strategy's task order
    # (same NPZ replayed N times consecutively, NPZ order randomized).
    has_replay = any(t.get("restore_data") and "_replay_actions" in (t["restore_data"] or {})
                     for t in episode_tasks)
    if not has_replay:
        episode_tasks.sort(key=lambda t: t["garment_type"])

    num_workers = min(args.num_workers, len(episode_tasks))

    # Assign globally unique ep_idx to each task before enqueueing.
    for i, t in enumerate(episode_tasks):
        t["ep_idx"] = i

    # Single shared queue — workers pull tasks dynamically for load balancing.
    # Garment-type sorting still helps: workers starting concurrently will
    # naturally pick same-type tasks from the front.  Under load imbalance,
    # a worker may switch types more often (~5s overhead), but this is far
    # cheaper than having idle workers while others still run.
    shared_task_queue: _q.Queue = _q.Queue()
    for t in episode_tasks:
        shared_task_queue.put(t)

    start_time = time.time()
    # Upper bound: a task with max_attempts > 1 may spawn up to N-1 retries on
    # failure (proxy-side), so count each task as max_attempts attempts. The
    # proxy emits phantom "discarded" events for unused attempts on early
    # success so the processed counter still converges to total_expected.
    total_expected = sum(
        int((t.get("restore_data") or {}).get("_max_attempts", 1))
        for t in episode_tasks
    )

    # Shared episode-done queue: proxies push episode metadata here as soon as
    # each PKL is written.  The dataset writer thread picks them up immediately
    # — no waiting for the entire worker session to finish.
    episode_done_queue: queue.Queue = queue.Queue()

    # Real-format mode: keep per-camera native shapes (top 1280x720, wrist 640x480)
    # by disabling storage downscale. Sim mode keeps the existing single-shape
    # downscale path so old behaviour is unchanged.
    if args.dataset_format == "real_bc":
        _storage_w, _storage_h = None, None
    else:
        _storage_w = args.dataset_width or args.camera_width
        _storage_h = args.dataset_height or args.camera_height

    def _worker_fn(worker_id):
        return run_worker_session(
            task_queue=shared_task_queue,
            port=args.port,
            max_steps=args.max_steps,
            worker_id=worker_id,
            server_url=server_url,
            video_dir=video_dir,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            top_camera_width=args.top_camera_width,
            top_camera_height=args.top_camera_height,
            enable_depth=args.depth,
            prior_file=getattr(args, "prior_file", None),
            noise_temperature=getattr(args, "noise_temperature", 1.0),
            explore_noise_prob=getattr(args, "explore_noise_prob", 0.0),
            explore_noise_scale=getattr(args, "explore_noise_scale", 0.0),
            aug_config=getattr(args, "_aug_config_dict", None),
            episode_done_queue=episode_done_queue,
            success_rates_file=getattr(args, "success_rates_file", None),
            default_rollout_type=getattr(args, "rollout_type", None),
            storage_width=_storage_w,
            storage_height=_storage_h,
            per_garment_type_inference_config=getattr(args, "_per_garment_type_inference_config", None),
            save_pkl=not args.no_save_pkl,
        )

    # Background dataset writer thread — processes episodes as they stream in
    # from proxies via episode_done_queue.  Sentinel None signals exit.
    _dataset_stop = threading.Event()

    discarded_count = 0

    def _dataset_writer_loop():
        while not _dataset_stop.is_set():
            try:
                ep = episode_done_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if ep is None:
                break
            nonlocal completed, discarded_count

            # Hard mining: discarded episodes (failures/duplicates) are counted
            # for progress but not written to the dataset.
            if ep.get("discarded"):
                discarded_count += 1
                elapsed = time.time() - start_time
                processed = completed + discarded_count
                logger.info(
                    f"Progress: {processed}/{total_expected} eps "
                    f"({completed} saved, {discarded_count} discarded, {elapsed/60:.1f}min)"
                )
                continue

            all_episodes.append(ep)
            completed += 1
            # Write keyframes first (reads temp pkl but never deletes it)
            if keyframe_writer is not None:
                try:
                    keyframe_writer.add_episode_from_temp(video_dir, ep)
                except Exception as kf_e:
                    logger.warning("Keyframe add_episode failed: %s", kf_e, exc_info=True)
            # Write full dataset (deletes temp pkl when done)
            if dataset_writer is not None:
                try:
                    if dataset_writer.add_episode_from_temp(video_dir, ep):
                        logger.info("Dataset: added %s seed=%s ep=%s", ep.get("garment"), ep.get("seed"), ep.get("ep_idx"))
                    else:
                        logger.warning("Dataset: temp file missing for %s seed=%s ep=%s", ep.get("garment"), ep.get("seed"), ep.get("ep_idx"))
                except Exception as ds_e:
                    logger.warning("Dataset add_episode failed: %s", ds_e, exc_info=True)

            elapsed = time.time() - start_time
            processed = completed + discarded_count
            logger.info(
                f"Progress: {processed}/{total_expected} eps "
                f"({completed} saved, {discarded_count} discarded, {elapsed/60:.1f}min)"
            )
            with open(video_path / "eval_results.json", "w") as jf:
                json.dump({"episodes": all_episodes}, jf, indent=2)
            _write_eval_summary(
                video_path, all_episodes,
                start_time=start_time, total_expected=total_expected,
                total_processed=processed,
                run_info=run_info,
            )

    _dataset_thread = threading.Thread(target=_dataset_writer_loop, daemon=False, name="dataset-writer")
    _dataset_thread.start()

    MAX_ORCHESTRATOR_RESPAWNS = 3  # max times to respawn a fully-dead worker

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            # Submit N workers, all pulling from a single shared task queue.
            # Proxies push episode results to episode_done_queue as they complete.
            active_futures = {pool.submit(_worker_fn, wid): wid for wid in range(num_workers)}
            respawn_counts: dict[int, int] = {wid: 0 for wid in range(num_workers)}
            logger.info(f"Started {num_workers} worker sessions, {total_expected} episodes queued")

            while active_futures:
                done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
                for f in done:
                    wid = active_futures.pop(f)
                    try:
                        result = f.result()
                    except Exception as e:
                        logger.error(f"Worker {wid} crashed: {e}")
                        result = {"hung": True}

                    remaining = shared_task_queue.qsize()
                    if remaining > 0 and respawn_counts[wid] < MAX_ORCHESTRATOR_RESPAWNS:
                        respawn_counts[wid] += 1
                        logger.warning(
                            f"Worker {wid} died with ~{remaining} tasks remaining in shared queue — "
                            f"respawning (attempt {respawn_counts[wid]}/{MAX_ORCHESTRATOR_RESPAWNS})"
                        )
                        new_future = pool.submit(_worker_fn, wid)
                        active_futures[new_future] = wid
                    elif remaining > 0 and not active_futures:
                        # No other workers alive to pick up remaining tasks
                        logger.error(
                            f"Worker {wid} died, {remaining} tasks in shared queue, "
                            f"no workers left — abandoning tasks"
                        )
                        failed += remaining
                    elif result.get("hung"):
                        logger.warning(f"Worker {wid} hung (no remaining tasks)")

        # All workers done — signal dataset writer to drain and exit
        episode_done_queue.put(None)
        _dataset_thread.join()

        # Summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"EVALUATION SUMMARY ({elapsed/60:.1f} min)")
        logger.info("=" * 60)

        for split_name in ["Seen", "Unseen"]:
            split_eps = [e for e in all_episodes if e.get("split") == split_name]
            if not split_eps:
                continue
            s_total = len(split_eps)
            s_success = sum(1 for e in split_eps if e.get("success"))
            s_return = sum(e["return"] for e in split_eps) / s_total
            s_dr = [e["dense_return"] for e in split_eps if "dense_return" in e]
            s_dr_str = f", avg_reward={sum(s_dr)/len(s_dr):.3f}" if s_dr else ""
            logger.info(f"  [{split_name}] {s_success}/{s_total} ({s_success/s_total:.1%}), avg_return={s_return:.1f}{s_dr_str}")
            by_garment: dict = {}
            for e in split_eps:
                by_garment.setdefault(e.get("garment", ""), []).append(e)
            for gname, eps in sorted(by_garment.items()):
                gs = sum(1 for e in eps if e.get("success"))
                gr = sum(e["return"] for e in eps) / len(eps)
                g_dr = [e["dense_return"] for e in eps if "dense_return" in e]
                g_dr_str = f", avg_reward={sum(g_dr)/len(g_dr):.3f}" if g_dr else ""
                logger.info(f"    {gname}: {gs}/{len(eps)} ({gs/len(eps):.1%}), avg_return={gr:.1f}{g_dr_str}")

        total = len(all_episodes)
        successes = sum(1 for e in all_episodes if e.get("success"))
        dr_all = [e["dense_return"] for e in all_episodes if "dense_return" in e]
        dr_str = f", avg_reward={sum(dr_all)/len(dr_all):.3f}" if dr_all else ""
        logger.info(f"  OVERALL: {successes}/{total} ({successes/total:.1%}){dr_str}" if total else "  OVERALL: 0/0")
        logger.info(f"  Failed/skipped: {failed}")
        if discarded_count > 0:
            logger.info(f"  Discarded (hard mining): {discarded_count}")
        logger.info("=" * 60)

        _write_eval_summary(
            video_path, all_episodes,
            start_time=start_time, total_expected=total_expected,
            total_processed=completed + discarded_count,
            run_info=run_info,
            eval_dataset_root=video_path / "eval_dataset" if dataset_writer is not None else None,
        )
        logger.info(f"Eval summary written to {video_path / 'eval_summary.json'} and {video_path / 'eval_summary.txt'}")

        # Finalize eval dataset and write run metadata. Post-rollout integrity
        # check inside finalize() raises RuntimeError if any per-camera mp4
        # frame count disagrees with the parquet. On failure we QUARANTINE the
        # dataset root to `*.REJECTED_CORRUPT` so it can never be uploaded to
        # HF or consumed by the trainer — silently leaving a corrupt dataset
        # in place is how bad data sneaks into training.
        if dataset_writer is not None:
            try:
                from lehome_solution.eval import EvalRunMetadata
                dataset_writer.finalize()
            except Exception as e:
                logger.error("REJECTING rollout — dataset finalize FAILED: %s", e)
                rejected_root = Path(str(dataset_writer.root) + ".REJECTED_CORRUPT")
                try:
                    if rejected_root.exists():
                        shutil.rmtree(rejected_root)
                    os.rename(str(dataset_writer.root), str(rejected_root))
                    logger.error("Quarantined corrupt dataset to %s", rejected_root)
                except Exception as mv_err:
                    logger.error("Failed to quarantine rejected dataset: %s", mv_err)
            else:
                try:
                    run_meta = EvalRunMetadata.from_args(
                        eval_run_id, video_dir, str(dataset_writer.root), args
                    )
                    dataset_writer.write_run_metadata(run_meta)
                except Exception as e:
                    logger.warning("Eval run metadata write failed: %s", e)
        if keyframe_writer is not None:
            try:
                keyframe_writer.finalize()
            except Exception as e:
                logger.warning("Keyframe dataset finalize: %s", e)

        finish_wandb(wandb_run, all_episodes, video_dir, wandb_step)

    finally:
        # Ensure dataset thread exits even if an exception occurred mid-run
        if _dataset_thread.is_alive():
            episode_done_queue.put(None)
            _dataset_thread.join(timeout=60)
        logger.info("Stopping server...")
        try:
            os.killpg(os.getpgid(server_proc.pid), 9)
        except Exception:
            pass
        # Close server log file handle
        log_fh = getattr(server_proc, '_log_fh', None)
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass
        logger.info("Done.")


if __name__ == "__main__":
    main()
