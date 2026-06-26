"""Compute normalization statistics for LeHome Challenge dataset.

Computes mean, std, min, max from dataset in parallel.
Supports per-timestamp normalization (with smooth sqrt fitting)
and action correlation matrices.
"""

import dataclasses
import json
import numpy as np
import tyro
from pathlib import Path

# Import LeHome-specific modules
from lehome_solution.shared import normalize

# Shared data loading
from data_utils import get_delta_transform_from_config, load_action_chunks


def compute_correlation(
    all_chunks: np.ndarray,
    action_horizon: int,
    max_correlation_samples: int = 2_000_000,
):
    """Compute full correlation matrix and averaged spatial/temporal correlations.

    Uses the REAL action dimensions only (no padding). The correlation matrix
    has shape [H*D_real, H*D_real] where D_real is the actual action dim (e.g. 12).
    """
    action_dim = all_chunks.shape[2]  # real action dim (e.g. 12)

    print("Computing full action correlation matrix...")
    print(f"Total action chunks: {all_chunks.shape[0]} of shape ({action_horizon}, {action_dim})")
    print(f"Correlation matrix will be {action_horizon * action_dim}x{action_horizon * action_dim}")

    # Flatten chunks to (num_samples, action_horizon * action_dim) — NO padding
    flattened = all_chunks.reshape(-1, action_horizon * action_dim)
    total_samples = flattened.shape[0]

    # Subsample if too many
    if total_samples > max_correlation_samples:
        print(f"Subsampling {max_correlation_samples} from {total_samples} chunks")
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(total_samples, size=max_correlation_samples, replace=False)
        flattened = flattened[sample_idx]
        print(f"Using {flattened.shape[0]} sampled chunks")
    else:
        print(f"Using all {total_samples} chunks")

    # Normalize each dimension to zero mean unit variance
    chunk_mean = np.mean(flattened, axis=0)
    chunk_std = np.std(flattened, axis=0)

    constant_dims = chunk_std < 1e-6
    print(f"Found {np.sum(constant_dims)} constant dimensions")

    normalized = flattened.copy()
    normalized[:, ~constant_dims] = (
        (flattened[:, ~constant_dims] - chunk_mean[~constant_dims]) / chunk_std[~constant_dims]
    )

    # Compute covariance (= correlation for normalized data)
    cov_matrix = np.cov(normalized, rowvar=False)

    # Normalize to proper correlation matrix
    diag_vals = np.diag(cov_matrix).copy()
    constant_mask = diag_vals < 1e-10
    diag_safe = diag_vals.copy()
    diag_safe[constant_mask] = 1.0
    normalizer = np.sqrt(diag_safe[:, None] @ diag_safe[None, :])
    cov_matrix = cov_matrix / normalizer
    cov_matrix[constant_mask, :] = 0.0
    cov_matrix[:, constant_mask] = 0.0
    np.fill_diagonal(cov_matrix, 1.0)

    # Regularize
    epsilon = 1e-6
    cov_reg = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])
    eigenvalues = np.linalg.eigvalsh(cov_reg)
    min_eig = np.min(eigenvalues)
    print(f"Min eigenvalue after regularization: {min_eig:.6e}")

    if min_eig <= 0:
        print("WARNING: Not positive definite, adding stronger regularization...")
        epsilon = max(1e-5, -min_eig + 1e-5)
        cov_reg = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

    # Cholesky
    chol_lower = np.linalg.cholesky(cov_reg)
    print(f"Cholesky decomposition successful! Shape: {chol_lower.shape}")

    # Averaged spatial correlation (action_dim x action_dim)
    spatial_corrs = []
    for t in range(action_horizon):
        s, e = t * action_dim, (t + 1) * action_dim
        ts_data = normalized[:, s:e]
        corr_t = np.cov(ts_data, rowvar=False)
        diag_t = np.diag(corr_t)
        diag_t_safe = np.where(diag_t < 1e-10, 1.0, diag_t)
        corr_t = corr_t / np.sqrt(diag_t_safe[:, None] @ diag_t_safe[None, :])
        spatial_corrs.append(corr_t)
    avg_spatial = np.mean(spatial_corrs, axis=0)
    np.fill_diagonal(avg_spatial, 1.0)
    print(f"Spatial correlation shape: {avg_spatial.shape}")

    # Averaged temporal correlation (action_horizon x action_horizon)
    temporal_corrs = []
    for d in range(action_dim):
        dim_indices = [t * action_dim + d for t in range(action_horizon)]
        if np.all(chunk_std[dim_indices] < 1e-6):
            continue
        dim_data = normalized[:, dim_indices]
        corr_d = np.cov(dim_data, rowvar=False)
        diag_d = np.diag(corr_d)
        corr_d = corr_d / np.sqrt(diag_d[:, None] @ diag_d[None, :])
        temporal_corrs.append(corr_d)

    if temporal_corrs:
        avg_temporal = np.mean(temporal_corrs, axis=0)
        np.fill_diagonal(avg_temporal, 1.0)
    else:
        avg_temporal = np.eye(action_horizon)
    print(f"Temporal correlation shape: {avg_temporal.shape}")

    return {
        "action_correlation_cholesky": chol_lower,
    }


def main(
    config_name: str,
    max_files: int | None = None,
    num_workers: int | None = None,
    per_timestamp: bool = False,
    correlation: bool = True,
    max_correlation_samples: int = 2_000_000,
    sample_fraction: float = 0.1,
    compute_quantiles_sample_size: int = 1_000_000,
    extra_roots: list[str] | None = None,
    all_sources: bool = False,
    pipeline_state: str | None = None,
    exp_name: str | None = None,
):
    """Compute normalization statistics for LeHome config.

    Args:
        config_name: Training config — registry name (e.g. pi_modified_bc_rl)
                     or a YAML recipe path (e.g. configs/train_real_bc.yaml)
        max_files: Maximum number of parquet files to process per dataset (None = all)
        num_workers: Number of parallel workers (None = auto)
        per_timestamp: Whether to compute per-timestamp statistics
        correlation: Whether to compute action correlation matrix
        max_correlation_samples: Max samples for correlation computation
        sample_fraction: Fraction of action chunks to sample per file
                        (default 0.1 = 10%). Mean/std/min/max always use ALL data.
        compute_quantiles_sample_size: Max samples for percentile computation
        extra_roots: Additional dataset roots to include in statistics computation
        all_sources: If True, automatically include all data_sources from the config
        pipeline_state: Path to pipeline_state.json — includes BC dataset from config
                       plus all RL datasets from the pipeline state
        exp_name: Override exp_name for pipeline checkpoint path (e.g. 'sim_rl').
                  Ensures assets are saved where the pipeline expects them.
    """
    from lehome_solution.training.yaml_train_config import resolve_train_config

    config = resolve_train_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Collect all dataset roots to process. Units flow through natively
    # (sim sources: radians; real sources: degrees) — no unit conversion
    # anywhere, mirroring the training-time loader.
    # (root, speed_factor, bucket)
    dataset_roots: list[tuple[str, float, str]] = []
    seen: set[str] = set()
    _spec_speed: dict[str, float] = {}  # root -> speed_factor from config
    _spec_bucket: dict[str, str] = {}   # root -> bucket label from config

    def _add(root: str) -> None:
        if root in seen:
            return
        speed = _spec_speed.get(root, 1.0)
        bucket = _spec_bucket.get(root, "")
        dataset_roots.append((root, speed, bucket))
        seen.add(root)

    if data_config.lehome_dataset_root:
        _add(data_config.lehome_dataset_root)

    if all_sources and hasattr(config.data, 'data_sources') and config.data.data_sources:
        for spec in config.data.data_sources:
            _spec_speed[spec.root] = getattr(spec, "speed_factor", 1.0)
            _spec_bucket[spec.root] = getattr(spec, "bucket", "")
            _add(spec.root)

    if pipeline_state:
        with open(pipeline_state) as f:
            state = json.load(f)
        # Add BC dataset from config data_sources
        if hasattr(config.data, 'data_sources') and config.data.data_sources:
            for spec in config.data.data_sources:
                _spec_speed[spec.root] = getattr(spec, "speed_factor", 1.0)
                _spec_bucket[spec.root] = getattr(spec, "bucket", "")
                _add(spec.root)
        # Add all RL datasets from pipeline state.
        for ds in state.get("rl_datasets", []):
            _add(ds["root"])

    if extra_roots:
        for root in extra_roots:
            _add(root)

    if not dataset_roots:
        raise ValueError("No dataset roots found. Set lehome_dataset_root in config or pass --extra_roots")

    delta_transform = get_delta_transform_from_config(config_name)
    print(f"Delta mask: {delta_transform.mask}")

    compute_per_timestamp = per_timestamp or data_config.use_per_timestamp_norm
    action_horizon = config.model.action_horizon

    if compute_per_timestamp:
        print(f"Computing per-timestamp statistics with action_horizon={action_horizon}")
    if correlation:
        print(f"Computing correlation matrix with action_horizon={action_horizon}")

    # Load data from all dataset roots, grouped by bucket for weighted resampling.
    bucket_data: dict[str, dict] = {}  # bucket -> {"states": [], "actions": [], "chunks": []}

    print(f"\nLoading data from {len(dataset_roots)} dataset(s):")
    for i, (root, speed, bucket) in enumerate(dataset_roots):
        speed_tag = f" speed_factor={speed}" if abs(speed - 1.0) > 1e-6 else ""
        bucket_tag = f" bucket={bucket}" if bucket else ""
        print(f"  [{i+1}/{len(dataset_roots)}]{speed_tag}{bucket_tag} {root}")
        try:
            data = load_action_chunks(
                data_root=root,
                delta_mask=delta_transform.mask,
                action_horizon=action_horizon,
                sample_fraction=sample_fraction if (compute_per_timestamp or correlation) else 1.0,
                max_files=max_files,
                num_workers=num_workers,
                speed_factor=speed,
            )
            bkey = bucket or "_default"
            if bkey not in bucket_data:
                bucket_data[bkey] = {"states": [], "actions": [], "chunks": []}
            bucket_data[bkey]["states"].append(data["states"])
            bucket_data[bkey]["actions"].append(data["actions"])
            if data["chunks"] is not None:
                bucket_data[bkey]["chunks"].append(data["chunks"])
        except FileNotFoundError as e:
            print(f"    WARNING: Skipping — {e}")

    # Bucket-weighted resampling: match training-time bucket_shares so each
    # bucket contributes proportionally to the stats.
    bucket_shares = None
    if hasattr(config.data, 'bucket_shares') and config.data.bucket_shares:
        bucket_shares = config.data.bucket_shares

    all_states_list = []
    all_actions_list = []
    all_chunks_list = []

    if bucket_shares and len(bucket_data) > 1:
        # Consolidate per-bucket arrays
        bucket_arrays: dict[str, dict] = {}
        for bkey, bdata in bucket_data.items():
            s = np.concatenate(bdata["states"], axis=0) if bdata["states"] else np.empty((0,))
            a = np.concatenate(bdata["actions"], axis=0) if bdata["actions"] else np.empty((0,))
            c = np.concatenate(bdata["chunks"], axis=0) if bdata["chunks"] else None
            bucket_arrays[bkey] = {"states": s, "actions": a, "chunks": c}

        # Compute target sample counts per bucket.  Anchor = bucket whose
        # (target_share / raw_count) is smallest — it keeps all samples,
        # others are downsampled.
        raw_counts = {b: len(d["states"]) for b, d in bucket_arrays.items()}
        total_raw = sum(raw_counts.values())

        # Only resample buckets that appear in bucket_shares.
        share_sum = sum(bucket_shares.get(b, 0) for b in bucket_arrays)
        ratios = {}
        for b in bucket_arrays:
            share = bucket_shares.get(b, 0)
            if share <= 0 or raw_counts[b] == 0:
                ratios[b] = 1.0
                continue
            ratios[b] = (share / share_sum) / (raw_counts[b] / total_raw)

        max_ratio = max(ratios.values())
        target_fracs = {b: ratios[b] / max_ratio for b in bucket_arrays}

        print(f"\nBucket-weighted resampling (bucket_shares={bucket_shares}):")
        for b in sorted(bucket_arrays):
            raw = raw_counts[b]
            target_n = int(raw * min(target_fracs[b], 1.0))
            share_pct = bucket_shares.get(b, 0) * 100
            raw_pct = raw / total_raw * 100
            print(f"  {b:15s}: raw={raw:8d} ({raw_pct:5.1f}%) target_share={share_pct:5.1f}% "
                  f"-> keep {target_n:8d} (frac={min(target_fracs[b], 1.0):.3f})")

        rng = np.random.RandomState(42)
        for b, arrs in bucket_arrays.items():
            frac = min(target_fracs[b], 1.0)
            n_states = len(arrs["states"])
            n_chunks = len(arrs["chunks"]) if arrs["chunks"] is not None else 0

            if frac < 1.0 - 1e-9 and n_states > 0:
                keep_s = max(1, int(n_states * frac))
                idx_s = rng.choice(n_states, size=keep_s, replace=False)
                all_states_list.append(arrs["states"][idx_s])
                all_actions_list.append(arrs["actions"][idx_s])
            elif n_states > 0:
                all_states_list.append(arrs["states"])
                all_actions_list.append(arrs["actions"])

            if n_chunks > 0:
                if frac < 1.0 - 1e-9:
                    keep_c = max(1, int(n_chunks * frac))
                    idx_c = rng.choice(n_chunks, size=keep_c, replace=False)
                    all_chunks_list.append(arrs["chunks"][idx_c])
                else:
                    all_chunks_list.append(arrs["chunks"])
    else:
        for bdata in bucket_data.values():
            all_states_list.extend(bdata["states"])
            all_actions_list.extend(bdata["actions"])
            all_chunks_list.extend(bdata["chunks"])

    states = np.concatenate(all_states_list, axis=0) if all_states_list else np.empty((0,))
    actions = np.concatenate(all_actions_list, axis=0) if all_actions_list else np.empty((0,))
    chunks = np.concatenate(all_chunks_list, axis=0) if all_chunks_list else None

    print(f"\nCombined: {len(states)} timesteps, {len(chunks) if chunks is not None else 0} action chunks from {len(dataset_roots)} datasets")

    # ---- Global statistics ----
    final_stats = {}
    for key, arr in [("state", states), ("actions", actions)]:
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        final_stats[key] = {"mean": mean, "std": std}

    # Compute actual percentiles for actions from chunks
    if chunks is not None:
        print("Computing actual percentiles for actions...")
        flat_actions = chunks.reshape(-1, chunks.shape[-1])
        if len(flat_actions) > compute_quantiles_sample_size:
            print(f"  Sampling {compute_quantiles_sample_size:,} from {len(flat_actions):,}")
            rng = np.random.RandomState(42)
            idx = rng.choice(len(flat_actions), size=compute_quantiles_sample_size, replace=False)
            flat_actions = flat_actions[idx]
        final_stats["actions"]["q01"] = np.percentile(flat_actions, 1, axis=0)
        final_stats["actions"]["q99"] = np.percentile(flat_actions, 99, axis=0)
        print(f"  Computed q01/q99 from {len(flat_actions):,} samples")
    else:
        final_stats["actions"]["q01"] = np.min(actions, axis=0)
        final_stats["actions"]["q99"] = np.max(actions, axis=0)

    final_stats["state"]["q01"] = np.min(states, axis=0)
    final_stats["state"]["q99"] = np.max(states, axis=0)

    # ---- Per-timestamp statistics (with smooth fitting) ----
    per_timestamp_stats = None
    if compute_per_timestamp and chunks is not None:
        H, D = chunks.shape[1], chunks.shape[2]
        per_timestamp_mean = np.zeros((H, D))
        per_timestamp_std = np.zeros((H, D))
        per_timestamp_q01 = np.zeros((H, D))
        per_timestamp_q99 = np.zeros((H, D))

        for t in range(H):
            td = chunks[:, t, :]
            per_timestamp_mean[t] = np.mean(td, axis=0)
            per_timestamp_std[t] = np.std(td, axis=0)
            per_timestamp_q01[t] = np.percentile(td, 1, axis=0)
            per_timestamp_q99[t] = np.percentile(td, 99, axis=0)

        # Fit smooth parametric functions:
        #   mean: a + b * t              (linear)
        #   std:  a + s * sqrt(t + e)    (sqrt with learnable e > 0)
        print("\nFitting smooth normalization curves...")
        mean_fit = normalize.fit_linear_norm(per_timestamp_mean)
        std_fit = normalize.fit_sqrt_norm(per_timestamp_std)

        print(f"\nSmooth fitting results:")
        print(f"  Mean fit (linear: a + b*t)")
        print(f"    R² per dim:        {mean_fit.r_squared[:D]}")
        print(f"    avg R²:            {np.mean(mean_fit.r_squared[:D]):.6f}")
        print(f"    max abs error:     {mean_fit.max_abs_error[:D]}")
        print(f"  Std fit  (sqrt: a + s*sqrt(t+e))")
        print(f"    R² per dim:        {std_fit.r_squared[:D]}")
        print(f"    avg R²:            {np.mean(std_fit.r_squared[:D]):.6f}")
        print(f"    max abs error:     {std_fit.max_abs_error[:D]}")
        print(f"    learned epsilons:  {std_fit.epsilons[:D]}")

        fitted_mean = mean_fit.fitted_values
        fitted_std = np.maximum(std_fit.fitted_values, 1e-6)

        # === Safety checks ===
        has_errors = False

        # ERROR: fitted std too low (< 0.1x raw) — numerical instability
        std_ratio_to_raw = fitted_std / np.maximum(per_timestamp_std, 1e-12)
        too_low = std_ratio_to_raw < 0.1
        if np.any(too_low):
            has_errors = True
            locs = np.argwhere(too_low)
            print(f"\n  ERROR: Fitted std < 0.1x raw at {len(locs)} (t, dim) locations!")
            for t_idx, d_idx in locs[:20]:
                print(f"    t={t_idx}, dim={d_idx}: fitted={fitted_std[t_idx, d_idx]:.6f}, "
                      f"raw={per_timestamp_std[t_idx, d_idx]:.6f}")
            if len(locs) > 20:
                print(f"    ... and {len(locs) - 20} more")

        # WARNING: fitted mean >2x off from raw (only for non-negligible means)
        mean_ratio = np.where(
            np.abs(per_timestamp_mean) > 1e-6, np.abs(fitted_mean / per_timestamp_mean), 1.0
        )
        mean_off = ((mean_ratio > 2.0) | (mean_ratio < 0.5)) & (np.abs(per_timestamp_mean) > 0.01)
        if np.any(mean_off):
            locs = np.argwhere(mean_off)
            print(f"\n  WARNING: Fitted mean >2x off at {len(locs)} locations")
            for t_idx, d_idx in locs[:10]:
                print(f"    t={t_idx}, dim={d_idx}: fitted={fitted_mean[t_idx, d_idx]:.6f}, "
                      f"raw={per_timestamp_mean[t_idx, d_idx]:.6f}")

        # WARNING: fitted std >2x off from raw
        std_off_ratio = np.where(per_timestamp_std > 1e-6, fitted_std / per_timestamp_std, 1.0)
        std_off = (std_off_ratio > 2.0) | (std_off_ratio < 0.5)
        if np.any(std_off):
            locs = np.argwhere(std_off)
            print(f"\n  WARNING: Fitted std >2x off at {len(locs)} locations")
            for t_idx, d_idx in locs[:10]:
                print(f"    t={t_idx}, dim={d_idx}: fitted={fitted_std[t_idx, d_idx]:.6f}, "
                      f"raw={per_timestamp_std[t_idx, d_idx]:.6f}")

        if has_errors:
            raise RuntimeError("Fitted normalization has critical issues (see ERROR above).")

        if not np.any(mean_off) and not np.any(std_off):
            print("\n  All safety checks passed.")

        # Scale quantiles by fitted/raw std ratio
        raw_std_safe = np.maximum(per_timestamp_std, 1e-8)
        ratio = fitted_std / raw_std_safe
        fitted_q01 = fitted_mean + (per_timestamp_q01 - per_timestamp_mean) * ratio
        fitted_q99 = fitted_mean + (per_timestamp_q99 - per_timestamp_mean) * ratio

        per_timestamp_stats = {
            "per_timestamp_mean": fitted_mean,
            "per_timestamp_std": fitted_std,
            "per_timestamp_q01": fitted_q01,
            "per_timestamp_q99": fitted_q99,
        }

    # ---- Correlation matrix ----
    correlation_stats = None
    if correlation and chunks is not None:
        correlation_stats = compute_correlation(
            chunks, action_horizon, max_correlation_samples
        )

    # ---- Build NormStats and save ----
    # Stats are stored at native action_dim (12D) — no padding needed.
    norm_stats_dict = {}
    for key in ["state", "actions"]:
        stats_kwargs = {}
        for stat_type in ["mean", "std", "q01", "q99"]:
            stats_kwargs[stat_type] = final_stats[key][stat_type]

        if per_timestamp_stats is not None and key == "actions":
            print("Adding per-timestamp statistics...")
            for stat_type in ["per_timestamp_mean", "per_timestamp_std",
                              "per_timestamp_q01", "per_timestamp_q99"]:
                stats_kwargs[stat_type] = per_timestamp_stats[stat_type]

        if correlation_stats is not None and key == "actions":
            stats_kwargs["action_correlation_cholesky"] = correlation_stats["action_correlation_cholesky"]

        norm_stats_dict[key] = normalize.NormStats(**stats_kwargs)

    asset_id = config.data.assets.asset_id or data_config.repo_id if hasattr(config.data, 'assets') else data_config.repo_id

    # Save to all asset locations so both standalone and pipeline usage work:
    # 1. {assets_base_dir}/{config_name}/{asset_id}/ — standalone scripts
    # 2. {checkpoint_base_dir}/{config_name}/{exp_name}/assets/{asset_id}/ — RL pipeline
    save_paths = set()
    standalone_path = config.assets_dirs / asset_id
    save_paths.add(str(standalone_path.resolve()))
    # Use --exp_name override if provided, otherwise fall back to config.exp_name
    effective_config = config
    if exp_name:
        effective_config = dataclasses.replace(config, exp_name=exp_name)
    try:
        pipeline_path = effective_config.checkpoint_dir / "assets" / asset_id
        save_paths.add(str(pipeline_path.resolve()))
    except ValueError:
        pass  # exp_name not set

    for p in sorted(save_paths):
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        normalize.save(p, norm_stats_dict)
        print(f"Saved to: {p}")

    # ---- Summary ----
    print("\nStatistics Summary (first 12 dims):")
    print("State means:  ", np.array(norm_stats_dict["state"].mean[:12]))
    print("Action means: ", np.array(norm_stats_dict["actions"].mean[:12]))

    if per_timestamp_stats is not None:
        pts = np.array(norm_stats_dict["actions"].per_timestamp_mean)
        print(f"\nPer-timestamp means shape: {pts.shape}")
        for t in range(min(3, pts.shape[0])):
            print(f"  t={t}: {pts[t, :5]}")

    if correlation_stats is not None:
        chol = np.array(norm_stats_dict["actions"].action_correlation_cholesky)
        print(f"\nCorrelation (Cholesky): {chol.shape}, {chol.nbytes / 1024 / 1024:.2f} MB")

    print(f"\nStatistics saved to {len(save_paths)} location(s)")


if __name__ == "__main__":
    tyro.cli(main)
