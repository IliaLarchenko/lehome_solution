"""Train FAST tokenizer for action encoding.

This script:
1. Loads action chunks from LeHome dataset (parquet, no video loading)
2. Applies delta transforms and quantile normalization
3. Trains FAST tokenizer on specified action dimensions
4. Saves tokenizer to assets directory
5. Reports compression statistics
"""

import dataclasses
import json
import numpy as np
import tyro
from pathlib import Path
from transformers import AutoProcessor

# Shared data loading
from data_utils import get_delta_transform_from_config, load_action_chunks


def train_fast_tokenizer(
    action_chunks: np.ndarray,
    vocab_size: int = 1024,
    scale: float = 10.0,
) -> AutoProcessor:
    """Train FAST tokenizer (BPE on DCT coefficients) on action chunks."""
    print(f"Training FAST tokenizer on {len(action_chunks)} action chunks...")
    print(f"  Chunk shape: {action_chunks.shape}")
    print(f"  Vocab size: {vocab_size}, DCT scale: {scale}")

    base_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast",
        trust_remote_code=True,
    )

    action_data_list = [action_chunks[i] for i in range(len(action_chunks))]

    print("Training new tokenizer (this may take a few minutes)...")
    tokenizer = base_tokenizer.fit(
        action_data_list,
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=action_chunks.shape[1],
        action_dim=action_chunks.shape[2],
    )
    print("Tokenizer training complete!")

    sample_chunk = action_chunks[0]
    encoded = tokenizer(sample_chunk[None])[0]
    if isinstance(encoded, list):
        encoded = np.array(encoded)
    print(f"Sample encoding: {len(encoded)} tokens for chunk shape {sample_chunk.shape}")

    return tokenizer


def compute_compression_stats(tokenizer, action_chunks: np.ndarray):
    """Compute compression statistics."""
    print("\nComputing compression statistics...")

    sample_size = min(1000, len(action_chunks))
    rng = np.random.RandomState(42)
    sample_chunks = action_chunks[rng.choice(len(action_chunks), size=sample_size, replace=False)]

    token_lengths = []
    for chunk in sample_chunks:
        encoded = tokenizer(chunk[None])[0]
        if isinstance(encoded, list):
            token_lengths.append(len(encoded))
        else:
            token_lengths.append(
                encoded.shape[0] if hasattr(encoded, "shape") else len(encoded)
            )

    token_lengths = np.array(token_lengths)
    input_size = action_chunks.shape[1] * action_chunks.shape[2]
    avg_tokens = np.mean(token_lengths)

    stats = {
        "compression_ratio": float(input_size / avg_tokens),
        "mean_token_length": float(np.mean(token_lengths)),
        "p99_token_length": float(np.percentile(token_lengths, 99)),
        "min_token_length": float(np.min(token_lengths)),
        "max_token_length": float(np.max(token_lengths)),
    }

    print(f"Compression Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}")

    return stats


def main(
    config_name: str,
    max_files: int | None = None,
    num_workers: int | None = None,
    sample_fraction: float = 1.0,
    encoded_dims: str = "0:12",
    vocab_size: int = 1024,
    scale: float = 10.0,
    extra_roots: list[str] | None = None,
    pipeline_state: str | None = None,
    all_sources: bool = False,
    exp_name: str | None = None,
):
    """Train FAST tokenizer for action encoding.

    Args:
        config_name: Training config — registry name (e.g. pi_modified_bc_rl)
                     or a YAML recipe path (e.g. configs/train_real_bc.yaml)
        max_files: Max parquet files to process per dataset (None = all)
        num_workers: Number of parallel workers (None = auto)
        sample_fraction: Fraction of chunks to sample per file (1.0 = all)
        encoded_dims: Comma-separated dimension ranges (e.g., "0:12")
        vocab_size: FAST vocabulary size
        scale: DCT scaling factor
        extra_roots: Additional dataset roots to include
        pipeline_state: Path to pipeline_state.json — includes BC + all RL datasets
        exp_name: Override exp_name for pipeline checkpoint path (e.g. 'sim_rl').
                  Ensures assets are saved where the pipeline expects them.
    """
    from lehome_solution.training.yaml_train_config import resolve_train_config

    config = resolve_train_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Collect all dataset roots with per-source speed_factor + bucket.
    # Units flow through natively (sim sources: radians; real sources:
    # degrees) — the tokenizer is trained on the same distribution the
    # training-time loader produces, with no unit conversion anywhere.
    dataset_roots: list[tuple[str, float, str]] = []
    _seen_roots: set[str] = set()
    _spec_speed: dict[str, float] = {}
    _spec_bucket: dict[str, str] = {}

    def _add_root(root: str) -> None:
        if root in _seen_roots:
            return
        _seen_roots.add(root)
        dataset_roots.append((root, _spec_speed.get(root, 1.0), _spec_bucket.get(root, "")))

    if data_config.lehome_dataset_root:
        _add_root(data_config.lehome_dataset_root)

    if all_sources and hasattr(config.data, 'data_sources') and config.data.data_sources:
        for spec in config.data.data_sources:
            _spec_speed[spec.root] = getattr(spec, "speed_factor", 1.0)
            _spec_bucket[spec.root] = getattr(spec, "bucket", "")
            _add_root(spec.root)

    if pipeline_state:
        import json as _json
        with open(pipeline_state) as f:
            state = _json.load(f)
        if hasattr(config.data, 'data_sources') and config.data.data_sources:
            for spec in config.data.data_sources:
                _spec_speed[spec.root] = getattr(spec, "speed_factor", 1.0)
                _spec_bucket[spec.root] = getattr(spec, "bucket", "")
                _add_root(spec.root)
        for ds in state.get("rl_datasets", []):
            _add_root(ds["root"])

    if extra_roots:
        for root in extra_roots:
            _add_root(root)

    if not dataset_roots:
        raise ValueError("No dataset roots found. Set lehome_dataset_root in config, or pass --pipeline_state / --extra_roots")

    # Parse encoded dimensions
    encoded_dim_ranges = []
    for range_str in encoded_dims.split(","):
        start, end = map(int, range_str.strip().split(":"))
        encoded_dim_ranges.append((start, end))
    total_encoded_dims = sum(end - start for start, end in encoded_dim_ranges)
    print(f"Encoding {total_encoded_dims} dimensions: {encoded_dims}")

    delta_transform = get_delta_transform_from_config(config_name)
    action_horizon = config.model.action_horizon
    print(f"Action horizon: {action_horizon}")
    print(f"Delta mask: {delta_transform.mask}")

    # Load action chunks from all dataset roots, grouped by bucket.
    bucket_chunks: dict[str, list] = {}
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
                sample_fraction=sample_fraction,
                max_files=max_files,
                num_workers=num_workers,
                speed_factor=speed,
            )
            if data["chunks"] is not None:
                bkey = bucket or "_default"
                if bkey not in bucket_chunks:
                    bucket_chunks[bkey] = []
                bucket_chunks[bkey].append(data["chunks"])
        except FileNotFoundError as e:
            print(f"    WARNING: Skipping — {e}")

    if not bucket_chunks:
        raise RuntimeError("No action chunks found! Check dataset paths and episode lengths.")

    # Bucket-weighted resampling to match training-time bucket_shares.
    bucket_shares = None
    if hasattr(config.data, 'bucket_shares') and config.data.bucket_shares:
        bucket_shares = config.data.bucket_shares

    all_chunks_list = []
    if bucket_shares and len(bucket_chunks) > 1:
        bucket_arrays = {b: np.concatenate(cl, axis=0) for b, cl in bucket_chunks.items()}
        raw_counts = {b: len(a) for b, a in bucket_arrays.items()}
        total_raw = sum(raw_counts.values())

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
            frac = min(target_fracs[b], 1.0)
            target_n = int(raw * frac)
            share_pct = bucket_shares.get(b, 0) * 100
            raw_pct = raw / total_raw * 100
            print(f"  {b:15s}: raw={raw:8d} ({raw_pct:5.1f}%) target_share={share_pct:5.1f}% "
                  f"-> keep {target_n:8d} (frac={frac:.3f})")

        rng = np.random.RandomState(42)
        for b, arr in bucket_arrays.items():
            frac = min(target_fracs[b], 1.0)
            if frac < 1.0 - 1e-9 and len(arr) > 0:
                keep = max(1, int(len(arr) * frac))
                idx = rng.choice(len(arr), size=keep, replace=False)
                all_chunks_list.append(arr[idx])
            elif len(arr) > 0:
                all_chunks_list.append(arr)
    else:
        for cl in bucket_chunks.values():
            all_chunks_list.extend(cl)

    chunks = np.concatenate(all_chunks_list, axis=0)  # [N, H, D]

    print(f"Loaded {len(chunks)} action chunks, shape: {chunks.shape}")

    # Extract encoded dimensions
    encoded_chunks = []
    for start, end in encoded_dim_ranges:
        encoded_chunks.append(chunks[:, :, start:end])
    encoded_chunks = np.concatenate(encoded_chunks, axis=-1)
    print(f"Extracted {encoded_chunks.shape[-1]} encoded dimensions")

    # Apply quantile normalization (FAST always uses quantile, not per-timestamp)
    print(f"\nBefore normalization:")
    print(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
    print(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")

    norm_stats = data_config.norm_stats
    if norm_stats is not None:
        action_stats = norm_stats["actions"]

        encoded_dim_indices = []
        for start, end in encoded_dim_ranges:
            encoded_dim_indices.extend(range(start, end))
        encoded_dim_indices = np.array(encoded_dim_indices)

        q01 = action_stats.q01[encoded_dim_indices]
        q99 = action_stats.q99[encoded_dim_indices]

        print(f"\nQuantile normalization (q01, q99):")
        for i, dim_idx in enumerate(encoded_dim_indices):
            print(f"  dim {dim_idx}: q01={q01[i]:7.4f}, q99={q99[i]:7.4f}, range={q99[i]-q01[i]:7.4f}")

        encoded_chunks = np.clip(encoded_chunks, q01, q99)
        encoded_chunks = 2.0 * (encoded_chunks - q01) / np.maximum(q99 - q01, 1e-6) - 1.0

        print(f"\nAfter normalization:")
        print(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
        print(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")
    else:
        print("WARNING: No normalization stats found, using raw delta actions")

    # Train FAST tokenizer
    tokenizer = train_fast_tokenizer(encoded_chunks, vocab_size=vocab_size, scale=scale)

    # Compression statistics
    compression_stats = compute_compression_stats(tokenizer, encoded_chunks)

    # Save to all asset locations so both standalone and pipeline usage work:
    # 1. {assets_base_dir}/{config_name}/{asset_id}/fast_tokenizer — standalone scripts
    # 2. {checkpoint_base_dir}/{config_name}/{exp_name}/assets/{asset_id}/fast_tokenizer — RL pipeline
    asset_id = data_config.asset_id
    metadata = {
        "vocab_size": vocab_size,
        "scale": scale,
        "encoded_dims": encoded_dims,
        "encoded_dim_ranges": encoded_dim_ranges,
        "total_encoded_dims": total_encoded_dims,
        "action_horizon": action_horizon,
        "num_training_chunks": len(encoded_chunks),
        "compression_stats": compression_stats,
    }

    save_paths = set()
    save_paths.add(str((config.assets_dirs / asset_id).resolve()))
    effective_config = config
    if exp_name:
        effective_config = dataclasses.replace(config, exp_name=exp_name)
    try:
        save_paths.add(str((effective_config.checkpoint_dir / "assets" / asset_id).resolve()))
    except ValueError:
        pass  # exp_name not set

    for p in sorted(save_paths):
        tok_dir = Path(p) / "fast_tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tok_dir)
        with open(tok_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved FAST tokenizer to {tok_dir}")

    print(f"Metadata: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    tyro.cli(main)
