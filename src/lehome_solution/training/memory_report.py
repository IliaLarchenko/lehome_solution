"""Parse JAX device memory profile and print a human-readable summary.

Uses jax.live_arrays() instead of parsing the pprof protobuf,
which avoids needing any extra dependencies.
"""

import logging

import jax
import jax.numpy as jnp


def print_memory_report(profile_path: str | None = None) -> None:
    """Log a summary of current JAX device memory usage.

    Uses jax.live_arrays() to enumerate all live arrays on device,
    grouped by shape and dtype. Also reports the backend memory stats.
    """
    # 1. Backend memory stats (most reliable)
    backend = jax.lib.xla_bridge.get_backend()
    for i, device in enumerate(jax.local_devices()):
        stats = device.memory_stats()
        if stats:
            peak = stats.get("peak_bytes_in_use", 0)
            current = stats.get("bytes_in_use", 0)
            limit = stats.get("bytes_limit", 0)
            logging.info(
                f"Device {i} memory: {current / 1e9:.2f} GB in use, "
                f"{peak / 1e9:.2f} GB peak, {limit / 1e9:.2f} GB limit"
            )

    # 2. Enumerate live arrays
    arrays = jax.live_arrays()
    by_shape = {}
    total_bytes = 0
    for arr in arrays:
        key = (arr.shape, arr.dtype)
        nbytes = arr.size * arr.dtype.itemsize
        total_bytes += nbytes
        if key not in by_shape:
            by_shape[key] = {"count": 0, "bytes": 0}
        by_shape[key]["count"] += 1
        by_shape[key]["bytes"] += nbytes

    # Sort by total bytes descending
    sorted_items = sorted(by_shape.items(), key=lambda x: -x[1]["bytes"])

    logging.info("=" * 90)
    logging.info(f"LIVE ARRAYS — Total: {total_bytes / 1e9:.2f} GB across {len(arrays)} arrays")
    logging.info("=" * 90)
    logging.info(f"{'Size (MB)':>10}  {'Count':>6}  {'Each (MB)':>10}  Shape + Dtype")
    logging.info("-" * 90)
    for (shape, dtype), info in sorted_items[:40]:
        each_mb = info["bytes"] / info["count"] / 1e6
        logging.info(
            f"{info['bytes'] / 1e6:>10.1f}  {info['count']:>6}  {each_mb:>10.2f}  {shape} {dtype}"
        )
    logging.info("=" * 90)
