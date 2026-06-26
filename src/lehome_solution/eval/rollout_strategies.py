"""Rollout strategies for RL pipeline garment selection.

Strategies:
1. Random: sample N garments randomly from the pool (default behavior)
2. Full: all garments with N episodes each (unbiased metrics estimation)
3. Curriculum: select garments near target success rate (configurable target_sr)
4. Hard mining: re-roll failed garments with state restoration
5. Success replay: replay successful episodes with different augmentations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lehome_solution.eval.eval_utils import garment_name_to_type

logger = logging.getLogger(__name__)


SUCCESS_RATES_FILE = "success_rates.json"


def load_success_rates(rl_datasets: list[dict]) -> dict | None:
    """Load success rates from success_rates.json in RL datasets.

    Reads the most recent file found across all datasets.
    Returns dict with "by_type" and "by_garment" keys, or None.
    """
    # Check datasets in reverse order (most recent first)
    for ds in reversed(rl_datasets):
        sr_path = Path(ds["root"]) / SUCCESS_RATES_FILE
        if sr_path.exists():
            try:
                with open(sr_path) as f:
                    data = json.load(f)
                logger.info("Loaded success rates from %s: types=%s",
                            sr_path,
                            {k: f"{v:.1%}" for k, v in sorted(data.get("by_type", {}).items())})
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", sr_path, e)
    return None


def load_success_rates_from_file(path: str | Path) -> dict | None:
    """Load success rates from a specific file path."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", p, e)
        return None


def compute_fr_weights(
    items: list[dict[str, Any]],
    success_rates: dict | None,
    default_fr: float = 0.5,
) -> np.ndarray:
    """Compute failure-rate-proportional weights for a list of items.

    Each item must have 'garment' and 'garment_type' keys.
    Weight = 1 - SR(garment), falling back to 1 - SR(type), then default_fr.

    Returns normalized probability array (sums to 1).
    """
    if not items:
        return np.array([], dtype=np.float64)

    sr_by_garment = success_rates.get("by_garment", {}) if success_rates else {}
    sr_by_type = success_rates.get("by_type", {}) if success_rates else {}

    # Count states per garment type to normalize pool size effects.
    # Without this, types with many states dominate regardless of FR.
    from collections import Counter
    type_counts = Counter(item.get("garment_type", "") for item in items)

    weights = np.empty(len(items), dtype=np.float64)
    for i, item in enumerate(items):
        garment = item.get("garment", "")
        garment_type = item.get("garment_type", "")
        if garment in sr_by_garment:
            fr = 1.0 - sr_by_garment[garment]
        elif garment_type in sr_by_type:
            fr = 1.0 - sr_by_type[garment_type]
        else:
            fr = default_fr
        fr = max(fr, 0.01)  # floor to prevent zero-weight items
        # Divide by number of states of this type so total weight per type
        # is proportional to FR, not FR × pool_size
        weights[i] = fr / type_counts[garment_type]

    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights[:] = 1.0 / len(weights)
    return weights


def full_strategy(
    garment_pool: list[str],
    **_kwargs,
) -> list[str]:
    """Return all garments in pool (unbiased metrics estimation)."""
    return list(garment_pool)


def curriculum_strategy(
    garment_pool: list[str],
    num_rollouts: int,
    rng: np.random.RandomState,
    success_rates: dict | None = None,
    target_sr: float = 0.5,
    **_kwargs,
) -> list[str]:
    """Two-level curriculum sampling.

    Step 1 — choose garment TYPE proportional to (1 - type_SR).
        Types with lower overall success rate get more rollouts.
    Step 2 — within the chosen type, choose a garment weighted by
        proximity to target_sr (Gaussian, configurable — default 0.5).

    Both use aggregate SR from recompute_advantages (success_rates.json).

    Args:
        garment_pool: all available garment names
        num_rollouts: total number of rollouts to generate
        success_rates: {"by_type": {type->SR}, "by_garment": {name->SR}}
        target_sr: target success rate for within-type Gaussian weighting
    """
    if not success_rates:
        logger.warning("Curriculum: no success_rates provided — falling back to uniform random sampling!")
        return list(rng.choice(garment_pool, size=num_rollouts, replace=True))

    type_sr = success_rates.get("by_type", {})
    garment_sr = success_rates.get("by_garment", {})

    # Group garments by type
    by_type: dict[str, list[str]] = {}
    for g in garment_pool:
        by_type.setdefault(garment_name_to_type(g), []).append(g)

    # Step 1: type-level weight = 1 - type_SR
    type_weights: dict[str, float] = {}
    for t in by_type:
        sr = type_sr.get(t, 0.0)  # no data → assume hardest
        type_weights[t] = max(1.0 - sr, 0.05)

    types = list(by_type.keys())
    tw = np.array([type_weights[t] for t in types])
    type_probs = tw / tw.sum()

    # Step 2: within-type Gaussian weights (proximity to target_sr)
    # sigma chosen so that SR at 0% or 100% gets ~10x less weight than target
    sigma = 0.233
    within_type_probs: dict[str, np.ndarray] = {}
    for t in types:
        garments = by_type[t]
        scores = []
        for g in garments:
            if g in garment_sr:
                diff = garment_sr[g] - target_sr
                scores.append(np.exp(-(diff ** 2) / (2 * sigma ** 2)))
            else:
                scores.append(1.0)  # no data → peak score
        s = np.array(scores)
        s = np.maximum(s, 0.02)
        within_type_probs[t] = s / s.sum()

    # Sample: pick type, then pick garment within type
    type_choices = rng.choice(len(types), size=num_rollouts, replace=True, p=type_probs)
    sampled = []
    for ti in type_choices:
        t = types[ti]
        garments = by_type[t]
        g = rng.choice(garments, p=within_type_probs[t])
        sampled.append(g)

    for t in sorted(types):
        logger.info("  Curriculum %s: type_weight=%.3f (SR=%.1f%%), n_garments=%d",
                    t, type_weights[t], type_sr.get(t, 0.0) * 100, len(by_type[t]))
    logger.info("Curriculum: %d rollouts from %d garments (target_sr=%.2f)",
                num_rollouts, len(garment_pool), target_sr)
    return sampled


def _read_state_metadata(npz_path: Path) -> dict[str, Any] | None:
    """Read metadata from a JSON sidecar or fall back to NPZ metadata key.

    Returns parsed metadata dict, or None on failure.
    """
    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read sidecar %s: %s, falling back to NPZ", json_path, e)

    # Fallback: read from NPZ metadata key (old format, no JSON sidecar)
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            meta_str = str(data["metadata"])
            return json.loads(meta_str)
    except Exception as e:
        logger.warning("Failed to read NPZ metadata %s: %s", npz_path, e)
        return None


def _write_state_sidecar(npz_path: Path, meta: dict[str, Any]):
    """Write a JSON metadata sidecar alongside an NPZ file."""
    json_path = npz_path.with_suffix(".json")
    try:
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
    except OSError as e:
        logger.warning("Failed to write sidecar %s: %s", json_path, e)


def get_failure_states(failure_states_dir: str | Path) -> list[dict[str, Any]]:
    """Read metadata from a failure states directory.

    Reads from JSON sidecars when available, falls back to NPZ metadata.
    Returns list of {garment, garment_type, seed, npz_path, status} dicts.
    Only returns states with status == "active" (or no status field).
    """
    fs_dir = Path(failure_states_dir)
    if not fs_dir.exists():
        return []

    results = []
    for npz_path in sorted(fs_dir.glob("*.npz")):
        meta = _read_state_metadata(npz_path)
        if meta is None:
            continue
        status = meta.get("status", "active")
        if status not in ("active",):
            continue
        results.append({
            "garment": meta.get("garment", ""),
            "garment_type": meta.get("garment_type", ""),
            "seed": meta.get("seed", 0),
            "npz_path": str(npz_path),
        })

    return results


def hard_mining_strategy(
    garment_pool: list[str],
    num_garments: int,
    rng: np.random.RandomState,
    failure_states: list[dict[str, Any]] | None = None,
    replays_per_state: int = 1,
    remove_on_success: bool = True,
    success_rates: dict | None = None,
    **_kwargs,
) -> tuple[list[str], list[dict] | None]:
    """Sample from NPZ failure states for state restoration replay.

    Samples failure states proportionally to garment failure rate (FR).
    Each sampled state emits a single task with ``max_attempts=replays_per_state``;
    the worker retries the task on failure until success or the attempt budget
    is exhausted.
    remove_on_success: if True, NPZ is removed from pool on success.

    If no failure states are available, returns empty lists (caller should skip).
    Samples with replacement when fewer states than requested.

    Returns:
        garments: list of garment names to evaluate
        task_overrides: list of {garment, seed, restore_npz, hard_mining_mode,
            max_attempts} dicts, or None if no failure states available
    """
    eligible = []
    if failure_states:
        eligible = [f for f in failure_states if f.get("garment", "") in garment_pool]

    if not eligible:
        logger.info("Hard mining: no failure states available, skipping")
        return [], None

    n_total = num_garments
    replace = len(eligible) < n_total

    # FR-proportional sampling
    fr_weights = compute_fr_weights(eligible, success_rates)
    indices = rng.choice(len(eligible), size=n_total, replace=replace, p=fr_weights)

    mode = "exact" if remove_on_success else "augs"
    task_overrides = []
    for idx in indices:
        f = eligible[idx]
        task_overrides.append({
            "garment": f["garment"],
            "seed": f.get("seed", 0),
            "restore_npz": f["npz_path"],
            "hard_mining_mode": mode,
            "max_attempts": int(replays_per_state),
        })

    selected_garments = list({ov["garment"] for ov in task_overrides})
    logger.info("Hard mining [remove_on_success=%s] (NPZ restore): %d tasks (up to %d attempts each, "
                "early-stop on success) from %d unique garments (pool=%d failure states)",
                remove_on_success, len(task_overrides), replays_per_state,
                len(selected_garments), len(eligible))
    return selected_garments, task_overrides


def get_success_states(
    success_states_dir: str | Path,
    min_actions: int = 60,
) -> list[dict[str, Any]]:
    """Read metadata from a success states directory.

    Reads from JSON sidecars when available, falls back to NPZ metadata.
    Returns list of {garment, garment_type, seed, npz_path, n_actions} dicts.
    Only returns states with status == "active" (or no status field)
    and n_actions >= min_actions.

    No side effects: does NOT delete short states.
    """
    ss_dir = Path(success_states_dir)
    if not ss_dir.exists():
        return []

    results = []
    skipped = 0
    for npz_path in sorted(ss_dir.glob("*.npz")):
        meta = _read_state_metadata(npz_path)
        if meta is None:
            continue
        status = meta.get("status", "active")
        if status not in ("active",):
            continue
        # n_actions from sidecar (preferred) or fall back to loading NPZ
        n_actions = meta.get("n_actions")
        if n_actions is None:
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    n_actions = int(data["actions"].shape[0]) if "actions" in data else 0
            except Exception:
                n_actions = 0
        else:
            n_actions = int(n_actions)
        if n_actions < min_actions:
            skipped += 1
            continue
        results.append({
            "garment": meta.get("garment", ""),
            "garment_type": meta.get("garment_type", ""),
            "seed": meta.get("seed", 0),
            "npz_path": str(npz_path),
            "n_actions": n_actions,
            "dagger": bool(meta.get("dagger", False)),
        })

    if skipped:
        logger.info("Skipped %d success states with < %d actions", skipped, min_actions)
    return results


def mark_states_in_use(states: list[dict[str, Any]], rollout_id: str = ""):
    """Mark selected states as in_use by updating their JSON sidecars.

    Creates sidecars if they don't exist (backward compat).
    """
    for s in states:
        npz_path = Path(s["npz_path"])
        if not npz_path.exists():
            continue
        meta = _read_state_metadata(npz_path)
        if meta is None:
            meta = {k: v for k, v in s.items() if k != "npz_path"}
        meta["status"] = "in_use"
        if rollout_id:
            meta["rollout_id"] = rollout_id
        _write_state_sidecar(npz_path, meta)


def mark_states_consumed(states: list[dict[str, Any]]):
    """Mark states as consumed by updating their JSON sidecars.

    Called after dataset write is confirmed. Consumed states can be
    safely deleted by cleanup_consumed_states().
    """
    for s in states:
        npz_path = Path(s["npz_path"])
        if not npz_path.exists():
            continue
        meta = _read_state_metadata(npz_path)
        if meta is None:
            meta = {k: v for k, v in s.items() if k != "npz_path"}
        meta["status"] = "consumed"
        _write_state_sidecar(npz_path, meta)


def cleanup_consumed_states(states_dir: str | Path) -> int:
    """Delete NPZ+JSON pairs with status="consumed".

    Returns number of removed state pairs.
    """
    sd = Path(states_dir)
    if not sd.exists():
        return 0

    removed = 0
    for npz_path in list(sd.glob("*.npz")):
        meta = _read_state_metadata(npz_path)
        if meta is None:
            continue
        if meta.get("status") == "consumed":
            npz_path.unlink()
            json_path = npz_path.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
            removed += 1

    if removed:
        logger.info("Cleaned up %d consumed states from %s", removed, sd)
    return removed


def _replay_strategy_common(
    garment_pool: list[str],
    num_states: int,
    replays_per_state: int,
    rng: np.random.RandomState,
    states: list[dict[str, Any]] | None,
    success_rates: dict | None,
    replay_key: str,
    label: str,
    dagger_only: bool = False,
    uniform_garment_sampling: bool = False,
) -> tuple[list[str], list[dict] | None]:
    """Shared sampling logic for success_replay and semi_success_replay.

    When ``uniform_garment_sampling=False`` (default): samples states
    proportionally to garment failure rate (FR) — focuses on harder garments.

    When ``uniform_garment_sampling=True``: every garment instance
    (e.g. ``Pant_Long_Seen_4``) gets equal probability regardless of how many
    states it has or its success rate. This is preferred for real-format
    dataset collection where balanced per-garment coverage matters.

    Args:
        replay_key: key set to True in each task override (e.g. "success_replay")
        label: human-readable label for log messages
        dagger_only: if True, only consider states saved from DAgger collection.
        uniform_garment_sampling: if True, sample uniformly per garment instance.
    """
    eligible = []
    if states:
        eligible = [s for s in states if s.get("garment", "") in garment_pool]
        if dagger_only:
            eligible = [s for s in eligible if s.get("dagger")]

    if not eligible:
        reason = "no dagger-origin states" if dagger_only else "no states available"
        logger.info("%s: %s, skipping", label, reason)
        return [], None

    if uniform_garment_sampling:
        from collections import Counter
        garment_counts = Counter(s.get("garment", "") for s in eligible)
        weights = np.array(
            [1.0 / garment_counts[s.get("garment", "")] for s in eligible],
            dtype=np.float64,
        )
        weights /= weights.sum()
        n_garments = len(garment_counts)
        logger.info(
            "%s: uniform garment sampling across %d garment instances (%d states)",
            label, n_garments, len(eligible),
        )
    else:
        # FR-proportional sampling (with replacement when pool < requested)
        weights = compute_fr_weights(eligible, success_rates)
    replace = num_states > len(eligible)
    n_sample = num_states  # sample with replacement if pool is small
    indices = rng.choice(len(eligible), size=n_sample, replace=replace, p=weights)

    task_overrides = []
    for idx in indices:
        s = eligible[idx]
        # One task per state; worker retries up to replays_per_state attempts
        # and early-stops on the first success.
        task_overrides.append({
            "garment": s["garment"],
            "seed": s.get("seed", 0),
            "restore_npz": s["npz_path"],
            replay_key: True,
            "max_attempts": int(replays_per_state),
        })

    selected_garments = list({ov["garment"] for ov in task_overrides})
    logger.info(
        "%s: %d tasks (up to %d attempts each, early-stop on success) from %d unique garments "
        "(pool=%d states, requested %d)",
        label, len(task_overrides), replays_per_state,
        len(selected_garments), len(eligible), num_states,
    )
    return selected_garments, task_overrides


def success_replay_strategy(
    garment_pool: list[str],
    num_states: int,
    replays_per_state: int,
    rng: np.random.RandomState,
    success_states: list[dict[str, Any]] | None = None,
    success_rates: dict | None = None,
    uniform_garment_sampling: bool = False,
    **_kwargs,
) -> tuple[list[str], list[dict] | None]:
    """Sample from success states for open-loop replay with different augmentations.

    Each success state is replayed `replays_per_state` times with different
    garment augmentations (100% probability). The saved actions are replayed
    open-loop; the policy is only called for value predictions.
    Only successful replays are saved to the dataset.
    """
    return _replay_strategy_common(
        garment_pool, num_states, replays_per_state, rng,
        states=success_states, success_rates=success_rates,
        replay_key="success_replay", label="Success replay",
        uniform_garment_sampling=uniform_garment_sampling,
    )


def semi_success_replay_strategy(
    garment_pool: list[str],
    num_states: int,
    replays_per_state: int,
    rng: np.random.RandomState,
    semi_success_states: list[dict[str, Any]] | None = None,
    success_rates: dict | None = None,
    dagger_only: bool = False,
    **_kwargs,
) -> tuple[list[str], list[dict] | None]:
    """Sample from semi-success states (reached first checkpoint but failed).

    Saved actions are replayed up to the checkpoint, then the policy takes
    over. Only successful completions are saved to the dataset (and also
    promoted to success states for future success_replay).

    When ``dagger_only`` is True, only states marked ``dagger=True`` in their
    metadata are eligible — the resulting rollouts are intended to be routed
    to the dagger dataset pool by the caller.
    """
    label = "Semi-success replay (dagger-only)" if dagger_only else "Semi-success replay"
    return _replay_strategy_common(
        garment_pool, num_states, replays_per_state, rng,
        states=semi_success_states, success_rates=success_rates,
        replay_key="semi_success_replay", label=label,
        dagger_only=dagger_only,
    )
