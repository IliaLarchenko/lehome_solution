"""Thompson Sampling optimization for inference hyperparameters.

Each parameter is optimized independently. Per-parameter Beta(alpha, beta)
posteriors are maintained and updated with per-garment-type-SR-normalized
rewards (baseline subtracted at the type level, not the garment level, to match
the total-SR optimization objective).

Only "full" (all-garment) rollouts update the prior. Partial / curriculum /
hard-mining / success-replay rollouts are excluded — they bias the bandit
against configs that happened to run during a harder-than-average subset.

Parameters:
    actions_to_execute: {3,5,10} — how many of 30 predicted actions to use
    k_execute: {1.0,1.2} — execute_in_n_steps = int(k * actions_to_execute)
    actions_to_keep: {0,3,6} — trailing actions for inpainting
    num_steps: fixed at 10 — diffusion denoising iterations
    time_threshold_inpaint: {0.4,0.5} — when inpainting correction activates
    cfg_scale: {5.0,7.0,9.0,11.0} — classifier-free guidance scale
    noise_temperature: {0.7,0.8,0.9} — scales initial noise covariance for exploration
    num_rollout_candidates: {1,2,3,4} — best-of-N at the policy server (cost ~N×
        action-expert; VLM prefix is shared across candidates).
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter spaces (ordered discrete values)
# ---------------------------------------------------------------------------

PARAM_SPACES: dict[str, list[float]] = {
    "actions_to_execute": [3, 5, 10],
    "k_execute": [1.0, 1.2],
    "actions_to_keep": [0, 3, 6],
    "time_threshold_inpaint": [0.4, 0.5],
    "cfg_scale": [5.0, 7.0, 9.0, 11.0],
    "noise_temperature": [0.7, 0.8, 0.9],
    "num_rollout_candidates": [1, 2, 3, 4],
}

# Global default config — the single set of "reasonable numbers" used for ALL
# garment types when nothing more specific is supplied (no per-garment-type
# config, no Thompson prior).  Derived as the per-parameter MODE across the
# four tuned per-garment-type sim-round configs (top_long / top_short /
# pant_long / pant_short).  Two parameters were tied and broken toward the
# smoother/safer value: actions_to_execute (5 vs 3 → 5), time_threshold_inpaint
# (0.4 vs 0.5 → 0.5).  num_steps is fixed at 10 (not optimized) to enable
# server-side batching.
DEFAULT_CONFIG = {
    "actions_to_execute": 5,
    "k_execute": 1.0,
    "actions_to_keep": 3,
    "num_steps": 10,
    "time_threshold_inpaint": 0.5,
    "cfg_scale": 7.0,
    "noise_temperature": 0.7,
    "num_rollout_candidates": 3,
}

from lehome_solution.constants import ACTION_HORIZON

# Bump this when the prior format changes to force a fresh start.
_PRIOR_VERSION = 2  # v2: per-garment-type optimization


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _init_params_block() -> dict:
    """Create a fresh {param_name: {value: Beta(1,1)}} dict."""
    return {
        param_name: {str(v): {"alpha": 1.0, "beta": 1.0} for v in values}
        for param_name, values in PARAM_SPACES.items()
    }


def init_prior() -> dict:
    """Create uniform prior: Beta(1, 1) for all parameter values.

    Also initializes per-garment-type sub-priors that accumulate evidence
    separately, so per-type views can be visualized without being reset each
    batch.
    """
    from lehome_solution.constants import GARMENT_TYPES

    prior = {
        "version": _PRIOR_VERSION,
        "params": _init_params_block(),
        "iteration": 0,
        "history": [],
        "per_garment_type": {
            gt: {"params": _init_params_block(), "n_episodes": 0}
            for gt in GARMENT_TYPES
        },
    }
    return prior


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _thompson_sample_param(
    param_prior: dict[str, dict],
    rng: np.random.RandomState,
) -> float:
    """Thompson-sample one parameter value from its Beta posteriors."""
    values = []
    samples = []
    for v_str, ab in param_prior.items():
        values.append(float(v_str))
        samples.append(rng.beta(ab["alpha"], ab["beta"]))
    best_idx = int(np.argmax(samples))
    return values[best_idx]


def _get_garment_type_params(prior: dict, garment_type: str | None) -> dict:
    """Return the params block for a garment type, falling back to combined."""
    if garment_type and "per_garment_type" in prior:
        entry = prior["per_garment_type"].get(garment_type)
        if entry and "params" in entry:
            return entry["params"]
    return prior["params"]


def sample_config(
    prior: dict,
    rng: np.random.RandomState,
    garment_type: str | None = None,
) -> dict:
    """Sample a full inference config from the prior via Thompson Sampling.

    Samples each parameter independently and rejects configs that violate
    constraints (actions_to_execute + actions_to_keep <= 30).

    Args:
        prior: Current prior state dict.
        rng: Random state for Thompson Sampling.
        garment_type: If provided, use the per-garment-type sub-prior.
    """
    params_prior = _get_garment_type_params(prior, garment_type)
    for _ in range(100):  # rejection loop for constraints
        config = {}
        for param_name in PARAM_SPACES:
            config[param_name] = _thompson_sample_param(params_prior[param_name], rng)

        # When ate < 4, the wrapper's cubic-spline interpolation cannot
        # resample the chunk (needs >= 4 anchor points). Force k_execute=1
        # so eins=ate and no interpolation happens. This is logged back
        # into the prior so the bandit posterior reflects what was actually
        # executed, not what was originally drawn.
        if int(config["actions_to_execute"]) < 4 and config["k_execute"] != 1.0:
            config["k_execute"] = 1.0

        # Derive execute_in_n_steps
        config["execute_in_n_steps"] = max(1, int(config["k_execute"] * config["actions_to_execute"]))
        # Cast integer params
        config["actions_to_execute"] = int(config["actions_to_execute"])
        config["actions_to_keep"] = int(config["actions_to_keep"])
        config["num_rollout_candidates"] = max(1, int(config["num_rollout_candidates"]))
        # num_steps is fixed (not optimized), add from DEFAULT_CONFIG
        config["num_steps"] = DEFAULT_CONFIG["num_steps"]

        # Constraint check
        if config["actions_to_execute"] + config["actions_to_keep"] <= ACTION_HORIZON:
            return config

    # Fallback to default if rejection sampling fails (shouldn't happen)
    logger.warning("Rejection sampling exhausted, using default config")
    result = dict(DEFAULT_CONFIG)
    if int(result["actions_to_execute"]) < 4 and result["k_execute"] != 1.0:
        result["k_execute"] = 1.0
    result["execute_in_n_steps"] = int(result["k_execute"] * result["actions_to_execute"])
    result["actions_to_execute"] = int(result["actions_to_execute"])
    result["actions_to_keep"] = int(result["actions_to_keep"])
    result["num_rollout_candidates"] = max(1, int(result["num_rollout_candidates"]))
    return result


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def update_prior(
    prior: dict,
    episodes: list[dict],
    *,
    decay_factor: float = 0.99,
    garment_type_success_rates: dict[str, float] | None = None,
    **_kwargs,
) -> dict:
    """Update prior with episode results using Thompson Sampling.

    Uses baseline-subtracted reward ``r = ep_success − SR(garment_type)`` split
    into non-negative deltas: ``α += max(r, 0)``, ``β += max(−r, 0)``. The
    per-type baseline matches the optimization objective (total SR across
    types) and gives a lower-variance baseline than per-garment SR.

    Args:
        prior: Current prior state dict.
        episodes: List of dicts with keys:
            - inference_config: dict with sampled parameter values
            - garment: str (garment name)
            - garment_type: str
            - reward: float (binary success; 1.0 success, 0.0 failure)
        decay_factor: Shrink posteriors toward uniform (Beta(1,1)) each iteration.
        garment_type_success_rates: Per-garment-TYPE SR (garment_type -> SR in
            [0, 1]), used as the baseline subtracted from reward. Falls back to
            0.5 when the type is missing.

    Returns:
        Updated prior dict.
    """
    if not episodes:
        return prior

    from lehome_solution.constants import GARMENT_TYPES

    sr = garment_type_success_rates or {}

    # Ensure per-garment-type sub-priors exist (backward compat for old files)
    if "per_garment_type" not in prior:
        prior["per_garment_type"] = {
            gt: {"params": _init_params_block(), "n_episodes": 0}
            for gt in GARMENT_TYPES
        }
    else:
        for gt in GARMENT_TYPES:
            if gt not in prior["per_garment_type"]:
                prior["per_garment_type"][gt] = {
                    "params": _init_params_block(), "n_episodes": 0,
                }

    # 1. Apply decay toward uniform (shrink alpha, beta toward 1.0) — both
    # combined and per-garment-type sub-priors decay on the same schedule.
    def _decay_params(params_block: dict):
        for param_name in PARAM_SPACES:
            for v_str, ab in params_block[param_name].items():
                ab["alpha"] = 1.0 + decay_factor * (ab["alpha"] - 1.0)
                ab["beta"] = 1.0 + decay_factor * (ab["beta"] - 1.0)

    _decay_params(prior["params"])
    for gt in GARMENT_TYPES:
        _decay_params(prior["per_garment_type"][gt]["params"])

    def _apply_update(params_block: dict, config: dict, delta_pos: float, delta_neg: float):
        for param_name, values in PARAM_SPACES.items():
            sampled_value = config.get(param_name)
            if sampled_value is None:
                continue
            str_values = [str(v) for v in values]
            sampled_str = str(sampled_value)
            if sampled_str not in str_values:
                try:
                    sampled_float = float(sampled_value)
                    idx = min(range(len(values)), key=lambda i: abs(values[i] - sampled_float))
                    sampled_str = str_values[idx]
                except (ValueError, TypeError):
                    continue
            params_block[param_name][sampled_str]["alpha"] += delta_pos
            params_block[param_name][sampled_str]["beta"] += delta_neg

    # 2. Update combined prior and the matching per-garment-type sub-prior.
    # r = ep_success − SR(garment_type); α += max(r, 0); β += max(−r, 0).
    for ep in episodes:
        config = ep.get("inference_config")
        if config is None:
            continue
        gt = ep.get("garment_type", "")
        reward = float(ep["reward"]) - sr.get(gt, 0.5)
        delta_pos = max(0.0, reward)
        delta_neg = max(0.0, -reward)

        _apply_update(prior["params"], config, delta_pos, delta_neg)

        if gt in prior["per_garment_type"]:
            _apply_update(prior["per_garment_type"][gt]["params"], config, delta_pos, delta_neg)
            prior["per_garment_type"][gt]["n_episodes"] += 1

    # 3. Cross-type regularization: shrink each type's posterior toward the
    # pooled success rate while preserving each type's own evidence (α+β).
    # Pool via SUM (so types with more episodes have proportionally more pull
    # on the pooled direction, matching their reliability), then rescale the
    # pooled (α,β) to match the current type's (α+β). The blend is a convex
    # combination, so each type's (α+β) is preserved exactly; only the ratio
    # α/(α+β) is pulled toward total_α/(total_α+total_β).
    _REGULARIZATION_WEIGHT = 0.1
    gt_entries = [prior["per_garment_type"][gt] for gt in GARMENT_TYPES
                  if gt in prior["per_garment_type"]]
    if len(gt_entries) > 1:
        for param_name in PARAM_SPACES:
            for v_str in gt_entries[0]["params"][param_name]:
                total_alpha = float(sum(e["params"][param_name][v_str]["alpha"] for e in gt_entries))
                total_beta = float(sum(e["params"][param_name][v_str]["beta"] for e in gt_entries))
                total_ab = total_alpha + total_beta
                if total_ab <= 0.0:
                    continue
                for e in gt_entries:
                    ab = e["params"][param_name][v_str]
                    own_ab = ab["alpha"] + ab["beta"]
                    scale = own_ab / total_ab
                    ref_alpha = total_alpha * scale
                    ref_beta = total_beta * scale
                    ab["alpha"] = (1.0 - _REGULARIZATION_WEIGHT) * ab["alpha"] + _REGULARIZATION_WEIGHT * ref_alpha
                    ab["beta"] = (1.0 - _REGULARIZATION_WEIGHT) * ab["beta"] + _REGULARIZATION_WEIGHT * ref_beta

    # 4. Record history
    iteration = prior.get("iteration", 0)
    history_entry = {
        "iteration": iteration,
        "n_episodes": len(episodes),
        "avg_reward": float(np.mean([ep["reward"] for ep in episodes])),
        "best_config": {gt: get_best_config(prior, garment_type=gt) for gt in GARMENT_TYPES},
    }
    prior.setdefault("history", []).append(history_entry)
    prior["iteration"] = iteration + 1

    return prior


# ---------------------------------------------------------------------------
# Best config extraction
# ---------------------------------------------------------------------------

def get_best_config(prior: dict, garment_type: str | None = None) -> dict:
    """Return the MAP config (highest alpha/(alpha+beta) per parameter).

    Args:
        prior: Current prior state dict.
        garment_type: If provided, use the per-garment-type sub-prior.
    """
    params = _get_garment_type_params(prior, garment_type)
    config = {}
    for param_name, values in PARAM_SPACES.items():
        best_v = None
        best_mean = -1.0
        for v in values:
            v_str = str(v)
            ab = params[param_name].get(v_str, {"alpha": 1.0, "beta": 1.0})
            mean = ab["alpha"] / (ab["alpha"] + ab["beta"])
            if mean > best_mean:
                best_mean = mean
                best_v = v
        config[param_name] = best_v

    # Derive execute_in_n_steps and cast types
    config["execute_in_n_steps"] = max(1, int(config["k_execute"] * config["actions_to_execute"]))
    config["actions_to_execute"] = int(config["actions_to_execute"])
    config["actions_to_keep"] = int(config["actions_to_keep"])
    config["num_rollout_candidates"] = max(1, int(config["num_rollout_candidates"]))
    config["num_steps"] = DEFAULT_CONFIG["num_steps"]
    return config


def _posterior_summary_for_params(params: dict, prefix: str) -> dict:
    """Compute posterior summary metrics for a single params block."""
    summary = {}
    for param_name, values in PARAM_SPACES.items():
        means = []
        for v in values:
            v_str = str(v)
            ab = params[param_name].get(v_str, {"alpha": 1.0, "beta": 1.0})
            means.append(ab["alpha"] / (ab["alpha"] + ab["beta"]))

        # Best value and its posterior mean
        best_idx = int(np.argmax(means))
        summary[f"{prefix}/{param_name}_best"] = values[best_idx]
        summary[f"{prefix}/{param_name}_best_prob"] = means[best_idx]

        # Entropy of the posterior means (normalized as probabilities)
        probs = np.array(means)
        probs = probs / probs.sum()  # normalize to prob distribution
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(values))
        summary[f"{prefix}/{param_name}_entropy_ratio"] = float(entropy / max_entropy)

    return summary


def get_posterior_summary(prior: dict) -> dict:
    """Get summary statistics for wandb logging.

    Returns per-garment-type best configs and entropy metrics.
    """
    from lehome_solution.constants import GARMENT_TYPES

    summary = {}

    # Per-garment-type summaries
    for gt in GARMENT_TYPES:
        gt_params = _get_garment_type_params(prior, gt)
        gt_summary = _posterior_summary_for_params(gt_params, f"inference_param_opt/{gt}")
        summary.update(gt_summary)

        best = get_best_config(prior, garment_type=gt)
        summary[f"inference_param_opt/{gt}/best_actions_to_execute"] = best["actions_to_execute"]
        summary[f"inference_param_opt/{gt}/best_execute_in_n_steps"] = best["execute_in_n_steps"]
        summary[f"inference_param_opt/{gt}/best_actions_to_keep"] = best["actions_to_keep"]
        summary[f"inference_param_opt/{gt}/best_num_steps"] = best["num_steps"]
        summary[f"inference_param_opt/{gt}/best_time_threshold_inpaint"] = best["time_threshold_inpaint"]
        summary[f"inference_param_opt/{gt}/best_noise_temperature"] = best.get("noise_temperature", 1.0)
        summary[f"inference_param_opt/{gt}/best_num_rollout_candidates"] = best.get("num_rollout_candidates", 1)
        n_eps = prior.get("per_garment_type", {}).get(gt, {}).get("n_episodes", 0)
        summary[f"inference_param_opt/{gt}/n_episodes"] = n_eps

    return summary


# ---------------------------------------------------------------------------
# Per-garment-type posterior views (for logging only, not used in decisions)
# ---------------------------------------------------------------------------

def get_garment_type_priors(prior: dict) -> dict[str, dict]:
    """Return the persisted per-garment-type sub-priors as prior-like dicts.

    Each sub-prior has the same shape as the combined prior ({"params",
    "iteration"}) so it can be passed directly to `plot_inference_prior`.
    Evidence accumulates across iterations on the same decay schedule as the
    combined prior (see `update_prior`) — not reset per batch.
    """
    from lehome_solution.constants import GARMENT_TYPES

    iteration = prior.get("iteration", 0)
    sub = prior.get("per_garment_type", {})
    result = {}
    for gt in GARMENT_TYPES:
        entry = sub.get(gt)
        if entry is None or "params" not in entry:
            # Missing sub-prior: return a uniform view rather than crashing
            result[gt] = {
                "params": _init_params_block(),
                "iteration": iteration,
                "n_episodes": 0,
            }
            continue
        result[gt] = {
            "params": entry["params"],
            "iteration": iteration,
            "n_episodes": entry.get("n_episodes", 0),
        }
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_prior(prior: dict, path: str | Path):
    """Save prior state to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prior, f, indent=2)
    logger.info("Saved inference prior to %s", path)


def load_prior(path: str | Path) -> dict:
    """Load prior state from JSON file.

    Reconciles with current PARAM_SPACES: if a parameter's set of values
    changed, its posteriors are reset to uniform Beta(1, 1).  Parameters
    whose value sets are unchanged keep their learned posteriors.

    If the prior version is outdated, a fresh prior is returned (forces reset
    on format changes).
    """
    from lehome_solution.constants import GARMENT_TYPES

    with open(path) as f:
        prior = json.load(f)

    # Version check — reset on format change
    if prior.get("version", 0) < _PRIOR_VERSION:
        logger.info("Prior version %d < %d, resetting to fresh prior",
                     prior.get("version", 0), _PRIOR_VERSION)
        fresh = init_prior()
        save_prior(fresh, path)
        return fresh

    def _reconcile(params_block: dict) -> tuple[list[str], list[str]]:
        """Targeted edit: drop removed arms, add new arms at Beta(1,1), preserve
        the rest. Existing arms keep their posteriors so history isn't wiped
        when we tune the search space.
        """
        added: list[str] = []
        dropped: list[str] = []
        for param_name, values in PARAM_SPACES.items():
            expected_keys = {str(v) for v in values}
            existing_block = params_block.setdefault(param_name, {})
            for k in list(existing_block.keys()):
                if k not in expected_keys:
                    del existing_block[k]
                    dropped.append(f"{param_name}={k}")
            for v in values:
                v_str = str(v)
                if v_str not in existing_block:
                    existing_block[v_str] = {"alpha": 1.0, "beta": 1.0}
                    added.append(f"{param_name}={v_str}")
        for old_name in list(params_block.keys()):
            if old_name not in PARAM_SPACES:
                del params_block[old_name]
                dropped.append(old_name)
        return added, dropped

    prior.setdefault("params", _init_params_block())
    added, dropped = _reconcile(prior["params"])
    if added or dropped:
        logger.info("Reconciled inference prior arms: added=%s dropped=%s", added, dropped)

    # Ensure per-garment-type sub-priors exist and are reconciled
    sub = prior.setdefault("per_garment_type", {})
    for gt in GARMENT_TYPES:
        entry = sub.setdefault(gt, {"params": _init_params_block(), "n_episodes": 0})
        entry.setdefault("params", _init_params_block())
        entry.setdefault("n_episodes", 0)
        gt_added, gt_dropped = _reconcile(entry["params"])
        if gt_added or gt_dropped:
            logger.info("Reconciled per-garment-type prior[%s] arms: added=%s dropped=%s",
                        gt, gt_added, gt_dropped)

    # Drop stale garment types
    for old_gt in list(sub.keys()):
        if old_gt not in GARMENT_TYPES:
            del sub[old_gt]

    return prior
