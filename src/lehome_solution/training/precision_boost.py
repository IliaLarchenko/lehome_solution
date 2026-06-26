"""Per-garment precision boost for advantage weighting.

For each garment, picks the top-K fraction of successful episodes (by binding-
condition margin on the final frame's ``check_distances``) and adds a
configurable bonus to every frame's advantage.

Metric: ``min_margin = min_i margin_i`` where
  margin_i = 1 - ratio_i   for '<=' conditions
           = ratio_i - 1   for '>=' conditions
Conditions whose stored ``success_distance`` is 0 are skipped — the launcher
stores those as ratio=0 regardless of physics (Top_Short_Seen_3 quirk).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from lehome_solution.constants import DISTANCE_CMPS

logger = logging.getLogger(__name__)

# Launcher PATCH 5 appends 3 "<=" conditions for pant_long with
# threshold 20cm × scale. They participate in min_margin alongside the
# per-garment config thresholds.
_PANT_LONG_EXTRA_SD = (20.0, 20.0, 20.0)

DEFAULT_CHALLENGE_ROOT = Path(
    "lehome-challenge/Assets/objects/Challenge_Garment/Release"
)


def _name_to_type(garment_name: str) -> str:
    parts = garment_name.split("_")
    if len(parts) < 2:
        return ""
    return f"{parts[0].lower()}_{parts[1].lower()}"


def load_garment_success_distances(
    challenge_root: Path | None = None,
) -> dict[str, list[float]]:
    """Load ``success_distance`` per garment from challenge config JSONs.

    Returns ``{garment_name: [sd_0, sd_1, ...]}``; pant_long entries are
    extended by 3 synthetic thresholds matching the launcher's extras.
    """
    root = challenge_root or DEFAULT_CHALLENGE_ROOT
    out: dict[str, list[float]] = {}
    for pattern in ("*_Seen_*", "*_Unseen_*"):
        for gdir in sorted(root.rglob(pattern)):
            if not gdir.is_dir():
                continue
            jsons = list(gdir.glob("*_obj_exp.json"))
            if not jsons:
                continue
            d = json.loads(jsons[0].read_text())
            sd = list(map(float, d.get("success_distance", [])))
            if _name_to_type(gdir.name) == "pant_long":
                sd = sd + list(_PANT_LONG_EXTRA_SD)
            out[gdir.name] = sd
    if not out:
        raise RuntimeError(
            f"No garment configs found under {root}. "
            "Is the lehome-challenge submodule initialized?"
        )
    return out


def compute_min_margin(
    check_distances,
    garment_type: str,
    success_distance: list[float] | None,
) -> float:
    """Binding-condition margin on the given frame. NaN if not computable.

    ``check_distances`` is either a numpy array or a length-N list of floats
    (or nested singleton lists, as parquet can store them).
    """
    cmps = DISTANCE_CMPS.get(garment_type)
    if cmps is None or check_distances is None:
        return float("nan")
    cd = np.asarray(check_distances, dtype=float).ravel()
    if cd.size < len(cmps):
        return float("nan")
    margins: list[float] = []
    for i, cmp in enumerate(cmps):
        if success_distance is not None and i < len(success_distance) and success_distance[i] == 0:
            continue  # stored-threshold=0 → ratio stored as 0, skip
        if cmp == "<=":
            margins.append(1.0 - float(cd[i]))
        else:
            margins.append(float(cd[i]) - 1.0)
    if not margins:
        return float("nan")
    return float(min(margins))


def apply_precision_boost(
    all_episodes: list[dict],
    all_advantages: list[list[float]],
    garment_success_distances: dict[str, list[float]],
    top_k: float,
    boost: float,
    min_successes: int,
) -> tuple[int, dict[str, dict]]:
    """Add ``boost`` to every frame advantage of the top-K% (per garment)
    successful episodes, ranked by ``min_margin`` on the last frame's
    ``check_distances`` (stored as ``ep['last_check_distances']``).

    Mutates ``all_advantages`` in place. Episodes without a last-frame
    check_distances (e.g. older init datasets with no column) are skipped.

    Returns:
        (n_boosted_total, info_by_garment)
    where info_by_garment[garment] has ``n_success``, ``threshold``,
    ``n_boosted``, and optionally ``skipped`` reason.
    """
    if boost == 0.0 or top_k <= 0.0:
        return 0, {}

    by_garment: dict[str, list[tuple[int, float]]] = {}
    missing_cd = 0
    for i, ep in enumerate(all_episodes):
        if not ep.get("success"):
            continue
        cd = ep.get("last_check_distances")
        if cd is None:
            missing_cd += 1
            continue
        sd = garment_success_distances.get(ep.get("garment", ""))
        m = compute_min_margin(cd, ep.get("garment_type", ""), sd)
        if not np.isfinite(m):
            continue
        by_garment.setdefault(ep["garment"], []).append((i, m))

    if missing_cd:
        logger.warning(
            "Precision boost: %d successful episodes missing last_check_distances "
            "(likely old init datasets without the column) — skipped.",
            missing_cd,
        )

    info: dict[str, dict] = {}
    n_boosted_total = 0
    percentile = 100.0 * (1.0 - top_k)
    for garment, ep_list in by_garment.items():
        if len(ep_list) < min_successes:
            info[garment] = dict(
                n_success=len(ep_list), threshold=None, n_boosted=0,
                skipped=f"< {min_successes} successes",
            )
            continue
        margins = np.asarray([m for _, m in ep_list], dtype=float)
        thr = float(np.percentile(margins, percentile))
        boosted_eps = [i for i, m in ep_list if m >= thr]
        info[garment] = dict(
            n_success=len(ep_list), threshold=thr, n_boosted=len(boosted_eps),
        )
        for ep_i in boosted_eps:
            advs = all_advantages[ep_i]
            for t in range(len(advs)):
                advs[t] = float(advs[t]) + boost
        n_boosted_total += len(boosted_eps)

    return n_boosted_total, info
