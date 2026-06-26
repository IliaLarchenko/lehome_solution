#!/usr/bin/env python3
"""Enrich a BC dataset with garment task labels (``--domain sim|real``).

Both official BC datasets ship with a generic task string ("fold the garment
on the table" / "Fold the Garment") and no per-episode garment label. The
training pipeline derives ``garment_type_id`` from the task name via
``garment_name_to_type()``, so the task labels must carry the garment.

The parquet rewrite is identical for both datasets; only the
episode → garment mapping differs:

* ``--domain sim`` (``data/lehome_challenge_merged/four_types_merged``):
  episodes are ordered by garment instance — 25 episodes per garment, garments
  listed in ``meta/garment_info.json`` insertion order. Tasks become the
  per-instance names (e.g. ``Top_Short_Seen_0``).

* ``--domain real`` (``data/lehome_real/four_types_merged``): the 500 episodes
  are the concatenation of the four per-garment subsets, 125 episodes each
  (verified by full-sequence episode-length match). Tasks become the 4 class names ``Top_Long / Top_Short /
  Pant_Long / Pant_Short`` with ``task_index = episode_index // 125``.

Rewrites: per-frame ``task_index`` in the data parquets, ``meta/tasks.parquet``,
the per-episode ``tasks`` column, and ``info.json`` ``total_tasks``.

Usage:
    uv run python scripts/enrich_garment_type.py --domain sim [--dry_run]
    uv run python scripts/enrich_garment_type.py --domain real [--dry_run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROOTS = {
    "sim": "data/lehome_challenge_merged/four_types_merged",
    "real": "data/lehome_real/four_types_merged",
}

# Real dataset: order matches the four_types_merged concatenation order —
# index = task_index. Prefix-matched by eval_utils.garment_name_to_type.
REAL_GARMENT_TASKS = ["Top_Long", "Top_Short", "Pant_Long", "Pant_Short"]
REAL_EPISODES_PER_GARMENT = 125


def build_mapping_sim(root: Path, info: dict) -> tuple[dict[int, str], list[str]]:
    """Episode → per-instance garment name, from garment_info.json order."""
    gi_path = root / "meta" / "garment_info.json"
    if not gi_path.exists():
        raise FileNotFoundError(f"No garment_info.json at {gi_path}")
    garment_info = json.loads(gi_path.read_text())

    ep_to_task: dict[int, str] = {}
    ep_idx = 0
    for garment_name, seeds in garment_info.items():
        for _ in range(len(seeds)):
            ep_to_task[ep_idx] = garment_name
            ep_idx += 1

    if len(ep_to_task) != info["total_episodes"]:
        raise RuntimeError(
            f"garment_info.json maps {len(ep_to_task)} episodes but dataset "
            f"has {info['total_episodes']}"
        )
    task_names = list(dict.fromkeys(garment_info.keys()))
    print(f"Mapped {len(ep_to_task)} episodes to {len(task_names)} garments")
    return ep_to_task, task_names


def build_mapping_real(root: Path, info: dict) -> tuple[dict[int, str], list[str]]:
    """Episode → garment class via episode_index // 125."""
    expected = REAL_EPISODES_PER_GARMENT * len(REAL_GARMENT_TASKS)
    if info["total_episodes"] != expected:
        raise RuntimeError(
            f"Expected {expected} episodes, dataset has {info['total_episodes']}. "
            f"Garment mapping assumption no longer holds — verify episode order."
        )
    ep_to_task = {
        ep: REAL_GARMENT_TASKS[ep // REAL_EPISODES_PER_GARMENT]
        for ep in range(expected)
    }
    print(f"Mapped {expected} episodes to {len(REAL_GARMENT_TASKS)} garment classes")
    return ep_to_task, list(REAL_GARMENT_TASKS)


def enrich(root: Path, ep_to_task: dict[int, str], task_names: list[str]) -> None:
    task_index = {name: i for i, name in enumerate(task_names)}

    # 1. Data parquets: rewrite per-frame task_index from episode_index.
    for pf in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(str(pf))
        new_task_indices = [
            task_index[ep_to_task[int(ep)]]
            for ep in table["episode_index"].to_pylist()
        ]
        idx = table.column_names.index("task_index")
        table = table.drop("task_index")
        table = table.add_column(idx, "task_index", pa.array(new_task_indices, type=pa.int64()))
        pq.write_table(table, str(pf))
        print(f"  data: {pf.relative_to(root)} -> {table.num_rows} rows rewritten")

    # 2. meta/tasks.parquet (index = task name, "task_index" column).
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(task_names)))},
        index=pd.Index(task_names, name="task"),
    )
    pq.write_table(
        pa.Table.from_pandas(tasks_df, preserve_index=True),
        root / "meta" / "tasks.parquet",
    )
    print(f"  meta: tasks.parquet -> {len(task_names)} tasks")

    # 3. meta/episodes/*.parquet: rewrite per-episode `tasks` column (list[str]).
    for pf in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        ep_df = pq.read_table(str(pf)).to_pandas()
        ep_df["tasks"] = [[ep_to_task[int(ep)]] for ep in ep_df["episode_index"]]
        pq.write_table(pa.Table.from_pandas(ep_df, preserve_index=False), str(pf))
        print(f"  meta: episodes/{pf.name} -> {len(ep_df)} episodes rewritten")

    # 4. info.json total_tasks.
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(task_names)
    info_path.write_text(json.dumps(info, indent=4))
    print(f"  meta: info.json total_tasks={len(task_names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich a BC dataset with garment task labels")
    parser.add_argument("--domain", required=True, choices=("sim", "real"))
    parser.add_argument("--dataset_root", default=None,
                        help="Defaults to the standard location for --domain")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root or DEFAULT_ROOTS[args.domain])
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    info = json.loads((root / "meta" / "info.json").read_text())

    build = build_mapping_sim if args.domain == "sim" else build_mapping_real
    ep_to_task, task_names = build(root, info)

    if args.dry_run:
        shown = set()
        for ep in sorted(ep_to_task):
            name = ep_to_task[ep]
            if name not in shown:
                shown.add(name)
                print(f"  ep {ep:>3d}+ -> task_index={task_names.index(name)} ({name})")
        return

    enrich(root, ep_to_task, task_names)
    print(f"\nDone. {len(ep_to_task)} episodes now carry garment task names.")


if __name__ == "__main__":
    main()
