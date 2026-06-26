#!/usr/bin/env python3
"""Download Hugging Face datasets to the folders the project configs expect.

Targets:
  - lehome/dataset_challenge_merged (four_types_merged subfolder only)
      -> data/lehome_challenge_merged/four_types_merged
  - lehome/dataset_challenge_real (four_types_merged subfolder only)
      -> data/lehome_real/four_types_merged
      (Real-robot teleop. Same task layout as the sim BC dataset but with
       different camera names.)

Usage:
  uv run python scripts/download_hf_assets.py [--four-types] [--real-four-types]
  uv run python scripts/download_hf_assets.py   # download both
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument(
        "--four-types",
        action="store_true",
        help="Download four_types_merged from dataset_challenge_merged to data/lehome_challenge_merged",
    )
    parser.add_argument(
        "--real-four-types",
        action="store_true",
        help="Download four_types_merged from dataset_challenge_real to data/lehome_real",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download both (default if no other option is set)",
    )
    args = parser.parse_args()

    do_all = args.all or not (args.four_types or args.real_four_types)

    if do_all or args.four_types:
        print(
            "Downloading lehome/dataset_challenge_merged (four_types_merged) -> data/lehome_challenge_merged ..."
        )
        Path("data/lehome_challenge_merged").mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="lehome/dataset_challenge_merged",
            repo_type="dataset",
            local_dir="data/lehome_challenge_merged",
            allow_patterns=["four_types_merged/**"],
        )
        print("  done.")

    if do_all or args.real_four_types:
        print(
            "Downloading lehome/dataset_challenge_real (four_types_merged) -> data/lehome_real ..."
        )
        Path("data/lehome_real").mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="lehome/dataset_challenge_real",
            repo_type="dataset",
            local_dir="data/lehome_real",
            allow_patterns=["four_types_merged/**"],
        )
        print("  done.")

    print("Download(s) complete.")


if __name__ == "__main__":
    main()
