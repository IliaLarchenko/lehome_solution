"""Shared eval utilities: garment mapping, garment list, Isaac Sim env.

Used by run_eval.py, eval_worker.py, and metadata.
Video saving is always done on the eval (client) side; the server never saves video.
"""

from __future__ import annotations

from pathlib import Path

from lehome_solution.constants import GARMENT_TYPE_PREFIX, GARMENT_TYPES

ALL_GARMENT_TYPES = list(GARMENT_TYPES)


def garment_name_to_type(garment_name: str) -> str:
    """Map garment name (e.g. Top_Long_Unseen_0) to garment_type (e.g. top_long)."""
    g = (garment_name or "").strip()
    if g.startswith("Top_Long"):
        return "top_long"
    if g.startswith("Top_Short"):
        return "top_short"
    if g.startswith("Pant_Long"):
        return "pant_long"
    if g.startswith("Pant_Short"):
        return "pant_short"
    return "unknown"


def get_garments(
    repo_root: Path | str,
    garment_type: str,
    subset: str = "unseen",
) -> list[str]:
    """Get garment names by scanning subdirectories (crash-safe, never reads stale list files).

    Workers of the same garment_type share one list file and serialize via file lock;
    there is no collision (no corrupt state), only ordered access.
    """
    repo_root = Path(repo_root)
    prefix = GARMENT_TYPE_PREFIX.get(garment_type, garment_type)
    garment_dir = (
        repo_root
        / "lehome-challenge"
        / "Assets"
        / "objects"
        / "Challenge_Garment"
        / "Release"
        / prefix
    )
    if not garment_dir.exists():
        return []
    all_garments = sorted(d.name for d in garment_dir.iterdir() if d.is_dir())
    if subset == "unseen":
        return [g for g in all_garments if "Unseen" in g]
    if subset == "seen":
        return [g for g in all_garments if "Seen" in g]
    return all_garments


def ensure_isaacsim_env(repo_root: Path | str | None = None) -> None:
    """Pre-accept Isaac Sim EULA and set Vulkan ICD so headless/CI runs don't block.

    Call once at eval startup. Does not touch server; video saving is always client-side.
    """
    import os
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

    if repo_root is None:
        return
    repo_root = Path(repo_root)
    eula_path = (
        repo_root
        / "lehome-challenge"
        / ".venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "isaacsim"
        / "kit"
        / "EULA_ACCEPTED"
    )
    if eula_path.parent.exists() and not eula_path.exists():
        eula_path.write_text("yes\n")
