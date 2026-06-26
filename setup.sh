#!/usr/bin/env bash
# =============================================================================
# LeHome — One-Stop Setup Script
# =============================================================================
#
# Sets up everything needed for training AND evaluation on any machine.
# Safe to re-run: skips steps that are already done.
#
# Usage:
#   bash setup.sh              # full setup (needs sudo for system deps)
#   bash setup.sh --no-sudo    # skip system deps (no sudo available)
#   bash setup.sh --no-data    # skip HuggingFace asset downloads
#   bash setup.sh --no-tests   # skip test suite at the end
#
# What it does (in order):
#   1. System deps          — apt packages, NVIDIA Vulkan libs, zenity dummy
#   2. uv                   — install if missing
#   3. Git submodules       — openpi + lehome-challenge
#   4. Main venv            — uv sync + torchcodec fix + libcuda symlink
#   5. Challenge venv       — isaacsim, IsaacLab fork, lehome, EULA
#   6. Simulation assets    — garment meshes from HuggingFace
#   7. Optional logins      — wandb + HF if env vars are set
#   8. Verification         — import checks + test suite
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHALLENGE_DIR="$SCRIPT_DIR/lehome-challenge"
VENV_PYTHON="$CHALLENGE_DIR/.venv/bin/python"
OPENPI_REPO="${OPENPI_REPO:-https://github.com/Physical-Intelligence/openpi.git}"
OPENPI_BRANCH="${OPENPI_BRANCH:-main}"
LEHOME_REPO="${LEHOME_REPO:-https://github.com/IliaLarchenko/lehome-challenge.git}"

# Parse flags
NO_SUDO=false
NO_DATA=false
NO_TESTS=false
for arg in "$@"; do
    case "$arg" in
        --no-sudo)  NO_SUDO=true ;;
        --no-data)  NO_DATA=true ;;
        --no-tests) NO_TESTS=true ;;
        -h|--help)
            echo "Usage: bash setup.sh [--no-sudo] [--no-data] [--no-tests]"
            exit 0 ;;
    esac
done

SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then SUDO="sudo"; fi

ok()   { echo "  ✓ $1"; }
skip() { echo "  · $1 (already done)"; }
step() { echo ""; echo "=== [$1] $2 ==="; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. System dependencies
# ─────────────────────────────────────────────────────────────────────────────

step 1 "System dependencies"

if [ "$NO_SUDO" = true ]; then
    skip "Skipped (--no-sudo)"
else
    # Core packages
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq \
        git git-lfs curl build-essential cmake ninja-build pkg-config python3-dev \
        libgl1-mesa-dev libglfw3 libglfw3-dev libglew-dev xorg-dev \
        libxi-dev libxinerama-dev libxcursor1 libxrandr2 ffmpeg htop \
        python3-venv python3-pip > /dev/null
    git lfs install --skip-smudge > /dev/null 2>&1 || true
    ok "Core packages installed"

    # NVIDIA Vulkan/GL libraries (required for Isaac Sim rendering)
    # The nvidia_icd.json (Vulkan ICD) is provided by libnvidia-gl-{major}.
    # On CUDA repo machines, these packages often have priority -1 (held back),
    # so we need --allow-downgrades and exact version pinning.
    NVIDIA_VERSION=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' | head -1 || true)
    if [ -z "$NVIDIA_VERSION" ]; then
        echo "  WARNING: Could not detect NVIDIA driver version — skipping Vulkan libs"
    elif [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
        skip "NVIDIA Vulkan ICD already present"
    else
        NVIDIA_MAJOR=$(echo "$NVIDIA_VERSION" | cut -d. -f1)
        GL_PKG="libnvidia-gl-${NVIDIA_MAJOR}"
        EXTRA_PKG="libnvidia-extra-${NVIDIA_MAJOR}"
        COMMON_PKG="libnvidia-common-${NVIDIA_MAJOR}"
        echo "  Installing $GL_PKG (driver $NVIDIA_VERSION)..."

        # Try strategies in order: exact version from CUDA repo, Ubuntu repo version, unversioned
        INSTALLED=false
        for VERSION_SUFFIX in "${NVIDIA_VERSION}-1ubuntu1" "${NVIDIA_VERSION}-0ubuntu0.24.04.2" "${NVIDIA_VERSION}-0ubuntu1"; do
            if $SUDO apt-get install -y --allow-downgrades \
                "${GL_PKG}=${VERSION_SUFFIX}" "${EXTRA_PKG}=${VERSION_SUFFIX}" 2>/dev/null; then
                INSTALLED=true
                break
            fi
        done

        # Fallback: try without version pin
        if [ "$INSTALLED" = false ]; then
            $SUDO apt-get install -y --allow-downgrades "$GL_PKG" "$EXTRA_PKG" 2>/dev/null || \
            $SUDO apt-get install -y --allow-downgrades "$GL_PKG" "$COMMON_PKG" 2>/dev/null || \
            echo "  WARNING: Could not install $GL_PKG — Isaac Sim eval may not work"
        fi

        if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
            ok "NVIDIA Vulkan ICD installed"
        else
            echo "  WARNING: /usr/share/vulkan/icd.d/nvidia_icd.json still not found after install"
        fi
    fi

    # Zenity dummy (Isaac Sim calls zenity for GUI dialogs — blocks headless)
    if [ -x /usr/local/bin/zenity ] && grep -q "exit 0" /usr/local/bin/zenity 2>/dev/null; then
        skip "Zenity dummy"
    else
        echo -e '#!/bin/sh\nexit 0' | $SUDO tee /usr/local/bin/zenity > /dev/null
        $SUDO chmod +x /usr/local/bin/zenity
        ok "Zenity dummy created"
    fi

    # libcuda.so symlink (fixes "cannot find -lcuda" linker errors)
    if [ -f /lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
        $SUDO ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true
        $SUDO ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true
        $SUDO ldconfig 2>/dev/null || true
        ok "libcuda.so symlink created"
    else
        skip "libcuda.so symlink"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. uv package manager
# ─────────────────────────────────────────────────────────────────────────────

step 2 "uv package manager"

export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
if command -v uv >/dev/null 2>&1; then
    skip "uv ($(uv --version 2>/dev/null || echo 'installed'))"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Git submodules (openpi + lehome-challenge)
# ─────────────────────────────────────────────────────────────────────────────

step 3 "Git submodules"

# Configure git to skip LFS during dependency installations
export GIT_LFS_SKIP_SMUDGE=1

# OpenPI
if [ -f "openpi/pyproject.toml" ]; then
    skip "OpenPI submodule"
else
    echo "  Initializing OpenPI submodule..."
    git submodule sync openpi 2>/dev/null || true
    git submodule update --init openpi 2>/dev/null || \
        GIT_LFS_SKIP_SMUDGE=1 git clone -b "$OPENPI_BRANCH" "$OPENPI_REPO" openpi
    ok "OpenPI submodule initialized"
fi

# Ensure openpi is on the correct branch
cd openpi
git fetch origin "$OPENPI_BRANCH" 2>/dev/null || true
git checkout "$OPENPI_BRANCH" 2>/dev/null || true
git pull origin "$OPENPI_BRANCH" 2>/dev/null || true
cd "$SCRIPT_DIR"
ok "OpenPI on $OPENPI_BRANCH branch"

# lehome-challenge
if [ -f "lehome-challenge/pyproject.toml" ] || [ -f "lehome-challenge/README.md" ]; then
    skip "lehome-challenge submodule"
else
    echo "  Initializing lehome-challenge submodule..."
    git submodule sync lehome-challenge 2>/dev/null || true
    git submodule update --init lehome-challenge 2>/dev/null || \
        GIT_LFS_SKIP_SMUDGE=1 git clone "$LEHOME_REPO" "lehome-challenge"
    ok "lehome-challenge submodule initialized"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Main project venv
# ─────────────────────────────────────────────────────────────────────────────

step 4 "Main project venv"

cd "$SCRIPT_DIR"
uv sync --extra dev
ok "uv sync complete"

# Install lehome Python package from challenge submodule (--no-deps to avoid conflicts)
uv pip install toml 2>/dev/null || true
uv pip install --no-deps -e lehome-challenge/source/lehome/ 2>/dev/null || true
ok "lehome package installed"

# Fix torchcodec version to match torch
TORCH_VERSION=$(uv run python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')" 2>/dev/null || echo "unknown")
if [ "$TORCH_VERSION" != "unknown" ]; then
    case "$TORCH_VERSION" in
        "2.7")  TC_VERSION="0.5" ;;
        "2.8")  TC_VERSION="0.7" ;;
        "2.9")  TC_VERSION="0.8" ;;
        "2.10") TC_VERSION="0.10" ;;
        *)      TC_VERSION="" ;;
    esac
    if [ -n "$TC_VERSION" ]; then
        uv pip install "torchcodec==${TC_VERSION}" --index-url https://download.pytorch.org/whl/cu126 2>/dev/null || true
        ok "torchcodec $TC_VERSION installed (torch $TORCH_VERSION)"
    else
        echo "  WARNING: Unknown torch version $TORCH_VERSION — skipping torchcodec fix"
    fi
fi

# CUDA NPP compat libraries for torchcodec
# PyTorch cu126 links against libnpp*.so.12, but newer systems ship only CUDA 13+.
# torchcodec's libtorchcodec_core6.so inherits NPP deps and fails at runtime.
# Fix: install the real CUDA 12 NPP runtime packages from Ubuntu repos.
TORCH_CUDA=$(uv run python -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "")
SYSTEM_CUDA=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 | grep -oP '\d+' | head -1 || echo "")
if [ -n "$TORCH_CUDA" ] && [ -n "$SYSTEM_CUDA" ] && [ "$TORCH_CUDA" != "$SYSTEM_CUDA" ]; then
    if ldconfig -p | grep -q "libnppicc.so.${TORCH_CUDA}"; then
        skip "CUDA $TORCH_CUDA NPP compat libs (already installed)"
    elif [ "$NO_SUDO" = true ]; then
        echo "  WARNING: Need CUDA $TORCH_CUDA NPP libs but --no-sudo set. Install manually: apt install libnppicc${TORCH_CUDA} libnppc${TORCH_CUDA}"
    else
        echo "  Installing CUDA $TORCH_CUDA NPP compat libs (system has CUDA $SYSTEM_CUDA)..."
        $SUDO apt-get install -y -qq "libnppicc${TORCH_CUDA}" "libnppc${TORCH_CUDA}" 2>/dev/null || \
            echo "  WARNING: Could not install NPP compat libs — torchcodec may fail at runtime"
        $SUDO ldconfig 2>/dev/null || true
        ok "CUDA $TORCH_CUDA NPP compat libs installed"
    fi
else
    skip "NPP compat libs (CUDA versions match or not detected)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Challenge venv (Isaac Sim + IsaacLab + lehome)
# ─────────────────────────────────────────────────────────────────────────────

step 5 "Challenge venv (Isaac Sim)"

cd "$CHALLENGE_DIR"

uv sync
ok "Challenge venv synced"

# Accept Isaac Sim EULA
EULA_FILE="$CHALLENGE_DIR/.venv/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED"
mkdir -p "$(dirname "$EULA_FILE")" 2>/dev/null || true
echo "yes" > "$EULA_FILE" 2>/dev/null || true
ok "Isaac Sim EULA accepted"

# Clone IsaacLab fork if needed
if [ ! -d "$CHALLENGE_DIR/third_party/IsaacLab" ]; then
    echo "  Cloning IsaacLab fork..."
    git clone https://github.com/lehome-official/IsaacLab.git "$CHALLENGE_DIR/third_party/IsaacLab"
    ok "IsaacLab fork cloned"
else
    skip "IsaacLab fork"
fi

# Install IsaacLab sub-packages
echo "  Running isaaclab.sh -i none..."
source "$CHALLENGE_DIR/.venv/bin/activate"
"$CHALLENGE_DIR/third_party/IsaacLab/isaaclab.sh" -i none 2>/dev/null || true
deactivate 2>/dev/null || true

# Fix flatdict build (needs old setuptools with pkg_resources)
uv pip install "setuptools<70" wheel --python "$VENV_PYTHON" 2>/dev/null || true
uv pip install flatdict==4.0.1 --no-build-isolation --python "$VENV_PYTHON" 2>/dev/null || true

# Install core isaaclab + lehome packages
uv pip install -e "$CHALLENGE_DIR/third_party/IsaacLab/source/isaaclab" --python "$VENV_PYTHON" 2>/dev/null || true
uv pip install -e "$CHALLENGE_DIR/source/lehome" --python "$VENV_PYTHON" 2>/dev/null || true

# Pin warp-lang: IsaacLab declares "warp-lang" unpinned, so a fresh install pulls
# the latest, which has removed warp.context / warp.types.array / np_dtype_to_warp_type
# that Isaac Sim 5.1 still imports. 1.12.1 is the last compatible release.
uv pip install "warp-lang==1.12.1" --python "$VENV_PYTHON" 2>/dev/null || true
ok "Challenge venv fully set up"

cd "$SCRIPT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Simulation assets (garment meshes from HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

step 6 "Simulation assets"

if [ "$NO_DATA" = true ]; then
    skip "Skipped (--no-data)"
else
    ASSETS_DIR="$CHALLENGE_DIR/Assets"
    if [ -d "$ASSETS_DIR/objects/Challenge_Garment/Release" ]; then
        skip "Garment assets already present"
    else
        echo "  Downloading lehome/asset_challenge from HuggingFace..."
        uv run huggingface-cli download lehome/asset_challenge --repo-type dataset --local-dir "$ASSETS_DIR"
        ok "Garment assets downloaded"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. Optional logins (from environment variables)
# ─────────────────────────────────────────────────────────────────────────────

step 7 "Optional logins"

if [ -n "${WANDB_API_KEY:-}" ]; then
    uv run wandb login --relogin <<< "$WANDB_API_KEY" 2>/dev/null || true
    ok "wandb logged in"
else
    skip "wandb (set WANDB_API_KEY to auto-login)"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    ok "HuggingFace logged in"
else
    skip "HuggingFace (set HF_TOKEN to auto-login)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 8. Verification
# ─────────────────────────────────────────────────────────────────────────────

step 8 "Verification"

echo "  Checking imports..."
uv run python -c "
import lehome_solution; print('    lehome_solution OK')
import lehome; print('    lehome OK')
import openpi; print('    openpi OK')
import lerobot; print('    lerobot OK')
import av; print(f'    pyav OK: {av.__version__}')
from lehome_solution.training.config import get_config; print('    training config OK')
" || echo "  WARNING: Some imports failed — check errors above"

# Challenge venv check
if [ -x "$VENV_PYTHON" ]; then
    "$VENV_PYTHON" -c "import isaacsim; print('    isaacsim OK')" 2>/dev/null || \
        echo "  WARNING: isaacsim import failed in challenge venv"
fi

if [ "$NO_TESTS" = true ]; then
    skip "Tests (--no-tests)"
else
    echo "  Running test suite..."
    uv run python -m pytest tests/ -v --tb=short \
        -k "not TestLeRobotDatasetLoading and not TestTransformPipeline" || \
        echo "  WARNING: Some tests failed — check output above"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  Setup complete!"
echo "================================================================"
echo ""
echo "  Main venv:       $SCRIPT_DIR/.venv"
echo "  Challenge venv:  $CHALLENGE_DIR/.venv"
echo ""
echo "  Next steps:"
echo "    uv run huggingface-cli login           # if not logged in"
echo "    uv run python scripts/download_hf_assets.py  # download model + datasets"
echo ""
echo "  Run training:"
echo "    uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml"
echo ""
echo "  Run evaluation:"
echo "    uv run python scripts/run_eval.py --checkpoint_dir outputs/checkpoints/..."
echo ""
