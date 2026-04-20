#!/bin/bash
set -euo pipefail

# Installs Qwen3-TTS-Openai-Fastapi into ./qwen3-tts/ with its own uv venv
# for the `optimized` backend (torch.compile + CUDA graphs + flash-attn +
# real token-by-token PCM streaming).
#
# Requirements:
#   - Linux + NVIDIA GPU (CUDA 12.1-compatible driver, typically ≥535).
#   - uv  (https://astral.sh/uv)
#   - A detected package manager: apt / dnf / pacman / zypper
#   - ~10 GB free disk for torch+flash-attn+venv, plus ~1.2 GB for the 0.6B
#     model on first request.
#
# flash-attn build: the optimized backend works best with flash-attn 2. We
# try the prebuilt wheel first (`--only-binary=:all:`); if no matching wheel
# exists for the host's (python, torch, cuda) combo we fall back to a source
# build via `--no-build-isolation`. Either way the install is REQUIRED — the
# script exits non-zero if both paths fail. Pass `MAX_JOBS=2` (or similar)
# in the environment on memory-constrained hosts (<16 GB RAM) to keep the
# source build from OOMing.
#
# Idempotency:
#   We stamp ./qwen3-tts/.installed with the checked-out commit SHA. If the
#   stamp matches the pinned ref AND the venv + flash-attn import still
#   work, re-running is a no-op.

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/qwen3-tts"
STAMP_FILE="$INSTALL_DIR/.installed"
VENV_PY="$INSTALL_DIR/.venv/bin/python"

REPO_URL="https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git"
# Pinned to the upstream default branch; bump this deliberately after
# verifying upstream changes don't break the optimized backend config.
PINNED_REF="main"

OS_NAME="$(uname -s)"
if [ "$OS_NAME" != "Linux" ]; then
    echo "Error: qwen3-tts is a Linux-only runtime (got $OS_NAME)." >&2
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi -L &>/dev/null; then
    echo "Error: qwen3-tts needs an NVIDIA GPU but no nvidia-smi detected." >&2
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "Error: uv is required but not found in PATH." >&2
    echo "  Install: https://astral.sh/uv/install.sh" >&2
    exit 1
fi

# --- Detect Linux package manager ---

detect_linux_pkg_mgr() {
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case " ${ID:-} ${ID_LIKE:-} " in
            *" debian "*|*" ubuntu "*)              echo apt; return ;;
            *" fedora "*|*" rhel "*|*" centos "*)   echo dnf; return ;;
            *" arch "*|*" manjaro "*)               echo pacman; return ;;
            *" suse "*|*" opensuse "*|*" opensuse-leap "*|*" opensuse-tumbleweed "*) echo zypper; return ;;
        esac
    fi
    for mgr in apt-get dnf pacman zypper; do
        if command -v "$mgr" &>/dev/null; then
            case "$mgr" in apt-get) echo apt ;; *) echo "$mgr" ;; esac
            return
        fi
    done
}

install_system_deps() {
    local pkg_mgr
    pkg_mgr=$(detect_linux_pkg_mgr)
    if [ -z "$pkg_mgr" ]; then
        echo "Error: no supported package manager (apt/dnf/pacman/zypper) detected." >&2
        exit 1
    fi
    local sudo_cmd=""
    if [ "$(id -u)" != "0" ]; then
        if command -v sudo &>/dev/null; then
            sudo_cmd="sudo"
        else
            echo "Error: system deps require root or sudo (package manager: $pkg_mgr)." >&2
            exit 1
        fi
    fi

    echo "Installing system deps via $pkg_mgr (ffmpeg, libsndfile, libsox, sox, build tools)..."
    case "$pkg_mgr" in
        apt)
            $sudo_cmd apt-get update
            $sudo_cmd apt-get install -y \
                ffmpeg libsndfile1 libsox-dev sox \
                build-essential git curl
            ;;
        dnf)
            $sudo_cmd dnf install -y ffmpeg libsndfile sox sox-devel gcc gcc-c++ make git curl
            ;;
        pacman)
            $sudo_cmd pacman -S --needed --noconfirm ffmpeg libsndfile sox base-devel git curl
            ;;
        zypper)
            $sudo_cmd zypper install -y ffmpeg libsndfile1 sox sox-devel gcc gcc-c++ make git curl
            ;;
    esac
}

clone_or_update() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        echo "Fetching $PINNED_REF..."
        git -C "$INSTALL_DIR" fetch --depth 1 origin "$PINNED_REF"
        git -C "$INSTALL_DIR" reset --hard "origin/$PINNED_REF" 2>/dev/null \
            || git -C "$INSTALL_DIR" reset --hard "$PINNED_REF"
    else
        echo "Cloning $REPO_URL into $INSTALL_DIR..."
        git clone --depth 1 --branch "$PINNED_REF" "$REPO_URL" "$INSTALL_DIR"
    fi
}

current_repo_sha() {
    git -C "$INSTALL_DIR" rev-parse HEAD
}

# Small helper: check if a module is importable in the venv.
venv_has() {
    "$VENV_PY" -c "import $1" &>/dev/null
}

venv_pip() {
    (cd "$INSTALL_DIR" && VIRTUAL_ENV="$INSTALL_DIR/.venv" uv pip "$@")
}

# --- Idempotency fast-path ---

if [ -x "$VENV_PY" ] && [ -f "$STAMP_FILE" ] && [ -d "$INSTALL_DIR/.git" ]; then
    stamped=$(cat "$STAMP_FILE")
    head_sha=$(current_repo_sha)
    if [ "$stamped" = "$head_sha" ] && venv_has flash_attn; then
        echo "qwen3-tts already installed at $head_sha — nothing to do."
        exit 0
    fi
    echo "qwen3-tts install incomplete or stale; resuming from wherever we left off..."
fi

# --- Run (each step is individually idempotent so re-runs don't re-download) ---

install_system_deps
clone_or_update

# nvcc needs to be present before we reach the flash-attn source build
# (which takes 20-60 min and can OOM on <16GB RAM boxes — see MAX_JOBS note
# below). Probe common install paths before giving up so the error message
# is actionable.
if ! command -v nvcc &>/dev/null; then
    for p in /usr/local/cuda/bin /opt/cuda/bin; do
        if [ -x "$p/nvcc" ]; then
            export PATH="$p:$PATH"
            break
        fi
    done
fi
if ! command -v nvcc &>/dev/null; then
    echo "Error: nvcc (CUDA Toolkit) is required to build flash-attn from source." >&2
    echo "  Install: https://developer.nvidia.com/cuda-downloads" >&2
    echo "  After install, ensure nvcc is in PATH (typically /usr/local/cuda/bin)." >&2
    exit 1
fi

# --- Venv ---

if [ ! -x "$VENV_PY" ]; then
    echo "Creating uv venv (Python 3.12 — best flash-attn wheel coverage)..."
    rm -rf "$INSTALL_DIR/.venv"
    (cd "$INSTALL_DIR" && uv venv --python 3.12 .venv)
fi

# --- Torch + CUDA (cu121) ---
#
# pyproject.toml leaves torch unpinned, so `pip install -e .[api]` pulls the
# CPU torch from PyPI. Install the CUDA wheel first to pin the resolver.
if ! venv_has torch; then
    echo "Installing torch + torchaudio (cu121)..."
    venv_pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# --- flash-attn build deps ---
#
# flash-attn's setup.py imports torch to detect the CUDA version at build
# time, and shells out to ninja + nvcc to compile the kernels. `wheel`,
# `setuptools`, `ninja`, and `packaging` must be present in the venv when
# we run with --no-build-isolation (otherwise the build subprocess can't
# find them). Installing them up front so the source build can succeed.
echo "Ensuring flash-attn build deps are present (wheel, setuptools, ninja, packaging)..."
venv_pip install -U wheel setuptools ninja packaging

# --- qwen-tts with [api] extra ---

if ! venv_has qwen_tts; then
    echo "Installing qwen-tts with the [api] extra..."
    venv_pip install -e ".[api]"
fi

# --- flash-attn (required, source build) ---
#
# We skip the "try prebuilt first" dance because flash-attn publishes its
# prebuilt wheels on GitHub releases (not PyPI), and uv resolving against
# PyPI with --only-binary produces an unsatisfiable-deps error rather than
# falling back cleanly. Go straight to source — slow but reliable given
# the nvcc check above and pre-installed build deps.
if ! venv_has flash_attn; then
    echo
    echo "Building flash-attn from source. This usually takes 20-60 minutes."
    echo "  On hosts with <16GB RAM, re-run this script with MAX_JOBS=2 in env."
    echo
    if ! venv_pip install -U flash-attn --no-build-isolation; then
        echo "Error: flash-attn install failed." >&2
        echo "  Likely causes:" >&2
        echo "    - Not enough RAM for the source build (try MAX_JOBS=2)." >&2
        echo "    - Mismatched torch / CUDA versions in the venv." >&2
        echo "    - nvcc on a different CUDA version than the torch wheel." >&2
        echo "  The optimized backend requires flash-attn; setup cannot continue." >&2
        exit 1
    fi
fi

# --- Seed ~/qwen3-tts/config.yaml (convenience only) ---
#
# Our launcher sets TTS_CONFIG=<repo>/config.yaml, so this copy is only
# here for users running the repo's own start_server.sh directly.
if [ -f "$INSTALL_DIR/config.yaml" ] && [ ! -f "$HOME/qwen3-tts/config.yaml" ]; then
    mkdir -p "$HOME/qwen3-tts"
    cp "$INSTALL_DIR/config.yaml" "$HOME/qwen3-tts/config.yaml"
fi

current_repo_sha > "$STAMP_FILE"
echo "Done. qwen3-tts installed at $INSTALL_DIR ($(current_repo_sha))."
