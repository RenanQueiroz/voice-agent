#!/bin/bash
set -euo pipefail

# Installs Kokoro-FastAPI into ./kokoro-fastapi/ and prepares a uv-managed
# virtualenv under ./kokoro-fastapi/.venv/.
#
# Requirements:
#   - Linux (this runtime is darwin-filtered; the script refuses to run on macOS).
#   - uv  (https://astral.sh/uv)
#   - A detected package manager: apt / dnf / pacman / zypper
#
# GPU acceleration:
#   Kokoro-FastAPI's `gpu` extra pulls torch==2.8.0+cu129 from
#   download.pytorch.org/whl/cu129. No host nvcc required — only an NVIDIA
#   driver new enough for CUDA 12.9 (~driver 550+).
#
# Idempotency:
#   We stamp ./kokoro-fastapi/.installed with the checked-out commit SHA.
#   Re-running the script compares it against `git ls-remote HEAD`; when
#   they match AND the venv looks healthy we skip all work.

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/kokoro-fastapi"
STAMP_FILE="$INSTALL_DIR/.installed"
VENV_PY="$INSTALL_DIR/.venv/bin/python"

REPO_URL="https://github.com/remsky/Kokoro-FastAPI.git"
# Pinned tag — bump deliberately after testing upstream changes. The
# upstream `VERSION` / `pyproject.toml` may report a higher version
# than the latest cut tag (the maintainer doesn't always tag releases);
# use whatever the most recent `git ls-remote --tags` shows.
PINNED_REF="v0.2.4"

OS_NAME="$(uname -s)"
if [ "$OS_NAME" != "Linux" ]; then
    echo "Error: kokoro-fastapi is a Linux-only runtime (got $OS_NAME)." >&2
    echo "  On macOS, use the mlx-audio 'kokoro' TTS entry instead." >&2
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

    echo "Installing system deps via $pkg_mgr (espeak-ng, libsndfile, ffmpeg, g++, cmake)..."
    case "$pkg_mgr" in
        apt)
            $sudo_cmd apt-get update
            $sudo_cmd apt-get install -y espeak-ng espeak-ng-data libsndfile1 ffmpeg g++ cmake git curl
            ;;
        dnf)
            $sudo_cmd dnf install -y espeak-ng libsndfile ffmpeg gcc-c++ cmake git curl
            ;;
        pacman)
            $sudo_cmd pacman -S --needed --noconfirm espeak-ng libsndfile ffmpeg gcc cmake git curl
            ;;
        zypper)
            $sudo_cmd zypper install -y espeak-ng libsndfile1 ffmpeg gcc-c++ cmake git curl
            ;;
    esac

    # Ubuntu packages espeak-ng data under an arch-qualified path; phonemizer
    # looks for /usr/share/espeak-ng-data. Symlink if needed (harmless if
    # the target already exists).
    if [ ! -d /usr/share/espeak-ng-data ] || [ -z "$(ls -A /usr/share/espeak-ng-data 2>/dev/null || true)" ]; then
        local arch_data
        arch_data=$(find /usr/lib -maxdepth 3 -type d -name espeak-ng-data 2>/dev/null | head -n1 || true)
        if [ -n "$arch_data" ]; then
            $sudo_cmd mkdir -p /usr/share/espeak-ng-data
            $sudo_cmd sh -c "ln -sf '$arch_data'/* /usr/share/espeak-ng-data/"
        fi
    fi
}

# --- Clone / update the repo ---

clone_or_update() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        echo "Fetching $PINNED_REF..."
        git -C "$INSTALL_DIR" fetch --tags --depth 1 origin "$PINNED_REF"
        git -C "$INSTALL_DIR" checkout "$PINNED_REF"
    else
        echo "Cloning $REPO_URL@$PINNED_REF into $INSTALL_DIR..."
        git clone --depth 1 --branch "$PINNED_REF" "$REPO_URL" "$INSTALL_DIR"
    fi
}

# Small helpers so steps can self-skip when already satisfied.
venv_has() {
    "$VENV_PY" -c "import $1" &>/dev/null
}

venv_pip() {
    (cd "$INSTALL_DIR" && VIRTUAL_ENV="$INSTALL_DIR/.venv" uv pip "$@")
}

# --- Idempotency fast-path: everything done, nothing to do ---

if [ -x "$VENV_PY" ] && [ -f "$STAMP_FILE" ]; then
    current_ref=$(cat "$STAMP_FILE")
    if [ "$current_ref" = "$PINNED_REF" ] && venv_has kokoro; then
        echo "kokoro-fastapi already installed at $PINNED_REF — nothing to do."
        exit 0
    fi
    echo "kokoro-fastapi install incomplete or stale; resuming from wherever we left off..."
fi

# --- Run (each step is individually idempotent so retries don't re-download) ---

install_system_deps
clone_or_update

# --- Venv (keep on retry; we re-use whatever is already in it) ---

if [ ! -x "$VENV_PY" ]; then
    echo "Creating uv venv (Python 3.10)..."
    rm -rf "$INSTALL_DIR/.venv"
    (cd "$INSTALL_DIR" && uv venv --python 3.10 .venv)
fi

# --- Install the project into the venv ---
#
# The [gpu] extra pulls torch==2.8.0+cu129 (~3 GB). Skip the whole
# install step when the package is already importable so retries don't
# re-fetch it. A stamped ref mismatch still forces a full re-install
# via the repo checkout above, but that's deliberate.
if ! venv_has kokoro; then
    echo "Installing Kokoro-FastAPI with the gpu extra (pulls torch+cu129, several GB)..."
    venv_pip install -e ".[gpu]"
fi

# --- Download the Kokoro model weights ---
#
# download_model.py is itself idempotent (it has a verify_files() check
# that short-circuits when both files are already present), but running
# it unconditionally still spends a couple seconds doing that check. The
# presence of the .pth short-circuits here as well.
MODEL_PTH="$INSTALL_DIR/api/src/models/v1_0/kokoro-v1_0.pth"
if [ ! -f "$MODEL_PTH" ]; then
    echo "Downloading Kokoro model..."
    (cd "$INSTALL_DIR" && "$VENV_PY" docker/scripts/download_model.py --output api/src/models/v1_0)
fi

echo "$PINNED_REF" > "$STAMP_FILE"
echo "Done. kokoro-fastapi installed at $INSTALL_DIR ($PINNED_REF)."
