#!/bin/bash
set -euo pipefail

# Downloads and extracts the latest llama.cpp prebuilt release for the
# current OS / architecture into ./llamacpp, overwriting existing files
# there. Skips the download when the installed version already matches
# the latest.
#
# Platform selection:
#   macOS arm64    → llama-{tag}-bin-macos-arm64.tar.gz
#   Linux x86_64   → llama-{tag}-bin-ubuntu-x64.zip (CPU)
#                    or CUDA build if an NVIDIA GPU is detected
#   Linux aarch64  → llama-{tag}-bin-ubuntu-arm64.zip
#
# CUDA detection is a simple `nvidia-smi` probe; we try the CUDA build
# first and fall back to the CPU build if the exact asset 404s, so the
# script stays useful even when the CUDA asset naming changes upstream.

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/llamacpp"
TMPFILE=$(mktemp /tmp/llamacpp-XXXXXX)
trap 'rm -f "$TMPFILE"' EXIT

UPDATED=0

LLAMA_REPO="ggml-org/llama.cpp"

# --- Detect platform ---

OS_NAME="$(uname -s)"
ARCH="$(uname -m)"

case "$OS_NAME" in
    Darwin)
        if [ "$ARCH" != "arm64" ]; then
            echo "Error: only Apple Silicon (arm64) is supported on macOS, detected $ARCH." >&2
            exit 1
        fi
        ;;
    Linux)
        case "$ARCH" in
            x86_64|aarch64) ;;
            *)
                echo "Error: unsupported Linux architecture $ARCH (expected x86_64 or aarch64)." >&2
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Error: unsupported OS $OS_NAME." >&2
        exit 1
        ;;
esac

# --- Find the latest release tag ---

echo "Checking llama.cpp..."
LLAMA_TAG=$(curl -sf "https://api.github.com/repos/$LLAMA_REPO/releases/latest" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")

if [ -z "$LLAMA_TAG" ]; then
    echo "Error: could not determine latest llama.cpp release tag" >&2
    exit 1
fi

# Tag is e.g. "b8763"; local version reports "8763".
LLAMA_LATEST="${LLAMA_TAG#b}"
LLAMA_LOCAL=$("$INSTALL_DIR/llama-cli" --version 2>&1 | grep -o 'version: [0-9]*' | cut -d' ' -f2 || echo "")

if [ "$LLAMA_LOCAL" = "$LLAMA_LATEST" ]; then
    echo "llama.cpp is already up to date ($LLAMA_TAG)."
else
    echo "Updating llama.cpp: ${LLAMA_LOCAL:-not installed} -> $LLAMA_LATEST"

    # --- Pick the asset for this platform ---

    CUDA_AVAILABLE=0
    if [ "$OS_NAME" = "Linux" ] && [ "$ARCH" = "x86_64" ] && command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            CUDA_AVAILABLE=1
        fi
    fi

    # Candidate list: each entry is "<asset-filename>". The first entry that
    # downloads successfully wins. On Linux+CUDA we try a CUDA build first;
    # otherwise we go straight to the CPU/Apple-Silicon asset.
    CANDIDATES=()
    case "$OS_NAME" in
        Darwin)
            CANDIDATES+=("llama-${LLAMA_TAG}-bin-macos-arm64.tar.gz")
            ;;
        Linux)
            if [ "$ARCH" = "x86_64" ]; then
                if [ "$CUDA_AVAILABLE" = "1" ]; then
                    # Asset names in the llama.cpp releases change over time
                    # (e.g. `-cuda-cu12.x-x64.zip`). Probe a couple of common
                    # variants before falling back to the CPU build.
                    CANDIDATES+=("llama-${LLAMA_TAG}-bin-ubuntu-x64-cuda.zip")
                    CANDIDATES+=("llama-${LLAMA_TAG}-bin-ubuntu-cuda-cu12.4-x64.zip")
                    CANDIDATES+=("llama-${LLAMA_TAG}-bin-ubuntu-cuda-cu12-x64.zip")
                fi
                CANDIDATES+=("llama-${LLAMA_TAG}-bin-ubuntu-x64.zip")
            else
                # Linux arm64 (e.g. Raspberry Pi 5, ARM servers)
                CANDIDATES+=("llama-${LLAMA_TAG}-bin-ubuntu-arm64.zip")
            fi
            ;;
    esac

    ASSET=""
    for candidate in "${CANDIDATES[@]}"; do
        url="https://github.com/$LLAMA_REPO/releases/download/$LLAMA_TAG/$candidate"
        echo "Trying $candidate..."
        if curl -fsSL --progress-bar -o "$TMPFILE" "$url"; then
            ASSET="$candidate"
            break
        fi
        echo "  not found, trying next candidate."
    done

    if [ -z "$ASSET" ]; then
        echo "Error: no matching llama.cpp asset found for $OS_NAME $ARCH (tag $LLAMA_TAG)." >&2
        echo "  See https://github.com/$LLAMA_REPO/releases/tag/$LLAMA_TAG for the current asset names." >&2
        exit 1
    fi

    echo "Extracting $ASSET to $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    case "$ASSET" in
        *.tar.gz)
            tar -xzf "$TMPFILE" --strip-components=1 -C "$INSTALL_DIR"
            ;;
        *.zip)
            if ! command -v unzip &> /dev/null; then
                echo "Error: unzip is required to extract $ASSET but is not installed." >&2
                echo "  Install it with your package manager (apt install unzip / dnf install unzip / pacman -S unzip)." >&2
                exit 1
            fi
            TMPDIR=$(mktemp -d)
            unzip -q "$TMPFILE" -d "$TMPDIR"
            # llama.cpp zips typically contain a `build/bin/…` layout. Find
            # the directory holding `llama-server` and copy its contents to
            # the install dir so we match the tar.gz strip-components=1 shape.
            SERVER_BIN="$(find "$TMPDIR" -type f -name 'llama-server' -print -quit)"
            if [ -z "$SERVER_BIN" ]; then
                echo "Error: llama-server not found inside $ASSET." >&2
                rm -rf "$TMPDIR"
                exit 1
            fi
            BIN_DIR="$(dirname "$SERVER_BIN")"
            cp -R "$BIN_DIR/." "$INSTALL_DIR/"
            # Copy shared libs alongside the binary (CUDA builds ship libggml-cuda.so etc.)
            LIB_DIR="$(dirname "$BIN_DIR")/lib"
            if [ -d "$LIB_DIR" ]; then
                cp -R "$LIB_DIR/." "$INSTALL_DIR/"
            fi
            rm -rf "$TMPDIR"
            ;;
        *)
            echo "Error: unrecognized archive extension for $ASSET." >&2
            exit 1
            ;;
    esac

    echo "llama.cpp $LLAMA_TAG installed ($ASSET)."
    UPDATED=1
fi

echo

if [ "$UPDATED" -eq 0 ]; then
    echo "Everything is already up to date."
else
    echo "Done. Binaries updated in $INSTALL_DIR"
fi
