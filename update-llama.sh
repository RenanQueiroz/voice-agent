#!/bin/bash
set -euo pipefail

# Downloads and extracts the latest llama.cpp prebuilt release
# for Apple Silicon into ./llama.cpp, overwriting any existing files there.
# Skips downloading if the installed version already matches the latest.
#
# Previously also updated llama-swap:
# Downloads and extracts the latest llama.cpp and llama-swap prebuilt releases

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/llama.cpp"
TMPFILE=$(mktemp /tmp/llamacpp-XXXXXX.tar.gz)
trap 'rm -f "$TMPFILE"' EXIT

UPDATED=0

# --- llama.cpp ---

LLAMA_REPO="ggml-org/llama.cpp"

echo "Checking llama.cpp..."
LLAMA_TAG=$(curl -sf "https://api.github.com/repos/$LLAMA_REPO/releases/latest" \
    | grep '"tag_name"' | head -1 | cut -d'"' -f4)

if [ -z "$LLAMA_TAG" ]; then
    echo "Error: could not determine latest llama.cpp release tag" >&2
    exit 1
fi

# Tag is e.g. "b8763", local version reports "8763"
LLAMA_LATEST="${LLAMA_TAG#b}"
LLAMA_LOCAL=$("$INSTALL_DIR/llama-cli" --version 2>&1 | grep -o 'version: [0-9]*' | cut -d' ' -f2 || echo "")

if [ "$LLAMA_LOCAL" = "$LLAMA_LATEST" ]; then
    echo "llama.cpp is already up to date ($LLAMA_TAG)."
else
    echo "Updating llama.cpp: ${LLAMA_LOCAL:-not installed} -> $LLAMA_LATEST"
    LLAMA_ASSET="llama-${LLAMA_TAG}-bin-macos-arm64.tar.gz"
    LLAMA_URL="https://github.com/$LLAMA_REPO/releases/download/$LLAMA_TAG/$LLAMA_ASSET"

    echo "Downloading $LLAMA_ASSET..."
    curl -fL --progress-bar -o "$TMPFILE" "$LLAMA_URL"

    echo "Extracting to $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    tar -xzf "$TMPFILE" --strip-components=1 -C "$INSTALL_DIR"

    echo "llama.cpp $LLAMA_TAG installed."
    UPDATED=1
fi

echo

# --- llama-swap ---
# llama-swap is disabled for now. Uncomment this block to include it again.
#
# SWAP_REPO="mostlygeek/llama-swap"
#
# echo "Checking llama-swap..."
# SWAP_TAG=$(curl -sf "https://api.github.com/repos/$SWAP_REPO/releases/latest" \
#     | grep '"tag_name"' | head -1 | cut -d'"' -f4)
#
# if [ -z "$SWAP_TAG" ]; then
#     echo "Error: could not determine latest llama-swap release tag" >&2
#     exit 1
# fi
#
# # Tag is e.g. "v200", local version reports "200"
# SWAP_LATEST="${SWAP_TAG#v}"
# SWAP_LOCAL=$("$INSTALL_DIR/llama-swap" --version 2>&1 | grep -o 'version: [0-9]*' | cut -d' ' -f2 || echo "")
#
# if [ "$SWAP_LOCAL" = "$SWAP_LATEST" ]; then
#     echo "llama-swap is already up to date ($SWAP_TAG)."
# else
#     echo "Updating llama-swap: ${SWAP_LOCAL:-not installed} -> $SWAP_LATEST"
#     SWAP_ASSET="llama-swap_${SWAP_LATEST}_darwin_arm64.tar.gz"
#     SWAP_URL="https://github.com/$SWAP_REPO/releases/download/$SWAP_TAG/$SWAP_ASSET"
#
#     echo "Downloading $SWAP_ASSET..."
#     curl -fL --progress-bar -o "$TMPFILE" "$SWAP_URL"
#
#     echo "Extracting to $INSTALL_DIR..."
#     mkdir -p "$INSTALL_DIR"
#     tar -xzf "$TMPFILE" --exclude='README.md' --exclude='LICENSE.md' -C "$INSTALL_DIR"
#
#     echo "llama-swap $SWAP_TAG installed."
#     UPDATED=1
# fi
#
# echo

if [ "$UPDATED" -eq 0 ]; then
    echo "Everything is already up to date."
else
    echo "Done. Binaries updated in $INSTALL_DIR"
fi
