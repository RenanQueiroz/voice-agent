#!/bin/bash
set -euo pipefail

# Downloads and extracts the latest llama.cpp prebuilt release
# for Apple Silicon into ./llama.cpp, overwriting any existing files there.
# Skips downloading if the installed version already matches the latest.

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/llamacpp"
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

if [ "$UPDATED" -eq 0 ]; then
    echo "Everything is already up to date."
else
    echo "Done. Binaries updated in $INSTALL_DIR"
fi