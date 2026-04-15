#!/bin/bash
set -euo pipefail

# Builds whisper.cpp from source (with Metal acceleration) and downloads
# the specified GGML model plus the Silero VAD model into ./whispercpp/.
#
# Usage:
#   ./setup-whisper.sh [model-name]
#   e.g.: ./setup-whisper.sh large-v3-turbo-q5_0
#
# Supported models:
#   tiny, tiny.en, base, base.en, small, small.en, small.en-tdrz,
#   medium, medium.en, large-v1, large-v2, large-v2-q5_0,
#   large-v3, large-v3-q5_0, large-v3-turbo, large-v3-turbo-q5_0

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/whispercpp"
MODELS_DIR="$INSTALL_DIR/models"
SRC_DIR="$INSTALL_DIR/src"
REPO="https://github.com/ggml-org/whisper.cpp.git"

MODEL="${1:-large-v3-turbo}"
WHISPER_HF="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
VAD_HF="https://huggingface.co/ggml-org/whisper-vad/resolve/main"
VAD_MODEL="silero-v5.1.2"

SUPPORTED_MODELS=(
    tiny tiny.en base base.en small small.en small.en-tdrz
    medium medium.en large-v1 large-v2 large-v2-q5_0
    large-v3 large-v3-q5_0 large-v3-turbo large-v3-turbo-q5_0
)

# --- Validate model name ---

valid=0
for m in "${SUPPORTED_MODELS[@]}"; do
    if [ "$m" = "$MODEL" ]; then valid=1; break; fi
done
if [ "$valid" -eq 0 ]; then
    echo "Error: unsupported model '$MODEL'" >&2
    echo "Supported: ${SUPPORTED_MODELS[*]}" >&2
    exit 1
fi

mkdir -p "$INSTALL_DIR" "$MODELS_DIR"

UPDATED=0

# --- Build whisper.cpp ---

if [ -f "$INSTALL_DIR/whisper-server" ]; then
    echo "whisper-server binary already exists, skipping build."
else
    echo "Building whisper.cpp..."

    # Ensure cmake is available
    if ! command -v cmake &> /dev/null; then
        echo "cmake not found, installing via Homebrew..."
        if ! command -v brew &> /dev/null; then
            echo "Error: cmake is required but neither cmake nor Homebrew are installed." >&2
            echo "Install cmake manually: https://cmake.org/download/" >&2
            exit 1
        fi
        brew install cmake > /dev/null 2>&1
        echo "cmake installed."
    fi

    if [ -d "$SRC_DIR/.git" ]; then
        echo "Updating existing source..."
        git -C "$SRC_DIR" pull --ff-only 2>/dev/null || true
    else
        echo "Cloning whisper.cpp..."
        rm -rf "$SRC_DIR"
        git clone --depth 1 "$REPO" "$SRC_DIR"
    fi

    # Metal is enabled by default on macOS (GPU acceleration on Apple Silicon)
    echo "Compiling with Metal support..."
    cmake -B "$SRC_DIR/build" -S "$SRC_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SRC_DIR/build" -j --config Release

    echo "Copying binaries..."
    cp "$SRC_DIR/build/bin/whisper-server" "$INSTALL_DIR/"
    cp "$SRC_DIR/build/bin/whisper-cli" "$INSTALL_DIR/" 2>/dev/null || true

    echo "whisper.cpp built successfully."
    UPDATED=1
fi

echo

# --- Download GGML model ---

MODEL_FILE="$MODELS_DIR/ggml-${MODEL}.bin"
if [ -f "$MODEL_FILE" ]; then
    echo "Model ggml-${MODEL}.bin already exists, skipping download."
else
    echo "Downloading ggml-${MODEL}.bin..."
    curl -fL --progress-bar -o "$MODEL_FILE" "$WHISPER_HF/ggml-${MODEL}.bin"
    echo "Model downloaded."
    UPDATED=1
fi

echo

# --- Download VAD model ---

VAD_FILE="$MODELS_DIR/ggml-${VAD_MODEL}.bin"
if [ -f "$VAD_FILE" ]; then
    echo "VAD model ggml-${VAD_MODEL}.bin already exists, skipping download."
else
    echo "Downloading VAD model ggml-${VAD_MODEL}.bin..."
    curl -fL --progress-bar -o "$VAD_FILE" "$VAD_HF/ggml-${VAD_MODEL}.bin"
    echo "VAD model downloaded."
    UPDATED=1
fi

echo

if [ "$UPDATED" -eq 0 ]; then
    echo "Everything is already up to date."
else
    echo "Done. Installed to $INSTALL_DIR"
    echo "  Binary:    $INSTALL_DIR/whisper-server"
    echo "  Model:     $MODEL_FILE"
    echo "  VAD model: $VAD_FILE"
fi
