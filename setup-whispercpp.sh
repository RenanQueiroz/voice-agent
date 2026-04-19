#!/bin/bash
set -euo pipefail

# Builds whisper.cpp from source and downloads the specified GGML model
# plus the Silero VAD model into ./whispercpp/.
#
# GPU acceleration:
#   macOS   → Metal (auto-enabled by whisper.cpp's cmake).
#   Linux   → CUDA if `nvidia-smi` is present (`-DGGML_CUDA=ON`), else CPU.
#
# Usage:
#   ./setup-whispercpp.sh [model-name]
#   e.g.: ./setup-whispercpp.sh large-v3-turbo-q5_0
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

OS_NAME="$(uname -s)"

# --- Install ffmpeg if missing ---
#
# whisper-server is launched with `--convert`, which shells out to ffmpeg to
# transcode incoming audio. Without ffmpeg on PATH, the server logs
# "ffmpeg is not found." and never binds to its port — every health probe
# times out and the role-start flow fails opaquely.

install_ffmpeg() {
    case "$OS_NAME" in
        Darwin)
            if ! command -v brew &> /dev/null; then
                echo "Error: ffmpeg is required but neither ffmpeg nor Homebrew are installed." >&2
                echo "Install ffmpeg manually: https://ffmpeg.org/download.html" >&2
                exit 1
            fi
            echo "ffmpeg not found, installing via Homebrew..."
            brew install ffmpeg > /dev/null 2>&1
            ;;
        Linux)
            if command -v apt-get &> /dev/null; then
                echo "ffmpeg not found, installing via apt..."
                sudo apt-get update -qq
                sudo apt-get install -y ffmpeg
            elif command -v dnf &> /dev/null; then
                echo "ffmpeg not found, installing via dnf..."
                sudo dnf install -y ffmpeg
            elif command -v pacman &> /dev/null; then
                echo "ffmpeg not found, installing via pacman..."
                sudo pacman -S --noconfirm ffmpeg
            elif command -v zypper &> /dev/null; then
                echo "ffmpeg not found, installing via zypper..."
                sudo zypper install -y ffmpeg
            else
                echo "Error: ffmpeg is required but no supported package manager was found." >&2
                echo "Install ffmpeg manually: https://ffmpeg.org/download.html" >&2
                exit 1
            fi
            ;;
        *)
            echo "Error: ffmpeg is required but this script doesn't know how to install it on $OS_NAME." >&2
            exit 1
            ;;
    esac
    echo "ffmpeg installed."
}

# --- Install cmake if missing ---

install_cmake() {
    case "$OS_NAME" in
        Darwin)
            if ! command -v brew &> /dev/null; then
                echo "Error: cmake is required but neither cmake nor Homebrew are installed." >&2
                echo "Install cmake manually: https://cmake.org/download/" >&2
                exit 1
            fi
            echo "cmake not found, installing via Homebrew..."
            brew install cmake > /dev/null 2>&1
            ;;
        Linux)
            if command -v apt-get &> /dev/null; then
                echo "cmake not found, installing via apt..."
                sudo apt-get update -qq
                sudo apt-get install -y cmake
            elif command -v dnf &> /dev/null; then
                echo "cmake not found, installing via dnf..."
                sudo dnf install -y cmake
            elif command -v pacman &> /dev/null; then
                echo "cmake not found, installing via pacman..."
                sudo pacman -S --noconfirm cmake
            elif command -v zypper &> /dev/null; then
                echo "cmake not found, installing via zypper..."
                sudo zypper install -y cmake
            else
                echo "Error: cmake is required but no supported package manager was found." >&2
                echo "Install cmake manually: https://cmake.org/download/" >&2
                exit 1
            fi
            ;;
        *)
            echo "Error: cmake is required but this script doesn't know how to install it on $OS_NAME." >&2
            exit 1
            ;;
    esac
    echo "cmake installed."
}

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

# --- Ensure ffmpeg is installed (required at runtime by whisper-server --convert) ---

if ! command -v ffmpeg &> /dev/null; then
    install_ffmpeg
fi

# --- Build whisper.cpp ---

if [ -f "$INSTALL_DIR/whisper-server" ]; then
    echo "whisper-server binary already exists, skipping build."
else
    echo "Building whisper.cpp..."

    if ! command -v cmake &> /dev/null; then
        install_cmake
    fi

    if [ -d "$SRC_DIR/.git" ]; then
        echo "Updating existing source..."
        git -C "$SRC_DIR" pull --ff-only 2>/dev/null || true
    else
        echo "Cloning whisper.cpp..."
        rm -rf "$SRC_DIR"
        git clone --depth 1 "$REPO" "$SRC_DIR"
    fi

    CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release)
    if [ "$OS_NAME" = "Linux" ] && command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
        echo "Compiling with CUDA support (nvidia-smi detected)..."
        CMAKE_FLAGS+=(-DGGML_CUDA=ON)
    elif [ "$OS_NAME" = "Darwin" ]; then
        # Metal is enabled by default on macOS (GPU acceleration on Apple Silicon).
        echo "Compiling with Metal support..."
    else
        echo "Compiling CPU-only build..."
    fi

    cmake -B "$SRC_DIR/build" -S "$SRC_DIR" "${CMAKE_FLAGS[@]}"
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

# --- Download Silero VAD ONNX model (for Python-side VAD) ---

SILERO_ONNX="$MODELS_DIR/silero_vad.onnx"
SILERO_ONNX_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
if [ -f "$SILERO_ONNX" ]; then
    echo "Silero VAD ONNX model already exists, skipping download."
else
    echo "Downloading Silero VAD ONNX model..."
    curl -fL --progress-bar -o "$SILERO_ONNX" "$SILERO_ONNX_URL"
    echo "Silero VAD ONNX model downloaded."
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
    echo "  Silero ONNX: $SILERO_ONNX"
fi
