#!/bin/bash
set -euo pipefail

# Builds whisper.cpp from source and downloads the specified GGML model
# plus the Silero VAD ONNX model used by the Python recorder.
#
# GPU acceleration:
#   macOS   → Metal (auto-enabled by whisper.cpp's cmake).
#   Linux   → CUDA if `nvidia-smi` is present (`-DGGML_CUDA=ON`),
#             else OpenBLAS if available (`-DGGML_BLAS=1`), else CPU.
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
STAMP_FILE="$INSTALL_DIR/.built-commit"
BUILD_MODE_FILE="$INSTALL_DIR/.build-mode"
REPO="https://github.com/ggml-org/whisper.cpp.git"

MODEL="${1:-large-v3-turbo-q5_0}"
WHISPER_HF="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

SUPPORTED_MODELS=(
    tiny tiny.en base base.en small small.en small.en-tdrz
    medium medium.en large-v1 large-v2 large-v2-q5_0
    large-v3 large-v3-q5_0 large-v3-turbo large-v3-turbo-q5_0
)

OS_NAME="$(uname -s)"

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

detect_build_mode() {
    if [ "$OS_NAME" = "Linux" ] && command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
        echo "linux-cuda"
    elif [ "$OS_NAME" = "Linux" ] && has_openblas; then
        echo "linux-openblas"
    elif [ "$OS_NAME" = "Darwin" ]; then
        echo "darwin-metal"
    else
        echo "cpu"
    fi
}

has_openblas() {
    if command -v pkg-config &> /dev/null && pkg-config --exists openblas; then
        return 0
    fi
    local has_lib=0
    for path in \
        /usr/lib/libopenblas.so \
        /usr/lib64/libopenblas.so \
        /usr/lib/*/libopenblas.so \
        /usr/local/lib/libopenblas.so; do
        if compgen -G "$path" > /dev/null; then
            has_lib=1
            break
        fi
    done
    if [ "$has_lib" -eq 0 ] && command -v ldconfig &> /dev/null && ldconfig -p 2>/dev/null | grep -q 'libopenblas\.so'; then
        has_lib=1
    fi
    if [ "$has_lib" -eq 0 ]; then
        return 1
    fi
    for path in \
        /usr/include/cblas.h \
        /usr/include/openblas/cblas.h \
        /usr/include/*/cblas.h \
        /usr/include/*/openblas*/cblas.h \
        /usr/local/include/cblas.h \
        /usr/local/include/openblas/cblas.h; do
        if compgen -G "$path" > /dev/null; then
            return 0
        fi
    done
    return 1
}

# --- Build whisper.cpp ---

echo "Checking whisper.cpp..."

BUILD_MODE="$(detect_build_mode)"
REMOTE_SHA="$(git ls-remote "$REPO" HEAD | awk '{print $1}')"
if [ -z "$REMOTE_SHA" ]; then
    echo "Error: could not resolve upstream HEAD of $REPO" >&2
    exit 1
fi

LOCAL_SHA=""
if [ -f "$STAMP_FILE" ]; then
    LOCAL_SHA="$(cat "$STAMP_FILE")"
fi

LOCAL_BUILD_MODE=""
if [ -f "$BUILD_MODE_FILE" ]; then
    LOCAL_BUILD_MODE="$(cat "$BUILD_MODE_FILE")"
fi

if [ "$LOCAL_SHA" = "$REMOTE_SHA" ] && [ "$LOCAL_BUILD_MODE" = "$BUILD_MODE" ] && [ -x "$INSTALL_DIR/whisper-server" ]; then
    echo "whisper.cpp is already up to date (commit ${REMOTE_SHA:0:12}, $BUILD_MODE)."
else
    if [ -n "$LOCAL_SHA" ]; then
        echo "Updating whisper.cpp: ${LOCAL_SHA:0:12} -> ${REMOTE_SHA:0:12} ($BUILD_MODE)"
    else
        echo "Building whisper.cpp from source at ${REMOTE_SHA:0:12} ($BUILD_MODE)"
    fi
    if [ -n "$LOCAL_BUILD_MODE" ] && [ "$LOCAL_BUILD_MODE" != "$BUILD_MODE" ]; then
        echo "Build mode changed: $LOCAL_BUILD_MODE -> $BUILD_MODE"
    fi

    if ! command -v cmake &> /dev/null; then
        install_cmake
    fi

    if [ ! -d "$SRC_DIR/.git" ]; then
        echo "Cloning whisper.cpp..."
        rm -rf "$SRC_DIR"
        git clone --depth 1 "$REPO" "$SRC_DIR"
    else
        echo "Fetching latest commit..."
        git -C "$SRC_DIR" fetch --depth 1 origin HEAD
        git -C "$SRC_DIR" reset --hard FETCH_HEAD
    fi

    BUILT_SHA="$(git -C "$SRC_DIR" rev-parse HEAD)"

    rm -rf "$SRC_DIR/build"

    CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release)
    case "$BUILD_MODE" in
        linux-openblas)
            echo "Compiling with OpenBLAS support..."
            CMAKE_FLAGS+=(-DGGML_BLAS=1 -DGGML_BLAS_VENDOR=OpenBLAS)
            ;;
        linux-cuda)
            echo "Compiling with CUDA support (nvidia-smi detected)..."
            CMAKE_FLAGS+=(-DGGML_CUDA=ON)
            ;;
        darwin-metal)
            # Metal is enabled by default on macOS (GPU acceleration on Apple Silicon).
            echo "Compiling with Metal support..."
            ;;
        *)
            echo "Compiling CPU-only build..."
            ;;
    esac

    cmake -B "$SRC_DIR/build" -S "$SRC_DIR" "${CMAKE_FLAGS[@]}"
    cmake --build "$SRC_DIR/build" -j --config Release

    echo "Copying binaries..."
    cp "$SRC_DIR/build/bin/whisper-server" "$INSTALL_DIR/"
    cp "$SRC_DIR/build/bin/whisper-cli" "$INSTALL_DIR/" 2>/dev/null || true

    echo "$BUILT_SHA" > "$STAMP_FILE"
    echo "$BUILD_MODE" > "$BUILD_MODE_FILE"

    echo "whisper.cpp built successfully (${BUILT_SHA:0:12}, $BUILD_MODE)."
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
    echo "  Silero ONNX: $SILERO_ONNX"
fi
