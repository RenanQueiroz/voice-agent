#!/bin/bash
set -euo pipefail

# Installs llama.cpp into ./llamacpp. Two paths:
#
#   macOS arm64, Linux CPU x86_64, Linux aarch64:
#     Download and extract the latest prebuilt release from GitHub.
#     Up-to-date check uses `llama-cli --version` vs. the release tag.
#
#   Linux x86_64 + NVIDIA (nvcc in PATH):
#     Clone ggml-org/llama.cpp and build from source with CUDA enabled,
#     targeting the host GPU (CMAKE_CUDA_ARCHITECTURES=native). The
#     upstream releases page has no Ubuntu+CUDA asset, so we build to
#     get CUDA on Linux. The last-built commit SHA is recorded in
#     ./llamacpp/.built-commit; if `git ls-remote HEAD` matches the
#     stamp we skip the rebuild entirely.
#
# If nvcc is missing (NVIDIA driver only, no CUDA Toolkit) we fall back
# to the CPU prebuilt with a warning — llama.cpp is still usable without
# a GPU, just slower.

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/llamacpp"
SRC_DIR="$DIR/llamacpp-src"
STAMP_FILE="$INSTALL_DIR/.built-commit"

TMPFILE=$(mktemp /tmp/llamacpp-XXXXXX)
TMPDIR_EXTRACT=""
cleanup() {
    rm -f "$TMPFILE"
    if [ -n "$TMPDIR_EXTRACT" ] && [ -d "$TMPDIR_EXTRACT" ]; then
        rm -rf "$TMPDIR_EXTRACT"
    fi
}
trap cleanup EXIT

LLAMA_REPO="ggml-org/llama.cpp"
LLAMA_REPO_URL="https://github.com/$LLAMA_REPO.git"

UPDATED=0

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

# --- Decide install mode ---

# On Linux x86_64 with an NVIDIA GPU we try the source-CUDA build. nvcc
# can live outside PATH (CUDA Toolkit typically installs to /usr/local/cuda);
# probe the common locations before giving up.
if ! command -v nvcc &>/dev/null; then
    for p in /usr/local/cuda/bin /opt/cuda/bin; do
        if [ -x "$p/nvcc" ]; then
            export PATH="$p:$PATH"
            break
        fi
    done
fi

INSTALL_MODE="prebuilt"
if [ "$OS_NAME" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        if command -v nvcc &>/dev/null; then
            INSTALL_MODE="source-cuda"
        else
            echo "NVIDIA GPU detected, but nvcc is not in PATH."
            echo "  Install the CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)"
            echo "  and ensure nvcc is reachable to build llama.cpp with CUDA."
            echo "  Falling back to the CPU prebuilt."
        fi
    fi
fi

# --- Build-tool auto-install (Linux) ---

detect_linux_pkg_mgr() {
    # Prefer /etc/os-release so we pick the distro's actual manager even
    # when several are installed. Fall back to PATH probing.
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case " ${ID:-} ${ID_LIKE:-} " in
            *" debian "*|*" ubuntu "*)     echo apt;    return ;;
            *" fedora "*|*" rhel "*|*" centos "*) echo dnf; return ;;
            *" arch "*|*" manjaro "*)      echo pacman; return ;;
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

install_build_tools() {
    local pkg_mgr
    pkg_mgr=$(detect_linux_pkg_mgr)
    if [ -z "$pkg_mgr" ]; then
        echo "Error: could not detect a supported Linux package manager." >&2
        return 1
    fi

    local sudo_cmd=""
    if [ "$(id -u)" != "0" ]; then
        if command -v sudo &>/dev/null; then
            sudo_cmd="sudo"
        else
            echo "Error: need root or sudo to install build tools via $pkg_mgr." >&2
            return 1
        fi
    fi

    echo "Installing build tools via $pkg_mgr (requires sudo)..."
    case "$pkg_mgr" in
        apt)
            $sudo_cmd apt-get update
            $sudo_cmd apt-get install -y cmake build-essential git
            ;;
        dnf)
            $sudo_cmd dnf install -y cmake gcc gcc-c++ make git
            ;;
        pacman)
            $sudo_cmd pacman -S --needed --noconfirm cmake gcc make git
            ;;
        zypper)
            $sudo_cmd zypper install -y cmake gcc gcc-c++ make git
            ;;
        *)
            echo "Error: unsupported package manager: $pkg_mgr" >&2
            return 1
            ;;
    esac
}

# --- Source build: Linux x86_64 + CUDA ---

install_source_cuda() {
    echo "Checking llama.cpp (source build, CUDA)..."

    # nvcc is part of the CUDA Toolkit (multi-GB, often needs NVIDIA's repo
    # configured first), so we don't auto-install it — error with a hint
    # instead. The rest of the toolchain is a single apt/dnf/pacman/zypper
    # line, which we run transparently if anything is missing.
    local basic_tools=(git cmake make gcc g++)
    local missing=()
    for cmd in "${basic_tools[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "Missing build tools: ${missing[*]} — installing via package manager..."
        install_build_tools
        missing=()
        for cmd in "${basic_tools[@]}"; do
            if ! command -v "$cmd" &>/dev/null; then
                missing+=("$cmd")
            fi
        done
        if [ "${#missing[@]}" -gt 0 ]; then
            echo "Error: still missing after install attempt: ${missing[*]}" >&2
            exit 1
        fi
    fi
    if ! command -v nvcc &>/dev/null; then
        echo "Error: nvcc not found. Install the CUDA Toolkit:" >&2
        echo "  https://developer.nvidia.com/cuda-downloads" >&2
        exit 1
    fi

    local remote_sha
    remote_sha=$(git ls-remote "$LLAMA_REPO_URL" HEAD | awk '{print $1}')
    if [ -z "$remote_sha" ]; then
        echo "Error: could not resolve upstream HEAD of $LLAMA_REPO_URL" >&2
        exit 1
    fi

    local local_sha=""
    if [ -f "$STAMP_FILE" ]; then
        local_sha=$(cat "$STAMP_FILE")
    fi

    if [ "$local_sha" = "$remote_sha" ] && [ -x "$INSTALL_DIR/llama-server" ]; then
        echo "llama.cpp is already up to date (commit ${remote_sha:0:12})."
        return 0
    fi

    if [ -n "$local_sha" ]; then
        echo "Updating llama.cpp: ${local_sha:0:12} -> ${remote_sha:0:12}"
    else
        echo "Building llama.cpp from source at ${remote_sha:0:12}"
    fi

    # Shallow clone/fetch — we only ever build tip-of-master, so full
    # history wastes hundreds of MB. Upstream can advance between
    # `ls-remote` above and the fetch below; that's fine, we stamp
    # whatever we actually checked out and the next run will notice.
    if [ ! -d "$SRC_DIR/.git" ]; then
        echo "Cloning $LLAMA_REPO_URL into $SRC_DIR (shallow)..."
        git clone --depth 1 "$LLAMA_REPO_URL" "$SRC_DIR"
    else
        echo "Fetching latest commit..."
        git -C "$SRC_DIR" fetch --depth 1 origin HEAD
        git -C "$SRC_DIR" reset --hard FETCH_HEAD
    fi

    local built_sha
    built_sha=$(git -C "$SRC_DIR" rev-parse HEAD)

    # Fresh build dir so upstream CMake / flag changes can't leak stale state.
    rm -rf "$SRC_DIR/build"

    echo "Configuring (GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=native)..."
    cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=native \
        -DLLAMA_BUILD_TESTS=OFF

    echo "Building (this takes several minutes)..."
    cmake --build "$SRC_DIR/build" --config Release -j"$(nproc)" \
        --target llama-server llama-cli

    local build_bin="$SRC_DIR/build/bin"
    if [ ! -x "$build_bin/llama-server" ]; then
        echo "Error: llama-server not found after build at $build_bin." >&2
        exit 1
    fi

    echo "Installing to $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    # Flat layout (binaries + shared libs in INSTALL_DIR) to match the
    # prebuilt path so servers.py doesn't care which mode produced the tree.
    find "$build_bin" -maxdepth 1 -type f -executable -exec cp -af {} "$INSTALL_DIR/" \;
    find "$SRC_DIR/build" -name '*.so*' -exec cp -af {} "$INSTALL_DIR/" \;

    echo "$built_sha" > "$STAMP_FILE"
    echo "llama.cpp built from source (${built_sha:0:12})."
    UPDATED=1
}

# --- Prebuilt: macOS arm64, Linux CPU / arm64 ---

install_prebuilt() {
    echo "Checking llama.cpp (prebuilt)..."

    local llama_tag
    llama_tag=$(curl -sf "https://api.github.com/repos/$LLAMA_REPO/releases/latest" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
    if [ -z "$llama_tag" ]; then
        echo "Error: could not determine latest llama.cpp release tag" >&2
        exit 1
    fi

    # Tag is e.g. "b8763"; local version reports "8763".
    local llama_latest="${llama_tag#b}"
    local llama_local=""
    if [ -x "$INSTALL_DIR/llama-cli" ]; then
        llama_local=$("$INSTALL_DIR/llama-cli" --version 2>&1 | grep -o 'version: [0-9]*' | cut -d' ' -f2 || echo "")
    fi

    # If a previous source build left a stamp, treat the install as dirty
    # and reinstall from prebuilt so we don't end up with mixed artifacts.
    if [ "$llama_local" = "$llama_latest" ] && [ ! -f "$STAMP_FILE" ]; then
        echo "llama.cpp is already up to date ($llama_tag)."
        return 0
    fi

    echo "Updating llama.cpp: ${llama_local:-not installed} -> $llama_latest"

    local candidates=()
    case "$OS_NAME" in
        Darwin)
            candidates+=("llama-${llama_tag}-bin-macos-arm64.tar.gz")
            ;;
        Linux)
            if [ "$ARCH" = "x86_64" ]; then
                candidates+=("llama-${llama_tag}-bin-ubuntu-x64.tar.gz")
            else
                candidates+=("llama-${llama_tag}-bin-ubuntu-arm64.tar.gz")
            fi
            ;;
    esac

    local asset="" url
    for candidate in "${candidates[@]}"; do
        url="https://github.com/$LLAMA_REPO/releases/download/$llama_tag/$candidate"
        echo "Downloading $candidate..."
        if curl -fsSL --progress-bar -o "$TMPFILE" "$url"; then
            asset="$candidate"
            break
        fi
        echo "  not found."
    done

    if [ -z "$asset" ]; then
        echo "Error: no matching llama.cpp asset for $OS_NAME $ARCH (tag $llama_tag)." >&2
        echo "  See https://github.com/$LLAMA_REPO/releases/tag/$llama_tag for asset names." >&2
        exit 1
    fi

    TMPDIR_EXTRACT=$(mktemp -d)

    echo "Extracting $asset..."
    case "$asset" in
        *.tar.gz)
            tar -xzf "$TMPFILE" -C "$TMPDIR_EXTRACT"
            ;;
        *.zip)
            if ! command -v unzip &>/dev/null; then
                echo "Error: unzip is required to extract $asset but is not installed." >&2
                echo "  Install it with your package manager (apt install unzip / dnf install unzip / pacman -S unzip)." >&2
                exit 1
            fi
            unzip -q "$TMPFILE" -d "$TMPDIR_EXTRACT"
            ;;
        *)
            echo "Error: unrecognized archive extension for $asset." >&2
            exit 1
            ;;
    esac

    local server_bin
    server_bin="$(find "$TMPDIR_EXTRACT" -type f -name 'llama-server' -print -quit)"
    if [ -z "$server_bin" ]; then
        echo "Error: llama-server not found inside $asset." >&2
        exit 1
    fi

    mkdir -p "$INSTALL_DIR"
    local bin_dir lib_dir
    bin_dir="$(dirname "$server_bin")"
    cp -R "$bin_dir/." "$INSTALL_DIR/"
    lib_dir="$(dirname "$bin_dir")/lib"
    if [ -d "$lib_dir" ]; then
        cp -R "$lib_dir/." "$INSTALL_DIR/"
    fi

    # We're on a prebuilt now; drop any stale source-build stamp so the
    # next run doesn't mistakenly short-circuit on commit comparison.
    rm -f "$STAMP_FILE"

    echo "llama.cpp $llama_tag installed ($asset)."
    UPDATED=1
}

# --- Run ---

if [ "$INSTALL_MODE" = "source-cuda" ]; then
    install_source_cuda
else
    install_prebuilt
fi

echo

if [ "$UPDATED" -eq 0 ]; then
    echo "Everything is already up to date."
else
    echo "Done. Binaries updated in $INSTALL_DIR"
fi
