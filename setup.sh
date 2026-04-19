#!/bin/bash
# Setup and update dependencies for the voice agent.
# Usage:
#   ./setup.sh           # Install all deps (core + local)
#   ./setup.sh --update  # Update all deps to latest versions
#
# The `local` extra resolves to the mlx stack on macOS only — on Linux the
# sys_platform marker in pyproject.toml makes it a no-op, so there's nothing
# extra to install for local Linux runtimes (llama.cpp / whisper.cpp ship as
# binaries via setup-llamacpp.sh / setup-whispercpp.sh).
#
# On Linux we also install PortAudio, which the `sounddevice` Python package
# dlopens at import time. The package has no Linux wheel that bundles it, so
# without the system lib the Textual app can't even start.

set -e

cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

OS_NAME=$(uname -s)

# --- Install PortAudio (required by the `sounddevice` Python package) ---
#
# macOS: the sounddevice wheel bundles PortAudio, nothing to do.
# Linux: we need the system shared library. Package name varies by distro:
#   apt/zypper → libportaudio2   (dnf/pacman call it "portaudio")

install_portaudio_linux() {
    # Cheap detection: if the shared object is already discoverable via
    # ldconfig, skip. This avoids a sudo prompt on every setup.sh run.
    if command -v ldconfig &> /dev/null && ldconfig -p 2>/dev/null | grep -q 'libportaudio\.so\.2'; then
        return 0
    fi

    if command -v apt-get &> /dev/null; then
        echo "Installing PortAudio via apt (libportaudio2)..."
        sudo apt-get update -qq
        sudo apt-get install -y libportaudio2
    elif command -v dnf &> /dev/null; then
        echo "Installing PortAudio via dnf (portaudio)..."
        sudo dnf install -y portaudio
    elif command -v pacman &> /dev/null; then
        echo "Installing PortAudio via pacman (portaudio)..."
        sudo pacman -S --noconfirm portaudio
    elif command -v zypper &> /dev/null; then
        echo "Installing PortAudio via zypper (libportaudio2)..."
        sudo zypper install -y libportaudio2
    else
        echo "Error: PortAudio is required but no supported package manager was found." >&2
        echo "Install the PortAudio runtime library manually (libportaudio2 / portaudio)." >&2
        exit 1
    fi
}

if [ "$OS_NAME" = "Linux" ]; then
    install_portaudio_linux
fi

if [ "$1" = "--update" ]; then
    echo "Updating all dependencies to latest versions ($OS_NAME)..."
    uv lock --upgrade
    uv sync --extra local
else
    echo "Installing dependencies ($OS_NAME)..."
    uv sync --extra local
fi

echo ""
echo "Done. Run 'uv run python -m voice-agent' to start."
