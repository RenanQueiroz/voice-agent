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

# --- Install audio system deps (required by the `sounddevice` Python package) ---
#
# macOS: the sounddevice wheel bundles PortAudio, nothing to do.
# Linux: three packages are needed:
#   libportaudio2       — sounddevice dlopens libportaudio.so.2 at import time.
#   pulseaudio-utils    — provides `pactl` (diagnostics; handy for the
#                         troubleshooting path documented in README).
#   libasound2-plugins  — the ALSA→PulseAudio bridge. On WSL2 there is no real
#                         ALSA device (only WSLg's PulseAudio socket at
#                         $PULSE_SERVER), so PortAudio's default ALSA host
#                         cannot find a mic. This plugin makes ALSA transparently
#                         route through PulseAudio, which gives PortAudio a
#                         working default input device. Also harmless on native
#                         Linux — usually already installed by the desktop.

install_audio_deps_linux() {
    local have_portaudio=0 have_pactl=0
    if command -v ldconfig &> /dev/null && ldconfig -p 2>/dev/null | grep -q 'libportaudio\.so\.2'; then
        have_portaudio=1
    fi
    if command -v pactl &> /dev/null; then
        have_pactl=1
    fi
    # Avoids a sudo prompt on every run once the environment is set up.
    # A fresh install will be missing at least libportaudio2, which triggers
    # the full three-package install (package managers are idempotent so
    # reinstalling the ALSA plugin if it happened to be present is a no-op).
    if [ "$have_portaudio" = "1" ] && [ "$have_pactl" = "1" ]; then
        return 0
    fi

    if command -v apt-get &> /dev/null; then
        echo "Installing audio deps via apt (libportaudio2 pulseaudio-utils libasound2-plugins)..."
        sudo apt-get update -qq
        sudo apt-get install -y libportaudio2 pulseaudio-utils libasound2-plugins
    elif command -v dnf &> /dev/null; then
        echo "Installing audio deps via dnf (portaudio pulseaudio-utils alsa-plugins-pulseaudio)..."
        sudo dnf install -y portaudio pulseaudio-utils alsa-plugins-pulseaudio
    elif command -v pacman &> /dev/null; then
        echo "Installing audio deps via pacman (portaudio libpulse alsa-plugins)..."
        sudo pacman -S --noconfirm portaudio libpulse alsa-plugins
    elif command -v zypper &> /dev/null; then
        echo "Installing audio deps via zypper (libportaudio2 pulseaudio-utils alsa-plugins-pulse)..."
        sudo zypper install -y libportaudio2 pulseaudio-utils alsa-plugins-pulse
    else
        echo "Error: Audio deps are required but no supported package manager was found." >&2
        echo "Install manually: PortAudio runtime lib + pulseaudio utilities + ALSA PulseAudio plugin." >&2
        exit 1
    fi
}

if [ "$OS_NAME" = "Linux" ]; then
    install_audio_deps_linux
fi

# --- Download Silero VAD ONNX model ---
#
# The app's input path is always Silero VAD — VADRecorder loads this ONNX
# regardless of whether any role is local or cloud. So every user needs
# this file, not just local-STT users. (setup-whispercpp.sh also grabs it
# as part of its whisper.cpp setup; both paths are idempotent.)

SILERO_ONNX="$(pwd)/whispercpp/models/silero_vad.onnx"
SILERO_ONNX_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
if [ ! -f "$SILERO_ONNX" ]; then
    echo "Downloading Silero VAD ONNX model..."
    mkdir -p "$(dirname "$SILERO_ONNX")"
    curl -fL --progress-bar -o "$SILERO_ONNX" "$SILERO_ONNX_URL"
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
