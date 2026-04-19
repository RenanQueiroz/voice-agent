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

set -e

cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

OS_NAME=$(uname -s)

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
