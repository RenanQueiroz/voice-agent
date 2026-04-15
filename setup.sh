#!/bin/bash
# Setup and update dependencies for the voice agent.
# Usage:
#   ./setup.sh           # Install all deps (core + local)
#   ./setup.sh --update  # Update all deps to latest versions

set -e

cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

if [ "$1" = "--update" ]; then
    echo "Updating all dependencies to latest versions..."
    uv lock --upgrade
    uv sync --extra local
    echo ""
    echo "Done. Run 'uv run python -m voice-agent' to start."
else
    echo "Installing dependencies..."
    uv sync --extra local
    echo ""
    echo "Done. Run 'uv run python -m voice-agent' to start."
fi
