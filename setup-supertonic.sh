#!/bin/bash
set -euo pipefail

# Installs the official Supertonic Python server into ./supertonic-server/
# with its own uv-managed virtualenv.
#
# Requirements:
#   - macOS or Linux
#   - uv (https://astral.sh/uv)
#
# Notes:
#   - Supertonic is CPU/ONNX by default; do not install GPU ONNX packages here.
#   - The server downloads model assets into the user's Supertonic cache on
#     first startup. ServerManager waits on /v1/health until that completes.
#   - Keep this in a separate Python 3.12 venv so ONNX Runtime wheel support
#     does not constrain the main app's Python version.

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/supertonic-server"
STAMP_FILE="$INSTALL_DIR/.installed"
VENV_PY="$INSTALL_DIR/.venv/bin/python"
VENV_SUPERTONIC="$INSTALL_DIR/.venv/bin/supertonic"
PACKAGE_SPEC="supertonic[serve]==1.3.1"

OS_NAME="$(uname -s)"
if [ "$OS_NAME" != "Darwin" ] && [ "$OS_NAME" != "Linux" ]; then
    echo "Error: supertonic is supported only on macOS and Linux (got $OS_NAME)." >&2
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "Error: uv is required but not found in PATH." >&2
    echo "  Install: https://astral.sh/uv/install.sh" >&2
    exit 1
fi

mkdir -p "$INSTALL_DIR"

venv_has_server() {
    [ -x "$VENV_SUPERTONIC" ] && "$VENV_PY" -c "import fastapi, supertonic, uvicorn" &>/dev/null
}

venv_pip() {
    (cd "$INSTALL_DIR" && VIRTUAL_ENV="$INSTALL_DIR/.venv" uv pip "$@")
}

if [ -x "$VENV_PY" ] && [ -f "$STAMP_FILE" ]; then
    current_spec=$(cat "$STAMP_FILE")
    if [ "$current_spec" = "$PACKAGE_SPEC" ] && venv_has_server; then
        echo "supertonic already installed ($PACKAGE_SPEC) — nothing to do."
        exit 0
    fi
    echo "supertonic install incomplete or stale; resuming from wherever we left off..."
fi

if [ ! -x "$VENV_PY" ]; then
    echo "Creating uv venv (Python 3.12)..."
    rm -rf "$INSTALL_DIR/.venv"
    (cd "$INSTALL_DIR" && uv venv --python 3.12 .venv)
fi

echo "Installing $PACKAGE_SPEC..."
venv_pip install "$PACKAGE_SPEC"

if ! venv_has_server; then
    echo "Error: supertonic server imports failed after install." >&2
    exit 1
fi

echo "$PACKAGE_SPEC" > "$STAMP_FILE"
echo "Done. supertonic installed at $INSTALL_DIR ($PACKAGE_SPEC)."
