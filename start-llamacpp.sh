#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_BIN="${LLAMA_BIN:-$DIR/llamacpp/llama-server}"
HOST="${LLAMACPP_HOST:-0.0.0.0}"
PORT="${LLAMACPP_PORT:-8080}"

if [ -n "${LLAMACPP_PRESET:-}" ]; then
    if [[ "$LLAMACPP_PRESET" = /* ]]; then
        PRESET="$LLAMACPP_PRESET"
    else
        PRESET="$DIR/$LLAMACPP_PRESET"
    fi
else
    PRESET="$DIR/llamacpp-models.ini"
fi

if [ ! -x "$LLAMA_BIN" ]; then
    echo "llama-server not found at $LLAMA_BIN; running setup-llamacpp.sh..."
    bash "$DIR/setup-llamacpp.sh"
fi

if [ ! -x "$LLAMA_BIN" ]; then
    echo "Error: llama-server binary not found after setup." >&2
    echo "Expected: $LLAMA_BIN" >&2
    exit 1
fi

if [ ! -f "$PRESET" ]; then
    echo "Error: llama.cpp preset file not found: $PRESET" >&2
    if [ -f "$DIR/llamacpp-models.ini.example" ]; then
        echo "Copy llamacpp-models.ini.example to llamacpp-models.ini and customize it." >&2
    fi
    exit 1
fi

echo "Starting llama-server on $HOST:$PORT"
echo "Using models preset: $PRESET"
exec "$LLAMA_BIN" \
    --host "$HOST" \
    --port "$PORT" \
    --models-preset "$PRESET" \
    "$@"
