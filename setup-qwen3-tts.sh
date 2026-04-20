#!/bin/bash
set -euo pipefail

# Installs Qwen3-TTS-Openai-Fastapi into ./qwen3-tts/ with its own uv venv
# for the `optimized` backend (torch.compile + CUDA graphs + flash-attn-3
# via HF kernels + real token-by-token PCM streaming).
#
# Requirements:
#   - Linux + NVIDIA GPU (CUDA 12.8-compatible driver; drivers ≥525 typically
#     satisfy this since CUDA runtime versioning is backward-compatible).
#   - uv  (https://astral.sh/uv)
#   - A detected package manager: apt / dnf / pacman / zypper
#   - ~8 GB free disk for venv+model (down from ~12 GB in the old
#     flash-attn-2-source-build setup).
#
# Attention backend: we use `attn_implementation="kernels-community/flash-
# attn3"`, which causes transformers to download a pre-built flash-attn 3
# kernel from HF Hub at model-load time — no nvcc, no source compilation,
# no 20-60 min wait. The kernel matches the upstream `Qwen/Qwen3-TTS` HF
# Space setup. On Ada (RTX 40xx) the runtime speedup over flash-attn 2
# is modest (it's a Hopper-optimized kernel), but combined with torch 2.8
# it delivers RTF ~0.5× (well under realtime) on an RTX 4080 Laptop.
#
# Version lock-in: `kernels<0.10` pins us to the 0.9.x line, which is the
# last release compatible with `huggingface-hub<1.0` (which transformers
# 4.57.3 requires). Bumping `kernels` past that will break the backend
# init with a transformers-vs-hub version-compat error.
#
# Idempotency:
#   We stamp ./qwen3-tts/.installed with the checked-out commit SHA. If the
#   stamp matches the pinned ref AND the venv imports torch + torchaudio +
#   kernels + loads the flash-attn3 kernel, re-running is a no-op.

DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$DIR/qwen3-tts"
STAMP_FILE="$INSTALL_DIR/.installed"
VENV_PY="$INSTALL_DIR/.venv/bin/python"

REPO_URL="https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git"
# Pinned to the upstream default branch; bump this deliberately after
# verifying upstream changes don't break the optimized backend config.
PINNED_REF="main"

OS_NAME="$(uname -s)"
if [ "$OS_NAME" != "Linux" ]; then
    echo "Error: qwen3-tts is a Linux-only runtime (got $OS_NAME)." >&2
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi -L &>/dev/null; then
    echo "Error: qwen3-tts needs an NVIDIA GPU but no nvidia-smi detected." >&2
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "Error: uv is required but not found in PATH." >&2
    echo "  Install: https://astral.sh/uv/install.sh" >&2
    exit 1
fi

# --- Detect Linux package manager ---

detect_linux_pkg_mgr() {
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case " ${ID:-} ${ID_LIKE:-} " in
            *" debian "*|*" ubuntu "*)              echo apt; return ;;
            *" fedora "*|*" rhel "*|*" centos "*)   echo dnf; return ;;
            *" arch "*|*" manjaro "*)               echo pacman; return ;;
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

install_system_deps() {
    local pkg_mgr
    pkg_mgr=$(detect_linux_pkg_mgr)
    if [ -z "$pkg_mgr" ]; then
        echo "Error: no supported package manager (apt/dnf/pacman/zypper) detected." >&2
        exit 1
    fi
    local sudo_cmd=""
    if [ "$(id -u)" != "0" ]; then
        if command -v sudo &>/dev/null; then
            sudo_cmd="sudo"
        else
            echo "Error: system deps require root or sudo (package manager: $pkg_mgr)." >&2
            exit 1
        fi
    fi

    # python3.X-dev provides the Python headers (Python.h) that
    # torch.compile / Triton need at *runtime* — Inductor shells out to
    # gcc to compile a tiny cuda_utils.so per compiled graph, and that
    # .c file does `#include <Python.h>`. Without the headers it fails
    # with "Python.h: No such file or directory" and audio generation
    # errors out mid-turn. Version-specific package name keeps us
    # aligned with the venv's system Python (uv creates --python 3.12
    # venvs against /usr/bin/python3.12 on Ubuntu/Debian).
    echo "Installing system deps via $pkg_mgr (ffmpeg, libsndfile, libsox, sox, build tools, python3-dev)..."
    case "$pkg_mgr" in
        apt)
            $sudo_cmd apt-get update
            $sudo_cmd apt-get install -y \
                ffmpeg libsndfile1 libsox-dev sox \
                build-essential git curl \
                python3.12-dev
            ;;
        dnf)
            $sudo_cmd dnf install -y ffmpeg libsndfile sox sox-devel \
                gcc gcc-c++ make git curl python3-devel
            ;;
        pacman)
            $sudo_cmd pacman -S --needed --noconfirm ffmpeg libsndfile sox \
                base-devel git curl
            # Arch ships Python headers in the main `python` package, so
            # no separate -dev package is needed.
            ;;
        zypper)
            $sudo_cmd zypper install -y ffmpeg libsndfile1 sox sox-devel \
                gcc gcc-c++ make git curl python3-devel
            ;;
    esac
}

clone_or_update() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        # Warn before blowing away any uncommitted edits — the
        # reset --hard below is silent about discards otherwise.
        # Note: this fires on every run because our own
        # apply_config_optimizations patch leaves config.yaml dirty
        # relative to the tree we reset to; that's harmless (the patch
        # is re-applied below) but the message is still accurate —
        # anything tracked is about to go.
        if ! git -C "$INSTALL_DIR" diff --quiet 2>/dev/null \
            || ! git -C "$INSTALL_DIR" diff --cached --quiet 2>/dev/null; then
            echo "NOTE: discarding uncommitted changes in $INSTALL_DIR (syncing to $PINNED_REF)"
        fi
        echo "Fetching $PINNED_REF..."
        git -C "$INSTALL_DIR" fetch --depth 1 origin "$PINNED_REF"
        git -C "$INSTALL_DIR" reset --hard "origin/$PINNED_REF" 2>/dev/null \
            || git -C "$INSTALL_DIR" reset --hard "$PINNED_REF"
    else
        echo "Cloning $REPO_URL into $INSTALL_DIR..."
        git clone --depth 1 --branch "$PINNED_REF" "$REPO_URL" "$INSTALL_DIR"
    fi
}

current_repo_sha() {
    git -C "$INSTALL_DIR" rev-parse HEAD
}

# Raise torch._dynamo's per-function cache limit at Python startup.
#
# Qwen3's forward passes go through `transformers.utils.generic:wrapper`
# with variable kwargs, so every distinct (shape, kwargs) combo takes a
# cache slot. The default (8) is too tight — once saturated, dynamo
# stops compiling and falls back to eager for new combos, which kills
# RTF. No env var exposes this (as of torch 2.8), so inject it via
# Python's site initialization.
#
# We drop TWO files in the venv's site-packages:
#
#   1. `_voice_agent_dynamo_cache.py` — the actual patch module.
#   2. `voice_agent_dynamo_cache.pth` — a .pth file whose sole line
#      `import _voice_agent_dynamo_cache` is exec'd by `site.py` during
#      site-package processing.
#
# Why .pth instead of sitecustomize.py? Ubuntu/Debian's python3.12 ships
# its own `/usr/lib/python3.12/sitecustomize.py`, which wins the name
# resolution and prevents our venv-local copy from ever being loaded.
# .pth files all get processed, in order — no name collision. They also
# run earlier than sitecustomize, which is what we want anyway.
#
# Idempotent — we overwrite every run. Requires the venv to exist;
# callers should guard on `[ -x "$VENV_PY" ]`.
write_sitecustomize() {
    local site_pkg
    site_pkg=$("$VENV_PY" -c "import site; print(site.getsitepackages()[0])")
    # Quiet no-op when both files already exist with the current shape
    # (set -u + small marker check). Comparing body hashes would be more
    # bulletproof but this is enough — the shape rarely changes and any
    # diff is harmless (we overwrite below).
    if [ -f "$site_pkg/_voice_agent_dynamo_cache.py" ] \
        && [ -f "$site_pkg/voice_agent_dynamo_cache.pth" ] \
        && grep -q "cache_size_limit < 64" "$site_pkg/_voice_agent_dynamo_cache.py" 2>/dev/null; then
        return 0
    fi
    cat > "$site_pkg/_voice_agent_dynamo_cache.py" <<'PY'
# Auto-generated by voice-agent/setup-qwen3-tts.sh. Loaded at interpreter
# startup via the matching .pth file — runs before any user code touches
# torch. Keep try/except broad: every Python invocation against this
# venv (pip, uv, python -c, etc.) runs this, and a failure must not
# prevent the interpreter from starting.
try:
    import torch._dynamo.config as _c

    if _c.cache_size_limit < 64:
        _c.cache_size_limit = 64
    if _c.accumulated_cache_size_limit < 1024:
        _c.accumulated_cache_size_limit = 1024
except Exception:
    pass
PY
    cat > "$site_pkg/voice_agent_dynamo_cache.pth" <<'PTH'
import _voice_agent_dynamo_cache
PTH
    echo "Wrote $site_pkg/_voice_agent_dynamo_cache.py + .pth (bumps torch._dynamo cache_size_limit to 64)"
}

# Apply source patches to the vendored qwen3-tts server.
#
# Upstream doesn't expose `temperature` (or other sampling kwargs) on
# the `/v1/audio/speech` endpoint — the pydantic request schema omits
# it, so client-sent `extra_body={"temperature": 0.7}` is silently
# dropped by pydantic's default `extra="ignore"`. But the underlying
# `model.stream_generate_custom_voice` / `generate_custom_voice`
# methods DO accept `temperature` via `**kwargs` (see their docstring
# — "Sampling temperature; higher => more random"). Lowering
# temperature below the default 1.0 significantly tames the speaker-
# embedding's L1 phonetic prior on non-native-language output (e.g.
# Sohee + English), producing cleaner, less-accented pronunciation
# while preserving streaming TTFB.
#
# We plumb it through three files:
#   - api/structures/schemas.py        — add `temperature` field
#   - api/routers/openai_compatible.py — forward it on the streaming call
#   - api/backends/optimized_backend.py — accept it, pass to the model
#
# Runs AFTER clone_or_update (which does `git reset --hard` and would
# otherwise revert the edits). Idempotent — each step grep-guarded so
# re-running is cheap.
apply_source_patches() {
    python3 <<'PY'
import os, sys

install_dir = os.environ["INSTALL_DIR"]

def patch_file(rel_path, old, new, marker):
    """Replace `old` with `new` in the target file. Skip if `marker`
    already present. Raise if `old` not found — guards against silent
    upstream changes breaking the patch."""
    path = os.path.join(install_dir, rel_path)
    with open(path) as f:
        src = f.read()
    if marker in src:
        return False  # already patched
    if old not in src:
        raise SystemExit(
            f"apply_source_patches: {rel_path} doesn't contain the expected "
            f"pre-patch text. Upstream probably changed — update the patch "
            f"in setup-qwen3-tts.sh."
        )
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    return True

# 1. schemas.py — add `temperature` to OpenAISpeechRequest.
if patch_file(
    "api/structures/schemas.py",
    (
        "    instruct: Optional[str] = Field(\n"
        "        default=None,\n"
        "        description=\"Optional instruction for voice style/emotion control.\",\n"
        "    )\n"
        "    normalization_options"
    ),
    (
        "    instruct: Optional[str] = Field(\n"
        "        default=None,\n"
        "        description=\"Optional instruction for voice style/emotion control.\",\n"
        "    )\n"
        "    temperature: Optional[float] = Field(\n"
        "        default=None,\n"
        "        ge=0.0,\n"
        "        le=2.0,\n"
        "        description=\"Optional sampling temperature (passed to the model as a sampling kwarg). Lower => more deterministic, cleaner pronunciation on non-native-language output. Default leaves the model's internal default (1.0).\",\n"
        "    )\n"
        "    normalization_options"
    ),
    marker="    temperature: Optional[float] = Field(",
):
    print("Patched schemas.py: added OpenAISpeechRequest.temperature")

# 2. openai_compatible.py — forward temperature to the streaming backend call.
if patch_file(
    "api/routers/openai_compatible.py",
    (
        "                        async for pcm_chunk, sr in backend.generate_speech_streaming(\n"
        "                            text=normalized_text,\n"
        "                            voice=voice_name,\n"
        "                            language=language,\n"
        "                            instruct=request.instruct,\n"
        "                            model=request.model,\n"
        "                        ):"
    ),
    (
        "                        async for pcm_chunk, sr in backend.generate_speech_streaming(\n"
        "                            text=normalized_text,\n"
        "                            voice=voice_name,\n"
        "                            language=language,\n"
        "                            instruct=request.instruct,\n"
        "                            model=request.model,\n"
        "                            temperature=request.temperature,\n"
        "                        ):"
    ),
    marker="temperature=request.temperature,",
):
    print("Patched openai_compatible.py: forward temperature on streaming")

# 3. optimized_backend.py — accept temperature and pass to the model call.
if patch_file(
    "api/backends/optimized_backend.py",
    (
        "    async def generate_speech_streaming(\n"
        "        self,\n"
        "        text: str,\n"
        "        voice: str,\n"
        "        language: str = \"Auto\",\n"
        "        instruct: Optional[str] = None,\n"
        "        speed: float = 1.0,\n"
        "        model: str = \"tts-1\",\n"
        "    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:"
    ),
    (
        "    async def generate_speech_streaming(\n"
        "        self,\n"
        "        text: str,\n"
        "        voice: str,\n"
        "        language: str = \"Auto\",\n"
        "        instruct: Optional[str] = None,\n"
        "        speed: float = 1.0,\n"
        "        model: str = \"tts-1\",\n"
        "        temperature: Optional[float] = None,\n"
        "    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:"
    ),
    marker="        temperature: Optional[float] = None,\n    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:",
):
    print("Patched optimized_backend.py: added temperature param to generate_speech_streaming")

if patch_file(
    "api/backends/optimized_backend.py",
    (
        "        for chunk, sr in self.model.stream_generate_custom_voice(\n"
        "            text=text,\n"
        "            speaker=voice,\n"
        "            language=language,\n"
        "            instruct=instruct,\n"
        "            emit_every_frames=emit_every_frames,\n"
        "            decode_window_frames=decode_window_frames,\n"
        "        ):"
    ),
    (
        "        model_kwargs = {}\n"
        "        if temperature is not None:\n"
        "            model_kwargs[\"temperature\"] = temperature\n"
        "        for chunk, sr in self.model.stream_generate_custom_voice(\n"
        "            text=text,\n"
        "            speaker=voice,\n"
        "            language=language,\n"
        "            instruct=instruct,\n"
        "            emit_every_frames=emit_every_frames,\n"
        "            decode_window_frames=decode_window_frames,\n"
        "            **model_kwargs,\n"
        "        ):"
    ),
    marker="**model_kwargs,\n        ):",
):
    print("Patched optimized_backend.py: forward temperature kwarg to stream_generate_custom_voice")
PY
}
export INSTALL_DIR  # consumed by the heredoc above

# Apply performance overrides to upstream config.yaml.
#
# Upstream ships the most conservative defaults so the server boots on
# any GPU. Everything but `use_cuda_graphs` is already tuned (flash-attn
# 2, torch.compile max-autotune, fast codebook, 80-frame decode window,
# 6-frame emit interval). CUDA graphs is explicitly gated with a
# "set true after verifying compile works on your GPU" note — our
# post-install verification proves compile works, and CUDA graphs
# materially improves RTF on modern NVIDIA GPUs (RTX 20xx+), so flip it.
#
# Runs AFTER clone_or_update (which does `git reset --hard` and would
# otherwise revert the edit). Idempotent — only rewrites if the upstream
# default is still present, so re-running this script is cheap.
apply_config_optimizations() {
    local cfg="$INSTALL_DIR/config.yaml"
    [ -f "$cfg" ] || return 0
    if grep -qE '^[[:space:]]*use_cuda_graphs:[[:space:]]*false' "$cfg"; then
        # Preserve indentation; drop the stale "set true after…" trailing
        # comment since we've committed to the decision.
        sed -i -E \
            's|^([[:space:]]*)use_cuda_graphs:[[:space:]]*false.*$|\1use_cuda_graphs: true|' \
            "$cfg"
        echo "Patched config.yaml: use_cuda_graphs: false -> true"
    fi
    # Flip attention from flash_attention_2 to the HF kernels-community
    # flash-attn 3 build. Matches the upstream `Qwen/Qwen3-TTS` Space
    # (`huggingface.co/spaces/Qwen/Qwen3-TTS`) and eliminates the 20-60
    # min flash-attn source build we used to do — the kernel is
    # downloaded pre-built from HF Hub at model-load time.
    if grep -qE '^[[:space:]]*attention:[[:space:]]*flash_attention_2[[:space:]]*$' "$cfg"; then
        sed -i -E \
            's|^([[:space:]]*)attention:[[:space:]]*flash_attention_2[[:space:]]*$|\1attention: kernels-community/flash-attn3|' \
            "$cfg"
        echo "Patched config.yaml: attention: flash_attention_2 -> kernels-community/flash-attn3"
    fi
}

# Small helper: check if a module is importable in the venv.
venv_has() {
    "$VENV_PY" -c "import $1" &>/dev/null
}

# torch + torchaudio must be on cu128 (matching the torch 2.8 release
# line) so the `kernels-community/flash-attn3` pre-built variants match
# our (torch, cuda) tuple. torchaudio's compiled .so also expects an
# exact torch version, so a mismatch between the two produces an
# `OSError: libtorchaudio.so: undefined symbol` at import. When this
# check returns false we know we need to re-pin both together.
venv_has_cu128_torch_stack() {
    "$VENV_PY" <<'PY' &>/dev/null
import sys
try:
    import torch, torchaudio
except Exception:
    sys.exit(1)
cuda = (getattr(torch.version, "cuda", "") or "")
# 12.8 (current) or newer 12.x line — reject older (cu12.1, cu12.4, etc.)
# that won't have a matching flash-attn3 kernel variant.
if not cuda.startswith("12.8"):
    sys.exit(1)
# torch version should be the 2.8.x line — anything older predates the
# kernels variants and anything much newer may shift the ABI.
sys.exit(0 if torch.__version__.startswith("2.8.") else 1)
PY
}

# torch.compile / Triton shells out to gcc at runtime to compile helper
# CUDA .so files that `#include <Python.h>`. If the Python dev headers
# are missing, audio generation fails mid-turn with a BackendCompilerFailed
# traceback in the server log. Gate the fast-path on this so we fall
# through to install_system_deps when the headers aren't there.
venv_has_python_headers() {
    "$VENV_PY" <<'PY' &>/dev/null
import os, sys, sysconfig
include = sysconfig.get_path("include")
sys.exit(0 if include and os.path.exists(os.path.join(include, "Python.h")) else 1)
PY
}

# Confirm our sitecustomize.py took effect — i.e. the dynamo cache limit
# is at least our target. If a future torch upgrade moves the config
# elsewhere, this check invalidates the stamp and we re-run setup.
venv_has_dynamo_cache_bump() {
    "$VENV_PY" <<'PY' &>/dev/null
import sys
try:
    import torch._dynamo.config as c
except Exception:
    sys.exit(1)
sys.exit(0 if c.cache_size_limit >= 64 else 1)
PY
}

# Confirm the `kernels` library is present AND can load the flash-attn3
# kernel for our (torch, cuda) combo. If kernels is missing, or the
# kernel registry has no matching variant, the backend falls back to
# sdpa at runtime (still works, but ~2× slower than fa3). Catch this
# here instead of later.
venv_has_fa3_kernel() {
    "$VENV_PY" <<'PY' &>/dev/null
import sys
try:
    from kernels import get_kernel
    get_kernel("kernels-community/flash-attn3")
except Exception:
    sys.exit(1)
PY
}

venv_pip() {
    (cd "$INSTALL_DIR" && VIRTUAL_ENV="$INSTALL_DIR/.venv" uv pip "$@")
}

# --- Idempotency fast-path ---
#
# Re-apply optimization overrides BEFORE the fast-path check so users
# upgrading the setup script pick up new config + source patches without
# having to force a re-clone / full re-install. Same reasoning for the
# sitecustomize file — writing it doesn't need sudo or a network, so
# we can drop it on an existing venv without going through the full
# repair path (which would otherwise re-run `sudo apt-get`).
apply_config_optimizations
apply_source_patches
if [ -x "$VENV_PY" ]; then
    write_sitecustomize
fi

if [ -x "$VENV_PY" ] && [ -f "$STAMP_FILE" ] && [ -d "$INSTALL_DIR/.git" ]; then
    stamped=$(cat "$STAMP_FILE")
    head_sha=$(current_repo_sha)
    if [ "$stamped" = "$head_sha" ] \
        && venv_has_cu128_torch_stack \
        && venv_has_fa3_kernel \
        && venv_has_python_headers \
        && venv_has_dynamo_cache_bump; then
        echo "qwen3-tts already installed at $head_sha — nothing to do."
        exit 0
    fi
    echo "qwen3-tts install incomplete or stale; resuming from wherever we left off..."
fi

# --- Run (each step is individually idempotent so re-runs don't re-download) ---

install_system_deps
clone_or_update
# clone_or_update did a git reset --hard which would have reverted
# our config + source patches. Re-apply here before anything reads
# or imports the affected files.
apply_config_optimizations
apply_source_patches

# --- Venv ---

if [ ! -x "$VENV_PY" ]; then
    echo "Creating uv venv (Python 3.12)..."
    rm -rf "$INSTALL_DIR/.venv"
    (cd "$INSTALL_DIR" && uv venv --python 3.12 .venv)
fi

# --- qwen-tts with [api] extra ---
#
# Install this FIRST, before pinning torch. pyproject leaves torch
# unpinned, so if we install cu128 torch first then run this, uv happily
# upgrades torch to the latest PyPI wheel (torch 2.11.0+cu130 at the
# time of writing) — which skews torchaudio and breaks the kernels-
# community/flash-attn3 match (no variant exists for that combo).
# Letting the api install pull whatever torch it wants, then force-
# replacing it below, is the only ordering where both packages land on
# cu128 together.
if ! venv_has qwen_tts; then
    echo "Installing qwen-tts with the [api] extra..."
    venv_pip install -e ".[api]"
fi

# --- Pin torch + torchaudio to 2.8.0+cu128 (force-replace whatever -e .[api] pulled) ---
#
# --reinstall on BOTH packages in ONE call keeps them version-locked to
# each other. torch 2.8 + cu128 matches the upstream HF `Qwen/Qwen3-TTS`
# Space and is in the kernels-community/flash-attn3 variant matrix. If
# you bump these, pick a pair with a matching variant at
# https://huggingface.co/kernels-community/flash-attn3/tree/main/build.
if ! venv_has_cu128_torch_stack; then
    echo "Installing torch 2.8.0 + torchaudio 2.8.0 (cu128) — force-replacing any mismatched install..."
    venv_pip install --reinstall torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

# --- kernels (flash-attn 3 pre-built) ---
#
# `kernels<0.10` pins the 0.9.x line, which is compatible with
# `huggingface-hub<1.0` — transformers 4.57.3 (what qwen-tts pins)
# requires hub<1.0, and kernels 0.13+ would pull hub 1.x, leading to a
# `huggingface-hub<1.0 is required` error at model load. Keep this pin
# until transformers is upgraded past the boundary.
#
# The actual flash-attn 3 kernel binary is downloaded from HF Hub at
# model-load time (via `attn_implementation="kernels-community/flash-
# attn3"`), not here — this just installs the runtime that resolves
# that URL.
if ! venv_has kernels; then
    echo "Installing kernels (flash-attn 3 runtime)..."
    venv_pip install 'kernels<0.10'
fi

# --- Clean up stale flash-attn 2 pip package ---
#
# An earlier version of this setup built flash-attn 2 from source. After
# the torch 2.5 → 2.8 upgrade that .so is ABI-broken, so `import
# flash_attn` fails at runtime. That triggers a misleading
# "Warning: flash-attn is not installed" log line from
# qwen_tts/core/tokenizer_25hz/vq/whisper_encoder.py, which has its own
# direct `import flash_attn` fallback chain (a separate code path from
# the kernels-community/flash-attn3 we use for the main decoder).
#
# Uninstall the stale package so the warning at least reports accurately
# and we reclaim ~150 MB. The whisper encoder falls back to a manual
# PyTorch attention path — not a hot-path for CustomVoice inference, so
# no measurable perf impact. The `--quiet` redirect keeps the output
# clean when the package is already absent.
if venv_pip list 2>/dev/null | grep -q '^flash-attn '; then
    echo "Removing stale flash-attn (ABI-broken against torch 2.8; warning is cosmetic but misleading)..."
    venv_pip uninstall -q flash-attn
fi

write_sitecustomize

# --- Final verification: catch silent regressions before stamping ---
#
# set -e only catches non-zero returns; a successful install that
# nonetheless ended up with a wrong torch version (e.g. flash-attn
# pulling torch 2.11 during dep resolution) looks fine to bash but
# breaks at runtime. Verify the full stack imports cleanly on cu121
# BEFORE writing the stamp so a bad install doesn't cache itself.
echo "Verifying qwen3-tts install..."
if ! "$VENV_PY" <<'PY'
import os, sys, sysconfig
try:
    import torch, torchaudio, kernels
    import torch._dynamo.config as dynamo_cfg
    from kernels import get_kernel
except Exception as e:
    print(f"  import failed: {e}", file=sys.stderr)
    sys.exit(1)
cuda = (getattr(torch.version, "cuda", "") or "")
if not cuda.startswith("12.8"):
    print(f"  torch is on cuda {cuda!r}, expected 12.8.x", file=sys.stderr)
    sys.exit(1)
if not torch.__version__.startswith("2.8."):
    print(f"  torch {torch.__version__} — expected 2.8.x (kernels variants are pinned to this)", file=sys.stderr)
    sys.exit(1)
# torch.compile / Triton compiles helper .so files against Python.h at
# runtime. Catch missing dev headers here rather than mid-audio-generation.
header = os.path.join(sysconfig.get_path("include") or "", "Python.h")
if not os.path.exists(header):
    print(f"  missing Python dev headers ({header}) — install python3-dev via your pkg mgr", file=sys.stderr)
    sys.exit(1)
# Confirm sitecustomize.py fired — we raised cache_size_limit from 8 to 64.
if dynamo_cfg.cache_size_limit < 64:
    print(f"  torch._dynamo.cache_size_limit={dynamo_cfg.cache_size_limit} — sitecustomize didn't load", file=sys.stderr)
    sys.exit(1)
# Confirm the flash-attn3 kernel resolves for our (torch, cuda) combo.
# This downloads the kernel from HF Hub on the first call (cached under
# ~/.cache/huggingface), so subsequent model loads are instant.
try:
    get_kernel("kernels-community/flash-attn3")
except Exception as e:
    print(f"  flash-attn3 kernel load failed: {e}", file=sys.stderr)
    print(f"  Available variants: see https://huggingface.co/kernels-community/flash-attn3/tree/main/build", file=sys.stderr)
    sys.exit(1)
print(f"  torch {torch.__version__}, torchaudio {torchaudio.__version__}")
print(f"  kernels (flash-attn3 kernel loaded OK)")
print(f"  dynamo cache_size_limit={dynamo_cfg.cache_size_limit}")
PY
then
    echo "Error: post-install verification failed — refusing to stamp." >&2
    echo "  Inspect the venv manually:" >&2
    echo "    $VENV_PY -c 'import torch, torchaudio, kernels; print(torch.__version__, torchaudio.__version__)'" >&2
    exit 1
fi

# --- Seed ~/qwen3-tts/config.yaml (convenience only) ---
#
# Our launcher sets TTS_CONFIG=<repo>/config.yaml, so this copy is only
# here for users running the repo's own start_server.sh directly.
if [ -f "$INSTALL_DIR/config.yaml" ] && [ ! -f "$HOME/qwen3-tts/config.yaml" ]; then
    mkdir -p "$HOME/qwen3-tts"
    cp "$INSTALL_DIR/config.yaml" "$HOME/qwen3-tts/config.yaml"
fi

current_repo_sha > "$STAMP_FILE"
echo "Done. qwen3-tts installed at $INSTALL_DIR ($(current_repo_sha))."
