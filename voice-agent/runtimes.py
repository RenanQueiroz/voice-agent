"""Registry of the local runtimes the app can drive.

Each role (STT / LLM / TTS) has a set of possible runtimes — whisper.cpp
for STT, llama.cpp / mlx-lm / mlx-vlm for LLM, mlx-audio / kokoro-fastapi
/ qwen3-tts for TTS. Users pick one per active local model via the
`runtime` field in models.toml.

This module is the single source of truth for:

- which runtime IDs are valid per role (config validation),
- which OS each runtime supports (catalog filtering, auto-fallback in
  config.py when a user's saved preference is filtered out),
- the Python module to import to check if a runtime is installed
  (`pip_module`, None for binary runtimes like llamacpp/whispercpp and
  for source-installed runtimes like kokoro-fastapi / qwen3-tts),
- the health-check endpoint to poll once the server is launched, plus
  an optional response-body check for servers that return 200 during
  warmup (Qwen3-TTS).

Adding another local TTS runtime is a matter of adding an entry here
and teaching `ServerManager._start_tts` to dispatch on the new
`runtime` value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx

Role = Literal["stt", "llm", "tts"]


def _default_ready_check(resp: httpx.Response) -> bool:
    """Most servers are ready as soon as they return any 2xx on the health path."""
    return resp.status_code == 200


@dataclass(frozen=True)
class Runtime:
    id: str
    role: Role
    supported_os: frozenset[str]  # members of platform_info.OS
    pip_module: str | None  # None for binary runtimes (llamacpp / whispercpp)
    pip_package: str | None  # package to `uv pip install` if missing
    health_path: str  # relative path polled by ServerManager._wait_ready
    # Decides "ready" from the health response. Default accepts any 200.
    # Qwen3-TTS needs a body check because /health returns 200 during
    # torch.compile warmup with `{"status": "initializing"}`.
    ready_check: Callable[[httpx.Response], bool] = field(default=_default_ready_check)


RUNTIMES: dict[str, Runtime] = {
    "whispercpp": Runtime(
        id="whispercpp",
        role="stt",
        supported_os=frozenset({"darwin", "linux"}),
        pip_module=None,
        pip_package=None,
        health_path="/",
    ),
    "llamacpp": Runtime(
        id="llamacpp",
        role="llm",
        supported_os=frozenset({"darwin", "linux"}),
        pip_module=None,
        pip_package=None,
        health_path="/health",
    ),
    "mlx-lm": Runtime(
        id="mlx-lm",
        role="llm",
        supported_os=frozenset({"darwin"}),
        pip_module="mlx_lm",
        pip_package="mlx-lm",
        health_path="/v1/models",
    ),
    "mlx-vlm": Runtime(
        id="mlx-vlm",
        role="llm",
        supported_os=frozenset({"darwin"}),
        pip_module="mlx_vlm",
        pip_package="mlx-vlm",
        health_path="/v1/models",
    ),
    "mlx-audio": Runtime(
        id="mlx-audio",
        role="tts",
        supported_os=frozenset({"darwin"}),
        pip_module="mlx_audio",
        pip_package="mlx-audio[server,tts]",
        health_path="/v1/models",
    ),
    # Source-installed FastAPI TTS servers (Linux + CUDA). Neither is a
    # pip module we import in-process — each ships as a checkout with its
    # own uv venv, driven by a setup-<runtime>.sh script. ServerManager
    # launches `<repo>/.venv/bin/python -m <entrypoint>`.
    "kokoro-fastapi": Runtime(
        id="kokoro-fastapi",
        role="tts",
        supported_os=frozenset({"linux"}),
        pip_module=None,
        pip_package=None,
        health_path="/health",
    ),
    "qwen3-tts": Runtime(
        id="qwen3-tts",
        role="tts",
        supported_os=frozenset({"linux"}),
        pip_module=None,
        pip_package=None,
        health_path="/health",
        # Qwen3 returns 200 while torch.compile warmup is still running,
        # with `{"status": "initializing"}` in the body. Only treat the
        # server as ready once the backend reports "healthy" (set to True
        # after the 3-pass warmup finishes in optimized_backend.py).
        ready_check=lambda r: (
            r.status_code == 200
            and (
                r.json().get("status")
                if r.headers.get("content-type", "").startswith("application/json")
                else None
            )
            == "healthy"
        ),
    ),
}


def runtimes_for_role(role: Role) -> list[str]:
    """All registered runtime IDs that apply to the given role."""
    return [r.id for r in RUNTIMES.values() if r.role == role]


def is_runtime_supported(runtime_id: str, os_tag: str) -> bool:
    """True when the runtime is both registered and runs on `os_tag`."""
    rt = RUNTIMES.get(runtime_id)
    return rt is not None and os_tag in rt.supported_os


def get_runtime(runtime_id: str) -> Runtime:
    """Fetch a registered runtime. Raises KeyError if unknown — callers
    should have validated via config-time parsing first."""
    return RUNTIMES[runtime_id]
