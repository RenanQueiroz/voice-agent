"""Registry of the local runtimes the app can drive.

Each role (STT / LLM / TTS) has a set of possible runtimes — whisper.cpp /
onnx-asr for STT, llama.cpp / mlx-vlm for LLM, mlx-audio / kokoro-fastapi
/ qwen3-tts / supertonic for TTS. Users pick one per active local model via the
`runtime` field in models.toml.

This module is the single source of truth for:

- which runtime IDs are valid per role (config validation),
- which OS each runtime supports (catalog filtering, auto-fallback in
  config.py when a user's saved preference is filtered out),
- the Python module to import to check if a runtime is installed
  (`pip_module`, None for binary runtimes like llamacpp/whispercpp and
  for isolated server runtimes like supertonic / kokoro-fastapi / qwen3-tts),
- whether the runtime launches a supervised server process or runs in-process,
- whether the runtime needs a local URL in config.toml,
- the health-check endpoint to poll once the server is launched, plus
  an optional response-body check for servers whose 200 response still
  needs semantic readiness validation (Qwen3-TTS, Supertonic).

Adding another local runtime is a matter of adding an entry here and teaching
`ServerManager` / `providers.create_pipeline` to dispatch on the new `runtime`
value.
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


def _json_status_ready(expected: str) -> Callable[[httpx.Response], bool]:
    def check(resp: httpx.Response) -> bool:
        if resp.status_code != 200:
            return False
        if not resp.headers.get("content-type", "").startswith("application/json"):
            return False
        return resp.json().get("status") == expected

    return check


@dataclass(frozen=True)
class Runtime:
    id: str
    role: Role
    supported_os: frozenset[str]  # members of platform_info.OS
    pip_module: str | None  # None for binary runtimes (llamacpp / whispercpp)
    pip_package: str | None  # package to `uv pip install` if missing
    health_path: str  # relative path polled by ServerManager._wait_ready
    managed_process: bool = True  # False for in-process runtimes like onnx-asr
    requires_url: bool = True  # False when no [local].<role>_url is needed
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
    "onnx-asr": Runtime(
        id="onnx-asr",
        role="stt",
        supported_os=frozenset({"darwin", "linux"}),
        pip_module="onnx_asr",
        pip_package="onnx-asr[cpu,hub]",
        health_path="",
        managed_process=False,
        requires_url=False,
    ),
    "llamacpp": Runtime(
        id="llamacpp",
        role="llm",
        supported_os=frozenset({"darwin", "linux"}),
        pip_module=None,
        pip_package=None,
        health_path="/health",
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
    # Isolated FastAPI TTS servers. They are not imported in-process — each
    # runs from its own uv venv, driven by a setup-<runtime>.sh script.
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
        ready_check=_json_status_ready("healthy"),
    ),
    "supertonic": Runtime(
        id="supertonic",
        role="tts",
        supported_os=frozenset({"darwin", "linux"}),
        # Official `supertonic serve` runs from its own uv venv. Keep it out
        # of the main app environment so ONNX Runtime wheel constraints don't
        # leak into the Python 3.14 project venv.
        pip_module=None,
        pip_package=None,
        health_path="/v1/health",
        ready_check=_json_status_ready("ok"),
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
