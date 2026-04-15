"""Configuration loaded from config.toml, with .env and environment variable overrides.

Priority (highest wins): environment variables > .env > config.toml
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(_PROJECT_ROOT / ".env")


class ConfigError(SystemExit):
    """Raised when a required config value is missing."""


def _load_toml() -> dict:
    path = _PROJECT_ROOT / "config.toml"
    if not path.exists():
        print(f"Error: config.toml not found at {path}", file=sys.stderr)
        raise ConfigError(1)
    with open(path, "rb") as f:
        return tomllib.load(f)


def _get(env_key: str, toml_value: str | int | bool | None) -> str:
    """Resolve a config value: env var > toml. Raises if neither is set."""
    env = os.getenv(env_key)
    if env is not None:
        return env
    if toml_value is not None:
        return str(toml_value)
    raise ConfigError(
        f"Missing config: set '{env_key}' as an environment variable or in config.toml"
    )


def _get_optional(env_key: str, toml_value: str | int | bool | None) -> str | None:
    """Resolve a config value: env var > toml. Returns None if not set."""
    env = os.getenv(env_key)
    if env is not None:
        return env
    if toml_value is not None:
        return str(toml_value)
    return None


@dataclass
class Settings:
    voice_mode: Literal["cloud", "local"]
    input_mode: Literal["push_to_talk", "vad"]

    # Cloud
    openai_api_key: str | None

    # Local server config
    mlx_audio_url: str | None
    mlx_llm_url: str | None
    mlx_llm_server: str | None  # "mlx-vlm", "mlx-lm", or "llamacpp"
    mlx_kv_bits: str | None  # KV cache quantization bits (mlx-vlm only)
    mlx_kv_quant_scheme: str | None  # KV cache quantization scheme (mlx-vlm only)
    llamacpp_preset: str | None  # path to llama-server models preset INI file

    # Cloud model names
    cloud_stt_model: str | None
    cloud_tts_model: str | None
    cloud_llm_model: str | None
    cloud_tts_voice: str | None

    # Local model names
    local_stt_model: str | None
    local_tts_model: str | None
    local_llm_model: str | None
    local_tts_voice: str | None

    # Agent
    agent_instructions: str

    # VAD
    vad_aggressiveness: int
    vad_silence_ms: int
    vad_energy_threshold: int

    # Features
    enable_mcp: bool

    # Display
    show_transcript: bool
    show_metrics: bool

    # Audio
    sample_rate: int

    @property
    def stt_model(self) -> str:
        if self.voice_mode == "local":
            if not self.local_stt_model:
                raise ConfigError("Missing config: local.stt_model in config.toml")
            return self.local_stt_model
        if not self.cloud_stt_model:
            raise ConfigError("Missing config: cloud.stt_model in config.toml")
        return self.cloud_stt_model

    @property
    def tts_model(self) -> str:
        if self.voice_mode == "local":
            if not self.local_tts_model:
                raise ConfigError("Missing config: local.tts_model in config.toml")
            return self.local_tts_model
        if not self.cloud_tts_model:
            raise ConfigError("Missing config: cloud.tts_model in config.toml")
        return self.cloud_tts_model

    @property
    def llm_model(self) -> str:
        if self.voice_mode == "local":
            if not self.local_llm_model:
                section = self.mlx_llm_server or "local"
                raise ConfigError(
                    f"Missing config: llm_model in [local.{section}] in config.toml"
                )
            return self.local_llm_model
        if not self.cloud_llm_model:
            raise ConfigError("Missing config: cloud.llm_model in config.toml")
        return self.cloud_llm_model

    @property
    def tts_voice(self) -> str | None:
        if self.voice_mode == "local":
            return self.local_tts_voice
        return self.cloud_tts_voice


def load_settings() -> Settings:
    t = _load_toml()
    general = t.get("general", {})
    cloud = t.get("cloud", {})
    local = t.get("local", {})
    vad = t.get("vad", {})
    agent = t.get("agent", {})
    audio = t.get("audio", {})
    display = t.get("display", {})

    voice_mode = _get("VOICE_MODE", general.get("voice_mode"))
    input_mode = _get("INPUT_MODE", general.get("input_mode"))

    # Validate mode values
    if voice_mode not in ("cloud", "local"):
        raise ConfigError(
            f"Invalid voice_mode: '{voice_mode}' (must be 'cloud' or 'local')"
        )
    if input_mode not in ("push_to_talk", "vad"):
        raise ConfigError(
            f"Invalid input_mode: '{input_mode}' (must be 'push_to_talk' or 'vad')"
        )

    # Resolve LLM server type first, then load from its subsection
    llm_server = _get_optional("MLX_LLM_SERVER", local.get("llm_server"))
    llm_section = local.get(llm_server, {}) if llm_server else {}

    settings = Settings(
        voice_mode=voice_mode,  # type: ignore[arg-type]
        input_mode=input_mode,  # type: ignore[arg-type]
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        # Local settings (optional when running cloud)
        mlx_audio_url=_get_optional("MLX_AUDIO_URL", local.get("audio_url")),
        mlx_llm_url=_get_optional(
            "MLX_LLM_URL", local.get("llm_url") or local.get("vlm_url")
        ),
        mlx_llm_server=llm_server,
        mlx_kv_bits=_get_optional("MLX_KV_BITS", llm_section.get("kv_bits")),
        mlx_kv_quant_scheme=_get_optional(
            "MLX_KV_QUANT_SCHEME", llm_section.get("kv_quant_scheme")
        ),
        llamacpp_preset=_get_optional("LLAMACPP_PRESET", llm_section.get("preset")),
        # Cloud models (optional when running local)
        cloud_stt_model=_get_optional("CLOUD_STT_MODEL", cloud.get("stt_model")),
        cloud_tts_model=_get_optional("CLOUD_TTS_MODEL", cloud.get("tts_model")),
        cloud_llm_model=_get_optional("CLOUD_LLM_MODEL", cloud.get("llm_model")),
        cloud_tts_voice=_get_optional("CLOUD_TTS_VOICE", cloud.get("tts_voice")),
        # Local models (optional when running cloud)
        local_stt_model=_get_optional("LOCAL_STT_MODEL", local.get("stt_model")),
        local_tts_model=_get_optional("LOCAL_TTS_MODEL", local.get("tts_model")),
        local_llm_model=_get_optional("LOCAL_LLM_MODEL", llm_section.get("llm_model")),
        local_tts_voice=_get_optional("LOCAL_TTS_VOICE", local.get("tts_voice")),
        # Required settings
        agent_instructions=_get("AGENT_INSTRUCTIONS", agent.get("instructions")),
        vad_aggressiveness=int(_get("VAD_AGGRESSIVENESS", vad.get("aggressiveness"))),
        vad_silence_ms=int(_get("VAD_SILENCE_MS", vad.get("silence_ms"))),
        vad_energy_threshold=int(
            _get("VAD_ENERGY_THRESHOLD", vad.get("energy_threshold"))
        ),
        enable_mcp=_get("ENABLE_MCP", general.get("enable_mcp")).lower() == "true",
        show_transcript=_get("SHOW_TRANSCRIPT", display.get("show_transcript")).lower()
        == "true",
        show_metrics=_get("SHOW_METRICS", display.get("show_metrics")).lower()
        == "true",
        sample_rate=int(_get("SAMPLE_RATE", audio.get("sample_rate"))),
    )

    # Validate mode-specific required fields
    try:
        settings.stt_model
        settings.tts_model
        settings.llm_model
    except ConfigError:
        raise
    if settings.voice_mode == "local":
        if not settings.mlx_audio_url:
            raise ConfigError("Missing config: local.audio_url in config.toml")
        if not settings.mlx_llm_url:
            raise ConfigError("Missing config: local.llm_url in config.toml")
        if settings.mlx_llm_server not in ("mlx-vlm", "mlx-lm", "llamacpp"):
            raise ConfigError(
                f"Invalid local.llm_server: '{settings.mlx_llm_server}'"
                " (must be 'mlx-vlm', 'mlx-lm', or 'llamacpp')"
            )
        if settings.mlx_llm_server == "llamacpp":
            if not settings.llamacpp_preset:
                raise ConfigError(
                    "Missing config: preset in [local.llamacpp] in config.toml"
                )
            preset_path = _PROJECT_ROOT / settings.llamacpp_preset
            if not preset_path.exists():
                raise ConfigError(
                    f"llamacpp preset file not found: {preset_path}\n"
                    "Copy models.ini.example to models.ini and customize it."
                )
    if settings.voice_mode == "cloud" and not settings.openai_api_key:
        raise ConfigError("Missing config: OPENAI_API_KEY in .env")

    # Append model-specific instruction snippets
    model_instructions = agent.get("model-instructions", {})
    if model_instructions:
        active_models = [
            settings.stt_model.lower(),
            settings.tts_model.lower(),
            settings.llm_model.lower(),
        ]
        for pattern, text in model_instructions.items():
            if any(pattern in m for m in active_models):
                settings.agent_instructions += str(text)

    return settings
