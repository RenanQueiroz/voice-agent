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

    # Local server URLs
    mlx_audio_url: str | None
    mlx_vlm_url: str | None

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
                raise ConfigError("Missing config: local.llm_model in config.toml")
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

    settings = Settings(
        voice_mode=voice_mode,  # type: ignore[arg-type]
        input_mode=input_mode,  # type: ignore[arg-type]
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        # Local settings (optional when running cloud)
        mlx_audio_url=_get_optional("MLX_AUDIO_URL", local.get("audio_url")),
        mlx_vlm_url=_get_optional("MLX_VLM_URL", local.get("vlm_url")),
        # Cloud models (optional when running local)
        cloud_stt_model=_get_optional("CLOUD_STT_MODEL", cloud.get("stt_model")),
        cloud_tts_model=_get_optional("CLOUD_TTS_MODEL", cloud.get("tts_model")),
        cloud_llm_model=_get_optional("CLOUD_LLM_MODEL", cloud.get("llm_model")),
        cloud_tts_voice=_get_optional("CLOUD_TTS_VOICE", cloud.get("tts_voice")),
        # Local models (optional when running cloud)
        local_stt_model=_get_optional("LOCAL_STT_MODEL", local.get("stt_model")),
        local_tts_model=_get_optional("LOCAL_TTS_MODEL", local.get("tts_model")),
        local_llm_model=_get_optional("LOCAL_LLM_MODEL", local.get("llm_model")),
        local_tts_voice=_get_optional("LOCAL_TTS_VOICE", local.get("tts_voice")),
        # Required settings
        agent_instructions=_get("AGENT_INSTRUCTIONS", agent.get("instructions")),
        vad_aggressiveness=int(_get("VAD_AGGRESSIVENESS", vad.get("aggressiveness"))),
        vad_silence_ms=int(_get("VAD_SILENCE_MS", vad.get("silence_ms"))),
        vad_energy_threshold=int(
            _get("VAD_ENERGY_THRESHOLD", vad.get("energy_threshold"))
        ),
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
        if not settings.mlx_vlm_url:
            raise ConfigError("Missing config: local.vlm_url in config.toml")
    if settings.voice_mode == "cloud" and not settings.openai_api_key:
        raise ConfigError("Missing config: OPENAI_API_KEY in .env")

    return settings
