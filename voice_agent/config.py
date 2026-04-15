"""Configuration loaded from config.toml, with .env and environment variable overrides.

Priority (highest wins): environment variables > .env > config.toml
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(_PROJECT_ROOT / ".env")


def _load_toml() -> dict:
    path = _PROJECT_ROOT / "config.toml"
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def _get(env_key: str, toml_value: str | int | None, default: str) -> str:
    """Resolve a config value: env var > toml > default."""
    env = os.getenv(env_key)
    if env is not None:
        return env
    if toml_value is not None:
        return str(toml_value)
    return default


@dataclass
class Settings:
    voice_mode: Literal["cloud", "local"]
    input_mode: Literal["push_to_talk", "vad"]

    # Cloud
    openai_api_key: str | None

    # Local server URLs
    mlx_audio_url: str
    mlx_vlm_url: str

    # Cloud model names
    cloud_stt_model: str
    cloud_tts_model: str
    cloud_llm_model: str
    cloud_tts_voice: str

    # Local model names
    local_stt_model: str
    local_tts_model: str
    local_llm_model: str
    local_tts_voice: str

    # Agent
    agent_instructions: str

    # VAD
    vad_aggressiveness: int
    vad_silence_ms: int
    vad_energy_threshold: int

    # Audio
    sample_rate: int

    @property
    def stt_model(self) -> str:
        return (
            self.local_stt_model if self.voice_mode == "local" else self.cloud_stt_model
        )

    @property
    def tts_model(self) -> str:
        return (
            self.local_tts_model if self.voice_mode == "local" else self.cloud_tts_model
        )

    @property
    def llm_model(self) -> str:
        return (
            self.local_llm_model if self.voice_mode == "local" else self.cloud_llm_model
        )

    @property
    def tts_voice(self) -> str:
        return (
            self.local_tts_voice if self.voice_mode == "local" else self.cloud_tts_voice
        )


def load_settings() -> Settings:
    t = _load_toml()
    general = t.get("general", {})
    cloud = t.get("cloud", {})
    local = t.get("local", {})
    vad = t.get("vad", {})
    agent = t.get("agent", {})
    audio = t.get("audio", {})

    return Settings(
        voice_mode=_get("VOICE_MODE", general.get("voice_mode"), "cloud"),  # type: ignore[arg-type]
        input_mode=_get("INPUT_MODE", general.get("input_mode"), "push_to_talk"),  # type: ignore[arg-type]
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        mlx_audio_url=_get(
            "MLX_AUDIO_URL", local.get("audio_url"), "http://localhost:8000"
        ),
        mlx_vlm_url=_get("MLX_VLM_URL", local.get("vlm_url"), "http://localhost:8080"),
        cloud_stt_model=_get(
            "CLOUD_STT_MODEL", cloud.get("stt_model"), "gpt-4o-transcribe"
        ),
        cloud_tts_model=_get(
            "CLOUD_TTS_MODEL", cloud.get("tts_model"), "gpt-4o-mini-tts"
        ),
        cloud_llm_model=_get("CLOUD_LLM_MODEL", cloud.get("llm_model"), "gpt-4o-mini"),
        cloud_tts_voice=_get("CLOUD_TTS_VOICE", cloud.get("tts_voice"), "alloy"),
        local_stt_model=_get(
            "LOCAL_STT_MODEL",
            local.get("stt_model"),
            "mlx-community/whisper-large-v3-turbo-asr-fp16",
        ),
        local_tts_model=_get(
            "LOCAL_TTS_MODEL", local.get("tts_model"), "mlx-community/Kokoro-82M-bf16"
        ),
        local_llm_model=_get(
            "LOCAL_LLM_MODEL",
            local.get("llm_model"),
            "mlx-community/gemma-4-e4b-it-4bit",
        ),
        local_tts_voice=_get("LOCAL_TTS_VOICE", local.get("tts_voice"), "af_heart"),
        agent_instructions=_get(
            "AGENT_INSTRUCTIONS",
            agent.get("instructions"),
            "You are a helpful voice assistant. Be concise and conversational.",
        ),
        vad_aggressiveness=int(
            _get("VAD_AGGRESSIVENESS", vad.get("aggressiveness"), "2")
        ),
        vad_silence_ms=int(_get("VAD_SILENCE_MS", vad.get("silence_ms"), "500")),
        vad_energy_threshold=int(_get("VAD_ENERGY_THRESHOLD", vad.get("energy_threshold"), "40")),
        sample_rate=int(_get("SAMPLE_RATE", audio.get("sample_rate"), "24000")),
    )
