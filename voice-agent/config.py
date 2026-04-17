"""Configuration loaded from config.toml + preferences.toml, with .env and
environment variable overrides.

Instead of a single `voice_mode` toggle, the user defines a catalog of
models per role (STT / LLM / TTS). Each catalog entry is a `ModelConfig`
that marks itself `provider = "cloud" | "local"` and carries the fields
relevant to that role/provider (e.g., `voice` for TTS, `server/preset/
kv_bits` for local LLMs). A gitignored `preferences.toml` records which
entry is active per role; on first run (or bad preference) we fall back
to the first catalog entry.

Priority (highest wins): environment variables > .env > config.toml.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from .preferences import load_preferences

_PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(_PROJECT_ROOT / ".env")


class ConfigError(SystemExit):
    """Raised when a required config value is missing or invalid."""


def _load_toml() -> dict:
    path = _PROJECT_ROOT / "config.toml"
    if not path.exists():
        print(f"Error: config.toml not found at {path}", file=sys.stderr)
        raise ConfigError(1)
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_models_toml() -> dict:
    """Load `models.toml` (the role catalogs). Missing file is an error —
    the app needs at least one model per role."""
    path = _PROJECT_ROOT / "models.toml"
    if not path.exists():
        raise ConfigError(
            f"models.toml not found at {path}. This file holds the "
            "[[stt]] / [[llm]] / [[tts]] catalogs — see the example in the repo."
        )
    with open(path, "rb") as f:
        return tomllib.load(f)


def _get(env_key: str, toml_value: Any) -> str:
    env = os.getenv(env_key)
    if env is not None:
        return env
    if toml_value is not None:
        return str(toml_value)
    raise ConfigError(
        f"Missing config: set '{env_key}' as an environment variable or in config.toml"
    )


def _get_optional(env_key: str, toml_value: Any) -> str | None:
    env = os.getenv(env_key)
    if env is not None:
        return env
    if toml_value is not None:
        return str(toml_value)
    return None


Role = Literal["stt", "llm", "tts"]
Provider = Literal["cloud", "local"]
LLMServer = Literal["mlx-vlm", "mlx-lm", "llamacpp"]
Vendor = Literal["openai", "gemini"]


HOSTED_TOOL_NAMES = {"web_search", "code_interpreter", "file_search"}
VENDOR_NAMES = {"openai", "gemini"}


@dataclass
class ModelConfig:
    """One entry in a role's model catalog."""

    name: str  # unique, user-facing; appears in the Switch modal
    role: Role
    provider: Provider
    model: str  # cloud model id or local model spec (mlx path / whisper.cpp name / preset alias)

    # Cloud-only: which API vendor serves this model. Defaults to OpenAI when
    # unset; set to "gemini" to route the request to Google's Generative
    # Language API (Gemini has an OpenAI-compatible endpoint for LLMs, and
    # we wrap Gemini's native TTS API in a custom model class).
    vendor: Vendor | None = None

    # TTS-only
    voice: str | None = None

    # Local-LLM-only
    server: LLMServer | None = None
    preset: str | None = (
        None  # llamacpp preset path (llamacpp-models.ini), relative to project root
    )
    kv_bits: str | None = None
    kv_quant_scheme: str | None = None
    audio_input: bool = False  # LLM accepts audio directly

    # Cloud-LLM-only: OpenAI-hosted tools (see providers._hosted_tools)
    hosted_tools: list[str] = field(default_factory=list)
    # For hosted_tools = ["file_search"]: required vector store IDs + optional
    # max_num_results. Ignored if file_search isn't enabled.
    file_search_vector_stores: list[str] = field(default_factory=list)
    file_search_max_results: int | None = None

    @property
    def display_name(self) -> str:
        """Human-readable label: `"{name} ({provider})"`. Used in the Switch
        modal and in per-turn metrics. The raw `name` remains the stable key
        for preferences.toml lookups."""
        return f"{self.name} ({self.provider})"


@dataclass
class ShellConfig:
    """Optional shell-command tool. Every invocation requires user approval
    unless `auto_approve` is set — in that case commands run silently the
    moment the agent calls them. Use with care."""

    enabled: bool = False
    auto_approve: bool = False
    timeout_seconds: int = 30
    max_output_bytes: int = 10_000
    cwd: str | None = None  # relative to project root; None = project root


@dataclass
class Settings:
    input_mode: Literal["push_to_talk", "vad"]
    openai_api_key: str | None
    gemini_api_key: str | None

    # Local server endpoints. Required only for roles whose active model is
    # local. Validated lazily.
    stt_url: str | None
    tts_url: str | None
    llm_url: str | None

    # Catalogs
    stt_models: list[ModelConfig] = field(default_factory=list)
    llm_models: list[ModelConfig] = field(default_factory=list)
    tts_models: list[ModelConfig] = field(default_factory=list)

    # Active selection (mutates at runtime via the Switch modal)
    active_stt: str = ""
    active_llm: str = ""
    active_tts: str = ""

    # VAD
    vad_threshold: float = 0.5
    vad_silence_ms: int = 500

    # Features
    enable_mcp: bool = False

    # Display
    show_transcript: bool = True
    show_metrics: bool = True

    # Audio
    sample_rate: int = 24000

    # Agent
    agent_instructions: str = ""
    tool_call_filler: str | None = None
    model_instruction_snippets: dict[str, str] = field(default_factory=dict)

    # Shell tool (requires user approval per invocation)
    shell: ShellConfig = field(default_factory=ShellConfig)

    # ── Active-model lookups ─────────────────────────────────

    @property
    def stt(self) -> ModelConfig:
        return self._find("stt", self.active_stt, self.stt_models)

    @property
    def llm(self) -> ModelConfig:
        return self._find("llm", self.active_llm, self.llm_models)

    @property
    def tts(self) -> ModelConfig:
        return self._find("tts", self.active_tts, self.tts_models)

    @staticmethod
    def _find(role: str, name: str, catalog: list[ModelConfig]) -> ModelConfig:
        for m in catalog:
            if m.name == name:
                return m
        raise ConfigError(
            f"Active {role.upper()} model '{name}' is not in the catalog."
        )

    # ── Active-model effective string values (for agent instructions etc.) ─

    @property
    def stt_model(self) -> str:
        return self.stt.model

    @property
    def llm_model(self) -> str:
        return self.llm.model

    @property
    def tts_model(self) -> str:
        return self.tts.model

    @property
    def tts_voice(self) -> str | None:
        return self.tts.voice

    def active_names(self) -> dict[str, str]:
        return {"stt": self.active_stt, "llm": self.active_llm, "tts": self.active_tts}


def _parse_catalog(role: Role, entries: list[dict]) -> list[ModelConfig]:
    if not entries:
        raise ConfigError(
            f"Missing config: at least one [[{role}]] entry required in config.toml"
        )
    models: list[ModelConfig] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ConfigError(
                f"[[{role}]] entry #{i + 1} is missing a string 'name' field"
            )
        if name in seen:
            raise ConfigError(f"Duplicate [[{role}]] name: '{name}'")
        seen.add(name)

        provider = entry.get("provider")
        if provider not in ("cloud", "local"):
            raise ConfigError(
                f"[[{role}]] '{name}' must set provider = 'cloud' | 'local' (got {provider!r})"
            )

        model = entry.get("model")
        if not model or not isinstance(model, str):
            raise ConfigError(f"[[{role}]] '{name}' is missing a string 'model' field")

        config = ModelConfig(name=name, role=role, provider=provider, model=model)

        vendor = entry.get("vendor")
        if vendor is not None:
            if provider != "cloud":
                raise ConfigError(
                    f"[[{role}]] '{name}' has vendor = {vendor!r} but "
                    f"provider = {provider!r}. The 'vendor' field only "
                    "applies to cloud models."
                )
            if vendor not in VENDOR_NAMES:
                raise ConfigError(
                    f"[[{role}]] '{name}' has unknown vendor = {vendor!r}. "
                    f"Allowed: {sorted(VENDOR_NAMES)}."
                )
            config.vendor = vendor

        if role == "tts":
            voice = entry.get("voice")
            if voice is not None and not isinstance(voice, str):
                raise ConfigError(
                    f"[[tts]] '{name}' has a non-string 'voice': {voice!r}"
                )
            config.voice = voice

        if role == "llm":
            hosted_tools = entry.get("hosted_tools") or []
            if hosted_tools:
                if provider != "cloud":
                    raise ConfigError(
                        f"[[llm]] '{name}' has hosted_tools but provider is "
                        f"'{provider}' — hosted tools only work with cloud OpenAI models."
                    )
                if config.vendor is not None and config.vendor != "openai":
                    raise ConfigError(
                        f"[[llm]] '{name}' has hosted_tools but vendor = "
                        f"{config.vendor!r}. Hosted tools (web_search, "
                        "code_interpreter, file_search) are OpenAI-only — "
                        "they don't work with Gemini or other vendors."
                    )
                if not isinstance(hosted_tools, list) or not all(
                    isinstance(t, str) for t in hosted_tools
                ):
                    raise ConfigError(
                        f"[[llm]] '{name}' hosted_tools must be a list of strings"
                    )
                unknown = [t for t in hosted_tools if t not in HOSTED_TOOL_NAMES]
                if unknown:
                    raise ConfigError(
                        f"[[llm]] '{name}' has unknown hosted_tools: {unknown}. "
                        f"Allowed: {sorted(HOSTED_TOOL_NAMES)}"
                    )
                config.hosted_tools = list(hosted_tools)
            vs = entry.get("file_search_vector_stores") or []
            if vs:
                if not isinstance(vs, list) or not all(isinstance(v, str) for v in vs):
                    raise ConfigError(
                        f"[[llm]] '{name}' file_search_vector_stores must be a list of strings"
                    )
                config.file_search_vector_stores = list(vs)
            fsmax = entry.get("file_search_max_results")
            if fsmax is not None:
                config.file_search_max_results = int(fsmax)
            if (
                "file_search" in config.hosted_tools
                and not config.file_search_vector_stores
            ):
                raise ConfigError(
                    f"[[llm]] '{name}' has hosted_tools = ['file_search', ...] "
                    f"but no file_search_vector_stores set."
                )

        if role == "llm" and provider == "local":
            server = entry.get("server")
            if server not in ("mlx-vlm", "mlx-lm", "llamacpp"):
                raise ConfigError(
                    f"[[llm]] '{name}' must set server = 'mlx-vlm' | 'mlx-lm' | 'llamacpp' "
                    f"(got {server!r})"
                )
            config.server = server
            config.audio_input = bool(entry.get("audio_input", False))
            kv_bits = entry.get("kv_bits")
            config.kv_bits = str(kv_bits) if kv_bits is not None else None
            kv_scheme = entry.get("kv_quant_scheme")
            config.kv_quant_scheme = str(kv_scheme) if kv_scheme is not None else None
            preset = entry.get("preset")
            if server == "llamacpp":
                if not preset or not isinstance(preset, str):
                    raise ConfigError(
                        f"[[llm]] '{name}' (server=llamacpp) needs a 'preset' "
                        f"path, e.g. preset = 'llamacpp-models.ini'"
                    )
                preset_path = _PROJECT_ROOT / preset
                if not preset_path.exists():
                    raise ConfigError(
                        f"[[llm]] '{name}' preset file not found: {preset_path}. "
                        "Copy llamacpp-models.ini.example to llamacpp-models.ini "
                        "and customize it."
                    )
                config.preset = preset

        models.append(config)
    return models


def _resolve_active(role: Role, pref: str | None, catalog: list[ModelConfig]) -> str:
    if pref:
        for m in catalog:
            if m.name == pref:
                return pref
        print(
            f"Warning: preferences.toml active {role} '{pref}' not found in "
            f"catalog; falling back to '{catalog[0].name}'.",
            file=sys.stderr,
        )
    return catalog[0].name


def load_settings() -> Settings:
    t = _load_toml()

    # Detect the old schema and error with a migration hint.
    if "cloud" in t or ("local" in t and "stt_model" in t.get("local", {})):
        raise ConfigError(
            "config.toml uses the old voice_mode/cloud/local schema.\n"
            "The app now uses per-role catalogs — see the top of the updated "
            "config.toml in this repo for the new format."
        )
    # And detect the previous interim format where model catalogs lived in
    # config.toml itself.
    if any(role in t for role in ("stt", "llm", "tts")):
        raise ConfigError(
            "config.toml still contains [[stt]] / [[llm]] / [[tts]] entries. "
            "These have moved to models.toml — cut them out of config.toml and "
            "paste them into models.toml (see models.toml in the repo for the "
            "new layout)."
        )

    general = t.get("general", {})
    local = t.get("local", {})
    vad = t.get("vad", {})
    agent = t.get("agent", {})
    audio = t.get("audio", {})
    display = t.get("display", {})
    shell_cfg = t.get("shell", {})

    input_mode = _get("INPUT_MODE", general.get("input_mode"))
    if input_mode not in ("push_to_talk", "vad"):
        raise ConfigError(
            f"Invalid input_mode: '{input_mode}' (must be 'push_to_talk' or 'vad')"
        )

    models_toml = _load_models_toml()
    stt_models = _parse_catalog("stt", list(models_toml.get("stt", [])))
    llm_models = _parse_catalog("llm", list(models_toml.get("llm", [])))
    tts_models = _parse_catalog("tts", list(models_toml.get("tts", [])))

    prefs = load_preferences()
    active_stt = _resolve_active("stt", prefs.get("stt"), stt_models)
    active_llm = _resolve_active("llm", prefs.get("llm"), llm_models)
    active_tts = _resolve_active("tts", prefs.get("tts"), tts_models)

    settings = Settings(
        input_mode=input_mode,  # type: ignore[arg-type]
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        stt_url=_get_optional("STT_URL", local.get("stt_url")),
        tts_url=_get_optional("TTS_URL", local.get("tts_url")),
        llm_url=_get_optional("LLM_URL", local.get("llm_url")),
        stt_models=stt_models,
        llm_models=llm_models,
        tts_models=tts_models,
        active_stt=active_stt,
        active_llm=active_llm,
        active_tts=active_tts,
        vad_threshold=float(_get("VAD_THRESHOLD", vad.get("threshold"))),
        vad_silence_ms=int(_get("VAD_SILENCE_MS", vad.get("silence_ms"))),
        enable_mcp=_get("ENABLE_MCP", general.get("enable_mcp")).lower() == "true",
        show_transcript=_get("SHOW_TRANSCRIPT", display.get("show_transcript")).lower()
        == "true",
        show_metrics=_get("SHOW_METRICS", display.get("show_metrics")).lower()
        == "true",
        sample_rate=int(_get("SAMPLE_RATE", audio.get("sample_rate"))),
        agent_instructions=_get("AGENT_INSTRUCTIONS", agent.get("instructions")),
        tool_call_filler=_get_optional(
            "TOOL_CALL_FILLER", agent.get("tool_call_filler")
        ),
        model_instruction_snippets={
            str(k): str(v) for k, v in agent.get("model-instructions", {}).items()
        },
        shell=ShellConfig(
            enabled=bool(
                str(
                    _get_optional("SHELL_ENABLED", shell_cfg.get("enabled")) or "false"
                ).lower()
                == "true"
            ),
            auto_approve=bool(
                str(
                    _get_optional("SHELL_AUTO_APPROVE", shell_cfg.get("auto_approve"))
                    or "false"
                ).lower()
                == "true"
            ),
            timeout_seconds=int(shell_cfg.get("timeout_seconds", 30)),
            max_output_bytes=int(shell_cfg.get("max_output_bytes", 10_000)),
            cwd=(
                str(shell_cfg.get("cwd")) if shell_cfg.get("cwd") is not None else None
            ),
        ),
    )

    _validate_active_requirements(settings)

    # Append model-specific instruction snippets for whatever's currently
    # active. Keys are matched case-insensitively against the active model
    # IDs (STT / LLM / TTS). Prefix a key with `re:` to interpret the rest
    # as a regex; otherwise it's a substring match.
    import re as _re

    active_model_names = [
        settings.stt_model.lower(),
        settings.tts_model.lower(),
        settings.llm_model.lower(),
    ]
    for pattern, text in settings.model_instruction_snippets.items():
        if pattern.startswith("re:"):
            try:
                rx = _re.compile(pattern[3:], _re.IGNORECASE)
            except _re.error as e:
                raise ConfigError(
                    f"[agent.model-instructions] invalid regex "
                    f"{pattern!r}: {e}"
                ) from e
            matched = any(rx.search(m) for m in active_model_names)
        else:
            matched = any(pattern.lower() in m for m in active_model_names)
        if matched:
            settings.agent_instructions += text

    return settings


def _validate_active_requirements(settings: Settings) -> None:
    """Check the URL/API-key requirements implied by the current active models."""
    active = [settings.stt, settings.llm, settings.tts]
    needs_openai = any(
        m.provider == "cloud" and (m.vendor is None or m.vendor == "openai")
        for m in active
    )
    needs_gemini = any(m.provider == "cloud" and m.vendor == "gemini" for m in active)
    needs_local_stt = settings.stt.provider == "local"
    needs_local_llm = settings.llm.provider == "local"
    needs_local_tts = settings.tts.provider == "local"

    if needs_openai and not settings.openai_api_key:
        raise ConfigError(
            "Active selection includes an OpenAI cloud model but "
            "OPENAI_API_KEY is not set (check .env)."
        )
    if needs_gemini and not settings.gemini_api_key:
        raise ConfigError(
            "Active selection includes a Gemini cloud model but "
            "GEMINI_API_KEY is not set (check .env)."
        )
    if needs_local_stt and not settings.stt_url:
        raise ConfigError(
            "Active STT model is local but [local].stt_url is not set in config.toml."
        )
    if needs_local_llm and not settings.llm_url:
        raise ConfigError(
            "Active LLM model is local but [local].llm_url is not set in config.toml."
        )
    if needs_local_tts and not settings.tts_url:
        raise ConfigError(
            "Active TTS model is local but [local].tts_url is not set in config.toml."
        )
