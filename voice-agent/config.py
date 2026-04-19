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
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from .platform_info import current_os
from .preferences import load_preferences
from .runtimes import is_runtime_supported, runtimes_for_role

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
Vendor = Literal["openai", "gemini"]


HOSTED_TOOL_NAMES = {"web_search", "code_interpreter", "file_search"}
VENDOR_NAMES = {"openai", "gemini"}

_ENV_REF_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: str) -> str:
    """Expand `${VAR_NAME}` references using the current env (post-dotenv).
    Unknown refs are left literal so the config error surfaces downstream
    (e.g. "api_key is 'missing'") rather than here."""
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


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

    # Cloud-only: per-model API key override. If set, takes precedence over
    # the vendor-wide key (OPENAI_API_KEY / GEMINI_API_KEY). Useful when one
    # key is rate-limited or scoped to a subset of models — e.g. a legacy
    # Gemini key that still works on the LLM endpoint while a newer key
    # handles TTS. Supports `${VAR_NAME}` env-var expansion so real secrets
    # never live in the committed models.toml.
    api_key: str | None = None

    # TTS-only
    voice: str | None = None

    # Local-TTS-only: voice cloning for models that support it (e.g. CSM).
    # `ref_audio` is a path to a short reference recording; `ref_text` is its
    # transcript. Both are sent to mlx-audio's `/v1/audio/speech` as-is. Paths
    # get resolved to absolute relative to the project root, so the spawned
    # mlx-audio server can find the file regardless of its cwd.
    ref_audio: str | None = None
    ref_text: str | None = None

    # Local-TTS-only: seconds of audio mlx-audio buffers before emitting each
    # streaming chunk. Lower = earlier first-byte (good on paper) but in
    # practice <2.0 caused stuttering for us, so providers.py defaults to
    # the server's own 2.0. Override per-entry if you want to try lower.
    streaming_interval: float | None = None

    # TTS-only: free-form style/emotion/pronunciation instruction. Applies to
    # local mlx-audio models that support it (e.g. Qwen3-TTS) and to OpenAI
    # cloud TTS models (gpt-4o-mini-tts and family, which call the parameter
    # `instructions`). Translated to the right wire name per provider. Rejected
    # on Gemini TTS, which steers via inline [audio tags] in the text instead.
    instruct: str | None = None

    # Local-TTS-only: sampling temperature. Lower = more deterministic /
    # consistent speech, higher = more variation. mlx-audio's server default
    # is 0.7; modify this value if the voice feels inconsistent between turns.
    temperature: float | None = None

    # TTS-only: how to chunk the streamed LLM output before sending to TTS.
    #   "sentence" (default) — flush per sentence for fastest first-audio.
    #                          Many requests per turn; good for low-latency
    #                          local models and OpenAI.
    #   "paragraph"          — flush on blank lines; fewer requests per turn
    #                          at the cost of waiting longer for first audio.
    #   "full"               — don't split at all; one TTS request per turn
    #                          after the LLM finishes. Highest latency but
    #                          minimum request count — use this with
    #                          rate-limited providers like Gemini TTS.
    split: str | None = None

    # Local-only: which runtime serves this model. One of the IDs registered
    # in `runtimes.RUNTIMES` (whispercpp for STT; llamacpp / mlx-lm / mlx-vlm
    # for LLM; mlx-audio for TTS). Catalog parsing filters out entries whose
    # runtime isn't supported on the current OS — mlx-* runtimes silently
    # drop off on Linux because there are no Linux wheels for the mlx stack.
    runtime: str | None = None

    # Local-LLM-only (llamacpp preset + mlx-vlm quantization knobs)
    preset: str | None = (
        None  # llamacpp preset path (llamacpp-models.ini), relative to project root
    )
    kv_bits: str | None = None
    kv_quant_scheme: str | None = None
    audio_input: bool = False  # LLM accepts audio directly

    # LLM-only: `reasoning_effort` hint. For Gemini 3 preview and GPT-5 models,
    # `minimal` (or `none` on Gemini 3) dramatically reduces TTFT because the
    # model skips internal reasoning before emitting tokens. Defaults to None,
    # which uses the provider's default (typically `medium` — slow for voice).
    # Accepts: "none" | "minimal" | "low" | "medium" | "high" | "xhigh".
    reasoning_effort: str | None = None

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

    # Notes about catalog/preference changes applied at load time (e.g. an
    # active model filtered out on this OS). The app mounts a NoticeCard
    # for each at startup so the user sees the swap.
    fallback_notes: list[str] = field(default_factory=list)

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

        api_key = entry.get("api_key")
        if api_key is not None:
            if not isinstance(api_key, str):
                raise ConfigError(
                    f"[[{role}]] '{name}' has non-string 'api_key': {api_key!r}"
                )
            if provider != "cloud":
                raise ConfigError(
                    f"[[{role}]] '{name}' has api_key but provider = "
                    f"{provider!r}. Per-model api_key only applies to cloud models."
                )
            config.api_key = _expand_env(api_key)

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

            ref_audio = entry.get("ref_audio")
            ref_text = entry.get("ref_text")
            if ref_audio is not None or ref_text is not None:
                if provider != "local":
                    raise ConfigError(
                        f"[[tts]] '{name}' has ref_audio/ref_text but provider "
                        f"= {provider!r}. Voice cloning is a local-server "
                        "feature (mlx-audio)."
                    )
                if not (ref_audio and ref_text):
                    raise ConfigError(
                        f"[[tts]] '{name}' needs both 'ref_audio' and "
                        "'ref_text' set — they're a pair (reference audio + "
                        "its transcript)."
                    )
                if not isinstance(ref_audio, str) or not isinstance(ref_text, str):
                    raise ConfigError(
                        f"[[tts]] '{name}' ref_audio and ref_text must both be strings."
                    )
                ref_path = Path(ref_audio)
                if not ref_path.is_absolute():
                    ref_path = (_PROJECT_ROOT / ref_audio).resolve()
                if not ref_path.exists():
                    raise ConfigError(
                        f"[[tts]] '{name}' ref_audio file not found: "
                        f"{ref_path}. Path is resolved relative to the "
                        "project root."
                    )
                config.ref_audio = str(ref_path)
                config.ref_text = ref_text

            interval = entry.get("streaming_interval")
            if interval is not None:
                if provider != "local":
                    raise ConfigError(
                        f"[[tts]] '{name}' has streaming_interval but "
                        f"provider = {provider!r}. Only mlx-audio (local) "
                        "honors this parameter."
                    )
                try:
                    config.streaming_interval = float(interval)
                except (TypeError, ValueError) as e:
                    raise ConfigError(
                        f"[[tts]] '{name}' streaming_interval must be a "
                        f"number, got {interval!r}"
                    ) from e
                if config.streaming_interval <= 0:
                    raise ConfigError(
                        f"[[tts]] '{name}' streaming_interval must be > 0, "
                        f"got {config.streaming_interval}"
                    )

            instruct = entry.get("instruct")
            if instruct is not None:
                if not isinstance(instruct, str):
                    raise ConfigError(
                        f"[[tts]] '{name}' instruct must be a string, got {instruct!r}"
                    )
                if provider == "cloud" and config.vendor == "gemini":
                    raise ConfigError(
                        f"[[tts]] '{name}' has instruct but vendor = 'gemini'. "
                        "Gemini TTS steers via inline [audio tags] in the "
                        "spoken text, not a separate instruction field."
                    )
                config.instruct = instruct

            temperature = entry.get("temperature")
            if temperature is not None:
                if provider != "local":
                    raise ConfigError(
                        f"[[tts]] '{name}' has temperature but provider = "
                        f"{provider!r}. Only mlx-audio (local) honors this "
                        "parameter."
                    )
                try:
                    config.temperature = float(temperature)
                except (TypeError, ValueError) as e:
                    raise ConfigError(
                        f"[[tts]] '{name}' temperature must be a number, "
                        f"got {temperature!r}"
                    ) from e
                if config.temperature < 0:
                    raise ConfigError(
                        f"[[tts]] '{name}' temperature must be >= 0, got "
                        f"{config.temperature}"
                    )

            split = entry.get("split")
            if split is not None:
                if split not in {"sentence", "paragraph", "full"}:
                    raise ConfigError(
                        f"[[tts]] '{name}' split must be one of "
                        f"'sentence' / 'paragraph' / 'full', got {split!r}"
                    )
                config.split = split

        if role == "llm":
            effort = entry.get("reasoning_effort")
            if effort is not None:
                if not isinstance(effort, str) or effort not in {
                    "none",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                }:
                    raise ConfigError(
                        f"[[llm]] '{name}' has invalid reasoning_effort "
                        f"{effort!r}. Allowed: none, minimal, low, medium, "
                        "high, xhigh."
                    )
                config.reasoning_effort = effort

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

        if provider == "local":
            if "server" in entry and "runtime" not in entry:
                raise ConfigError(
                    f"[[{role}]] '{name}' uses the old 'server' field. Rename "
                    "it to 'runtime' (e.g. runtime = \"llamacpp\") — the field "
                    "now applies to STT and TTS entries too."
                )
            runtime = entry.get("runtime")
            allowed = runtimes_for_role(role)
            if not runtime or runtime not in allowed:
                raise ConfigError(
                    f"[[{role}]] '{name}' must set runtime to one of "
                    f"{allowed} (got {runtime!r})."
                )
            config.runtime = runtime

            if role == "llm":
                config.audio_input = bool(entry.get("audio_input", False))
                kv_bits = entry.get("kv_bits")
                config.kv_bits = str(kv_bits) if kv_bits is not None else None
                kv_scheme = entry.get("kv_quant_scheme")
                config.kv_quant_scheme = (
                    str(kv_scheme) if kv_scheme is not None else None
                )
                preset = entry.get("preset")
                if runtime == "llamacpp":
                    if not preset or not isinstance(preset, str):
                        raise ConfigError(
                            f"[[llm]] '{name}' (runtime=llamacpp) needs a "
                            f"'preset' path, e.g. preset = 'llamacpp-models.ini'"
                        )
                    preset_path = _PROJECT_ROOT / preset
                    if not preset_path.exists():
                        raise ConfigError(
                            f"[[llm]] '{name}' preset file not found: "
                            f"{preset_path}. Copy llamacpp-models.ini.example "
                            "to llamacpp-models.ini and customize it."
                        )
                    config.preset = preset

        models.append(config)
    return models


def _filter_catalog_by_os(
    role: Role,
    catalog: list[ModelConfig],
    os_tag: str,
) -> tuple[list[ModelConfig], list[ModelConfig]]:
    """Split a catalog into (compatible, dropped) for `os_tag`.

    Cloud entries are always compatible. Local entries survive only when
    their `runtime` runs on the current OS (see `runtimes.RUNTIMES`).
    """
    compatible: list[ModelConfig] = []
    dropped: list[ModelConfig] = []
    for m in catalog:
        if m.provider == "cloud":
            compatible.append(m)
            continue
        if m.runtime and is_runtime_supported(m.runtime, os_tag):
            compatible.append(m)
        else:
            dropped.append(m)
    if not compatible:
        # Every entry for this role was filtered out — preserve the startup
        # error but make the cause clear.
        names = [f"{m.name} ({m.runtime})" for m in dropped]
        raise ConfigError(
            f"No compatible [[{role}]] entries on {os_tag}: all catalog "
            f"entries use runtimes that don't support this OS ({names}). "
            "Add a cloud entry (or a runtime that supports this OS) to "
            "models.toml."
        )
    return compatible, dropped


def _resolve_active(
    role: Role,
    pref: str | None,
    catalog: list[ModelConfig],
    dropped: list[ModelConfig],
    notes: list[str],
) -> str:
    """Pick the active model name, recording a note if a user preference got
    filtered out because it's not supported on the current OS."""
    if pref:
        for m in catalog:
            if m.name == pref:
                return pref
        dropped_match = next((m for m in dropped if m.name == pref), None)
        fallback = catalog[0].name
        if dropped_match is not None:
            notes.append(
                f"{role.upper()}: switched from '{pref}' to '{fallback}' — "
                f"the '{dropped_match.runtime}' runtime is not supported on "
                "this operating system."
            )
        else:
            print(
                f"Warning: preferences.toml active {role} '{pref}' not found "
                f"in catalog; falling back to '{fallback}'.",
                file=sys.stderr,
            )
        return fallback
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

    models_toml = _load_models_toml()
    os_tag = current_os()
    stt_models_all = _parse_catalog("stt", list(models_toml.get("stt", [])))
    llm_models_all = _parse_catalog("llm", list(models_toml.get("llm", [])))
    tts_models_all = _parse_catalog("tts", list(models_toml.get("tts", [])))
    stt_models, stt_dropped = _filter_catalog_by_os("stt", stt_models_all, os_tag)
    llm_models, llm_dropped = _filter_catalog_by_os("llm", llm_models_all, os_tag)
    tts_models, tts_dropped = _filter_catalog_by_os("tts", tts_models_all, os_tag)

    prefs = load_preferences()
    fallback_notes: list[str] = []
    active_stt = _resolve_active(
        "stt", prefs.get("stt"), stt_models, stt_dropped, fallback_notes
    )
    active_llm = _resolve_active(
        "llm", prefs.get("llm"), llm_models, llm_dropped, fallback_notes
    )
    active_tts = _resolve_active(
        "tts", prefs.get("tts"), tts_models, tts_dropped, fallback_notes
    )

    settings = Settings(
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
        fallback_notes=fallback_notes,
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

    # Validate snippet regexes up-front so a bad regex surfaces at startup
    # rather than on the first runtime model switch that happens to match it.
    # The actual matching + concatenation happens in compose_agent_instructions,
    # called from providers.create_agent every time a pipeline is built.
    for pattern in settings.model_instruction_snippets:
        if pattern.startswith("re:"):
            try:
                re.compile(pattern[3:], re.IGNORECASE)
            except re.error as e:
                raise ConfigError(
                    f"[agent.model-instructions] invalid regex {pattern!r}: {e}"
                ) from e

    return settings


def compose_agent_instructions(settings: Settings) -> str:
    """Base agent instructions + any model-specific snippets whose keys match
    the **currently active** STT / LLM / TTS model IDs. Recomputed on every
    pipeline rebuild so a runtime Settings-modal swap picks up the right
    snippets for the newly-active models.
    """
    text = settings.agent_instructions
    active_model_names = [
        settings.stt_model.lower(),
        settings.tts_model.lower(),
        settings.llm_model.lower(),
    ]
    for pattern, snippet in settings.model_instruction_snippets.items():
        if pattern.startswith("re:"):
            rx = re.compile(pattern[3:], re.IGNORECASE)
            matched = any(rx.search(m) for m in active_model_names)
        else:
            matched = any(pattern.lower() in m for m in active_model_names)
        if matched:
            text += snippet
    return text


def _validate_active_requirements(settings: Settings) -> None:
    """Check the URL/API-key requirements implied by the current active models.
    A model with its own `api_key` counts as self-sufficient — it doesn't
    need the vendor-wide env-var fallback."""
    active = [settings.stt, settings.llm, settings.tts]
    needs_openai = any(
        m.provider == "cloud"
        and (m.vendor is None or m.vendor == "openai")
        and not m.api_key
        for m in active
    )
    needs_gemini = any(
        m.provider == "cloud" and m.vendor == "gemini" and not m.api_key for m in active
    )
    needs_local_stt = settings.stt.provider == "local"
    needs_local_llm = settings.llm.provider == "local"
    needs_local_tts = settings.tts.provider == "local"

    if needs_openai and not settings.openai_api_key:
        raise ConfigError(
            "Active selection includes an OpenAI cloud model without a "
            "per-model api_key, and OPENAI_API_KEY is not set (check .env)."
        )
    if needs_gemini and not settings.gemini_api_key:
        raise ConfigError(
            "Active selection includes a Gemini cloud model without a "
            "per-model api_key, and GEMINI_API_KEY is not set (check .env)."
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
