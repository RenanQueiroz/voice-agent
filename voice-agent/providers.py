from __future__ import annotations

import asyncio
import base64
import io
import os
import platform
import re
import time
import wave
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx
import numpy as np
from openai import AsyncOpenAI, Omit

from agents import (
    Agent,
    CodeInterpreterTool,
    FileSearchTool,
    ModelSettings,
    Runner,
    Tool,
    WebSearchTool,
)
from openai.types.shared import Reasoning
from agents.mcp import MCPServer
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline
from agents.voice.input import AudioInput, StreamedAudioInput
from agents.voice.model import (
    STTModel,
    STTModelSettings,
    StreamedTranscriptionSession,
    TTSModelSettings,
)
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider
from agents.voice.models.openai_stt import OpenAISTTModel
from agents.voice.models.openai_tts import OpenAITTSModel
from agents.voice.pipeline_config import VoicePipelineConfig

from .config import ModelConfig, Settings, compose_agent_instructions
from .display import TurnMetrics

# Minimum character count before text is sent to TTS (matches SDK's _add_text threshold)
_MIN_TTS_CHARS = 20

if TYPE_CHECKING:
    from .display import Display


class StreamingTTSModel(OpenAITTSModel):
    """TTS model that requests server-side streaming from mlx-audio.

    Optionally carries `ref_audio` / `ref_text` for models that support voice
    cloning (e.g. CSM). mlx-audio expects both as strings in the JSON body —
    `ref_audio` is a filesystem path the server can read, not an upload.

    `streaming_interval` overrides mlx-audio's buffering window. We default
    to the server's own 2.0s — shorter values seem plausible on paper but in
    practice caused stuttering/underruns and glitchy audio, so opting into a
    lower value is per-TTS-entry.
    """

    _DEFAULT_STREAMING_INTERVAL = 2.0

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        streaming_interval: float | None = None,
        instruct: str | None = None,
        temperature: float | None = None,
    ):
        super().__init__(model=model, openai_client=openai_client)
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._instruct = instruct
        self._temperature = temperature
        self._streaming_interval = (
            streaming_interval
            if streaming_interval is not None
            else self._DEFAULT_STREAMING_INTERVAL
        )

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        # Some TTS tokenizers (notably CSM) choke on the Unicode "right single
        # quotation mark" (U+2019) that LLMs commonly emit for apostrophes —
        # the audio skips or mispronounces the word. Normalize to ASCII; it's
        # a lossless swap phonetically.
        text = text.replace("\u2019", "'")
        extra_body: dict[str, object] = {
            "stream": True,
            "streaming_interval": self._streaming_interval,
        }
        if self._ref_audio and self._ref_text:
            extra_body["ref_audio"] = self._ref_audio
            extra_body["ref_text"] = self._ref_text
        if self._instruct:
            extra_body["instruct"] = self._instruct
        if self._temperature is not None:
            extra_body["temperature"] = self._temperature
        response = self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=settings.voice or "default",
            input=text,
            response_format="pcm",
            extra_body=extra_body,
        )
        async with response as stream:
            async for chunk in stream.iter_bytes(chunk_size=1024):
                yield chunk


class QwenStreamingTTSModel(StreamingTTSModel):
    """Streaming adapter for Qwen3-TTS-Openai-Fastapi (optimized backend).

    Qwen3 emits int16 PCM @ 24 kHz on `response_format="pcm"` — the exact
    format `AudioPlayer` already expects, so the bytes pass through
    untouched. This subclass exists to (a) adjust the request body (the
    optimized backend rejects the mlx-audio-specific extras that the
    base `StreamingTTSModel` always sends — `streaming_interval`,
    `ref_audio`, `ref_text`), and (b) trim the first ~100 ms of every
    stream, which contains model-warmup noise regardless of
    `non_streaming_mode`. `instruct` (style control on CustomVoice
    models) and `temperature` map through as-is.

    Audio-quality note: `temperature` (set on the catalog entry) is
    the main knob for taming speaker-embedding L1-phonetic bleed-
    through on non-native-language output (e.g. Sohee + English comes
    out heavily Korean-accented at the model default of 1.0; 0.6-0.7
    sounds close to native English). Upstream's `/v1/audio/speech`
    schema doesn't expose `temperature` natively — `setup-qwen3-tts.sh`
    patches the schema + router + backend to forward it end-to-end.
    If that patch regresses, temperature from `extra_body` will be
    silently dropped by pydantic and you'll be back to the default.

    Earlier notes claimed pcm was float32 little-endian. That was wrong
    / out of date — a raw probe of current upstream builds confirms
    int16 LE mono at 24 kHz. If a future upstream change breaks that,
    the symptom is fast, high-pitched, garbled audio — re-probe with
    a curl to `/v1/audio/speech` and compare byte counts to the
    duration * sample rate * bytes-per-sample you expect.
    """

    # Qwen's streaming decode emits a short burst of moderate-amplitude
    # noise at the start of every response — a model-level warmup
    # artifact, not a transport glitch. Empirical probe (three runs at
    # temperature 0.5, 0.7, and default 1.0 — shape is temperature-
    # independent): rms ≈ 1200 at t=0, tapering through t=80ms (rms
    # ~60-260, still faintly audible on good headphones), and truly
    # silent from t=90ms onward. Since the voice agent splits on
    # sentences, each sentence hits this, producing the "garbled start,
    # fixes itself" symptom. 150 ms of trim (3600 samples × 2 bytes =
    # 7200 bytes at 24 kHz int16) cleanly clears the taper window with
    # margin, landing well inside the silence zone so no click at the
    # cut. The added ~50 ms of latency is imperceptible.
    _WARMUP_TRIM_BYTES = 7200

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        text = text.replace("\u2019", "'")
        extra_body: dict[str, object] = {"stream": True}
        if self._instruct:
            extra_body["instruct"] = self._instruct
        if self._temperature is not None:
            extra_body["temperature"] = self._temperature
        response = self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=settings.voice or "Vivian",
            input=text,
            response_format="pcm",
            extra_body=extra_body,
        )
        bytes_trimmed = 0
        async with response as stream:
            async for chunk in stream.iter_bytes(chunk_size=1024):
                if not chunk:
                    continue
                if bytes_trimmed < self._WARMUP_TRIM_BYTES:
                    need = self._WARMUP_TRIM_BYTES - bytes_trimmed
                    if len(chunk) <= need:
                        bytes_trimmed += len(chunk)
                        continue
                    chunk = chunk[need:]
                    bytes_trimmed = self._WARMUP_TRIM_BYTES
                yield chunk


def _resample_wav_16k(audio_input: AudioInput) -> bytes:
    """Convert AudioInput (24kHz int16) to a 16kHz WAV byte buffer."""
    buf = audio_input.buffer
    src_rate = audio_input.frame_rate
    dst_rate = 16000
    if src_rate != dst_rate:
        # Simple linear interpolation resampling
        duration = len(buf) / src_rate
        n_out = int(duration * dst_rate)
        indices = np.linspace(0, len(buf) - 1, n_out)
        buf = np.interp(indices, np.arange(len(buf)), buf.astype(np.float64))
        buf = buf.astype(np.int16)
    # Encode as WAV
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(dst_rate)
        wf.writeframes(buf.tobytes())
    return wav_buf.getvalue()


class WhisperCppSTTModel(STTModel):
    """STT model that calls whisper.cpp server's /inference endpoint with VAD."""

    def __init__(self, model_name: str, server_url: str):
        self._model_name = model_name
        self._server_url = server_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def model_name(self) -> str:
        return self._model_name

    async def transcribe(
        self,
        input: AudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> str:
        wav_bytes = _resample_wav_16k(input)
        fields: dict[str, str] = {
            "temperature": "0.0",
            "temperature_inc": "0.2",
            "response_format": "json",
        }
        if settings.language:
            fields["language"] = settings.language
        resp = await self._client.post(
            f"{self._server_url}/inference",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data=fields,
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()

    async def create_session(
        self,
        input: StreamedAudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> StreamedTranscriptionSession:
        raise NotImplementedError(
            "Streamed sessions not supported by WhisperCppSTTModel"
        )


class AudioPassthroughSTTModel(STTModel):
    """Pass-through STT that stores audio on the workflow for direct LLM input.

    When the LLM supports audio input, this skips the STT latency from the
    critical path. The real STT runs in the background for transcript display.
    """

    def __init__(
        self,
        workflow: TranscriptVoiceWorkflow,
        real_stt: WhisperCppSTTModel,
    ):
        self._workflow = workflow
        self._real_stt = real_stt

    @property
    def model_name(self) -> str:
        return self._real_stt.model_name

    async def transcribe(
        self,
        input: AudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> str:
        # Store audio for the workflow to send directly to LLM
        self._workflow.pending_audio_input = input
        # Fire real STT in background for transcript display
        asyncio.create_task(self._background_stt(input, settings, time.monotonic()))
        return ""

    async def _background_stt(
        self, input: AudioInput, settings: STTModelSettings, start_time: float
    ) -> None:
        try:
            text = await self._real_stt.transcribe(input, settings, False, False)
            elapsed = time.monotonic() - start_time
            self._workflow.display_background_transcription(text, elapsed)
        except Exception:
            pass

    async def create_session(
        self,
        input: StreamedAudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> StreamedTranscriptionSession:
        raise NotImplementedError(
            "Streamed sessions not supported by AudioPassthroughSTTModel"
        )


class TranscriptVoiceWorkflow(SingleAgentVoiceWorkflow):
    """Wraps SingleAgentVoiceWorkflow to print transcriptions in real-time."""

    def __init__(
        self,
        agent: Agent,
        display: Display,
        show_transcript: bool = True,
        show_metrics: bool = True,
        tool_call_filler: str | None = None,
    ):
        super().__init__(agent)
        self.display = display
        self.show_transcript = show_transcript
        self.show_metrics = show_metrics
        self.tool_call_filler = tool_call_filler
        self.last_metrics = TurnMetrics()
        self.turn_start_time: float = 0.0
        self._partial_response = ""
        self.pending_audio_input: AudioInput | None = None
        self._background_transcription: str | None = None

    def display_background_transcription(self, text: str, stt_seconds: float) -> None:
        """Called by AudioPassthroughSTTModel when background STT completes."""
        self._background_transcription = text
        if text and self.show_transcript:
            self.display.user_said(
                text, stt_seconds=stt_seconds if self.show_metrics else 0.0
            )
        self.last_metrics.stt_seconds = stt_seconds

    async def run(self, transcription: str) -> AsyncIterator[str]:
        transcription = transcription.strip()
        self.last_metrics = TurnMetrics()
        self._partial_response = ""

        # Build user message — multimodal with audio or text-only
        if self.pending_audio_input is not None:
            # Audio-input mode: send audio directly to LLM, skip STT latency
            audio_input = self.pending_audio_input
            self.pending_audio_input = None
            # Downsample to 16kHz to reduce payload (~33% smaller)
            wav_bytes = _resample_wav_16k(audio_input)
            audio_b64 = base64.b64encode(wav_bytes).decode()
            content: list[dict] = [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            ]
            self._input_history.append({"role": "user", "content": content})  # type: ignore[arg-type]
            # STT runs in background — display handled by display_background_transcription
        else:
            # Normal text mode: STT already ran
            if self.turn_start_time > 0:
                self.last_metrics.stt_seconds = time.monotonic() - self.turn_start_time
            stt_display = self.last_metrics.stt_seconds if self.show_metrics else 0.0
            if self.show_transcript:
                self.display.user_said(transcription, stt_seconds=stt_display)
            self._input_history.append({"role": "user", "content": transcription})

        llm_start = time.monotonic()
        first_token_time = 0.0
        token_count = 0
        agent_started = False
        filler_sent = False

        # Run the agent and intercept all events (tool calls + text)
        result = Runner.run_streamed(self._current_agent, self._input_history)
        async for event in result.stream_events():
            if event.type == "run_item_stream_event" and event.name == "tool_called":
                # If the model didn't generate any text before the tool call,
                # yield a static filler phrase so the user hears something
                if self.tool_call_filler and not filler_sent and not agent_started:
                    filler_sent = True
                    if self.show_transcript:
                        self.display.agent_start()
                        self.display.agent_chunk(self.tool_call_filler)
                        self.display.agent_end()
                    yield self.tool_call_filler + " "
                elif agent_started and self.show_transcript:
                    # End current agent block before showing tool info
                    self.display.agent_end()
                agent_started = False
                if self.show_transcript:
                    tool_name = getattr(event.item.raw_item, "name", "tool")
                    tool_args = getattr(event.item.raw_item, "arguments", "")
                    self.display.tool_call(tool_name, tool_args)
            elif event.type == "run_item_stream_event" and event.name == "tool_output":
                if self.show_transcript:
                    output = (
                        str(event.item.output) if hasattr(event.item, "output") else ""
                    )
                    # Truncate long outputs
                    if len(output) > 200:
                        output = output[:200] + "..."
                    self.display.tool_result(output)
            elif (
                event.type == "raw_response_event"
                and event.data.type == "response.output_text.delta"
            ):
                if first_token_time == 0.0:
                    first_token_time = time.monotonic() - llm_start
                if not agent_started and self.show_transcript:
                    self.display.agent_start()
                    agent_started = True
                chunk = event.data.delta
                self._partial_response += chunk
                token_count += 1
                if self.show_transcript:
                    self.display.agent_chunk(chunk)
                yield chunk

        # Update history and agent (replicate parent logic)
        self._input_history = result.to_input_list()
        self._current_agent = result.last_agent

        # Replace audio content in history with text transcription to prevent
        # the base64 audio blob from bloating context on subsequent LLM calls
        if self._background_transcription:
            for item in self._input_history:
                if not isinstance(item, dict) or item.get("role") != "user":
                    continue
                content = item.get("content")  # type: ignore[union-attr]
                if isinstance(content, list) and any(
                    isinstance(p, dict) and p.get("type") == "input_audio"
                    for p in content
                ):
                    item["content"] = self._background_transcription  # type: ignore[index]
            self._background_transcription = None

        self.last_metrics.llm_seconds = time.monotonic() - llm_start
        self.last_metrics.llm_first_token_seconds = first_token_time
        self.last_metrics.llm_tokens = token_count

        if self.show_transcript:
            self.display.agent_end()

    def save_partial_history(self) -> None:
        """Save partial LLM response to history on interruption."""
        if self._partial_response:
            self._input_history.append(
                {
                    "role": "assistant",
                    "content": self._partial_response + " [interrupted]",
                }
            )
            self._partial_response = ""
            if self.show_transcript:
                self.display.agent_end()


def _hosted_tools(llm: ModelConfig) -> list[Tool]:
    """Instantiate the OpenAI-hosted tools configured on a cloud LLM."""
    tools: list[Tool] = []
    for name in llm.hosted_tools:
        if name == "web_search":
            tools.append(WebSearchTool())
        elif name == "code_interpreter":
            tools.append(
                CodeInterpreterTool(
                    tool_config={
                        "type": "code_interpreter",
                        "container": {"type": "auto"},
                    }
                )
            )
        elif name == "file_search":
            tools.append(
                FileSearchTool(
                    vector_store_ids=list(llm.file_search_vector_stores),
                    max_num_results=llm.file_search_max_results,
                )
            )
    return tools


_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OPENAI_BASE = "https://api.openai.com/v1"


def create_agent(
    settings: Settings,
    mcp_servers: list[MCPServer] | None = None,
    app: object | None = None,
) -> Agent:
    llm = settings.llm
    if llm.provider == "local":
        client = AsyncOpenAI(
            base_url=f"{settings.llm_url}/v1",
            api_key="not-needed",
        )
        model = OpenAIChatCompletionsModel(
            model=llm.model,
            openai_client=client,
        )
    elif llm.vendor == "gemini":
        gemini_key = llm.api_key or settings.gemini_api_key or "missing"
        # Gemini exposes an OpenAI-compatible endpoint for chat completions,
        # so we keep reusing OpenAIChatCompletionsModel — only the base_url
        # and API key change.
        #
        # Careful: the OpenAI SDK auto-reads `OPENAI_ORG_ID` /
        # `OPENAI_PROJECT_ID` from env and attaches `openai-organization` /
        # `openai-project` headers. Google's gateway treats those as extra
        # credentials alongside our `Authorization: Bearer` and returns 400
        # "Multiple authentication credentials received". Passing the
        # `organization=` / `project=` kwargs as None or "" doesn't help
        # (the SDK still falls back to the env vars); the only reliable
        # way to omit them is `default_headers` with `Omit()` sentinels.
        #
        # We also scope the httpx client with `trust_env=False` so a system
        # proxy env var can't inject a third credential either.
        gemini_http = httpx.AsyncClient(trust_env=False, timeout=60.0)
        client = AsyncOpenAI(
            base_url=_GEMINI_OPENAI_BASE,
            api_key=gemini_key,
            http_client=gemini_http,
            default_headers={
                "OpenAI-Organization": Omit(),  # type: ignore[dict-item]
                "OpenAI-Project": Omit(),  # type: ignore[dict-item]
            },
        )
        model = OpenAIChatCompletionsModel(
            model=llm.model,
            openai_client=client,
        )
    else:
        # Cloud OpenAI LLM. We build the client explicitly (instead of
        # passing just the model name string and letting the SDK use its
        # default provider) so that leftover env vars — OPENAI_BASE_URL,
        # OPENAI_ORG_ID, OPENAI_PROJECT_ID — can't redirect the request or
        # sneak in extra auth headers. Those env vars have actually burned
        # us: when OPENAI_BASE_URL was set from a prior Gemini experiment,
        # "gpt-*" string models were being sent to Gemini's gateway and
        # rejected with "Multiple authentication credentials received".
        #
        # We use `OpenAIResponsesModel` (the /responses endpoint) — not
        # `OpenAIChatCompletionsModel` — because hosted tools (web_search,
        # code_interpreter, file_search) only work through Responses. The
        # default string-model path in the SDK also picks Responses.
        openai_http = httpx.AsyncClient(trust_env=False, timeout=60.0)
        openai_client = AsyncOpenAI(
            base_url=_OPENAI_BASE,
            api_key=llm.api_key or settings.openai_api_key or "missing",
            http_client=openai_http,
            default_headers={
                "OpenAI-Organization": Omit(),  # type: ignore[dict-item]
                "OpenAI-Project": Omit(),  # type: ignore[dict-item]
            },
        )
        model = OpenAIResponsesModel(
            model=llm.model,
            openai_client=openai_client,
        )

    instructions = _expand_instructions(compose_agent_instructions(settings))

    tools: list[Tool] = list(_hosted_tools(llm))

    if settings.shell.enabled and app is not None:
        from .shell import create_shell_tool, system_summary

        tools.append(create_shell_tool(app, settings.shell))  # type: ignore[arg-type]
        instructions += (
            "\n\nYou have access to the `run_shell_command` tool, which runs a "
            f"shell command on the user's computer. {system_summary()} "
            "When the user asks you to do something on their machine — open an "
            "app, read or edit a file, look something up with `curl`, play a "
            "sound, check system info, run a small script, and so on — USE this "
            "tool rather than saying you can't. The user approves every command "
            "before it runs, so it's always safe to propose one. "
            "Before each tool call, say a short sentence about what you're about "
            "to do (e.g. 'I'll open Safari for you'), then call the tool. After "
            "the command returns, summarize the result in plain speech — the "
            "user will hear your reply, not the raw output. Keep commands short "
            "and scoped; if the user declines, accept it and move on."
        )

    model_settings = ModelSettings()
    if llm.reasoning_effort:
        # Drops server-side thinking before the first token on Gemini 3
        # preview + GPT-5 reasoning models, where the default budget can
        # make TTFT catastrophic for voice (e.g. 16s on gemini-3.1-flash-
        # lite-preview vs ~1s with reasoning_effort="minimal").
        model_settings = ModelSettings(reasoning=Reasoning(effort=llm.reasoning_effort))  # type: ignore[arg-type]

    return Agent(
        name="Assistant",
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        mcp_servers=mcp_servers or [],
        tools=tools,
    )


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_instructions(text: str) -> str:
    """Expand variables in agent instructions.

    Supports:
      {date}        - today's date (e.g., "April 15, 2026")
      {time}        - current time (e.g., "2:30 PM")
      {datetime}    - date and time
      {os}          - operating system (e.g., "macOS")
      ${VAR_NAME}   - environment variable
    """
    now = datetime.now(tz=timezone.utc).astimezone()

    builtins = {
        "date": now.strftime("%B %d, %Y"),
        "time": now.strftime("%-I:%M %p"),
        "datetime": now.strftime("%B %d, %Y %-I:%M %p"),
        "os": platform.system(),
    }

    # Replace {name} builtins (single braces)
    for key, value in builtins.items():
        text = text.replace(f"{{{key}}}", value)

    # Replace ${VAR_NAME} env vars
    text = _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), text)

    return text


# A confident sentence boundary: one-or-more `.!?` followed by whitespace.
# This can't match mid-decimal ("3.14" has no space after the dot between
# the digits) and correctly matches ellipses and "?!" / "!?".
_SENT_BOUNDARY_RE = re.compile(r"[.!?]+\s+")

# Markdown hyperlink: `[display text](url)`. We keep the display text and
# drop the URL — "according to [kotlinlang.org](https://…)" becomes
# "according to kotlinlang.org" before hitting TTS. The prompt already
# asks for no hyperlinks, but models cite sources anyway. Audio tags like
# `[excited]` aren't followed by `(url)`, so they're left alone.
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# Bare URLs that slipped through without markdown wrapping. TTS reading
# "h-t-t-p-s-colon-slash-slash" out loud is brutal; strip them entirely.
_BARE_URL_RE = re.compile(r"https?://\S+")


def _clean_for_tts(text: str) -> str:
    """Strip markdown hyperlinks to their visible text and drop bare URLs.

    Applied inside the sentence splitter so every TTS path (local, OpenAI,
    Gemini) benefits — without mutating the chunks shown in the UI or stored
    in the conversation history. Called on each splitter invocation; since
    the transforms are idempotent, re-cleaning already-cleaned remainder
    text is a no-op.
    """
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("", text)
    return text


# Paragraph boundary: blank line (two newlines with only whitespace between).
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n[ \t]*\n")


def _paragraph_splitter(text_buffer: str) -> tuple[str, str]:
    """Flush only on paragraph breaks (blank lines). Still cleans markdown
    links. Good middle ground when per-sentence flushing exhausts rate
    limits (Gemini TTS) but full-response buffering adds too much latency."""
    text_buffer = _clean_for_tts(text_buffer)
    last = None
    for m in _PARAGRAPH_BOUNDARY_RE.finditer(text_buffer):
        last = m
    if last is None:
        return "", text_buffer
    return text_buffer[: last.end()], text_buffer[last.end() :]


def _no_split_splitter(text_buffer: str) -> tuple[str, str]:
    """Never flush mid-stream. The SDK's turn-end path (`_turn_done`) sends
    whatever's left as a single final TTS request, so the whole LLM
    response is synthesized in one call. Use with rate-limited providers
    (e.g. Gemini TTS) where each request costs a daily quota slot."""
    return "", _clean_for_tts(text_buffer)


def _select_splitter(mode: str | None) -> Callable[[str], tuple[str, str]]:
    if mode == "paragraph":
        return _paragraph_splitter
    if mode == "full":
        return _no_split_splitter
    # Default / "sentence"
    return _eager_sentence_splitter()


def _eager_sentence_splitter(
    min_length: int = _MIN_TTS_CHARS,
) -> Callable[[str], tuple[str, str]]:
    """Sentence splitter that sends complete sentences to TTS immediately,
    without getting fooled by decimals and abbreviations mid-stream.

    The SDK's default splitter both (a) holds back the last sentence waiting
    for a second one — defeating the point of speaking while waiting for a
    tool — and (b) treats any `.` / `!` / `?` as a sentence end, which breaks
    streamed decimals: when the model has emitted "The value is 3." but not
    yet "14 apples.", the splitter flushes "The value is 3." to TTS, which
    then reads back "three" with no decimals.

    This implementation:
      1. Flushes everything up to the *last* unambiguous boundary
         (punctuation followed by whitespace) — decimals can't match since
         they have no whitespace between the digits.
      2. For the "eager" case of a buffer that ends exactly at a
         sentence-ender with no trailing whitespace yet, it ONLY flushes
         when the char before the punctuation is a non-digit. A digit-then-
         dot at the very end of the buffer is ambiguous (could be decimal
         in progress) and is held back until more text arrives or a real
         whitespace separator appears.
    """

    def splitter(text_buffer: str) -> tuple[str, str]:
        text_buffer = _clean_for_tts(text_buffer)

        # 1) Unambiguous boundary inside the buffer (punctuation + space).
        last_match = None
        for m in _SENT_BOUNDARY_RE.finditer(text_buffer):
            last_match = m
        if last_match is not None:
            end = last_match.end()
            combined = text_buffer[:end]
            remaining = text_buffer[end:]
            if len(combined.strip()) >= min_length:
                return combined, remaining
            # If we found a boundary but what's before it is too short, fall
            # through to the eager check — the full buffer may still be long
            # enough to flush as one unit.

        # 2) Eager flush: buffer ends with sentence-ending punctuation and
        # nothing after. Only trust it if the char before the punctuation
        # isn't a digit (else we might be mid-decimal, waiting on "14 …").
        stripped = text_buffer.rstrip()
        if stripped and stripped[-1] in ".!?":
            prev = stripped[-2] if len(stripped) >= 2 else ""
            if not prev.isdigit() and len(stripped) >= min_length:
                return text_buffer, ""

        return "", text_buffer

    return splitter


def create_pipeline_config(settings: Settings) -> VoicePipelineConfig:
    tts = settings.tts
    if tts.provider == "local":
        provider = OpenAIVoiceModelProvider(
            base_url=f"{settings.tts_url}/v1",
            api_key="not-needed",
        )
    else:
        provider = OpenAIVoiceModelProvider()

    # For cloud OpenAI TTS the SDK's OpenAITTSModel reads
    # TTSModelSettings.instructions and sends it as the `instructions` body
    # param (which gpt-4o-mini-tts and family honor). For local mlx-audio we
    # send the same user-facing `instruct` value directly as `instruct` in
    # StreamingTTSModel.run(). Same config field, right wire name per provider.
    tts_settings_kwargs: dict[str, object] = {
        "voice": tts.voice if tts.voice else None,
        "text_splitter": _select_splitter(tts.split),
    }
    if tts.instruct and tts.provider == "cloud":
        tts_settings_kwargs["instructions"] = tts.instruct

    return VoicePipelineConfig(
        model_provider=provider,
        tts_settings=TTSModelSettings(**tts_settings_kwargs),  # type: ignore[arg-type]
        tracing_disabled=True,
    )


def create_pipeline(
    settings: Settings,
    display: Display,
    mcp_servers: list[MCPServer] | None = None,
) -> tuple[TranscriptVoiceWorkflow, VoicePipeline]:
    agent = create_agent(settings, mcp_servers=mcp_servers, app=display)
    workflow = TranscriptVoiceWorkflow(
        agent,
        display=display,
        show_transcript=settings.show_transcript,
        show_metrics=settings.show_metrics,
        tool_call_filler=settings.tool_call_filler,
    )
    config = create_pipeline_config(settings)

    tts = settings.tts
    tts_model: object = tts.model
    if tts.provider == "local":
        tts_client = AsyncOpenAI(
            base_url=f"{settings.tts_url}/v1",
            api_key="not-needed",
        )
        # Qwen3-TTS streams float32 PCM and rejects the mlx-audio extras
        # (streaming_interval / ref_audio / ref_text). Use a dedicated
        # subclass that converts float32 → int16 and sends only the fields
        # this backend accepts.
        if tts.runtime == "qwen3-tts":
            tts_model = QwenStreamingTTSModel(
                model=tts.model,
                openai_client=tts_client,
                instruct=tts.instruct,
                temperature=tts.temperature,
            )
        else:
            tts_model = StreamingTTSModel(
                model=tts.model,
                openai_client=tts_client,
                ref_audio=tts.ref_audio,
                ref_text=tts.ref_text,
                streaming_interval=tts.streaming_interval,
                instruct=tts.instruct,
                temperature=tts.temperature,
            )
    elif tts.vendor == "gemini":
        from .gemini_tts import GeminiTTSModel

        tts_model = GeminiTTSModel(
            model=tts.model,
            api_key=tts.api_key or settings.gemini_api_key or "missing",
        )
    elif tts.api_key:
        # Cloud OpenAI TTS with a per-model key override — bypass the
        # shared pipeline-level provider (which reads OPENAI_API_KEY) and
        # build the TTS client explicitly so this key is what's used.
        tts_client = AsyncOpenAI(
            api_key=tts.api_key,
            http_client=httpx.AsyncClient(trust_env=False, timeout=60.0),
            default_headers={
                "OpenAI-Organization": Omit(),  # type: ignore[dict-item]
                "OpenAI-Project": Omit(),  # type: ignore[dict-item]
            },
        )
        tts_model = OpenAITTSModel(model=tts.model, openai_client=tts_client)

    stt = settings.stt
    llm = settings.llm
    stt_model: str | STTModel = stt.model
    if stt.provider == "local" and settings.stt_url:
        whisper_stt = WhisperCppSTTModel(stt.model, settings.stt_url)
        # Audio-passthrough only makes sense when the LLM is also local and
        # accepts audio directly — otherwise the audio blob never reaches a
        # model that can consume it.
        if llm.provider == "local" and llm.audio_input:
            stt_model = AudioPassthroughSTTModel(workflow, whisper_stt)
        else:
            stt_model = whisper_stt
    elif stt.provider == "cloud" and stt.api_key:
        # Same override rationale as the cloud TTS branch above.
        stt_client = AsyncOpenAI(
            api_key=stt.api_key,
            http_client=httpx.AsyncClient(trust_env=False, timeout=60.0),
            default_headers={
                "OpenAI-Organization": Omit(),  # type: ignore[dict-item]
                "OpenAI-Project": Omit(),  # type: ignore[dict-item]
            },
        )
        stt_model = OpenAISTTModel(model=stt.model, openai_client=stt_client)

    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=stt_model,
        tts_model=tts_model,
        config=config,
    )
    return workflow, pipeline
