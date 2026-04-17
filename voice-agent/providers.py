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
from openai import AsyncOpenAI

from agents import (
    Agent,
    CodeInterpreterTool,
    FileSearchTool,
    Runner,
    Tool,
    WebSearchTool,
)
from agents.mcp import MCPServer
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline
from agents.voice.input import AudioInput, StreamedAudioInput
from agents.voice.model import (
    STTModel,
    STTModelSettings,
    StreamedTranscriptionSession,
    TTSModelSettings,
)
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider
from agents.voice.models.openai_tts import OpenAITTSModel
from agents.voice.pipeline_config import VoicePipelineConfig

from .config import ModelConfig, Settings
from .display import TurnMetrics

# Minimum character count before text is sent to TTS (matches SDK's _add_text threshold)
_MIN_TTS_CHARS = 20

if TYPE_CHECKING:
    from .display import Display


class StreamingTTSModel(OpenAITTSModel):
    """TTS model that requests server-side streaming from mlx-audio."""

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        response = self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=settings.voice or "default",
            input=text,
            response_format="pcm",
            extra_body={
                "stream": True,
            },
        )
        async with response as stream:
            async for chunk in stream.iter_bytes(chunk_size=1024):
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
        # Gemini exposes an OpenAI-compatible endpoint for chat completions,
        # so we keep reusing OpenAIChatCompletionsModel — only the base_url
        # and API key change.
        client = AsyncOpenAI(
            base_url=_GEMINI_OPENAI_BASE,
            api_key=settings.gemini_api_key or "missing",
        )
        model = OpenAIChatCompletionsModel(
            model=llm.model,
            openai_client=client,
        )
    else:
        model = llm.model  # type: ignore[assignment]

    instructions = _expand_instructions(settings.agent_instructions)

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

    return Agent(
        name="Assistant",
        instructions=instructions,
        model=model,
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

    return VoicePipelineConfig(
        model_provider=provider,
        tts_settings=TTSModelSettings(
            voice=tts.voice if tts.voice else None,  # type: ignore[arg-type]
            text_splitter=_eager_sentence_splitter(),
        ),
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
        tts_model = StreamingTTSModel(
            model=tts.model,
            openai_client=tts_client,
        )
    elif tts.vendor == "gemini":
        from .gemini_tts import GeminiTTSModel

        tts_model = GeminiTTSModel(
            model=tts.model,
            api_key=settings.gemini_api_key or "missing",
        )

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

    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=stt_model,
        tts_model=tts_model,
        config=config,
    )
    return workflow, pipeline
