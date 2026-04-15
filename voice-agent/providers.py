from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from agents import Agent
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline
from agents.voice.model import TTSModelSettings
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider
from agents.voice.models.openai_tts import OpenAITTSModel
from agents.voice.pipeline_config import VoicePipelineConfig
from agents.voice.utils import get_sentence_based_splitter

from .config import Settings
from .display import TurnMetrics

if TYPE_CHECKING:
    from .display import Display


class StreamingTTSModel(OpenAITTSModel):
    """TTS model that requests server-side streaming from mlx-audio."""

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        response = self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=settings.voice or "af_heart",
            input=text,
            response_format="pcm",
            extra_body={
                "stream": True,
            },
        )
        async with response as stream:
            async for chunk in stream.iter_bytes(chunk_size=1024):
                yield chunk


class TranscriptVoiceWorkflow(SingleAgentVoiceWorkflow):
    """Wraps SingleAgentVoiceWorkflow to print transcriptions in real-time."""

    def __init__(
        self,
        agent: Agent,
        display: Display,
        show_transcript: bool = True,
        show_metrics: bool = True,
    ):
        super().__init__(agent)
        self.display = display
        self.show_transcript = show_transcript
        self.show_metrics = show_metrics
        self.last_metrics = TurnMetrics()
        self.turn_start_time: float = 0.0
        self._partial_response = ""

    async def run(self, transcription: str) -> AsyncIterator[str]:
        self.last_metrics = TurnMetrics()
        self._partial_response = ""

        # STT time = time from pipeline.run() start to now (when transcription arrives)
        if self.turn_start_time > 0:
            self.last_metrics.stt_seconds = time.monotonic() - self.turn_start_time

        stt_display = self.last_metrics.stt_seconds if self.show_metrics else 0.0
        if self.show_transcript:
            self.display.user_said(transcription, stt_seconds=stt_display)
            self.display.agent_start()

        llm_start = time.monotonic()
        token_count = 0

        async for chunk in super().run(transcription):
            self._partial_response += chunk
            token_count += 1
            if self.show_transcript:
                self.display.agent_chunk(chunk)
            yield chunk

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


def create_agent(settings: Settings) -> Agent:
    if settings.voice_mode == "local":
        client = AsyncOpenAI(
            base_url=f"{settings.mlx_vlm_url}/v1",
            api_key="not-needed",
        )
        model = OpenAIChatCompletionsModel(
            model=settings.llm_model,
            openai_client=client,
        )
    else:
        model = settings.llm_model  # type: ignore[assignment]

    return Agent(
        name="Assistant",
        instructions=settings.agent_instructions,
        model=model,
    )


def create_pipeline_config(settings: Settings) -> VoicePipelineConfig:
    if settings.voice_mode == "local":
        provider = OpenAIVoiceModelProvider(
            base_url=f"{settings.mlx_audio_url}/v1",
            api_key="not-needed",
        )
    else:
        provider = OpenAIVoiceModelProvider()

    return VoicePipelineConfig(
        model_provider=provider,
        tts_settings=TTSModelSettings(
            voice=settings.tts_voice,  # type: ignore[arg-type]
            # Lower the sentence buffer so TTS starts sooner
            text_splitter=get_sentence_based_splitter(min_sentence_length=10),
        ),
        tracing_disabled=True,
    )


def create_pipeline(
    settings: Settings,
    display: Display,
) -> tuple[TranscriptVoiceWorkflow, VoicePipeline]:
    agent = create_agent(settings)
    workflow = TranscriptVoiceWorkflow(
        agent,
        display=display,
        show_transcript=settings.show_transcript,
        show_metrics=settings.show_metrics,
    )
    config = create_pipeline_config(settings)

    # For local mode, use StreamingTTSModel for server-side streaming
    tts_model: str | StreamingTTSModel = settings.tts_model
    if settings.voice_mode == "local":
        tts_client = AsyncOpenAI(
            base_url=f"{settings.mlx_audio_url}/v1",
            api_key="not-needed",
        )
        tts_model = StreamingTTSModel(
            model=settings.tts_model,
            openai_client=tts_client,
        )

    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=settings.stt_model,
        tts_model=tts_model,
        config=config,
    )
    return workflow, pipeline
