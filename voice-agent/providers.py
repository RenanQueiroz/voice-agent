from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from agents import Agent
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline
from agents.voice.model import TTSModelSettings
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider
from agents.voice.pipeline_config import VoicePipelineConfig

from .config import Settings


class TranscriptVoiceWorkflow(SingleAgentVoiceWorkflow):
    """Wraps SingleAgentVoiceWorkflow to print transcriptions in real-time."""

    def __init__(self, agent: Agent, show_transcript: bool = True):
        super().__init__(agent)
        self.show_transcript = show_transcript

    async def run(self, transcription: str) -> AsyncIterator[str]:
        if self.show_transcript:
            print(f"\n  You: {transcription}")
            print("  Agent: ", end="", flush=True)

        async for chunk in super().run(transcription):
            if self.show_transcript:
                print(chunk, end="", flush=True)
            yield chunk

        if self.show_transcript:
            print()  # newline after agent response


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
        tts_settings=TTSModelSettings(voice=settings.tts_voice),  # type: ignore[arg-type]
        tracing_disabled=True,
    )


def create_pipeline(
    settings: Settings,
) -> tuple[TranscriptVoiceWorkflow, VoicePipeline]:
    agent = create_agent(settings)
    workflow = TranscriptVoiceWorkflow(agent, show_transcript=settings.show_transcript)
    config = create_pipeline_config(settings)
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=settings.stt_model,
        tts_model=settings.tts_model,
        config=config,
    )
    return workflow, pipeline
