from __future__ import annotations

from openai import AsyncOpenAI

from agents import Agent
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline
from agents.voice.model import TTSModelSettings
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider
from agents.voice.pipeline_config import VoicePipelineConfig

from .config import Settings


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
) -> tuple[SingleAgentVoiceWorkflow, VoicePipeline]:
    agent = create_agent(settings)
    workflow = SingleAgentVoiceWorkflow(agent)
    config = create_pipeline_config(settings)
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=settings.stt_model,
        tts_model=settings.tts_model,
        config=config,
    )
    return workflow, pipeline
