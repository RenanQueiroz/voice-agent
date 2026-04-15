from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import openai
from openai import APIConnectionError
from agents import set_tracing_disabled
from agents.voice import AudioInput, VoicePipeline

from .audio import VADRecorder, play_response, read_key, record_push_to_talk
from .config import Settings, load_settings
from .providers import create_pipeline

if TYPE_CHECKING:
    from .servers import ServerManager

logging.getLogger("openai.agents").setLevel(logging.CRITICAL)
set_tracing_disabled(True)


def _print_connection_error(settings: Settings) -> None:
    print("\n-- Connection error: could not reach the API server.")
    if settings.voice_mode == "local":
        print(f"   Make sure the local servers are running:")
        print(f"     STT/TTS: {settings.mlx_audio_url}  (./scripts/start_mlx_audio.sh)")
        print(f"     LLM:     {settings.mlx_vlm_url}  (./scripts/start_mlx_vlm.sh)")
    else:
        print("   Check your internet connection and OPENAI_API_KEY in .env")


async def _run_push_to_talk(settings: Settings, server_manager: ServerManager | None = None) -> None:
    workflow, pipeline = create_pipeline(settings)

    print("Voice Agent Ready (STT -> LLM -> TTS pipeline)")
    print("Press K to start recording, K again to stop, Q to quit\n")

    while True:
        print("-- Ready. Press K to speak.")
        while True:
            key = read_key()
            if key == "k":
                break
            if key == "q":
                return
            await asyncio.sleep(0)

        print("-- Recording... (press K to stop)")
        buffer = await record_push_to_talk(settings)

        if len(buffer) < settings.sample_rate * 0.5:
            print("-- Too short, skipping.")
            continue

        print(f"-- Processing {len(buffer) / settings.sample_rate:.1f}s of audio...")
        audio_input = AudioInput(buffer=buffer)
        try:
            result = await pipeline.run(audio_input)
            await play_response(result, settings)
        except APIConnectionError:
            _print_connection_error(settings)
            return
        except openai.AuthenticationError as e:
            print(f"\n-- Auth error: {e.message}")
            print("   Check your OPENAI_API_KEY in .env")
            return
        except openai.RateLimitError as e:
            print(f"\n-- Rate limit / quota error: {e.message}")
            print(
                "   Check your plan and billing at"
                " https://platform.openai.com/settings/organization/billing"
            )
        except openai.APIError as e:
            print(f"\n-- API error: {e.message}")
            if server_manager:
                server_manager.print_server_logs()


async def _run_vad(settings: Settings, server_manager: ServerManager | None = None) -> None:
    workflow, pipeline = create_pipeline(settings)
    recorder = VADRecorder(settings)

    print("Voice Agent Ready (STT -> LLM -> TTS pipeline)")
    print(f"Mode: {settings.voice_mode} | Listening with VAD (Q to quit)\n")
    print("-- Listening... speak naturally.")

    quit_event = asyncio.Event()

    async def check_quit() -> None:
        loop = asyncio.get_event_loop()
        while not quit_event.is_set():
            key = await loop.run_in_executor(None, read_key)
            if key == "q":
                quit_event.set()
                return

    quit_task = asyncio.create_task(check_quit())

    try:
        while not quit_event.is_set():
            segment = await recorder.record_segment(quit_event=quit_event)

            if quit_event.is_set():
                break

            if len(segment) < settings.sample_rate * 0.3:
                continue

            print(f"-- Processing {len(segment) / settings.sample_rate:.1f}s of audio...")
            audio_input = AudioInput(buffer=segment)
            try:
                result = await pipeline.run(audio_input)
                recorder.pause()
                await play_response(result, settings)
                recorder.resume()
                print("-- Listening...")
            except APIConnectionError:
                _print_connection_error(settings)
                return
            except openai.AuthenticationError as e:
                print(f"\n-- Auth error: {e.message}")
                print("   Check your OPENAI_API_KEY in .env")
                return
            except openai.RateLimitError as e:
                print(f"\n-- Rate limit / quota error: {e.message}")
            except openai.APIError as e:
                print(f"\n-- API error: {e.message}")
                if server_manager:
                    server_manager.print_server_logs()
    finally:
        quit_task.cancel()


async def run(settings: Settings | None = None) -> None:
    if settings is None:
        settings = load_settings()

    server_manager = None
    if settings.voice_mode == "local":
        from .servers import ServerManager

        server_manager = ServerManager(settings)
        if not await server_manager.start():
            print("Failed to start local servers. Exiting.")
            return

    try:
        if settings.input_mode == "vad":
            await _run_vad(settings, server_manager)
        else:
            await _run_push_to_talk(settings, server_manager)
    finally:
        if server_manager:
            server_manager.stop()
