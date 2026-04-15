from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import httpx
import openai
from openai import APIConnectionError
from agents import set_tracing_disabled
from agents.voice import AudioInput

from .audio import VADRecorder, play_response, read_key, record_push_to_talk
from .config import Settings, load_settings
from .display import Display
from .providers import create_pipeline

if TYPE_CHECKING:
    from .servers import ServerManager

logging.getLogger("openai.agents").setLevel(logging.CRITICAL)
set_tracing_disabled(True)


async def _run_push_to_talk(
    settings: Settings,
    display: Display,
    server_manager: ServerManager | None = None,
) -> None:
    workflow, pipeline = create_pipeline(settings, display)
    display.ready_banner(settings)

    while True:
        display.ready_for_key()
        while True:
            key = read_key()
            if key == "k":
                break
            if key == "q":
                return
            await asyncio.sleep(0)

        display.recording_start()
        buffer = await record_push_to_talk(settings)

        if len(buffer) < settings.sample_rate * 0.5:
            display.recording_too_short()
            continue

        display.processing(len(buffer) / settings.sample_rate)
        audio_input = AudioInput(buffer=buffer)
        try:
            turn_start = time.monotonic()
            workflow.turn_start_time = turn_start
            result = await pipeline.run(audio_input)
            tts_total, tts_first = await play_response(result, display)
            if settings.show_metrics:
                m = workflow.last_metrics
                m.tts_seconds = tts_total
                m.tts_first_byte_seconds = tts_first
                m.total_seconds = time.monotonic() - turn_start
                display.metrics(m)
        except APIConnectionError:
            display.connection_error(settings)
            return
        except httpx.RemoteProtocolError:
            display.tts_stream_error()
            if server_manager:
                display.api_error_with_logs(
                    "TTS stream interrupted", server_manager.get_all_server_logs()
                )
        except openai.AuthenticationError as e:
            display.auth_error(e.message)
            return
        except openai.RateLimitError as e:
            display.rate_limit_error(e.message)
        except openai.APIError as e:
            if server_manager:
                display.api_error_with_logs(
                    e.message, server_manager.get_all_server_logs()
                )
            else:
                display.api_error(e.message)


async def _run_vad(
    settings: Settings,
    display: Display,
    server_manager: ServerManager | None = None,
) -> None:
    workflow, pipeline = create_pipeline(settings, display)
    recorder = VADRecorder(settings, display)
    display.ready_banner(settings)
    display.listening()

    quit_event = asyncio.Event()
    muted = False

    async def check_keys() -> None:
        nonlocal muted
        loop = asyncio.get_event_loop()
        while not quit_event.is_set():
            key = await loop.run_in_executor(None, read_key)
            if key == "q":
                quit_event.set()
                return
            if key == "m":
                muted = not muted
                if muted:
                    recorder.mute()
                    display.muted()
                else:
                    recorder.unmute()
                    display.unmuted()

    quit_task = asyncio.create_task(check_keys())

    try:
        while not quit_event.is_set():
            segment = await recorder.record_segment(quit_event=quit_event)

            if quit_event.is_set():
                break

            if len(segment) < settings.sample_rate * 0.3:
                continue

            display.processing(len(segment) / settings.sample_rate)
            audio_input = AudioInput(buffer=segment)
            try:
                turn_start = time.monotonic()
                workflow.turn_start_time = turn_start
                result = await pipeline.run(audio_input)
                recorder.pause()
                tts_total, tts_first = await play_response(result, display)
                if settings.show_metrics:
                    m = workflow.last_metrics
                    m.tts_seconds = tts_total
                    m.tts_first_byte_seconds = tts_first
                    m.total_seconds = time.monotonic() - turn_start
                    display.metrics(m)
                recorder.resume()
                display.listening()
            except APIConnectionError:
                display.connection_error(settings)
                return
            except httpx.RemoteProtocolError:
                display.tts_stream_error()
                if server_manager:
                    display.api_error_with_logs(
                        "TTS stream interrupted",
                        server_manager.get_all_server_logs(),
                    )
                recorder.resume()
                display.listening()
            except openai.AuthenticationError as e:
                display.auth_error(e.message)
                return
            except openai.RateLimitError as e:
                display.rate_limit_error(e.message)
            except openai.APIError as e:
                if server_manager:
                    display.api_error_with_logs(
                        e.message, server_manager.get_all_server_logs()
                    )
                else:
                    display.api_error(e.message)
    finally:
        quit_task.cancel()


async def run(settings: Settings | None = None) -> None:
    if settings is None:
        settings = load_settings()

    display = Display()

    server_manager = None
    if settings.voice_mode == "local":
        from .servers import ServerManager

        server_manager = ServerManager(settings, display)
        if not await server_manager.start():
            display.setup_failed()
            return

    try:
        if settings.input_mode == "vad":
            await _run_vad(settings, display, server_manager)
        else:
            await _run_push_to_talk(settings, display, server_manager)
    finally:
        if server_manager:
            server_manager.stop()
