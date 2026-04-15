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

from .audio import AudioPlayer, VADRecorder, read_key, record_push_to_talk
from .config import Settings, load_settings
from .display import Display
from .mcp import load_mcp_servers
from .providers import TranscriptVoiceWorkflow, create_pipeline

if TYPE_CHECKING:
    from .servers import ServerManager

logging.getLogger("openai.agents").setLevel(logging.CRITICAL)
set_tracing_disabled(True)


async def _process_turn(
    segment: bytes | object,
    workflow: TranscriptVoiceWorkflow,
    pipeline: object,
    display: Display,
    player: AudioPlayer,
    settings: Settings,
    server_manager: ServerManager | None,
) -> None:
    """Process a single conversation turn (STT → LLM → TTS → playback)."""
    import numpy as np

    audio_input = AudioInput(buffer=np.asarray(segment, dtype=np.int16))
    turn_start = time.monotonic()
    workflow.turn_start_time = turn_start

    try:
        result = await pipeline.run(audio_input)  # type: ignore[union-attr]
        tts_total, tts_first = await player.play(result, display)

        if settings.show_metrics:
            m = workflow.last_metrics
            m.tts_seconds = tts_total
            m.tts_first_byte_seconds = tts_first
            m.total_seconds = time.monotonic() - turn_start
            display.metrics(m)
    except APIConnectionError:
        display.connection_error(settings)
    except httpx.RemoteProtocolError:
        display.tts_stream_error()
        if server_manager:
            display.api_error_with_logs(
                "TTS stream interrupted", server_manager.get_all_server_logs()
            )
    except openai.AuthenticationError as e:
        display.auth_error(e.message)
    except openai.RateLimitError as e:
        display.rate_limit_error(e.message)
    except openai.APIError as e:
        if server_manager:
            display.api_error_with_logs(e.message, server_manager.get_all_server_logs())
        else:
            display.api_error(e.message)


async def _run_push_to_talk(
    settings: Settings,
    display: Display,
    server_manager: ServerManager | None = None,
    mcp_servers: list | None = None,
) -> None:
    workflow, pipeline = create_pipeline(settings, display, mcp_servers=mcp_servers)
    player = AudioPlayer()
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
        await _process_turn(
            buffer, workflow, pipeline, display, player, settings, server_manager
        )


async def _run_vad(
    settings: Settings,
    display: Display,
    server_manager: ServerManager | None = None,
    mcp_servers: list | None = None,
) -> None:
    workflow, pipeline = create_pipeline(settings, display, mcp_servers=mcp_servers)
    recorder = VADRecorder(settings, display)
    player = AudioPlayer()
    display.ready_banner(settings)
    display.listening()

    quit_event = asyncio.Event()
    interrupt_event = asyncio.Event()
    muted = False
    responding = False
    current_task: asyncio.Task[None] | None = None

    async def check_keys() -> None:
        nonlocal muted, responding
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
                    if not responding:
                        recorder.unmute()
                    display.unmuted()
            if key == " " and responding:
                interrupt_event.set()

    # Start background tasks
    key_task = asyncio.create_task(check_keys())
    vad_task = asyncio.create_task(recorder.run(quit_event))

    try:
        while not quit_event.is_set():
            # Wait for next speech segment from VAD
            try:
                segment = await asyncio.wait_for(recorder.segments.get(), timeout=0.5)
            except TimeoutError:
                continue

            if quit_event.is_set():
                break

            display.processing(len(segment) / settings.sample_rate)

            # Mute VAD during response to prevent echo from speakers
            responding = True
            interrupt_event.clear()
            if not muted:
                recorder.mute()

            # Run the turn
            current_task = asyncio.create_task(
                _process_turn(
                    segment,
                    workflow,
                    pipeline,
                    display,
                    player,
                    settings,
                    server_manager,
                )
            )

            # Wait for either turn completion or interruption
            interrupt_task = asyncio.create_task(interrupt_event.wait())
            done, _ = await asyncio.wait(
                [current_task, interrupt_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if interrupt_task in done:
                # User pressed Space to interrupt
                player.stop()
                current_task.cancel()
                try:
                    await current_task
                except asyncio.CancelledError:
                    pass
                workflow.save_partial_history()
                display.interrupted()
            else:
                interrupt_task.cancel()

            # Resume VAD
            responding = False
            if not muted:
                recorder.unmute()
                display.listening()
            else:
                display.muted()
    finally:
        # Clean up all tasks
        if current_task and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
        key_task.cancel()
        vad_task.cancel()
        try:
            await key_task
        except asyncio.CancelledError:
            pass
        try:
            await vad_task
        except asyncio.CancelledError:
            pass


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

    # Connect MCP servers
    mcp_servers = load_mcp_servers() if settings.enable_mcp else []
    for server in mcp_servers:
        try:
            await server.connect()
        except Exception as e:
            display.api_error(f"Failed to connect MCP server '{server.name}': {e}")
            return

    # Collect tool names for the footer
    if mcp_servers:
        all_tool_names: list[str] = []
        for server in mcp_servers:
            tools = await server.list_tools()
            all_tool_names.extend(t.name for t in tools)
        display.set_mcp_tools(all_tool_names)

    try:
        if settings.input_mode == "vad":
            await _run_vad(settings, display, server_manager, mcp_servers)
        else:
            await _run_push_to_talk(settings, display, server_manager, mcp_servers)
    finally:
        for server in mcp_servers:
            try:
                await server.cleanup()
            except Exception:
                pass
        display.stop_footer()
        if server_manager:
            server_manager.stop()
