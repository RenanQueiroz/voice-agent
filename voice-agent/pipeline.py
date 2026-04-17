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

from .audio import AudioPlayer, VADRecorder
from .config import Settings

if TYPE_CHECKING:
    from .app import VoiceAgentApp
    from .servers import ServerManager

logging.getLogger("openai.agents").setLevel(logging.CRITICAL)
set_tracing_disabled(True)


async def _process_turn(
    segment: bytes | object,
    app: VoiceAgentApp,
    player: AudioPlayer,
    settings: Settings,
    server_manager: ServerManager | None,
) -> None:
    """Process a single conversation turn (STT → LLM → TTS → playback).

    Reads `app.workflow` / `app.pipeline` fresh so a runtime model switch
    applies on the next turn without restarting the loop.
    """
    import numpy as np

    workflow = app.workflow
    pipeline = app.pipeline
    if workflow is None or pipeline is None:
        return

    audio_input = AudioInput(buffer=np.asarray(segment, dtype=np.int16))
    turn_start = time.monotonic()
    workflow.turn_start_time = turn_start

    try:
        result = await pipeline.run(audio_input)
        tts_total, tts_first = await player.play(result, app)

        if settings.show_metrics:
            m = workflow.last_metrics
            m.tts_seconds = tts_total
            m.tts_first_byte_seconds = tts_first
            m.total_seconds = time.monotonic() - turn_start
            app.metrics(m)
    except APIConnectionError:
        app.connection_error(settings)
    except httpx.RemoteProtocolError:
        app.tts_stream_error()
        if server_manager:
            app.api_error_with_logs(
                "TTS stream interrupted", server_manager.get_all_server_logs()
            )
    except openai.AuthenticationError as e:
        app.auth_error(e.message)
    except openai.RateLimitError as e:
        app.rate_limit_error(e.message)
    except openai.APIError as e:
        if server_manager:
            app.api_error_with_logs(e.message, server_manager.get_all_server_logs())
        else:
            app.api_error(e.message)


async def _run_vad(
    settings: Settings,
    app: VoiceAgentApp,
    server_manager: ServerManager | None = None,
    mcp_servers: list | None = None,
) -> None:
    recorder = VADRecorder(settings, app)
    player = AudioPlayer()
    app.ready_banner(settings)
    app.listening()

    # Watch app.is_muted and mirror it onto the recorder.
    async def mute_watcher() -> None:
        last = app.is_muted
        if last:
            recorder.mute()
        while not app.quit_event.is_set():
            await asyncio.sleep(0.05)
            if app.is_muted == last:
                continue
            last = app.is_muted
            if app.is_muted:
                recorder.mute()
            else:
                if not app.responding:
                    recorder.unmute()

    current_task: asyncio.Task[None] | None = None
    mute_task = asyncio.create_task(mute_watcher())
    vad_task = asyncio.create_task(recorder.run(app.quit_event))

    try:
        while not app.quit_event.is_set():
            try:
                segment = await asyncio.wait_for(recorder.segments.get(), timeout=0.5)
            except TimeoutError:
                continue

            if app.quit_event.is_set():
                break

            # Wait for any in-flight model switch to finish before starting a
            # turn, so we never run a turn against a half-torn-down pipeline.
            await app._switch_lock.acquire()
            try:
                app.processing(len(segment) / settings.sample_rate)

                # Mute VAD during response to prevent echo from speakers
                app.responding = True
                app.interrupt_event.clear()
                if not app.is_muted:
                    recorder.mute()

                current_task = asyncio.create_task(
                    _process_turn(segment, app, player, settings, server_manager)
                )

                interrupt_task = asyncio.create_task(app.interrupt_event.wait())
                done, _ = await asyncio.wait(
                    [current_task, interrupt_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if interrupt_task in done:
                    player.stop()
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
                    if app.workflow is not None:
                        app.workflow.save_partial_history()
                    app.interrupted()
                else:
                    interrupt_task.cancel()

                app.responding = False
                if not app.is_muted:
                    recorder.unmute()
                    app.listening()
                else:
                    app.muted()
            finally:
                app._switch_lock.release()
    finally:
        if current_task and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
        mute_task.cancel()
        vad_task.cancel()
        try:
            await mute_task
        except asyncio.CancelledError:
            pass
        try:
            await vad_task
        except asyncio.CancelledError:
            pass


async def run_pipeline_loops(
    settings: Settings,
    app: VoiceAgentApp,
    server_manager: ServerManager | None,
    mcp_servers: list,
) -> None:
    """Entry point used by VoiceAgentApp._run_pipeline.

    The heavy-lifting (server reconcile, MCP connect, initial pipeline build)
    is done by the app before we get here — we just drive the main loop and
    read `app.workflow` / `app.pipeline` per turn.
    """
    await _run_vad(settings, app, server_manager, mcp_servers)
