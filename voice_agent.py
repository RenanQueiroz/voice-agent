"""
Speech-to-speech voice agent using OpenAI Agents SDK (3-model pipeline).

Pipeline: STT (whisper-1) -> LLM agent (gpt-4o-mini) -> TTS (tts-1)
Each model can be independently swapped for a local model later.

Requires: OPENAI_API_KEY environment variable.

Usage: uv run python voice_agent.py
Controls: K = start/stop recording, Q = quit
"""

import asyncio
import logging
import sys
import termios
import tty
from pathlib import Path

import numpy as np
import openai
import sounddevice as sd
from dotenv import load_dotenv
from agents import Agent, set_tracing_disabled
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

load_dotenv(Path(__file__).parent / ".env")

logging.getLogger("openai.agents").setLevel(logging.CRITICAL)
set_tracing_disabled(True)

SAMPLE_RATE = 24000
CHANNELS = 1

MODEL = "gpt-4o-mini"
AGENT_INSTRUCTIONS = "You are a helpful voice assistant. Be concise and conversational."

agent = Agent(name="Assistant", instructions=AGENT_INSTRUCTIONS, model=MODEL)
workflow = SingleAgentVoiceWorkflow(agent)


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def record_audio() -> np.ndarray[tuple[int], np.dtype[np.int16]]:
    """Record audio from the microphone until K is pressed again."""
    loop = asyncio.get_event_loop()
    chunks: list[np.ndarray] = []

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    stream.start()

    stop = asyncio.Event()

    async def listen_for_stop():
        while True:
            key = await loop.run_in_executor(None, read_key)
            if key in ("k", "q"):
                stop.set()
                return key

    async def capture_audio():
        while not stop.is_set():
            if stream.read_available > 0:
                data, _ = stream.read(stream.read_available)
                chunks.append(data)
            await asyncio.sleep(0.02)

    key_task = asyncio.create_task(listen_for_stop())
    capture_task = asyncio.create_task(capture_audio())

    pressed = await key_task
    await capture_task

    stream.stop()
    stream.close()

    if pressed == "q":
        raise KeyboardInterrupt

    recording = np.concatenate(chunks, axis=0).flatten()
    return np.asarray(recording, dtype=np.int16)


async def play_response(result) -> None:
    """Stream TTS audio response to speakers."""
    player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
    player.start()
    try:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.write(event.data)
            elif event.type == "voice_stream_event_lifecycle":
                print(f"\r  [{event.event}]")
            elif event.type == "voice_stream_event_error":
                print(f"\r  Error: {event.error}")
    finally:
        player.stop()
        player.close()


async def main():
    print("Voice Agent Ready (STT -> LLM -> TTS pipeline)")
    print("Press K to start recording, K again to stop, Q to quit\n")

    while True:
        # Wait for K to start recording
        loop = asyncio.get_event_loop()
        print("-- Ready. Press K to speak.")
        while True:
            key = await loop.run_in_executor(None, read_key)
            if key == "k":
                break
            if key == "q":
                print("Goodbye!")
                return

        print("-- Recording... (press K to stop)")
        try:
            buffer = await record_audio()
        except KeyboardInterrupt:
            print("Goodbye!")
            return

        if len(buffer) < SAMPLE_RATE * 0.5:
            print("-- Too short, skipping.")
            continue

        print(f"-- Processing {len(buffer) / SAMPLE_RATE:.1f}s of audio...")
        audio_input = AudioInput(buffer=buffer)
        pipeline = VoicePipeline(workflow=workflow)
        try:
            result = await pipeline.run(audio_input)
            await play_response(result)
        except openai.AuthenticationError as e:
            print(f"\n-- Auth error: {e.message}")
            print("   Check your OPENAI_API_KEY in .env")
            return
        except openai.RateLimitError as e:
            print(f"\n-- Rate limit / quota error: {e.message}")
            print("   Check your plan and billing at https://platform.openai.com/settings/organization/billing")
        except openai.APIError as e:
            print(f"\n-- API error: {e.message}")


if __name__ == "__main__":
    asyncio.run(main())
