from __future__ import annotations

import asyncio
import select
import sys
import termios
import threading
import tty
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from .config import Settings

if TYPE_CHECKING:
    from .display import Display

CHANNELS = 1

# Shared terminal state so we only set cbreak once
_term_lock = threading.Lock()
_term_old: list | None = None


def _enter_cbreak() -> None:
    global _term_old
    with _term_lock:
        if _term_old is None:
            fd = sys.stdin.fileno()
            _term_old = termios.tcgetattr(fd)
            tty.setcbreak(fd)


def _restore_terminal() -> None:
    global _term_old
    with _term_lock:
        if _term_old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _term_old)
            _term_old = None


def read_key(timeout: float = 0.2) -> str | None:
    """Read a single keypress with a timeout. Returns None if no key pressed."""
    _enter_cbreak()
    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if ready:
        return sys.stdin.read(1).lower()
    return None


async def record_push_to_talk(settings: Settings) -> np.ndarray:
    """Record audio from the microphone until K is pressed again.
    Raises KeyboardInterrupt if Q is pressed."""
    chunks: list[np.ndarray] = []

    stream = sd.InputStream(
        samplerate=settings.sample_rate, channels=CHANNELS, dtype="int16"
    )
    stream.start()

    pressed = ""
    try:
        while True:
            if stream.read_available > 0:
                data, _ = stream.read(stream.read_available)
                chunks.append(data)
            key = read_key(timeout=0.02)
            if key in ("k", "q"):
                pressed = key
                break
            await asyncio.sleep(0)
    finally:
        stream.stop()
        stream.close()

    if pressed == "q":
        raise KeyboardInterrupt

    recording = np.concatenate(chunks, axis=0).flatten()
    return np.asarray(recording, dtype=np.int16)


def _downsample_24k_to_16k(audio: np.ndarray) -> np.ndarray:
    """Downsample 24kHz int16 audio to 16kHz for webrtcvad.
    Uses simple linear interpolation: take 2 out of every 3 samples."""
    n = len(audio) - (len(audio) % 3)
    trimmed = audio[:n].reshape(-1, 3)
    s0 = trimmed[:, 0]
    s1 = (
        (trimmed[:, 1].astype(np.int32) + trimmed[:, 2].astype(np.int32)) // 2
    ).astype(np.int16)
    return np.column_stack([s0, s1]).flatten()


class VADRecorder:
    """Continuously listens to the microphone and yields complete speech segments
    using webrtcvad for voice activity detection."""

    def __init__(self, settings: Settings, display: Display):
        import webrtcvad

        self.sample_rate = settings.sample_rate
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness)
        self.silence_threshold = settings.vad_silence_ms // 20
        self._muted = False
        self._energy_threshold = settings.vad_energy_threshold
        self._display = display
        self._stream: sd.InputStream | None = None

    def _open_stream(self) -> sd.InputStream:
        if self._stream is None or self._stream.closed:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, channels=CHANNELS, dtype="int16"
            )
            self._stream.start()
        return self._stream

    def _close_stream(self) -> None:
        if self._stream is not None and not self._stream.closed:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def mute(self) -> None:
        self._muted = True
        self._close_stream()

    def unmute(self) -> None:
        self._muted = False

    def pause(self) -> None:
        """Pause without releasing the mic (used during TTS playback)."""
        self._muted = True

    def resume(self) -> None:
        """Resume after pause (reopens mic)."""
        self._muted = False

    async def record_segment(
        self, quit_event: asyncio.Event | None = None
    ) -> np.ndarray:
        """Block until a complete speech segment is detected. Returns 24kHz int16 buffer.
        Returns an empty array if quit_event is set."""
        frame_duration_ms = 20
        frame_samples = int(self.sample_rate * frame_duration_ms / 1000)
        frame_16k_samples = int(16000 * frame_duration_ms / 1000)

        speech_buffer: list[np.ndarray] = []
        silence_count = 0
        is_speaking = False

        try:
            while True:
                if quit_event and quit_event.is_set():
                    return np.array([], dtype=np.int16)

                if self._muted:
                    self._close_stream()
                    await asyncio.sleep(0.05)
                    continue

                stream = self._open_stream()

                if stream.read_available < frame_samples:
                    await asyncio.sleep(0.005)
                    continue

                frame_24k, _ = stream.read(frame_samples)
                frame_24k = frame_24k.flatten().astype(np.int16)

                frame_16k = _downsample_24k_to_16k(frame_24k)[:frame_16k_samples]

                rms = int(np.sqrt(np.mean(frame_24k.astype(np.int32) ** 2)))
                vad_says_speech = self.vad.is_speech(frame_16k.tobytes(), 16000)
                is_speech = vad_says_speech and rms > self._energy_threshold

                # Live status indicator
                if is_speaking:
                    remaining_ms = (
                        self.silence_threshold - silence_count
                    ) * frame_duration_ms
                    if is_speech:
                        self._display.vad_speaking(rms)
                    else:
                        self._display.vad_silence(remaining_ms)
                elif is_speech:
                    self._display.vad_speaking(rms)

                if is_speech:
                    speech_buffer.append(frame_24k)
                    silence_count = 0
                    if not is_speaking:
                        is_speaking = True
                elif is_speaking:
                    speech_buffer.append(frame_24k)
                    silence_count += 1
                    if silence_count >= self.silence_threshold:
                        self._display.vad_clear()
                        break
        finally:
            self._close_stream()

        return np.concatenate(speech_buffer)


async def play_response(result, display: Display) -> tuple[float, float]:
    """Stream TTS audio response to speakers.
    Returns (tts_total_seconds, tts_first_byte_seconds)."""
    import time

    player = sd.OutputStream(samplerate=24000, channels=CHANNELS, dtype=np.int16)
    player.start()
    tts_start = time.monotonic()
    first_byte_time = 0.0
    try:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                if first_byte_time == 0.0:
                    first_byte_time = time.monotonic() - tts_start
                player.write(event.data)
            elif event.type == "voice_stream_event_lifecycle":
                if event.event == "turn_started":
                    display.turn_started()
                elif event.event == "turn_ended":
                    display.turn_ended()
                elif event.event == "session_ended":
                    display.session_ended()
            elif event.type == "voice_stream_event_error":
                display.api_error(str(event.error))
    finally:
        player.stop()
        player.close()
    return time.monotonic() - tts_start, first_byte_time
