from __future__ import annotations

import asyncio
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
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
    """Downsample 24kHz int16 audio to 16kHz for Silero VAD."""
    n = len(audio) - (len(audio) % 3)
    trimmed = audio[:n].reshape(-1, 3)
    s0 = trimmed[:, 0]
    s1 = (
        (trimmed[:, 1].astype(np.int32) + trimmed[:, 2].astype(np.int32)) // 2
    ).astype(np.int16)
    return np.column_stack([s0, s1]).flatten()


_PROJECT_ROOT = Path(__file__).parent.parent


class VADRecorder:
    """Continuously listens to the microphone and pushes complete speech segments
    to a queue using Silero VAD (ONNX) for voice activity detection."""

    def __init__(self, settings: Settings, display: Display):
        import onnxruntime

        self.sample_rate = settings.sample_rate
        self._vad_threshold = settings.vad_threshold

        # Silero VAD requires 512 samples at 16kHz (32ms chunks)
        self._frame_duration_ms = 32
        self.silence_threshold = settings.vad_silence_ms // self._frame_duration_ms

        self._muted = False
        self._display = display
        self._stream: sd.InputStream | None = None
        self.segments: asyncio.Queue[np.ndarray] = asyncio.Queue()

        # Load Silero VAD ONNX model
        model_path = _PROJECT_ROOT / "whispercpp" / "models" / "silero_vad.onnx"
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._ort = onnxruntime.InferenceSession(str(model_path), sess_options=opts)

        # Model state: single tensor [2, 1, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)
        # Context window: 64 samples prepended to each chunk (required by model)
        self._context = np.zeros(64, dtype=np.float32)

    def _silero_predict(self, frame_16k: np.ndarray) -> float:
        """Run Silero VAD on a 512-sample 16kHz frame. Returns speech probability."""
        audio_f32 = frame_16k.astype(np.float32) / 32768.0
        # Prepend context from previous chunk (model requires this overlap)
        x = np.concatenate([self._context, audio_f32]).reshape(1, -1)
        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": self._sr,
        }
        out, state_new = self._ort.run(None, ort_inputs)
        self._state[:] = state_new
        self._context[:] = audio_f32[-64:]
        return float(out.item())

    def _reset_vad_state(self) -> None:
        """Reset model state between speech segments."""
        self._state[:] = 0
        self._context[:] = 0

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

    async def run(self, quit_event: asyncio.Event) -> None:
        """Continuous background task. Detects speech segments and pushes them to self.segments."""
        frame_duration_ms = self._frame_duration_ms
        # 32ms at 24kHz = 768 samples, downsampled to 16kHz = 512 samples (Silero requirement)
        frame_samples = int(self.sample_rate * frame_duration_ms / 1000)
        frame_16k_samples = 512

        # Require this many consecutive speech frames before we start buffering.
        # 3 frames at 32ms = 96ms -- short enough to not clip speech onset.
        speech_start_threshold = 3

        # Pre-roll: keep last N frames so we don't clip speech onset
        pre_roll_size = 3  # 3 frames = ~96ms of audio before speech was detected
        from collections import deque

        while not quit_event.is_set():
            # Record one complete speech segment
            speech_buffer: list[np.ndarray] = []
            pending_frames: list[np.ndarray] = []
            pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_size)
            speech_frame_count = 0
            silence_count = 0
            is_speaking = False
            self._reset_vad_state()

            while not quit_event.is_set():
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

                speech_prob = self._silero_predict(frame_16k)
                is_speech = speech_prob > self._vad_threshold

                if not is_speaking:
                    if is_speech:
                        pending_frames.append(frame_24k)
                        speech_frame_count += 1
                        if speech_frame_count >= speech_start_threshold:
                            # Confirmed speech -- prepend pre-roll + pending
                            is_speaking = True
                            speech_buffer.extend(pre_roll)
                            speech_buffer.extend(pending_frames)
                            pending_frames.clear()
                            self._display.vad_speaking(int(speech_prob * 100))
                    else:
                        # Reset -- noise was transient
                        speech_frame_count = 0
                        pending_frames.clear()
                        pre_roll.append(frame_24k)
                else:
                    # Already speaking
                    remaining_ms = (
                        self.silence_threshold - silence_count
                    ) * frame_duration_ms
                    if is_speech:
                        speech_buffer.append(frame_24k)
                        silence_count = 0
                        self._display.vad_speaking(int(speech_prob * 100))
                    else:
                        speech_buffer.append(frame_24k)
                        silence_count += 1
                        self._display.vad_silence(remaining_ms)
                        if silence_count >= self.silence_threshold:
                            self._display.vad_clear()
                            break

            # Push completed segment to queue
            if speech_buffer:
                segment = np.concatenate(speech_buffer)
                if len(segment) >= self.sample_rate * 0.5:
                    await self.segments.put(segment)

        self._close_stream()


class AudioPlayer:
    """Interruptible audio player for TTS output."""

    def __init__(self) -> None:
        self._player: sd.OutputStream | None = None
        self._stopped = False

    def stop(self) -> None:
        """Stop playback immediately (called from outside on interruption)."""
        self._stopped = True
        if self._player and not self._player.closed:
            self._player.stop()

    async def play(self, result, display: Display) -> tuple[float, float]:
        """Stream TTS audio to speakers.
        Returns (tts_total_seconds, tts_first_byte_seconds)."""
        self._stopped = False
        self._player = sd.OutputStream(
            samplerate=24000, channels=CHANNELS, dtype=np.int16
        )
        self._player.start()
        tts_start = time.monotonic()
        first_byte_time = 0.0
        try:
            async for event in result.stream():
                if self._stopped:
                    break
                if event.type == "voice_stream_event_audio":
                    if first_byte_time == 0.0:
                        first_byte_time = time.monotonic() - tts_start
                    if not self._stopped:
                        self._player.write(event.data)
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
            if not self._player.closed:
                self._player.stop()
                self._player.close()
            self._player = None
        return time.monotonic() - tts_start, first_byte_time
