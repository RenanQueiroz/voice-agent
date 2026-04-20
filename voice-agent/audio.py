from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from .config import Settings

if TYPE_CHECKING:
    from .display import Display

CHANNELS = 1


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
        ort_out = self._ort.run(None, ort_inputs)
        self._state = np.array(ort_out[1])
        self._context[:] = audio_f32[-64:]
        return float(np.array(ort_out[0]).item())

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
        # 2 frames at 32ms = 64ms -- Silero VAD is accurate enough for a lower threshold.
        speech_start_threshold = 2

        # Pre-roll: keep last N frames so we don't clip speech onset.
        # 8 frames × 32ms = 256ms — generous buffer since Silero VAD
        # may confirm speech slightly after the actual onset.
        pre_roll_size = 8
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
            # abort() — NOT stop(). Two reasons:
            #  * stop() waits for buffered audio to play out, which is
            #    the opposite of what we want on interrupt.
            #  * play() is awaiting run_in_executor(self._player.write,
            #    …) at this moment. stop() is not spec'd thread-safe
            #    against an in-progress write(); on ALSA→PulseAudio
            #    (WSLg, pipewire pulse-compat) it leaves the plugin
            #    in a partially-torn-down state, and the next
            #    OutputStream() raises "pulse_prepare: Unable to
            #    create stream: Bad state" + pthread-mutex assertion
            #    failures. abort() is the one call PortAudio spec's
            #    as thread-safe mid-write — it unblocks the executor
            #    thread so play()'s finally can close() cleanly.
            try:
                self._player.abort()
            except Exception:
                pass

    async def play(self, result, display: Display) -> tuple[float, float]:
        """Stream TTS audio to speakers.
        Returns (tts_total_seconds, tts_first_byte_seconds)."""
        self._stopped = False
        # PortAudio output buffering. Two knobs working together:
        #
        #  * `latency="high"` tells PortAudio to request the host's
        #    generous latency tier (vs. ~5-10 ms default on low).
        #  * `blocksize=4800` (200 ms at 24 kHz) sets the per-write
        #    chunk the stream expects — larger blocksize means more
        #    samples can sit in flight before the hardware drains,
        #    giving producer jitter (TTS network hiccups, GC pauses,
        #    event-loop stalls) a wider window to recover in. An extra
        #    ~200 ms of one-way latency is inaudible for voice.
        #
        # We do NOT call `start()` yet — that would kick PortAudio into
        # consuming samples from an empty buffer immediately, and
        # because TTS TTFB is 100-500 ms the output would drain to
        # silence before the first chunk lands, producing an ALSA
        # underrun every single turn. Instead we start the stream
        # *after* queueing the first chunk below, so consumption and
        # production begin in lockstep.
        self._player = sd.OutputStream(
            samplerate=24000,
            channels=CHANNELS,
            dtype=np.int16,
            latency="high",
            blocksize=4800,  # 200 ms at 24 kHz
        )
        stream_started = False
        stream_start_mono = 0.0
        total_audio_seconds = 0.0
        tts_start = time.monotonic()
        first_byte_time = 0.0
        try:
            try:
                async for event in result.stream():
                    if self._stopped:
                        break
                    if event.type == "voice_stream_event_audio":
                        if first_byte_time == 0.0:
                            first_byte_time = time.monotonic() - tts_start
                        if self._stopped:
                            continue
                        if not stream_started:
                            # Prime the queue with the first chunk, then
                            # start — this way PortAudio starts consuming
                            # the moment data's available, no head-of-
                            # stream underrun.
                            self._player.start()
                            stream_started = True
                            stream_start_mono = time.monotonic()
                        # Track how much audio we've queued so we can
                        # compute "true playback done" below without
                        # relying on PortAudio/PulseAudio drain semantics.
                        # event.data is an int16 sample array, so one
                        # element == one sample.
                        total_audio_seconds += len(event.data) / 24000
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._player.write, event.data
                        )
                    elif event.type == "voice_stream_event_lifecycle":
                        if event.event == "turn_started":
                            display.turn_started()
                        elif event.event == "turn_ended":
                            display.turn_ended()
                        elif event.event == "session_ended":
                            display.session_ended()
                    elif event.type == "voice_stream_event_error":
                        display.api_error(str(event.error))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # `result.stream()` yields events for the whole pipeline
                # (STT → LLM → TTS), so ANY upstream failure re-raises here
                # — not just TTS. Surface it as a generic API error rather
                # than mislabeling every Gemini 503 / auth failure as "TTS".
                display.api_error(str(e))
        finally:
            if stream_started and self._stopped:
                # Interrupt path: external stop() already called abort()
                # to unblock the run_in_executor write() thread. Give
                # that thread a tick to actually return from PortAudio
                # before we close — close() is not safe to call while
                # another thread is still inside write(). abort() is
                # synchronous in libportaudio (<1 ms), so 50 ms is
                # generous.
                await asyncio.sleep(0.05)
            elif stream_started:
                # Graceful end-of-turn drain. Wait for the audio we
                # queued to actually reach the speaker before we close.
                # Without this:
                #   (1) The tail of the last sentence clips — close()
                #       discards PortAudio ring + PulseAudio buffer
                #       content that hadn't played yet.
                #   (2) play() returns while audio is still playing, so
                #       pipeline.py unmutes the mic on the same tick
                #       and the TTS tail bleeds back in through the
                #       microphone as self-echo.
                #
                # Three components contribute to the drain:
                #
                #   * (total_audio_seconds - elapsed) — how much audio
                #     is queued in PortAudio's ring past what's been
                #     consumed. Naturally scales with how far ahead the
                #     TTS producer got (~buffer depth for fast-RTF
                #     backends like Qwen3, ~0 for realtime-paced ones
                #     like kokoro).
                #
                #   * `startup_delay` — sd.OutputStream.start() returns
                #     before the first callback fires; audio doesn't
                #     actually start flowing until roughly one blocksize
                #     later. `elapsed` (measured from start()) therefore
                #     overestimates playback progress by this amount, so
                #     we add it back to the drain.
                #
                #   * `pulse_cushion` — PulseAudio's own downstream
                #     buffer on ALSA→PulseAudio paths (WSLg, pipewire
                #     pulse-compat), which PortAudio doesn't see. Under
                #     system load this jitters, which is why earlier
                #     tighter values showed "sometimes too early,
                #     sometimes too late" — we err toward slightly late.
                #
                # Skipped on explicit interrupt (see above branch).
                elapsed = time.monotonic() - stream_start_mono
                startup_delay = 4800 / 24000  # blocksize / sample_rate
                pulse_cushion = 0.3
                drain_seconds = (
                    (total_audio_seconds - elapsed)
                    + startup_delay
                    + pulse_cushion
                )
                if drain_seconds > 0:
                    await asyncio.sleep(drain_seconds)

            if not self._player.closed:
                if stream_started and not self._stopped:
                    try:
                        self._player.stop()
                    except Exception:
                        pass
                try:
                    self._player.close()
                except Exception:
                    pass
            self._player = None
        return time.monotonic() - tts_start, first_byte_time
