# Agent Guidelines

This document provides guidance for AI agents working on this project.

## Project Overview

This is a real-time speech-to-speech voice agent using the OpenAI Agents SDK's `VoicePipeline`. It supports both cloud (OpenAI API) and local (MLX on Apple Silicon) inference with a 3-model pipeline: STT (speech-to-text), LLM (language model), TTS (text-to-speech).

## Package Structure

The main package is `voice-agent/` (note: hyphen, not underscore). Run with `uv run python -m voice-agent`.

```
voice-agent/
  __main__.py       # Entry point, creates Display, handles KeyboardInterrupt
  config.py         # Settings dataclass, loads config.toml + .env, validates
  audio.py          # VADRecorder (continuous speech detection), AudioPlayer (interruptible),
                    #   read_key(), terminal cbreak management
  display.py        # Rich TUI: persistent footer via Live, TurnMetrics dataclass
  pipeline.py       # Main async loop: _run_vad() and _run_push_to_talk(),
                    #   _process_turn(), interruption handling
  providers.py      # TranscriptVoiceWorkflow (intercepts STT/LLM for display/metrics),
                    #   StreamingTTSModel, create_agent(), create_pipeline()
  servers.py        # ServerManager: starts/stops mlx-audio and mlx-vlm,
                    #   health checks, dependency installation, patching
```

## Configuration

- `config.toml` -- committed, all defaults. No hardcoded defaults in code.
- `.env` -- gitignored, secrets only (OPENAI_API_KEY).
- Environment variables override `.env` which override `config.toml`.
- `config.py` validates all required fields at startup and fails fast with `ConfigError`.
- Cloud-specific fields are optional when `voice_mode = "local"` and vice versa.

## Key Architecture Patterns

### Async concurrency model (VAD mode)

Three concurrent tasks run in `_run_vad()`:
1. **VAD task** (`recorder.run()`) -- continuous background speech detection, pushes segments to `asyncio.Queue`
2. **Key listener** (`check_keys()`) -- reads keys via `run_in_executor` (non-blocking)
3. **Turn processing** (`_process_turn()`) -- cancellable task for STT->LLM->TTS->playback

The main loop consumes from the VAD queue. If a new segment arrives while a turn is running, the current turn is cancelled, partial history is saved, and the new turn starts.

### Terminal input

`read_key(timeout=0.2)` uses `select()` with a timeout -- never blocks the event loop. Terminal is put in cbreak mode once (thread-safe via `_term_lock`) and restored on exit via `_restore_terminal()`.

### Display (Rich Live)

The `Display` class uses `rich.live.Live` to render a persistent footer at the bottom. All conversation output uses `live.console.print()` to scroll above the footer. The footer updates on state changes via `_set_state()` -> `_update_footer()`.

Direct `sys.stdout.write()` should be avoided -- it conflicts with Rich Live. Use `self._print()` for scrolling content.

Agent response text is buffered in `_agent_buffer` and printed in one shot via `_print()` in `agent_end()`. Streaming partial text to the console doesn't work well with Rich Live's cursor management.

### Model providers

Both mlx-audio and mlx-vlm expose OpenAI-compatible APIs. We reuse the SDK's existing `OpenAISTTModel`, `OpenAITTSModel`, and `OpenAIChatCompletionsModel` by pointing `AsyncOpenAI` clients at localhost. No custom model subclasses needed except `StreamingTTSModel` which adds `stream=True` to the TTS request.

### VAD algorithm

- Audio captured at 24kHz, downsampled to 16kHz for webrtcvad
- Pre-roll ring buffer (100ms) captures audio before speech onset
- Requires 3 consecutive speech frames (60ms) to confirm speech start
- Silence threshold (configurable, default 500ms) triggers segment completion
- Energy threshold filters low-level noise even when webrtcvad says "speech"
- Minimum segment duration (0.5s) discards short noise bursts

### Echo suppression

No acoustic echo cancellation -- the mic is muted during agent response to prevent feedback from speakers. User can press Space to interrupt instead of speaking over the agent.

### Server management (local mode)

`ServerManager` handles the full lifecycle:
1. Check/install system deps (brew packages from `model_deps.toml`)
2. Check/install pip packages (`mlx-audio[server,tts,stt]`, `mlx-vlm`, model-specific deps)
3. Patch known compatibility issues (misaki/espeak.py for Kokoro TTS)
4. Start server subprocesses with stdout redirected to `logs/`
5. Poll `/v1/models` endpoint until healthy (10 min timeout for model downloads)
6. On exit: SIGTERM -> wait 5s -> SIGKILL

### Partial history on interruption

`TranscriptVoiceWorkflow` tracks `_partial_response` during LLM streaming. On cancellation, `save_partial_history()` appends the partial text with `[interrupted]` marker to `_input_history` so the agent has context in the next turn.

## Development Commands

```bash
uv run pyright              # Type check
uv run ruff format .        # Format
uv run ruff check --fix .   # Lint
uv run python -m voice-agent  # Run
```

## Common Pitfalls

- **Package name has a hyphen**: `voice-agent/`, not `voice_agent/`. Python allows this but imports use it as-is.
- **Rich Live + stdout**: Don't use `sys.stdout.write()` for content that should scroll above the footer. Use `display._print()`. Raw stdout writes will conflict with Live's cursor positioning.
- **Config defaults**: There are no hardcoded defaults in `config.py`. All values must come from `config.toml` or environment variables. Adding a new config field requires adding it to both `config.toml` and `load_settings()`.
- **VAD runs in background**: `VADRecorder.run()` is a long-running async task. It must yield frequently (`await asyncio.sleep()`) or the event loop starves other tasks.
- **Blocking calls**: Any blocking I/O (key reads, HTTP calls) must use `run_in_executor` or async equivalents. Blocking the event loop freezes the VAD.
- **Terminal state**: Always call `_restore_terminal()` on exit. If the process crashes without restoring, the terminal will be in cbreak mode (no echo, no line buffering).
- **misaki/espeak.py patching**: The `_apply_patches()` method in `servers.py` fixes a known incompatibility between misaki and phonemizer 3.3. It also replaces the broken `espeakng_loader` paths with system espeak-ng paths. The patch deletes `.pyc` cache files to ensure the fix takes effect.
