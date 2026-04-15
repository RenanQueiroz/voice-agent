# Agent Guidelines

This document provides guidance for AI agents working on this project.

## Project Overview

This is a real-time speech-to-speech voice agent using the OpenAI Agents SDK's `VoicePipeline`. It supports both cloud (OpenAI API) and local (MLX on Apple Silicon) inference with a 3-model pipeline: STT (speech-to-text), LLM (language model), TTS (text-to-speech). It also supports MCP servers for giving the agent tools.

## Package Structure

The main package is `voice-agent/` (note: hyphen, not underscore). Run with `uv run python -m voice-agent`.

```
voice-agent/
  __init__.py
  __main__.py       # Entry point, creates Display, handles KeyboardInterrupt
  config.py         # Settings dataclass, loads config.toml + .env, validates
  audio.py          # VADRecorder (continuous speech detection), AudioPlayer (interruptible),
                    #   read_key(), terminal cbreak management
  display.py        # Rich TUI: persistent footer via Live, TurnMetrics dataclass
  pipeline.py       # Main async loop: _run_vad() and _run_push_to_talk(),
                    #   _process_turn(), interruption handling, MCP lifecycle
  providers.py      # TranscriptVoiceWorkflow (intercepts STT/LLM/tool events for display),
                    #   StreamingTTSModel, create_agent(), create_pipeline()
  servers.py        # ServerManager: starts/stops mlx-audio and mlx-vlm,
                    #   health checks, dependency installation, patching
  mcp.py            # Loads MCP server configs from mcp_servers.toml,
                    #   supports ${VAR} env substitution
```

## Configuration Files

- `config.toml` -- committed, all settings. No hardcoded defaults in code.
- `.env` -- gitignored, secrets only (API keys).
- `mcp_servers.toml` -- gitignored, MCP server definitions. Copy from `mcp_servers.toml.example`.
- `model_deps.toml` -- committed, maps model name patterns to pip/brew dependencies.
- Environment variables override `.env` which override `config.toml`.
- `config.py` validates all required fields at startup and fails fast with `ConfigError`.
- Cloud-specific fields are optional when `voice_mode = "local"` and vice versa.
- `tts_voice` is optional for both modes (some TTS models don't need it).

## Key Architecture Patterns

### Async concurrency model (VAD mode)

Three concurrent tasks run in `_run_vad()`:
1. **VAD task** (`recorder.run()`) -- continuous background speech detection, pushes segments to `asyncio.Queue`
2. **Key listener** (`check_keys()`) -- reads keys via `run_in_executor` (non-blocking)
3. **Turn processing** (`_process_turn()`) -- cancellable task for STT->LLM->TTS->playback

The main loop consumes from the VAD queue. When a turn is running and the user presses Space, the current task is cancelled, partial history is saved, and the agent resumes listening.

### Terminal input

`read_key(timeout=0.2)` uses `select()` with a timeout -- never blocks the event loop. Terminal is put in cbreak mode once (thread-safe via `_term_lock`) and restored on exit via `_restore_terminal()`.

### Display (Rich Live)

The `Display` class uses `rich.live.Live` to render a persistent footer at the bottom. All conversation output uses `live.console.print()` to scroll above the footer. The footer updates on state changes via `_set_state()` -> `_update_footer()`.

Direct `sys.stdout.write()` should be avoided -- it conflicts with Rich Live. Use `self._print()` for scrolling content.

Agent response text is buffered in `_agent_buffer` and printed in one shot via `_print()` in `agent_end()`. Use `rich.markup.escape()` on agent text to prevent Rich from interpreting brackets (e.g., `[laugh]`) as markup tags.

### Model providers

Both mlx-audio and mlx-vlm expose OpenAI-compatible APIs. We reuse the SDK's existing `OpenAISTTModel`, `OpenAITTSModel`, and `OpenAIChatCompletionsModel` by pointing `AsyncOpenAI` clients at localhost. No custom model subclasses needed except `StreamingTTSModel` which adds `stream=True` to the TTS request for server-side streaming.

### TranscriptVoiceWorkflow

This is the core interception layer in `providers.py`. It overrides `SingleAgentVoiceWorkflow.run()` to:
- Replicate the parent's logic (add to history, `Runner.run_streamed()`, update history)
- Intercept `run_item_stream_event` events for `tool_called` and `tool_output` to display tool usage
- Intercept `raw_response_event` with `response.output_text.delta` for streaming text
- Measure STT, LLM, and token timing
- Track `_partial_response` for interruption history preservation

**Important**: It does NOT call `super().run()` -- it replicates the parent logic directly to intercept all event types. When modifying, ensure the history update (`_input_history = result.to_input_list()`) and agent update (`_current_agent = result.last_agent`) are preserved.

### VAD algorithm

- Audio captured at 24kHz, downsampled to 16kHz for webrtcvad
- Pre-roll ring buffer (100ms) captures audio before speech onset
- Requires 3 consecutive speech frames (60ms) to confirm speech start
- Silence threshold (configurable, default 500ms) triggers segment completion
- Energy threshold filters low-level noise even when webrtcvad says "speech"
- Minimum segment duration (0.5s) discards short noise bursts

### Echo suppression

No acoustic echo cancellation -- the mic is muted during agent response to prevent feedback from speakers. User can press Space to interrupt instead of speaking over the agent. The VAD is paused during response and resumed after.

### MCP servers

MCP servers are defined in `mcp_servers.toml` (gitignored) and loaded by `mcp.py`. The lifecycle is managed in `pipeline.run()`:
1. `load_mcp_servers()` reads the TOML and creates server instances
2. Each server is connected via `await server.connect()`
3. Tool names are collected and shown in the footer
4. The connected server instances are passed to `create_agent()` -> `Agent(mcp_servers=...)`
5. On exit, `server.cleanup()` is called for each

**Critical**: MCP servers must be loaded exactly once. The same connected instances must be passed to the Agent. Loading them twice creates disconnected duplicates that silently fail.

Environment variable substitution (`${VAR_NAME}`) is supported in all string values in `mcp_servers.toml`, resolved from `.env` and the environment.

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
./setup.sh                  # Install deps
./setup.sh --update         # Update deps
```

## Common Pitfalls

- **Package name has a hyphen**: `voice-agent/`, not `voice_agent/`. Python allows this but imports use it as-is.
- **Rich Live + stdout**: Don't use `sys.stdout.write()` for content that should scroll above the footer. Use `display._print()`. Raw stdout writes will conflict with Live's cursor positioning.
- **Rich markup in agent text**: Agent responses may contain `[brackets]` (e.g., `[laugh]`). Always use `rich.markup.escape()` before passing agent text to Rich, or it will silently strip the bracketed content.
- **Config defaults**: There are no hardcoded defaults in `config.py`. All values must come from `config.toml` or environment variables. Adding a new config field requires adding it to both `config.toml` and `load_settings()`.
- **MCP server lifecycle**: Servers are loaded once in `pipeline.run()` and the same instances are passed to the Agent. Never call `load_mcp_servers()` in multiple places -- it creates separate disconnected instances.
- **VAD runs in background**: `VADRecorder.run()` is a long-running async task. It must yield frequently (`await asyncio.sleep()`) or the event loop starves other tasks.
- **Blocking calls**: Any blocking I/O (key reads, HTTP calls) must use `run_in_executor` or async equivalents. Blocking the event loop freezes the VAD.
- **Terminal state**: Always call `_restore_terminal()` on exit. If the process crashes without restoring, the terminal will be in cbreak mode (no echo, no line buffering).
- **misaki/espeak.py patching**: The `_apply_patches()` method in `servers.py` fixes a known incompatibility between misaki and phonemizer 3.3. It replaces both the library path and data path from `espeakng_loader` (broken CI build paths) with system espeak-ng paths. The patch deletes `.pyc` cache files to ensure the fix takes effect.
- **tts_voice is optional**: Some TTS models don't require a voice parameter. The config property returns `None` if not set, and providers handle `None` gracefully.
