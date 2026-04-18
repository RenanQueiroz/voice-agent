# Agent Guidelines

This document is for AI agents working on this project. It describes the architecture *as it is now* — if you find a discrepancy with the code, trust the code and update this file.

## Project overview

Real-time speech-to-speech voice agent built on the OpenAI Agents SDK's `VoicePipeline`. The UI is a fullscreen Textual app. Each role (STT / LLM / TTS) is configured independently as either cloud (OpenAI or Gemini) or local (whisper.cpp / mlx-audio / mlx-vlm / mlx-lm / llama-server); the user can mix-and-match and swap at runtime.

## Package structure

The package name has a hyphen: `voice-agent/`, not `voice_agent/`. Run with `uv run python -m voice-agent`.

```
voice-agent/
  __init__.py
  __main__.py       # Entry point; just `VoiceAgentApp(load_settings()).run()`
  app.py            # Textual App: compose, bindings, pipeline worker,
                    #   action_open_settings, switch_models,
                    #   action_reset_conversation, display-contract methods
                    #   (user_said, agent_*, metrics, server_*, …)
  app.tcss          # Textual CSS
  widgets.py        # UserTurn, AgentTurn, CopyButton, ToolCard, NoticeCard,
                    #   ErrorCard, ApprovalCard, StateRow, ModelRow, ToolsRow,
                    #   ControlRow, StatusFooter, ServerRow, SplashScreen,
                    #   SettingsScreen
  display.py        # TurnMetrics dataclass + TYPE_CHECKING-only `Display` alias
                    #   to VoiceAgentApp (kept so other modules can annotate
                    #   without an import cycle)
  audio.py          # VADRecorder (Silero ONNX), AudioPlayer
  pipeline.py       # Async _run_vad loop, _process_turn,
                    #   run_pipeline_loops entrypoint
  providers.py      # TranscriptVoiceWorkflow, WhisperCppSTTModel,
                    #   StreamingTTSModel, AudioPassthroughSTTModel,
                    #   create_agent / create_pipeline, _hosted_tools
  gemini_tts.py     # GeminiTTSModel — wraps Gemini's native generateContent
                    #   (OpenAI-compat doesn't cover TTS)
  shell.py          # run_shell_command tool + per-invocation approval flow
  servers.py        # ServerManager.reconcile() — per-role starter/stopper
  mcp.py            # load_mcp_servers() with ${ENV} expansion + `enabled` toggle
  preferences.py    # load_preferences / save_preferences (preferences.toml)
  config.py         # Settings + ModelConfig + _parse_catalog +
                    #   _validate_active_requirements + load_settings
```

## Configuration files

- `config.toml` — committed. Everything that isn't a model catalog: `[general]`, `[local]` server URLs, `[vad]`, `[display]`, `[audio]`, `[agent]`, `[shell]`. Catalogs moved out of here (see `models.toml`). Input is always Silero VAD — push-to-talk was removed.
- `models.toml` — committed. Catalog-style: `[[stt]] / [[llm]] / [[tts]]` arrays, each entry marks `provider = "cloud" | "local"`. Cloud entries can set `vendor = "gemini"`, per-model `api_key = "${VAR}"`, and on LLMs `reasoning_effort` / `hosted_tools`. This is what the Settings modal picks from.
- `preferences.toml` — gitignored. Three-line file `[active]` with the active `name` per role. Written by the Settings modal. Copy from `preferences.toml.example`.
- `.env` — gitignored. `OPENAI_API_KEY`, `GEMINI_API_KEY`, any custom keys referenced via `${VAR}` in a model `api_key`, plus env overrides.
- `mcp_servers.toml` — gitignored. MCP server definitions. Copy from `mcp_servers.toml.example`. Per-server `enabled = false` skips a server without deleting it.
- `llamacpp-models.ini` — gitignored. llama-server model preset file. Used only when any LLM entry has `server = "llamacpp"`. Copy from `llamacpp-models.ini.example`.
- `model_deps.toml` — committed. Maps name patterns to pip/brew deps.
- `setup-whispercpp.sh` / `setup-llamacpp.sh` — committed.

Priority: environment variable > `.env` > `config.toml`.

## Key architecture

### Textual App, not Rich Live

`VoiceAgentApp(App)` in [voice-agent/app.py](voice-agent/app.py) is the single UI entrypoint. It:

- composes a `VerticalScroll(#conversation)` above a docked `StatusFooter`
- pushes a `SplashScreen` modal during initial server setup and pops it when ready
- runs the entire pipeline (MCP connect + server reconcile + VAD loop) in a single Textual worker launched from `on_mount`
- exposes the former `Display` surface (`user_said`, `agent_start/chunk/end`, `tool_call/result`, `metrics`, `vad_*`, `processing`, `interrupted`, `server_*`, `api_error/*`, …) as methods. `providers.py`, `audio.py`, `servers.py`, and `pipeline.py` all call into these.

The `Display` name in type annotations is a `TYPE_CHECKING` alias for `VoiceAgentApp` in [voice-agent/display.py](voice-agent/display.py) — keeps import cycles at bay without introducing a real subclass.

### Key and click handling

Key bindings on `VoiceAgentApp.BINDINGS` route to `action_*` methods. Buttons in `ControlRow` dispatch the same actions via `on_button_pressed`. Never use raw `sys.stdout` writes; Textual owns the screen.

| Key       | Action                        | Notes                                 |
|-----------|-------------------------------|---------------------------------------|
| Space     | `action_interrupt`            | no-op unless `self.responding`        |
| M         | `action_toggle_mute`          | flips `self.is_muted`                 |
| R         | `action_reset_conversation`   | blocked while responding              |
| S         | `action_open_settings`        | blocked while responding              |
| Y / N     | `action_approve` / `action_decline` | resolves the pending `ApprovalCard` |
| Q, Ctrl+C | `action_quit`                 | sets `self.quit_event` + hard timeout |

### Per-role provider model

Config has a *catalog* per role, not a single `voice_mode`:

```toml
[[stt]]  name = "whisper-turbo"  provider = "local"  model = "..."
[[stt]]  name = "gpt-4o-transcribe" provider = "cloud" model = "..."
```

`ModelConfig` ([voice-agent/config.py](voice-agent/config.py)) has the fields for all roles; `_parse_catalog` validates role-specific ones (e.g., `server` / `preset` on local LLMs, `voice` on TTS, `hosted_tools` / `reasoning_effort` on cloud LLMs, `vendor` only on cloud, `api_key` only on cloud). `Settings.stt / llm / tts` are properties that look up the active entry by name.

`ModelConfig.display_name == f"{name} ({provider})"` is the label used in the Settings modal and per-turn metrics. `name` is the stable key used in `preferences.toml`.

**Per-model `api_key`.** Cloud entries can set `api_key = "${GEMINI_API_KEY_LEGACY}"` or literal. `_expand_env()` in [config.py](voice-agent/config.py) resolves `${VAR}` refs against the current env (post-dotenv). `_validate_active_requirements` treats a model with its own key as self-sufficient — the vendor-wide `OPENAI_API_KEY` / `GEMINI_API_KEY` is only required when *some* active cloud model for that vendor doesn't have its own key.

**`reasoning_effort` on LLMs.** Accepts `none | minimal | low | medium | high | xhigh`. Plumbed into the Agent via `ModelSettings(reasoning=Reasoning(effort=...))`, which the chat completions path reads as `reasoning_effort` and the Responses path reads via `reasoning.effort`. Defaults matter a lot for voice — on Gemini 3 preview flash and GPT-5, the default budget blows TTFT to ~16s; `"minimal"` drops it to ~1s. Hosted OpenAI tools require ≥ `"low"` (tool calls are part of the reasoning loop).

### Runtime switching

`VoiceAgentApp.action_open_settings` pushes a `SettingsScreen(ModalScreen)` with three `Select` dropdowns. On Apply, `switch_models(stt, llm, tts)`:

1. Holds `self._switch_lock` so no turn runs mid-swap
2. Mutates `self.settings.active_*`
3. `save_preferences(stt, llm, tts)` writes `preferences.toml` **immediately** (so a crash mid-reconcile doesn't desync the file)
4. Calls `self.server_manager.reconcile()` — see below
5. Calls `create_pipeline(...)` to rebuild `self.workflow` + `self.pipeline`
6. On any failure, reverts the three active fields, re-saves preferences with the previous selection, and mounts an `ErrorCard`

The turn loops in [voice-agent/pipeline.py](voice-agent/pipeline.py) acquire `app._switch_lock` around each turn and read `app.workflow` / `app.pipeline` **fresh** per turn, so a swap takes effect on the next turn with no loop restart.

### Reset action

`action_reset_conversation` (bound to `R` and the *Reset* button) calls `_reset_conversation()` under `_switch_lock`:

1. Rebuilds the pipeline via `create_pipeline(...)` — same code path as switching, but without calling `reconcile()` since no models changed. Gives us a fresh `TranscriptVoiceWorkflow` with an empty `_input_history`.
2. Removes every child of the `#conversation` `VerticalScroll`, clears `_current_agent_turn` / `_current_tool_card` / `_pending_user_turn` / `_last_metrics`.
3. Mounts a `NoticeCard("Conversation reset.")`.

Blocked while `self.responding` is true — mounts a notice instead.

### ServerManager (reconciler)

`ServerManager` in [voice-agent/servers.py](voice-agent/servers.py) is a per-role reconciler, not a "start all / stop all" batch.

- `_procs: dict[Role, Popen]` — at most one process per role
- `_started_for: dict[Role, str]` — which model name launched that process; a mismatch on reconcile means restart
- `reconcile()`: stop what's no longer needed, start what is, wait for health, optionally call `server_all_ready`
- `stop()`: terminate everything (called on app exit)

One-off details:

- **mlx-audio** (TTS port 8000) takes no model at startup — the model is in each request body, so swapping between local TTS entries does not restart it.
- **whisper-server** (STT port 9000) takes `-m <model>` at startup — swapping local STT models restarts.
- **LLM server** (port 8080) command depends on the active LLM's `server` field (`mlx-vlm` / `mlx-lm` / `llamacpp`); swapping between local LLMs restarts. Health endpoint differs per backend (`/v1/models` for mlx-*, `/health` for llamacpp).

Deps (`_ensure_whisper`, `_ensure_llm`, `_ensure_tts`) run the first time each role is started, not up front. `_apply_patches` still handles the misaki/phonemizer Kokoro quirk when the active TTS matches `"kokoro"`.

### Providers

In [voice-agent/providers.py](voice-agent/providers.py):

- `create_agent(settings, mcp_servers, app)` — three LLM paths:
  - **Local** → `OpenAIChatCompletionsModel` around `AsyncOpenAI(base_url=llm_url/v1)`.
  - **Cloud, `vendor = "gemini"`** → `OpenAIChatCompletionsModel` around `AsyncOpenAI(base_url=_GEMINI_OPENAI_BASE)`. The client uses `trust_env=False` and `default_headers={"OpenAI-Organization": Omit(), "OpenAI-Project": Omit()}` so env vars like `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` don't inject extra credentials that trip Gemini's "Multiple authentication credentials received" 400.
  - **Cloud OpenAI (default)** → `OpenAIResponsesModel` (not Chat Completions — hosted tools live on `/responses`) around `AsyncOpenAI(base_url=_OPENAI_BASE)`, same `Omit()` / `trust_env=False` defense. Without the explicit `base_url`, a leftover `OPENAI_BASE_URL` env var can silently redirect GPT requests to a non-OpenAI endpoint.
  - `reasoning_effort` is wired via `ModelSettings(reasoning=Reasoning(effort=...))` on the Agent.
  - Hosted tools (`settings.llm.hosted_tools`) go through `_hosted_tools()`. Validated at catalog-parse time to be OpenAI-only (rejected on Gemini or local).
  - Shell tool (if `settings.shell.enabled`) is attached with the approval-flow hook onto the app.
- `create_pipeline_config` — local TTS ⇒ `OpenAIVoiceModelProvider(base_url=tts_url/v1)`, cloud ⇒ default provider. Also sets `text_splitter=_eager_sentence_splitter()` (handles decimals so TTS doesn't speak `3.` as "three").
- `create_pipeline` — TTS branches:
  - Local ⇒ `StreamingTTSModel` (adds `stream=True` to the mlx-audio request).
  - `vendor = "gemini"` ⇒ `GeminiTTSModel` from `gemini_tts.py` (native `generateContent` with retries + RIFF header stripping).
  - Cloud OpenAI with a per-model `api_key` ⇒ explicit `OpenAITTSModel` around a dedicated `AsyncOpenAI` (so the override takes precedence over the provider's env-driven key).
  - Otherwise a bare model string — handed to the pipeline-config provider.
- STT branches similarly:
  - Local ⇒ `WhisperCppSTTModel`. Wrapped in `AudioPassthroughSTTModel` **only** when STT and LLM are both local and the active LLM has `audio_input = true`.
  - Cloud OpenAI with a per-model `api_key` ⇒ explicit `OpenAISTTModel` with a dedicated client.
  - Otherwise a bare model string.

### TranscriptVoiceWorkflow

The core interception layer. It overrides `SingleAgentVoiceWorkflow.run()` — it does **not** call `super().run()`; it replicates the parent's logic so it can intercept:

- `run_item_stream_event` → `tool_called` / `tool_output` → `display.tool_call / tool_result`
- `raw_response_event` with `response.output_text.delta` → `display.agent_chunk` + `_partial_response += chunk`

Preserve `self._input_history = result.to_input_list()` and `self._current_agent = result.last_agent` at the end of `run()` — the parent relies on those between turns.

`save_partial_history()` appends the partial assistant text with `[interrupted]` to history on Space-interrupt.

### Concurrency model

The pipeline runs three concurrent tasks in `_run_vad()` (the only input loop — push-to-talk was removed):

1. `recorder.run(quit_event)` — continuous Silero VAD pushing segments to `recorder.segments`
2. `mute_watcher()` — mirrors `app.is_muted` onto the recorder
3. the main loop — consumes `recorder.segments`, spawns `_process_turn` as a cancellable task, races it against `app.interrupt_event`

Key facts for the concurrency:

- **No `read_key()` / cbreak anymore.** Textual owns the terminal. All keys come in via `BINDINGS` and set `app.is_muted` or `app.interrupt_event`.
- **Pipeline state is on the app.** `app.workflow`, `app.pipeline`, `app.server_manager`, `app.mcp_servers`, `app._switch_lock` — loops read these fresh per turn.
- `VADRecorder.run()` must yield (`await asyncio.sleep()`) or the event loop starves the UI.
- Any blocking I/O has to go through `run_in_executor` (`AudioPlayer.play` already does).

### Display contract

The app implements ~30 methods that providers / audio / servers / pipeline call. Don't inline drawing anywhere else. The key families:

- **Conversation**: `user_said`, `agent_start/chunk/end`, `tool_call`, `tool_result`, `interrupted`, `metrics`
- **VAD / state**: `vad_speaking`, `vad_silence`, `vad_clear`, `listening`, `muted`, `unmuted`, `processing`, `ready_banner`
- **Server lifecycle (drives splash)**: `server_setup_start`, `server_starting`, `server_waiting`, `server_ready_one`, `server_all_ready`, `server_failed`, `server_timeout`, `server_install*`, `server_patched`, `setup_failed`
- **Errors (mount as inline `ErrorCard`)**: `api_error`, `api_error_with_logs`, `connection_error`, `auth_error`, `rate_limit_error`, `tts_stream_error`
- **Shell approval**: `request_shell_approval(command)` awaits the pending `ApprovalCard` and returns bool
- **Tools**: `set_mcp_tools(list[str])`

When a new role needs to call the app, add a method here (not a widget subclass).

`AudioPlayer.play()` wraps `result.stream()` — which yields events for the **whole pipeline** (STT → LLM → TTS) — so any upstream failure (Gemini 503, auth error, etc.) reraises out of there. The exception handler in `audio.py` routes those through `display.api_error(str(e))` unlabeled, not as "TTS error" (historical bug — mislabeled Gemini LLM failures as TTS). Don't reintroduce the prefix.

### Copy button on turn cards

`CopyButton(Static)` in [widgets.py](voice-agent/widgets.py) is mounted in the `card-header` `Horizontal` of every `UserTurn` and `AgentTurn`. It takes a `Callable[[], str]` rather than a raw string, so for `AgentTurn` (whose `text` reactive is still streaming) each click copies whatever's present at click-time.

`on_click` calls `self.app.copy_to_clipboard(text)` (Textual's built-in OSC-52 escape), flashes `✓ copied`, then resets after 1.5s. If the terminal doesn't forward OSC-52, the clipboard write is silently no-op — no fallback to subprocess `pbcopy` currently.

### UserTurn placeholder & STT ordering

Because `AudioPassthroughSTTModel` fires real STT in a background task, the agent often starts streaming **before** we have the transcription. To keep user/agent turn order visually correct:

- `app.processing(duration)` mounts an empty `UserTurn` with a dim `…` placeholder and stashes it in `self._pending_user_turn`.
- `app.user_said(text, stt_seconds)` fills the placeholder if one exists (mutates reactive attrs); otherwise it mounts a fresh one.

STT timing + model name are shown on the `UserTurn` (`STT [whisper-turbo (local)] 0.4s`). The `AgentTurn` metrics line carries LLM (with TTFT), TTS, and Total — e.g. `LLM [gemini-3.1-flash-lite-preview (cloud)] 1.4s (TTFT 1.0s) · TTS [...] 0.8s · Total 2.2s`. The `llm_first_token_seconds` field on `TurnMetrics` captures the time between `Runner.run_streamed` firing and the first `response.output_text.delta` event — useful for spotting providers that batch-stream (Gemini preview models at high reasoning budgets).

### Hosted OpenAI tools

Per-LLM in `models.toml`:

```toml
[[llm]]
provider         = "cloud"
vendor           = "openai"           # default; hosted tools are rejected on "gemini"
reasoning_effort = "low"              # ≥ "low" — tools are part of the reasoning loop
hosted_tools     = ["web_search", "code_interpreter"]
# for file_search, also:
# file_search_vector_stores = ["vs_abc123"]
# file_search_max_results   = 5
```

`_hosted_tools(llm)` in `providers.py` constructs `WebSearchTool()`, `CodeInterpreterTool(tool_config={"type": "code_interpreter", "container": {"type": "auto"}})`, and `FileSearchTool(vector_store_ids=…, max_num_results=…)`. Config-time validation rejects hosted tools on local LLMs, on non-OpenAI vendors (Gemini), and rejects `file_search` without vector stores. Hosted tools require the Responses API — hence why the OpenAI cloud path uses `OpenAIResponsesModel` rather than `OpenAIChatCompletionsModel`.

### Gemini integration

- **LLM** uses Gemini's OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`). We reuse `OpenAIChatCompletionsModel` with the alternate `base_url`. Caveats:
  - The compat endpoint rejects newer `AQ.`-prefix API keys with `Multiple authentication credentials received`. Point the LLM at a legacy `AIza`-prefix key via `api_key = "${GEMINI_API_KEY_LEGACY}"`.
  - `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` env vars auto-inject `openai-organization` / `openai-project` headers that Gemini treats as extra credentials → same 400. Our client passes `Omit()` for those headers.
  - `trust_env=False` on the httpx client prevents system proxy envs from adding a third credential.
- **TTS** has no OpenAI-compat counterpart. [`gemini_tts.py`](voice-agent/gemini_tts.py) wraps the native `generateContent` endpoint: one-shot request that returns base64 PCM, retried on 429/5xx with longer backoff for 429. Authenticates via `x-goog-api-key` (not `Authorization`), so the `AQ.` keys work here.
- `gemini-3.1+` TTS supports inline audio tags. The `[agent.model-instructions]` block in `config.toml` uses `"re:gemini-3\\.[1-9].*tts"` to attach tag-usage guidance only when a matching TTS is active.

### MCP servers

`load_mcp_servers()` in [voice-agent/mcp.py](voice-agent/mcp.py) reads `mcp_servers.toml`, expands `${VAR}` against env/`.env`, and returns connected `MCPServer` instances. Per-server `enabled = false` skips the entry. Called **exactly once** per app run from `VoiceAgentApp._run_pipeline`; same instances are passed to `create_agent(..., mcp_servers=…)`.

### Partial history on interruption

`TranscriptVoiceWorkflow._partial_response` accumulates as tokens stream. On Space, `pipeline._run_vad` cancels the turn task, calls `save_partial_history()`, and `app.interrupted()` mounts an "-interrupted" class on the current `AgentTurn`. Next turn sees the partial assistant reply in history with `[interrupted]`.

## Development commands

```bash
uv run pyright voice-agent              # Type check (whispercpp/ is vendored; ignore)
uv run ruff format voice-agent/         # Format
uv run ruff check --fix voice-agent/    # Lint
uv run python -m voice-agent            # Run
./setup.sh                              # Install deps (core + local)
./setup.sh --update                     # Update all deps
```

## Common pitfalls

- **Package name has a hyphen.** `voice-agent/`, not `voice_agent/`. Imports use the hyphen verbatim (`from voice-agent.app import …`). In Python code the hyphen works because we import via `importlib.import_module` in tests, and relative imports inside the package are fine.
- **`voice_mode` is gone.** Don't reintroduce it. Branch on `settings.<role>.provider` (and `settings.<role>.vendor` for cloud) instead. Each role is independent.
- **`input_mode` is gone.** Push-to-talk was removed — input is always Silero VAD. Don't add it back without explicit ask.
- **`ModelConfig.name` vs `display_name`.** Preferences / config use `name` (bare). Anything shown to the user should use `display_name` (adds `(local)` / `(cloud)`).
- **Hosted tools are OpenAI-cloud-only.** `_parse_catalog` raises on a local LLM or `vendor = "gemini"` LLM with `hosted_tools`. Keep that check. Hosted tools also require the Responses API, so the OpenAI cloud path must keep using `OpenAIResponsesModel`.
- **Don't strip the `Omit()` headers or `trust_env=False`** on the Gemini / OpenAI `AsyncOpenAI` clients in `create_agent`. They defend against env-var-induced auth conflicts (`OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`) and rogue proxies that trigger Gemini's "Multiple authentication credentials received" 400.
- **Gemini's OpenAI-compat endpoint doesn't accept `AQ.`-prefix keys.** Point Gemini LLM entries at a legacy `AIza` key via `api_key = "${GEMINI_API_KEY_LEGACY}"`. The TTS adapter uses `x-goog-api-key` on the native endpoint and accepts either key format.
- **`reasoning_effort` + hosted tools.** On OpenAI cloud LLMs, hosted tools need `reasoning_effort >= "low"` (tool calls are part of the reasoning loop). `"minimal"` or `"none"` will be rejected by OpenAI. On Gemini 3 preview, default reasoning budget makes TTFT ~16s — always set `reasoning_effort = "minimal"` unless you really need reasoning quality.
- **Audio-passthrough gating.** Only wrap STT with `AudioPassthroughSTTModel` when **both** STT and LLM are local (and the LLM's `audio_input = true`). Otherwise the audio never reaches a model that can consume it.
- **Don't capture `workflow` / `pipeline` in the loop.** Read `app.workflow` / `app.pipeline` fresh each turn so runtime swaps and resets apply. The `_switch_lock` prevents the swap/reset from racing a turn.
- **Textual owns the terminal.** No `sys.stdout.write`, no `tty.setcbreak`, no `termios` calls. Use `app._mount_card`, `app._set_state`, etc.
- **Rich markup in agent text.** The LLM may emit `[brackets]` (e.g. `[laugh]`) — Textual's `Text(text)` is fine (we construct it as plain text), but if you ever switch back to `Text.from_markup`, escape first with `rich.markup.escape`.
- **Pipeline errors surface through `AudioPlayer.play`.** `result.stream()` yields events for the whole STT→LLM→TTS chain. The `except` branch there routes through `display.api_error(str(e))` without a prefix — don't re-add `f"TTS error: {e}"`, it mislabels Gemini 503s etc.
- **`ServerManager.reconcile` is idempotent.** Call it freely (startup + every switch). The `_started_for` bookkeeping decides if a running process is still OK. `action_reset_conversation` deliberately does **not** call it (the active models haven't changed).
- **mlx-audio is model-agnostic.** Don't restart it when swapping local TTS — it takes the model in the request body. Only whisper-server and the LLM server restart on a model change.
- **MCP loading happens once.** In `VoiceAgentApp._run_pipeline`. Don't call `load_mcp_servers()` anywhere else; it creates *new* disconnected server instances and the Agent won't see the live ones. The reset/switch flows reuse `self.mcp_servers`.
- **VAD must yield.** `VADRecorder.run()` has `await asyncio.sleep(0.005)` when no data is available — keep it, or you'll starve the Textual event loop.
- **misaki/espeak patch.** `ServerManager._apply_patches` replaces library + data paths and deletes `.pyc` caches. Only runs for TTS roles whose active model name contains `"kokoro"`. Don't widen it blindly.
- **Whisper model name.** In local STT, `ModelConfig.model` is a whisper.cpp model file suffix like `large-v3-turbo-q5_0` (matches `ggml-{name}.bin`), *not* an MLX path. `setup-whispercpp.sh` manages these files.
- **`tts_voice` is optional.** Some local TTS models (chatterbox) don't take a voice. `ModelConfig.voice = None` is handled by `create_pipeline_config`.
- **Preferences fallback.** Unknown name in `preferences.toml` warns and falls back to the first catalog entry — don't crash on it.
- **Splash buffering.** `SplashScreen` methods (`log_line`, `set_waiting`, `set_ready`, `set_failed`) buffer until the screen is mounted, so they're safe to call from the moment the worker starts.
- **Sentence splitter is decimal-aware.** `_eager_sentence_splitter` in `providers.py` holds back `digit + "."` at buffer-end to avoid flushing mid-decimal (which would make TTS read `3.14` as "three"). If you touch it, keep that check.
