# Voice Agent

A real-time speech-to-speech voice agent built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). It uses a 3-model pipeline (STT, LLM, TTS) where each role can be cloud (OpenAI API) or local (MLX on Apple Silicon / whisper.cpp / llama.cpp) independently. The terminal UI is a fullscreen Textual app with clickable controls and flicker-free streaming.

## Features

- **Mix-and-match per-role providers**: pick STT, LLM, and TTS each from a cloud or local catalog in `config.toml`. Local servers only start for the roles you actually select.
- **Runtime model switching**: press `S` (or click *Switch models*) to open a modal picker; the active choice is saved to `preferences.toml`.
- **Fullscreen Textual TUI**: card-per-turn conversation scrolls above a persistent status footer; clickable Mute / Interrupt / Switch / Quit buttons plus keyboard shortcuts.
- **Voice Activity Detection (VAD)**: Silero VAD (ONNX) provides continuous listening with pre-roll; `push_to_talk` mode available as an alternative.
- **Interruption**: press Space (or click Interrupt) to cut the agent off mid-response.
- **Mute**: press M to release the mic entirely during a response.
- **MCP tools**: connect MCP servers via `mcp_servers.toml` with per-server `enabled` toggle.
- **OpenAI-hosted tools**: cloud LLM entries can enable `web_search`, `code_interpreter`, and `file_search` directly from `config.toml`.
- **Auto-setup for local roles**: installs Python deps, brew deps, builds whisper.cpp, downloads the LLM binary, and patches known compatibility issues.
- **Per-turn metrics**: STT, LLM, TTS, and total timing inline with each turn.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4) for local roles
- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **espeak-ng** (auto-installed via Homebrew for Kokoro TTS)
- An **OpenAI API key** for any cloud role

## Quick start

```bash
git clone <repo-url>
cd voice-agent
./setup.sh        # installs core + local dependencies
./start.sh        # runs the app
```

Or manually:

```bash
uv sync --extra local
uv run python -m voice-agent
```

If any of your active models are cloud, add your API key to `.env`:

```
OPENAI_API_KEY=sk-...
```

## Controls

All controls are also clickable buttons in the bottom footer. The Interrupt button only appears while the agent is responding.

### VAD mode

| Key     | Action                                           |
|---------|--------------------------------------------------|
| (speak) | Speech is detected automatically (Silero VAD)    |
| Space   | Interrupt the agent while it's speaking          |
| M       | Toggle microphone mute                           |
| S       | Open the Switch-models modal                     |
| Q       | Quit                                             |

### Push-to-talk mode

| Key | Action                          |
|-----|---------------------------------|
| K   | Start / stop recording          |
| S   | Open the Switch-models modal    |
| Q   | Quit                            |

## Configuration

Config lives in three files at the project root:

| File                | Purpose                                                                             | Checked in? |
|---------------------|-------------------------------------------------------------------------------------|-------------|
| `config.toml`       | Model catalogs, local server URLs, VAD, agent, display, audio                       | ✅          |
| `preferences.toml`  | Active model name per role; auto-written by the Switch modal                        | ❌          |
| `.env`              | Secrets (`OPENAI_API_KEY`)                                                          | ❌          |
| `mcp_servers.toml`  | MCP server definitions                                                              | ❌          |
| `models.ini`        | llama-server model preset file (only when any LLM uses `server = "llamacpp"`)       | ❌          |

Environment variables override `.env`, which overrides `config.toml`.

### config.toml structure

```toml
[general]
input_mode = "vad"                 # "vad" or "push_to_talk"
enable_mcp = true                  # load MCP servers from mcp_servers.toml

[local]                            # only consulted for roles whose active model is local
stt_url = "http://localhost:9000"  # whisper-server
tts_url = "http://localhost:8000"  # mlx-audio
llm_url = "http://localhost:8080"  # mlx-vlm / mlx-lm / llama-server

# ── STT catalog ─────────────────────────────────────
# The "(local)" / "(cloud)" suffix is added automatically from `provider`.
# The first entry is the default when preferences.toml is absent.
[[stt]]
name     = "whisper-turbo"
provider = "local"
model    = "large-v3-turbo-q5_0"   # whisper.cpp model file suffix

[[stt]]
name     = "gpt-4o-transcribe"
provider = "cloud"
model    = "gpt-4o-transcribe"

# ── LLM catalog ─────────────────────────────────────
[[llm]]
name        = "gemma-4-e4b-it (llamacpp)"
provider    = "local"
server      = "llamacpp"           # "mlx-vlm" | "mlx-lm" | "llamacpp"
model       = "gemma-4-e4b-it"     # must match a section name in the preset
preset      = "models.ini"
audio_input = true                 # pass mic audio straight to the LLM (local-STT + local-LLM only)

[[llm]]
name            = "gemma-4-e4b-it (mlx-vlm)"
provider        = "local"
server          = "mlx-vlm"
model           = "mlx-community/gemma-4-e4b-it-4bit"
audio_input     = true
kv_bits         = 3.5              # mlx-vlm only
kv_quant_scheme = "turboquant"     # mlx-vlm only

[[llm]]
name     = "gpt-5.4-mini"
provider = "cloud"
model    = "gpt-5.4-mini"
# Opt into OpenAI-hosted tools (cloud LLMs only):
# hosted_tools = ["web_search", "code_interpreter"]
# With "file_search" you must also set:
# file_search_vector_stores = ["vs_abc123"]
# file_search_max_results   = 5

# ── TTS catalog ─────────────────────────────────────
[[tts]]
name     = "kokoro"
provider = "local"
model    = "mlx-community/Kokoro-82M-bf16"
voice    = "af_heart"

[[tts]]
name     = "gpt-4o-mini-tts"
provider = "cloud"
model    = "gpt-4o-mini-tts"
voice    = "alloy"

[vad]
threshold  = 0.5                   # Silero speech probability threshold (0-1)
silence_ms = 500                   # silence before a segment is closed

[display]
show_transcript = true
show_metrics    = true             # inline STT / LLM / TTS / Total per turn

[audio]
sample_rate = 24000

[agent]
instructions     = """You are a helpful voice assistant. Today is {date}."""
tool_call_filler = "Give me just a moment."    # spoken while waiting for a tool call; optional
```

#### Runtime variables in `[agent].instructions`

| Variable      | Example                             |
|---------------|-------------------------------------|
| `{date}`      | April 15, 2026                      |
| `{time}`      | 2:30 PM                             |
| `{datetime}`  | April 15, 2026 2:30 PM              |
| `{os}`        | Darwin                              |
| `${VAR_NAME}` | Environment variable from `.env` or system |

### preferences.toml

Three lines, gitignored, written by the Switch modal (or you can create it manually):

```toml
[active]
stt = "whisper-turbo"
llm = "gpt-5.4-mini"
tts = "kokoro"
```

Each value matches the `name` field of an entry in the `config.toml` catalog (without the `(cloud)` / `(local)` suffix — that's added automatically for display). Missing or unknown names fall back to the first catalog entry and print a warning at startup. Copy `preferences.toml.example` to get started.

### .env

```
OPENAI_API_KEY=sk-...
```

Any `config.toml` scalar can be overridden via an env var — e.g. `INPUT_MODE=push_to_talk`, `STT_URL=http://...`, `VAD_THRESHOLD=0.6`.

### model_deps.toml

Maps model-name patterns to pip/brew deps. Matching deps are auto-installed the first time that role is started.

```toml
[kokoro]
deps = ["misaki", "num2words", "spacy", "phonemizer", "espeakng-loader"]
brew = ["espeak-ng"]
```

## Runtime model switching

Press `S` (or click the **Switch models** button) to open the modal. Pick one model per role and press *Apply*:

1. `ServerManager.reconcile()` stops any local server no longer needed, starts ones that are now needed, and restarts any whose active model changed.
2. The workflow + pipeline are rebuilt (fresh history — the previous conversation is not carried across a swap).
3. `preferences.toml` is rewritten with the new selection.

If a server fails to start, the active selection is reverted and an inline error card is mounted.

## MCP servers (tools)

The agent can use [MCP](https://modelcontextprotocol.io/) servers to get custom tools.

```bash
cp mcp_servers.toml.example mcp_servers.toml
```

```toml
# Stdio server (runs a local command)
[filesystem]
type    = "stdio"
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

# HTTP server (connects to a running server)
[my-api]
type = "http"
url  = "http://localhost:3000/mcp"

# Per-server toggle without deleting the entry
[experimental]
type    = "stdio"
command = "…"
enabled = false
```

Use `${VAR_NAME}` in any string value to pull secrets from `.env`. Tool calls and their results are rendered as cards in the conversation transcript.

## OpenAI-hosted tools (cloud LLMs)

Any cloud LLM entry can enable [hosted tools](https://openai.github.io/openai-agents-python/tools/):

```toml
[[llm]]
name         = "gpt-5.4-mini"
provider     = "cloud"
model        = "gpt-5.4-mini"
hosted_tools = ["web_search", "code_interpreter"]
```

Supported tools: `web_search`, `code_interpreter`, `file_search` (requires `file_search_vector_stores = [...]` and optional `file_search_max_results = N`). The app rejects `hosted_tools` on local LLMs at startup.

## Architecture

```
User speaks -> Mic (24kHz) -> Silero VAD -> Speech segment
    -> STT (whisper / gpt-4o-transcribe)         -> Text
    -> LLM (gemma / gpt-5.4-mini / …) -> [tool calls] -> Response text (streamed)
    -> TTS (kokoro / gpt-4o-mini-tts)           -> Audio (streamed)
    -> Speaker playback
```

### Package structure

```
voice-agent/
  __init__.py
  __main__.py       # Entry point; just launches VoiceAgentApp
  app.py            # Textual App: pipeline worker, display methods,
                    #   action_open_settings, switch_models, …
  app.tcss          # Textual CSS (cards, footer, splash, switch modal)
  widgets.py        # UserTurn / AgentTurn / ToolCard / NoticeCard / ErrorCard /
                    #   StateRow / ModelRow / ToolsRow / ControlRow / StatusFooter /
                    #   ServerRow / SplashScreen / ModelSwitchScreen
  display.py        # TurnMetrics dataclass + TYPE_CHECKING `Display` alias
  audio.py          # VADRecorder (Silero ONNX), AudioPlayer, record_push_to_talk
  pipeline.py       # Async loops (_run_vad / _run_push_to_talk), _process_turn,
                    #   run_pipeline_loops entrypoint
  providers.py      # TranscriptVoiceWorkflow, WhisperCppSTTModel,
                    #   StreamingTTSModel, AudioPassthroughSTTModel,
                    #   create_agent / create_pipeline, _hosted_tools
  servers.py        # ServerManager.reconcile() per-role starter
  mcp.py            # load_mcp_servers() with ${ENV} expansion + `enabled` toggle
  preferences.py    # load/save preferences.toml
  config.py         # Settings + ModelConfig, _parse_catalog, _validate_active_requirements
```

### Key design decisions

- **Per-role providers, not a single mode.** Each `[[stt]] / [[llm]] / [[tts]]` entry marks itself `provider = "cloud" | "local"`. The `ServerManager` is a reconciler: it starts only the local server(s) needed by the currently active selection and restarts on a swap.
- **OpenAI-compatible local servers.** mlx-audio, mlx-vlm, mlx-lm, and llama-server all expose `/v1/...` endpoints, so we reuse the SDK's `OpenAITTSModel` / `OpenAIChatCompletionsModel` and point `AsyncOpenAI` at localhost. Local STT uses a custom `WhisperCppSTTModel` hitting whisper-server's `/inference`.
- **Flicker-free streaming.** Each turn is its own widget; agent text is a `reactive` attribute that re-renders only that one widget per token. Rich `Live` is gone; Textual owns the screen.
- **Audio-passthrough** (`audio_input = true` on a local LLM) only engages when both STT and LLM are local. Otherwise the audio blob would be dropped.
- **Echo suppression by mic muting.** There is no AEC — we mute the microphone while the agent is speaking. Press Space to interrupt instead of speaking over it.

## Scripts

| Script                    | Purpose                                                             |
|---------------------------|---------------------------------------------------------------------|
| `./start.sh`              | Run the voice agent                                                 |
| `./setup.sh`              | Install all dependencies (core + local)                             |
| `./setup.sh --update`     | Update all dependencies                                             |
| `./setup-llamacpp.sh`     | Download/update the `llama-server` binary (llamacpp backend)        |
| `./setup-whispercpp.sh`   | Build whisper.cpp and download the selected whisper model           |

## Development

```bash
uv run pyright voice-agent
uv run ruff format voice-agent/
uv run ruff check --fix voice-agent/
uv run python -m voice-agent
```

## Troubleshooting

### Server logs

When any role is local, its server logs land in `logs/`:

- `logs/whisper-server_port_9000.log`
- `logs/mlx-audio_port_8000.log`
- `logs/llama-server_port_8080.log` / `logs/mlx-vlm_port_8080.log` / `logs/mlx-lm_port_8080.log`

On an API error the tail of the relevant log is rendered as an inline card.

### Local-server topology

Each local role has exactly one backing process, started on demand:

- **whisper-server** (STT, port 9000) — whisper.cpp HTTP server with built-in VAD. Restarted when you pick a different local STT model.
- **mlx-audio** (TTS, port 8000) — model-agnostic; the TTS model is specified per request, so swapping between local TTS models does **not** require a restart.
- **LLM server** (port 8080) — backend is per-entry:
  - `mlx-vlm` — Python module, supports `--kv-bits` / `--kv-quant-scheme`, health via `/v1/models`
  - `mlx-lm` — Python module, health via `/v1/models`
  - `llamacpp` — `llama-server` binary, health via `/health`, models from the preset file

The reconciler restarts the LLM server when you pick a different local LLM (different model or backend).

### VAD tuning

Adjust in `[vad]`:

- `threshold` — Silero speech-probability cutoff (0–1). Raise in noisy rooms.
- `silence_ms` — how long the trailing silence must be before a segment closes.

### Stale local selection

If you delete a local model or change its file layout, the next time that local server starts it'll fail and the selection reverts to your previous choice. Fix the config, then switch back via `S`.

### Missing dependencies

`model_deps.toml` maps patterns in the model name to pip/brew deps. The first time you start a role that matches, the deps install automatically. Add new entries there as needed.
