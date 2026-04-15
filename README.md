# Voice Agent

A real-time speech-to-speech voice agent built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). It uses a 3-model pipeline (STT, LLM, TTS) and supports both cloud (OpenAI API) and local (MLX on Apple Silicon) inference.

## Features

- **3-model pipeline**: Speech-to-Text -> LLM -> Text-to-Speech, each independently configurable
- **Cloud or local**: Switch between OpenAI API and local MLX models via config
- **Voice Activity Detection (VAD)**: Continuous listening with automatic speech detection using webrtcvad
- **Push-to-talk**: Alternative manual recording mode
- **Interruption**: Press Space to interrupt the agent mid-response
- **Mute**: Press M to toggle microphone on/off (fully releases the mic)
- **MCP support**: Connect MCP servers to give the agent tools (search, file access, APIs, etc.)
- **Auto-setup**: Local mode automatically installs dependencies, patches compatibility issues, downloads models, and starts servers
- **Persistent TUI**: Rich-based footer showing state, models, tools, controls, and metrics
- **Performance metrics**: STT, LLM (tokens/sec), and TTS timing after each turn
- **Conversation history**: Partial responses are preserved when interrupted

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4) for local mode
- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **espeak-ng** (installed automatically via Homebrew for local TTS models that need it)
- An **OpenAI API key** for cloud mode

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd voice-agent
./setup.sh
```

Or manually:

```bash
uv sync --extra local
```

### 2. Configure

Edit `config.toml` to choose your mode:

```toml
[general]
voice_mode = "local"    # "local" or "cloud"
input_mode = "vad"      # "vad" or "push_to_talk"
```

For cloud mode, add your API key to `.env`:

```
OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
./start.sh
```

Or:

```bash
uv run python -m voice-agent
```

In local mode, the agent will automatically:
1. Install `mlx-audio` and `mlx-vlm` if needed
2. Install model-specific dependencies (see `model_deps.toml`)
3. Install system packages via Homebrew (e.g., `espeak-ng`)
4. Patch known compatibility issues
5. Start the STT/TTS server (mlx-audio) and LLM server (mlx-vlm)
6. Wait for models to download and servers to be ready
7. Begin listening

## Controls

### VAD Mode

| Key     | Action                                  |
|---------|-----------------------------------------|
| (speak) | Automatically detected and processed    |
| Space   | Interrupt the agent while it's speaking |
| M       | Toggle microphone mute/unmute           |
| Q       | Quit                                    |

### Push-to-Talk Mode

| Key | Action               |
|-----|----------------------|
| K   | Start/stop recording |
| Q   | Quit                 |

## Configuration

All configuration lives in `config.toml` (committed) and `.env` (gitignored, secrets only).

Environment variables override `.env`, which overrides `config.toml`.

### config.toml

```toml
[general]
voice_mode = "local"        # "cloud" or "local"
input_mode = "vad"          # "vad" or "push_to_talk"

[cloud]
stt_model = "gpt-4o-transcribe"
tts_model = "gpt-4o-mini-tts"
llm_model = "gpt-4o-mini"
tts_voice = "alloy"         # optional

[local]
stt_model = "mlx-community/whisper-large-v3-turbo-asr-8bit"
tts_model = "mlx-community/Kokoro-82M-bf16"
llm_model = "mlx-community/gemma-4-e4b-it-4bit"
tts_voice = "af_heart"      # optional, some models don't need one
audio_url = "http://localhost:8000"
llm_server = "mlx-vlm"     # "mlx-vlm" for vision models, "mlx-lm" for text-only
llm_url = "http://localhost:8080"

[vad]
aggressiveness = 2          # 0-3, higher = more aggressive filtering
silence_ms = 500            # Silence duration before processing
energy_threshold = 60       # Minimum RMS energy for speech detection

[display]
show_transcript = true      # Show user/agent text in console
show_metrics = true         # Show STT/LLM/TTS timing

[agent]
instructions = "You are a helpful voice assistant. Today is {date}."

[audio]
sample_rate = 24000
```

Agent instructions support variable substitution:

| Variable | Example output |
|---|---|
| `{date}` | April 15, 2026 |
| `{time}` | 2:30 PM |
| `{datetime}` | April 15, 2026 2:30 PM |
| `{os}` | Darwin |
| `${VAR_NAME}` | Environment variable from `.env` or system |

### .env

```
OPENAI_API_KEY=sk-...
```

Any `config.toml` value can be overridden via environment variable. For example:
- `VOICE_MODE=cloud` overrides `general.voice_mode`
- `LOCAL_LLM_MODEL=mlx-community/some-model` overrides `local.llm_model`

### model_deps.toml

Maps model name patterns to their required pip and system dependencies. When a model name contains a pattern key, the listed dependencies are auto-installed before the server starts.

```toml
[kokoro]
deps = ["misaki", "num2words", "spacy", "phonemizer", "espeakng-loader"]
brew = ["espeak-ng"]
```

## MCP Servers (Tools)

The agent can use [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) servers to access external tools like search, file access, APIs, etc.

### Setup

```bash
cp mcp_servers.toml.example mcp_servers.toml
```

Edit `mcp_servers.toml` to add your servers:

```toml
# Stdio server (runs a local command)
[filesystem]
type = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
cache_tools = true

# HTTP server (connects to a running server)
[my-api]
type = "http"
url = "http://localhost:3000/mcp"
```

Use `${VAR_NAME}` to reference secrets from `.env`:

```toml
[search]
type = "http"
url = "http://localhost:3000/mcp"
env = { API_KEY = "${MY_SECRET_KEY}" }
```

`mcp_servers.toml` is gitignored so your server configurations stay private. Tool calls and results are shown in the conversation transcript.

## Architecture

```
User speaks -> Mic (24kHz) -> VAD (webrtcvad) -> Speech segment
    -> STT (whisper) -> Text transcription
    -> LLM (gemma/gpt) -> [Tool calls] -> Response text (streamed)
    -> TTS (kokoro/openai) -> Audio (streamed)
    -> Speaker playback
```

### Package Structure

```
voice-agent/
  __init__.py
  __main__.py       # Entry point
  config.py         # Settings from config.toml + .env, validation
  audio.py          # VADRecorder, AudioPlayer, mic/speaker I/O
  display.py        # Rich TUI with persistent footer
  pipeline.py       # Async conversation loop with interruption
  providers.py      # Model providers, streaming TTS, transcript workflow
  servers.py        # Local mlx-audio/mlx-vlm server lifecycle
  mcp.py            # MCP server loading from mcp_servers.toml
```

### Key Design Decisions

- **OpenAI-compatible endpoints**: Both mlx-audio and mlx-vlm expose OpenAI-compatible APIs, so we reuse the SDK's existing `OpenAISTTModel`, `OpenAITTSModel`, and `OpenAIChatCompletionsModel` pointed at localhost.
- **VAD with pre-roll**: A 100ms ring buffer captures audio before speech onset, preventing clipped words. Speech requires 3 consecutive frames (60ms) to confirm, filtering transient noise.
- **Echo suppression**: The mic is muted during agent response to prevent the agent from hearing its own voice through speakers. Press Space to interrupt instead.
- **Streaming**: LLM tokens stream via SSE, TTS audio streams via chunked HTTP (`stream=True`), and the SDK's sentence splitter sends text to TTS at sentence boundaries rather than waiting for the full response.
- **Tool visibility**: MCP tool calls and results are displayed in the conversation transcript so the user can see what the agent is doing.

## Scripts

| Script | Purpose |
|--------|---------|
| `./start.sh` | Run the voice agent |
| `./setup.sh` | Install all dependencies (core + local) |
| `./setup.sh --update` | Update all dependencies to latest versions |

## Development

```bash
# Type check
uv run pyright

# Lint and format
uv run ruff format .
uv run ruff check --fix .

# Run
uv run python -m voice-agent
```

## Troubleshooting

### Server logs

When running in local mode, server logs are saved to `logs/`:
- `logs/mlx-audio_port_8000.log` -- STT/TTS server
- `logs/mlx-vlm_port_8080.log` -- LLM server

API errors automatically display the relevant server log tail.

### Choosing between mlx-lm and mlx-vlm

The `llm_server` setting in `[local]` controls which server runs the LLM:

- **`mlx-vlm`**: For vision-language models (gemma-4, qwen2-vl, etc.)
- **`mlx-lm`**: For text-only models (gpt-oss, llama, qwen, etc.)

If your model isn't supported by one, try the other. Both expose the same OpenAI-compatible API.

### VAD too sensitive / not sensitive enough

Adjust in `config.toml`:
- `energy_threshold`: Raise to filter more noise (default 60, speech is typically 50-500+ RMS)
- `aggressiveness`: Lower (0-1) for quieter environments, higher (2-3) for noisy ones
- `silence_ms`: Raise if it cuts you off mid-sentence, lower for faster response

### Missing dependencies

The agent auto-installs model dependencies listed in `model_deps.toml`. If you hit a new missing module, add it to the appropriate section and restart.
