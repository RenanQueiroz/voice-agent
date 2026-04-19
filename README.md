# Voice Agent

A real-time speech-to-speech voice agent built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). It uses a 3-model pipeline (STT, LLM, TTS) where each role can be cloud (OpenAI API / Gemini) or local (whisper.cpp / llama.cpp / MLX on Apple Silicon) independently. The terminal UI is a fullscreen Textual app with clickable controls and flicker-free streaming.

Runs on **macOS** (all runtimes) and **Linux** (llama.cpp + whisper.cpp + any cloud role; MLX is Apple-Silicon-only and is filtered out of the catalog on Linux). On **Windows** run the app through [WSL2](https://learn.microsoft.com/windows/wsl/install).

## Features

- **Mix-and-match per-role providers**: pick STT, LLM, and TTS each from a cloud or local catalog in `models.toml`. Cloud LLMs / TTS can be OpenAI or Gemini. Local servers only start for the roles you actually select.
- **Runtime model switching**: press `S` (or click *Settings*) to open a modal picker; the active choice is saved to `preferences.toml`.
- **Per-model API key override**: use `api_key = "${VAR_NAME}"` on a cloud entry to point it at a different vendor key (e.g. a legacy Gemini key for the LLM while the newer key handles TTS).
- **Reasoning-effort knob**: `reasoning_effort = "minimal"` on a GPT-5 or Gemini-3 entry skips server-side thinking and drops TTFT from ~16s to ~1s for voice.
- **Fullscreen Textual TUI**: card-per-turn conversation scrolls above a persistent status footer; clickable Mute / Interrupt / Reset / Settings / Quit buttons plus keyboard shortcuts.
- **Per-card copy button**: a small `⧉ copy` control on every user and agent card pushes the message to the system clipboard (OSC-52).
- **Reset conversation**: press `R` (or click *Reset*) to clear history and start fresh without restarting the app.
- **Voice Activity Detection (VAD)**: Silero VAD (ONNX) provides continuous listening with pre-roll.
- **Interruption**: press Space (or click Interrupt) to cut the agent off mid-response.
- **Mute**: press M to release the mic entirely during a response.
- **Voice cloning** on local TTS models that support it (e.g. CSM): provide a reference WAV + transcript on a TTS entry and mlx-audio will clone that voice.
- **MCP tools**: connect MCP servers via `mcp_servers.toml` with per-server `enabled` toggle.
- **OpenAI-hosted tools**: cloud OpenAI LLM entries can enable `web_search`, `code_interpreter`, and `file_search` directly from `models.toml` (not supported on Gemini).
- **Shell tool (opt-in, with approval)**: the agent can propose shell commands that you approve/decline per invocation (or auto-approve if you trust the prompts).
- **Auto-setup for local roles**: installs Python deps, system packages via the detected package manager (brew/apt/dnf/pacman/zypper), builds whisper.cpp (with Metal on macOS, CUDA on Linux when an NVIDIA GPU is detected), installs llama.cpp (prebuilt on macOS / Linux CPU / Linux ARM; source build with CUDA on Linux+NVIDIA), and patches known compatibility issues.
- **OS-aware model catalog**: each local entry declares a `runtime` (`whispercpp` / `llamacpp` / `mlx-lm` / `mlx-vlm` / `mlx-audio`); entries whose runtime doesn't run on the current OS are filtered out of the Switch modal at startup. If `preferences.toml` points at a filtered entry, the app auto-falls back to the first compatible one and surfaces a notice.
- **Per-turn metrics**: STT, LLM (with TTFT), TTS, and total timing inline with each turn.

## Prerequisites

- **Operating system**:
  - **macOS** on Apple Silicon (M1/M2/M3/M4) — all local runtimes available.
  - **Linux** (x86_64 or aarch64) — `whispercpp` (STT) and `llamacpp` (LLM) for local roles; MLX-backed entries are filtered out since the `mlx` package has no Linux wheels. Local TTS is not supported on Linux today — pair a local STT/LLM with a cloud TTS (OpenAI or Gemini), or run everything cloud.
  - **Windows**: not supported natively — install [WSL2](https://learn.microsoft.com/windows/wsl/install) and run the app from a Linux shell.
- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** package manager — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **espeak-ng** (Kokoro TTS only; auto-installed at first use via brew/apt/dnf/pacman/zypper depending on your system).
- **NVIDIA GPU + driver** (optional, Linux only) — when `nvidia-smi` is present, `setup-whispercpp.sh` compiles whisper.cpp with `-DGGML_CUDA=ON`. `setup-llamacpp.sh` also builds llama.cpp from source with `-DGGML_CUDA=ON` against the host GPU (CMAKE_CUDA_ARCHITECTURES=native); this requires the **CUDA Toolkit** (nvcc) to be installed. Without nvcc, the script falls back to the CPU prebuilt.
- An **OpenAI** and/or **Gemini API key** for any cloud role.

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

Add the keys you need to `.env`:

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...       # only if any active model is a Gemini entry
```

Per-model override: any cloud entry can set `api_key = "${VAR_NAME}"` in `models.toml` to pull from a different env var, so you can have (say) two Gemini keys live side-by-side.

## Controls

All controls are also clickable buttons in the bottom footer. The Interrupt button is enabled only while the agent is responding. Input is Silero VAD — speak and the app picks it up automatically.

| Key       | Action                                                       |
|-----------|--------------------------------------------------------------|
| (speak)   | Speech is detected automatically (Silero VAD)                |
| Space     | Interrupt the agent while it's speaking                      |
| M         | Toggle microphone mute                                       |
| R         | Reset conversation (clear history + cards; fresh workflow)   |
| S         | Open the Settings modal (per-role model picker)              |
| Y / N     | Approve / decline a pending shell-tool proposal              |
| Q, Ctrl+C | Quit                                                         |

Each user and agent card has a `⧉ copy` control in the top-right that copies the message text to your clipboard (via OSC-52).

## Configuration

Config lives in three files at the project root:

| File                    | Purpose                                                                         | Checked in? |
|-------------------------|---------------------------------------------------------------------------------|-------------|
| `config.toml`           | Local server URLs, VAD, agent, display, audio, shell tool                       | ✅          |
| `models.toml`           | `[[stt]]` / `[[llm]]` / `[[tts]]` catalogs — the pickable models per role       | ✅          |
| `preferences.toml`      | Active model name per role; auto-written by the Settings modal                  | ❌          |
| `.env`                  | Secrets (`OPENAI_API_KEY`, `GEMINI_API_KEY`, any `api_key = "${VAR}"` targets)  | ❌          |
| `mcp_servers.toml`      | MCP server definitions                                                          | ❌          |
| `llamacpp-models.ini`   | llama-server preset (only when any LLM uses `runtime = "llamacpp"`)             | ❌          |

Environment variables override `.env`, which overrides `config.toml`.

### config.toml structure

```toml
[general]
enable_mcp = true                  # load MCP servers from mcp_servers.toml

[local]                            # only consulted for roles whose active model is local
# Each URL is served by whichever runtime is active for that role. The app
# filters runtimes by OS, so only the ones you need ever get consulted.
stt_url = "http://localhost:9000"  # whispercpp          (darwin + linux)
tts_url = "http://localhost:8000"  # mlx-audio           (darwin only)
llm_url = "http://localhost:8080"  # llamacpp / mlx-lm / mlx-vlm   (llamacpp: darwin+linux; mlx-*: darwin only)

[vad]
threshold  = 0.5                   # Silero speech probability threshold (0-1)
silence_ms = 500                   # silence before a segment is closed

[display]
show_transcript = true
show_metrics    = true             # inline STT / LLM / TTS / Total per turn

[audio]
sample_rate = 24000

[shell]                            # optional shell tool with per-call user approval
enabled      = false
auto_approve = false               # true = run silently, no prompt (dangerous)

[agent]
instructions     = """You are a helpful voice assistant. Today is {date}."""
tool_call_filler = "Give me just a moment."    # spoken while waiting for a tool call; optional
```

### models.toml structure

The per-role model lists live in `models.toml` and are combined at launch
with the `[local]` server URLs from `config.toml`. The `(local)` / `(cloud)`
suffix is added automatically from `provider` — don't include it in `name`.

Local entries must declare a `runtime`; allowed values per role:

| Role | Runtime       | OS support        |
|------|---------------|-------------------|
| stt  | `whispercpp`  | darwin, linux     |
| llm  | `llamacpp`    | darwin, linux     |
| llm  | `mlx-lm`      | darwin only       |
| llm  | `mlx-vlm`     | darwin only       |
| tts  | `mlx-audio`   | darwin only       |

Entries whose runtime doesn't run on the current OS are dropped from the
catalog at startup, and if `preferences.toml` points at a dropped entry
the app auto-falls back to the first compatible one.

```toml
# ── STT catalog ─────────────────────────────────────
# First entry is the default when preferences.toml is absent.
[[stt]]
name     = "whisper-turbo"
provider = "local"
runtime  = "whispercpp"
model    = "large-v3-turbo-q5_0"   # whisper.cpp model file suffix

[[stt]]
name     = "gpt-4o-transcribe"
provider = "cloud"
model    = "gpt-4o-transcribe"

# ── LLM catalog ─────────────────────────────────────
[[llm]]
name        = "gemma-4-e4b-it (llamacpp)"
provider    = "local"
runtime     = "llamacpp"                # "llamacpp" | "mlx-lm" | "mlx-vlm"
model       = "gemma-4-e4b-it"          # must match a section alias in the preset
preset      = "llamacpp-models.ini"
audio_input = true                      # pass mic audio straight to the LLM (local-STT + local-LLM only)

[[llm]]
name            = "gemma-4-e4b-it (mlx-vlm)"
provider        = "local"
runtime         = "mlx-vlm"
model           = "mlx-community/gemma-4-e4b-it-4bit"
audio_input     = true
kv_bits         = 3.5                   # mlx-vlm only
kv_quant_scheme = "turboquant"          # mlx-vlm only

[[llm]]
name             = "gpt-5.4-mini"
provider         = "cloud"
model            = "gpt-5.4-mini"
reasoning_effort = "minimal"            # "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
# Opt into OpenAI-hosted tools (OpenAI cloud LLMs only — not Gemini):
# hosted_tools = ["web_search", "code_interpreter"]
# With "file_search" you must also set:
# file_search_vector_stores = ["vs_abc123"]
# file_search_max_results   = 5

# Gemini LLM via its OpenAI-compatible chat endpoint.
[[llm]]
name             = "gemini-3.1-flash-lite-preview"
provider         = "cloud"
vendor           = "gemini"             # default is "openai"; set "gemini" to route to Google
model            = "gemini-3.1-flash-lite-preview"
reasoning_effort = "minimal"            # otherwise TTFT is ~16s on Gemini 3 preview
# Per-model key override. The OpenAI-compat endpoint rejects newer "AQ."-prefix
# keys, so point the LLM at a legacy AIza-prefix key:
api_key          = "${GEMINI_API_KEY_LEGACY}"

# ── TTS catalog ─────────────────────────────────────
# NOTE: No local TTS runtime is available on Linux today. Stick to cloud
# TTS entries on Linux, or pair a local STT/LLM with cloud TTS.
[[tts]]
name     = "kokoro"
provider = "local"
runtime  = "mlx-audio"
model    = "mlx-community/Kokoro-82M-bf16"
voice    = "af_heart"

# Voice cloning with CSM (local only, mlx-audio). Both fields required together.
# `ref_audio` is resolved relative to the project root.
[[tts]]
name      = "csm-my-voice"
provider  = "local"
runtime   = "mlx-audio"
model     = "mlx-community/csm-1b-8bit"
ref_audio = "voices/my-voice.wav"
ref_text  = "This is what my voice sounds like."

[[tts]]
name     = "gpt-4o-mini-tts"
provider = "cloud"
model    = "gpt-4o-mini-tts"
voice    = "alloy"

# Gemini TTS via native generateContent API (custom adapter in gemini_tts.py).
[[tts]]
name     = "gemini-3.1-flash-tts-preview"
provider = "cloud"
vendor   = "gemini"
model    = "gemini-3.1-flash-tts-preview"
voice    = "Sulafat"                    # see https://ai.google.dev/gemini-api/docs/speech-generation#voices
```

**Per-model `api_key` override.** Any cloud entry can set `api_key = "${VAR_NAME}"` to pull from a specific env var, overriding the vendor-wide `OPENAI_API_KEY` / `GEMINI_API_KEY`. Use this when one key is rate-limited or scoped to a subset of models. Literal values work but are checked in — prefer env-var references.

**`reasoning_effort` on reasoning-capable LLMs.** Gemini 3 preview flash and GPT-5 models do server-side thinking before emitting tokens; the default (typically `medium`) makes first-token latency unacceptable for voice. `"minimal"` drops it to ~1s. Hosted OpenAI tools require at least `"low"`, since the tool calls are themselves part of the reasoning process.

**Voice cloning on local TTS entries.** Set `ref_audio` + `ref_text` to clone a voice with models that support it (CSM is the current flagship; `mlx-community/csm-1b-8bit`). `ref_audio` is a path to a short reference WAV (resolved relative to the project root); `ref_text` is its verbatim transcript. Both must be set together, and only on local TTS entries — mlx-audio handles the cloning server-side. The config loader checks at startup that the file exists.

#### Runtime variables in `[agent].instructions`

| Variable      | Example                             |
|---------------|-------------------------------------|
| `{date}`      | April 15, 2026                      |
| `{time}`      | 2:30 PM                             |
| `{datetime}`  | April 15, 2026 2:30 PM              |
| `{os}`        | Darwin                              |
| `${VAR_NAME}` | Environment variable from `.env` or system |

### preferences.toml

Three lines, gitignored, written by the Settings modal (or you can create it manually):

```toml
[active]
stt = "whisper-turbo"
llm = "gpt-5.4-mini"
tts = "kokoro"
```

Each value matches the `name` field of an entry in `models.toml` (without the `(cloud)` / `(local)` suffix — that's added automatically for display). Missing or unknown names fall back to the first catalog entry and print a warning at startup. Names that *do* exist in the catalog but use a runtime that doesn't support the current OS (e.g. `kokoro` on Linux) fall back to the first compatible entry and mount an inline notice so the swap is visible. Copy `preferences.toml.example` to get started.

### .env

```
OPENAI_API_KEY=sk-...
```

Any `config.toml` scalar can be overridden via an env var — e.g. `STT_URL=http://...`, `VAD_THRESHOLD=0.6`.

### model_deps.toml

Maps model-name patterns to pip deps + per-package-manager system deps. Matching deps are auto-installed the first time that role is started; system packages use the OS-detected manager (brew on macOS; apt/dnf/pacman/zypper on Linux).

```toml
[kokoro]
deps = ["misaki", "num2words", "spacy", "phonemizer", "espeakng-loader"]

[kokoro.system]
brew   = ["espeak-ng"]
apt    = ["espeak-ng"]
dnf    = ["espeak-ng"]
pacman = ["espeak-ng"]
zypper = ["espeak-ng"]
```

## Runtime model switching

Press `S` (or click the **Settings** button) to open the modal. Pick one model per role and press *Apply*:

1. `preferences.toml` is rewritten immediately with the new selection (so a crash mid-swap doesn't desync it).
2. `ServerManager.reconcile()` stops any local server no longer needed, starts ones that are now needed, and restarts any whose active model changed.
3. The workflow + pipeline are rebuilt (fresh history — the previous conversation is not carried across a swap; use **Reset (R)** for the same effect without switching models).

If a server fails to start, the active selection is reverted, `preferences.toml` is rewritten with the previous selection, and an inline error card is mounted.

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

## OpenAI-hosted tools (OpenAI cloud LLMs only)

Cloud LLM entries with vendor `openai` (the default) can enable [hosted tools](https://openai.github.io/openai-agents-python/tools/):

```toml
[[llm]]
name             = "gpt-5.4-mini"
provider         = "cloud"
model            = "gpt-5.4-mini"
reasoning_effort = "low"             # hosted tools need ≥ "low"; "minimal"/"none" is rejected
hosted_tools     = ["web_search", "code_interpreter"]
```

Supported tools: `web_search`, `code_interpreter`, `file_search` (requires `file_search_vector_stores = [...]` and optional `file_search_max_results = N`). The app rejects `hosted_tools` on local LLMs and on non-OpenAI vendors (Gemini) at startup.

Hosted tools go through the OpenAI Responses API, which is why the OpenAI LLM path explicitly uses `OpenAIResponsesModel` (not `OpenAIChatCompletionsModel`).

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
                    #   action_open_settings, switch_models,
                    #   action_reset_conversation, …
  app.tcss          # Textual CSS (cards, footer, splash, settings modal)
  widgets.py        # UserTurn / AgentTurn / CopyButton / ToolCard / NoticeCard /
                    #   ErrorCard / ApprovalCard / StateRow / ModelRow / ToolsRow /
                    #   ControlRow / StatusFooter / ServerRow / SplashScreen /
                    #   SettingsScreen
  display.py        # TurnMetrics dataclass + TYPE_CHECKING `Display` alias
  audio.py          # VADRecorder (Silero ONNX), AudioPlayer
  pipeline.py       # Async _run_vad loop, _process_turn,
                    #   run_pipeline_loops entrypoint
  providers.py      # TranscriptVoiceWorkflow, WhisperCppSTTModel,
                    #   StreamingTTSModel, AudioPassthroughSTTModel,
                    #   create_agent / create_pipeline, _hosted_tools
  gemini_tts.py     # GeminiTTSModel (native generateContent, not OpenAI-compat)
  shell.py          # Shell tool + approval flow
  servers.py        # ServerManager.reconcile() per-role starter
  mcp.py            # load_mcp_servers() with ${ENV} expansion + `enabled` toggle
  preferences.py    # load/save preferences.toml
  config.py         # Settings + ModelConfig, _parse_catalog, _validate_active_requirements
```

### Key design decisions

- **Per-role providers, not a single mode.** Each `[[stt]] / [[llm]] / [[tts]]` entry marks itself `provider = "cloud" | "local"` (cloud entries can also set `vendor = "gemini"`). The `ServerManager` is a reconciler: it starts only the local server(s) needed by the currently active selection and restarts on a swap.
- **OpenAI-compatible local servers.** mlx-audio, mlx-vlm, mlx-lm, and llama-server all expose `/v1/...` endpoints, so we reuse the SDK's `OpenAITTSModel` / `OpenAIChatCompletionsModel` and point `AsyncOpenAI` at localhost. Local STT uses a custom `WhisperCppSTTModel` hitting whisper-server's `/inference`.
- **OpenAI LLMs use Responses, not Chat Completions.** To support hosted tools the OpenAI cloud LLM path constructs `OpenAIResponsesModel` with an explicit base URL + `Omit()` headers — this dodges env vars like `OPENAI_BASE_URL` / `OPENAI_ORG_ID` that would otherwise redirect traffic or trigger Gemini's "Multiple authentication credentials received" 400.
- **Gemini integration is split across two paths.** LLM goes through Gemini's OpenAI-compatible endpoint (we reuse `OpenAIChatCompletionsModel` with a different base URL). TTS does not have an OpenAI-compat counterpart, so `gemini_tts.py` wraps the native `generateContent` API and exposes it as a `TTSModel`.
- **Flicker-free streaming.** Each turn is its own widget; agent text is a `reactive` attribute that re-renders only that one widget per token. Rich `Live` is gone; Textual owns the screen.
- **Audio-passthrough** (`audio_input = true` on a local LLM) only engages when both STT and LLM are local. Otherwise the audio blob would be dropped.
- **Echo suppression by mic muting.** There is no AEC — we mute the microphone while the agent is speaking. Press Space to interrupt instead of speaking over it.

## Scripts

| Script                    | Purpose                                                             |
|---------------------------|---------------------------------------------------------------------|
| `./start.sh`              | Run the voice agent                                                 |
| `./setup.sh`              | Install all dependencies (core + local)                             |
| `./setup.sh --update`     | Update all dependencies                                             |
| `./setup-llamacpp.sh`     | Install/update `llama-server` (prebuilt, or source build w/ CUDA on Linux+NVIDIA) |
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

Each local role has exactly one backing process, started on demand, selected by the active model's `runtime`:

- **whisper-server** (STT, port 9000) — `runtime = "whispercpp"`. whisper.cpp HTTP server with built-in VAD. Runs on macOS and Linux. Restarted when you pick a different local STT model.
- **mlx-audio** (TTS, port 8000) — `runtime = "mlx-audio"`. macOS-only. Model-agnostic; the TTS model is specified per request, so swapping between local TTS models does **not** require a restart.
- **LLM server** (port 8080) — backend is per-entry:
  - `runtime = "llamacpp"` — `llama-server` binary, health via `/health`, models from the preset file. Runs on macOS and Linux.
  - `runtime = "mlx-lm"` — Python module, health via `/v1/models`. macOS-only.
  - `runtime = "mlx-vlm"` — Python module, supports `--kv-bits` / `--kv-quant-scheme`, health via `/v1/models`. macOS-only.

The reconciler restarts the LLM server when you pick a different local LLM (different model or runtime).

### VAD tuning

Adjust in `[vad]`:

- `threshold` — Silero speech-probability cutoff (0–1). Raise in noisy rooms.
- `silence_ms` — how long the trailing silence must be before a segment closes.

### Stale local selection

If you delete a local model or change its file layout, the next time that local server starts it'll fail and the selection reverts to your previous choice. Fix the config, then switch back via `S`.

### Missing dependencies

`model_deps.toml` maps patterns in the model name to pip deps + per-package-manager system deps. The first time you start a role that matches, the deps install automatically (pip via `uv pip install`; system packages via `brew`/`apt-get`/`dnf`/`pacman`/`zypper` depending on the host). Add new entries there as needed; for a new system dep, include one key per supported package manager under the `[<pattern>.system]` table.
