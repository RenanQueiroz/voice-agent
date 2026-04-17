"""Fullscreen Textual TUI for the voice agent.

`VoiceAgentApp` exposes the same method names as the old `Display` class so
that `pipeline.py`, `providers.py`, `audio.py`, and `servers.py` can call into
it unchanged. Internally those methods update reactive widgets and mount new
conversation cards.

The pipeline itself (MCP connect, server startup, VAD / push-to-talk loops)
runs as a Textual worker started in `on_mount` so the UI stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button

from .display import TurnMetrics
from .widgets import (
    AgentTurn,
    ErrorCard,
    ModelRow,
    NoticeCard,
    SplashScreen,
    StateRow,
    StatusFooter,
    ToolCard,
    ToolsRow,
    UserTurn,
)

if TYPE_CHECKING:
    from .config import Settings


logging.getLogger("openai.agents").setLevel(logging.CRITICAL)


class VoiceAgentApp(App[None]):
    """The main app. Hosts the conversation UI and drives the pipeline."""

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("space", "interrupt", "Interrupt", show=False),
        Binding("m", "toggle_mute", "Mute", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("k", "record_key", "Record", show=False),
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings

        # Pipeline-facing async primitives. The pipeline worker reads these;
        # key bindings and button clicks set them.
        self.quit_event = asyncio.Event()
        self.interrupt_event = asyncio.Event()
        self.record_key_queue: asyncio.Queue[None] = asyncio.Queue()

        # Shared UI state
        self.is_muted = False
        self.responding = False

        # Active cards being streamed into (updated by display methods)
        self._current_agent_turn: AgentTurn | None = None
        self._current_tool_card: ToolCard | None = None
        self._pending_user_turn: UserTurn | None = None
        self._last_metrics: TurnMetrics | None = None

        # Splash handle (set while setup is running)
        self._splash: SplashScreen | None = None

    # ── Compose & mount ───────────────────────────────────

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="conversation")
        yield StatusFooter()

    def on_mount(self) -> None:
        # Push the splash over the main UI. The pipeline worker pops it when
        # setup finishes.
        self._splash = SplashScreen()
        self.push_screen(self._splash)

        # Seed the footer with model labels
        self._update_models()

        # Kick off the pipeline in a worker (async, not a thread, so it shares
        # the event loop and can call our methods directly).
        self.run_worker(self._run_pipeline(), exclusive=True, name="pipeline")

    def _update_models(self) -> None:
        try:
            row = self.query_one(ModelRow)
        except Exception:
            return
        s = self.settings
        row.stt_label = s.stt_model
        row.llm_label = s.llm_model
        row.tts_label = s.tts_model

    # ── Pipeline worker ───────────────────────────────────

    async def _run_pipeline(self) -> None:
        """The whole backend lifecycle, in one place."""
        # Lazy imports to keep app startup fast and to break cycles.
        from .mcp import load_mcp_servers
        from .pipeline import run_pipeline_loops
        from .servers import ServerManager

        server_manager: ServerManager | None = None
        mcp_servers: list = []

        try:
            if self.settings.voice_mode == "local":
                server_manager = ServerManager(self.settings, self)
                ok = await server_manager.start()
                if not ok:
                    self.setup_failed()
                    await asyncio.sleep(3)
                    self.exit()
                    return

            mcp_servers = load_mcp_servers() if self.settings.enable_mcp else []
            for server in mcp_servers:
                try:
                    await server.connect()
                except Exception as e:
                    self.api_error(f"Failed to connect MCP server '{server.name}': {e}")
                    self.exit()
                    return

            if mcp_servers:
                tool_names: list[str] = []
                for server in mcp_servers:
                    tools = await server.list_tools()
                    tool_names.extend(t.name for t in tools)
                self.set_mcp_tools(tool_names)

            # Setup done — dismiss splash and show the main UI
            if self._splash is not None:
                self.pop_screen()
                self._splash = None

            self._set_state("listening")

            await run_pipeline_loops(self.settings, self, server_manager, mcp_servers)
        finally:
            for server in mcp_servers:
                try:
                    await server.cleanup()
                except Exception:
                    pass
            if server_manager is not None:
                server_manager.stop()
            self.exit()

    # ── Actions (keyboard + click both route here) ────────

    def action_interrupt(self) -> None:
        if self.responding:
            self.interrupt_event.set()

    def action_toggle_mute(self) -> None:
        self.is_muted = not self.is_muted
        # The pipeline watches `app.is_muted` to mute/unmute the recorder.
        if self.is_muted:
            self.muted_ui()
        else:
            self.unmuted_ui()

    async def action_quit(self) -> None:  # type: ignore[override]
        """Request a clean shutdown: the pipeline worker exits and calls self.exit()."""
        self.quit_event.set()
        # Unblock anything waiting on record_key_queue
        try:
            self.record_key_queue.put_nowait(None)
        except Exception:
            pass
        # Fallback — if the worker is wedged, hard-exit after a moment
        self.set_timer(2.0, self.exit)

    def action_record_key(self) -> None:
        """K toggles recording in push-to-talk mode."""
        try:
            self.record_key_queue.put_nowait(None)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-mute":
            self.action_toggle_mute()
        elif event.button.id == "btn-interrupt":
            self.action_interrupt()
        elif event.button.id == "btn-quit":
            self.run_worker(self.action_quit())

    # ── Conversation output ───────────────────────────────

    def _conversation(self) -> VerticalScroll:
        return self.query_one("#conversation", VerticalScroll)

    def _mount_card(self, widget) -> None:
        try:
            convo = self._conversation()
        except Exception:
            return
        convo.mount(widget)
        convo.scroll_end(animate=False)

    def user_said(self, text: str, stt_seconds: float = 0.0) -> None:
        # If we reserved a placeholder at processing() time (happens every
        # turn), fill it. Otherwise mount a fresh card. Reserving up-front
        # keeps the visual order correct when STT runs async alongside the
        # LLM (audio-passthrough mode).
        if self._pending_user_turn is not None:
            self._pending_user_turn.text = text
            self._pending_user_turn.stt_seconds = stt_seconds
            self._pending_user_turn = None
        else:
            self._mount_card(UserTurn(text, stt_seconds))

    def agent_start(self) -> None:
        card = AgentTurn()
        self._current_agent_turn = card
        self._mount_card(card)
        self._set_state("responding")
        self.responding = True
        self._set_interrupt_enabled(True)

    def agent_chunk(self, text: str) -> None:
        if self._current_agent_turn is not None:
            self._current_agent_turn.append(text)
            # Keep the tail of the response in view as it grows
            try:
                self._conversation().scroll_end(animate=False)
            except Exception:
                pass

    def agent_end(self) -> None:
        self._current_agent_turn = None

    def tool_call(self, name: str, args: str) -> None:
        card = ToolCard(name, args)
        self._current_tool_card = card
        self._mount_card(card)

    def tool_result(self, output: str) -> None:
        if self._current_tool_card is not None:
            self._current_tool_card.set_result(output)
            self._current_tool_card = None

    def interrupted(self) -> None:
        if self._current_agent_turn is not None:
            self._current_agent_turn.is_interrupted = True
            self._current_agent_turn.append(" [interrupted]")
            self._current_agent_turn = None
        self.responding = False
        self._set_interrupt_enabled(False)

    # ── Metrics ───────────────────────────────────────────

    def metrics(self, m: TurnMetrics) -> None:
        self._last_metrics = m
        # Attach metrics to the most recently mounted AgentTurn if still present
        try:
            convo = self._conversation()
            turns = list(convo.query(AgentTurn))
        except Exception:
            return
        if turns:
            turns[-1].set_metrics(m)

    # ── State transitions ─────────────────────────────────

    def _set_state(self, state: str) -> None:
        try:
            row = self.query_one(StateRow)
        except Exception:
            return
        row.state = state

    def _set_interrupt_enabled(self, enabled: bool) -> None:
        try:
            btn = self.query_one("#btn-interrupt", Button)
            btn.disabled = not enabled
        except Exception:
            pass

    def listening(self) -> None:
        self._set_state("listening")
        self.responding = False
        self._set_interrupt_enabled(False)

    def muted_ui(self) -> None:
        self._set_state("muted")
        try:
            row = self.query_one(StateRow)
            row.is_muted = True
        except Exception:
            pass
        try:
            self.query_one("#btn-mute", Button).label = "Unmute (M)"
        except Exception:
            pass

    def unmuted_ui(self) -> None:
        self._set_state("listening" if not self.responding else "responding")
        try:
            row = self.query_one(StateRow)
            row.is_muted = False
        except Exception:
            pass
        try:
            self.query_one("#btn-mute", Button).label = "Mute  (M)"
        except Exception:
            pass

    # Kept for compatibility with existing call sites in pipeline.py
    def muted(self) -> None:
        self.is_muted = True
        self.muted_ui()

    def unmuted(self) -> None:
        self.is_muted = False
        self.unmuted_ui()

    def recording_start(self) -> None:
        self._mount_card(NoticeCard("● Recording… (press K to stop)"))
        self._set_state("speaking")

    def recording_too_short(self) -> None:
        self._mount_card(NoticeCard("Too short, skipping."))
        self._set_state("listening")

    def ready_for_key(self) -> None:
        self._set_state("idle")

    def vad_speaking(self, rms: int) -> None:
        try:
            row = self.query_one(StateRow)
        except Exception:
            return
        row.vad_rms = rms
        row.state = "speaking"

    def vad_silence(self, remaining_ms: int) -> None:
        try:
            row = self.query_one(StateRow)
        except Exception:
            return
        row.vad_remaining_ms = remaining_ms
        row.state = "silence"

    def vad_clear(self) -> None:
        try:
            row = self.query_one(StateRow)
        except Exception:
            return
        row.vad_rms = 0
        row.vad_remaining_ms = 0

    def processing(self, duration: float) -> None:
        try:
            row = self.query_one(StateRow)
            row.processing_duration = duration
        except Exception:
            pass
        self._set_state("processing")
        # Reserve a slot for the user's transcription so that agent_start()
        # — which may fire before the async STT finishes — mounts its card
        # *below* the user turn, not above it.
        if self._pending_user_turn is None:
            placeholder = UserTurn()
            self._pending_user_turn = placeholder
            self._mount_card(placeholder)

    def set_mcp_tools(self, tool_names: list[str]) -> None:
        try:
            row = self.query_one(ToolsRow)
            row.tools = tuple(tool_names)
        except Exception:
            pass

    def ready_banner(self, settings: Settings) -> None:
        # The footer is already mounted — just update models and switch state.
        self.settings = settings
        self._update_models()
        self._set_state("listening")

    # ── Turn lifecycle (no-ops kept for API compat) ───────

    def turn_started(self) -> None:
        pass

    def turn_ended(self) -> None:
        pass

    def session_ended(self) -> None:
        pass

    def start_footer(self, settings: Settings) -> None:
        self.settings = settings
        self._update_models()

    def stop_footer(self) -> None:
        pass

    def goodbye(self) -> None:
        self.exit()

    # ── Server setup (drives splash) ──────────────────────

    def _splash_log(self, text: str) -> None:
        if self._splash is not None:
            self._splash.log_line(text)

    def server_setup_start(self) -> None:
        self._splash_log("Setting up local servers…")

    def server_installing_system(self, packages: list[str]) -> None:
        self._splash_log(f"Installing system packages: {', '.join(packages)}…")

    def server_installing(self, packages: list[str]) -> None:
        self._splash_log(f"Installing: {', '.join(packages)}…")

    def server_installed(self) -> None:
        self._splash_log("✓ Packages installed.")

    def server_install_failed(self, lines: list[str]) -> None:
        self._splash_log("✗ Install failed:")
        for line in lines:
            self._splash_log(f"  {line}")

    def server_patched(self, description: str) -> None:
        self._splash_log(f"✓ {description}")

    def server_starting(self, name: str) -> None:
        # Seed the row as "waiting" with 0s so it shows up immediately;
        # server_waiting() will update the elapsed count as time passes.
        if self._splash is not None:
            self._splash.set_waiting(name, 0)
        self._splash_log(f"Starting {name}…")

    def server_waiting(self, name: str, elapsed: int) -> None:
        if self._splash is not None:
            self._splash.set_waiting(name, elapsed)

    def server_ready_one(self, name: str) -> None:
        if self._splash is not None:
            self._splash.set_ready(name)

    def server_all_ready(self) -> None:
        self._splash_log("All servers ready.")

    def server_failed(self, name: str, log_lines: list[str]) -> None:
        if self._splash is not None:
            self._splash.set_failed(name)
        self._splash_log(f"✗ {name} exited unexpectedly:")
        for line in log_lines:
            self._splash_log(f"  {line}")

    def server_timeout(self, name: str, timeout: int) -> None:
        if self._splash is not None:
            self._splash.set_failed(name)
        self._splash_log(f"✗ {name} did not become ready within {timeout}s")

    def setup_failed(self) -> None:
        self._splash_log("Failed to start local servers. Exiting.")

    # ── Errors (mount as cards in the conversation) ───────

    def _mount_error(self, title: str, body: str) -> None:
        self._mount_card(ErrorCard(title, body))

    def api_error(self, message: str) -> None:
        self._mount_error("API Error", message)

    def api_error_with_logs(
        self, message: str, server_logs: dict[str, list[str]]
    ) -> None:
        parts = [message]
        for name, lines in server_logs.items():
            parts.append(f"\n{name} logs:")
            for line in lines:
                parts.append(f"  {line}")
        self._mount_error("API Error", "\n".join(parts))

    def connection_error(self, settings: Settings) -> None:
        if settings.voice_mode == "local":
            msg = (
                "Could not reach the local servers.\n"
                f"  STT/TTS: {settings.mlx_audio_url}\n"
                f"  LLM:     {settings.mlx_llm_url}"
            )
        else:
            msg = "Could not reach the OpenAI API. Check your connection and OPENAI_API_KEY."
        self._mount_error("Connection Error", msg)

    def auth_error(self, message: str) -> None:
        self._mount_error("Auth Error", f"{message}\n\nCheck OPENAI_API_KEY in .env")

    def rate_limit_error(self, message: str) -> None:
        self._mount_error("Rate Limit", message)

    def tts_stream_error(self) -> None:
        self._mount_error("TTS Error", "TTS server closed the connection mid-stream.")
