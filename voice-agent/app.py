"""Fullscreen Textual TUI for the voice agent.

`VoiceAgentApp` exposes the same method names as the old `Display` class so
that `pipeline.py`, `providers.py`, `audio.py`, and `servers.py` can call into
it unchanged. Internally those methods update reactive widgets and mount new
conversation cards.

The pipeline itself (MCP connect, server startup, VAD loop) runs as a
Textual worker started in `on_mount` so the UI stays responsive.
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
    ApprovalCard,
    ErrorCard,
    ModelRow,
    NoticeCard,
    SettingsScreen,
    SplashScreen,
    StateRow,
    StatusFooter,
    ToolCard,
    ToolsRow,
    UserTurn,
)

if TYPE_CHECKING:
    from agents.voice import VoicePipeline

    from .config import Settings
    from .providers import TranscriptVoiceWorkflow
    from .servers import ServerManager


logging.getLogger("openai.agents").setLevel(logging.CRITICAL)


class VoiceAgentApp(App[None]):
    """The main app. Hosts the conversation UI and drives the pipeline."""

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("space", "interrupt", "Interrupt", show=False),
        Binding("m", "toggle_mute", "Mute", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("s", "open_settings", "Settings", show=False),
        Binding("r", "reset_conversation", "Reset", show=False),
        Binding("y", "approve", "Approve", show=False),
        Binding("n", "decline", "Decline", show=False),
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings

        # Pipeline-facing async primitives. The pipeline worker reads these;
        # key bindings and button clicks set them.
        self.quit_event = asyncio.Event()
        self.interrupt_event = asyncio.Event()

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

        # Pipeline state — rebuilt by switch_models; read fresh per turn by
        # the loop functions in pipeline.py so swaps take effect cleanly.
        self.workflow: TranscriptVoiceWorkflow | None = None
        self.pipeline: VoicePipeline | None = None
        self.server_manager: ServerManager | None = None
        self.mcp_servers: list = []

        # Serialize switches and keep the pipeline from processing a turn
        # while we're reconciling servers / rebuilding the workflow.
        self._switch_lock = asyncio.Lock()

        # Pending shell-approval card + future. Only one at a time — the
        # agent's tool call is blocked awaiting this future.
        self._pending_approval: asyncio.Future[bool] | None = None
        self._pending_approval_card: ApprovalCard | None = None

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
        row.stt_label = s.stt.name
        row.llm_label = s.llm.name
        row.tts_label = s.tts.name

    # ── Pipeline worker ───────────────────────────────────

    async def _run_pipeline(self) -> None:
        """The whole backend lifecycle, in one place."""
        # Lazy imports to keep app startup fast and to break cycles.
        from .mcp import load_mcp_servers
        from .pipeline import run_pipeline_loops
        from .providers import create_pipeline
        from .servers import ServerManager

        self.server_manager = ServerManager(self.settings, self)
        self.mcp_servers = []

        try:
            ok = await self.server_manager.reconcile()
            if not ok:
                self.setup_failed()
                await asyncio.sleep(3)
                self.exit()
                return

            self.mcp_servers = load_mcp_servers() if self.settings.enable_mcp else []
            for server in self.mcp_servers:
                try:
                    await server.connect()
                except Exception as e:
                    self.api_error(f"Failed to connect MCP server '{server.name}': {e}")
                    self.exit()
                    return

            if self.mcp_servers:
                tool_names: list[str] = []
                for server in self.mcp_servers:
                    tools = await server.list_tools()
                    tool_names.extend(t.name for t in tools)
                self.set_mcp_tools(tool_names)

            # Build the initial workflow + pipeline
            self.workflow, self.pipeline = create_pipeline(
                self.settings, self, mcp_servers=self.mcp_servers
            )

            # Setup done — dismiss splash and show the main UI
            if self._splash is not None:
                self.pop_screen()
                self._splash = None

            # If any active model got auto-swapped at load time (e.g. a
            # Linux user's preferences.toml pointed at an mlx-audio TTS),
            # surface a NoticeCard per swap so the change isn't silent.
            for note in self.settings.fallback_notes:
                self._mount_card(NoticeCard(note))

            self._set_state("listening")

            await run_pipeline_loops(
                self.settings, self, self.server_manager, self.mcp_servers
            )
        finally:
            for server in self.mcp_servers:
                try:
                    await server.cleanup()
                except Exception:
                    pass
            if self.server_manager is not None:
                self.server_manager.stop()
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
        # Fallback — if the worker is wedged, hard-exit after a moment
        self.set_timer(2.0, self.exit)

    def action_approve(self) -> None:
        """Y: approve the pending shell command, if any."""
        card = self._pending_approval_card
        if card is not None:
            card._resolve(True)

    def action_decline(self) -> None:
        """N: decline the pending shell command, if any."""
        card = self._pending_approval_card
        if card is not None:
            card._resolve(False)

    async def request_shell_approval(self, command: str) -> bool:
        """Block until the user approves or declines the pending shell command.

        The `ApprovalCard` is normally mounted by `tool_call()` as soon as
        the SDK emits the `tool_called` event — that way the card always
        lands right after the ToolCard in the conversation, no matter how
        the SDK interleaves tool invocation with the event stream. We just
        wait here for the future that card stored on the app.

        If no `tool_call` event was seen (direct invocation in tests, or a
        non-streamed code path), mount a standalone card as a fallback.
        With `[shell].auto_approve`, we return True immediately without
        any UI.
        """
        if self.settings.shell.auto_approve:
            return True

        # Give the tool_call event handler a moment to set up the card.
        for _ in range(50):  # up to ~500ms
            if self._pending_approval is not None:
                break
            await asyncio.sleep(0.01)

        fut = self._pending_approval
        if fut is None:
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            card = ApprovalCard("Shell command", command)
            self._pending_approval = fut
            self._pending_approval_card = card
            self._mount_card(card)
        try:
            return await fut
        finally:
            self._pending_approval = None
            self._pending_approval_card = None

    def on_approval_card_decision(self, event: ApprovalCard.Decision) -> None:
        fut = self._pending_approval
        if fut is not None and not fut.done():
            fut.set_result(event.approved)

    def action_open_settings(self) -> None:
        """Open the model-switch modal. Blocked while responding."""
        if self.responding:
            self._mount_card(
                NoticeCard("Finish the current response before switching models.")
            )
            return
        if self._switch_lock.locked():
            return

        def _apply(result: tuple[str, str, str] | None) -> None:
            if result is None:
                return
            stt, llm, tts = result
            # No-op if nothing actually changed
            if (
                stt == self.settings.active_stt
                and llm == self.settings.active_llm
                and tts == self.settings.active_tts
            ):
                return
            self.run_worker(
                self.switch_models(stt, llm, tts),
                exclusive=False,
                name="switch",
            )

        self.push_screen(SettingsScreen(self.settings), _apply)

    async def switch_models(self, stt: str, llm: str, tts: str) -> None:
        """Apply a new set of active models: reconcile servers, rebuild pipeline."""
        from .providers import create_pipeline

        from .preferences import save_preferences

        async with self._switch_lock:
            if self.server_manager is None:
                return
            prev_stt = self.settings.active_stt
            prev_llm = self.settings.active_llm
            prev_tts = self.settings.active_tts

            self.settings.active_stt = stt
            self.settings.active_llm = llm
            self.settings.active_tts = tts

            # Persist the user's choice right away so preferences.toml always
            # reflects the latest selection, even if the local server takes a
            # long time to come up (or the user kills the app mid-startup).
            # If reconcile fails below, we revert and re-save the previous
            # selection so the file stays in sync with what's actually active.
            try:
                save_preferences(stt, llm, tts)
            except Exception as e:
                self._mount_error(
                    "Preferences not saved",
                    f"Could not write preferences.toml: {e}",
                )

            self._update_models()  # reflect the new names in the footer immediately
            self._set_state("processing")
            notice = NoticeCard("Switching models…")
            self._mount_card(notice)

            def _remove_notice() -> None:
                try:
                    notice.remove()
                except Exception:
                    pass

            def _rollback(reason_title: str, reason_body: str) -> None:
                self.settings.active_stt = prev_stt
                self.settings.active_llm = prev_llm
                self.settings.active_tts = prev_tts
                try:
                    save_preferences(prev_stt, prev_llm, prev_tts)
                except Exception:
                    pass
                _remove_notice()
                self._mount_error(reason_title, reason_body)
                self._update_models()
                self._set_state("listening")

            ok = await self.server_manager.reconcile()
            if not ok:
                _rollback(
                    "Switch failed",
                    "Could not start the required local server. Reverted to previous selection.",
                )
                return

            try:
                self.workflow, self.pipeline = create_pipeline(
                    self.settings, self, mcp_servers=self.mcp_servers
                )
            except Exception as e:
                _rollback("Switch failed", str(e))
                return

            _remove_notice()
            self._set_state("listening")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-mute":
            self.action_toggle_mute()
        elif event.button.id == "btn-interrupt":
            self.action_interrupt()
        elif event.button.id == "btn-reset":
            self.action_reset_conversation()
        elif event.button.id == "btn-settings":
            self.action_open_settings()
        elif event.button.id == "btn-quit":
            self.run_worker(self.action_quit())

    def action_reset_conversation(self) -> None:
        """Clear the chat history and all conversation cards."""
        if self.responding:
            self._mount_card(NoticeCard("Can't reset while the agent is responding."))
            return
        self.run_worker(self._reset_conversation(), exclusive=False, name="reset")

    async def _reset_conversation(self) -> None:
        """Rebuild the pipeline (fresh workflow ⇒ empty history) and clear cards.

        We rebuild the pipeline rather than poking at `workflow._input_history`
        directly so a future change to the workflow's internals doesn't leave
        the reset subtly broken. The server reconcile is skipped — the active
        models haven't changed — so this is cheap.
        """
        from .providers import create_pipeline

        async with self._switch_lock:
            try:
                self.workflow, self.pipeline = create_pipeline(
                    self.settings, self, mcp_servers=self.mcp_servers
                )
            except Exception as e:
                self._mount_error("Reset failed", str(e))
                return
            try:
                convo = self._conversation()
                for child in list(convo.children):
                    child.remove()
            except Exception:
                pass
            self._current_agent_turn = None
            self._current_tool_card = None
            self._pending_user_turn = None
            self._last_metrics = None
            self._mount_card(NoticeCard("Conversation reset."))
            self._set_state("listening")

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
        stt_name = self.settings.stt.display_name
        if self._pending_user_turn is not None:
            self._pending_user_turn.text = text
            self._pending_user_turn.stt_seconds = stt_seconds
            self._pending_user_turn.stt_name = stt_name
            self._pending_user_turn = None
        else:
            self._mount_card(UserTurn(text, stt_seconds, stt_name=stt_name))

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
        # If this is a shell command, mount the approval card right after
        # the ToolCard so the prompt visually follows the call. The shell
        # tool function awaits the pre-created future via
        # `request_shell_approval` — this guarantees the UI order is
        # ToolCard → ApprovalCard regardless of how the SDK interleaves
        # tool invocation with the event stream.
        #
        # If the user opted into `[shell].auto_approve`, skip the card and
        # pre-resolve the future so the tool runs immediately.
        if name == "run_shell_command":
            import json as _json

            command = args
            try:
                parsed = _json.loads(args) if isinstance(args, str) else args
                if isinstance(parsed, dict) and "command" in parsed:
                    command = str(parsed["command"])
            except Exception:
                pass
            loop = asyncio.get_event_loop()
            fut: asyncio.Future[bool] = loop.create_future()
            self._pending_approval = fut
            if self.settings.shell.auto_approve:
                fut.set_result(True)
                self._pending_approval_card = None
            else:
                approval = ApprovalCard("Shell command", command)
                self._mount_card(approval)
                self._pending_approval_card = approval

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
            turns[-1].set_metrics(
                m,
                llm_name=self.settings.llm.display_name,
                tts_name=self.settings.tts.display_name,
            )

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
            btn.display = enabled
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
            placeholder = UserTurn(stt_name=self.settings.stt.display_name)
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
        # Interrupt is only relevant while the agent is responding.
        self._set_interrupt_enabled(False)

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
        lines: list[str] = []
        if settings.stt.provider == "local":
            lines.append(f"  STT: {settings.stt_url}")
        if settings.tts.provider == "local":
            lines.append(f"  TTS: {settings.tts_url}")
        if settings.llm.provider == "local":
            lines.append(f"  LLM: {settings.llm_url}")
        if any(
            m.provider == "cloud" for m in (settings.stt, settings.llm, settings.tts)
        ):
            lines.append("  Cloud: https://api.openai.com")
        msg = "Could not reach a required endpoint.\n" + "\n".join(lines)
        self._mount_error("Connection Error", msg)

    def auth_error(self, message: str) -> None:
        self._mount_error("Auth Error", f"{message}\n\nCheck OPENAI_API_KEY in .env")

    def rate_limit_error(self, message: str) -> None:
        self._mount_error("Rate Limit", message)

    def tts_stream_error(self) -> None:
        self._mount_error("TTS Error", "TTS server closed the connection mid-stream.")
