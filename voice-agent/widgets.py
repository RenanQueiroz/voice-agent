"""Textual widgets for the voice-agent TUI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Label, Select, Static

from .display import TurnMetrics

if TYPE_CHECKING:
    from .config import Settings


class CopyButton(Static):
    """Tiny clickable 'copy' icon docked in a turn card's header.

    Reads the text on every click via the callable, so a `AgentTurn` whose
    text is still streaming will copy whatever's present at click-time."""

    def __init__(self, get_text: Callable[[], str]) -> None:
        super().__init__("⧉ copy", classes="card-copy")
        self._get_text = get_text
        self._copied = False

    def on_click(self) -> None:
        text = (self._get_text() or "").strip()
        if not text:
            return
        self.app.copy_to_clipboard(text)
        self._copied = True
        self.update("✓ copied")
        self.set_timer(1.5, self._reset_label)

    def _reset_label(self) -> None:
        if not self._copied:
            return
        self._copied = False
        self.update("⧉ copy")


# ── Turn cards ───────────────────────────────────────────


class UserTurn(Widget):
    """A card showing the user's transcribed utterance.

    Reactive so the app can mount an empty placeholder at turn start and fill
    it in later when the (possibly background) STT finishes. Reserving the
    slot up front keeps user/agent turns in the right visual order even when
    whisper runs async alongside the LLM.
    """

    text: reactive[str] = reactive("", layout=True)
    stt_seconds: reactive[float] = reactive(0.0, layout=True)
    stt_name: reactive[str] = reactive("")

    def __init__(
        self, text: str = "", stt_seconds: float = 0.0, stt_name: str | None = None
    ) -> None:
        super().__init__()
        self._body_ready = False
        self.set_reactive(UserTurn.text, text)
        self.set_reactive(UserTurn.stt_seconds, stt_seconds)
        self.set_reactive(UserTurn.stt_name, stt_name or "")

    def compose(self) -> ComposeResult:
        with Horizontal(classes="card-header"):
            yield Static("You", classes="label")
            yield CopyButton(lambda: self.text)
        yield Static("", id="body")
        yield Static("", classes="stt")

    def on_mount(self) -> None:
        self._body_ready = True
        self.watch_text(self.text)
        self._refresh_stt()

    def watch_text(self, text: str) -> None:
        if not self._body_ready:
            return
        try:
            body = self.query_one("#body", Static)
        except Exception:
            return
        body.update(Text(text) if text else Text("…", style="dim"))

    def watch_stt_seconds(self, _seconds: float) -> None:
        self._refresh_stt()

    def watch_stt_name(self, _name: str) -> None:
        self._refresh_stt()

    def _refresh_stt(self) -> None:
        if not self._body_ready:
            return
        try:
            stt = self.query_one(".stt", Static)
        except Exception:
            return
        if self.stt_seconds > 0:
            label = f"STT [{self.stt_name}]" if self.stt_name else "STT"
            stt.update(f"{label} {self.stt_seconds:.1f}s")
            stt.display = True
        else:
            stt.update("")
            stt.display = False


class AgentTurn(Widget):
    """A card showing the agent's streaming response."""

    text: reactive[str] = reactive("", layout=True)
    metrics_line: reactive[str] = reactive("", layout=True)
    is_interrupted: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._body_ready = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="card-header"):
            yield Static("Agent", classes="label")
            yield CopyButton(lambda: self.text)
        yield Static("", id="body")
        yield Static("", classes="metrics")

    def on_mount(self) -> None:
        self._body_ready = True
        # Metrics row stays hidden until there's something to show; otherwise
        # it reserves a blank line below the agent text (noticeable when
        # show_metrics is off).
        try:
            self.query_one(".metrics", Static).display = bool(self.metrics_line)
        except Exception:
            pass
        # Flush any text / metrics that arrived before compose finished.
        if self.text:
            self.watch_text(self.text)
        if self.metrics_line:
            self.watch_metrics_line(self.metrics_line)

    def watch_text(self, text: str) -> None:
        if not self._body_ready:
            return
        try:
            body = self.query_one("#body", Static)
        except Exception:
            return
        body.update(Text(text) if text else Text(""))

    def watch_metrics_line(self, line: str) -> None:
        if not self._body_ready:
            return
        try:
            metrics = self.query_one(".metrics", Static)
        except Exception:
            return
        metrics.update(line)
        metrics.display = bool(line)

    def watch_is_interrupted(self, value: bool) -> None:
        self.set_class(value, "-interrupted")

    def append(self, chunk: str) -> None:
        self.text = self.text + chunk

    def set_metrics(
        self,
        m: TurnMetrics,
        llm_name: str | None = None,
        tts_name: str | None = None,
    ) -> None:
        def _label(role: str, name: str | None) -> str:
            return f"{role} [{name}]" if name else role

        parts: list[str] = []
        if m.llm_seconds > 0:
            llm_part = f"{_label('LLM', llm_name)} {m.llm_seconds:.1f}s"
            if m.llm_first_token_seconds > 0:
                llm_part += f" (TTFT {m.llm_first_token_seconds:.1f}s)"
            parts.append(llm_part)
        if m.tts_seconds > 0:
            parts.append(f"{_label('TTS', tts_name)} {m.tts_seconds:.1f}s")
        if m.total_seconds > 0:
            parts.append(f"Total {m.total_seconds:.1f}s")
        self.metrics_line = " · ".join(parts)


class ToolCard(Widget):
    """A card showing a tool invocation and its result."""

    def __init__(self, name: str, args: str) -> None:
        super().__init__()
        self._name = name
        self._args = args
        self._result: str | None = None

    def compose(self) -> ComposeResult:
        args_short = self._args[:100] + "…" if len(self._args) > 100 else self._args
        yield Static(f"Tool · {escape(self._name)}", classes="label")
        yield Static(escape(args_short) if args_short else "", id="args")
        yield Static("", id="result")

    def set_result(self, output: str) -> None:
        self._result = output
        short = output[:200] + "…" if len(output) > 200 else output
        try:
            self.query_one("#result", Static).update(f"→ {escape(short)}")
        except Exception:
            pass


class ErrorCard(Widget):
    """A card showing an error in-line with the conversation."""

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="label")
        yield Static(escape(self._body))


class NoticeCard(Widget):
    """Small inline notice (e.g., 'Recording…', 'Press K to speak')."""

    DEFAULT_CSS = """
    NoticeCard { height: auto; width: 100%; padding: 0 2; margin: 0 0 1 0; color: $text-muted; }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(escape(self._text))


class ApprovalCard(Widget):
    """Blocks a tool call until the user approves or declines.

    The card posts a `Decision` message (Approve / Decline) when the user
    clicks a button, presses Y/N, or the card is dismissed by `resolve()`.
    """

    class Decision(Message):
        def __init__(self, card: "ApprovalCard", approved: bool) -> None:
            super().__init__()
            self.card = card
            self.approved = approved

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._done = False

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="label")
        yield Static(escape(self._body), classes="body")
        with Horizontal(classes="approval-buttons"):
            yield Button("Approve (Y)", id="approve", variant="success")
            yield Button("Decline (N)", id="decline", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._done:
            return
        if event.button.id == "approve":
            self._resolve(True)
        elif event.button.id == "decline":
            self._resolve(False)

    def _resolve(self, approved: bool) -> None:
        if self._done:
            return
        self._done = True
        # Replace the buttons with an inline status so the card stays as a
        # record of the decision.
        try:
            for b in self.query(Button):
                b.remove()
            self.mount(
                Static(
                    "✓ Approved" if approved else "✗ Declined",
                    classes="decision",
                )
            )
        except Exception:
            pass
        self.post_message(self.Decision(self, approved))


# ── Footer ───────────────────────────────────────────────


_STATE_GLYPHS: dict[str, tuple[str, str]] = {
    "idle": ("○", "dim"),
    "listening": ("●", "green"),
    "speaking": ("●", "bold green"),
    "responding": ("●", "magenta"),
    "silence": ("●", "yellow"),
    "processing": ("●", "yellow"),
    "muted": ("●", "red"),
}


class StateRow(Widget):
    """First row of the footer: state icon + label + VAD meter / countdown."""

    state: reactive[str] = reactive("idle")
    vad_rms: reactive[int] = reactive(0)
    vad_remaining_ms: reactive[int] = reactive(0)
    processing_duration: reactive[float] = reactive(0.0)
    is_muted: reactive[bool] = reactive(False)

    def render(self) -> Text:
        icon, style = _STATE_GLYPHS.get(self.state, ("○", "dim"))
        t = Text()
        t.append(f"{icon} ", style=style)
        if self.state == "speaking":
            bar_len = min(max(self.vad_rms, 0) // 5, 20)
            t.append("Speaking  ", style=style)
            t.append("█" * bar_len, style="green")
            t.append("░" * (20 - bar_len), style="green dim")
        elif self.state == "silence":
            t.append(f"Silence  {self.vad_remaining_ms}ms", style=style)
        elif self.state == "processing" and self.processing_duration > 0:
            t.append(
                f"Processing {self.processing_duration:.1f}s of audio…", style=style
            )
        else:
            t.append(self.state.capitalize(), style=style)
        if self.is_muted and self.state != "muted":
            t.append("  (muted)", style="red dim")
        return t


class ModelRow(Widget):
    """Second row of the footer: STT / LLM / TTS model names."""

    stt_label: reactive[str] = reactive("")
    llm_label: reactive[str] = reactive("")
    tts_label: reactive[str] = reactive("")

    def render(self) -> Text:
        t = Text()

        def short(model: str) -> str:
            return model.split("/")[-1] if "/" in model else model

        if self.stt_label:
            t.append("STT ", style="dim")
            t.append(short(self.stt_label), style="cyan")
            t.append("   ", style="dim")
        if self.llm_label:
            t.append("LLM ", style="dim")
            t.append(short(self.llm_label), style="cyan")
            t.append("   ", style="dim")
        if self.tts_label:
            t.append("TTS ", style="dim")
            t.append(short(self.tts_label), style="cyan")
        return t


class ToolsRow(Widget):
    """Third footer row: MCP tool list (hidden when there are no tools)."""

    tools: reactive[tuple[str, ...]] = reactive(tuple())

    def watch_tools(self, tools: tuple[str, ...]) -> None:
        self.display = bool(tools)

    def on_mount(self) -> None:
        self.display = bool(self.tools)

    def render(self) -> Text:
        t = Text()
        t.append("Tools ", style="dim")
        t.append(" ".join(self.tools), style="yellow")
        return t


class ControlRow(Horizontal):
    """Footer control row: clickable Mute / Interrupt / Reset / Settings / Quit."""

    def compose(self) -> ComposeResult:
        yield Button("Mute (M)", id="btn-mute", variant="default")
        yield Button(
            "Interrupt (Space)", id="btn-interrupt", variant="warning", disabled=True
        )
        yield Button("Reset (R)", id="btn-reset", variant="default")
        yield Button("Settings (S)", id="btn-settings", variant="default")
        yield Button("Quit (Q)", id="btn-quit", variant="error")


class StatusFooter(Vertical):
    """Persistent footer: state, models, tools, controls."""

    def compose(self) -> ComposeResult:
        yield StateRow()
        yield ModelRow()
        yield ToolsRow()
        yield ControlRow()


# ── Splash screen ────────────────────────────────────────


class ServerRow(Widget):
    """One row on the splash screen, per server. When a log_path is
    attached, the whole row becomes clickable and pushes a
    `ServerLogScreen` over the splash so the user can watch the long-
    running startup (e.g. Qwen3 warmup / HF model download)."""

    status: reactive[str] = reactive("pending")
    elapsed: reactive[int] = reactive(0)

    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, name: str, log_path: Path | None = None) -> None:
        super().__init__()
        self._name = name
        self._log_path = log_path
        self._spin_idx = 0

    def set_log_path(self, log_path: Path | None) -> None:
        self._log_path = log_path
        self.refresh()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        if self.status == "waiting":
            self._spin_idx = (self._spin_idx + 1) % len(self._SPINNER)
            self.refresh()

    def render(self) -> Text:
        t = Text()
        if self.status == "ready":
            t.append("  ✓  ", style="bold green")
        elif self.status == "failed":
            t.append("  ✗  ", style="bold red")
        elif self.status == "waiting":
            t.append(f"  {self._SPINNER[self._spin_idx]}  ", style="cyan")
        else:
            t.append("  ·  ", style="dim")
        t.append(self._name, style="bold")
        if self.status == "waiting" and self.elapsed > 0:
            t.append(f"   {self.elapsed}s", style="dim")
        elif self.status == "ready":
            t.append("   ready", style="green dim")
        # Only hint at clickability while the server is actually running
        # and writing to its log; after it's ready or failed there's no
        # useful live output to watch.
        if self._log_path is not None and self.status in ("waiting", "failed"):
            t.append("   [click to view output]", style="cyan dim")
        return t

    def on_click(self) -> None:
        if self._log_path is None:
            return
        self.app.push_screen(ServerLogScreen(self._name, self._log_path))


class SplashScreen(ModalScreen[None]):
    """Startup splash: shows per-server progress while ServerManager runs."""

    def __init__(self) -> None:
        super().__init__()
        # Pending messages if the splash API is called before compose has
        # finished mounting children.
        self._ready = False
        self._pending_log: list[str] = []
        # (name, status, elapsed, log_path)
        self._pending_rows: list[tuple[str, str, int, Path | None]] = []

    def compose(self) -> ComposeResult:
        with Container(id="splash-panel"):
            yield Label("Voice Agent · starting servers", id="splash-title")
            yield Vertical(id="splash-servers")
            yield VerticalScroll(id="splash-log")

    def on_mount(self) -> None:
        self._ready = True
        for text in self._pending_log:
            self._append_log(text)
        self._pending_log.clear()
        for name, status, elapsed, log_path in self._pending_rows:
            self._apply_row(name, status, elapsed, log_path)
        self._pending_rows.clear()

    # ── API used by VoiceAgentApp ─────────────────────────

    def log_line(self, text: str) -> None:
        if not self._ready:
            self._pending_log.append(text)
            return
        self._append_log(text)

    def _append_log(self, text: str) -> None:
        try:
            log = self.query_one("#splash-log", VerticalScroll)
        except Exception:
            return
        log.mount(Static(escape(text)))
        log.scroll_end(animate=False)

    def ensure_row(self, name: str, log_path: Path | None = None) -> ServerRow | None:
        if not self._ready:
            return None
        try:
            container = self.query_one("#splash-servers", Vertical)
        except Exception:
            return None
        for row in container.query(ServerRow):
            if row._name == name:
                # Upgrade the row's log_path lazily — we often create the
                # row on "Starting X…" before the launcher knows the log
                # path, and the path is attached by the next call.
                if log_path is not None:
                    row.set_log_path(log_path)
                return row
        row = ServerRow(name, log_path=log_path)
        container.mount(row)
        return row

    def _apply_row(
        self,
        name: str,
        status: str,
        elapsed: int,
        log_path: Path | None = None,
    ) -> None:
        row = self.ensure_row(name, log_path=log_path)
        if row is None:
            return
        row.status = status
        if status == "waiting":
            row.elapsed = elapsed

    def set_waiting(
        self, name: str, elapsed: int, log_path: Path | None = None
    ) -> None:
        if not self._ready:
            self._pending_rows.append((name, "waiting", elapsed, log_path))
            return
        self._apply_row(name, "waiting", elapsed, log_path)

    def set_ready(self, name: str) -> None:
        if not self._ready:
            self._pending_rows.append((name, "ready", 0, None))
            return
        self._apply_row(name, "ready", 0)

    def set_failed(self, name: str) -> None:
        if not self._ready:
            self._pending_rows.append((name, "failed", 0, None))
            return
        self._apply_row(name, "failed", 0)


# ── Server log viewer ────────────────────────────────────


class ServerLogScreen(ModalScreen[None]):
    """Tail a server's log file in a modal so the user can see what's
    happening during a long startup (e.g. Qwen3 downloading the model or
    compiling kernels). Auto-refreshes every 500ms by reading the bytes
    appended since the last tick.

    Not a live `tail -f` stream — the log file is written to by a
    subprocess we don't own; we poll the file size and read the delta.
    That's plenty responsive for human eyes and avoids extra threads.
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+c", "close", "Close"),
    ]

    def __init__(self, name: str, log_path: Path) -> None:
        super().__init__()
        self._name = name
        self._log_path = log_path
        # Byte offset into the log file of the last chunk we read. We
        # append only new bytes each tick so the scroll position stays
        # at the bottom without re-rendering the whole buffer.
        self._offset = 0
        self._auto_scroll = True
        # Accumulate rendered text on the screen rather than round-tripping
        # through Static.renderable (which isn't a public attribute).
        self._buffer = ""

    def compose(self) -> ComposeResult:
        with Container(id="log-panel"):
            yield Label(f"{self._name} — {self._log_path.name}", id="log-title")
            with VerticalScroll(id="log-body"):
                yield Static("", id="log-content", markup=False)
            with Horizontal(id="log-buttons"):
                yield Button("Close", id="log-close", variant="primary")

    def on_mount(self) -> None:
        self._poll()  # render whatever's already in the file
        self.set_interval(0.5, self._poll)

    def _poll(self) -> None:
        if not self._log_path.exists():
            return
        try:
            size = self._log_path.stat().st_size
        except OSError:
            return
        if size <= self._offset:
            # Also handle truncation (a new server replaced the log file).
            if size < self._offset:
                self._offset = 0
            else:
                return
        try:
            with self._log_path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read(size - self._offset)
            self._offset = size
        except OSError:
            return
        try:
            content = self.query_one("#log-content", Static)
            body = self.query_one("#log-body", VerticalScroll)
        except Exception:
            return
        self._buffer += chunk.decode(errors="replace")
        # Cap the rendered buffer so a long-running server doesn't grow
        # the widget unboundedly. 200 KB is ~4k lines, comfortably more
        # than any startup flow produces.
        if len(self._buffer) > 200_000:
            self._buffer = self._buffer[-200_000:]
        content.update(self._buffer)
        if self._auto_scroll:
            body.scroll_end(animate=False)

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "log-close":
            self.dismiss(None)


# ── Model switch modal ───────────────────────────────────


class SettingsScreen(ModalScreen[tuple[str, str, str] | None]):
    """Modal with three Select dropdowns (STT / LLM / TTS) + Apply/Cancel."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    def compose(self) -> ComposeResult:
        s = self._settings
        with Container(id="settings-panel"):
            yield Label("Settings", id="settings-title")
            with Vertical(id="settings-rows"):
                yield Label("STT", classes="settings-label")
                yield Select(
                    options=[(m.display_name, m.name) for m in s.stt_models],
                    value=s.active_stt,
                    id="settings-stt",
                    allow_blank=False,
                )
                yield Label("LLM", classes="settings-label")
                yield Select(
                    options=[(m.display_name, m.name) for m in s.llm_models],
                    value=s.active_llm,
                    id="settings-llm",
                    allow_blank=False,
                )
                yield Label("TTS", classes="settings-label")
                yield Select(
                    options=[(m.display_name, m.name) for m in s.tts_models],
                    value=s.active_tts,
                    id="settings-tts",
                    allow_blank=False,
                )
            with Horizontal(id="settings-buttons"):
                yield Button("Apply", id="settings-apply", variant="primary")
                yield Button("Cancel", id="settings-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-cancel":
            self.dismiss(None)
            return
        if event.button.id == "settings-apply":
            stt = self.query_one("#settings-stt", Select).value
            llm = self.query_one("#settings-llm", Select).value
            tts = self.query_one("#settings-tts", Select).value
            if isinstance(stt, str) and isinstance(llm, str) and isinstance(tts, str):
                self.dismiss((stt, llm, tts))
