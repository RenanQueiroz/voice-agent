"""Textual widgets for the voice-agent TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Label, Select, Static

from .display import TurnMetrics

if TYPE_CHECKING:
    from .config import Settings


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
        yield Static("You", classes="label")
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
        yield Static("Agent", classes="label")
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
            parts.append(f"{_label('LLM', llm_name)} {m.llm_seconds:.1f}s")
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
    """Footer control row: clickable Mute / Interrupt / Switch / Quit buttons."""

    def compose(self) -> ComposeResult:
        yield Button("Mute  (M)", id="btn-mute", variant="default")
        yield Button(
            "Interrupt  (Space)", id="btn-interrupt", variant="warning", disabled=True
        )
        yield Button("Switch models  (S)", id="btn-switch", variant="default")
        yield Button("Quit  (Q)", id="btn-quit", variant="error")


class StatusFooter(Vertical):
    """Persistent footer: state, models, tools, controls."""

    def compose(self) -> ComposeResult:
        yield StateRow()
        yield ModelRow()
        yield ToolsRow()
        yield ControlRow()


# ── Splash screen ────────────────────────────────────────


class ServerRow(Widget):
    """One row on the splash screen, per server."""

    status: reactive[str] = reactive("pending")
    elapsed: reactive[int] = reactive(0)

    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._spin_idx = 0

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
        return t


class SplashScreen(ModalScreen[None]):
    """Startup splash: shows per-server progress while ServerManager runs."""

    def __init__(self) -> None:
        super().__init__()
        # Pending messages if the splash API is called before compose has
        # finished mounting children.
        self._ready = False
        self._pending_log: list[str] = []
        self._pending_rows: list[tuple[str, str, int]] = []  # (name, status, elapsed)

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
        for name, status, elapsed in self._pending_rows:
            self._apply_row(name, status, elapsed)
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

    def ensure_row(self, name: str) -> ServerRow | None:
        if not self._ready:
            return None
        try:
            container = self.query_one("#splash-servers", Vertical)
        except Exception:
            return None
        for row in container.query(ServerRow):
            if row._name == name:
                return row
        row = ServerRow(name)
        container.mount(row)
        return row

    def _apply_row(self, name: str, status: str, elapsed: int) -> None:
        row = self.ensure_row(name)
        if row is None:
            return
        row.status = status
        if status == "waiting":
            row.elapsed = elapsed

    def set_waiting(self, name: str, elapsed: int) -> None:
        if not self._ready:
            self._pending_rows.append((name, "waiting", elapsed))
            return
        self._apply_row(name, "waiting", elapsed)

    def set_ready(self, name: str) -> None:
        if not self._ready:
            self._pending_rows.append((name, "ready", 0))
            return
        self._apply_row(name, "ready", 0)

    def set_failed(self, name: str) -> None:
        if not self._ready:
            self._pending_rows.append((name, "failed", 0))
            return
        self._apply_row(name, "failed", 0)


# ── Model switch modal ───────────────────────────────────


class ModelSwitchScreen(ModalScreen[tuple[str, str, str] | None]):
    """Modal with three Select dropdowns (STT / LLM / TTS) + Apply/Cancel."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    def compose(self) -> ComposeResult:
        s = self._settings
        with Container(id="switch-panel"):
            yield Label("Switch models", id="switch-title")
            with Vertical(id="switch-rows"):
                yield Label("STT", classes="switch-label")
                yield Select(
                    options=[(m.display_name, m.name) for m in s.stt_models],
                    value=s.active_stt,
                    id="switch-stt",
                    allow_blank=False,
                )
                yield Label("LLM", classes="switch-label")
                yield Select(
                    options=[(m.display_name, m.name) for m in s.llm_models],
                    value=s.active_llm,
                    id="switch-llm",
                    allow_blank=False,
                )
                yield Label("TTS", classes="switch-label")
                yield Select(
                    options=[(m.display_name, m.name) for m in s.tts_models],
                    value=s.active_tts,
                    id="switch-tts",
                    allow_blank=False,
                )
            with Horizontal(id="switch-buttons"):
                yield Button("Apply", id="switch-apply", variant="primary")
                yield Button("Cancel", id="switch-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "switch-cancel":
            self.dismiss(None)
            return
        if event.button.id == "switch-apply":
            stt = self.query_one("#switch-stt", Select).value
            llm = self.query_one("#switch-llm", Select).value
            tts = self.query_one("#switch-tts", Select).value
            if isinstance(stt, str) and isinstance(llm, str) and isinstance(tts, str):
                self.dismiss((stt, llm, tts))
