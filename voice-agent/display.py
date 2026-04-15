"""Centralized terminal display using Rich with a persistent footer."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from .config import Settings

console = Console()


@dataclass
class TurnMetrics:
    """Timing metrics for a single conversation turn."""

    stt_seconds: float = 0.0
    llm_seconds: float = 0.0
    llm_tokens: int = 0
    tts_seconds: float = 0.0
    tts_first_byte_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def llm_tokens_per_sec(self) -> float:
        return self.llm_tokens / self.llm_seconds if self.llm_seconds > 0 else 0.0


class Display:
    def __init__(self) -> None:
        self.console = console
        self._agent_streaming = False
        self._live: Live | None = None
        self._settings: Settings | None = None
        self._state = "idle"
        self._is_muted = False
        self._last_metrics: TurnMetrics | None = None
        self._vad_rms = 0
        self._vad_remaining_ms = 0
        self._processing_duration = 0.0

    # ── Footer management ────────────────────────────────────

    def start_footer(self, settings: Settings) -> None:
        """Start the persistent footer. Call once after settings are known."""
        self._settings = settings
        self._live = Live(
            self._render_footer(),
            console=self.console,
            refresh_per_second=4,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop_footer(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def _render_footer(self) -> Panel:
        """Render the persistent status bar."""
        s = self._settings
        bar = Text()

        # Line 1: Activity indicator
        state_styles = {
            "idle": ("○", "dim"),
            "listening": ("●", "green"),
            "speaking": ("●", "bold green"),
            "responding": ("●", "magenta"),
            "silence": ("●", "yellow"),
            "processing": ("●", "yellow"),
            "muted": ("●", "red"),
        }
        icon, style = state_styles.get(self._state, ("○", "dim"))
        bar.append(f" {icon} ", style=style)

        if self._state == "speaking":
            bar_len = min(self._vad_rms // 30, 20)
            energy_bar = "█" * bar_len + "░" * (20 - bar_len)
            bar.append("Speaking  ", style=style)
            bar.append(energy_bar, style="green")
        elif self._state == "silence":
            bar.append(f"Silence  {self._vad_remaining_ms}ms", style=style)
        elif self._state == "processing" and self._processing_duration > 0:
            bar.append(
                f"Processing {self._processing_duration:.1f}s of audio...", style=style
            )
        else:
            bar.append(self._state.capitalize(), style=style)

        if self._is_muted and self._state != "muted":
            bar.append("  ", style="dim")
            bar.append("(muted)", style="red dim")

        # Line 2: Model info
        if s:

            def _short(model: str) -> str:
                return model.split("/")[-1] if "/" in model else model

            bar.append("\n ")
            bar.append("STT:", style="dim")
            bar.append(_short(s.stt_model), style="cyan dim")
            bar.append("  ", style="dim")
            bar.append("LLM:", style="dim")
            bar.append(_short(s.llm_model), style="cyan dim")
            bar.append("  ", style="dim")
            bar.append("TTS:", style="dim")
            bar.append(_short(s.tts_model), style="cyan dim")

        # Line 3: Controls
        bar.append("\n ")
        if self._state in ("responding", "processing"):
            bar.append("Space", style="bold yellow")
            bar.append(" interrupt  ", style="dim")
        bar.append("M", style="bold yellow")
        bar.append(" mute  ", style="dim")
        bar.append("Q", style="bold yellow")
        bar.append(" quit", style="dim")

        return Panel(bar, border_style="blue", padding=(0, 1))

    def _update_footer(self) -> None:
        if self._live:
            self._live.update(self._render_footer())

    def _print(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        """Print above the footer."""
        if self._live:
            self._live.console.print(*args, **kwargs)  # type: ignore[arg-type]
        else:
            self.console.print(*args, **kwargs)  # type: ignore[arg-type]

    def _set_state(self, state: str) -> None:
        self._state = state
        self._update_footer()

    # ── Startup ──────────────────────────────────────────────

    def server_setup_start(self) -> None:
        self._print("\n[bold]Setting up local servers...[/]\n")

    def server_installing_system(self, packages: list[str]) -> None:
        self._print(f"  [dim]Installing system packages:[/] {', '.join(packages)}...")

    def server_installing(self, packages: list[str]) -> None:
        self._print(f"  [dim]Installing:[/] {', '.join(packages)}...")

    def server_installed(self) -> None:
        self._print("  [green]✓[/] Packages installed.")

    def server_install_failed(self, lines: list[str]) -> None:
        self._print("  [red]✗[/] Failed to install packages:")
        for line in lines:
            self._print(f"    [dim]{line}[/]")

    def server_patched(self, description: str) -> None:
        self._print(f"  [dim]✓ {description}[/]")

    def server_starting(self, name: str) -> None:
        self._print(f"  [dim]Starting {name}...[/]")

    def server_waiting(self, name: str, elapsed: int) -> None:
        # Use raw stdout for overwriting since _print adds newlines
        sys.stdout.write(f"\r  [⠋] Waiting for {name}... ({elapsed}s)")
        sys.stdout.flush()

    def server_ready_one(self, name: str) -> None:
        self._print(f"\r  [green]✓[/] {name} ready.          ")

    def server_all_ready(self) -> None:
        self._print("\n  [bold green]All servers ready.[/]\n")

    def server_failed(self, name: str, log_lines: list[str]) -> None:
        lines_text = "\n".join(f"  {line}" for line in log_lines)
        self._print(
            Panel(
                f"[red]{name}[/] exited unexpectedly:\n\n[dim]{lines_text}[/]",
                title="[red]Server Error[/]",
                border_style="red",
            )
        )

    def server_timeout(self, name: str, timeout: int) -> None:
        self._print(f"\n  [red]✗[/] {name} did not become ready within {timeout}s")

    def setup_failed(self) -> None:
        self._print("[red]Failed to start local servers. Exiting.[/]")

    # ── Ready Banner ─────────────────────────────────────────

    def ready_banner(self, settings: Settings) -> None:
        # Start the persistent footer instead of a one-time banner
        self.start_footer(settings)
        self._set_state("listening")

    # ── Conversation State ───────────────────────────────────

    def listening(self) -> None:
        self._set_state("listening")

    def muted(self) -> None:
        self._is_muted = True
        self.vad_clear()
        self._set_state("muted")

    def unmuted(self) -> None:
        self._is_muted = False
        self._set_state("listening")

    def recording_start(self) -> None:
        self._print("[bold green]●[/] Recording... [dim](press K to stop)[/]")

    def recording_too_short(self) -> None:
        self._print("[dim]  Too short, skipping.[/]")

    def ready_for_key(self) -> None:
        self._print("[green]●[/] [dim]Press[/] [bold]K[/] [dim]to speak.[/]")

    def vad_speaking(self, rms: int) -> None:
        self._vad_rms = rms
        self._state = "speaking"
        self._update_footer()

    def vad_silence(self, remaining_ms: int) -> None:
        self._vad_remaining_ms = remaining_ms
        self._state = "silence"
        self._update_footer()

    def vad_clear(self) -> None:
        self._vad_rms = 0
        self._vad_remaining_ms = 0

    def processing(self, duration: float) -> None:
        self._processing_duration = duration
        self._set_state("processing")

    def interrupted(self) -> None:
        self._print("[dim]  -- interrupted[/]")

    # ── Transcription ────────────────────────────────────────

    def user_said(self, text: str, stt_seconds: float = 0.0) -> None:
        self._print(f"\n  [bold cyan]You:[/] {text}")
        if stt_seconds > 0:
            self._print(f"  [dim]STT {stt_seconds:.1f}s[/]")

    def agent_start(self) -> None:
        self._agent_streaming = True
        self._agent_buffer = ""
        self._set_state("responding")

    def agent_chunk(self, text: str) -> None:
        self._agent_buffer += text

    def agent_end(self) -> None:
        if self._agent_streaming:
            if self._agent_buffer:
                self._print(f"\n  [bold magenta]Agent:[/] {self._agent_buffer}")
            self._agent_streaming = False
            self._agent_buffer = ""

    # ── Metrics ──────────────────────────────────────────────

    def metrics(self, m: TurnMetrics) -> None:
        self._last_metrics = m
        parts: list[str] = []
        if m.llm_seconds > 0:
            tok_s = f" ({m.llm_tokens_per_sec:.0f} tok/s)" if m.llm_tokens > 0 else ""
            parts.append(f"LLM {m.llm_seconds:.1f}s{tok_s}")
        if m.tts_seconds > 0:
            parts.append(f"TTS {m.tts_seconds:.1f}s")
        if m.total_seconds > 0:
            parts.append(f"Total {m.total_seconds:.1f}s")
        self._print(f"  [dim]{' │ '.join(parts)}[/]\n")

    # ── Lifecycle (no-op) ────────────────────────────────────

    def turn_started(self) -> None:
        pass

    def turn_ended(self) -> None:
        pass

    def session_ended(self) -> None:
        pass

    # ── Errors ───────────────────────────────────────────────

    def api_error(self, message: str) -> None:
        self._print(
            Panel(
                f"[red]{message}[/]",
                title="[red]API Error[/]",
                border_style="red",
            )
        )

    def api_error_with_logs(
        self, message: str, server_logs: dict[str, list[str]]
    ) -> None:
        parts = [f"[red]{message}[/]"]
        for name, lines in server_logs.items():
            parts.append(f"\n[bold]{name}[/] logs:")
            for line in lines:
                parts.append(f"  [dim]{line}[/]")
        self._print(
            Panel(
                "\n".join(parts),
                title="[red]API Error[/]",
                border_style="red",
            )
        )

    def connection_error(self, settings: Settings) -> None:
        if settings.voice_mode == "local":
            msg = (
                "Could not reach the local servers.\n\n"
                f"  STT/TTS: [cyan]{settings.mlx_audio_url}[/]\n"
                f"  LLM:     [cyan]{settings.mlx_vlm_url}[/]"
            )
        else:
            msg = "Could not reach the OpenAI API.\nCheck your internet connection and OPENAI_API_KEY in .env"
        self._print(Panel(msg, title="[red]Connection Error[/]", border_style="red"))

    def auth_error(self, message: str) -> None:
        self._print(
            Panel(
                f"{message}\n\nCheck your [bold]OPENAI_API_KEY[/] in .env",
                title="[red]Auth Error[/]",
                border_style="red",
            )
        )

    def rate_limit_error(self, message: str) -> None:
        self._print(
            Panel(
                f"{message}\n\nCheck your plan at [link]https://platform.openai.com/settings/organization/billing[/link]",
                title="[yellow]Rate Limit[/]",
                border_style="yellow",
            )
        )

    def tts_stream_error(self) -> None:
        self._print(
            Panel(
                "TTS server closed the connection mid-stream.",
                title="[red]TTS Error[/]",
                border_style="red",
            )
        )

    # ── Exit ─────────────────────────────────────────────────

    def goodbye(self) -> None:
        self.stop_footer()
        self.console.print("\n[dim]Goodbye![/]")
