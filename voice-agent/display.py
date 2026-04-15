"""Centralized terminal display using Rich."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from .config import Settings

console = Console()


class Display:
    def __init__(self) -> None:
        self.console = console
        self._agent_streaming = False

    # ── Startup ──────────────────────────────────────────────

    def server_setup_start(self) -> None:
        self.console.print("\n[bold]Setting up local servers...[/]\n")

    def server_installing_system(self, packages: list[str]) -> None:
        self.console.print(
            f"  [dim]Installing system packages:[/] {', '.join(packages)}..."
        )

    def server_installing(self, packages: list[str]) -> None:
        self.console.print(f"  [dim]Installing:[/] {', '.join(packages)}...")

    def server_installed(self) -> None:
        self.console.print("  [green]✓[/] Packages installed.")

    def server_install_failed(self, lines: list[str]) -> None:
        self.console.print("  [red]✗[/] Failed to install packages:")
        for line in lines:
            self.console.print(f"    [dim]{line}[/]")

    def server_patched(self, description: str) -> None:
        self.console.print(f"  [dim]✓ {description}[/]")

    def server_starting(self, name: str) -> None:
        self.console.print(f"  [dim]Starting {name}...[/]")

    def server_waiting(self, name: str, elapsed: int) -> None:
        sys.stdout.write(f"\r  [⠋] Waiting for {name}... ({elapsed}s)")
        sys.stdout.flush()

    def server_ready_one(self, name: str) -> None:
        self.console.print(f"\r  [green]✓[/] {name} ready.          ")

    def server_all_ready(self) -> None:
        self.console.print("\n  [bold green]All servers ready.[/]\n")

    def server_failed(self, name: str, log_lines: list[str]) -> None:
        lines_text = "\n".join(f"  {line}" for line in log_lines)
        self.console.print(
            Panel(
                f"[red]{name}[/] exited unexpectedly:\n\n[dim]{lines_text}[/]",
                title="[red]Server Error[/]",
                border_style="red",
            )
        )

    def server_timeout(self, name: str, timeout: int) -> None:
        self.console.print(
            f"\n  [red]✗[/] {name} did not become ready within {timeout}s"
        )

    def setup_failed(self) -> None:
        self.console.print("[red]Failed to start local servers. Exiting.[/]")

    # ── Ready Banner ─────────────────────────────────────────

    def ready_banner(self, settings: Settings) -> None:
        mode = settings.voice_mode
        input_mode = settings.input_mode.replace("_", " ")
        llm = (
            settings.llm_model.split("/")[-1]
            if "/" in settings.llm_model
            else settings.llm_model
        )
        tts = (
            settings.tts_model.split("/")[-1]
            if "/" in settings.tts_model
            else settings.tts_model
        )

        info = Text()
        info.append("Mode: ", style="dim")
        info.append(mode, style="bold")
        info.append("  │  ", style="dim")
        info.append("Input: ", style="dim")
        info.append(input_mode, style="bold")
        info.append("  │  ", style="dim")
        info.append("Q", style="bold yellow")
        info.append(" to quit", style="dim")
        info.append("\n")
        info.append("LLM: ", style="dim")
        info.append(llm, style="cyan")
        info.append("  │  ", style="dim")
        info.append("TTS: ", style="dim")
        info.append(tts, style="cyan")

        self.console.print(
            Panel(info, title="[bold]Voice Agent[/]", border_style="blue")
        )
        self.console.print()

    # ── Conversation State ───────────────────────────────────

    def listening(self) -> None:
        self.console.print("[green]●[/] [dim]Listening...[/]")

    def recording_start(self) -> None:
        self.console.print("[bold green]●[/] Recording... [dim](press K to stop)[/]")

    def recording_too_short(self) -> None:
        self.console.print("[dim]  Too short, skipping.[/]")

    def ready_for_key(self) -> None:
        self.console.print("[green]●[/] [dim]Press[/] [bold]K[/] [dim]to speak.[/]")

    def vad_speaking(self, rms: int) -> None:
        bar_len = min(rms // 30, 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        # Use ANSI codes directly since this overwrites the same line
        sys.stdout.write(f"\r  \033[1;32m●\033[0m Speaking  \033[32m{bar}\033[0m")
        sys.stdout.flush()

    def vad_silence(self, remaining_ms: int) -> None:
        sys.stdout.write(
            f"\r  \033[33m●\033[0m Silence   {remaining_ms:>4}ms remaining...   "
        )
        sys.stdout.flush()

    def vad_clear(self) -> None:
        sys.stdout.write("\r" + " " * 55 + "\r")
        sys.stdout.flush()

    def processing(self, duration: float) -> None:
        self.console.print(f"[yellow]●[/] Processing {duration:.1f}s of audio...")

    # ── Transcription ────────────────────────────────────────

    def user_said(self, text: str) -> None:
        self.console.print(f"\n  [bold cyan]You:[/] {text}")

    def agent_start(self) -> None:
        self._agent_streaming = True
        sys.stdout.write("  \033[1;35mAgent:\033[0m ")
        sys.stdout.flush()

    def agent_chunk(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def agent_end(self) -> None:
        if self._agent_streaming:
            sys.stdout.write("\n\n")
            sys.stdout.flush()
            self._agent_streaming = False

    # ── Lifecycle (no longer shown to user) ──────────────────

    def turn_started(self) -> None:
        pass  # handled by processing() state

    def turn_ended(self) -> None:
        pass  # handled by listening() state

    def session_ended(self) -> None:
        pass  # handled by goodbye()

    # ── Errors ───────────────────────────────────────────────

    def api_error(self, message: str) -> None:
        self.console.print(
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
        self.console.print(
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
        self.console.print(
            Panel(msg, title="[red]Connection Error[/]", border_style="red")
        )

    def auth_error(self, message: str) -> None:
        self.console.print(
            Panel(
                f"{message}\n\nCheck your [bold]OPENAI_API_KEY[/] in .env",
                title="[red]Auth Error[/]",
                border_style="red",
            )
        )

    def rate_limit_error(self, message: str) -> None:
        self.console.print(
            Panel(
                f"{message}\n\nCheck your plan at [link]https://platform.openai.com/settings/organization/billing[/link]",
                title="[yellow]Rate Limit[/]",
                border_style="yellow",
            )
        )

    def tts_stream_error(self) -> None:
        self.console.print(
            Panel(
                "TTS server closed the connection mid-stream.",
                title="[red]TTS Error[/]",
                border_style="red",
            )
        )

    # ── Exit ─────────────────────────────────────────────────

    def goodbye(self) -> None:
        self.console.print("\n[dim]Goodbye![/]")
