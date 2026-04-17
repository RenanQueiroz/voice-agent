"""Shell-command tool with human-in-the-loop approval.

The agent calls `run_shell_command(command)`. Inside the tool we mount an
`ApprovalCard` and await the user's decision — if declined, we return a
refusal string to the agent; if approved, we run the command via the
user's shell with a timeout, capture stdout+stderr, and return a
truncated transcript.

Safety:

* **Every invocation requires explicit user approval.** No allowlist /
  denylist shortcuts. The full command is displayed verbatim.
* The command runs with the user's own permissions in a subshell — this
  is a dual-use power tool. The feature is opt-in via `[shell].enabled`
  and is disabled by default.
* stdout+stderr are truncated to `max_output_bytes`; the process is
  killed at `timeout_seconds`.
* No environment stripping or path manipulation: we deliberately keep
  `env` / `cwd` predictable so the agent sees the same shell the user
  sees. That's the whole point of the tool.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from agents import FunctionTool, function_tool

from .config import ShellConfig

if TYPE_CHECKING:
    from .app import VoiceAgentApp

_PROJECT_ROOT = Path(__file__).parent.parent


def current_shell() -> str:
    """Best-effort detection of the user's interactive shell."""
    return os.environ.get("SHELL") or "/bin/sh"


def system_summary() -> str:
    """One-line platform summary for the agent's system prompt."""
    uname = platform.uname()
    return (
        f"OS: {uname.system} {uname.release} ({uname.machine}). "
        f"Shell: {current_shell()}."
    )


def create_shell_tool(app: VoiceAgentApp, cfg: ShellConfig) -> FunctionTool:
    """Build the `run_shell_command` function tool bound to this app + config."""

    cwd = (_PROJECT_ROOT / cfg.cwd).resolve() if cfg.cwd else _PROJECT_ROOT
    timeout = max(1, int(cfg.timeout_seconds))
    max_bytes = max(256, int(cfg.max_output_bytes))

    @function_tool(
        name_override="run_shell_command",
        description_override=(
            "Run a shell command on the user's computer. Use this proactively "
            "whenever the user asks you to DO something on their machine — do "
            "not refuse a request that can be accomplished by a shell command. "
            "The user must approve every command before it runs, so you can "
            "safely propose one whenever it helps.\n\n"
            "Common things you can do (examples are for macOS — adapt to the "
            "user's OS as reported in the system prompt):\n"
            "- Open apps or files: `open -a \"Safari\"`, `open ~/Documents/foo.pdf`.\n"
            "  (Linux: `xdg-open <path>`.)\n"
            "- Look things up on the web in plain text: `curl -s wttr.in/lisbon?format=3`, "
            "`curl -s https://ipinfo.io/ip`.\n"
            "- Check files and directories: `ls -lah <path>`, `find ~ -name '*.pdf'`, "
            "`cat <file>`, `head -n 50 <file>`.\n"
            "- System info: `uname -a`, `df -h`, `uptime`, `date`, `sw_vers`.\n"
            "- Clipboard on macOS: `pbcopy`, `pbpaste`.\n"
            "- Play a sound or speak: `afplay /System/Library/Sounds/Glass.aiff`, "
            "`say 'done'`.\n"
            "- AppleScript for anything macOS-GUI-related: `osascript -e '...'`.\n"
            "- Brief network probes: `ping -c 1 example.com`, `dig example.com +short`.\n\n"
            "Guidelines:\n"
            "- Pick the simplest command that answers the user's request. Chain with "
            "`&&` or pipes when it helps; avoid multi-step scripts unless necessary.\n"
            "- Avoid clearly destructive operations (`rm -rf`, `dd`, format, recursive "
            "`chmod`) unless the user explicitly asked for them.\n"
            "- If the user declines, the tool returns 'User declined…'. Accept the "
            "decision and move on without retrying.\n"
            "- After the command runs, summarize the result for the user in plain "
            "speech — they will hear your response, not the raw output."
        ),
    )
    async def run_shell_command(command: str) -> str:
        """Execute `command` in the user's shell after explicit approval.

        Args:
            command: The shell command to run (passed to /bin/sh -c).

        Returns:
            A transcript with the command, exit code, and truncated output,
            or a refusal string if the user declined.
        """
        command = command.strip()
        if not command:
            return "Error: empty command."

        approved = await app.request_shell_approval(command)
        if not approved:
            return "User declined to run this command."

        shell = current_shell()
        try:
            proc = await asyncio.create_subprocess_exec(
                shell,
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd),
            )
        except FileNotFoundError:
            return f"Error: shell '{shell}' not found."
        except Exception as e:
            return f"Error starting shell: {e}"

        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            return f"Error: command timed out after {timeout}s."

        rc = proc.returncode if proc.returncode is not None else -1
        out = (stdout_bytes or b"").decode("utf-8", errors="replace")
        truncated = len(out.encode("utf-8")) > max_bytes
        if truncated:
            out = out[:max_bytes] + "\n… (output truncated)"

        header = f"$ {shlex.quote(shell)} -c {shlex.quote(command)}\n(exit {rc})"
        body = out.rstrip() if out else "(no output)"
        return f"{header}\n{body}"

    return run_shell_command
