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
            "Run a shell command on the user's computer. Each call requires "
            "the user's explicit approval before the command executes; if the "
            "user declines, the tool returns 'User declined…'. Use sparingly "
            "and only when needed to answer the user's request. Prefer "
            "read-only or clearly-scoped commands."
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
