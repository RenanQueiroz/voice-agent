"""Load MCP server configurations from mcp_servers.toml."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

from agents.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHttp

_PROJECT_ROOT = Path(__file__).parent.parent

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR_NAME} references in strings, lists, and dicts."""
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_mcp_servers() -> list[MCPServer]:
    """Load MCP servers from mcp_servers.toml. Returns empty list if no servers configured."""
    path = _PROJECT_ROOT / "mcp_servers.toml"
    if not path.exists():
        return []

    with open(path, "rb") as f:
        config = tomllib.load(f)

    servers: list[MCPServer] = []

    for name, entry in config.items():
        entry = _expand_env(entry)
        # Per-server toggle. Defaults to True so omitting the key keeps the
        # existing behavior. Set `enabled = false` to skip this server
        # without deleting it from the file.
        if not entry.get("enabled", True):
            continue
        server_type = entry.get("type", "stdio")
        cache_tools = entry.get("cache_tools", True)

        if server_type == "stdio":
            command = entry.get("command")
            if not command:
                msg = f"MCP server '{name}': missing 'command'"
                raise ValueError(msg)

            params: dict = {"command": command}
            if "args" in entry:
                params["args"] = entry["args"]
            if "env" in entry:
                params["env"] = entry["env"]
            if "cwd" in entry:
                params["cwd"] = entry["cwd"]

            servers.append(
                MCPServerStdio(
                    params=params,  # type: ignore[arg-type]
                    cache_tools_list=cache_tools,
                    name=name,
                )
            )

        elif server_type == "http":
            url = entry.get("url")
            if not url:
                msg = f"MCP server '{name}': missing 'url'"
                raise ValueError(msg)

            servers.append(
                MCPServerStreamableHttp(
                    params={"url": url},  # type: ignore[arg-type]
                    cache_tools_list=cache_tools,
                    name=name,
                )
            )

        else:
            msg = f"MCP server '{name}': unknown type '{server_type}' (use 'stdio' or 'http')"
            raise ValueError(msg)

    return servers
