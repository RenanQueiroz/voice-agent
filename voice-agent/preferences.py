"""Persistent per-user model selections.

Stored in a gitignored `preferences.toml` at the project root. The file is
tiny and written by the app whenever the user applies a new set of active
models via the Switch modal.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
PREFERENCES_PATH = _PROJECT_ROOT / "preferences.toml"


def load_preferences(path: Path = PREFERENCES_PATH) -> dict[str, str]:
    """Return `{"stt": name, "llm": name, "tts": name}` as available.

    Missing file or missing keys return an empty dict (or partial). The
    caller is responsible for falling back to the first catalog entry.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}
    active = data.get("active", {})
    result: dict[str, str] = {}
    for role in ("stt", "llm", "tts"):
        name = active.get(role)
        if isinstance(name, str) and name:
            result[role] = name
    return result


def save_preferences(
    stt: str, llm: str, tts: str, path: Path = PREFERENCES_PATH
) -> None:
    """Write the active model names to `preferences.toml`.

    Simple hand-formatted output (three fields, one table) so we don't need
    a TOML writer dependency.
    """
    lines = [
        "# Saved by voice-agent. Change models from the Switch modal (press 's').",
        "[active]",
        f'stt = "{stt}"',
        f'llm = "{llm}"',
        f'tts = "{tts}"',
        "",
    ]
    path.write_text("\n".join(lines))
