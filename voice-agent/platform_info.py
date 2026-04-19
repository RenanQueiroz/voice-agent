"""Light OS detection used to gate runtimes and install system dependencies.

The runtime registry in `runtimes.py` maps each runtime to the set of
operating systems it runs on (e.g. mlx-* are darwin-only). Config parsing
filters catalog entries by `current_os()` so a Linux user never sees an
mlx entry in the Switch modal, and `ServerManager._ensure_system_deps`
uses `linux_package_manager()` to pick the right install command.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Literal

OS = Literal["darwin", "linux", "windows", "unknown"]
PackageManager = Literal["apt", "dnf", "pacman", "zypper"]


def current_os() -> OS:
    """Return the running OS as one of our canonical tags."""
    system = platform.system().lower()
    if system == "darwin":
        return "darwin"
    if system == "linux":
        return "linux"
    if system == "windows":
        return "windows"
    return "unknown"


def linux_package_manager() -> PackageManager | None:
    """Detect the system package manager on Linux.

    Checks /etc/os-release first (cheap and authoritative), then falls back
    to binary lookups. Returns None when we can't identify one — callers
    should treat that as "install manually".
    """
    if current_os() != "linux":
        return None

    os_release = Path("/etc/os-release")
    if os_release.exists():
        try:
            fields: dict[str, str] = {}
            for line in os_release.read_text().splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                fields[k.strip()] = v.strip().strip('"')
            id_like = (fields.get("ID", "") + " " + fields.get("ID_LIKE", "")).lower()
            if any(tag in id_like for tag in ("debian", "ubuntu")):
                return "apt"
            if any(tag in id_like for tag in ("fedora", "rhel", "centos")):
                return "dnf"
            if any(tag in id_like for tag in ("arch",)):
                return "pacman"
            if any(tag in id_like for tag in ("suse", "opensuse")):
                return "zypper"
        except Exception:
            pass

    for manager, binary in (
        ("apt", "apt-get"),
        ("dnf", "dnf"),
        ("pacman", "pacman"),
        ("zypper", "zypper"),
    ):
        if shutil.which(binary):
            return manager  # type: ignore[return-value]
    return None


def has_cuda() -> bool:
    """True if an NVIDIA CUDA stack looks usable — cheap heuristic only.

    Used by the setup scripts to pick CUDA-enabled llama.cpp/whisper.cpp
    builds. We check for `nvidia-smi` since it ships with the driver; the
    presence of a working GPU matters more than whether the full toolkit
    is installed, and prebuilt CUDA binaries bring their own runtime libs.
    """
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError:
        return False
