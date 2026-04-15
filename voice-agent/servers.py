"""Manages local mlx-audio and mlx-vlm server processes."""

from __future__ import annotations

import asyncio
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from .config import Settings

if TYPE_CHECKING:
    from .display import Display

_PROJECT_ROOT = Path(__file__).parent.parent

# How long to wait for servers to become healthy (includes model download time)
STARTUP_TIMEOUT = 600  # 10 minutes
_LOG_DIR = _PROJECT_ROOT / "logs"


class ServerManager:
    """Starts, health-checks, and stops local mlx-audio and mlx-vlm servers."""

    def __init__(self, settings: Settings, display: Display):
        self.settings = settings
        self.display = display
        self._processes: list[subprocess.Popen[str]] = []
        self._log_files: list[tuple[str, Path]] = []

    def _parse_port(self, url: str) -> int:
        parsed = urlparse(url)
        return parsed.port or 8000

    def _start_process(self, cmd: list[str], name: str) -> subprocess.Popen[str]:
        self.display.server_starting(name)
        _LOG_DIR.mkdir(exist_ok=True)
        log_path = (
            _LOG_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.log"
        )
        log_file = open(log_path, "w")  # noqa: SIM115
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._processes.append(proc)
        self._log_files.append((name, log_path))
        return proc

    async def _wait_for_health(
        self, url: str, name: str, proc: subprocess.Popen[str]
    ) -> bool:
        """Poll a server's endpoint until it responds or times out."""
        health_url = f"{url}/v1/models"
        elapsed = 0.0
        interval = 2.0
        last_elapsed = -1

        async with httpx.AsyncClient(timeout=5.0) as client:
            while elapsed < STARTUP_TIMEOUT:
                if proc.poll() is not None:
                    log_lines = self._get_log_tail(name)
                    self.display.server_failed(name, log_lines)
                    return False

                try:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        self.display.server_ready_one(name)
                        return True
                except httpx.ConnectError:
                    t = int(elapsed)
                    if t != last_elapsed:
                        self.display.server_waiting(name, t)
                        last_elapsed = t
                except httpx.ReadTimeout:
                    pass

                await asyncio.sleep(interval)
                elapsed += interval

        self.display.server_timeout(name, STARTUP_TIMEOUT)
        return False

    def _load_model_deps(self) -> dict[str, dict]:
        """Load model dependency mapping from model_deps.toml."""
        path = _PROJECT_ROOT / "model_deps.toml"
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            return tomllib.load(f)

    def _deps_for_model(self, model_name: str) -> list[str]:
        """Return extra pip packages needed for a given model."""
        lower = model_name.lower()
        deps: list[str] = []
        for pattern, entry in self._load_model_deps().items():
            if pattern in lower:
                deps.extend(entry.get("deps", []))
        return deps

    def _brew_for_model(self, model_name: str) -> list[str]:
        """Return brew packages needed for a given model."""
        lower = model_name.lower()
        pkgs: list[str] = []
        for pattern, entry in self._load_model_deps().items():
            if pattern in lower:
                pkgs.extend(entry.get("brew", []))
        return pkgs

    def _ensure_system_deps(self) -> bool:
        """Install system-level (brew) dependencies if missing."""
        missing: list[str] = []
        for model in [self.settings.local_tts_model, self.settings.local_stt_model]:
            for pkg in self._brew_for_model(model):
                if not shutil.which(pkg):
                    missing.append(pkg)

        if not missing:
            return True

        if not shutil.which("brew"):
            self.display.server_install_failed(
                [
                    f"Missing system packages: {', '.join(missing)}",
                    "Install them manually (Homebrew not found).",
                ]
            )
            return False

        self.display.server_installing_system(missing)
        result = subprocess.run(
            ["brew", "install", *missing],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.display.server_install_failed(result.stderr.strip().splitlines()[-10:])
            return False

        self.display.server_installed()
        return True

    def _ensure_packages(self) -> bool:
        """Install mlx-audio[server], mlx-vlm, model-specific deps, and system deps."""
        if not self._ensure_system_deps():
            return False

        packages: list[str] = []

        try:
            __import__("mlx_audio")
        except ImportError:
            packages.append("mlx-audio[server,tts,stt]")
        try:
            __import__("mlx_vlm")
        except ImportError:
            packages.append("mlx-vlm")

        for model in [self.settings.local_tts_model, self.settings.local_stt_model]:
            for dep in self._deps_for_model(model):
                import_name = (
                    dep.split("<")[0].split(">")[0].split("=")[0].split("!")[0]
                )
                try:
                    __import__(import_name.replace("-", "_"))
                except ImportError:
                    if dep not in packages:
                        packages.append(dep)

        if not packages:
            return True

        self.display.server_installing(packages)
        result = subprocess.run(
            ["uv", "pip", "install", *packages],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.display.server_install_failed(result.stderr.strip().splitlines()[-10:])
            return False

        self.display.server_installed()
        return True

    def _apply_patches(self) -> None:
        """Fix known compatibility issues in installed packages."""
        try:
            import misaki

            espeak_py = Path(misaki.__file__).parent / "espeak.py"
            if not espeak_py.exists():
                return

            content = espeak_py.read_text()
            patched = content

            for data_candidate in [
                Path("/opt/homebrew/share/espeak-ng-data"),
                Path("/usr/share/espeak-ng-data"),
            ]:
                if data_candidate.exists():
                    sys_data = data_candidate
                    break
            else:
                sys_data = None

            for lib_candidate in [
                Path("/opt/homebrew/lib/libespeak-ng.dylib"),
                Path("/usr/lib/libespeak-ng.so"),
            ]:
                if lib_candidate.exists():
                    sys_lib = lib_candidate
                    break
            else:
                sys_lib = None

            if sys_lib:
                patched = patched.replace(
                    "EspeakWrapper.set_library(espeakng_loader.get_library_path())",
                    f'EspeakWrapper.set_library("{sys_lib}")',
                )

            if sys_data:
                for old in [
                    "EspeakWrapper.set_data_path(espeakng_loader.get_data_path())",
                    "EspeakWrapper.data_path = espeakng_loader.get_data_path()",
                ]:
                    patched = patched.replace(
                        old,
                        f'EspeakWrapper.data_path = "{sys_data}"',
                    )
            elif "set_data_path" in patched:
                patched = patched.replace(
                    "EspeakWrapper.set_data_path(espeakng_loader.get_data_path())",
                    "EspeakWrapper.data_path = espeakng_loader.get_data_path()",
                )

            if patched != content:
                espeak_py.write_text(patched)
                pycache = espeak_py.parent / "__pycache__"
                if pycache.exists():
                    for pyc in pycache.glob("espeak*"):
                        pyc.unlink()
                self.display.server_patched(
                    "Patched misaki/espeak.py for system espeak-ng"
                )
        except Exception:
            pass

    async def start(self) -> bool:
        """Start both servers and wait for them to be healthy. Returns True if ready."""
        s = self.settings
        audio_port = self._parse_port(s.mlx_audio_url)
        vlm_port = self._parse_port(s.mlx_vlm_url)

        self.display.server_setup_start()

        if not self._ensure_packages():
            return False

        self._apply_patches()

        audio_proc = self._start_process(
            [
                sys.executable,
                "-m",
                "mlx_audio.server",
                "--host",
                "0.0.0.0",
                "--port",
                str(audio_port),
            ],
            f"mlx-audio (port {audio_port})",
        )

        vlm_proc = self._start_process(
            [
                sys.executable,
                "-m",
                "mlx_vlm.server",
                "--model",
                s.llm_model,
                "--port",
                str(vlm_port),
            ],
            f"mlx-vlm (port {vlm_port})",
        )

        audio_ok, vlm_ok = await asyncio.gather(
            self._wait_for_health(s.mlx_audio_url, "mlx-audio", audio_proc),
            self._wait_for_health(s.mlx_vlm_url, "mlx-vlm", vlm_proc),
        )

        if audio_ok and vlm_ok:
            self.display.server_all_ready()
            return True

        self.stop()
        return False

    def _get_log_tail(self, name: str, lines: int = 15) -> list[str]:
        for log_name, log_path in self._log_files:
            if name in log_name and log_path.exists():
                return log_path.read_text().strip().splitlines()[-lines:]
        return []

    def get_all_server_logs(self) -> dict[str, list[str]]:
        """Return recent logs from all servers."""
        logs: dict[str, list[str]] = {}
        for name, log_path in self._log_files:
            if log_path.exists():
                logs[name] = log_path.read_text().strip().splitlines()[-15:]
        return logs

    def stop(self) -> None:
        """Terminate all managed server processes."""
        for proc in self._processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in self._processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._processes.clear()
