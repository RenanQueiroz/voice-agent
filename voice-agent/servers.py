"""Manages local mlx-audio and mlx-vlm server processes."""

from __future__ import annotations

import asyncio
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import Settings

_PROJECT_ROOT = Path(__file__).parent.parent

# How long to wait for servers to become healthy (includes model download time)
STARTUP_TIMEOUT = 600  # 10 minutes
_LOG_DIR = _PROJECT_ROOT / "logs"


class ServerManager:
    """Starts, health-checks, and stops local mlx-audio and mlx-vlm servers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._processes: list[subprocess.Popen[str]] = []
        self._log_files: list[tuple[str, Path]] = []

    def _parse_port(self, url: str) -> int:
        parsed = urlparse(url)
        return parsed.port or 8000

    def _start_process(self, cmd: list[str], name: str) -> subprocess.Popen[str]:
        print(f"  Starting {name}...")
        _LOG_DIR.mkdir(exist_ok=True)
        log_path = (
            _LOG_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.log"
        )
        log_file = open(log_path, "w")
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
        last_msg = ""

        async with httpx.AsyncClient(timeout=5.0) as client:
            while elapsed < STARTUP_TIMEOUT:
                # Check if the process died
                if proc.poll() is not None:
                    print(f"\n  {name} exited unexpectedly (code {proc.returncode}):")
                    self._print_log_tail(name)
                    return False

                try:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        return True
                except httpx.ConnectError:
                    msg = f"  Waiting for {name}... ({int(elapsed)}s)"
                    if msg != last_msg:
                        print(msg)
                        last_msg = msg
                except httpx.ReadTimeout:
                    pass

                await asyncio.sleep(interval)
                elapsed += interval

        print(f"\n  {name} did not become ready within {STARTUP_TIMEOUT}s")
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
            print(f"  Missing system packages: {', '.join(missing)}")
            print("  Install them manually (Homebrew not found).")
            return False

        print(f"  Installing system packages: {', '.join(missing)}...")
        result = subprocess.run(
            ["brew", "install", *missing],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("  Failed to install system packages:")
            for line in result.stderr.strip().splitlines()[-10:]:
                print(f"    {line}")
            return False

        print("  System packages installed.")
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

        # Model-specific dependencies
        for model in [self.settings.local_tts_model, self.settings.local_stt_model]:
            for dep in self._deps_for_model(model):
                # Strip version specifiers for the import check (e.g. "misaki<0.9.4" -> "misaki")
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

        print(f"  Installing {', '.join(packages)}...")
        result = subprocess.run(
            ["uv", "pip", "install", *packages],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("  Failed to install packages:")
            for line in result.stderr.strip().splitlines()[-10:]:
                print(f"    {line}")
            return False

        print("  Installed successfully.")
        self._apply_patches()
        print()
        return True

    def _apply_patches(self) -> None:
        """Fix known compatibility issues in installed packages."""
        # See: https://github.com/Blaizzy/mlx-audio/issues/648
        # Issues in misaki/espeak.py:
        # 1. EspeakWrapper.set_data_path() removed in phonemizer 3.3
        # 2. espeakng_loader.get_data_path() returns broken CI build path
        # 3. espeakng_loader.get_library_path() loads a dylib with hardcoded
        #    broken paths -- must use system espeak-ng instead
        try:
            import misaki

            espeak_py = Path(misaki.__file__).parent / "espeak.py"
            if not espeak_py.exists():
                return

            content = espeak_py.read_text()
            patched = content

            # Find system espeak-ng paths (brew on macOS, standard on Linux)
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

            # Patch the library path (broken dylib from espeakng_loader pip package)
            if sys_lib:
                patched = patched.replace(
                    "EspeakWrapper.set_library(espeakng_loader.get_library_path())",
                    f'EspeakWrapper.set_library("{sys_lib}")',
                )

            # Patch the data path (broken CI build path from espeakng_loader)
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
                # Clear stale bytecode cache so the server picks up the patch
                pycache = espeak_py.parent / "__pycache__"
                if pycache.exists():
                    for pyc in pycache.glob("espeak*"):
                        pyc.unlink()
                print("  Patched misaki/espeak.py for system espeak-ng compatibility.")
        except Exception:
            pass

    async def start(self) -> bool:
        """Start both servers and wait for them to be healthy. Returns True if ready."""
        s = self.settings
        audio_port = self._parse_port(s.mlx_audio_url)
        vlm_port = self._parse_port(s.mlx_vlm_url)

        print("Setting up local servers...\n")

        if not self._ensure_packages():
            return False

        self._apply_patches()
        print("  Starting servers (first run may download models)...\n")

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

        # Wait for both in parallel
        audio_ok, vlm_ok = await asyncio.gather(
            self._wait_for_health(s.mlx_audio_url, "mlx-audio", audio_proc),
            self._wait_for_health(s.mlx_vlm_url, "mlx-vlm", vlm_proc),
        )

        if audio_ok and vlm_ok:
            print("\n  All servers ready.\n")
            return True

        self.stop()
        return False

    def _print_log_tail(self, name: str, lines: int = 15) -> None:
        for log_name, log_path in self._log_files:
            if name in log_name and log_path.exists():
                tail = log_path.read_text().strip().splitlines()[-lines:]
                for line in tail:
                    print(f"    {line}")
                break

    def print_server_logs(self) -> None:
        """Print recent server logs -- useful when debugging API errors."""
        for name, log_path in self._log_files:
            if log_path.exists():
                print(f"\n  -- {name} logs (last 15 lines):")
                self._print_log_tail(name)

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
