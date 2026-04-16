"""Manages local mlx-audio, whisper-server, and LLM server processes."""

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
    """Starts, health-checks, and stops local mlx-audio, whisper, and LLM servers."""

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
        self,
        url: str,
        name: str,
        proc: subprocess.Popen[str],
        health_path: str = "/v1/models",
    ) -> bool:
        """Poll a server's endpoint until it responds or times out."""
        health_url = f"{url}{health_path}"
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
        for model in [self.settings.tts_model]:
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
            packages.append("mlx-audio[server,tts]")

        # Install the right MLX LLM server based on config (llamacpp is a binary, not pip)
        llm_server = self.settings.mlx_llm_server or "mlx-vlm"
        if llm_server != "llamacpp":
            llm_import = "mlx_vlm" if llm_server == "mlx-vlm" else "mlx_lm"
            llm_pip = "mlx-vlm" if llm_server == "mlx-vlm" else "mlx-lm"
            try:
                __import__(llm_import)
            except ImportError:
                packages.append(llm_pip)

        for model in [self.settings.tts_model]:
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

    def _ensure_llamacpp(self) -> bool:
        """Run setup-llamacpp.sh to ensure the llama-server binary is available."""
        llama_bin = _PROJECT_ROOT / "llamacpp" / "llama-server"
        setup_script = _PROJECT_ROOT / "setup-llamacpp.sh"

        if not setup_script.exists():
            self.display.server_install_failed(
                [f"Setup script not found: {setup_script}"]
            )
            return False

        self.display.server_installing(["llamacpp"])
        result = subprocess.run(
            ["bash", str(setup_script)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.display.server_install_failed(result.stderr.strip().splitlines()[-10:])
            return False

        if not llama_bin.exists():
            self.display.server_install_failed(
                [
                    "llama-server binary not found after setup.",
                    f"Expected at: {llama_bin}",
                ]
            )
            return False

        self.display.server_installed()
        return True

    def _ensure_whisper(self) -> bool:
        """Run setup-whispercpp.sh to ensure whisper-server and models are available."""
        whisper_bin = _PROJECT_ROOT / "whispercpp" / "whisper-server"
        stt_model = self.settings.stt_model
        model_file = _PROJECT_ROOT / "whispercpp" / "models" / f"ggml-{stt_model}.bin"
        setup_script = _PROJECT_ROOT / "setup-whispercpp.sh"

        vad_onnx = _PROJECT_ROOT / "whispercpp" / "models" / "silero_vad.onnx"
        if whisper_bin.exists() and model_file.exists() and vad_onnx.exists():
            return True

        if not setup_script.exists():
            self.display.server_install_failed(
                [f"Setup script not found: {setup_script}"]
            )
            return False

        self.display.server_installing(["whisper.cpp"])
        result = subprocess.run(
            ["bash", str(setup_script), stt_model],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.display.server_install_failed(result.stderr.strip().splitlines()[-10:])
            return False

        if not whisper_bin.exists():
            self.display.server_install_failed(
                [
                    "whisper-server binary not found after setup.",
                    f"Expected at: {whisper_bin}",
                ]
            )
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
        """Start all servers and wait for them to be healthy. Returns True if ready."""
        s = self.settings
        assert s.mlx_audio_url is not None  # validated in load_settings
        assert s.stt_url is not None
        assert s.mlx_llm_url is not None
        assert s.mlx_llm_server is not None
        audio_url = s.mlx_audio_url
        stt_url = s.stt_url
        llm_url = s.mlx_llm_url
        llm_server = s.mlx_llm_server
        audio_port = self._parse_port(audio_url)
        stt_port = self._parse_port(stt_url)
        llm_port = self._parse_port(llm_url)

        self.display.server_setup_start()

        if not self._ensure_packages():
            return False

        self._apply_patches()

        if not self._ensure_whisper():
            return False

        # Start TTS server (mlx-audio)
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

        # Start STT server (whisper.cpp with VAD)
        whisper_bin = str(_PROJECT_ROOT / "whispercpp" / "whisper-server")
        model_path = str(
            _PROJECT_ROOT / "whispercpp" / "models" / f"ggml-{s.stt_model}.bin"
        )
        vad_model_path = str(
            _PROJECT_ROOT / "whispercpp" / "models" / "ggml-silero-v5.1.2.bin"
        )
        stt_cmd = [
            whisper_bin,
            "-m",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(stt_port),
            "--vad",
            "--vad-model",
            vad_model_path,
            "--convert",
        ]

        stt_proc = self._start_process(
            stt_cmd,
            f"whisper-server (port {stt_port})",
        )

        # Build LLM server command based on backend
        if llm_server == "llamacpp":
            if not self._ensure_llamacpp():
                return False
            llama_bin = str(_PROJECT_ROOT / "llamacpp" / "llama-server")
            llm_cmd = [
                llama_bin,
                "--host",
                "0.0.0.0",
                "--port",
                str(llm_port),
            ]
            if s.llamacpp_preset:
                preset_path = str(_PROJECT_ROOT / s.llamacpp_preset)
                llm_cmd.extend(["--models-preset", preset_path])
            llm_health_path = "/health"
        else:
            llm_module = (
                "mlx_vlm.server" if llm_server == "mlx-vlm" else "mlx_lm.server"
            )
            llm_cmd = [
                sys.executable,
                "-m",
                llm_module,
                "--model",
                s.llm_model,
                "--port",
                str(llm_port),
            ]
            if llm_server == "mlx-vlm":
                if s.mlx_kv_bits:
                    llm_cmd.extend(["--kv-bits", s.mlx_kv_bits])
                if s.mlx_kv_quant_scheme:
                    llm_cmd.extend(["--kv-quant-scheme", s.mlx_kv_quant_scheme])
            llm_health_path = "/v1/models"

        llm_proc = self._start_process(
            llm_cmd,
            f"{llm_server} (port {llm_port})",
        )

        audio_ok, stt_ok, llm_ok = await asyncio.gather(
            self._wait_for_health(audio_url, "mlx-audio", audio_proc),
            self._wait_for_health(stt_url, "whisper-server", stt_proc, health_path="/"),
            self._wait_for_health(
                llm_url, llm_server, llm_proc, health_path=llm_health_path
            ),
        )

        if audio_ok and stt_ok and llm_ok:
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
