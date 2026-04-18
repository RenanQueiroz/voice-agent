"""Manages local mlx-audio, whisper-server, and LLM server processes.

The manager is a per-role reconciler: it looks at the active `ModelConfig`
for each role (STT / LLM / TTS) and makes the running processes match.
Roles whose active model is cloud have no process; roles whose active
model is local get a process, restarted when the user picks a different
local model for that role.
"""

from __future__ import annotations

import asyncio
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

import httpx

from .config import ModelConfig, Settings

if TYPE_CHECKING:
    from .display import Display

_PROJECT_ROOT = Path(__file__).parent.parent

# How long to wait for servers to become healthy (includes model download time)
STARTUP_TIMEOUT = 600  # 10 minutes
_LOG_DIR = _PROJECT_ROOT / "logs"

Role = Literal["stt", "llm", "tts"]


class ServerManager:
    """Per-role reconciler for local server processes."""

    def __init__(self, settings: Settings, display: Display):
        self.settings = settings
        self.display = display
        self._procs: dict[Role, subprocess.Popen[str]] = {}
        self._log_files: dict[Role, Path] = {}
        # Identity of the active model that each running process was started
        # for. When the user picks a different local model for the same role,
        # this mismatches and we restart.
        self._started_for: dict[Role, str] = {}
        # Track which local role has had deps installed already this session.
        self._deps_ready: set[Role] = set()

    # ── Public API ────────────────────────────────────────

    async def reconcile(self) -> bool:
        """Bring running processes in line with the current active models.

        Returns True on success, False (and mounts/logs an error) on failure.
        """
        active = self._active_local_roles()

        # Stop processes that are no longer needed or whose model changed.
        for role in list(self._procs):
            wanted = active.get(role)
            if wanted is None or wanted.name != self._started_for.get(role):
                self._stop_role(role)

        # Start any missing processes.
        for role, model in active.items():
            if role in self._procs:
                continue
            ok = await self._start_role(role, model)
            if not ok:
                return False

        # Final health + readiness wait for anything we just started.
        to_wait: list[tuple[Role, ModelConfig]] = [
            (role, active[role])
            for role in active
            if role not in self._started_for  # newly launched
            or self._started_for[role] != active[role].name  # shouldn't happen
        ]
        for role, model in to_wait:
            ok = await self._wait_ready(role, model)
            if not ok:
                return False
            self._started_for[role] = model.name

        if active and all(role in self._procs for role in active):
            self.display.server_all_ready()
        return True

    def get_all_server_logs(self) -> dict[str, list[str]]:
        logs: dict[str, list[str]] = {}
        for role, log_path in self._log_files.items():
            if log_path.exists():
                logs[self._display_name(role)] = (
                    log_path.read_text().strip().splitlines()[-15:]
                )
        return logs

    def stop(self) -> None:
        for role in list(self._procs):
            self._stop_role(role)

    # ── Role starters ─────────────────────────────────────

    async def _start_role(self, role: Role, model: ModelConfig) -> bool:
        """Install deps (once per role) and start the process for this role."""
        if role not in self._deps_ready:
            ok = self._install_deps_for(role, model)
            if not ok:
                return False
            self._deps_ready.add(role)

        self._apply_patches(role, model)

        if role == "stt":
            return self._start_stt(model)
        if role == "llm":
            return self._start_llm(model)
        return self._start_tts(model)

    def _start_stt(self, model: ModelConfig) -> bool:
        port = self._parse_port(self._require_url("stt"))
        whisper_bin = _PROJECT_ROOT / "whispercpp" / "whisper-server"
        model_path = _PROJECT_ROOT / "whispercpp" / "models" / f"ggml-{model.model}.bin"
        vad_model_path = (
            _PROJECT_ROOT / "whispercpp" / "models" / "ggml-silero-v5.1.2.bin"
        )
        cmd = [
            str(whisper_bin),
            "-m",
            str(model_path),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--vad",
            "--vad-model",
            str(vad_model_path),
            "--convert",
        ]
        return self._launch("stt", cmd, f"whisper-server (port {port})")

    def _start_tts(self, model: ModelConfig) -> bool:
        port = self._parse_port(self._require_url("tts"))
        cmd = [
            sys.executable,
            "-m",
            "mlx_audio.server",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            # --workers accepts either an int (exact worker count) or a
            # 0 < float <= 1 (fraction of cores). mlx-audio defaults to 2
            # workers, which leaves most cores idle on Apple Silicon.
            # 1.0 = all cores — max throughput for local TTS.
            "--workers",
            "0.5",
        ]
        return self._launch("tts", cmd, f"mlx-audio (port {port})")

    def _start_llm(self, model: ModelConfig) -> bool:
        port = self._parse_port(self._require_url("llm"))
        server = model.server
        if server == "llamacpp":
            llama_bin = str(_PROJECT_ROOT / "llamacpp" / "llama-server")
            cmd = [
                llama_bin,
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ]
            if model.preset:
                cmd.extend(["--models-preset", str(_PROJECT_ROOT / model.preset)])
            return self._launch("llm", cmd, f"llama-server (port {port})")

        # mlx-vlm / mlx-lm
        module = "mlx_vlm.server" if server == "mlx-vlm" else "mlx_lm.server"
        cmd = [
            sys.executable,
            "-m",
            module,
            "--model",
            model.model,
            "--port",
            str(port),
        ]
        if server == "mlx-vlm":
            if model.kv_bits:
                cmd.extend(["--kv-bits", model.kv_bits])
            if model.kv_quant_scheme:
                cmd.extend(["--kv-quant-scheme", model.kv_quant_scheme])
        return self._launch("llm", cmd, f"{server} (port {port})")

    # ── Process plumbing ──────────────────────────────────

    def _launch(self, role: Role, cmd: list[str], display_name: str) -> bool:
        self.display.server_starting(display_name)
        _LOG_DIR.mkdir(exist_ok=True)
        log_path = (
            _LOG_DIR
            / f"{display_name.replace(' ', '_').replace('(', '').replace(')', '')}.log"
        )
        log_file = open(log_path, "w")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as e:
            self.display.server_failed(display_name, [str(e)])
            return False
        self._procs[role] = proc
        self._log_files[role] = log_path
        return True

    async def _wait_ready(self, role: Role, model: ModelConfig) -> bool:
        url = self._require_url(role)
        display_name = self._display_name(role)
        if role == "stt":
            return await self._wait_for_health(url, display_name, role, "/")
        if role == "llm" and model.server == "llamacpp":
            return await self._wait_for_health(url, display_name, role, "/health")
        return await self._wait_for_health(url, display_name, role, "/v1/models")

    async def _wait_for_health(
        self,
        url: str,
        display_name: str,
        role: Role,
        health_path: str,
    ) -> bool:
        proc = self._procs.get(role)
        if proc is None:
            return False
        health_url = f"{url.rstrip('/')}{health_path}"
        elapsed = 0.0
        interval = 2.0
        last_elapsed = -1

        async with httpx.AsyncClient(timeout=5.0) as client:
            while elapsed < STARTUP_TIMEOUT:
                if proc.poll() is not None:
                    log_lines = self._get_log_tail(role)
                    self.display.server_failed(display_name, log_lines)
                    self._procs.pop(role, None)
                    return False
                try:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        self.display.server_ready_one(display_name)
                        return True
                except httpx.ConnectError:
                    t = int(elapsed)
                    if t != last_elapsed:
                        self.display.server_waiting(display_name, t)
                        last_elapsed = t
                except httpx.ReadTimeout:
                    pass
                await asyncio.sleep(interval)
                elapsed += interval

        self.display.server_timeout(display_name, STARTUP_TIMEOUT)
        return False

    def _stop_role(self, role: Role) -> None:
        proc = self._procs.pop(role, None)
        self._log_files.pop(role, None)
        self._started_for.pop(role, None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ── Helpers ───────────────────────────────────────────

    def _active_local_roles(self) -> dict[Role, ModelConfig]:
        result: dict[Role, ModelConfig] = {}
        if self.settings.stt.provider == "local":
            result["stt"] = self.settings.stt
        if self.settings.llm.provider == "local":
            result["llm"] = self.settings.llm
        if self.settings.tts.provider == "local":
            result["tts"] = self.settings.tts
        return result

    def _require_url(self, role: Role) -> str:
        url = {
            "stt": self.settings.stt_url,
            "tts": self.settings.tts_url,
            "llm": self.settings.llm_url,
        }[role]
        if not url:
            raise RuntimeError(f"Missing {role}_url in [local] section of config.toml")
        return url

    @staticmethod
    def _parse_port(url: str) -> int:
        parsed = urlparse(url)
        return parsed.port or 8000

    def _display_name(self, role: Role) -> str:
        return {"stt": "whisper-server", "tts": "mlx-audio", "llm": "llm-server"}[role]

    def _get_log_tail(self, role: Role, lines: int = 15) -> list[str]:
        path = self._log_files.get(role)
        if path and path.exists():
            return path.read_text().strip().splitlines()[-lines:]
        return []

    # ── Dependency installers ─────────────────────────────

    def _install_deps_for(self, role: Role, model: ModelConfig) -> bool:
        if role == "stt":
            return self._ensure_whisper(model)
        if role == "llm":
            return self._ensure_llm(model)
        return self._ensure_tts(model)

    def _load_model_deps(self) -> dict[str, dict]:
        path = _PROJECT_ROOT / "model_deps.toml"
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            return tomllib.load(f)

    def _deps_for_model(self, model_name: str) -> list[str]:
        lower = model_name.lower()
        deps: list[str] = []
        for pattern, entry in self._load_model_deps().items():
            if pattern in lower:
                deps.extend(entry.get("deps", []))
        return deps

    def _brew_for_model(self, model_name: str) -> list[str]:
        lower = model_name.lower()
        pkgs: list[str] = []
        for pattern, entry in self._load_model_deps().items():
            if pattern in lower:
                pkgs.extend(entry.get("brew", []))
        return pkgs

    def _ensure_system_deps(self, model_name: str) -> bool:
        missing: list[str] = []
        for pkg in self._brew_for_model(model_name):
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
            ["brew", "install", *missing], capture_output=True, text=True
        )
        if result.returncode != 0:
            self.display.server_install_failed(result.stderr.strip().splitlines()[-10:])
            return False
        self.display.server_installed()
        return True

    def _ensure_tts(self, model: ModelConfig) -> bool:
        if not self._ensure_system_deps(model.model):
            return False
        packages: list[str] = []
        try:
            __import__("mlx_audio")
        except ImportError:
            packages.append("mlx-audio[server,tts]")
        for dep in self._deps_for_model(model.model):
            import_name = dep.split("<")[0].split(">")[0].split("=")[0].split("!")[0]
            try:
                __import__(import_name.replace("-", "_"))
            except ImportError:
                if dep not in packages:
                    packages.append(dep)
        if not packages:
            return True
        self.display.server_installing(packages)
        result = subprocess.run(
            ["uv", "pip", "install", *packages], capture_output=True, text=True
        )
        if result.returncode != 0:
            self.display.server_install_failed(result.stderr.strip().splitlines()[-10:])
            return False
        self.display.server_installed()
        return True

    def _ensure_llm(self, model: ModelConfig) -> bool:
        if model.server == "llamacpp":
            return self._ensure_llamacpp()
        llm_import = "mlx_vlm" if model.server == "mlx-vlm" else "mlx_lm"
        llm_pip = "mlx-vlm" if model.server == "mlx-vlm" else "mlx-lm"
        try:
            __import__(llm_import)
            return True
        except ImportError:
            pass
        self.display.server_installing([llm_pip])
        result = subprocess.run(
            ["uv", "pip", "install", llm_pip], capture_output=True, text=True
        )
        if result.returncode != 0:
            self.display.server_install_failed(result.stderr.strip().splitlines()[-10:])
            return False
        self.display.server_installed()
        return True

    def _ensure_llamacpp(self) -> bool:
        llama_bin = _PROJECT_ROOT / "llamacpp" / "llama-server"
        setup_script = _PROJECT_ROOT / "setup-llamacpp.sh"
        if llama_bin.exists():
            return True
        if not setup_script.exists():
            self.display.server_install_failed(
                [f"Setup script not found: {setup_script}"]
            )
            return False
        self.display.server_installing(["llamacpp"])
        result = subprocess.run(
            ["bash", str(setup_script)], capture_output=True, text=True
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

    def _ensure_whisper(self, model: ModelConfig) -> bool:
        whisper_bin = _PROJECT_ROOT / "whispercpp" / "whisper-server"
        model_file = _PROJECT_ROOT / "whispercpp" / "models" / f"ggml-{model.model}.bin"
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
            ["bash", str(setup_script), model.model],
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

    def _apply_patches(self, role: Role, model: ModelConfig) -> None:
        """Fix known compatibility issues (misaki/phonemizer for Kokoro)."""
        if role != "tts" or "kokoro" not in model.model.lower():
            return
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
