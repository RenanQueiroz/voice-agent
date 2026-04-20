"""Manages local mlx-audio, whisper-server, and LLM server processes.

The manager is a per-role reconciler: it looks at the active `ModelConfig`
for each role (STT / LLM / TTS) and makes the running processes match.
Roles whose active model is cloud have no process; roles whose active
model is local get a process, restarted when the user picks a different
local model for that role.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
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
from .platform_info import current_os, linux_package_manager
from .runtimes import RUNTIMES, Runtime, get_runtime


# Linux prctl constant; SIGKILL when parent dies. See prctl(2).
_PR_SET_PDEATHSIG = 1


def _subprocess_setup_linux() -> None:
    """Run in the child right after fork, before exec.

    Sets up two kernel-level cleanup guarantees so that aggressive
    user signals (e.g. Ctrl+C chains, hard crashes) don't leave
    orphan subprocesses holding VRAM and ports:

    1. `setsid()` — put the child in its own session + process group,
       so we can later `killpg(pid, SIGKILL)` and take out the whole
       subtree (main server + every inductor compile worker it spawns)
       in one syscall.
    2. `prctl(PR_SET_PDEATHSIG, SIGKILL)` — tell the kernel to send
       SIGKILL to this process if our parent process dies. Catches the
       case where our Python process is hard-killed without running
       its own cleanup handlers. Linux-only; on other kernels the
       setsid part still helps with normal shutdown.
    """
    os.setsid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        # If prctl isn't available (musl, non-Linux libc, etc.) we
        # still have setsid + our own graceful killpg on shutdown.
        pass


def _subprocess_setup_darwin() -> None:
    """Same as the linux variant but without pdeathsig (doesn't exist
    on darwin). setsid alone still enables killpg-based cleanup."""
    os.setsid()


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

        # Reap any orphan servers left over from a previous app session
        # that didn't shut down cleanly (e.g. user mashed Ctrl+C and our
        # cleanup handlers got skipped). Without this, reconcile() would
        # fail with "port already in use" AND the orphans would keep
        # holding VRAM / file handles until manually killed.
        self._reap_orphan_servers()

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
        runtime = model.runtime
        if runtime == "kokoro-fastapi":
            return self._start_kokoro_fastapi(port)
        if runtime == "qwen3-tts":
            return self._start_qwen3_tts(model, port)
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

    def _start_kokoro_fastapi(self, port: int) -> bool:
        repo = _PROJECT_ROOT / "kokoro-fastapi"
        py = repo / ".venv" / "bin" / "python"
        env = {
            **os.environ,
            "USE_GPU": "true",
            "USE_ONNX": "false",
            # The server reads source from api/src/main.py; it expects
            # both the repo root and the `api/` package on PYTHONPATH.
            "PYTHONPATH": f"{repo}:{repo / 'api'}",
            "MODEL_DIR": "src/models",
            "VOICES_DIR": "src/voices/v1_0",
            # phonemizer looks for espeak-ng under these paths at import time.
            "PHONEMIZER_ESPEAK_DATA": "/usr/share/espeak-ng-data",
            "PHONEMIZER_ESPEAK_PATH": "/usr/bin",
            "ESPEAK_DATA_PATH": "/usr/share/espeak-ng-data",
        }
        cmd = [
            str(py),
            "-m",
            "uvicorn",
            "api.src.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
        return self._launch(
            "tts", cmd, f"kokoro-fastapi (port {port})", cwd=repo, env=env
        )

    def _start_qwen3_tts(self, model: ModelConfig, port: int) -> bool:
        repo = _PROJECT_ROOT / "qwen3-tts"
        py = repo / ".venv" / "bin" / "python"
        cache = _PROJECT_ROOT / ".cache" / "qwen3-tts"
        cache.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            # Optimized backend: torch.compile + CUDA graphs + flash-attn +
            # token-by-token PCM streaming. Configured via config.yaml in the
            # repo; setup-qwen3-tts.sh seeds it from the upstream default.
            "TTS_BACKEND": "optimized",
            "TTS_CONFIG": str(repo / "config.yaml"),
            "HOST": "0.0.0.0",
            "PORT": str(port),
            # Persist torch.compile artifacts so subsequent boots skip the
            # ~75s compile cost. Same rationale for HF cache.
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "HF_HOME": str(cache / "hf"),
            # Voice library: profiles seeded by setup-qwen3-tts.sh at
            # <repo>/voice_library/profiles/<Name>/. Needed when the active
            # catalog entry uses voice = "clone:<Name>"; the openai_compatible
            # router reads this env var to find meta.json + ref audio.
            "VOICE_LIBRARY_DIR": str(repo / "voice_library"),
        }
        # Qwen3's optimized backend loads ONE model variant at boot (the
        # CustomVoice line for preset voices, or the Base line for voice
        # cloning — they share the encoder/decoder but differ in the
        # speaker-embedding head). Select per catalog entry via env var
        # (setup-qwen3-tts.sh patches _default_model_key to honor it).
        # Mismatch symptom: clone:<Name> voice on a CustomVoice model
        # returns "voice_cloning_not_supported" 400; a preset voice
        # (Sohee, Vivian, …) on a Base model returns "unknown voice" 400.
        # Catalog validation enforces pairing (clone: + -Base, preset +
        # -CustomVoice) at config load, so only a direct API misuse hits
        # this server-side.
        if model.model_variant:
            env["QWEN3_DEFAULT_MODEL"] = model.model_variant
        cmd = [str(py), "-m", "api.main"]
        return self._launch("tts", cmd, f"qwen3-tts (port {port})", cwd=repo, env=env)

    def _start_llm(self, model: ModelConfig) -> bool:
        port = self._parse_port(self._require_url("llm"))
        runtime = model.runtime
        if runtime == "llamacpp":
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
        module = "mlx_vlm.server" if runtime == "mlx-vlm" else "mlx_lm.server"
        cmd = [
            sys.executable,
            "-m",
            module,
            "--model",
            model.model,
            "--port",
            str(port),
        ]
        if runtime == "mlx-vlm":
            if model.kv_bits:
                cmd.extend(["--kv-bits", model.kv_bits])
            if model.kv_quant_scheme:
                cmd.extend(["--kv-quant-scheme", model.kv_quant_scheme])
        return self._launch("llm", cmd, f"{runtime} (port {port})")

    # ── Process plumbing ──────────────────────────────────

    def _launch(
        self,
        role: Role,
        cmd: list[str],
        display_name: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> bool:
        _LOG_DIR.mkdir(exist_ok=True)
        log_path = (
            _LOG_DIR
            / f"{display_name.replace(' ', '_').replace('(', '').replace(')', '')}.log"
        )
        # Announce with the log path so the splash can attach a "click to
        # view output" affordance to this server's row — lets the user
        # watch long startups (e.g. Qwen3's HF download + torch.compile)
        # instead of staring at a stalled spinner.
        self.display.server_starting(display_name, log_path)
        log_file = open(log_path, "w")  # noqa: SIM115
        # preexec_fn isolates the child in its own process group (via
        # setsid) AND — on Linux — registers PR_SET_PDEATHSIG=SIGKILL
        # so the kernel reaps it even if our Python process is hard-
        # killed. See _subprocess_setup_linux docstring.
        preexec: object | None
        if sys.platform.startswith("linux"):
            preexec = _subprocess_setup_linux
        elif sys.platform == "darwin":
            preexec = _subprocess_setup_darwin
        else:
            preexec = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(cwd) if cwd else None,
                env=env,
                preexec_fn=preexec,  # type: ignore[arg-type]
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
        # Runtime decides both the health path and when a response
        # counts as "ready" (Qwen3 reports 200 during torch.compile
        # warmup — its ready_check inspects the body).
        runtime = RUNTIMES.get(model.runtime or "")
        return await self._wait_for_health(url, display_name, role, runtime)

    async def _wait_for_health(
        self,
        url: str,
        display_name: str,
        role: Role,
        runtime: Runtime | None,
    ) -> bool:
        proc = self._procs.get(role)
        if proc is None:
            return False
        health_path = runtime.health_path if runtime else "/"
        ready_check = (
            runtime.ready_check if runtime else (lambda r: r.status_code == 200)
        )
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
                    try:
                        ready = ready_check(resp)
                    except Exception:
                        # Malformed JSON mid-warmup, etc. — not fatal; keep
                        # polling. Worst case we time out and the user sees
                        # a server_timeout log tail.
                        ready = False
                    if ready:
                        self.display.server_ready_one(display_name)
                        return True
                    # HTTP responded but ready_check said no (e.g. Qwen3
                    # "initializing" during warmup). Surface elapsed as a
                    # waiting heartbeat so the splash doesn't look frozen.
                    t = int(elapsed)
                    if t != last_elapsed:
                        self.display.server_waiting(display_name, t)
                        last_elapsed = t
                except httpx.ConnectError, httpx.ConnectTimeout:
                    # ConnectError: TCP refused (not listening yet).
                    # ConnectTimeout: SYN got no answer (process running but
                    # hasn't bound to the port — e.g. whisper-server stuck
                    # because ffmpeg is missing). Both mean "not ready yet";
                    # let the outer STARTUP_TIMEOUT bound the wait instead of
                    # letting the exception kill the worker.
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

    def _reap_orphan_servers(self) -> None:
        """Kill any subprocesses left behind by a previous app session.

        Matches processes by the first argument of their cmdline — we
        only terminate things whose entrypoint binary lives under one
        of the vendored server directories we control. This is
        deliberately narrow: a user's own un-related uvicorn / python
        process has no reason to launch from `./qwen3-tts/.venv/bin/`
        etc., so matching on that path means no collateral damage.

        Runs once per ServerManager (so once per app launch), sync, on
        Linux and macOS. `/proc` scan is the only reliable source on
        Linux; we fall back to `pgrep` on macOS.
        """
        signatures = [
            str(_PROJECT_ROOT / "kokoro-fastapi" / ".venv" / "bin" / "python"),
            str(_PROJECT_ROOT / "qwen3-tts" / ".venv" / "bin" / "python"),
            str(_PROJECT_ROOT / "whispercpp" / "whisper-server"),
            str(_PROJECT_ROOT / "llamacpp" / "llama-server"),
        ]
        my_pid = os.getpid()
        orphans: list[int] = []

        proc_dir = Path("/proc")
        if proc_dir.exists():
            # Linux path: read /proc/*/cmdline
            for entry in proc_dir.iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid == my_pid:
                    continue
                try:
                    raw = (entry / "cmdline").read_bytes()
                except OSError, PermissionError:
                    continue
                # cmdline is null-separated argv; we only care about argv[0]
                argv0 = raw.split(b"\x00", 1)[0].decode(errors="replace")
                if any(argv0.startswith(sig) for sig in signatures):
                    orphans.append(pid)
        else:
            # Fallback: use pgrep -f with each signature
            for sig in signatures:
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", sig],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                except FileNotFoundError, subprocess.TimeoutExpired:
                    return  # no pgrep available; give up
                for line in result.stdout.strip().splitlines():
                    try:
                        pid = int(line.strip())
                    except ValueError:
                        continue
                    if pid != my_pid:
                        orphans.append(pid)

        if not orphans:
            return

        # Send SIGTERM to the whole process group (each orphan should be
        # a session leader from its own _launch). SIGKILL stragglers
        # after a short grace period.
        for pid in orphans:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError, PermissionError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError, PermissionError:
                    pass
        # Short grace period, then SIGKILL any survivors.
        import time

        time.sleep(1.0)
        for pid in orphans:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError, PermissionError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError, PermissionError:
                    pass

    def _stop_role(self, role: Role) -> None:
        proc = self._procs.pop(role, None)
        self._log_files.pop(role, None)
        self._started_for.pop(role, None)
        if proc is None:
            return
        if proc.poll() is None:
            # Because _launch uses preexec_fn=os.setsid, the subprocess
            # is a session leader with pgid == pid and every descendant
            # (e.g. the 30+ torch inductor compile workers qwen spawns)
            # lives in the same session. Signal the whole group so we
            # don't leave VRAM-holding children behind.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError, PermissionError:
                # Fall back to the direct-proc signal if the group is
                # already gone or we can't signal it.
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError, PermissionError:
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
        if role == "tts":
            runtime = (
                self.settings.tts.runtime
                if self.settings.tts.provider == "local"
                else None
            )
            if runtime:
                return runtime
            return "mlx-audio"
        return {"stt": "whisper-server", "llm": "llm-server"}[role]

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

    def _system_packages_for_model(self, model_name: str, manager: str) -> list[str]:
        """Packages to install for this model under the given OS package
        manager (brew/apt/dnf/pacman/zypper), read from model_deps.toml."""
        lower = model_name.lower()
        pkgs: list[str] = []
        for pattern, entry in self._load_model_deps().items():
            if pattern not in lower:
                continue
            # New shape: [pattern.system] nested table keyed by manager.
            system = entry.get("system")
            if isinstance(system, dict):
                pkgs.extend(system.get(manager, []))
                continue
            # Back-compat with the original flat `brew = [...]` shape. Harmless
            # to keep; lets a user's un-migrated model_deps.toml still work on
            # macOS while they update it.
            if manager == "brew":
                pkgs.extend(entry.get("brew", []))
        return pkgs

    def _install_command(self, manager: str, packages: list[str]) -> list[str]:
        """Build the install command for the detected OS package manager.
        Linux commands use sudo — `shutil.which("sudo")` is implied; on
        hosts without sudo the command fails loudly, which is the right
        signal."""
        if manager == "brew":
            return ["brew", "install", *packages]
        if manager == "apt":
            return ["sudo", "apt-get", "install", "-y", *packages]
        if manager == "dnf":
            return ["sudo", "dnf", "install", "-y", *packages]
        if manager == "pacman":
            return ["sudo", "pacman", "-S", "--noconfirm", *packages]
        if manager == "zypper":
            return ["sudo", "zypper", "install", "-y", *packages]
        # Unknown manager — caller should have filtered this out already.
        return []

    def _ensure_system_deps(self, model_name: str) -> bool:
        os_tag = current_os()
        manager = "brew" if os_tag == "darwin" else linux_package_manager()
        if manager is None:
            # No recognized package manager. If model_deps has nothing for us
            # either, nothing to do; otherwise fail loudly.
            any_requested = any(
                self._system_packages_for_model(model_name, m)
                for m in ("brew", "apt", "dnf", "pacman", "zypper")
            )
            if not any_requested:
                return True
            self.display.server_install_failed(
                [
                    "System packages are required, but no supported package "
                    "manager was detected on this system.",
                    "Install manually (brew/apt/dnf/pacman/zypper).",
                ]
            )
            return False

        requested = self._system_packages_for_model(model_name, manager)
        missing = [pkg for pkg in requested if not shutil.which(pkg)]
        if not missing:
            return True
        if manager == "brew" and not shutil.which("brew"):
            self.display.server_install_failed(
                [
                    f"Missing system packages: {', '.join(missing)}",
                    "Install them manually (Homebrew not found).",
                ]
            )
            return False
        install_cmd = self._install_command(manager, missing)
        if not install_cmd:
            self.display.server_install_failed(
                [
                    f"Missing system packages: {', '.join(missing)}",
                    f"No install command wired up for package manager '{manager}'.",
                ]
            )
            return False
        self.display.server_installing_system(missing)
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.display.server_install_failed(result.stderr.strip().splitlines()[-10:])
            return False
        self.display.server_installed()
        return True

    def _ensure_tts(self, model: ModelConfig) -> bool:
        if not self._ensure_system_deps(model.model):
            return False
        # Source-installed Linux TTS runtimes: each has its own repo +
        # venv driven by a setup script. Skip the pip-install dance.
        if model.runtime == "kokoro-fastapi":
            return self._ensure_setup_script(
                "setup-kokoro-fastapi.sh",
                _PROJECT_ROOT / "kokoro-fastapi" / ".venv" / "bin" / "python",
                "kokoro-fastapi",
            )
        if model.runtime == "qwen3-tts":
            return self._ensure_setup_script(
                "setup-qwen3-tts.sh",
                _PROJECT_ROOT / "qwen3-tts" / ".venv" / "bin" / "python",
                "qwen3-tts",
            )
        packages: list[str] = []
        runtime = get_runtime(model.runtime) if model.runtime else None
        if runtime and runtime.pip_module and runtime.pip_package:
            try:
                __import__(runtime.pip_module)
            except ImportError:
                packages.append(runtime.pip_package)
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

    def _ensure_setup_script(
        self,
        script_name: str,
        sentinel: Path,
        display_label: str,
    ) -> bool:
        """Run a setup-<runtime>.sh script once, guarded by a sentinel path.

        Shared by llamacpp / kokoro-fastapi / qwen3-tts. The script is
        expected to be idempotent — the sentinel check is a fast-path to
        skip running it entirely when the install is already present.
        Script output streams to the splash via `server_install_failed`
        on failure (last 10 lines of stderr).
        """
        setup_script = _PROJECT_ROOT / script_name
        if sentinel.exists():
            return True
        if not setup_script.exists():
            self.display.server_install_failed(
                [f"Setup script not found: {setup_script}"]
            )
            return False
        self.display.server_installing([display_label])
        result = subprocess.run(
            ["bash", str(setup_script)], capture_output=True, text=True
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()[-10:]
            self.display.server_install_failed(tail)
            return False
        if not sentinel.exists():
            self.display.server_install_failed(
                [
                    f"{display_label} install reported success but sentinel "
                    f"is missing at: {sentinel}",
                ]
            )
            return False
        self.display.server_installed()
        return True

    def _ensure_llm(self, model: ModelConfig) -> bool:
        if model.runtime == "llamacpp":
            return self._ensure_llamacpp()
        runtime = get_runtime(model.runtime) if model.runtime else None
        if not runtime or not runtime.pip_module or not runtime.pip_package:
            return True
        try:
            __import__(runtime.pip_module)
            return True
        except ImportError:
            pass
        self.display.server_installing([runtime.pip_package])
        result = subprocess.run(
            ["uv", "pip", "install", runtime.pip_package],
            capture_output=True,
            text=True,
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
            # misaki isn't a project dep — it's pip-installed lazily by
            # _ensure_tts when Kokoro is the active TTS, so pyright can't
            # see it in static analysis.
            import misaki  # pyright: ignore[reportMissingImports]

            # Namespace packages have __file__ = None. Bail out quietly
            # rather than trying to patch a package with no on-disk module.
            if misaki.__file__ is None:
                return
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
