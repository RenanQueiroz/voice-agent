import sys

from .platform_info import current_os


def main() -> None:
    # Windows isn't supported directly — too many components (whisper.cpp /
    # llama.cpp setup scripts, mlx stack, Silero ONNX sounddevice path,
    # Textual terminal handling) are tuned for POSIX. WSL2 works for the
    # Linux path, so point Windows users there instead of letting them
    # hit a confusing mid-startup failure.
    if current_os() == "windows":
        print(
            "voice-agent does not support running directly on Windows.\n"
            "Please run it through WSL2 instead:\n"
            "  1. Install WSL2 and a Linux distribution "
            "(https://learn.microsoft.com/windows/wsl/install)\n"
            "  2. Clone this repo inside your WSL2 home directory.\n"
            "  3. Run ./setup.sh followed by 'uv run python -m voice-agent' "
            "from the WSL2 shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Lazy-import the rest so the Windows path above doesn't drag in
    # config parsing / Textual / dependencies.
    from .app import VoiceAgentApp
    from .config import load_settings

    settings = load_settings()
    VoiceAgentApp(settings).run()


if __name__ == "__main__":
    main()
