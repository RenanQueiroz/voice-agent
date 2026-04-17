from .app import VoiceAgentApp
from .config import load_settings


def main() -> None:
    settings = load_settings()
    VoiceAgentApp(settings).run()


if __name__ == "__main__":
    main()
