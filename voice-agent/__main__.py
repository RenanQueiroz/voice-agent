import asyncio

from .audio import _restore_terminal
from .config import load_settings
from .display import Display
from .pipeline import run


def main() -> None:
    settings = load_settings()
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        Display().goodbye()
    finally:
        _restore_terminal()


if __name__ == "__main__":
    main()
