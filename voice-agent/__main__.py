import asyncio

from .audio import _restore_terminal
from .config import load_settings
from .pipeline import run


def main() -> None:
    settings = load_settings()
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        _restore_terminal()


if __name__ == "__main__":
    main()
