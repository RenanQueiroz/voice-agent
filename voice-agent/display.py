"""Shared types for the voice-agent display.

The TUI itself lives in `voice-agent/app.py` (Textual). This module now only
holds data types that `pipeline.py` and `providers.py` import, plus a
TYPE_CHECKING-only `Display` alias so other modules can type-annotate the
display target without creating an import cycle with `app.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import VoiceAgentApp as Display  # re-exported for annotations

__all__ = ["TurnMetrics", "Display"]


@dataclass
class TurnMetrics:
    """Timing metrics for a single conversation turn."""

    stt_seconds: float = 0.0
    llm_seconds: float = 0.0
    llm_first_token_seconds: float = 0.0  # TTFT — useful for spotting slow
    # streaming endpoints (Gemini OpenAI-compat in particular) where total
    # time looks fine but first-byte is high.
    llm_tokens: int = 0
    tts_seconds: float = 0.0
    tts_first_byte_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def llm_tokens_per_sec(self) -> float:
        return self.llm_tokens / self.llm_seconds if self.llm_seconds > 0 else 0.0
