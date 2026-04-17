"""Gemini TTS adapter.

Google doesn't expose an OpenAI-compatible TTS endpoint, so we call the
native `generateContent` REST API with `responseModalities = ["AUDIO"]`
and wrap it to look like the SDK's `TTSModel` (a `model_name` property
plus an `async def run(text, settings) -> AsyncIterator[bytes]`).

Request shape (see https://ai.google.dev/gemini-api/docs/speech-generation):

    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    x-goog-api-key: <GEMINI_API_KEY>
    {
      "contents":  [{"parts": [{"text": "<text>"}]}],
      "generationConfig": {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
          "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "<voice>"}}
        }
      }
    }

The response's `candidates[0].content.parts[*].inlineData.data` is base64
L16 PCM at 24 kHz mono — exactly what our `AudioPlayer` expects. A few
server paths wrap it in a RIFF/WAV header; we strip that if present and
yield the raw PCM in 1024-byte chunks to match the streaming shape of
the OpenAI TTS path.

Gemini's TTS API does not stream, so we collect the full audio in one
request before yielding.

When Gemini returns no audio (safety block, bad finish reason, quota
error, etc.) we raise instead of silently yielding nothing — the SDK
turns the exception into a `voice_stream_event_error`, which we surface
as an error card in the UI. The failing response payload is also logged
to `logs/gemini-tts.log` so we can see why Gemini refused.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from agents.voice.model import TTSModel, TTSModelSettings

_DEFAULT_VOICE = "Kore"
_GEMINI_TTS_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_CHUNK_BYTES = 1024
_LOG_PATH = Path(__file__).parent.parent / "logs" / "gemini-tts.log"

# Retry on transient network/quota hiccups. Gemini TTS preview endpoints
# are flaky — especially 429, since the sentence splitter fires one HTTP
# call per sentence and the preview tier's RPM limit is low. We use
# exponential backoff with extra patience on 429s.
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFFS_S: tuple[float, ...] = (0.8, 1.8, 3.6)
_BACKOFF_429_S: tuple[float, ...] = (1.5, 3.5, 7.0)


class GeminiTTSError(RuntimeError):
    """Raised when Gemini's TTS response contains no usable audio."""


class GeminiTTSModel(TTSModel):
    """TTSModel wrapper around Gemini's `generateContent` API."""

    def __init__(self, model: str, api_key: str, timeout: float = 60.0):
        self._model = model
        self._api_key = api_key
        # trust_env=False so a system proxy / env-based network config can't
        # inject a second credential alongside our `x-goog-api-key` header.
        # Google's gateway rejects requests with more than one credential.
        self._client = httpx.AsyncClient(trust_env=False, timeout=timeout)

    @property
    def model_name(self) -> str:
        return self._model

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        voice = getattr(settings, "voice", None) or _DEFAULT_VOICE
        url = f"{_GEMINI_TTS_BASE}/{self._model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
        }
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        resp: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_BACKOFFS_S[attempt])
                    continue
                _log_failure(text, voice, None, f"HTTP error: {e}")
                raise GeminiTTSError(f"Gemini TTS request failed: {e}") from e

            if resp.status_code in _TRANSIENT_STATUSES and attempt < _MAX_RETRIES:
                delay = (
                    _BACKOFF_429_S[attempt]
                    if resp.status_code == 429
                    else _BACKOFFS_S[attempt]
                )
                _log_failure(
                    text,
                    voice,
                    resp,
                    f"Transient HTTP {resp.status_code}, retrying in {delay:.1f}s",
                )
                await asyncio.sleep(delay)
                continue
            break

        assert resp is not None  # guaranteed by the retry loop
        if resp.status_code >= 400:
            _log_failure(text, voice, resp, f"HTTP {resp.status_code}")
            raise GeminiTTSError(
                f"Gemini TTS HTTP {resp.status_code}: "
                f"{_short_body(resp.text)}"
            )

        try:
            payload = resp.json()
        except ValueError as e:
            _log_failure(text, voice, resp, f"Non-JSON body: {e}")
            raise GeminiTTSError("Gemini TTS returned a non-JSON response") from e

        pcm = b""
        for candidate in payload.get("candidates", []) or []:
            content = candidate.get("content") or {}
            for part in content.get("parts", []) or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and isinstance(inline, dict):
                    data = inline.get("data")
                    if isinstance(data, str):
                        pcm += base64.b64decode(data)

        if not pcm:
            reason = _diagnose_empty(payload)
            _log_failure(text, voice, resp, f"Empty audio: {reason}")
            raise GeminiTTSError(f"Gemini TTS returned no audio ({reason})")

        # A few serving paths wrap the PCM in a RIFF/WAV header; skip it if so.
        if pcm[:4] == b"RIFF":
            data_idx = pcm.find(b"data")
            if data_idx >= 0:
                pcm = pcm[data_idx + 8 :]

        for i in range(0, len(pcm), _CHUNK_BYTES):
            yield pcm[i : i + _CHUNK_BYTES]


def _diagnose_empty(payload: dict) -> str:
    """Return a short human-readable reason the response carried no audio."""
    feedback = payload.get("promptFeedback") or {}
    block = feedback.get("blockReason")
    if block:
        return f"promptFeedback.blockReason={block}"

    candidates = payload.get("candidates") or []
    if not candidates:
        return "no candidates returned"

    reasons: list[str] = []
    for cand in candidates:
        finish = cand.get("finishReason")
        if finish and finish != "STOP":
            reasons.append(f"finishReason={finish}")
        safety = cand.get("safetyRatings")
        if safety:
            blocked = [s for s in safety if s.get("blocked")]
            if blocked:
                reasons.append(
                    "safety-blocked: "
                    + ", ".join(str(b.get("category", "?")) for b in blocked)
                )
    if reasons:
        return "; ".join(reasons)

    error = payload.get("error") or {}
    if error:
        return f"error={error.get('message') or error.get('status')}"

    return "unknown — see logs/gemini-tts.log"


def _short_body(text: str, limit: int = 200) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _log_failure(
    text: str,
    voice: str,
    resp: httpx.Response | None,
    summary: str,
) -> None:
    """Append a diagnostic entry to `logs/gemini-tts.log`. Never raises."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "voice": voice,
            "text": text,
            "summary": summary,
            "status": getattr(resp, "status_code", None),
        }
        if resp is not None:
            try:
                entry["response"] = resp.json()
            except Exception:
                entry["response_text"] = _short_body(resp.text, 800)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
