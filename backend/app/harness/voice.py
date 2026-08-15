"""Sarvam speech services: transcription and synthesis.

Sarvam is built for Indian languages, which is why it fits this dataset — a
general-purpose STT model transcribing Kannada or Odia performs materially worse
than one trained for them.

Both directions degrade rather than crash when SARVAM_API_KEY is absent, so the
rest of the pipeline stays testable and the app stays usable via text input.

Note on language coverage: Sarvam does not support every language in the
dataset. Assamese, Nepali, Sanskrit and Urdu have no locale, so transcription
falls back to a related script and the result is flagged lower-confidence rather
than silently presented as reliable.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.languages import get_language, sarvam_locale_for, stt_supported
from app.core.logging import get_logger
from app.core.schemas import TranscriptionResult

logger = get_logger(__name__)


class VoiceError(Exception):
    """Speech service failure."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class VoiceUnavailable(VoiceError):
    """No API key configured."""


class SarvamVoice:
    """Sarvam STT (saarika) and TTS (bulbul)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        stt_model: str | None = None,
        tts_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.api_key = api_key or settings.sarvam_api_key
        self.stt_model = stt_model or settings.sarvam_stt_model
        self.tts_model = tts_model or settings.sarvam_tts_model
        self.stt_url = settings.sarvam_stt_url
        self.tts_url = settings.sarvam_tts_url

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Speech to text
    # ------------------------------------------------------------------

    # Sarvam accepts only MP3 and WAV — WebM/Opus, which is what browsers
    # record by default, is rejected with a 400. The frontend converts to WAV
    # before upload; these defaults match that, and anything else is
    # normalized below so a mislabelled content type cannot cause a rejection.
    ACCEPTED_CONTENT_TYPES = frozenset(
        {
            "audio/mpeg", "audio/mp3", "audio/mpeg3", "audio/x-mpeg-3",
            "audio/x-mp3", "audio/wav", "audio/x-wav", "audio/wave",
            "audio/pcm_s16le",
        }
    )

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> TranscriptionResult:
        """Transcribe audio.

        `language=None` asks Sarvam to auto-detect, which is the right default
        for a 14-language system — requiring the user to declare their language
        before speaking defeats the point of voice input.
        """
        if not self.configured:
            raise VoiceUnavailable(
                "SARVAM_API_KEY is not set — voice input is disabled. "
                "Add it to backend/.env to enable transcription."
            )

        if not audio_bytes:
            raise VoiceError("Empty audio payload.")
        if len(audio_bytes) > self.settings.max_audio_bytes:
            raise VoiceError(
                f"Audio exceeds {self.settings.max_audio_bytes // (1024 * 1024)} MB limit."
            )

        started = time.perf_counter()
        # "unknown" is Sarvam's auto-detect sentinel.
        locale = sarvam_locale_for(language) if language else "unknown"

        # Reject the unsupported container here rather than paying a network
        # round trip to learn the same thing from Sarvam's 400.
        normalized_type = content_type.split(";")[0].strip().lower()
        if normalized_type not in self.ACCEPTED_CONTENT_TYPES:
            raise VoiceError(
                f"Audio format {normalized_type!r} is not supported by Sarvam. "
                "Convert to WAV or MP3 before upload — browsers record WebM/Opus "
                "by default, which Sarvam rejects."
            )

        data: dict[str, str] = {"model": self.stt_model, "language_code": locale}
        files = {"file": (filename, audio_bytes, normalized_type)}

        try:
            async with httpx.AsyncClient(timeout=self.settings.stt_timeout_s) as client:
                response = await client.post(
                    self.stt_url,
                    headers={"api-subscription-key": self.api_key},
                    data=data,
                    files=files,
                )
        except httpx.TimeoutException as exc:
            raise VoiceError(f"Transcription timed out after {self.settings.stt_timeout_s}s",
                             retryable=True) from exc
        except httpx.HTTPError as exc:
            raise VoiceError(f"Transcription request failed: {exc}", retryable=True) from exc

        if response.status_code != 200:
            raise VoiceError(
                f"Sarvam STT returned {response.status_code}: {response.text[:200]}",
                retryable=response.status_code in (429, 500, 502, 503, 504),
            )

        payload: dict[str, Any] = response.json()
        text = (payload.get("transcript") or "").strip()
        detected = payload.get("language_code") or locale
        duration_ms = (time.perf_counter() - started) * 1000

        short_code = self._to_short_code(detected)
        # Flag languages Sarvam has no model for — the transcript may still be
        # usable but should not be presented as high confidence.
        confidence = None if stt_supported(short_code) else 0.5

        logger.info(
            "transcribed",
            language=short_code,
            chars=len(text),
            duration_ms=round(duration_ms, 1),
        )
        return TranscriptionResult(
            text=text,
            language_code=short_code,
            provider="sarvam",
            duration_ms=duration_ms,
            confidence=confidence,
        )

    @staticmethod
    def _to_short_code(sarvam_locale: str) -> str:
        """Map a Sarvam locale ('hi-IN') back to our short code ('hi')."""
        if not sarvam_locale or sarvam_locale == "unknown":
            return "hi"
        base = sarvam_locale.split("-")[0].lower()
        # Sarvam uses 'od' for Odia; the dataset registry uses 'or'.
        if base == "od":
            return "or"
        return base if get_language(base) else "hi"

    # ------------------------------------------------------------------
    # Text to speech
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        *,
        language: str = "hi",
        speaker: str | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """Synthesize speech. Returns (wav_bytes, metadata).

        Sarvam caps input length per request, so long answers are truncated at a
        sentence boundary rather than mid-word.
        """
        if not self.configured:
            raise VoiceUnavailable(
                "SARVAM_API_KEY is not set — audio playback is disabled."
            )
        if not text.strip():
            raise VoiceError("Empty text.")

        started = time.perf_counter()
        locale = sarvam_locale_for(language)
        trimmed = self._trim_for_tts(text)

        body = {
            "inputs": [trimmed],
            "target_language_code": locale,
            "speaker": speaker or self.settings.sarvam_tts_speaker,
            "model": self.tts_model,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.tts_timeout_s) as client:
                response = await client.post(
                    self.tts_url,
                    headers={
                        "api-subscription-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except httpx.TimeoutException as exc:
            raise VoiceError(f"Synthesis timed out after {self.settings.tts_timeout_s}s",
                             retryable=True) from exc
        except httpx.HTTPError as exc:
            raise VoiceError(f"Synthesis request failed: {exc}", retryable=True) from exc

        if response.status_code != 200:
            raise VoiceError(
                f"Sarvam TTS returned {response.status_code}: {response.text[:200]}",
                retryable=response.status_code in (429, 500, 502, 503, 504),
            )

        payload = response.json()
        audios = payload.get("audios") or []
        if not audios:
            raise VoiceError("Sarvam TTS returned no audio.")

        audio_bytes = base64.b64decode(audios[0])
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "synthesized", language=language, chars=len(trimmed),
            duration_ms=round(duration_ms, 1),
        )
        return audio_bytes, {
            "provider": "sarvam",
            "model": self.tts_model,
            "language": language,
            "locale": locale,
            "duration_ms": round(duration_ms, 2),
            "truncated": len(trimmed) < len(text),
            "bytes": len(audio_bytes),
        }

    @staticmethod
    def _trim_for_tts(text: str, max_chars: int = 1500) -> str:
        """Trim to the API limit on a sentence boundary where possible."""
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        for terminator in (". ", "। ", "۔ ", "! ", "? "):
            index = cut.rfind(terminator)
            if index > max_chars * 0.6:
                return cut[: index + 1]
        return cut

    def health(self) -> dict[str, Any]:
        return {
            "provider": "sarvam",
            "configured": self.configured,
            "stt_model": self.stt_model,
            "tts_model": self.tts_model,
            "stt_available": self.configured,
            "tts_available": self.configured,
        }


_voice: SarvamVoice | None = None


def get_voice() -> SarvamVoice:
    global _voice
    if _voice is None:
        _voice = SarvamVoice()
    return _voice
