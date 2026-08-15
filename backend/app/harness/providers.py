"""LLM provider abstraction with failover.

One interface, two implementations (Groq, Gemini), selected by an ordered
failover chain. Each provider carries its own circuit breaker so a persistently
failing provider is dropped from rotation rather than retried on every request —
without that, an outage at the primary provider adds its full timeout to every
single query.

Failure taxonomy matters here. Retrying a 401 is pointless and retrying a 400 is
worse than pointless, so errors are classified as retryable (429, 5xx, timeouts,
connection failures) or terminal (auth, malformed request) and handled
differently.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ProviderError(Exception):
    """Base for provider failures."""

    def __init__(self, message: str, *, provider: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class ProviderUnavailable(ProviderError):
    """Provider is not configured, or its circuit breaker is open."""


class AllProvidersFailed(Exception):
    """Every provider in the chain failed. Carries per-provider detail."""

    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        self.attempts = attempts
        summary = "; ".join(f"{a['provider']}: {a.get('error', 'unknown')}" for a in attempts)
        super().__init__(f"All providers failed — {summary}")


@dataclass(slots=True)
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    max_tokens: int = 1024
    temperature: float = 0.2
    json_schema: dict[str, Any] | None = None
    stop: list[str] | None = None


@dataclass(slots=True)
class GenerationResponse:
    text: str
    provider: str
    model: str
    duration_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Trips open after consecutive failures; half-opens after a cooldown.

    Prevents a dead provider from costing every request its timeout. State is
    per-provider and in-process, which is the right scope here — a single API
    instance's view of provider health is what matters for its own routing.
    """

    __slots__ = ("threshold", "cooldown_s", "_failures", "_opened_at", "_half_open")

    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None
        # Tracked explicitly rather than inferred from the failure count: with
        # threshold=1, `threshold - 1` is 0, which is indistinguishable from a
        # never-failed breaker.
        self._half_open = False

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            # Cooldown elapsed: half-open and let one probe request through.
            self._opened_at = None
            self._half_open = True
            self._failures = max(0, self.threshold - 1)
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open = False

    def record_failure(self) -> None:
        self._failures += 1
        self._half_open = False
        if self._failures >= self.threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning("circuit_breaker_opened", failures=self._failures)

    @property
    def state(self) -> str:
        if self.is_open:
            return "open"
        if self._half_open:
            return "half_open"
        return "half_open" if self._failures else "closed"


class LLMProvider(ABC):
    """One generation backend."""

    name: str = "base"

    def __init__(self, api_key: str | None, model: str, timeout_s: float) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        settings = get_settings()
        self.breaker = CircuitBreaker(
            threshold=settings.circuit_breaker_threshold,
            cooldown_s=settings.circuit_breaker_cooldown_s,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def available(self) -> bool:
        return self.configured and not self.breaker.is_open

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Run one generation. Raises ProviderError on failure."""

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "configured": self.configured,
            "circuit_state": self.breaker.state,
            "available": self.available,
        }


class GroqProvider(LLMProvider):
    """Groq — very low latency, which is why it leads the failover chain."""

    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        super().__init__(
            api_key or settings.groq_api_key,
            model or settings.groq_model,
            settings.generation_timeout_s,
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self.api_key, timeout=self.timeout_s)
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.configured:
            raise ProviderUnavailable("GROQ_API_KEY not set", provider=self.name)

        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if request.stop:
            kwargs["stop"] = request.stop

        try:
            completion = await self._get_client().chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalized below
            raise self._translate(exc) from exc

        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        return GenerationResponse(
            text=choice.message.content or "",
            provider=self.name,
            model=self.model,
            duration_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason,
        )

    def _translate(self, exc: Exception) -> ProviderError:
        message = str(exc)
        status = getattr(exc, "status_code", None)
        # 429 and 5xx are transient; auth and malformed-request errors are not.
        retryable = status in (408, 409, 429, 500, 502, 503, 504) or any(
            token in message.lower() for token in ("timeout", "connection", "temporarily")
        )
        return ProviderError(message, provider=self.name, retryable=retryable)


class GeminiProvider(LLMProvider):
    """Google Gemini — the failover target when Groq is unavailable."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        super().__init__(
            api_key or settings.gemini_api_key,
            model or settings.gemini_model,
            settings.generation_timeout_s,
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.configured:
            raise ProviderUnavailable("GEMINI_API_KEY not set", provider=self.name)

        from google.genai import types

        start = time.perf_counter()
        config = types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_sequences=request.stop or None,
            response_mime_type="application/json" if request.json_schema else None,
        )

        try:
            response = await asyncio.wait_for(
                self._get_client().aio.models.generate_content(
                    model=self.model, contents=request.user_prompt, config=config
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError(
                f"timed out after {self.timeout_s}s", provider=self.name, retryable=True
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalized below
            raise self._translate(exc) from exc

        usage = getattr(response, "usage_metadata", None)
        return GenerationResponse(
            text=response.text or "",
            provider=self.name,
            model=self.model,
            duration_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
        )

    def _translate(self, exc: Exception) -> ProviderError:
        message = str(exc)
        lowered = message.lower()
        retryable = any(
            token in lowered
            for token in ("429", "500", "503", "504", "timeout", "unavailable",
                          "resource_exhausted", "deadline")
        )
        return ProviderError(message, provider=self.name, retryable=retryable)


def build_providers() -> dict[str, LLMProvider]:
    """Instantiate every provider, configured or not.

    Unconfigured providers are kept so `/health` can report *why* they are
    unavailable rather than silently omitting them.
    """
    return {"groq": GroqProvider(), "gemini": GeminiProvider()}
