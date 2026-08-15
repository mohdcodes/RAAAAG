"""Guardrail pipeline.

Four layers, ordered so the cheapest checks run first and the expensive ones
only run on queries that survive:

    input safety   -> off-topic   -> [retrieval] -> confidence   -> [generation] -> grounding
    ~1ms              ~1ms                          <1ms                             ~5-20ms

Every layer returns a `GuardrailResult` carrying its verdict, score, threshold
and reasoning, and every one of those is surfaced in the UI. The system showing
*why* it declined is as much the deliverable as the decline itself.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import get_settings
from app.core.schemas import (
    AnswerStatus,
    GroundingClaim,
    GuardrailResult,
    GuardrailStage,
    GuardrailVerdict,
    RetrievedChunk,
)
from app.guardrails.input_guard import InputSafetyGuard, OffTopicGuard
from app.guardrails.output_guard import ConfidenceGuard, GroundingGuard

__all__ = [
    "ConfidenceGuard",
    "GroundingGuard",
    "GuardrailPipeline",
    "InputSafetyGuard",
    "OffTopicGuard",
    "REFUSAL_MESSAGES",
]


# User-facing refusal text per failure mode. Each says what happened and what
# the user can do about it — a bare "I can't help with that" teaches nothing.
REFUSAL_MESSAGES: dict[AnswerStatus, str] = {
    AnswerStatus.REFUSED_UNSAFE: (
        "This query was blocked by the input safety check. Try rephrasing it as a "
        "factual question about the dataset."
    ),
    AnswerStatus.REFUSED_OFF_TOPIC: (
        "This system answers factual questions from a retrieved web-passage corpus. "
        "It cannot chat, write content, or give personal advice."
    ),
    AnswerStatus.REFUSED_LOW_CONFIDENCE: (
        "No passage in the indexed corpus is relevant enough to answer this "
        "confidently. Rather than guess, the system is declining — try rephrasing, "
        "or ask about a topic covered by MS MARCO web passages."
    ),
    AnswerStatus.REFUSED_UNGROUNDED: (
        "A draft answer was generated but too much of it could not be traced back "
        "to the retrieved passages, so it was withheld as likely unreliable."
    ),
}


class GuardrailPipeline:
    """Runs all four guardrails and maps their verdicts to an outcome."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.enabled = settings.guardrails_enabled if enabled is None else enabled
        self.settings = settings
        self.input_guard = InputSafetyGuard()
        self.off_topic_guard = OffTopicGuard()
        self.confidence_guard = ConfidenceGuard(threshold=confidence_threshold)
        self.grounding_guard = GroundingGuard()

    # ------------------------------------------------------------------
    # Pre-retrieval
    # ------------------------------------------------------------------

    def check_input(
        self, query: str
    ) -> tuple[list[GuardrailResult], AnswerStatus | None, str]:
        """Screen a query before retrieval.

        Returns the guardrail results, a refusal status when blocked (None when
        the query may proceed), and the sanitized query text.
        """
        results: list[GuardrailResult] = []
        if not self.enabled:
            return results, None, query.strip()

        if self.settings.input_safety_enabled:
            safety = self.input_guard.check(query)
            results.append(safety)
            if safety.blocked:
                return results, AnswerStatus.REFUSED_UNSAFE, query.strip()

        sanitized = self.input_guard.sanitize(query)

        if self.settings.off_topic_check_enabled:
            off_topic = self.off_topic_guard.check(sanitized)
            results.append(off_topic)
            if off_topic.blocked:
                return results, AnswerStatus.REFUSED_OFF_TOPIC, sanitized

        return results, None, sanitized

    # ------------------------------------------------------------------
    # Post-retrieval
    # ------------------------------------------------------------------

    def check_retrieval(
        self, retrieved: Sequence[RetrievedChunk]
    ) -> tuple[GuardrailResult, AnswerStatus | None]:
        """Gate generation on retrieval quality."""
        result = self.confidence_guard.check(retrieved)
        status = AnswerStatus.REFUSED_LOW_CONFIDENCE if result.blocked else None
        return result, status

    # ------------------------------------------------------------------
    # Post-generation
    # ------------------------------------------------------------------

    def check_grounding(
        self, answer: str, retrieved: Sequence[RetrievedChunk]
    ) -> tuple[GuardrailResult, list[GroundingClaim], AnswerStatus | None]:
        """Verify the generated answer against retrieved context."""
        if not self.enabled or not self.settings.grounding_check_enabled:
            passthrough = GuardrailResult(
                stage=GuardrailStage.GROUNDING,
                verdict=GuardrailVerdict.PASS,
                reason="Grounding check disabled.",
            )
            return passthrough, [], None

        result, claims = self.grounding_guard.check(answer, retrieved)
        status = AnswerStatus.REFUSED_UNGROUNDED if result.blocked else None
        return result, claims, status

    @staticmethod
    def refusal_message(status: AnswerStatus, detail: str = "") -> str:
        base = REFUSAL_MESSAGES.get(status, "This query cannot be answered.")
        return f"{base}\n\n{detail}" if detail else base
