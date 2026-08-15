"""Output-side guardrails: retrieval confidence and answer grounding.

These are the two that actually prevent hallucination.

The confidence gate is the more important of the pair, and it is deliberately
placed *before* generation: if retrieval found nothing relevant, no prompt
engineering downstream will conjure a grounded answer, so the cheapest and most
reliable move is to refuse before spending an LLM call.

Grounding verification runs after generation and checks that what the model
actually said traces back to the retrieved passages. It uses lexical overlap
rather than an LLM judge — an LLM checking an LLM adds latency and a second
opportunity to hallucinate, while overlap is deterministic, fast, and
adequate for catching the failure mode that matters (fabricated specifics).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import (
    GroundingClaim,
    GuardrailResult,
    GuardrailStage,
    GuardrailVerdict,
    RetrievedChunk,
)
from app.core.timing import Stopwatch
from app.ingest.text_utils import normalize_text, split_sentences

logger = get_logger(__name__)


class ConfidenceGuard:
    """Refuses when retrieval is too weak to support any grounded answer.

    Threshold applies to the reranker's sigmoid-normalized score, which is
    bounded 0-1 and therefore stable across queries. Raw retrieval scores would
    not be — cosine similarity distributions shift with query length and
    language, so a fixed threshold on them would be meaningless.
    """

    stage = GuardrailStage.RETRIEVAL_CONFIDENCE

    def __init__(
        self, threshold: float | None = None, min_passages: int | None = None
    ) -> None:
        settings = get_settings()
        self.threshold = threshold if threshold is not None else settings.confidence_threshold
        self.min_passages = (
            min_passages if min_passages is not None else settings.min_context_passages
        )

    def check(self, retrieved: Sequence[RetrievedChunk]) -> GuardrailResult:
        watch = Stopwatch()

        if not retrieved:
            return GuardrailResult(
                stage=self.stage,
                verdict=GuardrailVerdict.BLOCK,
                score=0.0,
                threshold=self.threshold,
                reason="No passages retrieved for this query.",
                details={"retrieved_count": 0},
                duration_ms=watch.stop(),
            )

        top_score = retrieved[0].final_score
        above = [c for c in retrieved if c.final_score >= self.threshold]
        details = {
            "retrieved_count": len(retrieved),
            "above_threshold": len(above),
            "top_score": round(top_score, 4),
            "score_distribution": [round(c.final_score, 4) for c in retrieved[:5]],
        }

        if len(above) < self.min_passages:
            return GuardrailResult(
                stage=self.stage,
                verdict=GuardrailVerdict.BLOCK,
                score=top_score,
                threshold=self.threshold,
                reason=(
                    f"Top retrieval score {top_score:.3f} is below the "
                    f"{self.threshold:.2f} confidence threshold — the corpus does "
                    "not appear to contain an answer to this question."
                ),
                details=details,
                duration_ms=watch.stop(),
            )

        # Passing but close to the line: answer, and say it may be incomplete.
        if top_score < self.threshold * 1.4:
            return GuardrailResult(
                stage=self.stage,
                verdict=GuardrailVerdict.WARN,
                score=top_score,
                threshold=self.threshold,
                reason=(
                    f"Retrieval confidence {top_score:.3f} is only modestly above "
                    "threshold; the answer may be partial."
                ),
                details=details,
                duration_ms=watch.stop(),
            )

        return GuardrailResult(
            stage=self.stage,
            verdict=GuardrailVerdict.PASS,
            score=top_score,
            threshold=self.threshold,
            reason=f"{len(above)} passage(s) above the confidence threshold.",
            details=details,
            duration_ms=watch.stop(),
        )


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------

# Tokens carrying factual weight: numbers, dates, proper nouns, and any
# non-Latin word (Indic scripts have no case, so capitalization cannot be used
# to spot entities there).
_NUMERIC = re.compile(r"\b\d[\d,.\-/]*\b")
_TOKEN = re.compile(r"\w+", re.UNICODE)

# Hedges that signal the model is declining rather than asserting. Sentences
# containing these are not treated as factual claims needing support.
_HEDGE_MARKERS = (
    "i don't know", "i do not know", "not enough information", "cannot determine",
    "no information", "does not contain", "unable to answer", "not mentioned",
    "insufficient", "पर्याप्त नहीं", "जानकारी नहीं",
)


class GroundingGuard:
    """Verifies each answer sentence is supported by retrieved context.

    Support is measured by content-word overlap between the claim and each
    passage, with numeric tokens weighted heavily — fabricated dates, counts and
    statistics are the hallucination that most damages a RAG system's
    credibility, and they are exactly what overlap catches reliably.
    """

    stage = GuardrailStage.GROUNDING

    def __init__(
        self,
        *,
        support_threshold: float = 0.35,
        min_supported_ratio: float = 0.5,
        numeric_weight: float = 2.0,
    ) -> None:
        self.support_threshold = support_threshold
        self.min_supported_ratio = min_supported_ratio
        self.numeric_weight = numeric_weight

    def check(
        self, answer: str, retrieved: Sequence[RetrievedChunk]
    ) -> tuple[GuardrailResult, list[GroundingClaim]]:
        watch = Stopwatch()

        if not answer.strip():
            result = GuardrailResult(
                stage=self.stage,
                verdict=GuardrailVerdict.BLOCK,
                reason="Empty answer.",
                duration_ms=watch.stop(),
            )
            return result, []

        # A refusal is grounded by definition — it asserts nothing.
        lowered = answer.lower()
        if any(marker in lowered for marker in _HEDGE_MARKERS):
            result = GuardrailResult(
                stage=self.stage,
                verdict=GuardrailVerdict.PASS,
                score=1.0,
                reason="Answer declines to assert facts; nothing to verify.",
                details={"refusal_detected": True},
                duration_ms=watch.stop(),
            )
            return result, []

        if not retrieved:
            result = GuardrailResult(
                stage=self.stage,
                verdict=GuardrailVerdict.BLOCK,
                score=0.0,
                reason="Answer asserts facts but no context was retrieved.",
                duration_ms=watch.stop(),
            )
            return result, []

        passage_tokens = [
            (chunk.chunk.chunk_id, self._tokenize(chunk.chunk.retrieval_text))
            for chunk in retrieved
        ]

        claims: list[GroundingClaim] = []
        for sentence in split_sentences(answer):
            claim_tokens = self._tokenize(sentence)
            if not claim_tokens:
                continue

            # Numbers are treated as hard evidence, not weighted tokens. A claim
            # asserting "1847" against context saying "1602" is false regardless
            # of how many surrounding words match — and averaging lets those
            # matching words drown out the one part that is actually wrong.
            claim_numbers = {t for t in claim_tokens if _NUMERIC.fullmatch(t)}

            best_score = 0.0
            supporting: list[str] = []
            for chunk_id, tokens in passage_tokens:
                overlap = self._weighted_overlap(claim_tokens, tokens)
                unsupported_numbers = claim_numbers - set(tokens)
                if overlap >= self.support_threshold and not unsupported_numbers:
                    supporting.append(chunk_id)
                elif unsupported_numbers:
                    # Cap the reported confidence so the UI shows the claim as
                    # doubtful rather than merely borderline.
                    overlap = min(overlap, self.support_threshold * 0.9)
                best_score = max(best_score, overlap)

            claims.append(
                GroundingClaim(
                    claim=sentence,
                    supported=bool(supporting),
                    supporting_chunk_ids=supporting[:3],
                    confidence=round(best_score, 4),
                )
            )

        if not claims:
            result = GuardrailResult(
                stage=self.stage,
                verdict=GuardrailVerdict.WARN,
                reason="No verifiable claims found in the answer.",
                duration_ms=watch.stop(),
            )
            return result, claims

        supported = sum(1 for c in claims if c.supported)
        ratio = supported / len(claims)
        details = {
            "total_claims": len(claims),
            "supported_claims": supported,
            "supported_ratio": round(ratio, 3),
            "unsupported": [c.claim[:120] for c in claims if not c.supported][:3],
        }

        if ratio < self.min_supported_ratio:
            verdict, reason = (
                GuardrailVerdict.BLOCK,
                f"Only {supported}/{len(claims)} claims trace to retrieved context.",
            )
        elif ratio < 1.0:
            verdict, reason = (
                GuardrailVerdict.WARN,
                f"{len(claims) - supported} of {len(claims)} claims are weakly supported.",
            )
        else:
            verdict, reason = (
                GuardrailVerdict.PASS,
                f"All {len(claims)} claims trace to retrieved context.",
            )

        result = GuardrailResult(
            stage=self.stage,
            verdict=verdict,
            score=round(ratio, 4),
            threshold=self.min_supported_ratio,
            reason=reason,
            details=details,
            duration_ms=watch.stop(),
        )
        return result, claims

    def _tokenize(self, text: str) -> dict[str, float]:
        """Content tokens with weights; numerics weighted higher.

        Stopwords are excluded by length rather than a word list, because a
        14-language stopword set would be a maintenance burden for marginal
        gain. Short tokens are dropped in Latin script only — Indic words are
        frequently short and meaningful.
        """
        normalized = normalize_text(text).lower()
        weights: dict[str, float] = {}

        for match in _NUMERIC.finditer(normalized):
            weights[match.group()] = self.numeric_weight

        for match in _TOKEN.finditer(normalized):
            token = match.group()
            if token in weights:
                continue
            is_latin = token.isascii()
            if is_latin and len(token) <= 3:
                continue
            weights[token] = 1.0

        return weights

    def _weighted_overlap(
        self, claim: dict[str, float], passage: dict[str, float]
    ) -> float:
        """Weighted recall of claim tokens found in the passage.

        Recall, not Jaccard: a long passage containing every claim token should
        score 1.0, and Jaccard would penalize it for its length.
        """
        if not claim:
            return 0.0
        total = sum(claim.values())
        matched = sum(weight for token, weight in claim.items() if token in passage)
        return matched / total if total else 0.0
