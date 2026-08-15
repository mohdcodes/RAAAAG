"""RAG orchestrator.

Wires every stage together and times each one. The stage sequence, and which
stages count toward the <200ms retrieval budget:

    guardrail:input      counted    ~1-3ms      deterministic pattern matching
    embed:query          counted    ~15-25ms    e5-large forward pass, CPU
    search:vector        counted    ~3-10ms     Qdrant binary + rescore
    rerank:cross_encoder counted    ~30-60ms    bge-reranker over top-50
    guardrail:confidence counted    <1ms        threshold check
    ------------------------------------------- budget total ~55-105ms
    generate:llm         EXCLUDED   400-1500ms  third-party network call
    guardrail:grounding  EXCLUDED   ~5-20ms     runs after generation

Generation is excluded from the budget because it is a network call to a
third-party API whose latency is not ours to control — and because no LLM
completes in under 200ms. That exclusion is stated explicitly in the response
rather than hidden: `timing.retrieval_ms` and `timing.total_ms` are both
returned, and the UI shows both.

Every refusal path returns the same response shape as a success, so the frontend
renders guardrail decisions with the same machinery as answers.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.core.config import get_settings
from app.core.languages import get_language
from app.core.logging import get_logger, new_request_id
from app.core.schemas import (
    AnswerResponse,
    AnswerStatus,
    GuardrailResult,
    QueryRequest,
    RetrievalScope,
    RetrievedChunk,
    TimingBreakdown,
)
from app.core.timing import Stopwatch
from app.guardrails import GuardrailPipeline
from app.harness.generator import GenerationHarness
from app.harness.providers import AllProvidersFailed

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end query orchestration.

    Dependencies are injected so the pipeline can be tested without loading
    models or running Qdrant.
    """

    def __init__(
        self,
        *,
        embedder=None,
        vector_store=None,
        reranker=None,
        guardrails: GuardrailPipeline | None = None,
        generator: GenerationHarness | None = None,
    ) -> None:
        self.settings = get_settings()
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self.guardrails = guardrails or GuardrailPipeline()
        self.generator = generator or GenerationHarness()

    # Lazy accessors: models load on first use, not at import time, so the API
    # starts fast and tests that never retrieve never pay the model load.

    @property
    def embedder(self):
        if self._embedder is None:
            from app.retrieval.embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            from app.retrieval.vector_store import get_vector_store

            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def reranker(self):
        if self._reranker is None and self.settings.rerank_enabled:
            from app.retrieval.reranker import get_reranker

            self._reranker = get_reranker()
        return self._reranker

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def answer(self, request: QueryRequest) -> AnswerResponse:
        request_id = new_request_id()
        timing = TimingBreakdown()
        guardrail_results: list[GuardrailResult] = []
        warnings: list[str] = []

        # -- Stage 1: input guardrails ---------------------------------
        watch = Stopwatch()
        input_results, refusal, query = self.guardrails.check_input(request.text)
        timing.add("guardrail:input", watch.stop())
        guardrail_results.extend(input_results)

        language = request.language or self._detect_language(query)

        if refusal is not None:
            return self._refusal_response(
                request_id, query, language, refusal, guardrail_results, timing,
                detail=input_results[-1].reason if input_results else "",
            )

        # -- Stage 2: embed query --------------------------------------
        watch = Stopwatch()
        try:
            query_vector = self.embedder.encode_queries([query])[0]
        except Exception as exc:  # noqa: BLE001 - surfaced as an error response
            timing.add("embed:query", watch.stop())
            logger.error("embedding_failed", error=str(exc))
            return self._error_response(
                request_id, query, language, guardrail_results, timing,
                f"Query embedding failed: {exc}",
            )
        timing.add("embed:query", watch.stop())

        # -- Stage 3: vector search ------------------------------------
        watch = Stopwatch()
        try:
            candidates = self._search(query_vector, request, language)
        except Exception as exc:  # noqa: BLE001
            timing.add("search:vector", watch.stop())
            logger.error("search_failed", error=str(exc))
            return self._error_response(
                request_id, query, language, guardrail_results, timing,
                f"Vector search failed: {exc}",
            )
        timing.add("search:vector", watch.stop())

        # -- Stage 4: rerank -------------------------------------------
        watch = Stopwatch()
        if self.reranker is not None and candidates:
            try:
                retrieved = self.reranker.rerank(query, candidates, top_k=request.top_k)
            except Exception as exc:  # noqa: BLE001 - degrade, do not fail
                logger.warning("rerank_failed_using_dense_order", error=str(exc))
                warnings.append("Reranking unavailable; results ordered by vector score.")
                retrieved = list(candidates[: request.top_k])
        else:
            retrieved = list(candidates[: request.top_k])
        timing.add("rerank:cross_encoder", watch.stop())

        # -- Stage 5: confidence gate ----------------------------------
        watch = Stopwatch()
        confidence_result, low_confidence = self.guardrails.check_retrieval(retrieved)
        timing.add("guardrail:confidence", watch.stop())
        guardrail_results.append(confidence_result)

        if low_confidence is not None:
            # Refuse before spending an LLM call — retrieval found nothing
            # relevant, and no prompt can fix that.
            return self._refusal_response(
                request_id, query, language, low_confidence, guardrail_results, timing,
                retrieved=retrieved, detail=confidence_result.reason,
            )

        # -- Stage 6: generation (excluded from the retrieval budget) ---
        watch = Stopwatch()
        status = AnswerStatus.ANSWERED
        provider_used: str | None = None
        attempts: list[dict[str, Any]] = []
        try:
            answer_text, citations, metadata, attempts = await self.generator.answer(
                query, retrieved, language_name=self._language_name(language)
            )
            provider_used = metadata.get("provider")
            if metadata.get("parse_repaired"):
                warnings.append("Model output required JSON repair.")
            if not metadata.get("sufficient_context", True):
                warnings.append("Model reported the context was insufficient.")
        except AllProvidersFailed as exc:
            attempts = exc.attempts
            answer_text, citations = self.generator.extractive_fallback(retrieved)
            status = AnswerStatus.DEGRADED_EXTRACTIVE
            warnings.append("All generation providers failed; returned top passage.")
            logger.error("all_providers_failed", attempts=attempts)
        timing.add("generate:llm", watch.stop(), counted=False)

        # -- Stage 7: grounding verification ---------------------------
        grounding_claims = []
        if status is AnswerStatus.ANSWERED:
            watch = Stopwatch()
            grounding_result, grounding_claims, ungrounded = self.guardrails.check_grounding(
                answer_text, retrieved
            )
            timing.add("guardrail:grounding", watch.stop(), counted=False)
            guardrail_results.append(grounding_result)

            if ungrounded is not None:
                return self._refusal_response(
                    request_id, query, language, ungrounded, guardrail_results, timing,
                    retrieved=retrieved, claims=grounding_claims,
                    detail=grounding_result.reason, provider=provider_used,
                    attempts=attempts,
                )

        logger.info(
            "query_answered",
            status=status.value,
            retrieval_ms=round(timing.retrieval_ms, 2),
            total_ms=round(timing.total_ms, 2),
            within_budget=timing.meets_budget,
            provider=provider_used,
        )

        return AnswerResponse(
            query=query,
            detected_language=language,
            answer=answer_text,
            status=status,
            citations=citations,
            retrieved=retrieved if request.include_debug else [],
            guardrails=guardrail_results,
            grounding_claims=grounding_claims,
            timing=timing,
            provider_used=provider_used,
            provider_attempts=attempts,
            strategy_used=request.strategy or self.settings.chunking_strategy,
            request_id=request_id,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _search(
        self, query_vector, request: QueryRequest, language: str
    ) -> list[RetrievedChunk]:
        """Vector search, scoped by the requested retrieval mode."""
        languages: Sequence[str] | None = None
        english_only = False

        if request.scope is RetrievalScope.SAME_LANGUAGE:
            languages = [language]
        elif request.scope is RetrievalScope.ENGLISH_ONLY:
            english_only = True
        # ALL_LANGUAGES leaves both unset: cross-lingual retrieval is the point,
        # since e5 shares a vector space across languages.

        return self.vector_store.search(
            query_vector,
            limit=self.settings.retrieval_candidates,
            languages=languages,
            english_only=english_only,
            strategy=request.strategy,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect language by script.

        Unicode ranges are decisive for Indic scripts and cost microseconds,
        unlike a statistical detector. The ambiguous case is Devanagari, shared
        by Hindi, Marathi, Nepali and Sanskrit — Hindi is the right default
        there by corpus frequency, and the user can override it explicitly.
        """
        ranges = (
            ((0x0980, 0x09FF), "bn"),  # Bengali/Assamese
            ((0x0A00, 0x0A7F), "pa"),  # Gurmukhi
            ((0x0A80, 0x0AFF), "gu"),  # Gujarati
            ((0x0B00, 0x0B7F), "or"),  # Odia
            ((0x0B80, 0x0BFF), "ta"),  # Tamil
            ((0x0C00, 0x0C7F), "te"),  # Telugu
            ((0x0C80, 0x0CFF), "kn"),  # Kannada
            ((0x0D00, 0x0D7F), "ml"),  # Malayalam
            ((0x0600, 0x06FF), "ur"),  # Perso-Arabic
            ((0x0900, 0x097F), "hi"),  # Devanagari — checked last
        )
        counts: dict[str, int] = {}
        for char in text:
            point = ord(char)
            for (low, high), code in ranges:
                if low <= point <= high:
                    counts[code] = counts.get(code, 0) + 1
                    break
        return max(counts, key=counts.get) if counts else "en"

    @staticmethod
    def _language_name(code: str) -> str:
        lang = get_language(code)
        return lang.name if lang else "the same language as the question"

    def _refusal_response(
        self,
        request_id: str,
        query: str,
        language: str,
        status: AnswerStatus,
        guardrails: list[GuardrailResult],
        timing: TimingBreakdown,
        *,
        retrieved: Sequence[RetrievedChunk] = (),
        claims=None,
        detail: str = "",
        provider: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
    ) -> AnswerResponse:
        """A refusal is a first-class outcome, shaped exactly like an answer."""
        logger.info(
            "query_refused", status=status.value,
            retrieval_ms=round(timing.retrieval_ms, 2), reason=detail[:120],
        )
        return AnswerResponse(
            query=query,
            detected_language=language,
            answer=self.guardrails.refusal_message(status, detail),
            status=status,
            citations=[],
            retrieved=list(retrieved),
            guardrails=guardrails,
            grounding_claims=claims or [],
            timing=timing,
            provider_used=provider,
            provider_attempts=attempts or [],
            request_id=request_id,
        )

    def _error_response(
        self,
        request_id: str,
        query: str,
        language: str,
        guardrails: list[GuardrailResult],
        timing: TimingBreakdown,
        message: str,
    ) -> AnswerResponse:
        return AnswerResponse(
            query=query,
            detected_language=language,
            answer=f"The request could not be completed: {message}",
            status=AnswerStatus.ERROR,
            guardrails=guardrails,
            timing=timing,
            request_id=request_id,
            warnings=[message],
        )

    def health(self) -> dict[str, Any]:
        return {
            "guardrails_enabled": self.guardrails.enabled,
            "rerank_enabled": self.settings.rerank_enabled,
            "confidence_threshold": self.settings.confidence_threshold,
            "generation": self.generator.health(),
            "vector_store": self.vector_store.health(),
        }


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
