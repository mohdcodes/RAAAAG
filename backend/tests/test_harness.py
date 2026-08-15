"""Harness tests.

The harness exists to survive provider failure, so these tests are mostly about
what happens when things go wrong: transient errors, hard failures, dead
providers, and models that return malformed JSON.
"""

from __future__ import annotations

import pytest

from app.core.schemas import Chunk, ChunkMetadata, RetrievedChunk
from app.harness.generator import GenerationHarness
from app.harness.providers import (
    AllProvidersFailed,
    CircuitBreaker,
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    ProviderError,
)


class FakeProvider(LLMProvider):
    """Scriptable provider: each call pops the next outcome from `script`."""

    def __init__(self, name: str, script: list, *, configured: bool = True) -> None:
        super().__init__("fake-key" if configured else None, f"{name}-model", 5.0)
        self.name = name
        self.script = list(script)
        self.calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        outcome = self.script.pop(0) if self.script else Exception("script exhausted")
        if isinstance(outcome, Exception):
            raise outcome
        return GenerationResponse(
            text=outcome, provider=self.name, model=self.model, duration_ms=1.0
        )


def retryable(msg: str = "429 rate limited") -> ProviderError:
    return ProviderError(msg, provider="fake", retryable=True)


def terminal(msg: str = "401 unauthorized") -> ProviderError:
    return ProviderError(msg, provider="fake", retryable=False)


def make_chunk(text: str, score: float = 0.9, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                doc_hash="h1", language="en", flores_code="eng_Latn",
                is_english=True, strategy="passage_native",
            ),
        ),
        rerank_score=score,
        fused_score=score,
    )


REQUEST = GenerationRequest(system_prompt="sys", user_prompt="user")


# ------------------------------------------------------------------ breaker


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        breaker = CircuitBreaker(threshold=3, cooldown_s=60)
        for _ in range(2):
            breaker.record_failure()
        assert not breaker.is_open
        breaker.record_failure()
        assert breaker.is_open
        assert breaker.state == "open"

    def test_success_resets(self):
        breaker = CircuitBreaker(threshold=3, cooldown_s=60)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert not breaker.is_open

    def test_half_opens_after_cooldown(self):
        """Cooldown 0 means the next check half-opens immediately."""
        breaker = CircuitBreaker(threshold=1, cooldown_s=0.0)
        breaker.record_failure()
        assert not breaker.is_open  # probe allowed through
        assert breaker.state == "half_open"


# ------------------------------------------------------------------ failover


class TestFailover:
    async def test_primary_success_skips_secondary(self):
        primary = FakeProvider("groq", ['{"answer":"ok","citations":[],"sufficient_context":true}'])
        secondary = FakeProvider("gemini", ["should not be called"])
        harness = GenerationHarness(
            {"groq": primary, "gemini": secondary}, order=["groq", "gemini"], max_retries=0
        )

        response, attempts = await harness.generate(REQUEST)
        assert response.provider == "groq"
        assert secondary.calls == 0
        assert attempts[-1]["status"] == "success"

    async def test_falls_over_to_secondary(self):
        primary = FakeProvider("groq", [terminal()])
        secondary = FakeProvider("gemini", ["recovered"])
        harness = GenerationHarness(
            {"groq": primary, "gemini": secondary}, order=["groq", "gemini"], max_retries=0
        )

        response, attempts = await harness.generate(REQUEST)
        assert response.provider == "gemini"
        assert response.text == "recovered"
        assert [a["status"] for a in attempts] == ["failed", "success"]

    async def test_retryable_error_is_retried(self):
        provider = FakeProvider("groq", [retryable(), retryable(), "third time lucky"])
        harness = GenerationHarness({"groq": provider}, order=["groq"], max_retries=2)

        response, attempts = await harness.generate(REQUEST)
        assert response.text == "third time lucky"
        assert provider.calls == 3
        assert len([a for a in attempts if a["status"] == "failed"]) == 2

    async def test_terminal_error_is_not_retried(self):
        """Retrying a 401 wastes the user's latency budget for nothing."""
        provider = FakeProvider("groq", [terminal(), "never reached"])
        harness = GenerationHarness({"groq": provider}, order=["groq"], max_retries=3)

        with pytest.raises(AllProvidersFailed):
            await harness.generate(REQUEST)
        assert provider.calls == 1

    async def test_unconfigured_provider_skipped_with_reason(self):
        unconfigured = FakeProvider("groq", [], configured=False)
        working = FakeProvider("gemini", ["ok"])
        harness = GenerationHarness(
            {"groq": unconfigured, "gemini": working}, order=["groq", "gemini"]
        )

        response, attempts = await harness.generate(REQUEST)
        assert response.provider == "gemini"
        assert attempts[0]["status"] == "skipped"
        assert "not configured" in attempts[0]["error"]

    async def test_all_providers_failed_carries_detail(self):
        harness = GenerationHarness(
            {"groq": FakeProvider("groq", [terminal("boom")]),
             "gemini": FakeProvider("gemini", [terminal("bang")])},
            order=["groq", "gemini"], max_retries=0,
        )

        with pytest.raises(AllProvidersFailed) as excinfo:
            await harness.generate(REQUEST)
        assert len(excinfo.value.attempts) == 2
        assert "boom" in str(excinfo.value)
        assert "bang" in str(excinfo.value)

    async def test_open_circuit_skips_provider(self):
        provider = FakeProvider("groq", ["unused"])
        provider.breaker = CircuitBreaker(threshold=1, cooldown_s=60)
        provider.breaker.record_failure()
        backup = FakeProvider("gemini", ["ok"])
        harness = GenerationHarness(
            {"groq": provider, "gemini": backup}, order=["groq", "gemini"]
        )

        response, attempts = await harness.generate(REQUEST)
        assert response.provider == "gemini"
        assert provider.calls == 0
        assert "circuit open" in attempts[0]["error"]


# ------------------------------------------------------------------ parsing


class TestAnswerParsing:
    def test_clean_json(self):
        parsed = GenerationHarness._parse_answer(
            '{"answer":"A corporation is a legal entity.","citations":[1,2],'
            '"sufficient_context":true}'
        )
        assert parsed["answer"] == "A corporation is a legal entity."
        assert parsed["citations"] == [1, 2]
        assert not parsed.get("_repaired")

    def test_markdown_fenced_json(self):
        parsed = GenerationHarness._parse_answer(
            '```json\n{"answer":"Fenced.","citations":[1],"sufficient_context":true}\n```'
        )
        assert parsed["answer"] == "Fenced."

    def test_json_with_surrounding_prose(self):
        parsed = GenerationHarness._parse_answer(
            'Sure! Here is the answer:\n{"answer":"Embedded.","citations":[1],'
            '"sufficient_context":true}\nHope that helps.'
        )
        assert parsed["answer"] == "Embedded."
        assert parsed["_repaired"] is True

    def test_truncated_json_recovers_answer(self):
        """Hitting max_tokens mid-JSON must not lose the answer text."""
        parsed = GenerationHarness._parse_answer(
            '{"answer":"This got cut off at the token limit","citations":[1'
        )
        assert "cut off" in parsed["answer"]
        assert parsed["_repaired"] is True

    def test_plain_text_becomes_answer(self):
        parsed = GenerationHarness._parse_answer("Just plain prose, no JSON at all.")
        assert parsed["answer"] == "Just plain prose, no JSON at all."
        assert parsed["_repaired"] is True

    def test_empty_marks_insufficient(self):
        parsed = GenerationHarness._parse_answer("")
        assert parsed["sufficient_context"] is False

    def test_missing_optional_fields_defaulted(self):
        parsed = GenerationHarness._parse_answer('{"answer":"Only answer field."}')
        assert parsed["citations"] == []
        assert parsed["sufficient_context"] is True


# ------------------------------------------------------------------ context


class TestContextBuilding:
    def test_numbers_passages_and_builds_citations(self):
        harness = GenerationHarness({})
        context, citations = harness._build_context(
            [make_chunk("First passage.", 0.9, "a"), make_chunk("Second passage.", 0.8, "b")],
            max_chars=6000,
        )
        assert "[1] First passage." in context
        assert "[2] Second passage." in context
        assert [c.marker for c in citations] == [1, 2]

    def test_respects_char_budget(self):
        harness = GenerationHarness({})
        chunks = [make_chunk("x" * 400, 0.9, f"c{i}") for i in range(10)]
        _, citations = harness._build_context(chunks, max_chars=1000)
        assert len(citations) < 10

    def test_uses_wide_context_for_small_to_big(self):
        """Parent context must reach the LLM, not the narrow embedded fragment."""
        chunk = make_chunk("narrow", 0.9)
        chunk.chunk.context_text = "the full parent passage with much more detail"
        harness = GenerationHarness({})
        context, _ = harness._build_context([chunk], max_chars=6000)
        assert "full parent passage" in context


# ------------------------------------------------------------------ degradation


class TestExtractiveFallback:
    def test_returns_top_passage_when_llm_unavailable(self):
        answer, citations = GenerationHarness.extractive_fallback(
            [make_chunk("A corporation is a legal entity recognized in law.", 0.9)]
        )
        assert "corporation is a legal entity" in answer
        assert "unavailable" in answer.lower()
        assert len(citations) == 1

    def test_handles_no_retrieval(self):
        answer, citations = GenerationHarness.extractive_fallback([])
        assert citations == []
        assert "No relevant passages" in answer


class TestHealth:
    def test_reports_provider_state(self):
        harness = GenerationHarness(
            {"groq": FakeProvider("groq", []), "gemini": FakeProvider("gemini", [], configured=False)}
        )
        health = harness.health()
        assert health["any_available"] is True
        states = {p["provider"]: p for p in health["providers"]}
        assert states["groq"]["available"] is True
        assert states["gemini"]["configured"] is False
