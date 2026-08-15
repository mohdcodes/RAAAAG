"""API tests.

Uses FastAPI's TestClient with a stubbed pipeline, so these run without Qdrant,
without models, and without API keys.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.analytics import LatencyStore, get_latency_store
from app.core.schemas import (
    AnswerResponse,
    AnswerStatus,
    Chunk,
    ChunkMetadata,
    RetrievedChunk,
    TimingBreakdown,
)


@pytest.fixture
def client(monkeypatch):
    """TestClient with the RAG pipeline stubbed out."""
    from app.api import routes

    class StubPipeline:
        def __init__(self) -> None:
            self.vector_store = StubStore()
            self.generator = StubGenerator()

        async def answer(self, request):
            timing = TimingBreakdown()
            timing.add("guardrail:input", 1.2)
            timing.add("embed:query", 18.4)
            timing.add("search:vector", 5.1)
            timing.add("rerank:cross_encoder", 41.0)
            timing.add("guardrail:confidence", 0.3)
            timing.add("generate:llm", 620.0, counted=False)
            return AnswerResponse(
                query=request.text,
                detected_language="en",
                answer="A corporation is a company recognized in law. [1]",
                status=AnswerStatus.ANSWERED,
                retrieved=[
                    RetrievedChunk(
                        chunk=Chunk(
                            chunk_id="c1",
                            text="A corporation is a company recognized in law.",
                            metadata=ChunkMetadata(
                                doc_hash="h1", language="en", flores_code="eng_Latn",
                                is_english=True, strategy="parent_child",
                            ),
                        ),
                        rerank_score=0.91,
                    )
                ],
                timing=timing,
                provider_used="groq",
                request_id="test123",
            )

    class StubStore:
        def health(self):
            return {"reachable": True, "target_collection_present": True}

        def collection_info(self):
            return {"exists": True, "points_count": 1234, "name": "msmarco_xi"}

        def scroll(self, **kwargs):
            return (
                [{"chunk_id": "c1", "text": "sample passage", "language": "en"}],
                "next_page_token",
            )

    class StubGenerator:
        def health(self):
            return {"any_available": True, "providers": []}

    monkeypatch.setattr(routes, "get_pipeline", lambda: StubPipeline())
    monkeypatch.setattr("app.main.get_settings", lambda: __import__(
        "app.core.config", fromlist=["get_settings"]
    ).get_settings())

    from app.main import app

    get_latency_store().clear()
    return TestClient(app)


class TestQueryEndpoint:
    def test_returns_answer_with_timing(self, client):
        response = client.post("/api/query", json={"text": "what is a corporation?"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "answered"
        assert body["answer"]
        assert body["timing"]["stages"]

    def test_rejects_empty_query(self, client):
        assert client.post("/api/query", json={"text": ""}).status_code == 422

    def test_rejects_overlong_query(self, client):
        assert client.post("/api/query", json={"text": "a" * 2000}).status_code == 422

    def test_records_to_analytics(self, client):
        client.post("/api/query", json={"text": "what is a corporation?"})
        summary = client.get("/api/analytics/latency").json()
        assert summary["samples"] == 1

    def test_request_id_header(self, client):
        response = client.post("/api/query", json={"text": "test query here"})
        assert response.headers["X-Request-ID"]
        assert float(response.headers["X-Process-Time"]) >= 0


class TestHealthEndpoint:
    def test_reports_components(self, client):
        body = client.get("/api/health").json()
        assert body["status"] in ("ready", "degraded")
        assert "vector_store" in body["components"]
        assert "generation" in body["components"]
        assert "voice" in body["components"]

    def test_health_returns_200_even_when_degraded(self, client):
        """A monitoring endpoint that fails to respond tells you nothing."""
        assert client.get("/api/health").status_code == 200


class TestDatasetEndpoints:
    def test_preview_pages(self, client):
        body = client.get("/api/dataset/preview?limit=10").json()
        assert body["rows"]
        assert body["next_offset"] == "next_page_token"

    def test_preview_caps_limit(self, client):
        assert client.get("/api/dataset/preview?limit=9999").status_code == 200

    def test_stats_include_provenance(self, client):
        body = client.get("/api/dataset/stats").json()
        assert body["dataset"]["id"] == "ai4bharat/MSMARCO-XI"
        assert body["dataset"]["total_rows_upstream"] == 11_451_314
        assert len(body["languages"]) == 14

    def test_languages_endpoint(self, client):
        body = client.get("/api/languages").json()
        assert len(body["languages"]) == 14
        assert "hi" in body["all_codes"]


class TestStrategiesEndpoint:
    def test_lists_registered_strategies(self, client):
        body = client.get("/api/strategies").json()
        names = {s["name"] for s in body["strategies"]}
        assert {"passage_native", "parent_child", "semantic", "late_chunking"} <= names


class TestAnalyticsEndpoints:
    def test_empty_summary(self, client):
        body = client.get("/api/analytics/latency").json()
        assert body["samples"] == 0

    def test_reset_clears(self, client):
        client.post("/api/query", json={"text": "what is a corporation?"})
        client.post("/api/analytics/reset")
        assert client.get("/api/analytics/latency").json()["samples"] == 0


class TestVoiceEndpoints:
    def test_transcribe_without_key_returns_503(self, client):
        response = client.post(
            "/api/voice/transcribe",
            files={"file": ("test.webm", b"fake audio bytes", "audio/webm")},
        )
        # 503 when unconfigured, 502 if a key exists but the call fails.
        assert response.status_code in (502, 503)

    def test_speak_without_key_returns_503(self, client):
        response = client.post("/api/voice/speak", json={"text": "hello", "language": "hi"})
        assert response.status_code in (502, 503)


class TestLatencyStore:
    def test_percentiles(self):
        store = LatencyStore()
        for ms in range(1, 101):
            timing = TimingBreakdown()
            timing.add("search:vector", float(ms))
            store.record(timing, status=AnswerStatus.ANSWERED)

        result = store.percentiles_for("search:vector")
        assert result.samples == 100
        assert result.p50 == pytest.approx(50.5, abs=0.5)
        assert result.p70 == pytest.approx(70.3, abs=0.5)
        assert result.p100 == 100.0

    def test_budget_compliance_measures_retrieval_only(self):
        """The 200ms claim covers retrieval, not the LLM call."""
        store = LatencyStore()
        timing = TimingBreakdown()
        timing.add("search:vector", 50.0)
        timing.add("generate:llm", 900.0, counted=False)
        store.record(timing, status=AnswerStatus.ANSWERED)

        summary = store.summary()
        assert summary["budget_compliance"]["percentage"] == 100.0
        assert "excludes LLM" in summary["budget_compliance"]["measures"]

    def test_ring_buffer_bounded(self):
        store = LatencyStore(max_samples=10)
        for _ in range(50):
            timing = TimingBreakdown()
            timing.add("x", 1.0)
            store.record(timing, status=AnswerStatus.ANSWERED)
        assert store.summary()["samples"] == 10

    def test_status_counts(self):
        store = LatencyStore()
        for status in (AnswerStatus.ANSWERED, AnswerStatus.REFUSED_LOW_CONFIDENCE):
            timing = TimingBreakdown()
            timing.add("x", 1.0)
            store.record(timing, status=status)
        counts = store.summary()["status_counts"]
        assert counts["answered"] == 1
        assert counts["refused_low_confidence"] == 1
