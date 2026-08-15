"""Typed contracts for every stage boundary.

The harness requirement is satisfied by these: each pipeline stage takes and
returns a validated model, so a malformed hand-off fails loudly at the boundary
instead of silently corrupting a downstream stage.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class QueryType(str, Enum):
    DESCRIPTION = "DESCRIPTION"
    NUMERIC = "NUMERIC"
    ENTITY = "ENTITY"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    UNKNOWN = "UNKNOWN"


class GuardrailVerdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class GuardrailStage(str, Enum):
    INPUT_SAFETY = "input_safety"
    OFF_TOPIC = "off_topic"
    RETRIEVAL_CONFIDENCE = "retrieval_confidence"
    GROUNDING = "grounding"


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED_UNSAFE = "refused_unsafe"
    REFUSED_OFF_TOPIC = "refused_off_topic"
    REFUSED_LOW_CONFIDENCE = "refused_low_confidence"
    REFUSED_UNGROUNDED = "refused_ungrounded"
    DEGRADED_EXTRACTIVE = "degraded_extractive"
    ERROR = "error"

    @property
    def is_refusal(self) -> bool:
        return self.value.startswith("refused_")


class RetrievalScope(str, Enum):
    ALL_LANGUAGES = "all"
    SAME_LANGUAGE = "same"
    ENGLISH_ONLY = "english"


# --------------------------------------------------------------------------
# Chunk / corpus models
# --------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Metadata carried by every chunk. Enables metadata-aware filtering."""

    doc_hash: str = Field(description="Stable content hash — synthesized doc ID")
    language: str
    flores_code: str
    is_english: bool
    query_type: QueryType = QueryType.UNKNOWN
    is_selected: bool = Field(
        default=False, description="MS MARCO relevance label for its source query"
    )
    source_query_ids: list[int] = Field(default_factory=list)
    strategy: str = Field(description="Chunking strategy that produced this chunk")
    parent_hash: str | None = Field(
        default=None, description="Parent doc for small-to-big retrieval"
    )
    char_count: int = 0
    token_estimate: int = 0
    position: int = 0


class Chunk(BaseModel):
    """A unit of text that gets embedded and indexed."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    # Wider context returned instead of `text` for sentence-window/parent-child.
    context_text: str | None = None

    @property
    def retrieval_text(self) -> str:
        """Text handed to the LLM — the wide context when one exists."""
        return self.context_text or self.text


class RetrievedChunk(BaseModel):
    """A chunk plus its scores from one retrieval pass."""

    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None
    rank: int = 0

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fused_score


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------


class GuardrailResult(BaseModel):
    """One guardrail's decision. Rendered inspectably in the UI."""

    stage: GuardrailStage
    verdict: GuardrailVerdict
    score: float | None = None
    threshold: float | None = None
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.verdict is GuardrailVerdict.BLOCK


class GroundingClaim(BaseModel):
    """A single claim from the answer, checked against retrieved context."""

    claim: str
    supported: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


class StageTiming(BaseModel):
    stage: str
    duration_ms: float
    # False for stages excluded from the <200ms claim (network-bound calls).
    counted_in_retrieval_budget: bool = True


class TimingBreakdown(BaseModel):
    """Per-stage timing. Drives the UI waterfall and the percentile analytics."""

    stages: list[StageTiming] = Field(default_factory=list)

    def add(self, stage: str, ms: float, *, counted: bool = True) -> None:
        self.stages.append(
            StageTiming(stage=stage, duration_ms=ms, counted_in_retrieval_budget=counted)
        )

    @property
    def retrieval_ms(self) -> float:
        """Sum of stages inside the <200ms budget."""
        return sum(s.duration_ms for s in self.stages if s.counted_in_retrieval_budget)

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.stages)

    def get(self, stage: str) -> float:
        return next((s.duration_ms for s in self.stages if s.stage == stage), 0.0)

    @property
    def meets_budget(self) -> bool:
        return self.retrieval_ms < 200.0


# --------------------------------------------------------------------------
# Requests / responses
# --------------------------------------------------------------------------


class TranscriptionResult(BaseModel):
    text: str
    language_code: str
    provider: str = "sarvam"
    duration_ms: float = 0.0
    confidence: float | None = None
    audio_seconds: float | None = None


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    language: str | None = Field(default=None, description="Auto-detected if omitted")
    scope: RetrievalScope = RetrievalScope.ALL_LANGUAGES
    top_k: int = Field(default=5, ge=1, le=50)
    strategy: str | None = None
    include_debug: bool = True


class Citation(BaseModel):
    chunk_id: str
    doc_hash: str
    text: str
    language: str
    score: float
    marker: int = Field(description="1-based [n] marker used in the answer")


class AnswerResponse(BaseModel):
    """The full pipeline result. Everything the UI renders comes from here."""

    query: str
    detected_language: str
    answer: str
    status: AnswerStatus
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    guardrails: list[GuardrailResult] = Field(default_factory=list)
    grounding_claims: list[GroundingClaim] = Field(default_factory=list)
    timing: TimingBreakdown = Field(default_factory=TimingBreakdown)
    provider_used: str | None = None
    provider_attempts: list[dict[str, Any]] = Field(default_factory=list)
    strategy_used: str | None = None
    request_id: str = ""
    warnings: list[str] = Field(default_factory=list)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    language: str = "hi"
    speaker: str | None = None


# --------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------


class RetrievalMetrics(BaseModel):
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    queries_evaluated: int = 0


class LatencyPercentiles(BaseModel):
    stage: str
    p50: float
    p70: float
    p90: float
    p95: float
    p100: float
    mean: float
    samples: int


class StrategyBenchmark(BaseModel):
    """One chunking strategy's measured result — the comparison table row."""

    strategy: str
    metrics: RetrievalMetrics
    chunk_count: int
    avg_chunk_chars: float
    index_build_seconds: float
    embedding_count: int
    latency: LatencyPercentiles | None = None
    notes: str = ""
