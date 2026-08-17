"""Wire types for the full query pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.timing import TimingBreakdown


class RefusalReason(str, Enum):
    """Distinct reasons matter. "Nothing relevant is indexed" and "I won't
    answer that" are different products and deserve different UI treatment."""

    UNSAFE_INPUT = "unsafe_input"
    PROMPT_INJECTION = "prompt_injection"
    OFF_TOPIC = "off_topic"
    LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
    UNGROUNDED_OUTPUT = "ungrounded_output"
    INVALID_CITATIONS = "invalid_citations"
    GENERATION_FAILED = "generation_failed"


REFUSAL_MESSAGES: dict[RefusalReason, str] = {
    RefusalReason.UNSAFE_INPUT:
        "I can't help with that request.",
    RefusalReason.PROMPT_INJECTION:
        "That request looks like an attempt to change my instructions, so I won't act on it.",
    RefusalReason.OFF_TOPIC:
        "That question is outside what this knowledge base covers, so I don't have grounded information to answer it.",
    RefusalReason.LOW_RETRIEVAL_CONFIDENCE:
        "I couldn't find passages relevant enough to answer that confidently.",
    RefusalReason.UNGROUNDED_OUTPUT:
        "I drafted an answer but couldn't verify it against the retrieved sources, so I'm not going to state it.",
    RefusalReason.INVALID_CITATIONS:
        "The draft answer cited sources that don't exist, so I've withheld it.",
    RefusalReason.GENERATION_FAILED:
        "The answer service is unavailable right now.",
}


class GuardrailCheck(BaseModel):
    id: str                       # G1..G6
    name: str
    passed: bool
    score: float | None = None
    threshold: float | None = None
    detail: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    lang: str | None = None
    speak: bool = False


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    lang: str | None = None
    score: float


class SentenceSupport(BaseModel):
    sentence: str
    supported: bool
    best_score: float
    supporting_chunk_id: str | None = None


class QueryResponse(BaseModel):
    query: str
    detected_lang: str

    answered: bool
    answer: str | None = None
    refusal_reason: RefusalReason | None = None
    refusal_message: str | None = None

    citations: list[Citation] = Field(default_factory=list)
    guardrails: list[GuardrailCheck] = Field(default_factory=list)
    groundedness: float | None = None
    sentence_support: list[SentenceSupport] = Field(default_factory=list)

    generated_by: Literal[
        "gemini", "gemini_fallback", "extractive_fallback", "none"
    ] = "none"
    tool_hops: int = 0
    retrieval: dict[str, Any] = Field(default_factory=dict)
    timing: TimingBreakdown


class GeneratedAnswer(BaseModel):
    """Structured output contract for the LLM.

    Never parse free text: a model that occasionally wraps JSON in prose, or
    emits a citation as a sentence instead of an id, breaks a regex parser
    silently. This is passed as `response_schema` so the SDK validates it and
    the model retries on mismatch.
    """

    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    used_context: bool = True
    self_confidence: Literal["high", "medium", "low"] = "medium"
