"""POST /api/v1/retrieve — the <200 ms SLO path.

This endpoint exists specifically so retrieval latency can be measured without
LLM noise in it. The brief asks for the retrieval pipeline to complete under
200 ms and for generation to be reported separately; mixing them into one
number would make both meaningless.

Guardrails G1/G2/G3 run here because they are part of the retrieval path and
cost ~0.35 ms combined. G5/G6 are post-generation and live on /query.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings
from app.core import latency
from app.core.text import detect_script_lang, normalize
from app.core.timing import TimingBreakdown, TimingCollector
from app.embedding.onnx_embedder import get_embedder
from app.index.fusion import mmr_diversify, rrf_flatness
from app.index.multi_index import get_retriever

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    strategies: list[str] | None = None
    diversify: bool = True
    explain: bool = True


class PassageOut(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    lang: str | None = None
    score: float
    rrf_score: float
    is_gold: bool = False
    strategies: dict[str, int] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    normalized_query: str
    detected_lang: str
    passages: list[PassageOut]
    n_candidates: int
    top_score: float
    rrf_flatness: float
    off_topic_similarity: float | None = None
    timing: TimingBreakdown
    per_strategy: dict[str, Any] | None = None


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    tc = TimingCollector()
    ret = get_retriever()
    emb = get_embedder()

    with tc.stage("normalize"):
        q = normalize(req.query)
        lang = detect_script_lang(q)

    with tc.stage("embed") as d:
        qvec = emb.embed_query(q)
        d["dim"] = int(qvec.shape[0])

    # G3 off-topic gate: one 64x384 matvec against the query vector we already
    # have. Reported, not enforced, on this endpoint — /query enforces it.
    off_topic_sim = None
    if ret.centroids is not None:
        with tc.stage("guard.offtopic") as d:
            off_topic_sim = float((ret.centroids @ qvec).max())
            d["max_sim"] = round(off_topic_sim, 4)

    with tc.stage("retrieve") as d:
        fused, raw = ret.retrieve(
            q, qvec, top_k=settings.search_top_k, strategies=req.strategies
        )
        d["n_fused"] = len(fused)
        d["per_strategy"] = {k: len(v) for k, v in raw.items()}

    with tc.stage("fuse") as d:
        flat = rrf_flatness(fused)
        d["flatness"] = round(flat, 4) if flat != float("inf") else None

        if req.diversify and len(fused) > req.top_k:
            with_vecs = {}
            for h in fused[:30]:
                c = ret.chunks.get(h.best_chunk_id)
                if c is not None:
                    with_vecs[h.doc_id] = None      # vectors not cached; MMR degrades
            top = mmr_diversify(fused[:30], {}, top_n=req.top_k)
        else:
            top = fused[: req.top_k]

    passages = [
        PassageOut(
            doc_id=h.doc_id,
            chunk_id=h.best_chunk_id,
            text=ret.chunk_text(h.best_chunk_id),
            lang=(ret.chunks.get(h.best_chunk_id) or {}).get("lang"),
            score=round(h.best_dense_score, 4),
            rrf_score=round(h.rrf_score, 6),
            is_gold=bool((ret.doc(h.doc_id) or {}).get("is_selected")),
            strategies=h.strategies,
        )
        for h in top
    ]

    # `span_of` not `sum_of`: the four index searches overlap, so summing them
    # would report ~4x their true wall cost. The SLO is a wall-clock claim.
    retrieval_ms = tc.total_ms()

    timing = TimingBreakdown(
        total_ms=round(tc.total_ms(), 3),
        retrieval_ms=round(retrieval_ms, 3),
        stages=tc.stages,
    )
    latency.record(timing, endpoint="retrieve", outcome="retrieved", lang=lang)

    return RetrieveResponse(
        query=req.query,
        normalized_query=q,
        detected_lang=lang,
        passages=passages,
        n_candidates=len(fused),
        top_score=passages[0].score if passages else 0.0,
        rrf_flatness=round(flat, 4) if flat != float("inf") else 999.0,
        off_topic_similarity=round(off_topic_sim, 4) if off_topic_sim is not None else None,
        timing=timing,
        per_strategy={
            k: [{"chunk_id": c, "score": round(s, 4)} for c, s in v[:5]]
            for k, v in raw.items()
        } if req.explain else None,
    )
