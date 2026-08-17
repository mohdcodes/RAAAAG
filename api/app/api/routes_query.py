"""POST /api/v1/query — the full pipeline.

Stage order, and why:

    normalize -> G1 -> G2 -> embed -> G3 -> retrieve -> fuse -> G4
                                                 |
                              <200 ms SLO ends here
                                                 |
                              -> generate -> G6 -> G5 -> assemble

Every guardrail SHORT-CIRCUITS. The moment one halts, we skip straight to
assembly and never pay for the LLM call — so a refusal is both cheaper and
faster than an answer, which is exactly the right incentive.

G6 (citation validity) runs before G5 (groundedness) deliberately: G6 is free
set-membership, G5 costs ~15 ms of embedding. Checking the cheap invalidator
first means a fabricated-citation answer never pays for the expensive check.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from app.config import settings
from app.core import latency
from app.core.text import detect_script_lang, normalize
from app.core.timing import TimingBreakdown, TimingCollector
from app.embedding.onnx_embedder import get_embedder
from app.guardrails.pipeline import (
    g1_input_sanity,
    g2_safety,
    g3_off_topic,
    g4_retrieval_confidence,
    g5_groundedness,
    g6_validate_citations,
)
from app.index.fusion import rrf_flatness
from app.index.multi_index import get_retriever
from app.llm.gemini_client import get_llm
from app.schemas.query import (
    REFUSAL_MESSAGES,
    Citation,
    GuardrailCheck,
    QueryRequest,
    QueryResponse,
    RefusalReason,
    SentenceSupport,
)

router = APIRouter()


def _refuse(
    *,
    req: QueryRequest,
    lang: str,
    reason: RefusalReason,
    checks: list[GuardrailCheck],
    tc: TimingCollector,
    retrieval: dict | None = None,
    citations: list[Citation] | None = None,
) -> QueryResponse:
    timing = TimingBreakdown(
        total_ms=round(tc.total_ms(), 3),
        retrieval_ms=round(tc.sum_of("normalize", "guards.input", "embed",
                                     "guard.offtopic", "retrieve", "fuse"), 3),
        stages=tc.stages,
    )
    # Refusals are measured too, and deliberately so: a guardrail short-circuit
    # is the FAST path, so dropping them would bias the percentiles upward and
    # flatter the numbers.
    latency.record(timing, endpoint="query", outcome=reason.value, lang=lang)
    return QueryResponse(
        query=req.query,
        detected_lang=lang,
        answered=False,
        refusal_reason=reason,
        refusal_message=REFUSAL_MESSAGES[reason],
        citations=citations or [],
        guardrails=checks,
        generated_by="none",
        retrieval=retrieval or {},
        timing=timing,
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    tc = TimingCollector()
    ret = get_retriever()
    emb = get_embedder()
    llm = get_llm()
    checks: list[GuardrailCheck] = []

    # --- normalize + input guards (G1, G2) ------------------------------
    with tc.stage("normalize"):
        q = normalize(req.query)
        lang = req.lang or detect_script_lang(q)

    with tc.stage("guards.input"):
        c1, r1 = g1_input_sanity(q)
        checks.append(c1)
        if r1:
            return _refuse(req=req, lang=lang, reason=r1, checks=checks, tc=tc)

        c2, r2 = g2_safety(q)
        checks.append(c2)
        if r2:
            return _refuse(req=req, lang=lang, reason=r2, checks=checks, tc=tc)

    # --- embed ----------------------------------------------------------
    with tc.stage("embed"):
        qvec = emb.embed_query(q)

    # --- G3 off-topic (reuses the vector just computed) -----------------
    with tc.stage("guard.offtopic"):
        c3, r3 = g3_off_topic(qvec, ret.centroids)
        checks.append(c3)
    if r3:
        return _refuse(req=req, lang=lang, reason=r3, checks=checks, tc=tc)

    # --- retrieve + fuse ------------------------------------------------
    with tc.stage("retrieve") as d:
        fused, raw = ret.retrieve(q, qvec, top_k=settings.search_top_k)
        d["n_fused"] = len(fused)
        d["per_strategy"] = {k: len(v) for k, v in raw.items()}

    with tc.stage("fuse"):
        top = fused[: req.top_k]
        flatness = rrf_flatness(fused)

    contexts = [
        {
            "chunk_id": h.best_chunk_id,
            "doc_id": h.doc_id,
            "text": ret.chunk_text(h.best_chunk_id),
            "lang": (ret.chunks.get(h.best_chunk_id) or {}).get("lang"),
            "score": h.best_dense_score,
        }
        for h in top
    ]
    contexts = [c for c in contexts if c["text"]]

    retrieval_meta = {
        "n_candidates": len(fused),
        "top_score": round(top[0].best_dense_score, 4) if top else 0.0,
        "flatness": round(flatness, 4) if flatness != float("inf") else None,
        "per_strategy": {k: len(v) for k, v in raw.items()},
    }
    citations = [
        Citation(doc_id=c["doc_id"], chunk_id=c["chunk_id"], text=c["text"],
                 lang=c["lang"], score=round(c["score"], 4))
        for c in contexts
    ]

    # --- G4 retrieval confidence ----------------------------------------
    with tc.stage("guard.confidence"):
        c4, r4 = g4_retrieval_confidence(
            retrieval_meta["top_score"], flatness
        )
        checks.append(c4)
    if r4 or not contexts:
        return _refuse(
            req=req, lang=lang,
            reason=r4 or RefusalReason.LOW_RETRIEVAL_CONFIDENCE,
            checks=checks, tc=tc, retrieval=retrieval_meta,
            citations=citations,       # show what we DID find; failure stays legible
        )

    # ================= <200 ms SLO boundary ends here ==================

    with tc.stage("generate") as d:
        gen = llm.generate(q, contexts)
        d["model"] = gen.generated_by
        d["provider_ms"] = round(gen.provider_ms, 1)
        if gen.error:
            d["error"] = gen.error[:200]

    if gen.answer is None:
        return _refuse(req=req, lang=lang, reason=RefusalReason.GENERATION_FAILED,
                       checks=checks, tc=tc, retrieval=retrieval_meta,
                       citations=citations)

    # The model may correctly report that the context does not answer the
    # question. Honour that rather than surfacing a hedged non-answer.
    if not gen.answer.used_context:
        checks.append(GuardrailCheck(id="G4b", name="model_abstained", passed=False,
                                     detail="model reported context insufficient"))
        return _refuse(req=req, lang=lang,
                       reason=RefusalReason.LOW_RETRIEVAL_CONFIDENCE,
                       checks=checks, tc=tc, retrieval=retrieval_meta,
                       citations=citations)

    # --- G6 citations (free) then G5 groundedness (~15 ms) --------------
    available = {c["chunk_id"] for c in contexts}
    with tc.stage("guard.citations"):
        c6, r6, valid_ids = g6_validate_citations(gen.answer.cited_chunk_ids, available)
        checks.append(c6)
    if r6:
        return _refuse(req=req, lang=lang, reason=r6, checks=checks, tc=tc,
                       retrieval=retrieval_meta, citations=citations)

    with tc.stage("guard.groundedness"):
        ctx_vecs = np.vstack([
            emb.embed_passages([c["text"] for c in contexts])
        ]) if contexts else np.zeros((0, emb.dim), dtype=np.float32)
        c5, r5, ground_score, support = g5_groundedness(
            gen.answer.answer, ctx_vecs, [c["chunk_id"] for c in contexts], emb
        )
        checks.append(c5)
    if r5:
        return _refuse(req=req, lang=lang, reason=r5, checks=checks, tc=tc,
                       retrieval=retrieval_meta, citations=citations)

    cited = [c for c in citations if c.chunk_id in set(valid_ids)] or citations[:3]

    timing = TimingBreakdown(
        total_ms=round(tc.total_ms(), 3),
        retrieval_ms=round(
            tc.sum_of("normalize", "guards.input", "embed",
                      "guard.offtopic", "retrieve", "fuse", "guard.confidence"), 3
        ),
        generation_ms=round(tc.sum_of("generate"), 3),
        stages=tc.stages,
    )
    latency.record(timing, endpoint="query", outcome="answered", lang=lang)

    return QueryResponse(
        query=req.query,
        detected_lang=lang,
        answered=True,
        answer=gen.answer.answer,
        citations=cited,
        guardrails=checks,
        groundedness=round(ground_score, 3),
        sentence_support=[SentenceSupport(**s) for s in support],
        generated_by=gen.generated_by,  # type: ignore[arg-type]
        tool_hops=gen.tool_hops,
        retrieval=retrieval_meta,
        timing=timing,
    )
