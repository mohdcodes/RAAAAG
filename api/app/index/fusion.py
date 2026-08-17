"""Reciprocal Rank Fusion across chunking strategies.

    RRF(d) = sum_s  w_s / (k + rank_s(d))        k = 60

Two design decisions carry the weight here.

**Why RRF rather than normalizing and averaging scores.** The scores are not
commensurable. Cosine distributions shift systematically with chunk length —
semantic's ~344-token chunks score differently from fixed's ~258-token ones for
the same query — and BM25 is unbounded, with a scale that depends on corpus
statistics. Min-max or z-score normalization across those is arbitrary, and
unstable per query: one outlier reshapes the whole distribution. RRF consumes
only RANKS, so it is scale-free, needs no calibration, and degrades gracefully
if one strategy returns garbage.

**Why fuse at the DOCUMENT level, not the chunk level.** A document takes its
best rank within each strategy, then those ranks fuse. Chunk-level fusion lets
a strategy that emits many overlapping chunks stuff the top of the result list
with near-duplicates of one passage, drowning out documents that several
independent strategies agreed on. Document-level fusion rewards exactly the
signal we want: agreement ACROSS strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings


@dataclass
class FusedHit:
    doc_id: str
    rrf_score: float
    best_chunk_id: str
    best_dense_score: float
    strategies: dict[str, int] = field(default_factory=dict)   # strategy -> best rank
    per_strategy_score: dict[str, float] = field(default_factory=dict)


def rrf_fuse(
    ranked_lists: dict[str, list[tuple[str, float]]],
    chunk_to_doc: dict[str, str],
    weights: dict[str, float] | None = None,
    k: int | None = None,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Fuse per-strategy chunk rankings into one document ranking.

    Args:
        ranked_lists: strategy -> [(chunk_id, score)] ordered best-first.
        chunk_to_doc: chunk_id -> doc_id.
        weights: per-strategy weight; missing strategies default to 1.0.
        k: RRF constant. 60 is the standard value from Cormack et al.; it damps
           the difference between ranks 1 and 2 so a single strategy's top hit
           cannot dominate agreement across several.
    """
    weights = weights or settings.fusion_weights
    k = k or settings.rrf_k

    best_rank: dict[str, dict[str, int]] = {}
    best_score: dict[str, dict[str, float]] = {}
    best_chunk: dict[str, tuple[str, float]] = {}

    for strategy, hits in ranked_lists.items():
        seen_docs: set[str] = set()
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            doc_id = chunk_to_doc.get(chunk_id)
            if doc_id is None:
                continue
            # Only a document's BEST rank within this strategy counts, so a
            # strategy emitting five overlapping chunks of one passage votes
            # once, not five times.
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            best_rank.setdefault(doc_id, {})[strategy] = rank
            best_score.setdefault(doc_id, {})[strategy] = score

            # Track the single best-scoring chunk across all DENSE strategies
            # for display; BM25 scores are on a different scale entirely.
            if strategy != "bm25":
                cur = best_chunk.get(doc_id)
                if cur is None or score > cur[1]:
                    best_chunk[doc_id] = (chunk_id, score)

    fused: list[FusedHit] = []
    for doc_id, ranks in best_rank.items():
        rrf = sum(weights.get(s, 1.0) / (k + r) for s, r in ranks.items())
        chunk_id, dense = best_chunk.get(doc_id, (next(iter(ranks)), 0.0))
        fused.append(
            FusedHit(
                doc_id=doc_id,
                rrf_score=rrf,
                best_chunk_id=chunk_id,
                best_dense_score=dense,
                strategies=dict(ranks),
                per_strategy_score=best_score.get(doc_id, {}),
            )
        )

    fused.sort(key=lambda h: (-h.rrf_score, -h.best_dense_score))
    return fused[:top_n] if top_n else fused


def rrf_flatness(fused: list[FusedHit], depth: int | None = None) -> float:
    """rrf[0] / rrf[depth-1] — how much the top result actually stands out.

    Feeds guardrail G4. Scores can all be individually acceptable while the
    retriever has no real opinion about which is best; that is a different
    failure from "everything scored low" and should also refuse.

    IMPORTANT — this ratio is tightly bounded, which is easy to get wrong.
    With k=60, a document ranked 1st in every strategy versus one ranked Nth
    in every strategy differs by only (60+N)/(60+1):

        N=5  -> 1.066      N=10 -> 1.147
        N=20 -> 1.312      N=50 -> 1.803

    So any threshold above ~1.07 at depth 5 is unreachable and refuses
    everything. Depth 10 is used to widen the usable range.
    """
    depth = depth or settings.flatness_depth
    if len(fused) < depth:
        return float("inf") if fused else 0.0
    tail = fused[depth - 1].rrf_score
    return fused[0].rrf_score / tail if tail > 0 else float("inf")


def mmr_diversify(
    fused: list[FusedHit],
    doc_vectors: dict[str, "object"],
    lambda_: float | None = None,
    top_n: int = 10,
) -> list[FusedHit]:
    """Maximal Marginal Relevance over the fused list.

    MS MARCO passages repeat heavily across queries, so the top-k can easily be
    five near-identical restatements of one fact — which wastes the LLM's
    context window and makes the source list look broken to a judge.
    """
    import numpy as np

    lambda_ = lambda_ if lambda_ is not None else settings.mmr_lambda
    if not fused:
        return []

    selected: list[FusedHit] = [fused[0]]
    candidates = fused[1:]

    while candidates and len(selected) < top_n:
        best_i, best_val = 0, -1e9
        for i, cand in enumerate(candidates):
            cv = doc_vectors.get(cand.doc_id)
            if cv is None:
                redundancy = 0.0
            else:
                sims = [
                    float(np.dot(cv, doc_vectors[s.doc_id]))
                    for s in selected
                    if s.doc_id in doc_vectors
                ]
                redundancy = max(sims) if sims else 0.0
            val = lambda_ * cand.rrf_score * 100 - (1 - lambda_) * redundancy
            if val > best_val:
                best_i, best_val = i, val
        selected.append(candidates.pop(best_i))

    return selected
