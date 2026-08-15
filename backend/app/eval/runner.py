"""Evaluation runner.

Measures two things the brief asks for, on the same set of queries:

  1. retrieval accuracy — Recall@1/5/10, MRR@10, nDCG@10 against is_selected
  2. latency percentiles — P50/P70/P90/P95/P100 per stage

Running both together matters: an accuracy number from a different query set
than the latency number invites the suspicion that each was measured under
whichever conditions flattered it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.core.schemas import LatencyPercentiles, RetrievalMetrics
from app.core.timing import percentile
from app.eval.metrics import aggregate
from app.ingest.loader import EvalQuery

logger = get_logger(__name__)


class RetrievalEvaluator:
    """Runs queries through retrieval and scores them against qrels."""

    def __init__(self, embedder, vector_store, reranker=None) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def evaluate(
        self,
        queries: Sequence[EvalQuery],
        *,
        top_k: int = 10,
        candidates: int = 200,
        strategy: str | None = None,
        use_reranker: bool = True,
        warmup: int = 3,
    ) -> dict[str, Any]:
        """Evaluate retrieval over labelled queries.

        The first few queries are discarded as warmup: lazy model loading and
        cold caches make them wildly unrepresentative, and including them would
        inflate P100 by an order of magnitude.
        """
        labelled = [q for q in queries if q.has_labels]
        if not labelled:
            raise ValueError("No queries carry relevance labels.")

        logger.info(
            "eval_start",
            queries=len(labelled), top_k=top_k,
            reranker=bool(self.reranker and use_reranker),
        )

        results: list[tuple[list[str], set[str]]] = []
        timings: dict[str, list[float]] = {
            "embed": [], "search": [], "rerank": [], "total": []
        }
        per_query: list[dict[str, Any]] = []

        for index, query in enumerate(labelled):
            is_warmup = index < warmup
            query_start = time.perf_counter()

            t0 = time.perf_counter()
            vector = self.embedder.encode_queries([query.text])[0]
            embed_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            retrieved = self.vector_store.search(
                vector, limit=candidates, strategy=strategy
            )
            search_ms = (time.perf_counter() - t0) * 1000

            rerank_ms = 0.0
            if self.reranker is not None and use_reranker and retrieved:
                t0 = time.perf_counter()
                retrieved = self.reranker.rerank(query.text, retrieved, top_k=top_k)
                rerank_ms = (time.perf_counter() - t0) * 1000
            else:
                retrieved = retrieved[:top_k]

            total_ms = (time.perf_counter() - query_start) * 1000

            # Chunks map back to source passages by doc_hash, which is what the
            # qrels are keyed on. Dedup preserves rank order.
            doc_ids: list[str] = []
            for item in retrieved:
                doc_hash = item.chunk.metadata.doc_hash
                if doc_hash not in doc_ids:
                    doc_ids.append(doc_hash)

            if not is_warmup:
                results.append((doc_ids, set(query.relevant_hashes)))
                timings["embed"].append(embed_ms)
                timings["search"].append(search_ms)
                timings["rerank"].append(rerank_ms)
                timings["total"].append(total_ms)
                per_query.append(
                    {
                        "query_id": query.query_id,
                        "query": query.text[:120],
                        "language": query.language,
                        "relevant": len(query.relevant_hashes),
                        "hit_at_1": bool(doc_ids[:1]) and doc_ids[0] in query.relevant_hashes,
                        "hit_at_10": any(d in query.relevant_hashes for d in doc_ids[:10]),
                        "total_ms": round(total_ms, 2),
                    }
                )

            if index % 50 == 0 and index:
                logger.info("eval_progress", done=index, total=len(labelled))

        metrics = aggregate(results)
        latency = {
            stage: self._percentiles(stage, values)
            for stage, values in timings.items()
        }

        within_budget = sum(1 for t in timings["total"] if t < 200.0)
        summary = {
            "metrics": metrics.model_dump(),
            "latency": {k: v.model_dump() for k, v in latency.items()},
            "budget": {
                "threshold_ms": 200.0,
                "within": within_budget,
                "total": len(timings["total"]),
                "percentage": round(
                    100.0 * within_budget / len(timings["total"]), 2
                ) if timings["total"] else 0.0,
                "measures": "embed + search + rerank (no LLM generation)",
            },
            "config": {
                "top_k": top_k,
                "candidates": candidates,
                "strategy": strategy,
                "reranker_used": bool(self.reranker and use_reranker),
                "warmup_discarded": warmup,
            },
            "per_query": per_query[:200],
        }

        logger.info(
            "eval_complete",
            recall_at_10=round(metrics.recall_at_10, 4),
            mrr_at_10=round(metrics.mrr_at_10, 4),
            p50_ms=latency["total"].p50,
        )
        return summary

    @staticmethod
    def _percentiles(stage: str, values: list[float]) -> LatencyPercentiles:
        if not values:
            return LatencyPercentiles(
                stage=stage, p50=0, p70=0, p90=0, p95=0, p100=0, mean=0, samples=0
            )
        return LatencyPercentiles(
            stage=stage,
            p50=round(percentile(values, 50), 3),
            p70=round(percentile(values, 70), 3),
            p90=round(percentile(values, 90), 3),
            p95=round(percentile(values, 95), 3),
            p100=round(percentile(values, 100), 3),
            mean=round(sum(values) / len(values), 3),
            samples=len(values),
        )


def save_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("eval_report_saved", path=str(path))
    return path


def format_report(report: dict[str, Any]) -> str:
    """Human-readable summary for the terminal and the submission."""
    metrics: dict = report["metrics"]
    latency: dict = report["latency"]
    budget: dict = report["budget"]

    lines = [
        "",
        "=" * 64,
        "  RETRIEVAL EVALUATION",
        "=" * 64,
        f"  queries evaluated : {metrics['queries_evaluated']}",
        "",
        "  ACCURACY (vs MS MARCO is_selected labels)",
        f"    Recall@1        : {metrics['recall_at_1']:.4f}",
        f"    Recall@5        : {metrics['recall_at_5']:.4f}",
        f"    Recall@10       : {metrics['recall_at_10']:.4f}",
        f"    MRR@10          : {metrics['mrr_at_10']:.4f}",
        f"    nDCG@10         : {metrics['ndcg_at_10']:.4f}",
        "",
        "  LATENCY (ms)",
        f"    {'stage':<10} {'P50':>9} {'P70':>9} {'P90':>9} {'P100':>9}",
    ]
    for stage in ("embed", "search", "rerank", "total"):
        p = latency.get(stage)
        if p:
            lines.append(
                f"    {stage:<10} {p['p50']:>9.2f} {p['p70']:>9.2f} "
                f"{p['p90']:>9.2f} {p['p100']:>9.2f}"
            )

    lines += [
        "",
        f"  BUDGET: {budget['within']}/{budget['total']} queries under 200ms "
        f"({budget['percentage']}%)",
        f"          measures {budget['measures']}",
        "=" * 64,
        "",
    ]
    return "\n".join(lines)
