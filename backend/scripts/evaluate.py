"""Measure retrieval accuracy and latency percentiles.

    python scripts/evaluate.py --queries 200

Writes data/runs/chunking_benchmark.json, which the Metrics tab reads, and
prints a summary. Accuracy is scored against the dataset's own `is_selected`
labels — the same ground truth MS MARCO ships.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# FAISS and torch both bundle OpenMP; on Windows loading both aborts without
# this. Must precede any import that pulls either in.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.eval.runner import RetrievalEvaluator, format_report
from app.ingest.loader import MSMarcoXILoader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="hi")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--rerank", action="store_true", help="Enable the cross-encoder")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")

    result = MSMarcoXILoader.load_saved(settings.processed_dir, args.language)
    labelled = [q for q in result.queries if q.has_labels][: args.queries]
    if not labelled:
        print("No labelled queries found — run the ingest first.")
        return 1

    from app.retrieval.embedder import get_embedder
    from app.retrieval.faiss_store import get_faiss_store

    embedder = get_embedder()
    embedder._ensure_loaded()

    store = get_faiss_store()
    if not store.ensure_loaded() or store.count() == 0:
        print("No FAISS index found — run the ingest first.")
        return 1

    reranker = None
    if args.rerank:
        from app.retrieval.reranker import get_reranker

        reranker = get_reranker()

    print(f"evaluating {len(labelled)} queries against {store.count():,} vectors…\n")

    evaluator = RetrievalEvaluator(embedder, store, reranker)
    report = evaluator.evaluate(
        labelled,
        top_k=args.top_k,
        candidates=args.candidates,
        use_reranker=args.rerank,
    )
    print(format_report(report))

    # Shape it like the benchmark file so the Metrics tab can read either.
    payload = {
        "language": args.language,
        "eval_queries": report["metrics"]["queries_evaluated"],
        "embedding_model": settings.embedding_model,
        "reranker_used": args.rerank,
        "results": [
            {
                "strategy": settings.chunking_strategy,
                "description": "active index",
                "metrics": report["metrics"],
                "latency": report["latency"],
                "budget": report["budget"],
                "chunk_count": store.count(),
                "avg_chunk_chars": 0,
                "expansion_ratio": 0,
                "embedding_seconds": 0,
            }
        ],
        "winner": settings.chunking_strategy,
        "caveat": (
            "MS MARCO labels are sparse — a passage that answers the query but "
            "was never marked relevant counts as a miss, so absolute recall "
            "understates real quality."
        ),
    }
    path = settings.runs_dir / "chunking_benchmark.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
