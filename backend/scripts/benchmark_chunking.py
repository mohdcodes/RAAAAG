"""Benchmark every chunking strategy head to head.

Answers the brief's chunking requirement with measurement rather than
assertion: each strategy is chunked, embedded and indexed into its own
collection, then evaluated on the same labelled queries with the same retrieval
settings. The only variable is the chunking.

Usage:
    python scripts/benchmark_chunking.py --language hi --limit 1000
    python scripts/benchmark_chunking.py --language hi --strategies parent_child,semantic

Output is written to data/runs/chunking_benchmark.json, which the UI's
comparison view reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.eval.runner import RetrievalEvaluator
from app.ingest.chunking import available_strategies, get_strategy
from app.ingest.loader import MSMarcoXILoader

logger = get_logger(__name__)

# Every registered strategy except `semantic`, which needs an embedder injected
# and costs an extra forward pass per sentence — included only when asked for.
DEFAULT_STRATEGIES = [
    "passage_native",
    "fixed_size",
    "recursive_character",
    "sentence",
    "sentence_window",
    "parent_child",
    "late_chunking",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare chunking strategies")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--limit", type=int, default=1000, help="Queries to load")
    parser.add_argument("--eval-queries", type=int, default=200, help="Queries to evaluate")
    parser.add_argument("--strategies", default=None, help="Comma-separated subset")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--keep-collections", action="store_true",
                        help="Do not delete per-strategy collections afterwards")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    strategies = (
        [s.strip() for s in args.strategies.split(",")]
        if args.strategies
        else DEFAULT_STRATEGIES
    )
    for name in strategies:
        if name not in available_strategies():
            logger.error("unknown_strategy", name=name, available=available_strategies())
            return 1

    print(f"\n{'=' * 70}")
    print("  CHUNKING STRATEGY BENCHMARK")
    print(f"{'=' * 70}")
    print(f"  language   : {args.language}")
    print(f"  strategies : {', '.join(strategies)}")
    print(f"  eval on    : {args.eval_queries} labelled queries")
    print(f"{'=' * 70}\n")

    # -- Load corpus once; every strategy sees identical input ---------
    loader = MSMarcoXILoader(cache_dir=settings.raw_dir)
    corpus_file = settings.processed_dir / f"corpus_{args.language}.jsonl"
    if corpus_file.exists():
        print(f"  using cached corpus: {corpus_file.name}")
        result = MSMarcoXILoader.load_saved(settings.processed_dir, args.language)
    else:
        print("  downloading corpus...")
        result = loader.build_corpus(args.language, limit=args.limit)
        MSMarcoXILoader.save(result, settings.processed_dir, args.language)

    passages = result.passages
    eval_queries = [q for q in result.queries if q.has_labels][: args.eval_queries]
    print(f"  passages   : {len(passages):,}")
    print(f"  labelled   : {len(eval_queries)} queries\n")

    if not eval_queries:
        print("  No labelled queries — cannot measure accuracy.\n")
        return 1

    from app.retrieval.embedder import get_embedder
    from app.retrieval.vector_store import VectorStore

    embedder = get_embedder()
    embedder._ensure_loaded()
    reranker = None
    if not args.no_rerank:
        from app.retrieval.reranker import get_reranker

        reranker = get_reranker()

    rows: list[dict] = []

    for name in strategies:
        print(f"  {'-' * 66}")
        print(f"  STRATEGY: {name}")
        started = time.perf_counter()

        strategy = get_strategy(name)
        if name == "semantic":
            strategy.set_embedder(embedder.embed_fn())

        chunks, stats = strategy.chunk_all(passages)
        print(f"    chunks       : {stats.output_chunks:,} "
              f"({stats.expansion_ratio:.2f}x, avg {stats.avg_chars:.0f} chars)")

        embed_started = time.perf_counter()
        vectors = embedder.encode_passages(
            [c.text for c in chunks], batch_size=settings.embedding_batch_size
        )
        embed_seconds = time.perf_counter() - embed_started
        print(f"    embedded     : {embed_seconds:.1f}s "
              f"({len(chunks) / embed_seconds:.1f}/sec)")

        collection = f"bench_{name}"
        store = VectorStore(collection=collection, dimension=embedder.dimension)
        store.create_collection(recreate=True)
        store.upsert_chunks(chunks, vectors, batch_size=256)
        time.sleep(2)

        evaluator = RetrievalEvaluator(embedder, store, reranker)
        report = evaluator.evaluate(
            eval_queries,
            top_k=args.top_k,
            candidates=args.candidates,
            use_reranker=not args.no_rerank,
        )

        metrics = report["metrics"]
        latency = report["latency"]["total"]
        print(f"    Recall@10    : {metrics['recall_at_10']:.4f}")
        print(f"    MRR@10       : {metrics['mrr_at_10']:.4f}")
        print(f"    nDCG@10      : {metrics['ndcg_at_10']:.4f}")
        print(f"    latency P50  : {latency['p50']:.1f}ms   P100: {latency['p100']:.1f}ms")

        rows.append(
            {
                "strategy": name,
                "description": strategy.description,
                "metrics": metrics,
                "latency": report["latency"],
                "budget": report["budget"],
                "chunk_count": stats.output_chunks,
                "avg_chunk_chars": round(stats.avg_chars, 1),
                "expansion_ratio": round(stats.expansion_ratio, 3),
                "chunking_seconds": round(stats.duration_seconds, 2),
                "embedding_seconds": round(embed_seconds, 1),
                "total_seconds": round(time.perf_counter() - started, 1),
            }
        )

        if not args.keep_collections:
            store.delete_collection()

    rows.sort(key=lambda r: r["metrics"]["recall_at_10"], reverse=True)

    # -- Summary table -------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  RESULTS (ranked by Recall@10)")
    print(f"{'=' * 70}")
    print(f"  {'strategy':<22}{'R@10':>8}{'MRR@10':>9}{'nDCG@10':>9}{'chunks':>9}{'P50ms':>8}")
    print(f"  {'-' * 66}")
    for row in rows:
        m = row["metrics"]
        print(
            f"  {row['strategy']:<22}{m['recall_at_10']:>8.4f}{m['mrr_at_10']:>9.4f}"
            f"{m['ndcg_at_10']:>9.4f}{row['chunk_count']:>9,}"
            f"{row['latency']['total']['p50']:>8.1f}"
        )
    print(f"{'=' * 70}")
    print(f"  WINNER: {rows[0]['strategy']} "
          f"(Recall@10 {rows[0]['metrics']['recall_at_10']:.4f})")
    print(f"{'=' * 70}\n")

    output = {
        "language": args.language,
        "passages": len(passages),
        "eval_queries": len(eval_queries),
        "embedding_model": settings.embedding_model,
        "reranker_used": not args.no_rerank,
        "top_k": args.top_k,
        "candidates": args.candidates,
        "results": rows,
        "winner": rows[0]["strategy"],
        "caveat": (
            "MS MARCO is_selected labels are sparse: a passage that answers the "
            "query but was never marked relevant counts as a miss, so absolute "
            "recall understates real quality. Figures are comparable between "
            "strategies, which is what this benchmark is for."
        ),
    }
    path = settings.runs_dir / "chunking_benchmark.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
