"""Ingest pipeline: dataset -> chunks -> embeddings -> Qdrant.

Usage:
    python scripts/ingest.py --languages hi --limit 500
    python scripts/ingest.py --languages hi,ta,bn --limit 5000 --strategy parent_child
    python scripts/ingest.py --all-languages --limit 20000

Embedding dominates the runtime on CPU, so the corpus is cached to disk after
the download/dedup step. Re-running with the same languages skips straight to
embedding unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a plain script from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.core.config import get_settings
from app.core.languages import ALL_CODES, get_language
from app.core.logging import configure_logging, get_logger
from app.ingest.chunking import get_strategy
from app.ingest.chunking.base import SourcePassage
from app.ingest.loader import MSMarcoXILoader

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI into Qdrant")
    parser.add_argument(
        "--languages", default="hi",
        help="Comma-separated language codes (e.g. hi,ta,bn)",
    )
    parser.add_argument("--all-languages", action="store_true", help="Ingest all 14")
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Max queries per language (each yields up to ~20 passages)",
    )
    parser.add_argument("--strategy", default=None, help="Chunking strategy")
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--collection", default=None, help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size")
    parser.add_argument("--recreate", action="store_true", help="Drop the collection first")
    parser.add_argument("--refresh", action="store_true", help="Re-download, ignore cache")
    parser.add_argument("--no-english", action="store_true", help="Skip English passages")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only, do not embed")
    return parser.parse_args()


def load_passages(
    loader: MSMarcoXILoader,
    languages: list[str],
    *,
    limit: int,
    split: str,
    include_english: bool,
    processed_dir: Path,
    refresh: bool,
) -> tuple[list[SourcePassage], list, dict]:
    """Load and dedup passages across languages, caching per language."""
    all_passages: dict[str, SourcePassage] = {}
    all_queries = []
    per_language: dict[str, dict] = {}

    for language in languages:
        cache_marker = processed_dir / f"corpus_{language}.jsonl"
        use_cache = cache_marker.exists() and not refresh

        if use_cache:
            logger.info("using_cached_corpus", language=language)
            result = MSMarcoXILoader.load_saved(processed_dir, language)
        else:
            logger.info("downloading", language=language, limit=limit, split=split)
            started = time.perf_counter()
            result = loader.build_corpus(
                language, split=split, limit=limit, include_english=include_english
            )
            result.stats["download_seconds"] = round(time.perf_counter() - started, 2)
            MSMarcoXILoader.save(result, processed_dir, language)

        per_language[language] = result.stats
        all_queries.extend(result.queries)

        # Cross-language dedup: English passages are shared across every
        # language file, so without this they would be indexed 14 times.
        for passage in result.passages:
            existing = all_passages.get(passage.doc_hash)
            if existing is None:
                all_passages[passage.doc_hash] = passage
            else:
                existing.source_query_ids.extend(passage.source_query_ids)
                existing.is_selected = existing.is_selected or passage.is_selected

    return list(all_passages.values()), all_queries, per_language


def embed_chunks(embedder, chunks, batch_size: int) -> np.ndarray:
    """Embed chunk texts with progress reporting."""
    texts = [chunk.text for chunk in chunks]
    total = len(texts)
    logger.info("embedding_start", chunks=total, batch_size=batch_size)

    started = time.perf_counter()
    vectors: list[np.ndarray] = []
    report_every = max(1, (total // batch_size) // 20)

    for index, start in enumerate(range(0, total, batch_size)):
        batch = texts[start : start + batch_size]
        vectors.append(embedder.encode_passages(batch, batch_size=batch_size))

        if index % report_every == 0 and start:
            elapsed = time.perf_counter() - started
            rate = start / elapsed
            remaining = (total - start) / rate if rate else 0
            logger.info(
                "embedding_progress",
                done=start, total=total,
                rate_per_sec=round(rate, 1),
                eta_seconds=round(remaining),
            )

    matrix = np.vstack(vectors)
    elapsed = time.perf_counter() - started
    logger.info(
        "embedding_complete",
        chunks=total, seconds=round(elapsed, 1),
        rate_per_sec=round(total / elapsed, 1) if elapsed else 0,
    )
    return matrix


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    languages = (
        list(ALL_CODES) if args.all_languages
        else [code.strip() for code in args.languages.split(",") if code.strip()]
    )
    for code in languages:
        if get_language(code) is None:
            logger.error("unknown_language", code=code, known=list(ALL_CODES))
            return 1

    strategy_name = args.strategy or settings.chunking_strategy
    collection = args.collection or settings.qdrant_collection
    batch_size = args.batch_size or settings.embedding_batch_size

    print(f"\n{'=' * 62}")
    print("  MSMARCO-XI INGEST")
    print(f"{'=' * 62}")
    print(f"  languages : {', '.join(languages)}")
    print(f"  limit     : {args.limit} queries/language")
    print(f"  strategy  : {strategy_name}")
    print(f"  collection: {collection}")
    print(f"{'=' * 62}\n")

    overall = time.perf_counter()

    # -- 1. Load ------------------------------------------------------
    loader = MSMarcoXILoader(cache_dir=settings.raw_dir)
    try:
        passages, queries, per_language = load_passages(
            loader, languages,
            limit=args.limit, split=args.split,
            include_english=not args.no_english,
            processed_dir=settings.processed_dir,
            refresh=args.refresh,
        )
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("load_failed", error=str(exc))
        print(f"\nFAILED during download: {exc}\n")
        return 1

    if not passages:
        print("\nNo passages loaded — nothing to index.\n")
        return 1

    print(f"  unique passages : {len(passages):,}")
    print(f"  queries         : {len(queries):,}")

    # -- 2. Chunk -----------------------------------------------------
    strategy = get_strategy(strategy_name)
    if strategy_name == "semantic":
        from app.retrieval.embedder import get_embedder

        strategy.set_embedder(get_embedder().embed_fn())

    chunks, stats = strategy.chunk_all(passages)
    print(f"  chunks          : {stats.output_chunks:,} "
          f"({stats.expansion_ratio:.2f}x expansion, avg {stats.avg_chars:.0f} chars)")

    if args.dry_run:
        print("\n--dry-run: stopping before embedding.\n")
        return 0

    # -- 3. Embed -----------------------------------------------------
    from app.retrieval.embedder import get_embedder

    embedder = get_embedder()
    print(f"\n  loading {settings.embedding_model} (first run downloads ~2GB)...")
    embedder._ensure_loaded()
    print(f"  device: {embedder.device}, dim: {embedder.dimension}\n")

    vectors = embed_chunks(embedder, chunks, batch_size)

    # -- 4. Index -----------------------------------------------------
    from app.retrieval.vector_store import VectorStore

    store = VectorStore(collection=collection, dimension=embedder.dimension)
    store.create_collection(recreate=args.recreate)
    indexed = store.upsert_chunks(chunks, vectors, batch_size=256)

    time.sleep(2)  # let Qdrant finish indexing before counting
    count = store.count()

    # -- 5. Report ----------------------------------------------------
    elapsed = time.perf_counter() - overall
    manifest = {
        "languages": languages,
        "limit_per_language": args.limit,
        "split": args.split,
        "strategy": strategy_name,
        "collection": collection,
        "unique_passages": len(passages),
        "queries": len(queries),
        "chunks": stats.output_chunks,
        "expansion_ratio": round(stats.expansion_ratio, 3),
        "avg_chunk_chars": round(stats.avg_chars, 1),
        "vectors_indexed": indexed,
        "collection_count": count,
        "embedding_model": settings.embedding_model,
        "dimension": embedder.dimension,
        "binary_quantization": settings.binary_quantization,
        "total_seconds": round(elapsed, 1),
        "per_language": per_language,
    }
    manifest_path = settings.runs_dir / f"ingest_{collection}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 62}")
    print("  INGEST COMPLETE")
    print(f"{'=' * 62}")
    print(f"  vectors in collection : {count:,}")
    print(f"  elapsed               : {elapsed / 60:.1f} min")
    print(f"  manifest              : {manifest_path}")
    print(f"{'=' * 62}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
