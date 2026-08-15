"""Rebuild a FAISS index from vectors already stored in Qdrant.

Embedding is the expensive step; moving backends should not repeat it. This
pulls the stored vectors and payloads out of a Qdrant collection and writes a
FAISS index from them.

    python scripts/migrate_to_faiss.py --source msmarco_demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.schemas import Chunk, ChunkMetadata, QueryType
from app.retrieval.faiss_store import FaissStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="msmarco_demo", help="Qdrant collection")
    parser.add_argument("--target", default=None, help="FAISS index name")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, timeout=120)
    info = client.get_collection(args.source)
    total = info.points_count
    print(f"source={args.source} points={total}", flush=True)

    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    offset = None

    while True:
        points, offset = client.scroll(
            args.source, limit=1000, offset=offset, with_payload=True, with_vectors=True
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            vector = point.vector if isinstance(point.vector, list) else None
            if not vector:
                continue

            try:
                query_type = QueryType(payload.get("query_type", "UNKNOWN"))
            except ValueError:
                query_type = QueryType.UNKNOWN

            chunks.append(
                Chunk(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    text=payload.get("text", ""),
                    context_text=payload.get("context_text"),
                    metadata=ChunkMetadata(
                        doc_hash=payload.get("doc_hash", ""),
                        language=payload.get("language", ""),
                        flores_code=payload.get("flores_code", ""),
                        is_english=payload.get("is_english", False),
                        query_type=query_type,
                        is_selected=payload.get("is_selected", False),
                        source_query_ids=payload.get("source_query_ids", []),
                        strategy=payload.get("strategy", ""),
                        parent_hash=payload.get("parent_hash"),
                        char_count=payload.get("char_count", 0),
                        position=payload.get("position", 0),
                    ),
                )
            )
            vectors.append(vector)

        print(f"  pulled {len(chunks)}/{total}", flush=True)
        if offset is None:
            break

    if not chunks:
        print("No vectors found — nothing to migrate.", flush=True)
        return 1

    matrix = np.asarray(vectors, dtype=np.float32)
    print(f"matrix {matrix.shape}", flush=True)

    store = FaissStore(
        name=args.target or settings.qdrant_collection,
        dimension=matrix.shape[1],
        binary=True,
    )
    stats = store.build(chunks, matrix)
    print(f"built: {stats}", flush=True)

    # A vector must retrieve itself — the cheapest correctness check there is.
    hits = store.search(matrix[7], limit=3)
    correct = bool(hits) and hits[0].chunk.chunk_id == chunks[7].chunk_id
    print(
        f"self-retrieval: {correct} (score {hits[0].dense_score:.4f})"
        if hits
        else "self-retrieval: NO HITS",
        flush=True,
    )

    latencies = []
    for i in range(50):
        started = time.perf_counter()
        store.search(matrix[i % len(matrix)], limit=10)
        latencies.append((time.perf_counter() - started) * 1000)
    latencies.sort()
    print(
        f"search p50={latencies[25]:.2f}ms "
        f"p90={latencies[45]:.2f}ms p100={latencies[-1]:.2f}ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
