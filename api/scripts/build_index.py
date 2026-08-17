"""Chunk the corpus with every strategy, embed, and build all indexes.

Produces, in data/index/:
    {strategy}.faiss          one IndexFlatIP per dense strategy
    {strategy}_ids.json       row -> chunk_id (must stay in sync with the above)
    chunks.parquet            every chunk from every strategy + display text
    documents.parquet         parent doc store (copied for self-containment)
    centroids.npy             k-means centroids -> powers guardrail G3
    bm25_*.{npz,npy,json}     lexical index over the sentence chunks
    index_manifest.json       counts + config used, for /stats and the README

Usage:
    python scripts/build_index.py                       # all strategies
    python scripts/build_index.py --strategies fixed,sentence
    python scripts/build_index.py --limit 20000         # fast dev loop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ingest is a THROUGHPUT problem; serving is a LATENCY problem. They want
# opposite thread counts, and this must be set before onnxruntime is imported.
#   serving: 4 threads — more makes a single 384-dim matmul slower, because
#            sync dominates and work lands on E-cores.
#   ingest:  all cores — we are saturating with large batches, not racing a
#            single small one.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking.base import Document  # noqa: E402
from app.chunking.registry import STRATEGIES  # noqa: E402
from app.config import settings  # noqa: E402
from app.embedding.onnx_embedder import get_embedder  # noqa: E402
from app.index.bm25_store import BM25Store  # noqa: E402
from app.index.faiss_store import FaissStore, build_centroids  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--limit", type=int, default=0, help="cap documents (dev loop)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--onnx", default=None,
        help="override ONNX file, e.g. onnx/model.onnx (fp32). This CPU "
             "(Core Ultra, no AVX-512/VNNI) has no hardware int8 path, so the "
             "qint8_avx512_vnni build is emulated and can be SLOWER than fp32.",
    )
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    settings.onnx_intra_op_threads = args.threads
    if args.onnx:
        settings.embed_onnx_file = args.onnx

    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = set(wanted) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"unknown strategies: {unknown}. known: {list(STRATEGIES)}")

    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)

    doc_path = settings.corpus_dir / "documents.parquet"
    if not doc_path.exists():
        raise SystemExit(f"{doc_path} missing — run scripts/build_corpus.py first")

    rows = pq.read_table(doc_path).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    docs = [Document(**{k: v for k, v in r.items() if k in Document.model_fields})
            for r in rows]
    print(f"corpus: {len(docs):,} documents")

    emb = get_embedder()
    manifest: dict = {
        "n_documents": len(docs),
        "embed_model": settings.embed_model_id,
        "embed_onnx": settings.embed_onnx_file,
        "index_type": settings.index_type,
        "dim": settings.embed_dim,
        "fusion_weights": settings.fusion_weights,
        "strategies": {},
    }

    all_chunk_rows: list[dict] = []
    all_vectors: list[np.ndarray] = []      # for the G3 centroids

    for strategy in wanted:
        chunker = STRATEGIES[strategy]
        t0 = time.perf_counter()

        chunks = []
        for doc in tqdm(docs, desc=f"[{strategy}] chunk", unit="doc"):
            chunks.extend(chunker.chunk(doc))
        chunk_ms = (time.perf_counter() - t0) * 1000

        if not chunks:
            print(f"[{strategy}] produced no chunks, skipping")
            continue

        t1 = time.perf_counter()
        texts = [c.text for c in chunks]
        # Length-sorted bulk path: the tokenizer pads every batch to its
        # longest member, so feeding corpus order mixes 40-token and 500-token
        # chunks and computes attention over mostly padding. Sorting first
        # makes each batch internally uniform; order is restored on return.
        vectors = emb.embed_passages_bulk(
            texts, batch_size=args.batch_size, progress=True
        )
        embed_ms = (time.perf_counter() - t1) * 1000
        rate = len(texts) / max(embed_ms / 1000, 1e-9)

        t2 = time.perf_counter()
        store = FaissStore(strategy)
        store.build(vectors, [c.chunk_id for c in chunks])
        store.save(out)
        index_ms = (time.perf_counter() - t2) * 1000

        all_vectors.append(vectors)
        for c in chunks:
            all_chunk_rows.append({
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "strategy": c.strategy,
                "text": c.text, "display_text": c.display_text,
                "ordinal": c.ordinal, "n_tokens": c.n_tokens,
                "lang": c.meta.get("lang"), "variant": c.meta.get("variant"),
                "query_type": c.meta.get("query_type"),
                "query_id": c.meta.get("query_id"),
                "is_selected": c.meta.get("is_selected"),
            })

        toks = sorted(c.n_tokens for c in chunks)
        manifest["strategies"][strategy] = {
            "n_chunks": len(chunks),
            "chunks_per_doc": round(len(chunks) / len(docs), 3),
            "tokens_p50": toks[len(toks) // 2],
            "tokens_max": toks[-1],
            "chunk_ms": round(chunk_ms),
            "embed_ms": round(embed_ms),
            "index_ms": round(index_ms),
            "vectors_mb": round(vectors.nbytes / 1e6, 1),
        }
        manifest["strategies"][strategy]["embed_chunks_per_s"] = round(rate, 1)
        print(f"[{strategy}] {len(chunks):,} chunks | chunk {chunk_ms/1000:.0f}s | "
              f"embed {embed_ms/1000:.0f}s ({rate:.0f} chunks/s) | "
              f"{vectors.nbytes/1e6:.0f} MB")

    # --- shared artifacts ------------------------------------------------

    pq.write_table(pa.Table.from_pylist(all_chunk_rows),
                   out / "chunks.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(rows),
                   out / "documents.parquet", compression="zstd")

    # BM25 over the sentence chunks (whole-sentence units are the right lexical
    # granularity; fixed windows would split terms across chunk boundaries).
    bm_src = [r for r in all_chunk_rows if r["strategy"] == "sentence"] or all_chunk_rows
    bm = BM25Store()
    t3 = time.perf_counter()
    bm.build([r["text"] for r in bm_src], [r["chunk_id"] for r in bm_src])
    bm.save(out)
    manifest["bm25"] = {
        "n_chunks": len(bm_src),
        "vocab_size": len(bm.vocab),
        "build_ms": round((time.perf_counter() - t3) * 1000),
        "matrix_mb": round(bm.matrix.data.nbytes / 1e6, 1),
    }
    print(f"[bm25] {len(bm_src):,} chunks, vocab {len(bm.vocab):,}")

    stacked = np.vstack(all_vectors)
    cents = build_centroids(stacked)
    np.save(out / "centroids.npy", cents)
    manifest["centroids"] = {"k": int(cents.shape[0]), "dim": int(cents.shape[1])}
    print(f"[centroids] {cents.shape[0]} x {cents.shape[1]} -> centroids.npy")

    manifest["total_chunks"] = len(all_chunk_rows)
    manifest["total_vectors_mb"] = round(stacked.nbytes / 1e6, 1)
    (out / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*64}")
    print(f"total chunks : {len(all_chunk_rows):,}")
    print(f"total vectors: {stacked.nbytes/1e6:.0f} MB resident")
    print(f"artifacts    : {out}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
