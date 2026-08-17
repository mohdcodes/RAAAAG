"""The retriever: searches every strategy in parallel and fuses the results.

Concurrency note that drives the whole latency budget: FAISS releases the GIL
during `search`, so running the four dense indexes in a ThreadPoolExecutor
gives real parallelism. Wall time is therefore max(per-index), not the sum —
about 12 ms instead of about 48 ms. That single fact is what leaves room for
generation inside the budget.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from app.chunking.registry import DENSE_STRATEGIES
from app.config import settings
from app.core.timing import TimingCollector
from app.index.bm25_store import BM25Store
from app.index.faiss_store import FaissStore
from app.index.fusion import FusedHit, rrf_fuse


class MultiIndexRetriever:
    """Loads every index once at startup and serves queries from RAM.

    One process-wide instance. `uvicorn --workers 1` is not negotiable: each
    worker would hold its own full copy of every index.
    """

    def __init__(self, index_dir: Path | None = None) -> None:
        self.dir = index_dir or settings.index_dir
        self.stores: dict[str, FaissStore] = {}
        self.bm25: BM25Store | None = None
        self.centroids: np.ndarray | None = None
        self.chunks: dict[str, dict[str, Any]] = {}
        self.docs: dict[str, dict[str, Any]] = {}
        self.chunk_to_doc: dict[str, str] = {}
        self.manifest: dict[str, Any] = {}
        self._pool = ThreadPoolExecutor(max_workers=len(DENSE_STRATEGIES) + 1)

    def load(self) -> "MultiIndexRetriever":
        if not self.dir.exists():
            raise RuntimeError(
                f"{self.dir} missing — run scripts/build_corpus.py then "
                f"scripts/build_index.py"
            )

        for strategy in DENSE_STRATEGIES:
            if (self.dir / f"{strategy}.faiss").exists():
                self.stores[strategy] = FaissStore(strategy).load(self.dir)

        if (self.dir / "bm25_matrix.npz").exists():
            self.bm25 = BM25Store().load(self.dir)

        if (self.dir / "centroids.npy").exists():
            self.centroids = np.load(self.dir / "centroids.npy")

        for row in pq.read_table(self.dir / "chunks.parquet").to_pylist():
            self.chunks[row["chunk_id"]] = row
            self.chunk_to_doc[row["chunk_id"]] = row["doc_id"]

        for row in pq.read_table(self.dir / "documents.parquet").to_pylist():
            self.docs[row["doc_id"]] = row

        mpath = self.dir / "index_manifest.json"
        if mpath.exists():
            self.manifest = json.loads(mpath.read_text(encoding="utf-8"))

        if not self.stores:
            raise RuntimeError(f"no FAISS indexes found in {self.dir}")
        return self

    # --- search ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        qvec: np.ndarray,
        top_k: int | None = None,
        strategies: list[str] | None = None,
        timing: TimingCollector | None = None,
    ) -> tuple[list[FusedHit], dict[str, list[tuple[str, float]]]]:
        """Search all strategies concurrently, then fuse. Returns (fused, raw)."""
        top_k = top_k or settings.search_top_k
        active = [s for s in (strategies or list(self.stores)) if s in self.stores]

        def dense(strategy: str) -> tuple[str, list[tuple[str, float]]]:
            return strategy, self.stores[strategy].search(qvec, top_k)

        futures = [self._pool.submit(dense, s) for s in active]
        raw: dict[str, list[tuple[str, float]]] = {}

        if self.bm25 is not None and (strategies is None or "bm25" in strategies):
            bm_future = self._pool.submit(self.bm25.search, query, top_k)
        else:
            bm_future = None

        for f in futures:
            name, hits = f.result()
            raw[name] = hits
        if bm_future is not None:
            raw["bm25"] = bm_future.result()

        if timing is not None:
            for name, hits in raw.items():
                timing.record(f"retrieve.{name}", 0.0, n_hits=len(hits))

        fused = rrf_fuse(raw, self.chunk_to_doc, weights=settings.fusion_weights)
        return fused, raw

    # --- accessors ------------------------------------------------------

    def chunk_text(self, chunk_id: str) -> str:
        """Display text — for S4 this strips the synthesized metadata header,
        which must never reach the LLM or the user."""
        c = self.chunks.get(chunk_id)
        return (c or {}).get("display_text") or (c or {}).get("text") or ""

    def doc(self, doc_id: str) -> dict[str, Any]:
        return self.docs.get(doc_id, {})

    def stats(self) -> dict[str, Any]:
        return {
            "strategies": {s: st.ntotal for s, st in self.stores.items()},
            "bm25_chunks": len(self.bm25.ids) if self.bm25 else 0,
            "bm25_vocab": len(self.bm25.vocab) if self.bm25 else 0,
            "documents": len(self.docs),
            "chunks": len(self.chunks),
            "centroids": int(self.centroids.shape[0]) if self.centroids is not None else 0,
            "manifest": self.manifest,
        }


_retriever: MultiIndexRetriever | None = None


def get_retriever() -> MultiIndexRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MultiIndexRetriever().load()
    return _retriever
