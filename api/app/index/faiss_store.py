"""FAISS vector store — one independent index per chunking strategy.

Why one index PER STRATEGY rather than a single index with a `strategy` field:
RRF needs each strategy's OWN ranked list. A single merged index returns one
merged ordering, from which per-strategy ranks cannot be recovered — that
would break both the fusion math and the ablation table that justifies the
whole chunking design.

Why IndexFlatIP (exact) rather than IVF/HNSW at this scale:
  1. Exact search is ~5-15 ms per index here, under 10% of the 200 ms budget.
     ANN buys speed we do not need and pays for it in recall.
  2. The G3/G4 guardrail thresholds are calibrated against absolute cosine
     values. Approximate search perturbs those query-to-query, which would make
     the refusal threshold drift and produce random "I don't know" answers.
  3. Approximate recall would confound the ablation: with ~0.95 per-index
     recall you cannot distinguish "this strategy is worse" from "this index
     missed it".

Vectors are L2-normalized before `add`, so inner product IS cosine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.config import settings

os.environ.setdefault("OMP_NUM_THREADS", str(settings.omp_threads))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

faiss.omp_set_num_threads(settings.omp_threads)


class FaissStore:
    """A single strategy's vector index plus its row -> chunk_id mapping.

    FAISS returns row ordinals, never your identifiers. `ids[row] -> chunk_id`
    is what converts them back. If the index and the id map are ever rebuilt
    out of step, lookups silently return the WRONG text — so both are written
    and loaded together, and `load` asserts their lengths match.
    """

    def __init__(self, strategy: str, dim: int | None = None) -> None:
        self.strategy = strategy
        self.dim = dim or settings.embed_dim
        self.index: faiss.Index | None = None
        self.ids: list[str] = []

    # --- build ---------------------------------------------------------

    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        if len(vectors) != len(chunk_ids):
            raise ValueError(
                f"{self.strategy}: {len(vectors)} vectors vs {len(chunk_ids)} ids"
            )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        # Defensive: normalize even though the embedder already did, because
        # IP-as-cosine silently degrades to magnitude ranking otherwise.
        faiss.normalize_L2(vectors)

        if settings.index_type == "hnsw":
            base = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = 200
        else:
            base = faiss.IndexFlatIP(self.dim)

        self.index = faiss.IndexIDMap2(base)
        self.index.add_with_ids(vectors, np.arange(len(chunk_ids), dtype=np.int64))
        self.ids = list(chunk_ids)

    # --- persistence ---------------------------------------------------

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dirpath / f"{self.strategy}.faiss"))
        (dirpath / f"{self.strategy}_ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, dirpath: Path) -> "FaissStore":
        self.index = faiss.read_index(str(dirpath / f"{self.strategy}.faiss"))
        self.ids = json.loads(
            (dirpath / f"{self.strategy}_ids.json").read_text(encoding="utf-8")
        )
        if self.index.ntotal != len(self.ids):
            raise RuntimeError(
                f"{self.strategy}: index has {self.index.ntotal} vectors but id map "
                f"has {len(self.ids)} entries — rebuild the index; lookups would "
                f"return wrong text."
            )
        return self

    # --- search --------------------------------------------------------

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine)] best-first."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
        k = min(k, self.index.ntotal)
        scores, rows = self.index.search(q, k)
        out: list[tuple[str, float]] = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            out.append((self.ids[int(row)], float(score)))
        return out

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0


def build_centroids(vectors: np.ndarray, k: int | None = None) -> np.ndarray:
    """K-means centroids over the corpus, for the G3 off-topic gate.

    This is the entire cost of off-topic detection: a 64x384 matvec against the
    query vector we already computed. No extra model, no extra network call, no
    extra embedding pass — which is precisely why it fits the latency budget.
    """
    k = k or settings.kmeans_centroids
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    k = min(k, max(1, len(vectors) // 10))
    km = faiss.Kmeans(vectors.shape[1], k, niter=20, verbose=False, seed=1234)
    km.train(vectors)
    cents = np.ascontiguousarray(km.centroids, dtype=np.float32)
    faiss.normalize_L2(cents)
    return cents
