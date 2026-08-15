"""FAISS vector store.

In-process, no server, no Docker. The index is a file the API memory-maps at
startup, which removes a whole class of deployment problems and takes the
network hop out of the query path entirely — the largest single term in the
latency budget after embedding.

Binary quantization is implemented directly here rather than delegated:

  * `IndexBinaryFlat` over bit-packed vectors, searched by Hamming distance
  * an oversampled candidate set rescored against full-precision float vectors

That is the same two-stage design Qdrant runs internally, but with the vectors
in the same process, so a search is a function call instead of an HTTP request.
"""

from __future__ import annotations

import json
import os
import pickle
import time

# FAISS and PyTorch each ship an OpenMP runtime; loading both aborts the
# process on Windows. Set before `import faiss` anywhere in the process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import Chunk, ChunkMetadata, QueryType, RetrievedChunk

logger = get_logger(__name__)


@dataclass(slots=True)
class StoredChunk:
    """Payload kept alongside each vector."""

    chunk_id: str
    text: str
    context_text: str | None
    doc_hash: str
    language: str
    flores_code: str
    is_english: bool
    query_type: str
    is_selected: bool
    strategy: str
    parent_hash: str | None
    char_count: int
    position: int
    source_query_ids: list[int] = field(default_factory=list)


class FaissStore:
    """Binary-quantized FAISS index with full-precision rescoring."""

    def __init__(
        self,
        index_dir: Path | None = None,
        name: str | None = None,
        dimension: int | None = None,
        *,
        binary: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.index_dir = index_dir or settings.index_dir
        self.name = name or settings.qdrant_collection
        self.dimension = dimension or settings.embedding_dim
        self.binary = settings.binary_quantization if binary is None else binary

        self._binary_index = None  # IndexBinaryFlat
        self._float_vectors: np.ndarray | None = None  # for rescoring
        self._chunks: list[StoredChunk] = []
        # language -> row indices, for metadata-filtered search
        self._by_language: dict[str, np.ndarray] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.index_dir / f"{self.name}.faiss"

    @property
    def _vectors_path(self) -> Path:
        return self.index_dir / f"{self.name}.vectors.npy"

    @property
    def _payload_path(self) -> Path:
        return self.index_dir / f"{self.name}.payload.pkl"

    @property
    def _manifest_path(self) -> Path:
        return self.index_dir / f"{self.name}.manifest.json"

    def exists(self) -> bool:
        return self._payload_path.exists() and (
            self._index_path.exists() or self._vectors_path.exists()
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @staticmethod
    def pack_binary(vectors: np.ndarray) -> np.ndarray:
        """Pack float vectors to bits: positive -> 1, else 0.

        384 float32 dims (1536 bytes) become 48 bytes — a 32x reduction. Sign
        alone preserves enough angular information for a first-pass filter,
        which is why the rescore stage exists.
        """
        return np.packbits((vectors > 0).astype(np.uint8), axis=1)

    def build(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        *,
        save: bool = True,
    ) -> dict[str, Any]:
        """Build the index from chunks and their embeddings."""
        import faiss

        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) != vectors ({len(vectors)})")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != configured {self.dimension}"
            )

        started = time.perf_counter()
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        # Normalize so inner product equals cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1.0, norms)

        self._float_vectors = vectors
        self._chunks = [self._to_stored(chunk) for chunk in chunks]

        if self.binary:
            packed = self.pack_binary(vectors)
            index = faiss.IndexBinaryFlat(self.dimension)
            index.add(packed)
            self._binary_index = index
            logger.info(
                "faiss_binary_index_built",
                vectors=len(vectors),
                bytes_per_vector=packed.shape[1],
                reduction=f"{vectors.nbytes / packed.nbytes:.1f}x",
            )
        else:
            index = faiss.IndexFlatIP(self.dimension)
            index.add(vectors)
            self._binary_index = index

        self._build_language_map()
        self._loaded = True

        stats = {
            "name": self.name,
            "vectors": len(vectors),
            "dimension": self.dimension,
            "binary_quantization": self.binary,
            "build_seconds": round(time.perf_counter() - started, 2),
            "languages": {k: int(len(v)) for k, v in self._by_language.items()},
        }
        if save:
            self.save(stats)
        return stats

    def _build_language_map(self) -> None:
        buckets: dict[str, list[int]] = {}
        for row, chunk in enumerate(self._chunks):
            buckets.setdefault(chunk.language, []).append(row)
        self._by_language = {
            lang: np.asarray(rows, dtype=np.int64) for lang, rows in buckets.items()
        }

    @staticmethod
    def _to_stored(chunk: Chunk) -> StoredChunk:
        meta = chunk.metadata
        return StoredChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            context_text=chunk.context_text,
            doc_hash=meta.doc_hash,
            language=meta.language,
            flores_code=meta.flores_code,
            is_english=meta.is_english,
            query_type=meta.query_type.value,
            is_selected=meta.is_selected,
            strategy=meta.strategy,
            parent_hash=meta.parent_hash,
            char_count=meta.char_count,
            position=meta.position,
            source_query_ids=list(meta.source_query_ids[:20]),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, stats: dict[str, Any] | None = None) -> None:
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)

        if self.binary:
            faiss.write_index_binary(self._binary_index, str(self._index_path))
        else:
            faiss.write_index(self._binary_index, str(self._index_path))

        np.save(self._vectors_path, self._float_vectors)
        with self._payload_path.open("wb") as handle:
            pickle.dump(self._chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)

        manifest = stats or {}
        manifest.update(
            {
                "name": self.name,
                "dimension": self.dimension,
                "binary_quantization": self.binary,
                "count": len(self._chunks),
            }
        )
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("faiss_index_saved", path=str(self._index_path), count=len(self._chunks))

    def load(self) -> bool:
        """Load from disk. Returns False when no index exists."""
        import faiss

        if not self.exists():
            logger.warning("faiss_index_missing", path=str(self._index_path))
            return False

        started = time.perf_counter()
        with self._payload_path.open("rb") as handle:
            self._chunks = pickle.load(handle)
        # mmap so startup does not pay the full read for a large index.
        self._float_vectors = np.load(self._vectors_path, mmap_mode="r")

        if self._index_path.exists():
            self._binary_index = (
                faiss.read_index_binary(str(self._index_path))
                if self.binary
                else faiss.read_index(str(self._index_path))
            )
        else:
            # Vectors present without an index file: rebuild it in memory.
            packed = self.pack_binary(np.asarray(self._float_vectors))
            self._binary_index = faiss.IndexBinaryFlat(self.dimension)
            self._binary_index.add(packed)

        self._build_language_map()
        self._loaded = True
        logger.info(
            "faiss_index_loaded",
            count=len(self._chunks),
            seconds=round(time.perf_counter() - started, 2),
        )
        return True

    def ensure_loaded(self) -> bool:
        return True if self._loaded else self.load()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        languages: Sequence[str] | None = None,
        english_only: bool = False,
        strategy: str | None = None,
        oversampling: float | None = None,
        **_ignored: Any,
    ) -> list[RetrievedChunk]:
        """Two-stage search: Hamming shortlist, then float rescore."""
        if not self.ensure_loaded() or not self._chunks:
            return []

        query = np.ascontiguousarray(query_vector.astype(np.float32).reshape(1, -1))
        norm = np.linalg.norm(query)
        if norm:
            query = query / norm

        allowed = self._allowed_rows(languages, english_only, strategy)
        if allowed is not None and allowed.size == 0:
            return []

        if not self.binary:
            return self._search_float(query, limit, allowed)

        # Stage 1 — Hamming over packed bits. Oversample because binary
        # distance is approximate; a true nearest neighbour can sit outside the
        # top-k of the binary ranking.
        factor = oversampling or self.settings.quantization_oversampling
        shortlist = min(len(self._chunks), max(limit * int(factor), limit * 3, 64))
        if allowed is not None:
            # Filtering shrinks the candidate pool, so widen the shortlist to
            # avoid it being consumed entirely by filtered-out rows.
            shortlist = min(len(self._chunks), shortlist * 4)

        packed_query = self.pack_binary(query)
        _, indices = self._binary_index.search(packed_query, shortlist)
        candidates = indices[0]
        candidates = candidates[candidates >= 0]

        if allowed is not None:
            candidates = candidates[np.isin(candidates, allowed)]
        if candidates.size == 0:
            return []

        # Stage 2 — exact cosine against full-precision vectors.
        vectors = np.asarray(self._float_vectors[candidates])
        scores = vectors @ query[0]
        order = np.argsort(-scores)[:limit]

        return [
            self._to_retrieved(int(candidates[i]), float(scores[i]), rank)
            for rank, i in enumerate(order)
        ]

    def _search_float(
        self, query: np.ndarray, limit: int, allowed: np.ndarray | None
    ) -> list[RetrievedChunk]:
        search_k = min(len(self._chunks), limit * 10 if allowed is not None else limit)
        scores, indices = self._binary_index.search(query, search_k)
        results: list[RetrievedChunk] = []
        for score, row in zip(scores[0], indices[0]):
            if row < 0:
                continue
            if allowed is not None and row not in allowed:
                continue
            results.append(self._to_retrieved(int(row), float(score), len(results)))
            if len(results) >= limit:
                break
        return results

    def _allowed_rows(
        self,
        languages: Sequence[str] | None,
        english_only: bool,
        strategy: str | None,
    ) -> np.ndarray | None:
        """Row indices matching the metadata filters, or None for no filter."""
        selected: np.ndarray | None = None

        if english_only:
            selected = self._by_language.get("en", np.empty(0, dtype=np.int64))
        elif languages:
            parts = [self._by_language.get(code) for code in languages]
            parts = [p for p in parts if p is not None and p.size]
            selected = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

        if strategy:
            matching = np.asarray(
                [i for i, c in enumerate(self._chunks) if c.strategy == strategy],
                dtype=np.int64,
            )
            selected = matching if selected is None else np.intersect1d(selected, matching)

        return selected

    def _to_retrieved(self, row: int, score: float, rank: int) -> RetrievedChunk:
        stored = self._chunks[row]
        try:
            query_type = QueryType(stored.query_type)
        except ValueError:
            query_type = QueryType.UNKNOWN

        chunk = Chunk(
            chunk_id=stored.chunk_id,
            text=stored.text,
            context_text=stored.context_text,
            metadata=ChunkMetadata(
                doc_hash=stored.doc_hash,
                language=stored.language,
                flores_code=stored.flores_code,
                is_english=stored.is_english,
                query_type=query_type,
                is_selected=stored.is_selected,
                source_query_ids=stored.source_query_ids,
                strategy=stored.strategy,
                parent_hash=stored.parent_hash,
                char_count=stored.char_count,
                position=stored.position,
            ),
        )
        return RetrievedChunk(
            chunk=chunk, dense_score=score, fused_score=score, rank=rank
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self._chunks) if self.ensure_loaded() else 0

    def collection_info(self) -> dict[str, Any]:
        if not self.exists():
            return {"exists": False, "name": self.name}
        manifest = {}
        if self._manifest_path.exists():
            try:
                manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        return {
            "exists": True,
            "name": self.name,
            "points_count": manifest.get("count", self.count()),
            "dimension": self.dimension,
            "binary_quantization": self.binary,
            "backend": "faiss",
            "status": "green",
            **{k: v for k, v in manifest.items() if k not in ("name", "count")},
        }

    def scroll(
        self,
        *,
        limit: int = 50,
        offset: Any = None,
        languages: Sequence[str] | None = None,
        strategy: str | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Page through stored chunks — backs the dataset preview."""
        if not self.ensure_loaded():
            return [], None

        rows = self._allowed_rows(languages, False, strategy)
        indices = rows.tolist() if rows is not None else list(range(len(self._chunks)))

        start = int(offset) if offset else 0
        page = indices[start : start + limit]
        payload = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "context_text": c.context_text,
                "doc_hash": c.doc_hash,
                "language": c.language,
                "query_type": c.query_type,
                "is_selected": c.is_selected,
                "strategy": c.strategy,
                "char_count": c.char_count,
            }
            for c in (self._chunks[i] for i in page)
        ]
        next_offset = start + limit if start + limit < len(indices) else None
        return payload, next_offset

    def health(self) -> dict[str, Any]:
        return {
            "reachable": True,  # in-process: nothing to be unreachable
            "backend": "faiss",
            "index_path": str(self._index_path),
            "target_collection_present": self.exists(),
            "count": self.count() if self.exists() else 0,
        }

    def delete_collection(self) -> None:
        for path in (
            self._index_path,
            self._vectors_path,
            self._payload_path,
            self._manifest_path,
        ):
            path.unlink(missing_ok=True)
        self._chunks = []
        self._float_vectors = None
        self._binary_index = None
        self._loaded = False

    # Compatibility shims so this can stand in for the Qdrant store.
    def create_collection(self, *, recreate: bool = False) -> None:
        if recreate:
            self.delete_collection()

    def upsert_chunks(
        self, chunks: Sequence[Chunk], vectors: np.ndarray, **_: Any
    ) -> int:
        self.build(chunks, vectors)
        return len(chunks)


_store: FaissStore | None = None


def get_faiss_store() -> FaissStore:
    global _store
    if _store is None:
        _store = FaissStore()
        _store.load()
    return _store
