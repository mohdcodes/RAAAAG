"""Qdrant vector store with binary quantization.

Binary quantization is the core latency mechanism here. Each float32 dimension
collapses to one bit (positive -> 1, else 0), so a 1024-dim vector goes from
4096 bytes to 128 — a 32x reduction. Comparison becomes XOR + popcount over
packed bits instead of 1024 float multiply-adds, which is roughly an order of
magnitude faster and, more importantly, keeps the whole index in RAM where
random access is not disk-bound.

The accuracy that quantization costs is bought back with a two-stage search:

  1. search the binary index with oversampling (ask for 3x the needed
     candidates, since binary distances are approximate)
  2. rescore those candidates against full-precision vectors held on disk

Qdrant implements both stages natively, so this module configures it rather
than reimplementing it.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import Chunk, ChunkMetadata, QueryType, RetrievedChunk

logger = get_logger(__name__)


class VectorStore:
    """Qdrant wrapper: collection lifecycle, batched upsert, filtered search."""

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        *,
        api_key: str | None = None,
        dimension: int | None = None,
    ) -> None:
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.api_key = api_key or settings.qdrant_api_key
        self.dimension = dimension or settings.embedding_dim
        self.settings = settings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=int(self.settings.qdrant_timeout_s),
            )
        return self._client

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def create_collection(self, *, recreate: bool = False) -> None:
        """Create the collection with binary quantization configured."""
        from qdrant_client import models

        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            logger.info("collection_exists", collection=self.collection)
            return
        if exists:
            logger.warning("recreating_collection", collection=self.collection)
            self.client.delete_collection(self.collection)

        quantization = None
        if self.settings.binary_quantization:
            quantization = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    # Keep the compact binary vectors in RAM; full-precision
                    # vectors stay on disk and are read only during rescore.
                    always_ram=True,
                )
            )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
                # Full-precision vectors on disk: they are only touched during
                # the rescore stage, so paying disk latency there is fine and
                # it keeps RAM for the binary index.
                on_disk=True,
            ),
            quantization_config=quantization,
            hnsw_config=models.HnswConfigDiff(
                m=self.settings.hnsw_m,
                ef_construct=self.settings.hnsw_ef_construct,
            ),
            # Sparse vectors power the BM25 half of hybrid search.
            sparse_vectors_config={
                "text": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
            if self.settings.hybrid_search
            else None,
        )

        # Payload indexes: without these, metadata filters degrade to full scans.
        for field, schema in (
            ("language", models.PayloadSchemaType.KEYWORD),
            ("strategy", models.PayloadSchemaType.KEYWORD),
            ("query_type", models.PayloadSchemaType.KEYWORD),
            ("is_english", models.PayloadSchemaType.BOOL),
            ("doc_hash", models.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(
                collection_name=self.collection, field_name=field, field_schema=schema
            )

        logger.info(
            "collection_created",
            collection=self.collection,
            dimension=self.dimension,
            binary_quantization=self.settings.binary_quantization,
        )

    def delete_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
            logger.info("collection_deleted", collection=self.collection)

    def collection_info(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.collection):
            return {"exists": False, "name": self.collection}
        info = self.client.get_collection(self.collection)
        return {
            "exists": True,
            "name": self.collection,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
            "binary_quantization": self.settings.binary_quantization,
            "dimension": self.dimension,
        }

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        *,
        batch_size: int = 256,
        sparse_vectors: Sequence[dict[int, float]] | None = None,
    ) -> int:
        """Insert chunks with their vectors.

        Point IDs are derived deterministically from chunk_id, so re-running
        ingest updates rather than duplicates.
        """
        from qdrant_client import models

        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) != vectors ({len(vectors)})")

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = []

            for offset, (chunk, vector) in enumerate(zip(batch, batch_vectors)):
                payload = self._chunk_payload(chunk)
                point_vector: Any = vector.tolist()

                if sparse_vectors is not None:
                    sparse = sparse_vectors[start + offset]
                    point_vector = {
                        "": vector.tolist(),
                        "text": models.SparseVector(
                            indices=list(sparse.keys()), values=list(sparse.values())
                        ),
                    }

                points.append(
                    models.PointStruct(
                        id=self._point_id(chunk.chunk_id),
                        vector=point_vector,
                        payload=payload,
                    )
                )

            self.client.upsert(self.collection, points=points, wait=False)
            total += len(points)
            if start % (batch_size * 10) == 0:
                logger.info("upserting", done=total, total=len(chunks))

        logger.info("upsert_complete", count=total, collection=self.collection)
        return total

    @staticmethod
    def _point_id(chunk_id: str) -> int:
        """Stable 63-bit integer ID derived from the chunk ID.

        Qdrant accepts unsigned ints or UUIDs. Hashing keeps re-ingest
        idempotent without maintaining a separate ID mapping.
        """
        import hashlib

        digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") >> 1

    @staticmethod
    def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
        meta = chunk.metadata
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "context_text": chunk.context_text,
            "doc_hash": meta.doc_hash,
            "language": meta.language,
            "flores_code": meta.flores_code,
            "is_english": meta.is_english,
            "query_type": meta.query_type.value,
            "is_selected": meta.is_selected,
            "source_query_ids": meta.source_query_ids[:50],  # cap payload size
            "strategy": meta.strategy,
            "parent_hash": meta.parent_hash,
            "char_count": meta.char_count,
            "token_estimate": meta.token_estimate,
            "position": meta.position,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        languages: Sequence[str] | None = None,
        strategy: str | None = None,
        english_only: bool = False,
        score_threshold: float | None = None,
        oversampling: float | None = None,
        rescore: bool | None = None,
        ef_search: int | None = None,
    ) -> list[RetrievedChunk]:
        """Dense search with binary-quantized two-stage retrieval."""
        from qdrant_client import models

        settings = self.settings
        query_filter = self._build_filter(
            languages=languages, strategy=strategy, english_only=english_only
        )

        quantization_params = None
        if settings.binary_quantization:
            quantization_params = models.QuantizationSearchParams(
                ignore=False,
                # Oversample: binary distances are approximate, so pull extra
                # candidates before rescoring to avoid losing true neighbours.
                oversampling=oversampling
                if oversampling is not None
                else settings.quantization_oversampling,
                # Rescore against full-precision vectors — this is what
                # recovers the accuracy quantization gives up.
                rescore=rescore if rescore is not None else settings.quantization_rescore,
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            search_params=models.SearchParams(
                hnsw_ef=ef_search or settings.hnsw_ef_search,
                quantization=quantization_params,
            ),
            with_payload=True,
        ).points

        return [self._to_retrieved(point, rank) for rank, point in enumerate(results)]

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        sparse_query: dict[int, float],
        *,
        limit: int = 10,
        languages: Sequence[str] | None = None,
        strategy: str | None = None,
        prefetch_limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Dense + sparse search fused with Reciprocal Rank Fusion.

        RRF is used rather than score averaging because dense cosine scores and
        BM25 scores live on incompatible scales; fusing on rank sidesteps the
        normalization problem entirely.
        """
        from qdrant_client import models

        query_filter = self._build_filter(languages=languages, strategy=strategy)
        prefetch_limit = prefetch_limit or max(limit * 4, 100)

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=query_vector.tolist(),
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse_query.keys()),
                        values=list(sparse_query.values()),
                    ),
                    using="text",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

        return [self._to_retrieved(point, rank) for rank, point in enumerate(results)]

    def _build_filter(
        self,
        *,
        languages: Sequence[str] | None = None,
        strategy: str | None = None,
        english_only: bool = False,
    ):
        from qdrant_client import models

        conditions = []
        if english_only:
            conditions.append(
                models.FieldCondition(key="is_english", match=models.MatchValue(value=True))
            )
        elif languages:
            conditions.append(
                models.FieldCondition(
                    key="language", match=models.MatchAny(any=list(languages))
                )
            )
        if strategy:
            conditions.append(
                models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy))
            )
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _to_retrieved(point, rank: int) -> RetrievedChunk:
        payload = point.payload or {}
        try:
            query_type = QueryType(payload.get("query_type", "UNKNOWN"))
        except ValueError:
            query_type = QueryType.UNKNOWN

        chunk = Chunk(
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
                token_estimate=payload.get("token_estimate", 0),
                position=payload.get("position", 0),
            ),
        )
        return RetrievedChunk(
            chunk=chunk, dense_score=point.score, fused_score=point.score, rank=rank
        )

    # ------------------------------------------------------------------
    # Dataset preview support
    # ------------------------------------------------------------------

    def scroll(
        self,
        *,
        limit: int = 50,
        offset: Any = None,
        languages: Sequence[str] | None = None,
        strategy: str | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Page through stored points — backs the UI's dataset preview."""
        points, next_offset = self.client.scroll(
            collection_name=self.collection,
            limit=limit,
            offset=offset,
            scroll_filter=self._build_filter(languages=languages, strategy=strategy),
            with_payload=True,
            with_vectors=False,
        )
        return [p.payload or {} for p in points], next_offset

    def health(self) -> dict[str, Any]:
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            return {
                "reachable": True,
                "url": self.url,
                "collections": collections,
                "target_collection_present": self.collection in collections,
            }
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"reachable": False, "url": self.url, "error": str(exc)}


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
