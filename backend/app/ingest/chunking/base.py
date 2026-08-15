"""Chunking strategy interface and registry.

Strategies are pluggable so the benchmark can run all of them over identical
input and produce a like-for-like comparison. Each declares whether it needs its
own embedding pass — the parent-child and late-chunking strategies reuse
passage-native vectors, which is what makes benchmarking all six affordable on
CPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Iterable

from app.core.schemas import Chunk, ChunkMetadata, QueryType
from app.ingest.text_utils import doc_hash, estimate_tokens, normalize_text


@dataclass(slots=True)
class SourcePassage:
    """One passage from the dataset, before chunking.

    Deduplicated across queries: `source_query_ids` and `is_selected` accumulate
    every query this passage appeared under.
    """

    text: str
    language: str
    flores_code: str
    is_english: bool
    doc_hash: str = ""
    query_type: QueryType = QueryType.UNKNOWN
    is_selected: bool = False
    source_query_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)
        if not self.doc_hash:
            self.doc_hash = doc_hash(self.text)


@dataclass(slots=True)
class ChunkingStats:
    """Measured properties of a strategy's output — feeds the comparison table."""

    strategy: str
    input_passages: int = 0
    output_chunks: int = 0
    total_chars: int = 0
    min_chars: int = 0
    max_chars: int = 0
    duration_seconds: float = 0.0

    @property
    def avg_chars(self) -> float:
        return self.total_chars / self.output_chunks if self.output_chunks else 0.0

    @property
    def expansion_ratio(self) -> float:
        """Chunks produced per input passage — the embedding cost multiplier."""
        return self.output_chunks / self.input_passages if self.input_passages else 0.0


class ChunkingStrategy(ABC):
    """Base class for all chunking strategies.

    Subclasses set `name` and implement `chunk_passage`. The base handles
    metadata construction, ID assignment and stats so strategies only express
    their actual splitting logic.
    """

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    # False when this strategy reuses another's vectors (parent-child, late).
    requires_own_embeddings: ClassVar[bool] = True

    def __init__(self, **params: object) -> None:
        self.params = params

    @abstractmethod
    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        """Split one passage into chunks. Must set chunk_id and metadata."""

    def chunk_all(
        self,
        passages: Iterable[SourcePassage],
        *,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[list[Chunk], ChunkingStats]:
        import time

        started = time.perf_counter()
        stats = ChunkingStats(strategy=self.name)
        chunks: list[Chunk] = []
        sizes: list[int] = []

        for index, passage in enumerate(passages):
            if not passage.text:
                continue
            stats.input_passages += 1
            produced = self.chunk_passage(passage)
            for chunk in produced:
                sizes.append(len(chunk.text))
                chunks.append(chunk)
            if progress and index % 1000 == 0:
                progress(index)

        stats.output_chunks = len(chunks)
        stats.total_chars = sum(sizes)
        stats.min_chars = min(sizes) if sizes else 0
        stats.max_chars = max(sizes) if sizes else 0
        stats.duration_seconds = time.perf_counter() - started
        return chunks, stats

    # -- helpers for subclasses -------------------------------------------

    def build_metadata(
        self,
        passage: SourcePassage,
        *,
        text: str,
        position: int = 0,
        parent_hash: str | None = None,
    ) -> ChunkMetadata:
        return ChunkMetadata(
            doc_hash=passage.doc_hash,
            language=passage.language,
            flores_code=passage.flores_code,
            is_english=passage.is_english,
            query_type=passage.query_type,
            is_selected=passage.is_selected,
            source_query_ids=list(passage.source_query_ids),
            strategy=self.name,
            parent_hash=parent_hash,
            char_count=len(text),
            token_estimate=estimate_tokens(text),
            position=position,
        )

    def make_chunk(
        self,
        passage: SourcePassage,
        text: str,
        *,
        position: int = 0,
        context_text: str | None = None,
        parent_hash: str | None = None,
    ) -> Chunk:
        """Build a chunk with a deterministic ID.

        IDs embed the strategy name so multiple strategies can coexist in one
        collection without colliding.
        """
        return Chunk(
            chunk_id=f"{self.name}:{passage.doc_hash}:{position}",
            text=text,
            context_text=context_text,
            metadata=self.build_metadata(
                passage, text=text, position=position, parent_hash=parent_hash
            ),
        )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type[ChunkingStrategy]] = {}


def register(cls: type[ChunkingStrategy]) -> type[ChunkingStrategy]:
    """Class decorator that adds a strategy to the registry."""
    if cls.name in _REGISTRY:
        raise ValueError(f"Duplicate chunking strategy name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, **params: object) -> ChunkingStrategy:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown chunking strategy {name!r}. Available: {available}")
    return _REGISTRY[name](**params)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def strategy_info() -> list[dict[str, object]]:
    """Registry metadata for the UI comparison view."""
    return [
        {
            "name": cls.name,
            "description": cls.description,
            "requires_own_embeddings": cls.requires_own_embeddings,
        }
        for cls in sorted(_REGISTRY.values(), key=lambda c: c.name)
    ]
