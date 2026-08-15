"""Baseline chunking strategies.

These are the control group. The brief explicitly rejects "a single naive
fixed-size approach" — but implementing the naive approaches and *measuring*
them is what turns the sophisticated strategies from an assertion into a
result. If parent-child does not beat fixed-size on this data, that is a finding
worth having.
"""

from __future__ import annotations

from typing import ClassVar

from app.core.schemas import Chunk
from app.ingest.chunking.base import ChunkingStrategy, SourcePassage, register
from app.ingest.text_utils import normalize_text, split_sentences


@register
class PassageNativeChunking(ChunkingStrategy):
    """Index each MS MARCO passage whole, exactly as the dataset ships it.

    This is the most important baseline for this specific dataset. MS MARCO
    passages are already human-curated web snippets of ~50-120 words — they are
    retrieval units by construction. Any strategy that splits them further has
    to justify the extra vectors it creates.
    """

    name: ClassVar[str] = "passage_native"
    description: ClassVar[str] = (
        "One chunk per dataset passage, respecting the corpus's own boundaries"
    )
    requires_own_embeddings: ClassVar[bool] = True

    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        text = normalize_text(passage.text)
        if not text:
            return []
        return [self.make_chunk(passage, text, position=0)]


@register
class FixedSizeChunking(ChunkingStrategy):
    """Fixed character windows with overlap — the classic naive approach.

    Splits blindly on character counts, so it cuts mid-sentence and mid-word.
    Overlap exists purely to reduce the chance that a cut destroys the one
    sentence that answers the query.
    """

    name: ClassVar[str] = "fixed_size"
    description: ClassVar[str] = "Fixed character windows with overlap (naive control)"
    requires_own_embeddings: ClassVar[bool] = True

    def __init__(self, chunk_chars: int = 400, overlap_chars: int = 80, **params: object) -> None:
        super().__init__(**params)
        if overlap_chars >= chunk_chars:
            raise ValueError("overlap_chars must be less than chunk_chars")
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars

    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        text = normalize_text(passage.text)
        if not text:
            return []
        if len(text) <= self.chunk_chars:
            return [self.make_chunk(passage, text, position=0)]

        step = self.chunk_chars - self.overlap_chars
        chunks: list[Chunk] = []
        start = 0
        position = 0
        while start < len(text):
            piece = text[start : start + self.chunk_chars].strip()
            if piece:
                chunks.append(self.make_chunk(passage, piece, position=position))
                position += 1
            if start + self.chunk_chars >= len(text):
                break
            start += step
        return chunks


@register
class RecursiveCharacterChunking(ChunkingStrategy):
    """Recursive splitting on a separator hierarchy.

    Tries to break on the most semantically meaningful separator available,
    falling back to progressively weaker ones. The separator list is
    script-aware: Devanagari danda and Urdu full stop rank alongside the Latin
    period, which off-the-shelf implementations miss entirely.
    """

    name: ClassVar[str] = "recursive_character"
    description: ClassVar[str] = (
        "Recursive split on a script-aware separator hierarchy (control)"
    )
    requires_own_embeddings: ClassVar[bool] = True

    # Strongest to weakest boundary.
    SEPARATORS: ClassVar[tuple[str, ...]] = (
        "\n\n", "\n", "। ", "॥ ", "۔ ", ". ", "? ", "! ", "؟ ", "; ", ", ", " ", "",
    )

    def __init__(self, chunk_chars: int = 400, overlap_chars: int = 60, **params: object) -> None:
        super().__init__(**params)
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars

    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        text = normalize_text(passage.text)
        if not text:
            return []
        pieces = self._split(text, list(self.SEPARATORS))
        merged = self._merge(pieces)
        return [
            self.make_chunk(passage, piece, position=i)
            for i, piece in enumerate(merged)
            if piece.strip()
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_chars:
            return [text]
        if not separators:
            return [text[i : i + self.chunk_chars] for i in range(0, len(text), self.chunk_chars)]

        separator, *rest = separators
        if separator == "":
            return [text[i : i + self.chunk_chars] for i in range(0, len(text), self.chunk_chars)]

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split(text, rest)

        out: list[str] = []
        for part in parts:
            piece = part + separator if separator.strip() else part
            out.extend([piece] if len(piece) <= self.chunk_chars else self._split(piece, rest))
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        """Recombine small fragments up to the target size, honouring overlap."""
        merged: list[str] = []
        buffer = ""
        for piece in pieces:
            if not buffer:
                buffer = piece
            elif len(buffer) + len(piece) <= self.chunk_chars:
                buffer += piece
            else:
                merged.append(buffer.strip())
                tail = buffer[-self.overlap_chars :] if self.overlap_chars else ""
                buffer = (tail + piece) if tail else piece
        if buffer.strip():
            merged.append(buffer.strip())
        return merged


@register
class SentenceChunking(ChunkingStrategy):
    """Group whole sentences up to a size budget — never cuts mid-sentence.

    A meaningful step up from fixed-size at essentially zero extra cost, and a
    fairer baseline for judging the semantic strategies.
    """

    name: ClassVar[str] = "sentence"
    description: ClassVar[str] = "Whole sentences grouped to a size budget"
    requires_own_embeddings: ClassVar[bool] = True

    def __init__(
        self, target_chars: int = 400, overlap_sentences: int = 1, **params: object
    ) -> None:
        super().__init__(**params)
        self.target_chars = target_chars
        self.overlap_sentences = overlap_sentences

    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        sentences = split_sentences(passage.text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self.make_chunk(passage, sentences[0], position=0)]

        groups: list[list[str]] = []
        current: list[str] = []
        size = 0
        for sentence in sentences:
            if current and size + len(sentence) > self.target_chars:
                groups.append(current)
                current = current[-self.overlap_sentences :] if self.overlap_sentences else []
                size = sum(len(s) for s in current)
            current.append(sentence)
            size += len(sentence)
        if current:
            groups.append(current)

        return [
            self.make_chunk(passage, " ".join(group), position=i)
            for i, group in enumerate(groups)
        ]
