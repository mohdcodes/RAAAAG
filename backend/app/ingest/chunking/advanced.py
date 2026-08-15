"""Advanced chunking strategies.

The shared idea across all three: the text you *embed* and the text you *return*
do not have to be the same. Small units embed precisely; wide units give the LLM
enough context to answer. Decoupling them is the single biggest structural win
available in RAG chunking.
"""

from __future__ import annotations

from typing import Callable, ClassVar

import numpy as np

from app.core.schemas import Chunk
from app.ingest.chunking.base import ChunkingStrategy, SourcePassage, register
from app.ingest.text_utils import normalize_text, split_sentences


@register
class SentenceWindowChunking(ChunkingStrategy):
    """Embed one sentence, return a window of its neighbours.

    Retrieval precision comes from the narrow embedded unit (a single sentence
    is a tight semantic target, so the query vector matches it sharply).
    Answer quality comes from the wide returned context. Classic small-to-big.

    Cost: one vector per sentence, so ~3-5x the vectors of passage-native.
    """

    name: ClassVar[str] = "sentence_window"
    description: ClassVar[str] = (
        "Embed single sentences, return a ±N-sentence window as context"
    )
    requires_own_embeddings: ClassVar[bool] = True

    def __init__(self, window_size: int = 2, **params: object) -> None:
        super().__init__(**params)
        self.window_size = window_size

    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        sentences = split_sentences(passage.text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        for i, sentence in enumerate(sentences):
            lo = max(0, i - self.window_size)
            hi = min(len(sentences), i + self.window_size + 1)
            window = " ".join(sentences[lo:hi])
            chunks.append(
                self.make_chunk(
                    passage,
                    sentence,  # embedded
                    position=i,
                    context_text=window,  # returned to the LLM
                    parent_hash=passage.doc_hash,
                )
            )
        return chunks


@register
class ParentChildChunking(ChunkingStrategy):
    """Embed sentence groups, return the whole parent passage.

    Same small-to-big principle as sentence-window, but the returned context is
    the complete original passage rather than a neighbourhood. For MS MARCO
    specifically this is a strong fit: passages are short enough that returning
    one whole costs little context budget, and they are self-contained by
    construction.

    Reuses passage-native embeddings when child == parent (single-sentence
    passages), which keeps the benchmark affordable.
    """

    name: ClassVar[str] = "parent_child"
    description: ClassVar[str] = (
        "Embed small sentence groups, return the full parent passage as context"
    )
    requires_own_embeddings: ClassVar[bool] = True

    def __init__(
        self, child_sentences: int = 2, child_overlap: int = 1, **params: object
    ) -> None:
        super().__init__(**params)
        if child_overlap >= child_sentences:
            raise ValueError("child_overlap must be less than child_sentences")
        self.child_sentences = child_sentences
        self.child_overlap = child_overlap

    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        parent = normalize_text(passage.text)
        sentences = split_sentences(parent)
        if not sentences:
            return []

        # Short passage: child and parent coincide, so emit a single chunk.
        if len(sentences) <= self.child_sentences:
            return [
                self.make_chunk(
                    passage, parent, position=0, context_text=parent,
                    parent_hash=passage.doc_hash,
                )
            ]

        step = self.child_sentences - self.child_overlap
        chunks: list[Chunk] = []
        position = 0
        start = 0
        while start < len(sentences):
            child = " ".join(sentences[start : start + self.child_sentences])
            if child.strip():
                chunks.append(
                    self.make_chunk(
                        passage, child, position=position, context_text=parent,
                        parent_hash=passage.doc_hash,
                    )
                )
                position += 1
            if start + self.child_sentences >= len(sentences):
                break
            start += step
        return chunks


@register
class SemanticChunking(ChunkingStrategy):
    """Split where meaning shifts, not where the character count runs out.

    Embeds each sentence, walks the sequence, and cuts where cosine similarity
    between consecutive sentences drops below a percentile-derived threshold —
    i.e. where the topic actually changes.

    The threshold is a *percentile of this passage's own similarity
    distribution* rather than an absolute value. An absolute threshold cannot
    work across 14 languages, because embedding similarity distributions differ
    per script and per language.

    Requires an embedding function, so it is materially more expensive at ingest
    than every other strategy here.
    """

    name: ClassVar[str] = "semantic"
    description: ClassVar[str] = (
        "Split at embedding-similarity breakpoints (percentile-adaptive)"
    )
    requires_own_embeddings: ClassVar[bool] = True

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        breakpoint_percentile: float = 25.0,
        min_sentences: int = 1,
        max_chars: int = 800,
        **params: object,
    ) -> None:
        super().__init__(**params)
        self.embed_fn = embed_fn
        self.breakpoint_percentile = breakpoint_percentile
        self.min_sentences = min_sentences
        self.max_chars = max_chars

    def set_embedder(self, embed_fn: Callable[[list[str]], np.ndarray]) -> None:
        """Inject the embedder after construction.

        Lets the registry build this strategy without an embedder present, so
        `available_strategies()` and the UI listing work without loading a model.
        """
        self.embed_fn = embed_fn

    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        # Check sentence count first: a single-sentence passage has no
        # adjacent pairs to compare, so it needs no embedder at all.
        sentences = split_sentences(passage.text)
        if len(sentences) <= 1:
            text = normalize_text(passage.text)
            return [self.make_chunk(passage, text, position=0)] if text else []

        if self.embed_fn is None:
            raise RuntimeError(
                "SemanticChunking requires an embedder — call set_embedder() "
                "or pass embed_fn= before chunking."
            )

        vectors = self.embed_fn(sentences)
        vectors = _l2_normalize(vectors)
        # Cosine similarity between each adjacent pair.
        similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)

        if similarities.size == 0:
            return [self.make_chunk(passage, " ".join(sentences), position=0)]

        threshold = float(np.percentile(similarities, self.breakpoint_percentile))
        breakpoints = {i + 1 for i, sim in enumerate(similarities) if sim < threshold}

        # With few sentences the percentile can land exactly on the minimum, so
        # a strict `<` finds nothing and the passage never splits. When there is
        # a genuine spread in similarity, cut at the weakest link instead.
        if not breakpoints and similarities.size > 1:
            spread = float(similarities.max() - similarities.min())
            if spread > 1e-6:
                breakpoints = {int(np.argmin(similarities)) + 1}

        groups: list[list[str]] = []
        current: list[str] = []
        for i, sentence in enumerate(sentences):
            crossed = i in breakpoints and len(current) >= self.min_sentences
            too_long = current and sum(len(s) for s in current) + len(sentence) > self.max_chars
            if current and (crossed or too_long):
                groups.append(current)
                current = []
            current.append(sentence)
        if current:
            groups.append(current)

        parent = normalize_text(passage.text)
        return [
            self.make_chunk(
                passage, " ".join(group), position=i,
                context_text=parent, parent_hash=passage.doc_hash,
            )
            for i, group in enumerate(groups)
        ]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)
