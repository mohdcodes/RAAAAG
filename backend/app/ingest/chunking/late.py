"""Late chunking.

Conventional chunking splits text, then embeds each piece in isolation — so a
chunk saying "he founded it in 1998" embeds without knowing who "he" is or what
"it" refers to. The pronouns are semantically empty in isolation.

Late chunking inverts the order: run the *whole* passage through the transformer
once, so every token attends to full-passage context, then mean-pool token
embeddings over each chunk's span. Each chunk vector is therefore
context-enriched — the "he" tokens carry information about the antecedent.

Reference: Günther et al., "Late Chunking: Contextual Chunk Embeddings Using
Long-Context Embedding Models" (arXiv:2409.04701).

Cost note: one forward pass per passage, shared with passage-native. That is why
this strategy is affordable to benchmark despite being the most sophisticated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from app.core.schemas import Chunk
from app.ingest.chunking.base import ChunkingStrategy, SourcePassage, register
from app.ingest.text_utils import normalize_text, split_sentences


@dataclass(slots=True)
class ChunkSpan:
    """A chunk's character span, used to map text onto token positions."""

    text: str
    char_start: int
    char_end: int
    position: int


@register
class LateChunking(ChunkingStrategy):
    """Produce chunk spans for context-aware pooling at embedding time.

    This class computes *spans*; the actual pooling happens in the embedding
    layer, which is the only place with access to token-level outputs. The
    embedder detects this strategy and calls `pool_spans` instead of embedding
    each chunk independently.
    """

    name: ClassVar[str] = "late_chunking"
    description: ClassVar[str] = (
        "Embed the full passage once, then mean-pool token vectors per chunk span"
    )
    # Shares passage-native's forward pass rather than needing a separate one.
    requires_own_embeddings: ClassVar[bool] = False

    def __init__(
        self, target_chars: int = 300, min_chars: int = 80, **params: object
    ) -> None:
        super().__init__(**params)
        self.target_chars = target_chars
        self.min_chars = min_chars

    def compute_spans(self, text: str) -> list[ChunkSpan]:
        """Character spans for one passage, grouped on sentence boundaries.

        Spans are located by scanning forward from the previous span's end, so
        a sentence that legitimately repeats within the passage cannot collapse
        onto an earlier offset.
        """
        text = normalize_text(text)
        sentences = split_sentences(text)
        if not sentences:
            return []

        spans: list[ChunkSpan] = []
        current: list[str] = []
        current_start: int | None = None
        cursor = 0

        for sentence in sentences:
            found = text.find(sentence, cursor)
            start = found if found != -1 else cursor
            end = start + len(sentence)
            cursor = end

            if current_start is None:
                current_start = start

            projected = end - current_start
            if current and projected > self.target_chars:
                joined = " ".join(current)
                spans.append(
                    ChunkSpan(joined, current_start, current_start + len(joined), len(spans))
                )
                current, current_start = [], start

            current.append(sentence)

        if current and current_start is not None:
            joined = " ".join(current)
            spans.append(
                ChunkSpan(joined, current_start, current_start + len(joined), len(spans))
            )

        return self._merge_tiny(spans)

    def _merge_tiny(self, spans: list[ChunkSpan]) -> list[ChunkSpan]:
        """Fold spans below min_chars into their predecessor."""
        if len(spans) <= 1:
            return spans
        merged: list[ChunkSpan] = [spans[0]]
        for span in spans[1:]:
            previous = merged[-1]
            if len(span.text) < self.min_chars:
                merged[-1] = ChunkSpan(
                    f"{previous.text} {span.text}",
                    previous.char_start,
                    span.char_end,
                    previous.position,
                )
            else:
                merged.append(ChunkSpan(span.text, span.char_start, span.char_end, len(merged)))
        return merged

    def chunk_passage(self, passage: SourcePassage) -> list[Chunk]:
        parent = normalize_text(passage.text)
        spans = self.compute_spans(parent)
        if not spans:
            return []
        return [
            self.make_chunk(
                passage, span.text, position=span.position,
                context_text=parent, parent_hash=passage.doc_hash,
            )
            for span in spans
        ]

    @staticmethod
    def pool_spans(
        token_embeddings: np.ndarray,
        offset_mapping: list[tuple[int, int]],
        spans: list[ChunkSpan],
        attention_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Mean-pool token embeddings over each chunk's character span.

        Args:
            token_embeddings: (n_tokens, dim) from a single full-passage pass.
            offset_mapping: per-token (char_start, char_end) from the tokenizer.
            spans: chunk spans from `compute_spans`.
            attention_mask: optional (n_tokens,) mask; padding is excluded.

        Returns:
            (n_spans, dim), L2-normalized. Spans that capture no tokens fall
            back to a mean over all real tokens rather than emitting a zero
            vector, which would otherwise poison similarity search.
        """
        if token_embeddings.ndim != 2:
            raise ValueError(f"expected (n_tokens, dim), got {token_embeddings.shape}")

        dim = token_embeddings.shape[1]
        valid = (
            np.ones(len(offset_mapping), dtype=bool)
            if attention_mask is None
            else attention_mask.astype(bool)
        )
        # Special tokens carry (0, 0) offsets and must not pollute pooling.
        real = np.array(
            [
                valid[i] and not (start == 0 and end == 0)
                for i, (start, end) in enumerate(offset_mapping)
            ],
            dtype=bool,
        )
        fallback = (
            token_embeddings[real].mean(axis=0)
            if real.any()
            else np.zeros(dim, dtype=token_embeddings.dtype)
        )

        out = np.zeros((len(spans), dim), dtype=np.float32)
        for i, span in enumerate(spans):
            selected = [
                idx
                for idx, (start, end) in enumerate(offset_mapping)
                if real[idx] and start < span.char_end and end > span.char_start
            ]
            out[i] = token_embeddings[selected].mean(axis=0) if selected else fallback

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)
