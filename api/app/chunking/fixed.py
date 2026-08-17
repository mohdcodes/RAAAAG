"""S1 — fixed-size token window with overlap.

The recall anchor every other strategy is measured against, and the brief's
"overlap handling" requirement.

Windows are sized in TOKENS and sliced using the tokenizer's character offset
map, never by character count. Two reasons:
  - The 512-token model limit is a token limit; a character cap tuned on
    English overflows Malayalam by ~66% (measured).
  - Offset-based slicing lands on real character boundaries, so a chunk edge
    never lands mid-grapheme and severs a combining mark from its base — which
    would corrupt the rendered text for every Indic script.
"""

from __future__ import annotations

from app.chunking.base import Chunk, Document
from app.config import settings
from app.embedding.onnx_embedder import get_embedder


class FixedTokenChunker:
    name = "fixed"

    def __init__(self, size: int | None = None, overlap: int | None = None) -> None:
        self.size = size or settings.fixed_chunk_tokens
        self.overlap = overlap if overlap is not None else settings.fixed_overlap_tokens
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than window size")
        self.stride = self.size - self.overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        emb = get_embedder()
        offsets = emb.encode_offsets(doc.text)
        if not offsets:
            return []

        # Short document: one chunk, no windowing.
        if len(offsets) <= self.size:
            return [
                Chunk.build(
                    doc=doc, strategy=self.name, ordinal=0, text=doc.text,
                    n_tokens=len(offsets), window=[0, len(offsets)],
                )
            ]

        chunks: list[Chunk] = []
        ordinal = 0
        for start in range(0, len(offsets), self.stride):
            window = offsets[start : start + self.size]
            if not window:
                break
            # Trailing window smaller than the overlap adds no new content —
            # every token in it already appeared in the previous chunk.
            if start > 0 and len(window) <= self.overlap:
                break

            char_start = window[0][0]
            char_end = window[-1][1]
            text = doc.text[char_start:char_end].strip()
            if not text:
                continue

            chunks.append(
                Chunk.build(
                    doc=doc, strategy=self.name, ordinal=ordinal, text=text,
                    n_tokens=len(window), window=[start, start + len(window)],
                )
            )
            ordinal += 1

        return chunks
