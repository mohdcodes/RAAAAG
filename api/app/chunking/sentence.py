"""S2 — sentence-boundary chunking with one-sentence overlap.

Fixed-size windows cut mid-sentence, which strands the subject of a clause in
one chunk and its predicate in another. This strategy packs whole sentences up
to a token budget instead, so every chunk is a self-contained statement.

The whole strategy depends on `core.text.split_sentences` handling Indic
terminators (danda, double danda, Urdu full stop). With an ASCII-only splitter
this would silently degrade to one-chunk-per-document for 10 of the corpus's
14 languages — identical output to S1 on short passages, and no error raised.
"""

from __future__ import annotations

from app.chunking.base import Chunk, Document, enforce_token_limit
from app.config import settings
from app.core.text import split_sentences
from app.embedding.onnx_embedder import get_embedder


class SentenceWindowChunker:
    name = "sentence"

    def __init__(self, max_tokens: int | None = None, overlap_sentences: int = 1) -> None:
        self.max_tokens = max_tokens or settings.sentence_max_tokens
        self.overlap_sentences = overlap_sentences

    def chunk(self, doc: Document) -> list[Chunk]:
        emb = get_embedder()
        sentences = split_sentences(doc.text)
        if not sentences:
            return []

        counts = [emb.count_tokens(s) for s in sentences]

        groups: list[list[int]] = []
        current: list[int] = []
        running = 0

        for i, n in enumerate(counts):
            # A single sentence over budget becomes its own chunk rather than
            # being dropped — truncation happens later at the model's 512 limit.
            if n >= self.max_tokens:
                if current:
                    groups.append(current)
                    current, running = [], 0
                groups.append([i])
                continue

            if running + n > self.max_tokens and current:
                groups.append(current)
                # Carry the tail sentence(s) forward so a fact spanning the
                # boundary survives in at least one chunk intact.
                current = current[-self.overlap_sentences :] if self.overlap_sentences else []
                running = sum(counts[j] for j in current)

            current.append(i)
            running += n

        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        ordinal = 0
        for idxs in groups:
            text = " ".join(sentences[j] for j in idxs).strip()
            if not text:
                continue
            # Backstop: a single sentence can exceed the budget (Indic passages
            # frequently run 1000+ tokens with no danda). See enforce_token_limit.
            for piece, n_tok in enforce_token_limit(text, self.max_tokens):
                chunks.append(
                    Chunk.build(
                        doc=doc, strategy=self.name, ordinal=ordinal, text=piece,
                        n_tokens=n_tok, n_sentences=len(idxs),
                    )
                )
                ordinal += 1
        return chunks
