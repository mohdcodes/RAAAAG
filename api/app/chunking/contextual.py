"""S4 — metadata-aware contextual chunking.

The brief's "metadata-aware chunking" requirement, implemented as Anthropic-
style contextual retrieval at ZERO LLM cost.

The usual contextual-retrieval recipe pays an LLM to write a one-line summary
situating each chunk in its document. This dataset makes that unnecessary: it
already records the query each passage was retrieved for (`Eng_Query`), plus
the question type. Prepending that as a header gives every chunk its situating
context for free — no generation pass, no per-chunk API spend.

Critically, the header is embedded but NEVER shown:

    text         = "[numeric] [hi] Q: what is a corporation | <body>"   <- embedded
    display_text = "<body>"                                             <- to the LLM

Skipping that split is the failure mode here: the model starts quoting the
scaffolding back at the user ("According to the numeric hi context...").
"""

from __future__ import annotations

from app.chunking.base import Chunk, Document, enforce_token_limit
from app.config import settings
from app.core.text import split_sentences
from app.embedding.onnx_embedder import get_embedder


class ContextualHeaderChunker:
    name = "contextual"

    def __init__(self, max_tokens: int | None = None) -> None:
        self.max_tokens = max_tokens or settings.sentence_max_tokens

    def _header(self, doc: Document) -> str:
        bits: list[str] = []
        if doc.query_type:
            bits.append(f"[{doc.query_type}]")
        if doc.lang:
            bits.append(f"[{doc.lang}]")
        if doc.eng_query:
            bits.append(f"Q: {doc.eng_query}")
        return " ".join(bits)

    def chunk(self, doc: Document) -> list[Chunk]:
        emb = get_embedder()
        header = self._header(doc)
        header_tokens = emb.count_tokens(header) if header else 0

        sentences = split_sentences(doc.text)
        if not sentences:
            return []

        counts = [emb.count_tokens(s) for s in sentences]
        # Reserve room for the header so header+body still fits the model limit.
        budget = max(self.max_tokens - header_tokens, 64)

        groups: list[list[int]] = []
        current: list[int] = []
        running = 0
        for i, n in enumerate(counts):
            if running + n > budget and current:
                groups.append(current)
                current, running = [], 0
            current.append(i)
            running += n
        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        ordinal = 0
        for idxs in groups:
            body = " ".join(sentences[j] for j in idxs).strip()
            if not body:
                continue
            # Split the BODY against the reserved budget, then re-attach the
            # header to each piece — so every piece keeps its context and the
            # combined header+body still fits the model limit.
            for piece, n_tok in enforce_token_limit(body, budget):
                embedded = f"{header} | {piece}" if header else piece
                chunks.append(
                    Chunk.build(
                        doc=doc,
                        strategy=self.name,
                        ordinal=ordinal,
                        text=embedded,           # header included -> embedded
                        display_text=piece,      # header stripped -> shown
                        n_tokens=header_tokens + n_tok,
                        has_header=bool(header),
                    )
                )
                ordinal += 1
        return chunks
