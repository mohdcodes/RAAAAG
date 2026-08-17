"""Shared chunking contract.

Every strategy emits the same `Chunk` type so `build_index.py` and the
retriever can iterate `STRATEGIES` without special-casing, and so adding a
sixth strategy is a one-line registry change.

The `text` / `display_text` split is what makes metadata-aware chunking work:
S4 embeds a synthesized header for retrieval signal but must never show that
header to the LLM, or the model starts quoting scaffolding back at the user.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class Document(BaseModel):
    """One passage from the corpus, pre-chunking."""

    doc_id: str
    text: str
    lang: str
    variant: str            # "en" | "native"
    query_id: int
    query_type: str
    source_query: str
    eng_query: str
    is_selected: int
    answer: str | None = None
    eng_answer: str | None = None
    n_chars: int = 0


class Chunk(BaseModel):
    chunk_id: str           # f"{strategy}:{doc_id}:{ordinal}"
    doc_id: str
    strategy: str
    text: str               # what gets EMBEDDED
    display_text: str       # what is shown to the user / sent to the LLM
    ordinal: int
    n_tokens: int
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        doc: Document,
        strategy: str,
        ordinal: int,
        text: str,
        n_tokens: int,
        display_text: str | None = None,
        **extra_meta: Any,
    ) -> "Chunk":
        return cls(
            chunk_id=f"{strategy}:{doc.doc_id}:{ordinal}",
            doc_id=doc.doc_id,
            strategy=strategy,
            text=text,
            display_text=display_text if display_text is not None else text,
            ordinal=ordinal,
            n_tokens=n_tokens,
            meta={
                "lang": doc.lang,
                "variant": doc.variant,
                "query_type": doc.query_type,
                "query_id": doc.query_id,
                "is_selected": doc.is_selected,
                **extra_meta,
            },
        )


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def enforce_token_limit(text: str, max_tokens: int) -> list[tuple[str, int]]:
    """Hard-split `text` so no piece exceeds `max_tokens`. Returns (text, n_tokens).

    Every sentence-grouping strategy needs this as a backstop. Grouping by
    sentence assumes sentences are smaller than the budget, and in this corpus
    that assumption fails: many Hindi and Tamil passages contain long runs with
    no danda at all, producing single "sentences" of 1100-1250 tokens.

    Without this the oversized chunk is handed to the tokenizer, which silently
    truncates at 512 and drops the tail — no exception, no warning, just
    permanently unretrievable text. Measured before this guard: 6 violations
    across 3 sample documents, up to 1229 tokens.

    Splits on tokenizer character offsets so a cut never lands mid-grapheme.
    """
    from app.embedding.onnx_embedder import get_embedder

    emb = get_embedder()
    n = emb.count_tokens(text)
    if n <= max_tokens:
        return [(text, n)]

    offsets = emb.encode_offsets(text)
    if not offsets:
        return [(text, n)]

    pieces: list[tuple[str, int]] = []
    for start in range(0, len(offsets), max_tokens):
        window = offsets[start : start + max_tokens]
        if not window:
            break
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            pieces.append((piece, len(window)))
    return pieces or [(text, n)]
