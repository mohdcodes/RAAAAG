"""S3 — semantic (embedding-similarity) splitting.

The brief asks for "semantic vs fixed-size splitting", and this is the real
version of it: split where the *meaning* shifts, not where a token counter
happens to land.

Method: embed each sentence, take cosine distance between consecutive
sentences, and place a breakpoint wherever that distance exceeds the Nth
percentile OF THAT DOCUMENT'S OWN distance distribution.

The per-document percentile matters. A global constant threshold (say
"split when distance > 0.3") behaves completely differently on a tightly
focused passage than on a rambling one — one gets zero breakpoints and the
other gets a breakpoint everywhere. A percentile adapts to each document's own
coherence, so "unusually large topic shift *for this text*" means the same
thing everywhere.

Cost note: this embeds every sentence in the corpus, roughly 3x the ingest
embedding work of the other strategies — and costs exactly ZERO at query time.
That is the right place to spend, and the README should say so explicitly.
"""

from __future__ import annotations

import numpy as np

from app.chunking.base import Chunk, Document, enforce_token_limit
from app.config import settings
from app.core.text import split_sentences
from app.embedding.onnx_embedder import get_embedder


class SemanticSimilarityChunker:
    name = "semantic"

    def __init__(
        self,
        percentile: float | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.percentile = percentile or settings.semantic_percentile
        self.min_tokens = min_tokens or settings.semantic_min_tokens
        self.max_tokens = max_tokens or settings.semantic_max_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        emb = get_embedder()
        sentences = split_sentences(doc.text)
        if not sentences:
            return []
        if len(sentences) < 3:
            # Too few sentences for a distance distribution to mean anything.
            text = " ".join(sentences)
            return [
                Chunk.build(
                    doc=doc, strategy=self.name, ordinal=i, text=piece,
                    n_tokens=n_tok, breakpoints=0,
                )
                for i, (piece, n_tok) in enumerate(
                    enforce_token_limit(text, self.max_tokens)
                )
            ]

        vecs = emb.embed_passages(sentences)
        # Vectors are L2-normed, so the dot product is cosine similarity.
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
        distances = 1.0 - sims

        threshold = float(np.percentile(distances, self.percentile))
        counts = [emb.count_tokens(s) for s in sentences]

        groups: list[list[int]] = []
        current: list[int] = [0]
        running = counts[0]

        for i, dist in enumerate(distances):
            nxt = i + 1
            n = counts[nxt]
            # Split on a genuine topic shift, or when the budget forces it.
            # The min_tokens guard stops a run of high distances from producing
            # a stream of one-sentence chunks with no context.
            shift = dist > threshold and running >= self.min_tokens
            overflow = running + n > self.max_tokens

            if shift or overflow:
                groups.append(current)
                current, running = [nxt], n
            else:
                current.append(nxt)
                running += n

        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        ordinal = 0
        for idxs in groups:
            text = " ".join(sentences[j] for j in idxs).strip()
            if not text:
                continue
            for piece, n_tok in enforce_token_limit(text, self.max_tokens):
                chunks.append(
                    Chunk.build(
                        doc=doc, strategy=self.name, ordinal=ordinal, text=piece,
                        n_tokens=n_tok, n_sentences=len(idxs),
                        threshold=round(threshold, 4),
                    )
                )
                ordinal += 1
        return chunks
