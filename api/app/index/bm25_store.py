"""S5 — BM25 lexical retrieval on a SciPy CSR matrix.

Not "vast chunking" theatre: it fixes a failure mode dense retrievers really
have. Embeddings are lossy about exact rare tokens — years, model numbers,
product codes, proper nouns seen rarely in training. A query for "748 GW" can
rank a passage about solar capacity generally above the one stating that exact
figure. BM25 does not have that problem.

Implemented directly on `scipy.sparse` rather than with `rank_bm25`, which
stores one Python dict per document — for 500k+ chunks that is a large amount
of RAM and materially slower. Scoring here is a single sparse matvec.

Tokenization is Unicode-aware (`\\w+` with re.UNICODE). An ASCII-only pattern
would produce ZERO tokens for Devanagari and Tamil, silently making BM25
contribute nothing for two of the three corpus languages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_TOKEN = re.compile(r"\w+", re.UNICODE)

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Store:
    def __init__(self) -> None:
        self.matrix: sp.csr_matrix | None = None   # [n_docs, n_terms] weighted tf
        self.idf: np.ndarray | None = None
        self.vocab: dict[str, int] = {}
        self.ids: list[str] = []

    def build(self, texts: list[str], chunk_ids: list[str]) -> None:
        self.ids = list(chunk_ids)
        n_docs = len(texts)

        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        lengths = np.zeros(n_docs, dtype=np.float32)

        for i, text in enumerate(texts):
            toks = tokenize(text)
            lengths[i] = len(toks)
            counts: dict[int, int] = {}
            for t in toks:
                j = vocab.get(t)
                if j is None:
                    j = len(vocab)
                    vocab[t] = j
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i)
                cols.append(j)
                vals.append(float(c))

        self.vocab = vocab
        n_terms = len(vocab)
        tf = sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32
        )

        avgdl = float(lengths.mean()) if n_docs else 0.0
        # Precompute the full BM25 tf saturation term so query time is one matvec.
        #   w = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
        denom_base = K1 * (1 - B + B * (lengths / (avgdl or 1.0)))
        tf = tf.tocoo()
        weighted = tf.data * (K1 + 1.0) / (tf.data + denom_base[tf.row])
        self.matrix = sp.csr_matrix(
            (weighted, (tf.row, tf.col)), shape=(n_docs, n_terms), dtype=np.float32
        )

        df = np.asarray((self.matrix > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.matrix is None or self.idf is None:
            return []
        toks = tokenize(query)
        cols = [self.vocab[t] for t in toks if t in self.vocab]
        if not cols:
            return []

        qvec = np.zeros(self.matrix.shape[1], dtype=np.float32)
        for j in cols:
            qvec[j] += self.idf[j]

        scores = self.matrix @ qvec          # one sparse matvec
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in top if scores[int(i)] > 0]

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        sp.save_npz(dirpath / "bm25_matrix.npz", self.matrix)
        np.save(dirpath / "bm25_idf.npy", self.idf)
        (dirpath / "bm25_vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8"
        )
        (dirpath / "bm25_ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, dirpath: Path) -> "BM25Store":
        self.matrix = sp.load_npz(dirpath / "bm25_matrix.npz").tocsr()
        self.idf = np.load(dirpath / "bm25_idf.npy")
        self.vocab = json.loads((dirpath / "bm25_vocab.json").read_text(encoding="utf-8"))
        self.ids = json.loads((dirpath / "bm25_ids.json").read_text(encoding="utf-8"))
        return self
