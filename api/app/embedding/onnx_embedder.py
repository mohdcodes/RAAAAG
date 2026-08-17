"""multilingual-e5-small embedding via ONNX Runtime (CPU, int8).

Four details here are load-bearing, and each fails *silently* — retrieval keeps
running and returns plausible results while quality degrades. tests/ asserts
each one. Figures below were measured on this machine, not assumed:

1. Asymmetric prefixes: "query: " on queries, "passage: " on documents.
   Measured on a 6-query ranking probe: omitting them was neutral for ranking
   (MRR 1.000 either way), but *swapping* them cost ~20% of the gold-vs-best-
   distractor margin (0.107 -> 0.085). The stronger reason to keep them is
   absolute-scale stability: the G3/G4 guardrail thresholds are calibrated
   against raw cosine values, so the prefix convention must be identical
   between calibration and serving or every threshold silently drifts.
2. Mean pooling, attention-mask weighted. e5's 1_Pooling config sets
   pooling_mode_mean_tokens. Pooling over pad positions skews every vector
   toward whatever padding embeds to. (The bge-small-en-v1.5 checkout on this
   machine uses CLS pooling instead — never share pooling code between them.)
3. L2 normalization. This is what makes faiss.IndexFlatIP return cosine.
   Without it, inner product ranks by magnitude, so longer chunks win
   regardless of relevance.
4. Token-based truncation at 512. Measured tokens-per-char vs English:
   Tamil 1.18x, Hindi 1.27x, Telugu 1.48x, Malayalam 1.66x. So 512 tokens is
   ~2330 chars of English but only ~1400 of Malayalam — a character-based cap
   tuned on English silently truncates the tail of every Malayalam chunk.
"""

from __future__ import annotations

import os

from app.config import settings

# MUST precede the onnxruntime import — see app/config.py docstring.
os.environ.setdefault("OMP_NUM_THREADS", str(settings.omp_threads))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


class E5Embedder:
    """Single shared ONNX session. Construct once in the app lifespan.

    Constructing per request would add 200-500 ms of graph optimization and
    arena allocation to every call — enough on its own to blow the latency
    budget, and it would define P100 in any benchmark that skipped warmup.
    """

    def __init__(
        self,
        model_id: str | None = None,
        onnx_file: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model_id = model_id or settings.embed_model_id
        self.onnx_file = onnx_file or settings.embed_onnx_file
        self.max_tokens = max_tokens or settings.embed_max_tokens
        self.dim = settings.embed_dim

        tok_path = hf_hub_download(self.model_id, "onnx/tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=self.max_tokens)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")  # XLM-R pad id

        model_path = hf_hub_download(self.model_id, self.onnx_file)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.onnx_intra_op_threads
        opts.inter_op_num_threads = settings.onnx_inter_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

        self.warm()

    def warm(self) -> None:
        """Run one throwaway inference so the first real request isn't paying
        for graph optimization and arena allocation."""
        self.embed_queries(["warmup"])

    # --- tokenization -----------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Token length under the *same* tokenizer the model uses.

        All chunkers size their windows with this. Sizing by characters instead
        is the single most common way multilingual chunking breaks.
        """
        prev = self.tokenizer.truncation
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=True).ids)
        finally:
            if prev:
                self.tokenizer.enable_truncation(max_length=self.max_tokens)
            self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def encode_offsets(self, text: str) -> list[tuple[int, int]]:
        """Character offsets per token, so fixed-size chunking can slice on real
        character boundaries and never sever a combining mark."""
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False).offsets)
        finally:
            self.tokenizer.enable_truncation(max_length=self.max_tokens)
            self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    # --- embedding --------------------------------------------------------

    def _forward(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)

        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)

        hidden = self.session.run(None, feeds)[0]  # [B, T, 384]

        # Mask-weighted mean pool. Pad positions contribute nothing.
        m = mask.astype(np.float32)[..., None]
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        vecs = summed / counts

        # L2 normalize -> inner product == cosine.
        norms = np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        return (vecs / norms).astype(np.float32)

    def _embed(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i : i + batch_size]]
            out.append(self._forward(batch))
        return np.vstack(out)

    def embed_passages_bulk(
        self, texts: list[str], batch_size: int = 64, progress: bool = False
    ) -> np.ndarray:
        """Ingest-only path: length-sorted batching.

        The tokenizer pads every batch to its LONGEST member. Feeding texts in
        corpus order mixes 40-token and 500-token chunks in one batch, so the
        short ones are padded ~12x and the model computes attention over
        mostly-padding. Measured at batch 256 in corpus order: ~13 chunks/s,
        which is ~13 hours for this corpus.

        Sorting by length first makes each batch internally uniform, so padding
        waste collapses. Results are restored to the caller's original order.

        Deliberately separate from `embed_passages`: this optimization is only
        valid when embedding many texts at once and would add pointless sorting
        overhead to the single-query serving path.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)

        rng = range(0, len(order), batch_size)
        if progress:
            from tqdm import tqdm
            rng = tqdm(rng, desc="embed", unit="batch")

        for start in rng:
            idx = order[start : start + batch_size]
            vecs = self._forward([self.passage_prefix_str + texts[i] for i in idx])
            for slot, i in enumerate(idx):
                out[i] = vecs[slot]
        return out

    @property
    def passage_prefix_str(self) -> str:
        return settings.passage_prefix

    def embed_queries(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """Embed with the "query: " prefix. Returns float32[N, 384], L2-normed."""
        return self._embed(
            texts, settings.query_prefix, batch_size or settings.embed_batch_size
        )

    def embed_passages(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """Embed with the "passage: " prefix. Returns float32[N, 384], L2-normed."""
        return self._embed(
            texts, settings.passage_prefix, batch_size or settings.embed_batch_size
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Single query -> float32[384]."""
        return self.embed_queries([text])[0]


_embedder: E5Embedder | None = None


def get_embedder() -> E5Embedder:
    """Process-wide singleton. The app lifespan primes this at startup."""
    global _embedder
    if _embedder is None:
        _embedder = E5Embedder()
    return _embedder
