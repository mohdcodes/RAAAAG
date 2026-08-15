"""Embedding layer.

Wraps `intfloat/multilingual-e5-large` with the details that actually move
retrieval quality:

  * e5 requires "query: " / "passage: " prefixes. Omitting them is the single
    most common e5 misuse and measurably costs recall — the model was trained
    with them and treats their absence as a distribution shift.
  * Mean pooling over the attention mask, not CLS. e5 is trained for mean
    pooling; using CLS silently degrades quality.
  * L2 normalization, so cosine similarity reduces to a dot product.
  * Token-level output for late chunking, which needs pre-pooling embeddings.

CPU is the target, so batches are modest and threads are pinned.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    """Sentence embedder with query/passage asymmetry and token-level access."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache_dir: Path | None = None,
        max_length: int | None = None,
        device: str | None = None,
        num_threads: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.cache_dir = cache_dir or settings.model_cache_dir
        self.max_length = max_length or settings.embedding_max_length
        self.query_prefix = settings.embedding_query_prefix
        self.passage_prefix = settings.embedding_passage_prefix
        self.batch_size = settings.embedding_batch_size
        self._device = device
        self._num_threads = num_threads
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading — importing torch is slow, so defer until first use.
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._num_threads:
            torch.set_num_threads(self._num_threads)
        elif os.cpu_count():
            # Torch defaults to physical-core count, which left half this
            # machine idle. Use all but one core: measured throughput improves
            # and one core stays free to keep the API responsive during ingest.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "loading_embedder", model=self.model_name, device=device,
            threads=torch.get_num_threads(),
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=str(self.cache_dir)
        )
        model = AutoModel.from_pretrained(self.model_name, cache_dir=str(self.cache_dir))
        model.eval()
        model.to(device)
        self._model = model
        self._device = device

        self.dimension = int(model.config.hidden_size)
        logger.info("embedder_ready", dimension=self.dimension, device=device)

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device or "cpu"

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_queries(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        """Embed queries with the e5 'query: ' prefix."""
        return self._encode([f"{self.query_prefix}{t}" for t in texts], **kwargs)

    def encode_passages(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        """Embed passages with the e5 'passage: ' prefix."""
        return self._encode([f"{self.passage_prefix}{t}" for t in texts], **kwargs)

    def encode_raw(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        """Embed without a prefix.

        Used by semantic chunking, where sentences are compared to each other
        rather than to a query, so neither prefix applies.
        """
        return self._encode(list(texts), **kwargs)

    def _encode(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        import torch

        batch_size = batch_size or self.batch_size
        out: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)

                hidden = self._model(**encoded).last_hidden_state
                pooled = self._mean_pool(hidden, encoded["attention_mask"])
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))

                if show_progress and start % (batch_size * 20) == 0:
                    logger.info("encoding", done=start, total=len(texts))

        return np.vstack(out)

    @staticmethod
    def _mean_pool(hidden_state, attention_mask):
        """Mean pooling over real tokens.

        e5 is trained for mean pooling; CLS pooling degrades it. Padding is
        masked out so batch composition cannot affect a sequence's embedding.
        """
        import torch

        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        summed = torch.sum(hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    # ------------------------------------------------------------------
    # Late chunking support
    # ------------------------------------------------------------------

    def encode_with_tokens(
        self, text: str, *, add_passage_prefix: bool = True
    ) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
        """Encode one text, returning token-level embeddings and offsets.

        Late chunking needs the token embeddings *before* pooling, plus the
        character offsets that map tokens back onto chunk spans.

        The prefix is stripped from the returned offsets so they align with the
        original text's character positions rather than the prefixed string.
        """
        self._ensure_loaded()
        import torch

        prefix = self.passage_prefix if add_passage_prefix else ""
        encoded = self._tokenizer(
            f"{prefix}{text}",
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.inference_mode():
            hidden = self._model(**encoded).last_hidden_state[0]

        token_embeddings = hidden.cpu().numpy().astype(np.float32)
        attention_mask = encoded["attention_mask"][0].cpu().numpy()

        # Shift offsets back into the original text's coordinate space.
        shift = len(prefix)
        adjusted: list[tuple[int, int]] = []
        for start, end in offsets:
            if start == 0 and end == 0:  # special token
                adjusted.append((0, 0))
            else:
                adjusted.append((max(0, start - shift), max(0, end - shift)))

        return token_embeddings, adjusted, attention_mask

    def embed_fn(self):
        """Callable for SemanticChunking's sentence-similarity comparisons."""
        return lambda sentences: self.encode_raw(sentences)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide singleton — the model is ~2 GB and must load once."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
