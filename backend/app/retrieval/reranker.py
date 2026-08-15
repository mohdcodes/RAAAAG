"""Cross-encoder reranking.

Bi-encoders (the embedder) encode query and passage independently, so they never
see the pair together — fast enough to search millions of vectors, but blind to
term-level interaction. A cross-encoder feeds `[query, passage]` through the
model jointly with full attention across both, which is far more accurate and
far too slow to run over a whole corpus.

Standard resolution, and the one used here: bi-encoder retrieves a wide
candidate pool, cross-encoder reorders the top ~50. That is where most of the
measured accuracy gain in this pipeline comes from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import RetrievedChunk

logger = get_logger(__name__)


class Reranker:
    """bge-reranker-v2-m3 cross-encoder — multilingual, matches e5's coverage."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache_dir: Path | None = None,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.cache_dir = cache_dir or settings.model_cache_dir
        self.batch_size = settings.rerank_batch_size
        self.max_length = max_length
        self._device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if os.cpu_count():
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("loading_reranker", model=self.model_name, device=device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=str(self.cache_dir)
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, cache_dir=str(self.cache_dir)
        )
        model.eval()
        model.to(device)
        self._model = model
        self._device = device
        logger.info("reranker_ready", device=device)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Relevance logits for each (query, passage) pair."""
        self._ensure_loaded()
        if not passages:
            return np.zeros(0, dtype=np.float32)

        import torch

        scores: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = passages[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [query] * len(batch),
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**encoded).logits.view(-1).float()
                scores.append(logits.cpu().numpy())

        return np.concatenate(scores).astype(np.float32)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int | None = None,
        normalize: bool = True,
    ) -> list[RetrievedChunk]:
        """Reorder candidates by cross-encoder score.

        Scores are sigmoid-normalized to 0-1 by default so the confidence
        guardrail can apply a stable threshold — raw logits are unbounded and
        would make any fixed threshold meaningless.

        Reranking uses `retrieval_text`, so small-to-big strategies are judged
        on the wide context the LLM will actually receive, not the narrow
        embedded fragment.
        """
        if not candidates:
            return []

        raw = self.score(query, [c.chunk.retrieval_text for c in candidates])
        values = _sigmoid(raw) if normalize else raw

        for candidate, value in zip(candidates, values):
            candidate.rerank_score = float(value)

        ordered = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        for rank, candidate in enumerate(ordered):
            candidate.rank = rank

        return ordered[:top_k] if top_k else ordered


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid — naive exp overflows on large logits."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
