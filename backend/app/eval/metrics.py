"""Retrieval metrics.

Computed against MS MARCO's `is_selected` labels, which are binary — so nDCG
here carries no graded relevance and is reported for comparability with
published numbers rather than as a richer signal than Recall.

An important caveat that belongs next to any number this module produces: MS
MARCO is sparsely labelled. A passage that genuinely answers the query but was
never marked `is_selected` counts as a miss, so absolute Recall against a pooled
corpus reads pessimistically. The numbers are useful for comparing strategies
against each other; they understate real-world quality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.core.schemas import RetrievalMetrics


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents appearing in the top k.

    With MS MARCO's typical single positive per query this is equivalent to
    hit-rate, but the general form is kept so multi-positive queries score
    correctly.
    """
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Reciprocal of the rank of the first relevant document, 0 if none in k."""
    if not relevant:
        return 0.0
    for index, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Normalized discounted cumulative gain with binary gains."""
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    # Ideal ranking puts every relevant document at the top.
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def aggregate(
    results: Sequence[tuple[Sequence[str], set[str]]]
) -> RetrievalMetrics:
    """Aggregate per-query results into mean metrics.

    Queries with no relevant labels are skipped rather than scored zero —
    counting them would conflate "the system failed" with "the dataset has no
    ground truth here".
    """
    scored = [(retrieved, relevant) for retrieved, relevant in results if relevant]
    if not scored:
        return RetrievalMetrics()

    n = len(scored)
    return RetrievalMetrics(
        recall_at_1=sum(recall_at_k(r, g, 1) for r, g in scored) / n,
        recall_at_5=sum(recall_at_k(r, g, 5) for r, g in scored) / n,
        recall_at_10=sum(recall_at_k(r, g, 10) for r, g in scored) / n,
        mrr_at_10=sum(reciprocal_rank(r, g, 10) for r, g in scored) / n,
        ndcg_at_10=sum(ndcg_at_k(r, g, 10) for r, g in scored) / n,
        queries_evaluated=n,
    )
