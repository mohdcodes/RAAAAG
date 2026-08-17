"""Strategy registry.

`build_index.py` and the retriever both iterate this dict, so the set of
strategies is defined in exactly one place and adding a sixth is a one-line
change. The keys must match the keys in `settings.fusion_weights`.
"""

from __future__ import annotations

from app.chunking.base import Chunker
from app.chunking.contextual import ContextualHeaderChunker
from app.chunking.fixed import FixedTokenChunker
from app.chunking.semantic import SemanticSimilarityChunker
from app.chunking.sentence import SentenceWindowChunker

STRATEGIES: dict[str, Chunker] = {
    FixedTokenChunker.name: FixedTokenChunker(),
    SentenceWindowChunker.name: SentenceWindowChunker(),
    SemanticSimilarityChunker.name: SemanticSimilarityChunker(),
    ContextualHeaderChunker.name: ContextualHeaderChunker(),
}

# BM25 is a retrieval strategy but not a chunker — it indexes S2's chunks.
# Listed here so fusion weights and the ablation table stay in sync.
DENSE_STRATEGIES = list(STRATEGIES.keys())
ALL_STRATEGIES = DENSE_STRATEGIES + ["bm25"]
