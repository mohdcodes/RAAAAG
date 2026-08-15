"""Chunking strategies.

Importing this package registers every strategy. Six are available:

  Controls (prove the sophisticated ones earn their cost)
    fixed_size           blind character windows with overlap
    recursive_character  script-aware separator hierarchy
    sentence             whole sentences grouped to a budget

  Dataset-native
    passage_native       one chunk per MS MARCO passage

  Small-to-big (embed narrow, return wide)
    sentence_window      embed a sentence, return its neighbourhood
    parent_child         embed sentence groups, return the full passage

  Context-aware
    semantic             cut at embedding-similarity breakpoints
    late_chunking        pool token vectors from one full-passage forward pass
"""

from app.ingest.chunking.advanced import (
    ParentChildChunking,
    SemanticChunking,
    SentenceWindowChunking,
)
from app.ingest.chunking.base import (
    Chunk,
    ChunkingStats,
    ChunkingStrategy,
    SourcePassage,
    available_strategies,
    get_strategy,
    register,
    strategy_info,
)
from app.ingest.chunking.baselines import (
    FixedSizeChunking,
    PassageNativeChunking,
    RecursiveCharacterChunking,
    SentenceChunking,
)
from app.ingest.chunking.late import ChunkSpan, LateChunking

__all__ = [
    "Chunk",
    "ChunkSpan",
    "ChunkingStats",
    "ChunkingStrategy",
    "FixedSizeChunking",
    "LateChunking",
    "ParentChildChunking",
    "PassageNativeChunking",
    "RecursiveCharacterChunking",
    "SemanticChunking",
    "SentenceChunking",
    "SentenceWindowChunking",
    "SourcePassage",
    "available_strategies",
    "get_strategy",
    "register",
    "strategy_info",
]
