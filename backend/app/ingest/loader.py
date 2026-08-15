"""MSMARCO-XI loader.

The dataset ships one row per query with ~10 passages nested inline, and no
passage IDs. This module turns that into two things a retrieval system needs:

  1. a deduplicated passage corpus with synthesized, content-addressed IDs
  2. a qrels table mapping query -> relevant doc hashes, for measuring accuracy

Streaming is the default because the full dataset is ~52 GB; a per-language
validation file is ~460 MB and is the practical unit of work.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from app.core.languages import ENGLISH, get_language
from app.core.logging import get_logger
from app.core.schemas import QueryType
from app.ingest.chunking.base import SourcePassage
from app.ingest.text_utils import doc_hash, normalize_text

logger = get_logger(__name__)


@dataclass(slots=True)
class EvalQuery:
    """A query plus its ground-truth relevant passages.

    `relevant_hashes` comes from MS MARCO's `is_selected` flags and is what
    Recall@k / MRR@10 / nDCG@10 are computed against.
    """

    query_id: int
    text: str
    language: str
    flores_code: str
    query_type: QueryType
    relevant_hashes: list[str] = field(default_factory=list)
    candidate_hashes: list[str] = field(default_factory=list)
    english_text: str = ""
    reference_answer: str = ""

    @property
    def has_labels(self) -> bool:
        return bool(self.relevant_hashes)


@dataclass(slots=True)
class IngestResult:
    passages: list[SourcePassage]
    queries: list[EvalQuery]
    stats: dict[str, object] = field(default_factory=dict)


class MSMarcoXILoader:
    """Loads and normalizes MSMARCO-XI parquet files."""

    DATASET_ID = "ai4bharat/MSMARCO-XI"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Row iteration
    # ------------------------------------------------------------------

    def iter_rows(
        self,
        language: str,
        *,
        split: str = "validation",
        limit: int | None = None,
    ) -> Iterator[dict]:
        """Stream rows for one language.

        Language selection is by *filename*, not config — the dataset exposes a
        single `default` config despite what its README claims.
        """
        from datasets import load_dataset

        lang = get_language(language)
        if lang is None:
            raise ValueError(f"Unknown language: {language}")

        if split == "train" and not lang.has_train:
            raise ValueError(
                f"{lang.name} has no train file in this dataset — validation only."
            )

        data_file = lang.train_file if split == "train" else lang.val_file
        logger.info("loading_dataset", language=lang.code, file=data_file, limit=limit)

        dataset = load_dataset(
            self.DATASET_ID,
            data_files={split: data_file},
            split=split,
            streaming=True,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        for index, row in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            yield row

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def build_corpus(
        self,
        language: str,
        *,
        split: str = "validation",
        limit: int | None = None,
        include_english: bool = True,
        include_translated: bool = True,
    ) -> IngestResult:
        """Build a deduplicated corpus and qrels for one language.

        Both English and translated passages are indexed by default — that is
        what makes cross-lingual retrieval work: a Hindi query can match an
        English passage because the embedder shares a vector space across
        languages.
        """
        lang = get_language(language)
        if lang is None:
            raise ValueError(f"Unknown language: {language}")

        # doc_hash -> SourcePassage, accumulating labels across queries.
        corpus: dict[str, SourcePassage] = {}
        queries: list[EvalQuery] = []
        skipped_rows = 0
        duplicate_hits = 0

        for row in self.iter_rows(language, split=split, limit=limit):
            parsed = self._parse_row(row, lang.code, include_english, include_translated)
            if parsed is None:
                skipped_rows += 1
                continue
            query, row_passages = parsed

            for passage in row_passages:
                existing = corpus.get(passage.doc_hash)
                if existing is None:
                    corpus[passage.doc_hash] = passage
                else:
                    # Same passage under another query: merge the evidence.
                    duplicate_hits += 1
                    existing.source_query_ids.extend(passage.source_query_ids)
                    existing.is_selected = existing.is_selected or passage.is_selected

            queries.append(query)

        stats = {
            "language": lang.code,
            "split": split,
            "rows_read": len(queries) + skipped_rows,
            "rows_skipped": skipped_rows,
            "queries": len(queries),
            "unique_passages": len(corpus),
            "duplicate_passage_hits": duplicate_hits,
            "labeled_queries": sum(1 for q in queries if q.has_labels),
        }
        logger.info("corpus_built", **stats)
        return IngestResult(list(corpus.values()), queries, stats)

    def _parse_row(
        self,
        row: dict,
        language: str,
        include_english: bool,
        include_translated: bool,
    ) -> tuple[EvalQuery, list[SourcePassage]] | None:
        """Turn one dataset row into a query plus its passages.

        Returns None for rows too malformed to use — the dataset is
        machine-translated at scale and some rows have empty or misaligned
        passage arrays.
        """
        passages_field = row.get("passages") or {}
        english = passages_field.get("English_passages") or []
        translated = passages_field.get("Translated_passages") or []
        selected = passages_field.get("is_selected") or []

        query_text = normalize_text(row.get("query") or "")
        english_query = normalize_text(row.get("Eng_Query") or "")
        if not query_text and not english_query:
            return None

        lang = get_language(language)
        flores = row.get("target_lang") or (lang.flores if lang else language)

        try:
            query_type = QueryType(row.get("query_type") or "UNKNOWN")
        except ValueError:
            query_type = QueryType.UNKNOWN

        out: list[SourcePassage] = []
        relevant: list[str] = []
        candidates: list[str] = []
        query_id = int(row.get("query_id") or 0)

        # is_selected is positionally aligned with the passage arrays; a
        # missing entry means "not selected" rather than an error.
        def flag(i: int) -> bool:
            return bool(selected[i]) if i < len(selected) else False

        if include_translated:
            for i, text in enumerate(translated):
                passage = self._make_passage(
                    text, language, flores, query_type, flag(i), query_id, is_english=False
                )
                if passage:
                    out.append(passage)
                    candidates.append(passage.doc_hash)
                    if passage.is_selected:
                        relevant.append(passage.doc_hash)

        if include_english:
            for i, text in enumerate(english):
                passage = self._make_passage(
                    text, ENGLISH.code, ENGLISH.flores, query_type, flag(i), query_id,
                    is_english=True,
                )
                if passage:
                    out.append(passage)
                    candidates.append(passage.doc_hash)
                    if passage.is_selected:
                        relevant.append(passage.doc_hash)

        if not out:
            return None

        query = EvalQuery(
            query_id=query_id,
            text=query_text or english_query,
            language=language,
            flores_code=flores,
            query_type=query_type,
            relevant_hashes=relevant,
            candidate_hashes=candidates,
            english_text=english_query,
            reference_answer=normalize_text(row.get("Answer") or row.get("Eng_Answer") or ""),
        )
        return query, out

    @staticmethod
    def _make_passage(
        text: str,
        language: str,
        flores: str,
        query_type: QueryType,
        is_selected: bool,
        query_id: int,
        *,
        is_english: bool,
    ) -> SourcePassage | None:
        cleaned = normalize_text(text or "")
        # Anything this short is a translation artifact, not a passage.
        if len(cleaned) < 20:
            return None
        return SourcePassage(
            text=cleaned,
            language=language,
            flores_code=flores,
            is_english=is_english,
            doc_hash=doc_hash(cleaned),
            query_type=query_type,
            is_selected=is_selected,
            source_query_ids=[query_id],
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save(result: IngestResult, out_dir: Path, language: str) -> dict[str, Path]:
        """Persist corpus, qrels and stats as JSONL for reuse across runs."""
        out_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = out_dir / f"corpus_{language}.jsonl"
        queries_path = out_dir / f"queries_{language}.jsonl"
        stats_path = out_dir / f"stats_{language}.json"

        with corpus_path.open("w", encoding="utf-8") as handle:
            for passage in result.passages:
                handle.write(
                    json.dumps(
                        {
                            "doc_hash": passage.doc_hash,
                            "text": passage.text,
                            "language": passage.language,
                            "flores_code": passage.flores_code,
                            "is_english": passage.is_english,
                            "query_type": passage.query_type.value,
                            "is_selected": passage.is_selected,
                            "source_query_ids": passage.source_query_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        with queries_path.open("w", encoding="utf-8") as handle:
            for query in result.queries:
                handle.write(
                    json.dumps(
                        {
                            "query_id": query.query_id,
                            "text": query.text,
                            "english_text": query.english_text,
                            "language": query.language,
                            "flores_code": query.flores_code,
                            "query_type": query.query_type.value,
                            "relevant_hashes": query.relevant_hashes,
                            "candidate_hashes": query.candidate_hashes,
                            "reference_answer": query.reference_answer,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        stats_path.write_text(
            json.dumps(result.stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "corpus_saved",
            corpus=str(corpus_path),
            passages=len(result.passages),
            queries=len(result.queries),
        )
        return {"corpus": corpus_path, "queries": queries_path, "stats": stats_path}

    @staticmethod
    def load_saved(out_dir: Path, language: str) -> IngestResult:
        """Reload a previously saved corpus, skipping the download."""
        corpus_path = out_dir / f"corpus_{language}.jsonl"
        queries_path = out_dir / f"queries_{language}.jsonl"
        stats_path = out_dir / f"stats_{language}.json"

        if not corpus_path.exists():
            raise FileNotFoundError(f"No saved corpus at {corpus_path}")

        passages: list[SourcePassage] = []
        with corpus_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                passages.append(
                    SourcePassage(
                        text=record["text"],
                        language=record["language"],
                        flores_code=record["flores_code"],
                        is_english=record["is_english"],
                        doc_hash=record["doc_hash"],
                        query_type=QueryType(record["query_type"]),
                        is_selected=record["is_selected"],
                        source_query_ids=record["source_query_ids"],
                    )
                )

        queries: list[EvalQuery] = []
        if queries_path.exists():
            with queries_path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    queries.append(
                        EvalQuery(
                            query_id=record["query_id"],
                            text=record["text"],
                            language=record["language"],
                            flores_code=record["flores_code"],
                            query_type=QueryType(record["query_type"]),
                            relevant_hashes=record["relevant_hashes"],
                            candidate_hashes=record["candidate_hashes"],
                            english_text=record.get("english_text", ""),
                            reference_answer=record.get("reference_answer", ""),
                        )
                    )

        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        return IngestResult(passages, queries, stats)


def build_qrels(queries: list[EvalQuery]) -> dict[int, set[str]]:
    """query_id -> set of relevant doc hashes, for eval."""
    qrels: dict[int, set[str]] = defaultdict(set)
    for query in queries:
        if query.relevant_hashes:
            qrels[query.query_id].update(query.relevant_hashes)
    return dict(qrels)
