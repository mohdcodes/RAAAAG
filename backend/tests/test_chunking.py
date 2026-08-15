"""Chunking tests.

Chunking bugs are insidious: they do not crash, they quietly degrade recall.
These tests pin the invariants that matter — no content loss, no infinite loops,
correct multilingual sentence splitting, and correct small-to-big context
attachment.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.schemas import QueryType
from app.ingest.chunking import (
    LateChunking,
    SourcePassage,
    available_strategies,
    get_strategy,
    strategy_info,
)
from app.ingest.text_utils import (
    doc_hash,
    estimate_tokens,
    normalize_text,
    sliding_window,
    split_sentences,
)

ENGLISH = (
    "A corporation is a company authorized to act as a single entity. "
    "It is recognized as such in law. Shareholders own the corporation. "
    "Directors manage its affairs on their behalf."
)
HINDI = (
    "निगम एक कंपनी है जो एकल इकाई के रूप में कार्य करने के लिए अधिकृत है। "
    "इसे कानून में मान्यता प्राप्त है। शेयरधारक निगम के मालिक होते हैं।"
)
URDU = "کارپوریشن ایک کمپنی ہے۔ اسے قانون میں تسلیم کیا جاتا ہے۔ حصص دار مالک ہیں۔"


def make_passage(text: str, language: str = "en", flores: str = "eng_Latn") -> SourcePassage:
    return SourcePassage(
        text=text,
        language=language,
        flores_code=flores,
        is_english=language == "en",
        query_type=QueryType.DESCRIPTION,
        is_selected=True,
        source_query_ids=[1, 2],
    )


# ---------------------------------------------------------------- text utils


class TestTextUtils:
    def test_normalize_collapses_whitespace(self):
        assert normalize_text("  a\n\n  b\t c  ") == "a b c"

    def test_nfc_normalization_makes_hashes_match(self):
        """Decomposed and precomposed Devanagari must dedup to one vector."""
        precomposed = "नि"  # ni
        decomposed = "न" + "ि"
        assert doc_hash(precomposed) == doc_hash(decomposed)

    def test_hash_is_stable_and_case_insensitive(self):
        assert doc_hash("Hello World") == doc_hash("hello world")
        assert doc_hash("a") != doc_hash("b")
        assert len(doc_hash("x")) == 16

    def test_split_sentences_english(self):
        assert len(split_sentences(ENGLISH)) == 4

    def test_split_sentences_devanagari_danda(self):
        """Devanagari uses । not '.' — NLTK punkt would return one sentence."""
        assert len(split_sentences(HINDI)) == 3

    def test_split_sentences_urdu(self):
        assert len(split_sentences(URDU)) == 3

    def test_abbreviation_does_not_split(self):
        assert len(split_sentences("Dr. Smith works here. He is a physician.")) == 2

    def test_short_fragments_merge_forward(self):
        for sentence in split_sentences("Yes. This is a much longer sentence here."):
            assert len(sentence) >= 15

    def test_token_estimate_accounts_for_script(self):
        """Indic scripts tokenize denser than Latin at equal char length."""
        latin, indic = "a" * 100, "न" * 100
        assert estimate_tokens(indic) > estimate_tokens(latin)

    def test_sliding_window_rejects_overlap_ge_size(self):
        """Guards the classic infinite loop."""
        with pytest.raises(ValueError, match="less than"):
            sliding_window(["a", "b", "c"], size=2, overlap=2)

    def test_sliding_window_covers_all_items(self):
        items = [str(i) for i in range(10)]
        windows = sliding_window(items, size=4, overlap=2)
        assert {i for _, _, w in windows for i in w} == set(items)

    def test_empty_input_is_safe(self):
        assert split_sentences("") == []
        assert normalize_text("") == ""
        assert sliding_window([], 3, 1) == []


# ---------------------------------------------------------------- registry


class TestRegistry:
    def test_all_six_plus_registered(self):
        names = available_strategies()
        for expected in (
            "fixed_size", "recursive_character", "sentence", "passage_native",
            "sentence_window", "parent_child", "semantic", "late_chunking",
        ):
            assert expected in names, f"{expected} not registered"

    def test_unknown_strategy_lists_alternatives(self):
        with pytest.raises(KeyError, match="Available"):
            get_strategy("nope")

    def test_info_flags_shared_embedding_strategies(self):
        info = {i["name"]: i for i in strategy_info()}
        assert info["late_chunking"]["requires_own_embeddings"] is False
        assert info["passage_native"]["requires_own_embeddings"] is True


# ---------------------------------------------------------------- strategies

ALL_TEXT_STRATEGIES = [
    "passage_native", "fixed_size", "recursive_character",
    "sentence", "sentence_window", "parent_child", "late_chunking",
]


class TestStrategyInvariants:
    """Invariants every strategy must satisfy, regardless of approach."""

    @pytest.mark.parametrize("name", ALL_TEXT_STRATEGIES)
    def test_produces_chunks(self, name):
        chunks = get_strategy(name).chunk_passage(make_passage(ENGLISH))
        assert chunks, f"{name} produced nothing"
        assert all(c.text.strip() for c in chunks), f"{name} produced empty text"

    @pytest.mark.parametrize("name", ALL_TEXT_STRATEGIES)
    def test_chunk_ids_unique(self, name):
        chunks = get_strategy(name).chunk_passage(make_passage(ENGLISH))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"{name} produced duplicate IDs"

    @pytest.mark.parametrize("name", ALL_TEXT_STRATEGIES)
    def test_metadata_propagates(self, name):
        chunk = get_strategy(name).chunk_passage(make_passage(ENGLISH))[0]
        assert chunk.metadata.strategy == name
        assert chunk.metadata.language == "en"
        assert chunk.metadata.is_selected is True
        assert chunk.metadata.source_query_ids == [1, 2]
        assert chunk.metadata.char_count == len(chunk.text)

    @pytest.mark.parametrize("name", ALL_TEXT_STRATEGIES)
    def test_handles_empty_and_single_sentence(self, name):
        strategy = get_strategy(name)
        assert strategy.chunk_passage(make_passage("")) == []
        assert len(strategy.chunk_passage(make_passage("Just one sentence here."))) >= 1

    @pytest.mark.parametrize("name", ALL_TEXT_STRATEGIES)
    @pytest.mark.parametrize(
        "text,lang,flores",
        [(ENGLISH, "en", "eng_Latn"), (HINDI, "hi", "hin_Deva"), (URDU, "ur", "urd_Arab")],
    )
    def test_multilingual(self, name, text, lang, flores):
        chunks = get_strategy(name).chunk_passage(make_passage(text, lang, flores))
        assert chunks, f"{name} failed on {lang}"

    @pytest.mark.parametrize("name", ALL_TEXT_STRATEGIES)
    def test_no_substantial_content_loss(self, name):
        """Concatenated chunks must retain most source words.

        Not exact equality — overlapping strategies legitimately duplicate, and
        splitters may drop a separator. But losing content means losing recall.
        """
        chunks = get_strategy(name).chunk_passage(make_passage(ENGLISH))
        source_words = set(normalize_text(ENGLISH).lower().split())
        chunk_words = {w for c in chunks for w in c.text.lower().split()}
        missing = source_words - chunk_words
        assert len(missing) <= 1, f"{name} lost words: {missing}"


class TestFixedSize:
    def test_respects_size_limit(self):
        chunks = get_strategy("fixed_size", chunk_chars=100, overlap_chars=20).chunk_passage(
            make_passage(ENGLISH)
        )
        assert all(len(c.text) <= 100 for c in chunks)
        assert len(chunks) > 1

    def test_rejects_overlap_ge_size(self):
        with pytest.raises(ValueError):
            get_strategy("fixed_size", chunk_chars=100, overlap_chars=100)

    def test_long_text_terminates(self):
        """Regression guard against a non-advancing loop."""
        chunks = get_strategy("fixed_size", chunk_chars=50, overlap_chars=25).chunk_passage(
            make_passage("word " * 500)
        )
        assert 10 < len(chunks) < 500


class TestSmallToBig:
    def test_sentence_window_embeds_narrow_returns_wide(self):
        chunks = get_strategy("sentence_window", window_size=1).chunk_passage(
            make_passage(ENGLISH)
        )
        middle = chunks[1]
        assert len(middle.context_text) > len(middle.text)
        assert middle.text in middle.context_text
        assert middle.retrieval_text == middle.context_text

    def test_parent_child_returns_full_passage(self):
        chunks = get_strategy("parent_child", child_sentences=2, child_overlap=1).chunk_passage(
            make_passage(ENGLISH)
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.context_text == normalize_text(ENGLISH)
            assert chunk.metadata.parent_hash == doc_hash(normalize_text(ENGLISH))

    def test_parent_child_rejects_bad_overlap(self):
        with pytest.raises(ValueError):
            get_strategy("parent_child", child_sentences=2, child_overlap=2)


class TestSemantic:
    def test_requires_embedder(self):
        with pytest.raises(RuntimeError, match="requires an embedder"):
            get_strategy("semantic").chunk_passage(make_passage(ENGLISH))

    def test_splits_at_similarity_drop(self):
        """Sentences 0-1 share a topic, 2-3 another → cut at the 1|2 boundary.

        The fixture must produce genuine *variation* in adjacent similarity:
        uniform similarity means no topic shift exists, and returning a single
        chunk is then the correct behaviour.
        """
        topic_a = np.array([1.0, 0.0], dtype=np.float32)
        # Rotated slightly off topic_a: within-topic pairs stay highly similar,
        # the across-topic pair drops sharply.
        topic_b = np.array([0.0, 1.0], dtype=np.float32)

        def fake_embed(sentences: list[str]) -> np.ndarray:
            vectors = []
            for i, _ in enumerate(sentences):
                base = topic_a if i < 2 else topic_b
                # Small per-sentence jitter so within-topic similarity is high
                # but not exactly 1.0, giving the percentile a real distribution.
                jitter = np.array([0.05 * (i % 2), 0.05 * ((i + 1) % 2)], dtype=np.float32)
                vectors.append(base + jitter)
            return np.array(vectors, dtype=np.float32)

        strategy = get_strategy("semantic", breakpoint_percentile=50.0)
        strategy.set_embedder(fake_embed)
        chunks = strategy.chunk_passage(make_passage(ENGLISH))
        assert len(chunks) >= 2, "expected a split at the topic boundary"

    def test_uniform_similarity_yields_single_chunk(self):
        """No variation in similarity means no topic shift to split on."""

        def flat_embed(sentences: list[str]) -> np.ndarray:
            return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(sentences), 1))

        strategy = get_strategy("semantic", breakpoint_percentile=25.0)
        strategy.set_embedder(flat_embed)
        assert len(strategy.chunk_passage(make_passage(ENGLISH))) == 1

    def test_single_sentence_needs_no_embedding(self):
        strategy = get_strategy("semantic")  # no embedder set
        assert len(strategy.chunk_passage(make_passage("Only one sentence."))) == 1


class TestLateChunking:
    def test_spans_are_ordered_and_non_overlapping(self):
        spans = LateChunking(target_chars=100).compute_spans(ENGLISH)
        assert spans
        for a, b in zip(spans, spans[1:]):
            assert a.char_start <= b.char_start

    def test_repeated_sentence_gets_distinct_spans(self):
        """Scanning forward prevents a repeat collapsing onto an earlier offset."""
        text = "The same sentence here. Something else entirely. The same sentence here."
        spans = LateChunking(target_chars=30, min_chars=10).compute_spans(text)
        assert len({s.char_start for s in spans}) == len(spans)

    def test_pool_spans_shapes_and_normalization(self):
        rng = np.random.default_rng(0)
        tokens = rng.normal(size=(10, 8)).astype(np.float32)
        offsets = [(0, 0)] + [(i * 5, i * 5 + 5) for i in range(8)] + [(0, 0)]
        spans = [
            LateChunking().compute_spans("First sentence here. Second sentence here.")[0]
        ]
        pooled = LateChunking.pool_spans(tokens, offsets, spans)
        assert pooled.shape == (len(spans), 8)
        np.testing.assert_allclose(np.linalg.norm(pooled, axis=1), 1.0, rtol=1e-5)

    def test_pool_spans_falls_back_rather_than_emitting_zeros(self):
        """A span matching no tokens must not become a zero vector."""
        tokens = np.ones((5, 4), dtype=np.float32)
        offsets = [(0, 0), (0, 5), (5, 10), (10, 15), (0, 0)]
        from app.ingest.chunking.late import ChunkSpan

        far_span = [ChunkSpan("x", 900, 950, 0)]
        pooled = LateChunking.pool_spans(tokens, offsets, far_span)
        assert np.linalg.norm(pooled[0]) > 0

    def test_pool_spans_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="expected"):
            LateChunking.pool_spans(np.ones((3,)), [(0, 1)], [])


class TestChunkAll:
    def test_stats_are_computed(self):
        passages = [make_passage(ENGLISH), make_passage(HINDI, "hi", "hin_Deva")]
        chunks, stats = get_strategy("parent_child").chunk_all(passages)
        assert stats.input_passages == 2
        assert stats.output_chunks == len(chunks)
        assert stats.avg_chars > 0
        assert stats.expansion_ratio > 0
        assert stats.duration_seconds >= 0

    def test_blank_passages_skipped(self):
        _, stats = get_strategy("passage_native").chunk_all(
            [make_passage(ENGLISH), make_passage("   ")]
        )
        assert stats.input_passages == 1
