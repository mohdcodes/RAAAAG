"""Text utilities for multilingual chunking.

Everything here must work across 14 Indic scripts plus Latin. That rules out
most off-the-shelf sentence splitters: NLTK's punkt is trained on European
languages and does not know Devanagari danda (।), Urdu full stop (۔), or the
fact that Indic scripts do not use spaces the way Latin does.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Sentence terminators across the scripts in this dataset.
#   .  ?  !   Latin
#   ।  ॥      Devanagari danda / double danda (Hindi, Marathi, Nepali, Sanskrit)
#   ۔  ؟      Urdu full stop / question mark
#   。 ！ ？   full-width (defensive; occasionally appears in scraped web text)
_TERMINATORS = r"[.!?।॥۔؟。！？]"

# Split after a terminator followed by whitespace. Lookbehind keeps the
# terminator attached to the sentence it ends.
_SENTENCE_SPLIT = re.compile(rf"(?<={_TERMINATORS})\s+")

# Abbreviations that must not trigger a split. Deliberately short: over-eager
# protection is worse than an occasional extra boundary.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
        "inc", "ltd", "co", "corp", "no", "vol", "fig", "eg", "ie",
        "approx", "dept", "est", "min", "max", "avg",
    }
)

_WHITESPACE = re.compile(r"\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def normalize_text(text: str) -> str:
    """Normalize for hashing and dedup.

    NFC matters for Indic scripts: the same visual grapheme can be encoded as
    precomposed or decomposed sequences, and without normalization identical
    passages would hash differently and fail to dedup.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def doc_hash(text: str, *, length: int = 16) -> str:
    """Stable content-addressed ID.

    The dataset has no passage_id, so identity is derived from content. Same
    passage appearing under many queries collapses to one vector.
    """
    normalized = normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


def estimate_tokens(text: str) -> int:
    """Approximate token count without loading a tokenizer.

    Indic scripts run roughly 2.5 chars/token under XLM-R style vocabularies
    versus ~4 for English. Used only for budgeting chunk sizes, so an
    approximation is fine; exact counts come from the real tokenizer at
    embedding time.
    """
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ratio = 2.5 if non_ascii > len(text) * 0.3 else 4.0
    return max(1, int(len(text) / ratio))


def split_sentences(text: str, *, min_chars: int = 15) -> list[str]:
    """Split into sentences across Latin and Indic scripts.

    Short fragments are merged forward rather than emitted alone — a stray
    "Dr." or a two-word fragment makes a poor retrieval unit.
    """
    text = normalize_text(text)
    if not text:
        return []

    raw = _SENTENCE_SPLIT.split(text)
    merged: list[str] = []

    for part in raw:
        part = part.strip()
        if not part:
            continue
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        elif merged and len(part) < min_chars:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)

    # A leading fragment can end up shorter than min_chars; fold it forward.
    if len(merged) > 1 and len(merged[0]) < min_chars:
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)

    return merged


def _ends_with_abbreviation(sentence: str) -> bool:
    if not sentence.endswith("."):
        return False
    last = sentence[:-1].split()[-1].lower() if sentence[:-1].split() else ""
    return last.strip(".") in _ABBREVIATIONS


def split_paragraphs(text: str) -> list[str]:
    parts = _PARAGRAPH_SPLIT.split(text or "")
    return [normalize_text(p) for p in parts if normalize_text(p)]


def sliding_window(
    items: list[str], size: int, overlap: int
) -> list[tuple[int, int, list[str]]]:
    """Windows over a list with overlap, as (start, end, slice) triples.

    Guards against the classic infinite loop when overlap >= size.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")
    if not items:
        return []

    step = size - overlap
    windows: list[tuple[int, int, list[str]]] = []
    start = 0
    while start < len(items):
        end = min(start + size, len(items))
        windows.append((start, end, items[start:end]))
        if end >= len(items):
            break
        start += step
    return windows


def truncate_to_chars(text: str, max_chars: int) -> str:
    """Truncate on a word boundary when one is reasonably close."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars * 0.8:
        cut = cut[:last_space]
    return cut.rstrip() + "…"
