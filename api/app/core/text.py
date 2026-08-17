"""Unicode normalization, script detection, and Indic-aware sentence splitting.

The sentence splitter is the single highest-leverage correctness detail in the
chunking layer. `nltk.punkt` and every `[.!?]` regex return the *entire passage
as one sentence* for Devanagari, Bengali, Gujarati, Punjabi, Odia and Urdu,
because those scripts do not use the ASCII full stop. Sentence-window and
semantic chunking would both silently degrade to "one chunk per document" for
10 of the corpus's 14 languages, with no error raised anywhere.
"""

from __future__ import annotations

import re
import unicodedata

# । danda (U+0964) and ॥ double danda (U+0965) terminate sentences across the
# Devanagari-derived scripts: Hindi, Bengali, Marathi, Gujarati, Punjabi, Odia,
# Assamese, Nepali, Sanskrit.
# ۔ (U+06D4) full stop and ؟ (U+061F) question mark are the Urdu equivalents.
# । is also used in Tamil/Telugu/Kannada/Malayalam text alongside the ASCII stop.
_TERMINATORS = r"[.!?।॥۔؟。！？…]"

# Split *after* a terminator run, provided the next non-space char starts a new
# sentence. Lookbehind keeps the terminator attached to the sentence it ends.
_SENT_SPLIT = re.compile(rf"(?<={_TERMINATORS})[\s]+")

# Abbreviations that must not trigger a split. Latin-script only by nature.
_ABBREV = re.compile(
    r"\b(?:[A-Z]|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Rev|Hon|vs|etc|Inc|Ltd|Co|Corp"
    r"|e\.g|i\.e|a\.m|p\.m|U\.S|U\.K|No)\.$",
    re.IGNORECASE,
)

# Unicode block ranges → language hint. Ranges are disjoint, so first match wins.
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "hi"),   # Devanagari — hi/mr/ne/sa share it
    (0x0980, 0x09FF, "bn"),   # Bengali — bn/as share it
    (0x0A00, 0x0A7F, "pa"),   # Gurmukhi
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    (0x0B00, 0x0B7F, "or"),   # Odia
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0C80, 0x0CFF, "kn"),   # Kannada
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
    (0x0600, 0x06FF, "ur"),   # Arabic block — Urdu in this corpus
]


def normalize(text: str) -> str:
    """NFC-normalize and collapse whitespace.

    NFC matters for Indic scripts: the same visual grapheme can arrive as a
    precomposed codepoint or as base + combining mark. Un-normalized text
    tokenizes into different token sequences for identical-looking strings,
    so a query would fail to match a passage that renders identically.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "")  # ZWSP, BOM
    return re.sub(r"\s+", " ", text).strip()


def detect_script_lang(text: str) -> str:
    """Best-effort language hint from the dominant script.

    Cheap (single pass, no model) and sufficient here: we only need it to pick a
    TTS voice and tag the response. Devanagari cannot distinguish hi/mr/ne/sa,
    so it returns the most likely member; the STT provider's own language_code
    is preferred when available.
    """
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if cp < 0x0590:                       # Latin/ASCII fast path
            if ch.isalpha():
                counts["en"] = counts.get("en", 0) + 1
            continue
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return "en"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def split_sentences(text: str, min_chars: int = 2) -> list[str]:
    """Split into sentences, honouring Indic terminators.

    Falls back to returning the whole text as one sentence when no terminator is
    present — callers (sentence/semantic chunkers) must handle a single-element
    result, which is the correct behaviour for a short passage.
    """
    text = normalize(text)
    if not text:
        return []

    parts = _SENT_SPLIT.split(text)

    # Re-join fragments produced by splitting on an abbreviation's period.
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)

    return [s for s in merged if len(s) >= min_chars]


def clean_for_speech(text: str) -> str:
    """Strip markdown so TTS doesn't read asterisks and bullets aloud."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    return normalize(text)
