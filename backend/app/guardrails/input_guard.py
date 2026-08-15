"""Input-side guardrails: safety screening and off-topic detection.

Both run before retrieval, so both must be fast and deterministic — no LLM
calls. Pattern matching in the low single-digit milliseconds keeps them well
inside the latency budget, and determinism means a blocked query is blocked
reproducibly rather than at the whim of a sampled model.

Deliberately tuned to under-block. A false positive on a legitimate question is
a worse outcome here than letting an odd query through to the confidence gate,
which will refuse it anyway when retrieval finds nothing relevant.
"""

from __future__ import annotations

import re
import unicodedata

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.schemas import GuardrailResult, GuardrailStage, GuardrailVerdict
from app.core.timing import Stopwatch

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Prompt injection
# --------------------------------------------------------------------------

# Attempts to override system instructions or extract the prompt. These target
# the *structure* of an injection rather than any particular wording, so they
# survive paraphrase better than a keyword blocklist.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\b", re.I),
    re.compile(r"\bdisregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|instructions)\b", re.I),
    re.compile(r"\bforget\s+(everything|all|your)\s+(you|instructions|rules|above)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(a|an|no longer)\b", re.I),
    re.compile(r"\bact\s+as\s+(if\s+you|a|an)\b.{0,40}\b(unrestricted|jailbroken|dan)\b", re.I),
    re.compile(r"\b(reveal|show|print|repeat|output)\s+(your|the)\s+"
               r"(system\s+)?(prompt|instructions|rules)\b", re.I),
    re.compile(r"\bsystem\s*[:>]\s*", re.I),
    re.compile(r"<\s*/?\s*(system|instruction|prompt)\s*>", re.I),
    re.compile(r"\b(developer|admin|root)\s+mode\b", re.I),
    re.compile(r"\bbypass\s+(all\s+)?(your\s+)?(safety|filter|guardrail|restriction)", re.I),
)

# Requests for operational harm. Scoped to content this system could plausibly
# be misused to produce — not a general-purpose content filter.
_UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+to\s+(make|build|synthesize|construct)\s+"
               r"(a\s+)?(bomb|explosive|weapon|nerve\s+agent|bioweapon)", re.I),
    re.compile(r"\b(synthesize|manufacture)\s+(meth|methamphetamine|fentanyl|ricin|sarin)\b", re.I),
    re.compile(r"\bhow\s+to\s+(kill|murder|poison)\s+(someone|a\s+person|people|my)\b", re.I),
    re.compile(r"\b(child|minor|underage)\s+(porn|sexual|abuse\s+material)\b", re.I),
    re.compile(r"\bhow\s+to\s+(hack|breach|ddos)\s+.{0,30}\b(without\s+permission|illegally)\b", re.I),
)

# Zero-width and bidi-control characters used to smuggle hidden instructions
# past a visual review. Stripped before matching, then flagged.
_INVISIBLE_CHARS = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


class InputSafetyGuard:
    """Screens for prompt injection, unsafe requests, and malformed input."""

    stage = GuardrailStage.INPUT_SAFETY

    def __init__(self, max_chars: int | None = None) -> None:
        self.max_chars = max_chars or get_settings().max_query_chars

    def check(self, query: str) -> GuardrailResult:
        watch = Stopwatch()
        details: dict[str, object] = {}

        if not query or not query.strip():
            return self._result(
                GuardrailVerdict.BLOCK, "Empty query.", watch, details
            )

        # Normalize before matching so homoglyph and encoding tricks do not
        # slip past patterns that only know ASCII.
        stripped = _INVISIBLE_CHARS.sub("", query)
        if stripped != query:
            details["invisible_characters_removed"] = len(query) - len(stripped)
        normalized = unicodedata.normalize("NFKC", stripped)

        if len(normalized) > self.max_chars:
            details["length"] = len(normalized)
            return self._result(
                GuardrailVerdict.BLOCK,
                f"Query exceeds {self.max_chars} characters.",
                watch,
                details,
            )

        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(normalized):
                details["matched"] = "unsafe_content"
                return self._result(
                    GuardrailVerdict.BLOCK,
                    "Query requests content this system will not help with.",
                    watch,
                    details,
                )

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(normalized):
                details["matched"] = "prompt_injection"
                return self._result(
                    GuardrailVerdict.BLOCK,
                    "Query appears to contain instructions aimed at overriding "
                    "system behaviour.",
                    watch,
                    details,
                )

        # Hidden characters alone are suspicious but not conclusive — warn and
        # let retrieval proceed on the cleaned text.
        if details.get("invisible_characters_removed"):
            return self._result(
                GuardrailVerdict.WARN,
                "Hidden characters were removed from the query.",
                watch,
                details,
            )

        return self._result(GuardrailVerdict.PASS, "No safety concerns.", watch, details)

    def sanitize(self, query: str) -> str:
        """Cleaned query text for downstream stages."""
        return unicodedata.normalize("NFKC", _INVISIBLE_CHARS.sub("", query)).strip()

    def _result(
        self,
        verdict: GuardrailVerdict,
        reason: str,
        watch: Stopwatch,
        details: dict[str, object],
    ) -> GuardrailResult:
        return GuardrailResult(
            stage=self.stage,
            verdict=verdict,
            reason=reason,
            details=details,
            duration_ms=watch.stop(),
        )


# --------------------------------------------------------------------------
# Off-topic detection
# --------------------------------------------------------------------------

# Queries aimed at the assistant itself rather than the corpus. MS MARCO is a
# web-search corpus; none of these have grounded answers in it.
_META_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(hi|hello|hey|namaste|नमस्ते|hola)\b[\s!.?]*$", re.I),
    re.compile(r"\bwhat\s+(model|llm|ai)\s+are\s+you\b", re.I),
    re.compile(r"\bwho\s+(made|built|created|trained)\s+you\b", re.I),
    re.compile(r"\bare\s+you\s+(chatgpt|claude|gemini|gpt|an?\s+ai|human|real)\b", re.I),
    re.compile(r"\b(write|generate|compose)\s+(me\s+)?(a\s+)?"
               r"(poem|song|story|essay|code|script|program)\b", re.I),
    re.compile(r"\b(translate|summarize)\s+this\b", re.I),
)

# Requests for advice the corpus cannot ground and that carry real-world risk
# if answered from a general-purpose model instead.
_ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(should|shall)\s+i\s+(invest|buy|sell|take|quit|marry|divorce)\b", re.I),
    re.compile(r"\bdiagnose\s+(my|me)\b", re.I),
    re.compile(r"\bwhat('?s|\s+is)\s+wrong\s+with\s+(me|my)\b", re.I),
    re.compile(r"\b(my|our)\s+(lawyer|doctor|symptoms)\b.{0,40}\bshould\s+i\b", re.I),
)


class OffTopicGuard:
    """Detects queries the corpus cannot answer.

    Cheap pattern matching only. Genuine topical mismatch — a real question
    about something simply absent from the corpus — is caught downstream by the
    retrieval confidence gate, which is the more reliable signal because it is
    grounded in what the index actually contains.
    """

    stage = GuardrailStage.OFF_TOPIC

    def check(self, query: str) -> GuardrailResult:
        watch = Stopwatch()
        text = query.strip()
        details: dict[str, object] = {}

        for pattern in _META_PATTERNS:
            if pattern.search(text):
                details["matched"] = "meta_or_generative"
                return GuardrailResult(
                    stage=self.stage,
                    verdict=GuardrailVerdict.BLOCK,
                    reason=(
                        "This is a search system over a web-passage corpus, not a "
                        "general-purpose assistant. Ask a factual question instead."
                    ),
                    details=details,
                    duration_ms=watch.stop(),
                )

        for pattern in _ADVICE_PATTERNS:
            if pattern.search(text):
                details["matched"] = "personal_advice"
                return GuardrailResult(
                    stage=self.stage,
                    verdict=GuardrailVerdict.BLOCK,
                    reason=(
                        "This system answers factual questions from retrieved "
                        "passages and cannot give personal, medical, legal or "
                        "financial advice."
                    ),
                    details=details,
                    duration_ms=watch.stop(),
                )

        # A single word is rarely a searchable question, but it can be an entity
        # lookup, so warn rather than block.
        if len(text.split()) < 2:
            details["word_count"] = len(text.split())
            return GuardrailResult(
                stage=self.stage,
                verdict=GuardrailVerdict.WARN,
                reason="Very short query — retrieval may be imprecise.",
                details=details,
                duration_ms=watch.stop(),
            )

        return GuardrailResult(
            stage=self.stage,
            verdict=GuardrailVerdict.PASS,
            reason="Query looks answerable from the corpus.",
            details=details,
            duration_ms=watch.stop(),
        )
