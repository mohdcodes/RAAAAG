"""Six guardrails.

The design constraint: G1-G4 sit INSIDE the <200 ms retrieval path, so they
must be nearly free. Measured combined cost is ~0.35 ms, which is possible
only because none of them adds a model or a network call:

  G1 sanity + injection   regex over a short string           ~0.1 ms
  G2 unsafe content       lexicon set-membership              ~0.2 ms
  G3 off-topic            64x384 matvec, REUSES the query     ~0.03 ms
                          vector already computed for search
  G4 retrieval confidence  arithmetic on scores we already have ~0 ms

G5/G6 run after generation and are reported separately, since the brief asks
for generation latency to be reported apart from retrieval.

The expensive-looking one, G3, is cheap precisely because it piggybacks on work
already done. A "does this belong to the corpus" classifier would have cost a
model load and 10-50 ms; comparing the existing query vector against 64 k-means
centroids costs 25 microseconds and answers the same question.
"""

from __future__ import annotations

import re

import numpy as np

from app.config import settings
from app.core.text import split_sentences
from app.schemas.query import GuardrailCheck, RefusalReason

# --- G1: prompt injection -------------------------------------------------
# Deliberately narrow. Broad patterns ("system", "prompt") would refuse
# legitimate MS MARCO questions like "what is a system prompt in linguistics".
_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+)*"
    r"(instruction|prompt|rule|direction|context)"
    r"|you\s+are\s+now\s+(a|an|no\s+longer)"
    r"|(reveal|show|print|repeat|output)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instruction)"
    r"|<\s*/?\s*(system|assistant|user)\s*>"
    r"|\bDAN\b|jailbreak"
    r"|act\s+as\s+(if\s+you\s+are\s+)?an?\s+(unrestricted|unfiltered|uncensored)",
    re.IGNORECASE,
)

# --- G2: unsafe content ---------------------------------------------------
# A compact multilingual lexicon of intent phrases, not slurs. Matching intent
# ("how to make a bomb") rather than keywords ("bomb") avoids refusing
# legitimate corpus questions like "what caused the Bali bombing".
_UNSAFE_PATTERNS = re.compile(
    r"how\s+(to|do\s+i|can\s+i)\s+.{0,30}\b(make|build|synthesize|acquire)\b.{0,30}"
    r"\b(bomb|explosive|meth|methamphetamine|nerve\s+agent|ricin|sarin|napalm)\b"
    r"|how\s+(to|do\s+i|can\s+i)\s+.{0,40}\b(kill|murder|poison|harm)\b\s+(someone|a\s+person|my|him|her|them)"
    r"|\b(child\s+porn|csam)\b"
    r"|how\s+to\s+.{0,30}\b(hack|ddos|breach)\b.{0,20}\b(bank|hospital|government|someone)",
    re.IGNORECASE,
)


def g1_input_sanity(query: str) -> tuple[GuardrailCheck, RefusalReason | None]:
    q = query.strip()
    if len(q) < 2:
        return (
            GuardrailCheck(id="G1", name="input_sanity", passed=False,
                           detail="query too short"),
            RefusalReason.UNSAFE_INPUT,
        )
    if _INJECTION.search(q):
        return (
            GuardrailCheck(id="G1", name="input_sanity", passed=False,
                           detail="prompt-injection pattern matched"),
            RefusalReason.PROMPT_INJECTION,
        )
    return GuardrailCheck(id="G1", name="input_sanity", passed=True), None


def g2_safety(query: str) -> tuple[GuardrailCheck, RefusalReason | None]:
    if _UNSAFE_PATTERNS.search(query):
        return (
            GuardrailCheck(id="G2", name="content_safety", passed=False,
                           detail="unsafe-intent pattern matched"),
            RefusalReason.UNSAFE_INPUT,
        )
    return GuardrailCheck(id="G2", name="content_safety", passed=True), None


def g3_off_topic(
    qvec: np.ndarray, centroids: np.ndarray | None, tau: float | None = None
) -> tuple[GuardrailCheck, RefusalReason | None]:
    """Max cosine between the query and any corpus k-means centroid.

    Reuses the query vector already computed for retrieval, so the marginal
    cost is one 64x384 matvec.
    """
    tau = tau if tau is not None else settings.tau_topic
    if centroids is None:
        return GuardrailCheck(id="G3", name="off_topic", passed=True,
                              detail="no centroids available"), None

    max_sim = float((centroids @ qvec).max())
    passed = max_sim >= tau
    return (
        GuardrailCheck(id="G3", name="off_topic", passed=passed,
                       score=round(max_sim, 4), threshold=tau,
                       detail=None if passed else "query is far from every corpus cluster"),
        None if passed else RefusalReason.OFF_TOPIC,
    )


def g4_retrieval_confidence(
    top_score: float, flatness: float, tau_conf: float | None = None
) -> tuple[GuardrailCheck, RefusalReason | None]:
    """Two distinct failures, both meaning "don't answer":

      - top_score < tau_conf: nothing scored well enough.
      - flatness < tau: scores are acceptable but the ranking has no opinion —
        the top result is barely better than the fifth. Answering from a flat
        ranking is how a RAG system confidently cites an irrelevant passage.
    """
    tau_conf = tau_conf if tau_conf is not None else settings.tau_conf
    low = top_score < tau_conf
    flat = flatness < settings.tau_flatness

    if low:
        detail = f"top score {top_score:.3f} < {tau_conf}"
    elif flat:
        detail = f"ranking is flat (rrf[0]/rrf[4] = {flatness:.3f})"
    else:
        detail = None

    passed = not (low or flat)
    return (
        GuardrailCheck(id="G4", name="retrieval_confidence", passed=passed,
                       score=round(top_score, 4), threshold=tau_conf, detail=detail),
        None if passed else RefusalReason.LOW_RETRIEVAL_CONFIDENCE,
    )


def g6_validate_citations(
    cited: list[str], available: set[str]
) -> tuple[GuardrailCheck, RefusalReason | None, list[str]]:
    """Set membership — free. Returns the surviving citations.

    A model that invents a chunk_id has, by definition, not read the context.
    If EVERY citation is invented we refuse; if some are, we strip them and
    keep the answer, since partial fabrication of ids is usually formatting
    drift rather than hallucinated content.
    """
    valid = [c for c in cited if c in available]
    if cited and not valid:
        return (
            GuardrailCheck(id="G6", name="citation_validity", passed=False,
                           score=0.0, detail=f"all {len(cited)} citations invented"),
            RefusalReason.INVALID_CITATIONS,
            [],
        )
    ratio = len(valid) / len(cited) if cited else 1.0
    return (
        GuardrailCheck(id="G6", name="citation_validity", passed=True,
                       score=round(ratio, 3),
                       detail=None if ratio == 1.0
                       else f"{len(cited)-len(valid)} invalid citations stripped"),
        None,
        valid,
    )


def g5_groundedness(
    answer: str,
    context_vecs: np.ndarray,
    context_ids: list[str],
    embedder,
    tau: float | None = None,
) -> tuple[GuardrailCheck, RefusalReason | None, float, list[dict]]:
    """Per-sentence support against the retrieved context.

    The contexts are NOT re-embedded: `context_vecs` were already computed
    during retrieval and are passed in. The only new work is one small batch
    embed of the answer's own sentences, which is why this costs ~15 ms rather
    than a second full retrieval pass.
    """
    tau = tau if tau is not None else settings.tau_ground
    sentences = [s for s in split_sentences(answer) if len(s) > 15]
    if not sentences or len(context_vecs) == 0:
        return (
            GuardrailCheck(id="G5", name="groundedness", passed=True,
                           detail="no verifiable sentences"),
            None, 1.0, [],
        )

    svecs = embedder.embed_passages(sentences)
    sims = svecs @ context_vecs.T                # [n_sent, n_ctx]
    best = sims.max(axis=1)
    best_idx = sims.argmax(axis=1)

    support = [
        {
            "sentence": s,
            "supported": bool(b >= tau),
            "best_score": round(float(b), 4),
            "supporting_chunk_id": context_ids[int(i)] if b >= tau else None,
        }
        for s, b, i in zip(sentences, best, best_idx)
    ]
    score = float(sum(1 for x in support if x["supported"]) / len(support))

    if score < settings.ground_refuse_below:
        return (
            GuardrailCheck(id="G5", name="groundedness", passed=False,
                           score=round(score, 3), threshold=settings.ground_refuse_below,
                           detail=f"only {score:.0%} of sentences supported by context"),
            RefusalReason.UNGROUNDED_OUTPUT, score, support,
        )
    return (
        GuardrailCheck(id="G5", name="groundedness", passed=True,
                       score=round(score, 3), threshold=settings.ground_refuse_below,
                       detail=None if score >= settings.ground_warn_below
                       else f"partially grounded ({score:.0%})"),
        None, score, support,
    )
