"""Rolling latency accumulator — every request, not just benchmark runs.

`scripts/benchmark.py` answers "what is P50 over 300 held-out queries" once, in
a controlled run. This answers "what has this server actually been doing" and
updates on every pass through the pipeline, so the percentiles reflect real
traffic — including the awkward queries a curated benchmark set never contains.

Design notes:

* In-memory and bounded. A deque capped at MAX_SAMPLES keeps this O(1) per
  request and immune to a long-running process growing without limit. It does
  NOT survive a restart, which is the honest tradeoff for adding zero I/O to a
  latency-sensitive path — persisting each request to disk would put a write in
  the very budget we are trying to measure.

* Locked. Endpoints are sync `def`, so FastAPI runs them in a threadpool and
  several can append concurrently even at --workers 1. `deque.append` is
  atomic, but the read path takes a consistent snapshot, so both sides lock.

* Percentiles are nearest-rank, matching scripts/benchmark.py exactly. Two
  different definitions of P50 across two reported numbers would be worse than
  having only one of them.
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict, deque
from typing import Any

from app.core.timing import TimingBreakdown

# ~1k requests is minutes of demo traffic and a few hundred KB resident.
MAX_SAMPLES = 1000

# The retrieval SLO this project is measured against.
BUDGET_MS = 200.0


class _Sample:
    __slots__ = ("total_ms", "retrieval_ms", "generation_ms", "stages", "outcome", "lang", "endpoint")

    def __init__(
        self,
        total_ms: float,
        retrieval_ms: float,
        generation_ms: float | None,
        stages: dict[str, float],
        outcome: str,
        lang: str,
        endpoint: str,
    ) -> None:
        self.total_ms = total_ms
        self.retrieval_ms = retrieval_ms
        self.generation_ms = generation_ms
        self.stages = stages
        self.outcome = outcome
        self.lang = lang
        self.endpoint = endpoint


_samples: deque[_Sample] = deque(maxlen=MAX_SAMPLES)
_lock = threading.Lock()


def record(
    timing: TimingBreakdown,
    *,
    endpoint: str,
    outcome: str,
    lang: str = "??",
) -> None:
    """Called once per completed request. Must never raise into the response
    path — a metrics bug should not turn a good answer into a 500."""
    try:
        stages: dict[str, float] = defaultdict(float)
        for s in timing.stages:
            # Same stage twice in one request (the per-strategy retrieve.*
            # records) is additive, not last-wins.
            stages[s.stage] += s.ms

        sample = _Sample(
            total_ms=timing.total_ms,
            retrieval_ms=timing.retrieval_ms,
            generation_ms=timing.generation_ms,
            stages=dict(stages),
            outcome=outcome,
            lang=lang or "??",
            endpoint=endpoint,
        )
        with _lock:
            _samples.append(sample)
    except Exception:  # noqa: BLE001 - metrics must not break the request
        pass


def reset() -> int:
    with _lock:
        n = len(_samples)
        _samples.clear()
    return n


def _pct(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank. Identical to scripts/benchmark.py so the two agree."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1,
                   int(round(p / 100 * len(sorted_vals) + 0.5)) - 1))
    return sorted_vals[k]


def _dist(name: str, vals: list[float]) -> dict[str, Any]:
    s = sorted(vals)
    return {
        "stage": name,
        "n": len(s),
        "p50": round(_pct(s, 50), 3),
        "p70": round(_pct(s, 70), 3),
        "p90": round(_pct(s, 90), 3),
        "p95": round(_pct(s, 95), 3),
        "p99": round(_pct(s, 99), 3),
        "p100": round(_pct(s, 100), 3),
        "mean": round(statistics.mean(s), 3) if s else 0.0,
        "stdev": round(statistics.pstdev(s), 3) if len(s) > 1 else 0.0,
    }


def summary() -> dict[str, Any]:
    with _lock:
        samples = list(_samples)

    if not samples:
        return {
            "samples": 0,
            "window": MAX_SAMPLES,
            "budget_ms": BUDGET_MS,
            "note": "No requests recorded yet on this server process.",
        }

    retrieval = [s.retrieval_ms for s in samples]
    within = sum(1 for v in retrieval if v < BUDGET_MS)

    # `retrieval_ms` is comparable across both endpoints, so it pools. `total`
    # is not: a /retrieve request has no generation, so its total is just its
    # retrieval, and pooling the two would drag the end-to-end figure toward
    # whichever endpoint happened to be called more. End-to-end means the full
    # pipeline, so it is computed over /query only.
    full = [s for s in samples if s.endpoint == "query"]

    stage_vals: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        for k, v in s.stages.items():
            stage_vals[k].append(v)

    lang_vals: dict[str, list[float]] = defaultdict(list)
    outcomes: dict[str, int] = defaultdict(int)
    endpoints: dict[str, int] = defaultdict(int)
    for s in samples:
        lang_vals[s.lang].append(s.retrieval_ms)
        outcomes[s.outcome] += 1
        endpoints[s.endpoint] += 1

    generation = [s.generation_ms for s in full if s.generation_ms is not None]

    return {
        "samples": len(samples),
        "window": MAX_SAMPLES,
        "budget_ms": BUDGET_MS,
        "retrieval": _dist("retrieval_ms", retrieval),
        "total": _dist("total_ms", [s.total_ms for s in full]) if full else None,
        "generation": _dist("generation_ms", generation) if generation else None,
        "stages": sorted(
            (_dist(k, v) for k, v in stage_vals.items()),
            key=lambda d: -d["mean"],
        ),
        "budget_compliance": {
            "threshold_ms": BUDGET_MS,
            "within": within,
            "total": len(retrieval),
            "percentage": round(100 * within / len(retrieval), 2),
            "measures": "retrieval_ms only — generation is a third-party call "
                        "and is reported separately",
        },
        "by_language": {
            k: {"n": len(v), "p50": round(_pct(sorted(v), 50), 3),
                "p100": round(_pct(sorted(v), 100), 3)}
            for k, v in sorted(lang_vals.items())
        },
        "outcomes": dict(outcomes),
        "endpoints": dict(endpoints),
        "note": "Rolling window over live traffic on this process; resets on "
                "restart. For a controlled measurement over held-out queries, "
                "see scripts/benchmark.py.",
    }
