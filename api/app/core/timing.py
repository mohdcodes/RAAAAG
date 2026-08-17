"""Latency instrumentation.

Every duration in this project comes from `time.perf_counter_ns()`.

Never `time.time()`: on Windows its granularity is ~15.6 ms (the default timer
interrupt period), so an 8 ms embed step measures as either 0 ms or 15.6 ms.
A benchmark built on it produces numbers that look entirely plausible and are
fiction. `perf_counter` maps to QueryPerformanceCounter (~100 ns resolution).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from pydantic import BaseModel, Field


class StageTiming(BaseModel):
    stage: str
    ms: float
    start_ms: float          # offset from collector creation
    end_ms: float
    ok: bool = True
    detail: dict[str, Any] | None = None


class TimingCollector:
    """Records per-stage wall time as offsets from a single origin.

    Offsets (not bare durations) are recorded so the frontend can render a true
    waterfall — including overlapping stages, e.g. the parallel per-strategy
    index searches, which genuinely run concurrently and would be misrepresented
    by a strictly-sequential stacked bar.
    """

    def __init__(self) -> None:
        self._origin_ns = time.perf_counter_ns()
        self._stages: list[StageTiming] = []

    def _now_ms(self) -> float:
        return (time.perf_counter_ns() - self._origin_ns) / 1e6

    @contextmanager
    def stage(self, name: str, **detail: Any) -> Iterator[dict[str, Any]]:
        """Time a block. Mutate the yielded dict to attach detail after the fact.

        >>> with tc.stage("retrieve") as d:
        ...     hits = search()
        ...     d["n_hits"] = len(hits)
        """
        start = self._now_ms()
        payload: dict[str, Any] = dict(detail)
        ok = True
        try:
            yield payload
        except Exception:
            ok = False
            raise
        finally:
            end = self._now_ms()
            self._stages.append(
                StageTiming(
                    stage=name,
                    ms=end - start,
                    start_ms=start,
                    end_ms=end,
                    ok=ok,
                    detail=payload or None,
                )
            )

    def record(self, name: str, ms: float, *, ok: bool = True, **detail: Any) -> None:
        """Record a stage whose duration was measured elsewhere (e.g. a provider
        reported its own time, or a thread pool returned a per-task duration)."""
        end = self._now_ms()
        self._stages.append(
            StageTiming(
                stage=name, ms=ms, start_ms=max(0.0, end - ms), end_ms=end,
                ok=ok, detail=dict(detail) or None,
            )
        )

    @property
    def stages(self) -> list[StageTiming]:
        return list(self._stages)

    def total_ms(self) -> float:
        return self._now_ms()

    def sum_of(self, *names: str) -> float:
        """Sum specific stages by name. Used to compute `retrieval_ms` — the
        number the <200 ms SLO is measured against — without including
        generation, which the brief requires reported separately."""
        wanted = set(names)
        return sum(s.ms for s in self._stages if s.stage in wanted)

    def span_of(self, *names: str) -> float:
        """Wall-clock span covering the named stages (max end - min start).

        Differs from `sum_of` whenever stages overlap: the four FAISS searches
        run in parallel, so their sum is ~4x their true wall cost. The SLO is a
        wall-clock claim, so retrieval latency uses this, not `sum_of`.
        """
        sel = [s for s in self._stages if s.stage in set(names)]
        if not sel:
            return 0.0
        return max(s.end_ms for s in sel) - min(s.start_ms for s in sel)


class TimingBreakdown(BaseModel):
    """Wire format. `retrieval_ms` is the <200 ms SLO figure; `generation_ms`
    is deliberately a separate field so the two are never conflated."""

    total_ms: float
    retrieval_ms: float
    generation_ms: float | None = None
    stt_ms: float | None = None
    tts_ms: float | None = None
    stages: list[StageTiming] = Field(default_factory=list)


class Stopwatch:
    """Single-shot timer for code outside a collector."""

    def __init__(self) -> None:
        self._start = time.perf_counter_ns()

    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self._start) / 1e6

    def reset(self) -> None:
        self._start = time.perf_counter_ns()
