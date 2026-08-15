"""Timing instrumentation.

Uses `perf_counter` (monotonic, highest available resolution) rather than
wall-clock, so measurements are immune to NTP adjustments. Every pipeline stage
is wrapped by one of these so the UI waterfall and the percentile analytics get
their numbers from a single source of truth.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

from app.core.schemas import TimingBreakdown


class Stopwatch:
    """Single-span timer. `elapsed_ms` is valid during and after the span."""

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._end: float | None = None

    def stop(self) -> float:
        self._end = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0


@contextmanager
def track(
    breakdown: TimingBreakdown, stage: str, *, counted: bool = True
) -> Iterator[Stopwatch]:
    """Time a sync block and record it.

    `counted=False` marks stages excluded from the <200ms retrieval budget —
    third-party network calls (STT, generation, TTS) whose latency is not ours
    to control.
    """
    watch = Stopwatch()
    try:
        yield watch
    finally:
        breakdown.add(stage, watch.stop(), counted=counted)


@asynccontextmanager
async def atrack(
    breakdown: TimingBreakdown, stage: str, *, counted: bool = True
) -> AsyncIterator[Stopwatch]:
    """Async counterpart to `track`."""
    watch = Stopwatch()
    try:
        yield watch
    finally:
        breakdown.add(stage, watch.stop(), counted=counted)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile.

    `pct` is 0-100. Interpolating matters at small sample sizes: with 20
    samples, nearest-rank P70 would silently snap to P75.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
