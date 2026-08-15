"""Latency analytics store.

Records per-stage timings for every query and computes the P50/P70/P90/P95/P100
percentiles the brief asks for. Deliberately in-process and bounded: a ring
buffer of recent queries needs no database, survives restarts by being
regenerable from a benchmark run, and costs nothing on the query path.

The brief asks for percentiles "measured across a reasonable number of test
queries — not a single best-case run", so this accumulates continuously and
reports the sample count alongside every figure.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

from app.core.schemas import AnswerStatus, LatencyPercentiles, TimingBreakdown
from app.core.timing import percentile


class LatencyStore:
    """Thread-safe bounded store of query timings."""

    def __init__(self, max_samples: int = 5000) -> None:
        self.max_samples = max_samples
        self._records: deque[dict[str, Any]] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def record(
        self,
        timing: TimingBreakdown,
        *,
        status: AnswerStatus,
        language: str = "",
        provider: str | None = None,
        query_chars: int = 0,
    ) -> None:
        entry = {
            "stages": {s.stage: round(s.duration_ms, 3) for s in timing.stages},
            "retrieval_ms": round(timing.retrieval_ms, 3),
            "total_ms": round(timing.total_ms, 3),
            "meets_budget": timing.meets_budget,
            "status": status.value,
            "language": language,
            "provider": provider,
            "query_chars": query_chars,
        }
        with self._lock:
            self._records.append(entry)

    def _snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def percentiles_for(self, key: str, records: list[dict] | None = None) -> LatencyPercentiles:
        """Percentiles for a stage name, or the synthetic keys
        'retrieval_ms' / 'total_ms'."""
        records = records if records is not None else self._snapshot()

        if key in ("retrieval_ms", "total_ms"):
            values = [r[key] for r in records if key in r]
        else:
            values = [r["stages"][key] for r in records if key in r.get("stages", {})]

        if not values:
            return LatencyPercentiles(
                stage=key, p50=0, p70=0, p90=0, p95=0, p100=0, mean=0, samples=0
            )

        return LatencyPercentiles(
            stage=key,
            p50=round(percentile(values, 50), 3),
            p70=round(percentile(values, 70), 3),
            p90=round(percentile(values, 90), 3),
            p95=round(percentile(values, 95), 3),
            p100=round(percentile(values, 100), 3),
            mean=round(sum(values) / len(values), 3),
            samples=len(values),
        )

    def summary(self) -> dict[str, Any]:
        """Full analytics payload for the dashboard."""
        records = self._snapshot()
        if not records:
            return {
                "samples": 0,
                "stages": [],
                "retrieval": self.percentiles_for("retrieval_ms", records).model_dump(),
                "total": self.percentiles_for("total_ms", records).model_dump(),
                "budget_compliance": None,
                "status_counts": {},
                "note": "No queries recorded yet.",
            }

        stage_names: list[str] = []
        for record in records:
            for name in record.get("stages", {}):
                if name not in stage_names:
                    stage_names.append(name)

        within = sum(1 for r in records if r.get("meets_budget"))
        status_counts: dict[str, int] = {}
        for record in records:
            status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

        return {
            "samples": len(records),
            "stages": [self.percentiles_for(name, records).model_dump() for name in stage_names],
            "retrieval": self.percentiles_for("retrieval_ms", records).model_dump(),
            "total": self.percentiles_for("total_ms", records).model_dump(),
            "budget_compliance": {
                "threshold_ms": 200.0,
                "within_budget": within,
                "total": len(records),
                "percentage": round(100.0 * within / len(records), 2),
                # Only retrieval stages are claimed to be under 200ms;
                # generation is a third-party network call reported separately.
                "measures": "retrieval_ms (excludes LLM generation and TTS)",
            },
            "status_counts": status_counts,
            "languages": self._counts(records, "language"),
            "providers": self._counts(records, "provider"),
        }

    @staticmethod
    def _counts(records: list[dict], field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            value = record.get(field)
            if value:
                counts[value] = counts.get(value, 0) + 1
        return counts

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._snapshot()[-limit:][::-1]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def export(self, path: Path) -> Path:
        """Write raw samples to JSON for the submission's latency report."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "records": self._snapshot()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


_store: LatencyStore | None = None


def get_latency_store() -> LatencyStore:
    global _store
    if _store is None:
        _store = LatencyStore()
    return _store
