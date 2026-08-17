"""Latency benchmark → P50 / P70 / P100.

Three methodology decisions that determine whether the numbers mean anything:

1. **Queries come from data/raw/heldout.jsonl** — real dataset queries that
   were deliberately NEVER INDEXED (the last 15% of rows, split off at download
   time). Benchmarking on indexed queries would measure a best case that no
   real user ever hits.

2. **Warmup is mandatory and discarded.** The first ONNX inference pays for
   graph optimization and arena allocation (200-500 ms). Including it would let
   one request define P100 and make the whole table a lie.

3. **`time.perf_counter_ns()`, never `time.time()`.** On Windows `time.time()`
   has ~15.6 ms granularity, so an 8 ms stage measures as 0 or 15.6 ms —
   producing plausible, entirely fictitious numbers.

Retrieval and full-pipeline modes are reported SEPARATELY, because the brief's
<200 ms target is a retrieval target and mixing in LLM time would make both
figures meaningless.

Usage:
    python scripts/benchmark.py --n 300 --mode retrieval
    python scripts/benchmark.py --n 100 --mode full
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.text import detect_script_lang, normalize  # noqa: E402
from app.embedding.onnx_embedder import get_embedder  # noqa: E402
from app.guardrails.pipeline import (  # noqa: E402
    g1_input_sanity, g2_safety, g3_off_topic, g4_retrieval_confidence,
)
from app.index.fusion import rrf_flatness  # noqa: E402
from app.index.multi_index import get_retriever  # noqa: E402


def percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile. P100 is the max by definition."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(p / 100 * len(sorted_vals) + 0.5)) - 1))
    return sorted_vals[k]


def load_queries(n: int, seed: int = 42) -> list[dict]:
    path = settings.raw_dir / "heldout.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing — run scripts/build_corpus.py first")
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]

    # Stratify by language so percentiles are not dominated by one script's
    # tokenization cost — Malayalam tokenizes 1.66x denser than English, so an
    # unstratified sample would silently shift with the language mix.
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("query"):
            by_lang[r.get("lang", "??")].append(r)

    rng = random.Random(seed)
    per = max(1, n // max(1, len(by_lang)))
    out: list[dict] = []
    for lang, items in by_lang.items():
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out[:n]


def run_retrieval(query: str) -> dict[str, float]:
    """One full retrieval-path request, timed per stage."""
    emb, ret = get_embedder(), get_retriever()
    t: dict[str, float] = {}

    t0 = time.perf_counter_ns()
    q = normalize(query)
    detect_script_lang(q)
    t["normalize"] = (time.perf_counter_ns() - t0) / 1e6

    t1 = time.perf_counter_ns()
    g1_input_sanity(q)
    g2_safety(q)
    t["guards.input"] = (time.perf_counter_ns() - t1) / 1e6

    t2 = time.perf_counter_ns()
    qvec = emb.embed_query(q)
    t["embed"] = (time.perf_counter_ns() - t2) / 1e6

    t3 = time.perf_counter_ns()
    g3_off_topic(qvec, ret.centroids)
    t["guard.offtopic"] = (time.perf_counter_ns() - t3) / 1e6

    t4 = time.perf_counter_ns()
    fused, _ = ret.retrieve(q, qvec, top_k=settings.search_top_k)
    t["retrieve"] = (time.perf_counter_ns() - t4) / 1e6

    t5 = time.perf_counter_ns()
    flat = rrf_flatness(fused)
    top = fused[0].best_dense_score if fused else 0.0
    g4_retrieval_confidence(top, flat)
    t["fuse+guard"] = (time.perf_counter_ns() - t5) / 1e6

    t["TOTAL"] = (time.perf_counter_ns() - t0) / 1e6
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--out", default=str(settings.data_dir / "bench"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading indexes…")
    get_embedder()
    get_retriever()

    queries = load_queries(args.n)
    langs = defaultdict(int)
    for q in queries:
        langs[q.get("lang", "??")] += 1
    print(f"{len(queries)} held-out queries (never indexed): {dict(langs)}")

    # Warmup — discarded. Without this the first request defines P100.
    print(f"warmup ×{args.warmup} (discarded)…")
    for q in queries[: args.warmup]:
        run_retrieval(q["query"])

    if args.mode == "full":
        import urllib.request

        def one(qtext: str) -> dict[str, float]:
            body = json.dumps({"query": qtext, "top_k": 5}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8000/api/v1/query", data=body,
                headers={"Content-Type": "application/json"},
            )
            t0 = time.perf_counter_ns()
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())
            wall = (time.perf_counter_ns() - t0) / 1e6
            tm = d.get("timing", {})
            return {
                "TOTAL": wall,
                "server_total": tm.get("total_ms", 0.0),
                "retrieval": tm.get("retrieval_ms", 0.0),
                "generation": tm.get("generation_ms") or 0.0,
                "_answered": 1.0 if d.get("answered") else 0.0,
            }
    else:
        one = run_retrieval

    per_stage: dict[str, list[float]] = defaultdict(list)
    per_lang: dict[str, list[float]] = defaultdict(list)
    rows = []

    print(f"measuring {len(queries)} queries (mode={args.mode})…")
    for i, q in enumerate(queries, 1):
        try:
            t = one(q["query"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] failed: {str(exc)[:80]}")
            continue
        for k, v in t.items():
            per_stage[k].append(v)
        per_lang[q.get("lang", "??")].append(t["TOTAL"])
        rows.append({"lang": q.get("lang"), "query": q["query"][:80], **t})
        if i % 50 == 0:
            print(f"  {i}/{len(queries)}…")

    totals = sorted(per_stage["TOTAL"])
    n = len(totals)

    print(f"\n{'='*72}")
    print(f"  LATENCY — mode={args.mode}, n={n}, warmup={args.warmup} discarded")
    print(f"{'='*72}")
    print(f"  P50   {percentile(totals,50):8.2f} ms")
    print(f"  P70   {percentile(totals,70):8.2f} ms")
    print(f"  P90   {percentile(totals,90):8.2f} ms")
    print(f"  P95   {percentile(totals,95):8.2f} ms")
    print(f"  P99   {percentile(totals,99):8.2f} ms")
    print(f"  P100  {percentile(totals,100):8.2f} ms   <- max, single worst run")
    print(f"  mean  {statistics.mean(totals):8.2f} ms")
    print(f"  stdev {statistics.pstdev(totals):8.2f} ms")

    if args.mode == "retrieval":
        ok = sum(1 for t in totals if t < 200)
        print(f"\n  SLO <200 ms: {ok}/{n} ({100*ok/n:.1f}%)  "
              f"{'PASS' if ok == n else 'FAIL'}")

    print(f"\n  {'stage':<16}{'P50':>9}{'P70':>9}{'P100':>9}{'share':>8}")
    print(f"  {'-'*50}")
    base = percentile(totals, 50) or 1.0
    for stage, vals in sorted(per_stage.items(), key=lambda kv: -statistics.mean(kv[1])):
        if stage.startswith("_"):
            continue
        s = sorted(vals)
        marker = "  <-- total" if stage == "TOTAL" else ""
        print(f"  {stage:<16}{percentile(s,50):9.2f}{percentile(s,70):9.2f}"
              f"{percentile(s,100):9.2f}{100*percentile(s,50)/base:7.1f}%{marker}")

    print(f"\n  {'language':<16}{'n':>5}{'P50':>9}{'P70':>9}{'P100':>9}")
    print(f"  {'-'*50}")
    for lang, vals in sorted(per_lang.items()):
        s = sorted(vals)
        print(f"  {lang:<16}{len(s):5d}{percentile(s,50):9.2f}"
              f"{percentile(s,70):9.2f}{percentile(s,100):9.2f}")

    if "_answered" in per_stage:
        a = per_stage["_answered"]
        print(f"\n  answered: {int(sum(a))}/{len(a)} "
              f"({100*sum(a)/len(a):.0f}%) — the rest were guardrail refusals")

    summary = {
        "mode": args.mode, "n": n, "warmup_discarded": args.warmup,
        "percentiles": {
            f"P{p}": round(percentile(totals, p), 3)
            for p in (50, 70, 90, 95, 99, 100)
        },
        "mean_ms": round(statistics.mean(totals), 3),
        "stdev_ms": round(statistics.pstdev(totals), 3),
        "slo_200ms_pass_rate": round(
            sum(1 for t in totals if t < 200) / n, 4) if n else 0,
        "stages": {
            k: {"P50": round(percentile(sorted(v), 50), 3),
                "P70": round(percentile(sorted(v), 70), 3),
                "P100": round(percentile(sorted(v), 100), 3)}
            for k, v in per_stage.items() if not k.startswith("_")
        },
        "by_language": {
            k: {"n": len(v), "P50": round(percentile(sorted(v), 50), 3),
                "P100": round(percentile(sorted(v), 100), 3)}
            for k, v in per_lang.items()
        },
        "index": get_retriever().stats()["strategies"],
    }
    (out_dir / f"bench_{args.mode}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    import csv
    with (out_dir / f"bench_{args.mode}.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\n  wrote {out_dir}/bench_{args.mode}.json and .csv")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
