"""End-to-end VOICE latency: speech in -> spoken answer out.

Measures the four legs a user actually waits through:

    STT (Sarvam)  ->  retrieval (local)  ->  generation (Gemini)  ->  TTS (Sarvam)

Real audio is used, not a stub: each probe question is first synthesized with
TTS, and that audio file is what gets fed to STT. That keeps the measurement
honest — a canned WAV of studio speech would understate STT time.

Note on the <200 ms target: it applies to the RETRIEVAL pipeline. No hosted LLM
or speech API returns in 200 ms, so the network legs are reported separately
rather than folded in — which is also why the brief asks for generation latency
to be reported apart from retrieval.

Usage:
    python scripts/benchmark_voice.py --n 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

WEB = "http://localhost:3000"
API = "http://127.0.0.1:8000"

PROBES = [
    ("what is a corporation", "en"),
    ("what is an ITIN number", "en"),
    ("how does photosynthesis work", "en"),
    ("what is chromedriver used for", "en"),
    ("कॉर्पोरेशन क्या है", "hi"),
    ("मधुमेह का कारण क्या है", "hi"),
    ("निगम कैसे काम करता है", "hi"),
    ("पौधे भोजन कैसे बनाते हैं", "hi"),
]


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(p / 100 * len(s) + 0.5)) - 1))
    return s[k]


def post_json(url: str, payload: dict, timeout: int = 90):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter_ns()
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    return raw, (time.perf_counter_ns() - t0) / 1e6


def post_audio(url: str, audio: bytes, filename: str, mime: str, timeout: int = 90):
    b = f"----{uuid.uuid4().hex}"
    head = (
        f'--{b}\r\nContent-Disposition: form-data; name="audio"; '
        f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
    ).encode()
    body = head + audio + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"},
    )
    t0 = time.perf_counter_ns()
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    return raw, (time.perf_counter_ns() - t0) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    probes = (PROBES * ((args.n // len(PROBES)) + 1))[: args.n]

    # --- synthesize the probe audio once, up front (not counted) ---------
    print(f"synthesizing {len(probes)} probe clips via TTS…")
    clips: list[tuple[str, str, bytes]] = []
    for text, lang in probes:
        raw, _ = post_json(f"{WEB}/api/tts", {"text": text, "lang": lang})
        clips.append((text, lang, raw))
    print(f"  got {sum(len(c[2]) for c in clips)/1024:.0f} KB of audio\n")

    legs: dict[str, list[float]] = {
        "stt": [], "retrieval": [], "generation": [], "tts": [],
        "server_total": [], "e2e": [],
    }
    rows = []
    answered = 0

    total = len(clips)
    for i, (text, lang, audio) in enumerate(clips, 1):
        warm = i <= args.warmup
        try:
            e0 = time.perf_counter_ns()

            stt_raw, stt_ms = post_audio(
                f"{WEB}/api/stt", audio, "input.mp3", "audio/mpeg")
            stt = json.loads(stt_raw)
            transcript = stt.get("text") or text

            q_raw, q_ms = post_json(
                f"{API}/api/v1/query", {"query": transcript, "top_k": 5})
            q = json.loads(q_raw)
            tm = q.get("timing", {})

            spoken = q.get("answer") or q.get("refusal_message") or ""
            tts_ms = 0.0
            if spoken:
                _, tts_ms = post_json(
                    f"{WEB}/api/tts", {"text": spoken[:900], "lang": lang})

            e2e = (time.perf_counter_ns() - e0) / 1e6

            if warm:
                print(f"  [{i}/{total}] warmup discarded ({e2e:.0f} ms)")
                continue

            legs["stt"].append(stt_ms)
            legs["retrieval"].append(tm.get("retrieval_ms", 0.0))
            legs["generation"].append(tm.get("generation_ms") or 0.0)
            legs["tts"].append(tts_ms)
            legs["server_total"].append(tm.get("total_ms", 0.0))
            legs["e2e"].append(e2e)
            answered += 1 if q.get("answered") else 0

            rows.append({
                "query": text, "lang": lang, "transcript": transcript,
                "answered": q.get("answered"), "stt_ms": round(stt_ms, 1),
                "retrieval_ms": round(tm.get("retrieval_ms", 0.0), 2),
                "generation_ms": round(tm.get("generation_ms") or 0.0, 1),
                "tts_ms": round(tts_ms, 1), "e2e_ms": round(e2e, 1),
            })
            flag = "ans" if q.get("answered") else f"REFUSE:{q.get('refusal_reason')}"
            print(f"  [{i}/{total}] {lang} {flag:22s} "
                  f"stt={stt_ms:6.0f} ret={tm.get('retrieval_ms',0):6.1f} "
                  f"gen={(tm.get('generation_ms') or 0):7.0f} tts={tts_ms:6.0f} "
                  f"| e2e={e2e:7.0f} ms")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{total}] FAILED: {str(exc)[:110]}")

    n = len(legs["e2e"])
    if not n:
        raise SystemExit("no successful runs")

    print(f"\n{'='*76}")
    print(f"  END-TO-END VOICE LATENCY — n={n} ({args.warmup} warmup discarded)")
    print(f"{'='*76}")
    print(f"  {'leg':<26}{'P50':>10}{'P70':>10}{'P100':>10}{'share':>9}")
    print(f"  {'-'*65}")
    base = pct(legs["e2e"], 50) or 1.0
    for name, label in [
        ("stt", "1. Speech-to-text"),
        ("retrieval", "2. Retrieval (SLO path)"),
        ("generation", "3. Generation (Gemini)"),
        ("tts", "4. Text-to-speech"),
    ]:
        v = legs[name]
        print(f"  {label:<26}{pct(v,50):10.1f}{pct(v,70):10.1f}"
              f"{pct(v,100):10.1f}{100*pct(v,50)/base:8.1f}%")
    print(f"  {'-'*65}")
    v = legs["e2e"]
    print(f"  {'TOTAL (user waits)':<26}{pct(v,50):10.1f}{pct(v,70):10.1f}"
          f"{pct(v,100):10.1f}{100:8.1f}%")
    print(f"\n  mean {statistics.mean(v):.0f} ms | stdev {statistics.pstdev(v):.0f} ms")
    print(f"  answered {answered}/{n} (rest were guardrail refusals)")

    r = legs["retrieval"]
    ok = sum(1 for x in r if x < 200)
    print(f"\n  RETRIEVAL SLO <200 ms: {ok}/{n} ({100*ok/n:.0f}%) "
          f"{'PASS' if ok == n else 'FAIL'}  [P100 = {pct(r,100):.1f} ms]")
    print(f"  Network legs (STT+gen+TTS) are hosted-API calls and are reported")
    print(f"  separately; none of them can meet a 200 ms budget.")

    out = settings.data_dir / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_voice.json").write_text(json.dumps({
        "n": n,
        "legs": {k: {"P50": round(pct(v, 50), 2), "P70": round(pct(v, 70), 2),
                     "P100": round(pct(v, 100), 2)} for k, v in legs.items()},
        "answered": answered,
        "retrieval_slo_pass_rate": round(ok / n, 4),
        "runs": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {out}/bench_voice.json")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()
