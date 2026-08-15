# Voice-Enabled Cross-Lingual RAG — Design

**Date:** 2026-08-15
**Status:** Approved (architecture section), implementation starting

## Purpose

A voice-first RAG system over `ai4bharat/MSMARCO-XI`. A user speaks a question in
any of 14 Indic languages; the system transcribes it, retrieves grounded context
across all languages, generates a cited answer, and can read that answer back
aloud. Every stage is observable in the UI: timings, retrieved chunks, and
guardrail verdicts.

## Dataset Facts (verified 2026-08-15)

Verified against the HF datasets-server API, not the README — the README is stale
(it claims JSONL files and per-language configs; neither exists).

| Property | Value |
|---|---|
| Access | Public, ungated, no token required |
| Size | 55.6 GB parquet / 11,451,314 rows / ~136.6 GB uncompressed |
| Configs | **One** (`default`) — languages split by *filename*, not config |
| Splits | train 10,080,140 rows · validation 1,371,174 rows |
| Format | Parquet only, 27 files (13 train + 14 validation) |
| Naming | `train/<lang3>train.parquet`, `validation/<lang3>val.parquet` |
| Gap | **No Telugu train file** — Telugu is validation-only |

**Schema** — one row per query, passages nested inline. There is no
corpus/queries/qrels triple and **no passage_id field**:

```
source_lang   string              e.g. "eng_Latn"
target_lang   string              e.g. "asm_Beng"  (FLORES lang_Script codes)
meta          struct{model_name, temperature, max_tokens, top_p,
                    frequency_penalty, presence_penalty}
Answer        string              translated answer
query_id      int64
query_type    string              DESCRIPTION|NUMERIC|ENTITY|LOCATION|PERSON
passages      struct{English_passages:    Sequence[string],
                    Translated_passages:  Sequence[string],
                    is_selected:          Sequence[int64]}
Eng_Query     string
Eng_Answer    string
query         string              translated query
```

**Implications that drive the design:**

1. **Synthesize doc IDs.** No `passage_id` exists. Use a stable content hash
   (SHA-1 of normalized text, truncated) so the same passage recurring across
   queries dedups to one vector.
2. **Relevance labels exist.** `is_selected` is a binary qrel — standard MS MARCO
   shape, ~10 passages per query, one marked relevant. This makes Recall@k,
   MRR@10 and nDCG@10 directly computable. Accuracy claims will be measured, not
   asserted.
3. **Labels are binary and sparse.** nDCG carries no graded relevance, and MS
   MARCO is known to be incompletely labeled, so absolute Recall against a large
   pooled corpus reads pessimistically. Report the pooling method alongside the
   numbers.
4. **Every row is bilingual.** English and translated text sit side by side —
   this is what makes cross-lingual retrieval possible for free.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Scope | 14 languages, validation split, capped per language | Full coverage within a CPU budget; cap is configurable |
| Hardware | CPU-only ingest (local Windows machine) | What's available; ONNX-optimized to compensate |
| Embedder | `intfloat/multilingual-e5-large` (1024-dim) | Strong multilingual retrieval, proven on MS MARCO-shaped tasks |
| Vector DB | Qdrant (Docker) with binary quantization | Native BQ + oversample/rescore; no hand-rolled two-stage search |
| STT | Sarvam `saarika` | Purpose-built for Indic speech |
| TTS | Sarvam `bulbul` | Same provider closes the voice loop |
| Generation | Groq **and** Gemini with automatic failover | User-provided keys; resilience is an explicit harness requirement |
| Chunking | 6 strategies benchmarked, winner used for full index | Evidence for the "vast chunking" requirement at 1× full-ingest cost |
| Guardrails | All four layers, every verdict surfaced in UI | Brief requires knowing *when not to answer* |
| Cross-lingual | Retrieve across all languages, answer in query's language | Uses the dataset's parallel structure; most useful behaviour |
| Deployment | User's own VM, Docker Compose, index built locally and shipped | VM specs TBD — parameterized until known |
| Build order | Vertical slice (one language, full path) then breadth | Demoable early, de-risked |

## Latency Budget

The brief's "<200ms for everything through final output" is not physically
achievable when final output includes an LLM call — a single generation is
400-1500ms and STT is 300-800ms, both network-bound third-party calls. The
system is therefore designed and reported as:

**Under the 200ms bar** (measured, P50/P70/P100):

| Stage | Budget |
|---|---|
| Input guardrail screening | ~1-3ms |
| Query embedding (ONNX, CPU) | ~15-25ms |
| Binary vector search | ~3-8ms |
| Full-precision rescore (top-200) | ~5-10ms |
| Cross-encoder rerank (top-50) | ~30-60ms |
| Confidence gate | <1ms |
| **Total query-time retrieval** | **~55-105ms** |

**Reported separately, same percentile treatment:** STT, generation, grounding
verification, TTS.

Chunking is an ingest-time cost, not a query-time one. The submission states this
explicitly rather than quietly excluding it.

**Binary quantization math:** 1024 dims × float32 = 4KB/vector → 128 bytes
quantized, a 32× reduction. 3M vectors: 12GB → 384MB, RAM-resident on a modest
VM. Qdrant searches Hamming distance, oversamples 3×, rescores against
full-precision vectors on disk. Recall retention will be *measured on this
dataset*, not cited from literature.

## Chunking Strategies

Six strategies, compared on a ~5k labeled-query subset, scored by
Recall@1/5/10, MRR@10, nDCG@10, chunk count, and build time.

| # | Strategy | Own embedding pass? |
|---|---|---|
| 1 | Passage-native (MS MARCO's own boundaries) | Yes — baseline |
| 2 | Fixed-size + overlap (control group) | Yes (subset only) |
| 3 | Recursive character splitting (control) | Yes (subset only) |
| 4 | Semantic (embedding-similarity breakpoints) | Yes (subset only) |
| 5 | Sentence-window (embed narrow, return wide) | Yes (subset only) |
| 6 | Parent-child small-to-big | **No** — reuses #1 vectors + parent map |
| 7 | Late chunking (pool spans from one forward pass) | Shares #1's forward pass |

All carry metadata: `language`, `query_type`, `is_selected`, `source_query_id`,
`doc_hash`. Metadata-aware filtering is therefore available to every strategy.

Controls (#2, #3) are included deliberately — proving the sophisticated
strategies beat naive ones is stronger evidence than only showing the
sophisticated ones.

**Cost control:** benchmark on the subset, then one full-scale embedding pass
using the winner. Embeddings persist to disk; server startup memory-maps the
index and never recomputes.

## Guardrails

Four layers, each rendering its verdict, score, and reasoning in the UI.

1. **Input safety** — prompt-injection and unsafe-content screening before
   retrieval.
2. **Off-topic detection** — is this answerable from an MS MARCO web corpus at
   all?
3. **Retrieval confidence gate** — refuse when top-k similarity falls below
   threshold. This is the primary hallucination defense: no context, no answer.
4. **Grounding verification** — every generated claim must trace to a retrieved
   passage; citations attached, ungrounded claims flagged.

Refusal is a first-class outcome with its own UI treatment, not an error state.

## Harness

Structured orchestration, not a raw prompt call:

- Typed request/response contracts (Pydantic) at every stage boundary
- Retry with exponential backoff on transient provider failures
- Circuit-breaking failover: Groq → Gemini → graceful degradation to
  extractive-only answering
- Structured LLM output (JSON schema) for answers, citations, and grounding
  verdicts
- Per-stage timing instrumentation feeding the analytics store
- Error recovery at each boundary — a failed stage degrades the response rather
  than failing the request

## UI

Ubuntu font throughout. Not a Claude-style interface — a purpose-built
retrieval console.

1. **Chat** — push-to-talk with live waveform, text input fallback,
   click-to-hear TTS on answers, inline citations
2. **Observability panel** — per-query timing waterfall, retrieved chunks with
   scores, guardrail verdicts
3. **Dataset preview** — paginated browser with language filter, showing
   queries, passages and `is_selected` labels
4. **Latency analytics** — P50/P70/P90/P100 per stage across all queries run
5. **Chunking comparison** — benchmark results table

## Security

- API keys server-side only, never exposed to the browser
- `.env` gitignored from the first commit
- Per-IP rate limiting, global daily request cap, max audio duration
- Public URL must not be able to drain API credits

## Open Items

- **VM provider and specs** — RAM determines index size, which determines the
  per-language row cap. Parameterized until known.
- **API keys** — user supplies Sarvam, Groq, Gemini in `backend/.env`. System
  runs with graceful degradation before they land.

## Explicitly Not Claimed

- **Not 100% accuracy.** No RAG system achieves this. The system measures real
  Recall@k/MRR@10/nDCG@10 against `is_selected` labels and optimizes hard
  (hybrid search + reranking). Strong MS MARCO systems reach ~85-95% Recall@10;
  actual measured numbers get reported, whatever they are.
- **Not <200ms including generation.** See Latency Budget above.
