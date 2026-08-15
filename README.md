# VoiceRAG — MSMARCO-XI

Voice-enabled cross-lingual retrieval-augmented generation over
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).
Speak a question in any of 14 Indic languages, get a grounded answer with
citations, and hear it read back — with every pipeline stage measurable in the
UI.

```
voice → Sarvam STT → guardrails → embed → binary-quantized vector search
      → cross-encoder rerank → confidence gate → Groq/Gemini → grounding check
```

## Quick start

```bash
# 1. Vector database
docker run -d --name hhg-qdrant -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.12.4

# 2. Backend
cd backend
python -m venv venv && ./venv/Scripts/activate    # Windows
pip install -r requirements.txt
cp .env.example .env                              # add your API keys
python scripts/ingest.py --languages hi --limit 2000
uvicorn app.main:app --reload

# 3. Frontend
cd ../frontend && npm install && npm run dev
```

Open http://localhost:3000.

### Or with Docker Compose

```bash
cp backend/.env.example .env    # fill in keys
docker compose up -d
```

## API keys

All three are optional — the system degrades rather than failing:

| Key | Enables | Without it |
|---|---|---|
| `SARVAM_API_KEY` | Voice input + audio playback | Text input still works |
| `GROQ_API_KEY` | Answer generation (primary) | Falls over to Gemini |
| `GEMINI_API_KEY` | Answer generation (fallback) | Falls back to extractive answers |

Put them in `backend/.env`, which is gitignored. Never commit or paste keys.

## Requirements addressed

**1. Speech-to-text — Sarvam.** `saarika:v2` for transcription with
auto-detect, `bulbul:v2` for playback. Chosen over ElevenLabs because the
corpus is Indic and Sarvam is built for these languages. Four dataset languages
(Assamese, Nepali, Sanskrit, Urdu) have no Sarvam locale; those transcripts are
flagged lower-confidence rather than silently presented as reliable.

**2. Chunking — seven strategies, benchmarked.**

| Strategy | Idea |
|---|---|
| `passage_native` | MS MARCO's own passage boundaries (baseline) |
| `fixed_size` | Blind character windows with overlap (control) |
| `recursive_character` | Script-aware separator hierarchy (control) |
| `sentence` | Whole sentences grouped to a budget |
| `sentence_window` | Embed one sentence, return its neighbourhood |
| `parent_child` | Embed sentence groups, return the full passage |
| `semantic` | Cut at embedding-similarity breakpoints, percentile-adaptive |
| `late_chunking` | Pool token vectors from one full-passage forward pass |

Controls are included deliberately: proving the sophisticated strategies beat
naive ones is stronger evidence than only showing the sophisticated ones. Run
`python scripts/benchmark_chunking.py --language hi` to produce the comparison
table, which the UI's Chunking tab reads.

Sentence splitting handles Devanagari danda (।), Urdu full stop (۔) and Latin
terminators — off-the-shelf splitters trained on European text treat a Hindi
paragraph as one sentence.

**3. Latency.** The sub-200ms budget covers **query-time retrieval**: input
guardrails, query embedding, vector search, rerank, and the confidence gate.
Generation is excluded and reported separately, because a single LLM call is
400-1500ms and no system completes one in 200ms. Chunking is an ingest-time
cost, not a query-time one. The UI draws budgeted stages solid and excluded
stages hatched, so the claim is visually precise about its scope.

Binary quantization is the main mechanism: 384 float32 dims → 384 bits, a 32×
memory cut, with Hamming-distance search over packed bits and a full-precision
rescore of the oversampled top candidates.

**4. Latency analytics.** `/api/analytics/latency` and the Latency tab report
P50/P70/P90/P95/P100 per stage with sample counts, accumulated across every
query rather than a single best-case run.

**5. Harness.** Typed Pydantic contracts at every stage boundary; ordered
provider failover with per-provider circuit breakers; retry with exponential
backoff and jitter for transient errors only (auth failures fail over
immediately rather than burning the latency budget); structured JSON output
with repair for markdown fences, surrounding prose and token-limit truncation;
graceful degradation to extractive answers when every provider is down.

**6. Guardrails — four layers, all surfaced in the UI.**

| Layer | When | What |
|---|---|---|
| Input safety | Pre-retrieval | Prompt injection, unsafe requests, zero-width smuggling |
| Topic scope | Pre-retrieval | Meta, generative and personal-advice queries |
| Confidence gate | Post-retrieval | Refuses before generation when nothing relevant was found |
| Grounding | Post-generation | Traces every answer sentence back to retrieved passages |

Refusals are first-class outcomes with their own UI treatment, not errors.

## Dataset notes

Verified against the HF datasets-server, not the README (which is stale about
both file format and config layout):

- 55.6 GB parquet, 11,451,314 rows, **one** `default` config — languages are
  split by *filename*, not config
- One row per query with ~10 passages nested inline; **no passage IDs**, so doc
  IDs are synthesized as content hashes and duplicates merged
- `is_selected` gives binary relevance labels, which makes Recall@k / MRR@10 /
  nDCG@10 directly measurable
- Telugu has a validation file but **no train file**
- Labels are sparse: a passage that answers the query but was never marked
  relevant counts as a miss, so absolute recall understates real quality

## Measured on this hardware (8-core CPU, no GPU)

| Finding | Number |
|---|---|
| e5-large embedding | 3.4 passages/sec |
| e5-small embedding | 32.2 passages/sec |
| Qdrant search (5k vectors, binary quantized) | p50 15.6ms, p100 30.1ms |
| Cross-lingual check: Hindi query → correct English passage | 0.823 |
| Cross-lingual check: Hindi query → unrelated passage | 0.700 |

The 9.5× embedding gap is why the pipeline uses e5-small: it is the difference
between a 17-hour and a 7-day ingest for all 14 languages. The reranker
recovers much of the accuracy difference since it does the final ordering.

## What is not claimed

- **Not 100% accuracy.** No RAG system achieves that. Accuracy is measured
  against `is_selected` labels and reported as measured.
- **Not sub-200ms including generation.** See the latency section.

## Tests

```bash
cd backend && ./venv/Scripts/python -m pytest tests/ -v
```

220 tests covering chunking across three scripts, guardrail behaviour in both
directions (legitimate queries must pass, attacks must not), provider failover
and circuit breaking, JSON repair, and the orchestrator's refusal paths.

## Layout

```
backend/
  app/core/         config, languages, typed schemas, timing, logging
  app/ingest/       dataset loader, text utils, chunking strategies
  app/retrieval/    embedder, Qdrant store, cross-encoder reranker
  app/guardrails/   input safety, topic scope, confidence, grounding
  app/harness/      providers, generation, voice, orchestrator
  app/eval/         retrieval metrics and evaluation runner
  app/api/          FastAPI routes, analytics store
  scripts/          ingest.py, benchmark_chunking.py
frontend/
  app/              Next.js app router, Ubuntu-font design system
  components/       chat, waveform, timing waterfall, guardrails, dataset
  lib/              typed API client, recorder hook
docs/superpowers/specs/   design document
```
