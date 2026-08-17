<div align="center">

# 🎙️ RAAAG!!!

**Voice-enabled, cross-lingual Retrieval-Augmented Generation over 14 Indic languages.**

Ask a question by voice or text in any of 14 Indic languages, get a grounded answer with
citations — and see every pipeline stage measured in millisecond.





[![Live Demo](https://img.shields.io/badge/demo-live-22c55e?style=flat-square)](https://tide-altered-when-pulse.trycloudflare.com/)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15.5-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19.1-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-1.15-4B8BBE?style=flat-square)](https://github.com/facebookresearch/faiss)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.28-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Caddy](https://img.shields.io/badge/Caddy-HTTPS-1F88C0?style=flat-square&logo=caddy&logoColor=white)](https://caddyserver.com/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](#-license)

**[▶ Try the live demo](https://tide-altered-when-pulse.trycloudflare.com/)**

</div>

---

## 📋 Table of contents

- [Why this exists](#-why-this-exists)
- [Live demo](#-live-demo)
- [Architecture](#-architecture)
- [Measured performance](#-measured-performance)
- [Quick start](#-quick-start)
- [Configuration](#-configuration)
- [API reference](#-api-reference)
- [Retrieval design](#-retrieval-design)
- [Chunking strategies](#-chunking-strategies)
- [Guardrails](#-guardrails)
- [Latency budget](#-latency-budget)
- [Dataset notes](#-dataset-notes)
- [Project layout](#-project-layout)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [What is *not* claimed](#-what-is-not-claimed)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Why this exists

Most RAG demos are English-only, single-strategy, and quietly untimed. This one is
built the other way around:

| | |
|---|---|
| 🌏 **Cross-lingual by default** | A Hindi question retrieves the correct English passage. One shared embedding space, no translation hop. |
| 🧩 **Four chunking strategies at once** | `fixed`, `sentence`, `semantic` and `contextual` are all indexed and fused per query — not chosen once at build time. |
| ⚡ **Sub-200ms retrieval, proven** | Every stage is timed and returned in the response body. The demo above answers in **~54ms**. |
| 🛡️ **Refusal as a first-class outcome** | Five guardrails, each with an explicit threshold. Refusing is a designed result, not an error. |
| 🖥️ **CPU-only** | int8-quantized ONNX embeddings + FAISS. No GPU anywhere in the hot path. |

---

## 🌐 Live demo

**<https://tide-altered-when-pulse.trycloudflare.com/>**

Live index statistics, straight from `/api/health` on the deployed instance:

| Metric | Value |
|---|---|
| Documents indexed | **151,675** |
| Total chunks across strategies | **655,137** |
| BM25 vocabulary | **89,427** terms |
| Embedding model | `intfloat/multilingual-e5-small` (int8 ONNX) |
| Vector dimensions | 384 |
| Cold-start load time | ~7.4s (once, at boot) |

> ⚠️ The demo runs through a Cloudflare quick-tunnel, so the hostname rotates when
> the tunnel restarts. If the link is dead, run it locally — [Quick start](#-quick-start)
> takes about five minutes.

---

## 🏗️ Architecture

```
                    ┌─────────────── Next.js 15 / React 19 ───────────────┐
   🎤 voice  ─────▶ │  recorder → WAV  ·  live timing waterfall  ·  cites │
   ⌨️  text   ─────▶ └──────────────────────────┬──────────────────────────┘
                                                │  POST /api/v1/{retrieve,query}
                    ┌───────────────────────────▼──────────────────────────┐
                    │                  FastAPI  (CPU only)                 │
                    │                                                      │
                    │  normalize ──▶ G1 sanity ──▶ G2 safety               │
                    │       │                                              │
                    │       ▼                                              │
                    │  embed  ·  e5-small int8 ONNX  ·  384-dim  (~4ms)    │
                    │       │                                              │
                    │       ▼                                              │
                    │  G3 off-topic gate  ·  64 k-means centroids          │
                    │       │                                              │
                    │       ▼                                              │
                    │  ┌────────── parallel retrieval ──────────┐          │
                    │  │ FAISS: fixed │ sentence │ semantic │   │          │
                    │  │         contextual   +   BM25       │   │          │
                    │  └────────────────┬───────────────────────┘          │
                    │                   ▼                                  │
                    │  Reciprocal Rank Fusion (k=60, weighted)             │
                    │                   ▼                                  │
                    │  MMR diversity (λ=0.7)                               │
                    │                   ▼                                  │
                    │  G4 confidence gate ──▶ refuse if top < τ            │
                    │                   ▼                                  │
                    │  Gemini generation  (excluded from 200ms budget)     │
                    │                   ▼                                  │
                    │  G5 groundedness  ·  G6 citation validation          │
                    └──────────────────────────────────────────────────────┘
```

**Design decisions worth calling out**

- **No PyTorch in the request path.** Embeddings run through `onnxruntime` with an
  int8-quantized graph. `transformers` is present for `AutoTokenizer` only.
- **`OMP_NUM_THREADS` is set before importing onnxruntime or faiss** — see the
  comment at the top of [api/app/main.py](api/app/main.py). Moving those lines below
  the other imports silently costs latency.
- **Models and indexes load exactly once**, in the FastAPI `lifespan` hook. Loading
  per-request would add 200–500ms to every call and would define P100 in any
  benchmark that skipped warmup.

---

## 📊 Measured performance

Captured from the live deployment, `POST /api/retrieve`, query `"symptoms of diabetes"`:

| Stage | Time | In 200ms budget? |
|---|---|:--:|
| `normalize` | 0.04 ms | ✅ |
| `embed` (e5-small int8, 384-dim) | 3.98 ms | ✅ |
| `guard.offtopic` | 0.04 ms | ✅ |
| `retrieve` (5 indexes + RRF fusion) | 49.78 ms | ✅ |
| **Total retrieval** | **54.15 ms** | ✅ **3.7× under budget** |
| `generation` (Gemini) | 400–1500 ms | ❌ excluded — see [Latency budget](#-latency-budget) |

Retrieval quality signals returned inline with every response:

| Signal | Example | Meaning |
|---|---|---|
| `top_score` | `0.9125` | Best passage cosine similarity |
| `n_candidates` | `127` | Unique chunks after cross-index fusion |
| `rrf_flatness` | `1.3938` | Score spread — flat means nothing stood out |
| `off_topic_similarity` | `0.928` | Nearest corpus centroid |

---

## 🚀 Quick start

### Prerequisites

- Python **3.12**
- Node.js **20+**
- ~8 GB free disk for the corpus and indexes
- No GPU required

### 1️⃣ Backend

```bash
cd api
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY for generation
```

### 2️⃣ Build the corpus and indexes

Both steps run **fully offline** — no API key needed.

```bash
python scripts/build_corpus.py     # download + normalize MSMARCO-XI
python scripts/build_index.py      # embed + build FAISS/BM25 indexes
```

> ⏱️ Index building is the slow part and only happens once. Chunking is an
> **ingest-time** cost and never touches the query path.

### 3️⃣ Run the API

```bash
uvicorn app.main:app --reload      # http://localhost:8000
# or, on Windows:
./run.ps1
```

Verify it came up cleanly:

```bash
curl http://localhost:8000/health
```

### 4️⃣ Frontend

```bash
cd ../frontend
npm install
npm run dev                        # http://localhost:3000
```

### 🐳 Or with Docker Compose

```bash
cd deploy
cp ../api/.env.example .env         # fill in keys
docker compose up -d
```

This brings up the API, the Next.js app, and Caddy for HTTPS termination on one
hostname.

---

## ⚙️ Configuration

All settings live in [api/app/config.py](api/app/config.py) and are overridable via
`api/.env`. The defaults below are the deployed values.

<details>
<summary><b>Embedding</b></summary>

| Setting | Default |
|---|---|
| `embed_model_id` | `intfloat/multilingual-e5-small` |
| `embed_onnx_file` | `onnx/model_qint8_avx512_vnni.onnx` |
| `embed_dim` | `384` |
| `embed_max_tokens` | `512` |
| `embed_batch_size` | `256` |
| `query_prefix` / `passage_prefix` | `"query: "` / `"passage: "` |

E5 models are trained with asymmetric prefixes; dropping them measurably degrades
retrieval, so they are applied automatically.
</details>

<details>
<summary><b>Threading</b></summary>

| Setting | Default |
|---|---|
| `omp_threads` | `4` |
| `onnx_intra_op_threads` | `4` |
| `onnx_inter_op_threads` | `1` |
</details>

<details>
<summary><b>Retrieval &amp; fusion</b></summary>

| Setting | Default | Purpose |
|---|---|---|
| `search_top_k` | `50` | Per-strategy candidate depth before fusion |
| `rrf_k` | `60` | RRF smoothing constant |
| `mmr_lambda` | `0.7` | Relevance vs. diversity trade-off |
| `kmeans_centroids` | `64` | Powers the G3 off-topic gate |

Fusion weights — `contextual` is boosted because it carries document context into
the chunk; `bm25` is damped because lexical matching misfires across scripts:

| Index | Weight |
|---|---|
| `fixed` | 1.0 |
| `sentence` | 1.0 |
| `semantic` | 1.0 |
| `contextual` | **1.2** |
| `bm25` | **0.8** |
</details>

<details>
<summary><b>Guardrail thresholds</b></summary>

| Setting | Default | Gate |
|---|---|---|
| `tau_topic` | `0.70` | G3 off-topic |
| `tau_conf` | `0.72` | G4 low retrieval confidence |
| `tau_flatness` | `1.04` | G4 flat-score detection |
| `flatness_depth` | `10` | Depth for the flatness check |
| `tau_ground` | `0.55` | G5 per-sentence support cosine |
| `ground_refuse_below` | `0.60` | G5 hard refusal |
| `ground_warn_below` | `0.85` | G5 warning band |
</details>

<details>
<summary><b>Generation &amp; voice</b></summary>

| Setting | Default |
|---|---|
| `gemini_model` | `gemini-3.5-flash-lite` |
| `gemini_fallback_model` | `gemini-3.1-flash-lite` |
| `generation_timeout_ms` | `15000` |
| `max_tool_hops` | `1` |
| `elevenlabs_stt_model` | `scribe_v2` |
</details>

### 🔑 API keys

| Key | Enables | Without it |
|---|---|---|
| `GEMINI_API_KEY` | Answer generation | Retrieval still works; no generated prose |
| `ELEVENLABS_API_KEY` | Voice input / playback | Text input still works |
| `HF_TOKEN` | Faster HuggingFace downloads | Anonymous downloads still work |

> 🔒 `api/.env` is gitignored. Never commit or paste real keys. Corpus building,
> indexing, chunking, embedding and retrieval all run **without any key at all**.

---

## 🔌 API reference

Interactive docs at `/docs` when running locally.

### `POST /api/v1/retrieve`

Retrieval only — no LLM call, no generation latency.

```bash
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "what is machine learning", "k": 3}'
```

<details>
<summary><b>Response (abridged, from the live instance)</b></summary>

```json
{
  "query": "what is machine learning",
  "normalized_query": "what is machine learning",
  "detected_lang": "en",
  "n_candidates": 127,
  "top_score": 0.9056,
  "rrf_flatness": 1.3938,
  "off_topic_similarity": 0.928,
  "passages": [
    {
      "doc_id": "1056989:hi:4:en",
      "chunk_id": "semantic:1056989:hi:4:en:0",
      "text": "What is Deep Learning and how is it different than traditional machine learning? ...",
      "lang": "en",
      "score": 0.9056,
      "rrf_score": 0.081491,
      "is_gold": false,
      "strategies": { "fixed": 1, "sentence": 2, "semantic": 1, "contextual": 1, "bm25": 2 }
    }
  ],
  "timing": {
    "total_ms": 54.149,
    "retrieval_ms": 54.148,
    "generation_ms": null,
    "stages": [
      { "stage": "normalize",       "ms": 0.042, "ok": true },
      { "stage": "embed",           "ms": 3.984, "ok": true, "detail": { "dim": 384 } },
      { "stage": "guard.offtopic",  "ms": 0.040, "ok": true, "detail": { "max_sim": 0.928 } },
      { "stage": "retrieve",        "ms": 49.782, "ok": true, "detail": { "n_fused": 127 } }
    ]
  }
}
```

Note `strategies` — it shows the per-index rank each chunk earned, so you can see
*why* fusion promoted it.
</details>

### `POST /api/v1/query`

Full pipeline: retrieval → generation → groundedness and citation checks.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "मधुमेह के लक्षण क्या हैं"}'
```

### Other endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness, startup time, full index stats |
| `GET` | `/api/v1/stats` | Corpus and index statistics |
| `GET` | `/api/v1/metrics` | Accumulated per-stage latency percentiles |
| `POST` | `/api/v1/metrics/reset` | Clear accumulated metrics |
| `GET` | `/api/v1/config` | Effective (non-secret) configuration |

---

## 🔍 Retrieval design

**Every query hits five indexes in parallel**, then results are fused. This is the
core architectural bet: rather than picking one chunking strategy at build time,
index all of them and let per-query evidence decide.

**Reciprocal Rank Fusion** (`k=60`) combines the ranked lists. RRF is used instead of
score averaging because raw scores from FAISS cosine and BM25 aren't on comparable
scales — ranks are.

**MMR** (`λ=0.7`) then removes near-duplicates. Without it the top-5 is frequently
five overlapping windows over the same sentence, which wastes the generator's
context on redundancy.

**Cross-lingual retrieval** falls out of the shared e5 embedding space — a Hindi
query and its English answer land near each other with no translation step. Note the
example above: an English query matched a passage under `doc_id` `1056989:hi:4:en`.

---

## 🧩 Chunking strategies

All four are built at ingest time and searched simultaneously.

| Strategy | Idea | Live chunk count |
|---|---|---|
| `fixed` | Token windows, 256 tokens with 64 overlap — the control | **155,110** |
| `sentence` | Whole sentences grouped to a ≤200-token budget | **157,976** |
| `semantic` | Cut at embedding-similarity breakpoints (95th percentile, 64–384 tokens) | **181,707** |
| `contextual` | Chunk carries surrounding document context — weighted highest at fusion | **160,381** |

A blind `fixed` control is included on purpose: showing that the sophisticated
strategies beat a naive baseline is stronger evidence than only showing the
sophisticated ones.

**Script-aware sentence splitting.** Devanagari danda (।), Urdu full stop (۔) and
Latin terminators are all handled. Off-the-shelf splitters trained on European text
treat an entire Hindi paragraph as one sentence, which silently destroys every
sentence-based strategy.

---

## 🛡️ Guardrails

Six checks, each with an explicit threshold, each reported in the response so a
refusal is always explainable.

| Gate | When | Checks |
|---|---|---|
| **G1** sanity | Pre-retrieval | Empty, degenerate, or zero-width-smuggled input |
| **G2** safety | Pre-retrieval | Prompt injection and unsafe requests |
| **G3** off-topic | Pre-retrieval | Similarity to 64 corpus centroids vs. `tau_topic` (0.70) |
| **G4** confidence | Post-retrieval | Top score vs. `tau_conf` (0.72) **and** score flatness vs. `tau_flatness` (1.04) |
| **G5** groundedness | Post-generation | Per-sentence support; refuse below 0.60, warn below 0.85 |
| **G6** citations | Post-generation | Every citation resolves to a retrieved chunk |

G4's flatness check matters more than the raw threshold: a query can clear the score
bar while every candidate is equally mediocre. Flat scores mean nothing actually
stood out, which is a different failure from "nothing scored well" and needs its own
gate.

Implementation: [api/app/guardrails/pipeline.py](api/app/guardrails/pipeline.py).

---

## ⏱️ Latency budget

**The sub-200ms budget covers query-time retrieval:** normalize, embed, off-topic
gate, multi-index search, fusion, MMR, and the confidence gate. Live measurement:
**54ms**.

**Generation is excluded and reported separately.** A single LLM call is 400–1500ms;
no system completes one in 200ms, and claiming otherwise would require redefining
either "latency" or "generation". The response returns `retrieval_ms` and
`generation_ms` as separate fields so the distinction is machine-readable, and the
UI renders budgeted stages solid and excluded stages hatched.

**Chunking is not in the budget** because it is an ingest-time cost. It happens once
in `build_index.py` and never during a query.

**Why it's fast on CPU:**
- int8-quantized ONNX embeddings — ~4ms per query
- Indexes and model loaded once at startup, never per request
- Thread counts pinned before onnxruntime/faiss import
- FAISS flat indexes: exact search, no recall loss from approximation

---

## 📚 Dataset notes

Built on [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).
Verified against the HF datasets-server rather than the dataset README, which is
stale about both file format and config layout:

- **55.6 GB** parquet, **11,451,314** rows, and exactly **one** `default` config —
  languages are split by *filename*, not by config
- One row per query with ~10 passages nested inline
- **No passage IDs** — document IDs are synthesized as content hashes and duplicates
  merged
- `is_selected` gives binary relevance labels, making Recall@k / MRR@10 / nDCG@10
  directly measurable
- **Telugu has a validation file but no train file**
- Labels are sparse: a passage that genuinely answers the query but was never marked
  relevant counts as a miss, so **absolute recall understates real quality**

Read parquet directly with `pyarrow` — the `datasets` library is deliberately avoided
in the ingest path.

---

## 📁 Project layout

```
api/                            # FastAPI backend (deployed service)
├── app/
│   ├── main.py                 # lifespan loading, route mounting
│   ├── config.py               # all tunables, .env-overridable
│   ├── api/
│   │   ├── routes_retrieve.py  # POST /api/v1/retrieve
│   │   └── routes_query.py     # POST /api/v1/query
│   ├── chunking/               # fixed · sentence · semantic · contextual
│   ├── embedding/
│   │   └── onnx_embedder.py    # int8 ONNX, CPU only
│   ├── index/
│   │   ├── faiss_store.py      # per-strategy vector indexes
│   │   ├── bm25_store.py       # lexical index
│   │   ├── fusion.py           # RRF + MMR
│   │   └── multi_index.py      # parallel search across all indexes
│   ├── guardrails/pipeline.py  # G1–G6
│   ├── llm/gemini_client.py    # generation with fallback model
│   ├── core/                   # latency, timing, text normalization
│   └── schemas/query.py        # typed request/response contracts
├── scripts/
│   ├── build_corpus.py         # download + normalize (offline)
│   ├── build_index.py          # embed + index (offline)
│   ├── benchmark.py            # retrieval benchmarks
│   └── benchmark_voice.py      # STT/TTS benchmarks
└── requirements.txt

frontend/                       # Next.js 15 · React 19
├── app/                        # app router, design system
├── components/                 # Chat · VoiceInput · Metrics · Result
└── lib/                        # typed client, recorder, WAV encoding

backend/                        # earlier Qdrant/Groq implementation (superseded)
deploy/                         # docker-compose · Caddyfile · deploy scripts
docs/                           # design documents
```

> ℹ️ **`api/` vs `backend/`** — `api/` is the current, deployed service (FAISS +
> ONNX + Gemini). `backend/` is the earlier implementation (Qdrant + Groq/Sarvam),
> kept for reference. New work belongs in `api/`.

---

## 🧪 Testing

```bash
cd api
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check .
```

The earlier `backend/` implementation carries a 220-test suite covering chunking
across three scripts, guardrails in both directions (legitimate queries must pass,
attacks must not), provider failover, JSON repair, and refusal paths:

```bash
cd backend && python -m pytest tests/ -v
```

---

## 🚢 Deployment

Deployed on an **Oracle Cloud ARM64** instance via Docker Compose behind Caddy.

```bash
cd deploy
docker compose up -d
```

Caddy terminates HTTPS on a single hostname and routes:

| Path | Upstream |
|---|---|
| `/api/*` | `api:8000` (120s header timeout for cold model load) |
| `/docs*`, `/openapi.json` | `api:8000` |
| everything else | `web:3000` |

**HTTPS is mandatory, not cosmetic** — browsers block `getUserMedia` on insecure
origins, so voice input is dead over plain HTTP regardless of what the app does.

`sslip.io` resolves any IP-shaped hostname to that IP, so the deployment works with
no registrar and no DNS record; Caddy provisions a Let's Encrypt certificate on
first start. To move to a real domain, replace the hostname in
[deploy/Caddyfile](deploy/Caddyfile) and point its A record at the host — nothing
else changes.

---

## ⚖️ What is *not* claimed

Stated plainly, because RAG benchmarks are easy to overstate:

- **Not 100% accuracy.** No RAG system achieves that. Accuracy is measured against
  `is_selected` labels and reported as measured.
- **Not sub-200ms including generation.** The budget covers retrieval; generation is
  reported separately. See [Latency budget](#-latency-budget).
- **Not uniform quality across all 14 languages.** Corpus coverage and embedding
  quality vary per language.
- **Recall figures understate real quality**, because the dataset's relevance labels
  are sparse.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository and create a branch: `git checkout -b feature/your-feature`
2. Make changes in `api/` (not the superseded `backend/`)
3. Run `pytest tests/ -v` and `ruff check .`
4. Commit, push, and open a pull request describing what changed and why

For latency-sensitive changes, include before/after numbers from
`scripts/benchmark.py` — this project treats measured latency as a contract.

---

## 📄 License

Released under the [MIT License](https://opensource.org/licenses/MIT).

The MSMARCO-XI dataset is subject to its own
[license terms](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).

---

<div align="center">

**[▶ Live demo](https://tide-altered-when-pulse.trycloudflare.com/)** · Built with FastAPI, FAISS, ONNX Runtime and Next.js

</div>
