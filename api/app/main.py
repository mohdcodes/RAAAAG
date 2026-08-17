"""FastAPI application.

`OMP_NUM_THREADS` is set before any import that reaches onnxruntime or faiss.
Moving these lines below the other imports silently costs latency — see
app/config.py.
"""

from __future__ import annotations

import os

from app.config import settings

os.environ.setdefault("OMP_NUM_THREADS", str(settings.omp_threads))

import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.api import routes_query, routes_retrieve  # noqa: E402
from app.core import latency  # noqa: E402
from app.embedding.onnx_embedder import get_embedder  # noqa: E402
from app.index.multi_index import get_retriever  # noqa: E402

STATE: dict = {"ready": False, "error": None, "startup_ms": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX session and every index ONCE.

    Doing this per request would add 200-500 ms of graph optimization and index
    deserialization to every call — on its own enough to blow the budget, and
    it would define P100 in any benchmark that skipped warmup.
    """
    t0 = time.perf_counter()
    try:
        emb = get_embedder()           # constructor also warms the session
        ret = get_retriever()
        STATE["ready"] = True
        STATE["startup_ms"] = round((time.perf_counter() - t0) * 1000)
        s = ret.stats()
        print(
            f"[startup] ready in {STATE['startup_ms']} ms | "
            f"dim={emb.dim} | docs={s['documents']:,} chunks={s['chunks']:,} | "
            f"indexes={s['strategies']}"
        )
    except Exception as exc:  # keep the server up so /health can explain why
        STATE["error"] = str(exc)
        print(f"[startup] FAILED: {exc}")
    yield


app = FastAPI(
    title="Voice RAG API",
    version="0.1.0",
    description="Voice-enabled RAG over MSMARCO-XI",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_retrieve.router, prefix="/api/v1", tags=["retrieval"])
app.include_router(routes_query.router, prefix="/api/v1", tags=["query"])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if STATE["ready"] else "degraded",
        "ready": STATE["ready"],
        "error": STATE["error"],
        "startup_ms": STATE["startup_ms"],
    }


@app.get("/api/v1/stats")
def stats() -> dict:
    if not STATE["ready"]:
        return {"ready": False, "error": STATE["error"]}
    return get_retriever().stats()


@app.get("/api/v1/metrics")
def metrics() -> dict:
    """Live latency distribution over every request this process has served.

    Complements scripts/benchmark.py rather than replacing it: the benchmark is
    a controlled run over held-out queries and is the number to cite; this is
    what the server has actually experienced, including real user queries no
    curated set contains. Both use the same nearest-rank percentile so they are
    directly comparable.
    """
    return latency.summary()


@app.post("/api/v1/metrics/reset")
def metrics_reset() -> dict:
    return {"cleared": latency.reset()}


@app.get("/api/v1/config")
def config() -> dict:
    """Tunables surfaced so the frontend and README can show what was used."""
    return {
        "embed_model": settings.embed_model_id,
        "embed_dim": settings.embed_dim,
        "index_type": settings.index_type,
        "search_top_k": settings.search_top_k,
        "rrf_k": settings.rrf_k,
        "fusion_weights": settings.fusion_weights,
        "chunking": {
            "fixed": {
                "size": settings.fixed_chunk_tokens,
                "overlap": settings.fixed_overlap_tokens,
            },
            "sentence": {"max_tokens": settings.sentence_max_tokens},
            "semantic": {
                "percentile": settings.semantic_percentile,
                "min_tokens": settings.semantic_min_tokens,
                "max_tokens": settings.semantic_max_tokens,
            },
        },
        "thresholds": {
            "tau_topic": settings.tau_topic,
            "tau_conf": settings.tau_conf,
            "tau_flatness": settings.tau_flatness,
        },
    }
