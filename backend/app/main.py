"""FastAPI application.

Rate limiting is not optional here: the deployment target is a public URL that
judges will visit, and every query can trigger paid API calls to Sarvam, Groq
and Gemini. Without per-IP limits and a global daily cap, a shared link could
drain the account.
"""

from __future__ import annotations

import os

# FAISS and PyTorch each bundle their own OpenMP runtime. On Windows, loading
# both aborts the process ("multiple copies of the OpenMP runtime"). This must
# be set before either library is imported, hence its position above every
# other import. Threads are pinned to one OpenMP runtime below to avoid the
# oversubscription this flag would otherwise permit.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) - 1)))

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger, new_request_id

settings = get_settings()
configure_logging(settings.log_level, json_output=settings.environment == "production")
logger = get_logger(__name__)


class GlobalUsageCap:
    """Daily request ceiling across all clients.

    Per-IP limits alone do not protect against a link circulating widely, so
    this backstops them with an absolute daily budget.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0
        self.window_start = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        if now - self.window_start >= 86_400:
            self.count = 0
            self.window_start = now
        if self.count >= self.limit:
            return False
        self.count += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.count)


usage_cap = GlobalUsageCap(settings.global_daily_cap)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "starting",
        environment=settings.environment,
        embedding_model=settings.embedding_model,
        binary_quantization=settings.binary_quantization,
        providers=settings.available_providers() or ["none configured"],
        stt=settings.has_stt,
    )
    if not settings.has_generation:
        logger.warning("no_generation_keys — answers will degrade to extractive")
    if not settings.has_stt:
        logger.warning("no_sarvam_key — voice input and playback disabled")
    yield
    logger.info("shutting_down")


app = FastAPI(
    title="Voice RAG — MSMARCO-XI",
    description=(
        "Voice-enabled cross-lingual retrieval-augmented generation over "
        "ai4bharat/MSMARCO-XI, with per-stage latency instrumentation and "
        "four-layer guardrails."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    # The frontend reads TTS metadata from this header; without exposing it,
    # CORS strips it silently.
    expose_headers=["X-TTS-Metadata", "X-Request-ID", "X-Process-Time"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request ID, time the request, and enforce the global cap."""
    request_id = new_request_id()
    started = time.perf_counter()

    # Health and docs must stay reachable even when the cap is exhausted,
    # otherwise the deployment looks dead rather than rate-limited.
    exempt = request.url.path in ("/api/health", "/docs", "/openapi.json", "/")
    if settings.rate_limit_enabled and not exempt and request.method == "POST":
        if not usage_cap.allow():
            logger.warning("global_cap_exceeded", path=request.url.path)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "The daily request cap for this demo deployment has been "
                        "reached. It resets in 24 hours."
                    ),
                    "request_id": request_id,
                },
            )

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Return structured JSON for unhandled errors.

    The frontend renders failures inline in the chat, so an HTML error page
    would break the UI rather than inform it.
    """
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred.",
            "error": str(exc)[:200],
            "path": request.url.path,
        },
    )


app.include_router(router)


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "service": "Voice RAG — MSMARCO-XI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "dataset": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI",
        "requests_remaining_today": usage_cap.remaining,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
    )
