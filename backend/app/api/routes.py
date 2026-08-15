"""API routes.

Endpoint groups:

    /api/query        text -> grounded answer
    /api/voice/*      transcribe, synthesize
    /api/dataset/*    corpus preview and stats
    /api/analytics/*  latency percentiles
    /api/health       component status
    /api/strategies   chunking registry + benchmark results

Errors return structured JSON rather than raising, because the frontend renders
failures inline in the chat rather than as toast notifications — a failed query
should still show its timing and guardrail state.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.api.analytics import get_latency_store
from app.core.config import get_settings
from app.core.languages import ALL_CODES, LANGUAGES, STT_SUPPORTED_CODES
from app.core.logging import get_logger
from app.core.schemas import AnswerResponse, QueryRequest, TTSRequest
from app.harness.pipeline import get_pipeline
from app.harness.voice import VoiceError, VoiceUnavailable, get_voice
from app.ingest.chunking import strategy_info

logger = get_logger(__name__)
router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


@router.post("/query", response_model=AnswerResponse)
async def query(request: QueryRequest) -> AnswerResponse:
    """Run the full RAG pipeline on a text query."""
    pipeline = get_pipeline()
    response = await pipeline.answer(request)

    get_latency_store().record(
        response.timing,
        status=response.status,
        language=response.detected_language,
        provider=response.provider_used,
        query_chars=len(request.text),
    )
    return response


# --------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------


@router.post("/voice/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> dict[str, Any]:
    """Transcribe uploaded audio via Sarvam."""
    settings = get_settings()
    audio = await file.read()

    if len(audio) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds {settings.max_audio_bytes // (1024 * 1024)} MB limit.",
        )

    try:
        result = await get_voice().transcribe(
            audio,
            language=language,
            filename=file.filename or "audio.webm",
            content_type=file.content_type or "audio/webm",
        )
    except VoiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return result.model_dump()


@router.post("/voice/ask")
async def voice_ask(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    scope: str = Form(default="all"),
    top_k: int = Form(default=5),
) -> dict[str, Any]:
    """Transcribe then answer in one round trip.

    Saves a network round trip versus calling /transcribe and /query
    separately, which matters on mobile connections.
    """
    settings = get_settings()
    audio = await file.read()
    if len(audio) > settings.max_audio_bytes:
        raise HTTPException(status_code=413, detail="Audio too large.")

    try:
        transcription = await get_voice().transcribe(
            audio,
            language=language,
            filename=file.filename or "audio.webm",
            content_type=file.content_type or "audio/webm",
        )
    except VoiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not transcription.text.strip():
        raise HTTPException(
            status_code=422, detail="No speech detected in the audio."
        )

    response = await get_pipeline().answer(
        QueryRequest(
            text=transcription.text,
            language=transcription.language_code,
            scope=scope,  # type: ignore[arg-type]
            top_k=top_k,
        )
    )
    # STT is a third-party network call, so it is reported but excluded from
    # the retrieval budget.
    response.timing.add("stt:sarvam", transcription.duration_ms, counted=False)

    get_latency_store().record(
        response.timing,
        status=response.status,
        language=response.detected_language,
        provider=response.provider_used,
        query_chars=len(transcription.text),
    )

    return {"transcription": transcription.model_dump(), "response": response.model_dump()}


@router.post("/voice/speak")
async def speak(request: TTSRequest) -> Response:
    """Synthesize speech for an answer — the click-to-hear feature."""
    try:
        audio, metadata = await get_voice().synthesize(
            request.text, language=request.language, speaker=request.speaker
        )
    except VoiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-TTS-Metadata": json.dumps(metadata),
            "Cache-Control": "no-store",
        },
    )


# --------------------------------------------------------------------------
# Dataset preview
# --------------------------------------------------------------------------


@router.get("/dataset/preview")
async def dataset_preview(
    limit: int = 25,
    offset: str | None = None,
    language: str | None = None,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Page through indexed chunks — backs the UI's dataset browser."""
    store = get_pipeline().vector_store
    limit = max(1, min(limit, 100))

    try:
        rows, next_offset = store.scroll(
            limit=limit,
            offset=offset,
            languages=[language] if language else None,
            strategy=strategy,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        raise HTTPException(
            status_code=503, detail=f"Vector store unavailable: {exc}"
        ) from exc

    return {
        "rows": rows,
        "next_offset": str(next_offset) if next_offset is not None else None,
        "count": len(rows),
        "filters": {"language": language, "strategy": strategy},
    }


@router.get("/dataset/stats")
async def dataset_stats() -> dict[str, Any]:
    """Corpus statistics and dataset provenance."""
    store = get_pipeline().vector_store
    settings = get_settings()

    try:
        info = store.collection_info()
    except Exception as exc:  # noqa: BLE001
        info = {"exists": False, "error": str(exc)}

    return {
        "collection": info,
        "dataset": {
            "id": settings.hf_dataset_id,
            "url": f"https://huggingface.co/datasets/{settings.hf_dataset_id}",
            "split": settings.hf_split,
            "max_queries_per_language": settings.max_queries_per_language,
            # Verified against the HF datasets-server, not the README, which is
            # stale about both file format and config layout.
            "total_rows_upstream": 11_451_314,
            "upstream_size_bytes": 55_619_599_557,
            "note": (
                "Passages are nested per query with binary is_selected labels; "
                "the dataset ships no passage IDs, so doc IDs are synthesized "
                "as content hashes and duplicates are merged."
            ),
        },
        "languages": [
            {
                "code": lang.code,
                "name": lang.name,
                "native_name": lang.native_name,
                "script": lang.script,
                "flores": lang.flores,
                "stt_supported": bool(lang.sarvam_locale),
                "has_train_split": lang.has_train,
            }
            for lang in LANGUAGES
        ],
    }


@router.get("/languages")
async def languages() -> dict[str, Any]:
    return {
        "languages": [
            {
                "code": lang.code,
                "name": lang.name,
                "native_name": lang.native_name,
                "script": lang.script,
                "stt_supported": bool(lang.sarvam_locale),
            }
            for lang in LANGUAGES
        ],
        "all_codes": list(ALL_CODES),
        "stt_supported_codes": list(STT_SUPPORTED_CODES),
    }


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


@router.get("/strategies")
async def strategies() -> dict[str, Any]:
    """Chunking registry plus benchmark results when they exist."""
    settings = get_settings()
    benchmark_path = settings.runs_dir / "chunking_benchmark.json"
    benchmark = None
    if benchmark_path.exists():
        try:
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            benchmark = {"error": "benchmark file is not valid JSON"}

    return {
        "strategies": strategy_info(),
        "active": settings.chunking_strategy,
        "benchmark": benchmark,
    }


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------


@router.get("/analytics/latency")
async def latency_analytics() -> dict[str, Any]:
    return get_latency_store().summary()


@router.get("/analytics/recent")
async def recent_queries(limit: int = 50) -> dict[str, Any]:
    return {"queries": get_latency_store().recent(max(1, min(limit, 200)))}


@router.post("/analytics/reset")
async def reset_analytics() -> dict[str, str]:
    get_latency_store().clear()
    return {"status": "cleared"}


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


@router.get("/health")
async def health() -> JSONResponse:
    """Component-level health.

    Returns 200 even when components are down, with per-component detail —
    a monitoring endpoint that fails to respond tells you nothing.
    """
    settings = get_settings()
    pipeline = get_pipeline()

    try:
        vector_health = pipeline.vector_store.health()
    except Exception as exc:  # noqa: BLE001
        vector_health = {"reachable": False, "error": str(exc)}

    voice_health = get_voice().health()
    generation_health = pipeline.generator.health()

    ready = bool(vector_health.get("target_collection_present")) and generation_health[
        "any_available"
    ]

    return JSONResponse(
        {
            "status": "ready" if ready else "degraded",
            "environment": settings.environment,
            "components": {
                "vector_store": vector_health,
                "generation": generation_health,
                "voice": voice_health,
                "guardrails": {
                    "enabled": settings.guardrails_enabled,
                    "confidence_threshold": settings.confidence_threshold,
                },
            },
            "config": {
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "binary_quantization": settings.binary_quantization,
                "chunking_strategy": settings.chunking_strategy,
                "retrieval_top_k": settings.retrieval_top_k,
            },
        }
    )
