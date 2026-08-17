"""Application settings.

IMPORTANT: `OMP_NUM_THREADS` must be set before onnxruntime/faiss are imported
anywhere in the process. `app.main` does that at module top, before importing
anything that touches them. Changing that import order silently costs latency:
on this hybrid P/E-core CPU, letting OpenMP spawn a thread per core makes a
single 384-dim query matmul *slower*, because sync dominates and work lands on
E-cores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

API_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = API_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=API_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ---
    data_dir: Path = DATA_DIR
    raw_dir: Path = DATA_DIR / "raw"
    corpus_dir: Path = DATA_DIR / "corpus"
    index_dir: Path = DATA_DIR / "index"

    # --- embedding model ---
    # multilingual-e5-small: 384-dim, 12 layers, XLM-R tokenizer (vocab 250037),
    # mean pooling, requires asymmetric "query: " / "passage: " prefixes.
    embed_model_id: str = "intfloat/multilingual-e5-small"
    embed_onnx_file: str = "onnx/model_qint8_avx512_vnni.onnx"
    embed_dim: int = 384
    embed_max_tokens: int = 512
    embed_batch_size: int = 256
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    # --- threading (see module docstring) ---
    omp_threads: int = 4
    onnx_intra_op_threads: int = 4
    onnx_inter_op_threads: int = 1

    # --- index ---
    index_type: Literal["flat", "hnsw"] = "flat"
    search_top_k: int = 50           # per-strategy candidate depth before fusion
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    kmeans_centroids: int = 64       # powers the G3 off-topic gate

    fusion_weights: dict[str, float] = {
        "fixed": 1.0,
        "sentence": 1.0,
        "semantic": 1.0,
        "contextual": 1.2,
        "bm25": 0.8,
    }

    # --- chunking ---
    fixed_chunk_tokens: int = 256
    fixed_overlap_tokens: int = 64
    sentence_max_tokens: int = 200
    semantic_percentile: float = 95.0
    semantic_min_tokens: int = 64
    semantic_max_tokens: int = 384

    # --- guardrail thresholds (calibrate_thresholds.py overwrites these) ---
    # G3: e5 cosines are anisotropic and compress into ~[0.80, 0.95] even for
    # gibberish, so this gate has weak discriminative power. Measured on the
    # small index, in-corpus and off-topic distributions OVERLAP. Set low
    # (report, rarely refuse) until calibrate_thresholds.py runs against the
    # full corpus. G4 and G5 carry the real weight.
    tau_topic: float = 0.70
    tau_conf: float = 0.72           # G4 low retrieval confidence (e5 scale)

    # G4 flatness: rrf[0]/rrf[depth]. NOTE the bounded range — with rrf_k=60 a
    # document ranked 1st in every strategy vs one ranked 10th in every
    # strategy differs by only (60+10)/(60+1) = 1.147. An earlier value of 1.15
    # was therefore mathematically unreachable and refused 100% of queries.
    tau_flatness: float = 1.04
    flatness_depth: int = 10
    tau_ground: float = 0.55         # G5 per-sentence support cosine
    ground_refuse_below: float = 0.60
    ground_warn_below: float = 0.85

    # --- generation ---
    gemini_api_key: str = ""
    # Measured on this key, 2026-08: gemini-2.5-flash returns 429 (free-tier
    # quota exhausted) and gemini-2.5-flash-lite returns 404 ("no longer
    # available to new users") despite still appearing in models.list().
    # Latencies measured on an identical structured-output prompt:
    #   gemini-3.5-flash-lite   984 ms   <- primary
    #   gemini-3.1-flash-lite  1199 ms   <- fallback (different family)
    #   gemini-3.6-flash       3484 ms
    #   gemini-3.5-flash      13952 ms
    # The fallback is deliberately from a different model generation, so a
    # family-wide outage or quota block does not take out both.
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_model: str = "gemini-3.1-flash-lite"
    max_tool_hops: int = 1
    generation_timeout_ms: int = 15_000

    # --- voice ---
    elevenlabs_api_key: str = ""
    elevenlabs_stt_model: str = "scribe_v2"

    # --- server ---
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    log_level: str = "INFO"


settings = Settings()
