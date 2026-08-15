"""Central configuration.

Every tunable lives here so the pipeline can be re-pointed (different VM,
different index size, different provider) without touching logic. Values are
read from environment/.env; defaults are chosen so the system boots and
degrades gracefully when API keys are absent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- Environment ----------------
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # CORS: the Next.js origin(s) allowed to call this API.
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # ---------------- Provider keys ----------------
    # Absent keys are tolerated: STT/TTS/generation degrade rather than crash.
    sarvam_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    # ---------------- Speech ----------------
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_tts_url: str = "https://api.sarvam.ai/text-to-speech"
    sarvam_stt_model: str = "saarika:v2"
    sarvam_tts_model: str = "bulbul:v2"
    sarvam_tts_speaker: str = "anushka"
    stt_timeout_s: float = 15.0
    tts_timeout_s: float = 20.0
    max_audio_seconds: int = 30
    max_audio_bytes: int = 10 * 1024 * 1024  # 10 MB

    # ---------------- Generation ----------------
    # Ordered failover chain. First healthy provider wins.
    generation_provider_order: list[str] = Field(default=["groq", "gemini"])
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    generation_timeout_s: float = 30.0
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.2
    provider_max_retries: int = 2
    # Consecutive failures before a provider is tripped out of rotation.
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_s: float = 60.0

    # ---------------- Embeddings ----------------
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    embedding_max_length: int = 512
    use_onnx: bool = True
    # e5 models require these prefixes; omitting them measurably hurts recall.
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # ---------------- Reranking ----------------
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True
    rerank_candidates: int = 50
    rerank_batch_size: int = 16

    # ---------------- Qdrant ----------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "msmarco_xi"
    qdrant_timeout_s: float = 10.0
    # Binary quantization: 32x memory reduction (4KB -> 128B per 1024-dim vector).
    # Oversampling pulls extra binary candidates, then rescores against
    # full-precision vectors to recover recall lost to quantization.
    binary_quantization: bool = True
    quantization_oversampling: float = 3.0
    quantization_rescore: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    hnsw_ef_search: int = 128

    # ---------------- Retrieval ----------------
    retrieval_top_k: int = 10
    retrieval_candidates: int = 200  # pre-rerank pool
    hybrid_search: bool = True
    hybrid_alpha: float = 0.7  # weight on dense vs sparse
    # Refuse to answer below this reranked score — the primary hallucination
    # defense. Tuned against the eval set, not guessed.
    confidence_threshold: float = 0.35
    min_context_passages: int = 1

    # ---------------- Ingest ----------------
    hf_dataset_id: str = "ai4bharat/MSMARCO-XI"
    hf_split: str = "validation"
    # Per-language query cap. The main lever on index size and ingest time;
    # raise once VM RAM is known.
    max_queries_per_language: int = 20_000
    ingest_batch_size: int = 512
    chunking_strategy: str = "parent_child"

    # ---------------- Paths ----------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    runs_dir: Path = PROJECT_ROOT / "data" / "runs"
    model_cache_dir: Path = PROJECT_ROOT / "data" / "models"

    # ---------------- Rate limiting ----------------
    # A public URL must not be able to drain API credits.
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 20
    rate_limit_per_day: int = 500
    global_daily_cap: int = 5_000

    # ---------------- Guardrails ----------------
    guardrails_enabled: bool = True
    input_safety_enabled: bool = True
    off_topic_check_enabled: bool = True
    grounding_check_enabled: bool = True
    max_query_chars: int = 1_000

    def ensure_dirs(self) -> None:
        """Create data directories. Safe to call repeatedly."""
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.index_dir,
            self.runs_dir,
            self.model_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def has_stt(self) -> bool:
        return bool(self.sarvam_api_key)

    @property
    def has_generation(self) -> bool:
        return bool(self.groq_api_key or self.gemini_api_key)

    def available_providers(self) -> list[str]:
        """Configured generation providers, in failover order."""
        keys = {"groq": self.groq_api_key, "gemini": self.gemini_api_key}
        return [p for p in self.generation_provider_order if keys.get(p)]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
