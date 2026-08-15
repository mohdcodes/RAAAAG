/**
 * Types mirroring the backend Pydantic schemas in app/core/schemas.py.
 * Keep these in sync — they are the API contract.
 */

export type AnswerStatus =
  | "answered"
  | "refused_unsafe"
  | "refused_off_topic"
  | "refused_low_confidence"
  | "refused_ungrounded"
  | "degraded_extractive"
  | "error";

export type GuardrailVerdict = "pass" | "warn" | "block";

export type GuardrailStage =
  | "input_safety"
  | "off_topic"
  | "retrieval_confidence"
  | "grounding";

export type RetrievalScope = "all" | "same" | "english";

export interface ChunkMetadata {
  doc_hash: string;
  language: string;
  flores_code: string;
  is_english: boolean;
  query_type: string;
  is_selected: boolean;
  source_query_ids: number[];
  strategy: string;
  parent_hash: string | null;
  char_count: number;
  token_estimate: number;
  position: number;
}

export interface Chunk {
  chunk_id: string;
  text: string;
  context_text: string | null;
  metadata: ChunkMetadata;
}

export interface RetrievedChunk {
  chunk: Chunk;
  dense_score: number;
  sparse_score: number;
  fused_score: number;
  rerank_score: number | null;
  rank: number;
}

export interface GuardrailResult {
  stage: GuardrailStage;
  verdict: GuardrailVerdict;
  score: number | null;
  threshold: number | null;
  reason: string;
  details: Record<string, unknown>;
  duration_ms: number;
}

export interface GroundingClaim {
  claim: string;
  supported: boolean;
  supporting_chunk_ids: string[];
  confidence: number;
}

export interface StageTiming {
  stage: string;
  duration_ms: number;
  /** False for stages excluded from the <200ms retrieval budget. */
  counted_in_retrieval_budget: boolean;
}

export interface TimingBreakdown {
  stages: StageTiming[];
}

export interface Citation {
  chunk_id: string;
  doc_hash: string;
  text: string;
  language: string;
  score: number;
  marker: number;
}

export interface AnswerResponse {
  query: string;
  detected_language: string;
  answer: string;
  status: AnswerStatus;
  citations: Citation[];
  retrieved: RetrievedChunk[];
  guardrails: GuardrailResult[];
  grounding_claims: GroundingClaim[];
  timing: TimingBreakdown;
  provider_used: string | null;
  provider_attempts: Array<Record<string, unknown>>;
  strategy_used: string | null;
  request_id: string;
  warnings: string[];
}

export interface TranscriptionResult {
  text: string;
  language_code: string;
  provider: string;
  duration_ms: number;
  confidence: number | null;
  audio_seconds: number | null;
}

export interface LatencyPercentiles {
  stage: string;
  p50: number;
  p70: number;
  p90: number;
  p95: number;
  p100: number;
  mean: number;
  samples: number;
}

export interface LatencySummary {
  samples: number;
  stages: LatencyPercentiles[];
  retrieval: LatencyPercentiles;
  total: LatencyPercentiles;
  budget_compliance: {
    threshold_ms: number;
    within_budget: number;
    total: number;
    percentage: number;
    measures: string;
  } | null;
  status_counts: Record<string, number>;
  languages?: Record<string, number>;
  providers?: Record<string, number>;
  note?: string;
}

export interface LanguageInfo {
  code: string;
  name: string;
  native_name: string;
  script: string;
  flores?: string;
  stt_supported: boolean;
  has_train_split?: boolean;
}

export interface DatasetStats {
  collection: {
    exists: boolean;
    points_count?: number;
    vectors_count?: number;
    status?: string;
    name?: string;
    binary_quantization?: boolean;
    dimension?: number;
    /** "faiss" | "qdrant" */
    backend?: string;
    error?: string;
  };
  dataset: {
    id: string;
    url: string;
    split: string;
    max_queries_per_language: number;
    total_rows_upstream: number;
    upstream_size_bytes: number;
    note: string;
  };
  languages: LanguageInfo[];
}

export interface DatasetPreviewRow {
  chunk_id?: string;
  text?: string;
  context_text?: string | null;
  doc_hash?: string;
  language?: string;
  query_type?: string;
  is_selected?: boolean;
  strategy?: string;
  char_count?: number;
  source_query_ids?: number[];
  [key: string]: unknown;
}

export interface StrategyInfo {
  name: string;
  description: string;
  requires_own_embeddings: boolean;
}

export interface BenchmarkRow {
  strategy: string;
  description: string;
  metrics: {
    recall_at_1: number;
    recall_at_5: number;
    recall_at_10: number;
    mrr_at_10: number;
    ndcg_at_10: number;
    queries_evaluated: number;
  };
  latency: Record<string, LatencyPercentiles>;
  chunk_count: number;
  avg_chunk_chars: number;
  expansion_ratio: number;
  embedding_seconds: number;
}

export interface HealthResponse {
  status: "ready" | "degraded";
  environment: string;
  components: {
    vector_store: Record<string, unknown>;
    generation: {
      order: string[];
      providers: Array<{
        provider: string;
        model: string;
        configured: boolean;
        circuit_state: string;
        available: boolean;
      }>;
      any_available: boolean;
    };
    voice: {
      provider: string;
      configured: boolean;
      stt_available: boolean;
      tts_available: boolean;
    };
    guardrails: { enabled: boolean; confidence_threshold: number };
  };
  config: Record<string, unknown>;
}

/** A single turn in the chat transcript. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  language?: string;
  response?: AnswerResponse;
  transcription?: TranscriptionResult;
  isLoading?: boolean;
  error?: string;
  timestamp: number;
}
