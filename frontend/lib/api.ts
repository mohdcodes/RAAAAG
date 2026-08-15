/**
 * Backend API client.
 *
 * Every call returns typed data or throws an ApiError carrying the backend's
 * structured detail, so the chat UI can render failures inline with the same
 * treatment as answers rather than as opaque toasts.
 */

import type {
  AnswerResponse,
  DatasetPreviewRow,
  DatasetStats,
  HealthResponse,
  LanguageInfo,
  LatencySummary,
  RetrievalScope,
  StrategyInfo,
  TranscriptionResult,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData
          ? {}
          : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
    });
  } catch (cause) {
    // Network-level failure: the backend is unreachable, not erroring.
    throw new ApiError(
      `Cannot reach the API at ${API_BASE}. Is the backend running?`,
      0,
      String(cause),
    );
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? body.error ?? detail;
    } catch {
      // Response body was not JSON; keep the status text.
    }
    throw new ApiError(detail, response.status, detail);
  }

  return response.json() as Promise<T>;
}

export interface QueryOptions {
  language?: string | null;
  scope?: RetrievalScope;
  topK?: number;
  strategy?: string | null;
  includeDebug?: boolean;
}

export function query(
  text: string,
  options: QueryOptions = {},
): Promise<AnswerResponse> {
  return request<AnswerResponse>("/api/query", {
    method: "POST",
    body: JSON.stringify({
      text,
      language: options.language ?? null,
      scope: options.scope ?? "all",
      top_k: options.topK ?? 10,
      strategy: options.strategy ?? null,
      include_debug: options.includeDebug ?? true,
    }),
  });
}

export function transcribe(
  audio: Blob,
  language?: string | null,
): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append("file", audio, "recording.webm");
  if (language) form.append("language", language);
  return request<TranscriptionResult>("/api/voice/transcribe", {
    method: "POST",
    body: form,
  });
}

/** Transcribe and answer in one round trip — saves a hop on slow connections. */
export function voiceAsk(
  audio: Blob,
  options: { language?: string | null; scope?: RetrievalScope; topK?: number } = {},
): Promise<{ transcription: TranscriptionResult; response: AnswerResponse }> {
  const form = new FormData();
  form.append("file", audio, "recording.webm");
  if (options.language) form.append("language", options.language);
  form.append("scope", options.scope ?? "all");
  form.append("top_k", String(options.topK ?? 10));
  return request("/api/voice/ask", { method: "POST", body: form });
}

/** Synthesize speech; returns an object URL the caller must revoke. */
export async function speak(
  text: string,
  language = "hi",
): Promise<{ url: string; metadata: Record<string, unknown> }> {
  const response = await fetch(`${API_BASE}/api/voice/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language }),
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, response.status, detail);
  }

  const header = response.headers.get("X-TTS-Metadata");
  const blob = await response.blob();
  return {
    url: URL.createObjectURL(blob),
    metadata: header ? JSON.parse(header) : {},
  };
}

export function health(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function datasetStats(): Promise<DatasetStats> {
  return request<DatasetStats>("/api/dataset/stats");
}

export function datasetPreview(params: {
  limit?: number;
  offset?: string | null;
  language?: string | null;
  strategy?: string | null;
}): Promise<{
  rows: DatasetPreviewRow[];
  next_offset: string | null;
  count: number;
}> {
  const search = new URLSearchParams();
  search.set("limit", String(params.limit ?? 25));
  if (params.offset) search.set("offset", params.offset);
  if (params.language) search.set("language", params.language);
  if (params.strategy) search.set("strategy", params.strategy);
  return request(`/api/dataset/preview?${search}`);
}

export function languages(): Promise<{
  languages: LanguageInfo[];
  all_codes: string[];
  stt_supported_codes: string[];
}> {
  return request("/api/languages");
}

export function strategies(): Promise<{
  strategies: StrategyInfo[];
  active: string;
  benchmark: Record<string, unknown> | null;
}> {
  return request("/api/strategies");
}

export function latencyAnalytics(): Promise<LatencySummary> {
  return request<LatencySummary>("/api/analytics/latency");
}

export function resetAnalytics(): Promise<{ status: string }> {
  return request("/api/analytics/reset", { method: "POST" });
}
