"use client";

/**
 * One chat turn.
 *
 * Refusals are rendered as first-class outcomes with their own colour and
 * explanation, not as errors — a system declining to answer because retrieval
 * found nothing relevant is working correctly, and the UI should say so.
 */

import { useEffect, useRef, useState } from "react";
import type { AnswerStatus, ChatMessage } from "@/lib/types";
import { speak } from "@/lib/api";
import { GuardrailPanel } from "./GuardrailPanel";
import { RetrievedChunks } from "./RetrievedChunks";
import { TimingWaterfall } from "./TimingWaterfall";

const STATUS_META: Record<
  AnswerStatus,
  { label: string; color: string; description: string }
> = {
  answered: { label: "Answered", color: "var(--green)", description: "" },
  refused_unsafe: {
    label: "Blocked — input safety",
    color: "var(--red)",
    description: "The query was blocked before retrieval.",
  },
  refused_off_topic: {
    label: "Out of scope",
    color: "var(--yellow)",
    description: "This system answers factual questions from the indexed corpus.",
  },
  refused_low_confidence: {
    label: "Declined — no relevant context",
    color: "var(--yellow)",
    description: "Retrieval scored below the confidence threshold, so no answer was generated.",
  },
  refused_ungrounded: {
    label: "Withheld — ungrounded",
    color: "var(--red)",
    description: "The draft answer could not be traced back to the retrieved passages.",
  },
  degraded_extractive: {
    label: "Degraded — extractive only",
    color: "var(--yellow)",
    description: "All generation providers failed; showing the top retrieved passage.",
  },
  error: { label: "Error", color: "var(--red)", description: "" },
};

export function Message({ message }: { message: ChatMessage }) {
  if (message.role === "user") return <UserMessage message={message} />;
  return <AssistantMessage message={message} />;
}

function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="animate-slide-up flex justify-end">
      <div className="max-w-[75%] rounded-[var(--radius-lg)] rounded-br-sm border border-[var(--accent-dim)] bg-[#f0883e14] px-3.5 py-2">
        <p className="text-[13.5px] leading-relaxed text-[var(--text)] indic">
          {message.text}
        </p>
        {message.transcription && (
          <div className="mt-1.5 flex items-center gap-2 border-t border-[var(--accent-dim)] pt-1.5 text-[10.5px] text-[var(--text-dim)]">
            <MicBadge />
            <span>
              transcribed · {message.transcription.language_code} ·{" "}
              <span className="tabular">
                {message.transcription.duration_ms.toFixed(0)}ms
              </span>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function AssistantMessage({ message }: { message: ChatMessage }) {
  const [showDetail, setShowDetail] = useState(false);
  const [audioState, setAudioState] = useState<"idle" | "loading" | "playing">("idle");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null);

  // Revoke the object URL on unmount; leaking them holds the audio in memory.
  useEffect(
    () => () => {
      audioRef.current?.pause();
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  if (message.isLoading) return <LoadingMessage />;

  if (message.error) {
    return (
      <div className="animate-slide-up rounded-[var(--radius-lg)] border border-[#f8514944] bg-[#f8514911] px-3.5 py-2.5">
        <p className="text-[12.5px] text-[var(--red)]">{message.error}</p>
      </div>
    );
  }

  const response = message.response;
  if (!response) return null;

  const meta = STATUS_META[response.status];
  const isRefusal = response.status.startsWith("refused_");

  const handleSpeak = async () => {
    if (audioState === "playing") {
      audioRef.current?.pause();
      setAudioState("idle");
      return;
    }
    setAudioState("loading");
    try {
      const { url } = await speak(response.answer, response.detected_language);
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      urlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => setAudioState("idle");
      audio.onerror = () => setAudioState("idle");
      await audio.play();
      setAudioState("playing");
    } catch {
      setAudioState("idle");
    }
  };

  return (
    <div className="animate-slide-up space-y-2.5">
      <div
        className="rounded-[var(--radius-lg)] rounded-bl-sm border bg-[var(--bg-raised)] px-3.5 py-3"
        style={{ borderColor: isRefusal ? `${meta.color}44` : "var(--border)" }}
      >
        <div className="mb-2 flex items-center gap-2">
          <span
            className="rounded-sm px-1.5 py-0.5 text-[9.5px] font-medium uppercase tracking-wide"
            style={{ background: `${meta.color}1e`, color: meta.color }}
          >
            {meta.label}
          </span>
          {response.provider_used && (
            <span className="text-[10px] text-[var(--text-dim)]">
              via {response.provider_used}
            </span>
          )}
          <span className="flex-1" />
          <button
            type="button"
            onClick={handleSpeak}
            title="Listen to this answer"
            className="flex items-center gap-1 rounded-[var(--radius)] px-1.5 py-0.5 text-[10.5px] text-[var(--text-dim)] transition-colors hover:bg-[var(--bg-overlay)] hover:text-[var(--accent)]"
          >
            {audioState === "loading" ? (
              <span className="animate-spin-slow inline-block h-3 w-3 rounded-full border border-[var(--text-dim)] border-t-transparent" />
            ) : (
              <SpeakerIcon active={audioState === "playing"} />
            )}
            {audioState === "playing" ? "Stop" : "Listen"}
          </button>
        </div>

        <p className="whitespace-pre-wrap text-[13.5px] leading-relaxed text-[var(--text)] indic">
          {response.answer}
        </p>

        {response.citations.length > 0 && (
          <div className="mt-2.5 flex flex-wrap gap-1 border-t border-[var(--border)] pt-2">
            {response.citations.map((citation) => (
              <span
                key={citation.chunk_id}
                title={citation.text}
                className="tabular cursor-help rounded-sm border border-[var(--accent-dim)] bg-[#f0883e14] px-1.5 py-0.5 text-[10px] text-[var(--accent)]"
              >
                [{citation.marker}] {citation.score.toFixed(2)}
              </span>
            ))}
          </div>
        )}

        {response.warnings.length > 0 && (
          <div className="mt-2 space-y-0.5">
            {response.warnings.map((warning, index) => (
              <p key={index} className="text-[10.5px] text-[var(--yellow)]">
                ⚠ {warning}
              </p>
            ))}
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={() => setShowDetail(!showDetail)}
        className="flex w-full items-center gap-2 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-2.5 py-1.5 text-[11px] text-[var(--text-dim)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-muted)]"
      >
        <span
          className="transition-transform duration-150"
          style={{ transform: showDetail ? "rotate(90deg)" : "none" }}
        >
          ▸
        </span>
        <span>Pipeline detail</span>
        <span className="flex-1" />
        <span
          className="tabular"
          style={{
            color: retrievalMs(response) < 200 ? "var(--green)" : "var(--yellow)",
          }}
        >
          {retrievalMs(response).toFixed(0)}ms retrieval
        </span>
        <span className="tabular text-[var(--text-dim)]">
          {response.retrieved.length} chunks
        </span>
      </button>

      {showDetail && (
        <div className="animate-slide-up space-y-4 rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-raised)] p-3">
          <TimingWaterfall timing={response.timing} />
          <GuardrailPanel
            guardrails={response.guardrails}
            claims={response.grounding_claims}
          />
          <RetrievedChunks
            chunks={response.retrieved}
            citedIds={response.citations.map((c) => c.chunk_id)}
          />
          {response.provider_attempts.length > 1 && (
            <ProviderAttempts attempts={response.provider_attempts} />
          )}
        </div>
      )}
    </div>
  );
}

function ProviderAttempts({ attempts }: { attempts: Array<Record<string, unknown>> }) {
  return (
    <div className="space-y-1">
      <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
        Provider failover
      </span>
      {attempts.map((attempt, index) => (
        <div
          key={index}
          className="flex items-center gap-2 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-2 py-1 text-[11px]"
        >
          <span className="text-[var(--text-muted)]">{String(attempt.provider)}</span>
          <span
            style={{
              color:
                attempt.status === "success" ? "var(--green)" : "var(--text-dim)",
            }}
          >
            {String(attempt.status)}
          </span>
          {attempt.error ? (
            <span className="flex-1 truncate text-[var(--text-dim)]">
              {String(attempt.error)}
            </span>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function LoadingMessage() {
  return (
    <div className="animate-slide-up space-y-2 rounded-[var(--radius-lg)] rounded-bl-sm border border-[var(--border)] bg-[var(--bg-raised)] px-3.5 py-3">
      <div className="shimmer h-3 w-3/4 rounded" />
      <div className="shimmer h-3 w-1/2 rounded" />
      <p className="pt-1 text-[10.5px] text-[var(--text-dim)]">
        retrieving · reranking · generating
      </p>
    </div>
  );
}

function retrievalMs(response: { timing: { stages: Array<{ duration_ms: number; counted_in_retrieval_budget: boolean }> } }) {
  return response.timing.stages
    .filter((s) => s.counted_in_retrieval_budget)
    .reduce((sum, s) => sum + s.duration_ms, 0);
}

function MicBadge() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2h2v2a5 5 0 0 0 10 0v-2h2z" />
    </svg>
  );
}

function SpeakerIcon({ active }: { active: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      {active && <path d="M15.5 8.5a5 5 0 0 1 0 7" />}
    </svg>
  );
}
