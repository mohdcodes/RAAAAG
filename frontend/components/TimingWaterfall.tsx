"use client";

/**
 * Per-stage latency waterfall.
 *
 * The visual separation matters as much as the numbers: stages inside the
 * <200ms retrieval budget are drawn solid, and stages excluded from it (the
 * third-party LLM call, TTS) are drawn hatched and grouped under a separate
 * total. The claim being made is precise, so the chart has to be precise about
 * what it is claiming.
 */

import type { TimingBreakdown } from "@/lib/types";

const STAGE_LABELS: Record<string, string> = {
  "guardrail:input": "Input guardrails",
  "embed:query": "Query embedding",
  "search:vector": "Vector search",
  "rerank:cross_encoder": "Cross-encoder rerank",
  "guardrail:confidence": "Confidence gate",
  "generate:llm": "LLM generation",
  "guardrail:grounding": "Grounding check",
  "stt:sarvam": "Speech-to-text",
};

const STAGE_COLORS: Record<string, string> = {
  "guardrail:input": "var(--purple)",
  "embed:query": "var(--cyan)",
  "search:vector": "var(--accent)",
  "rerank:cross_encoder": "var(--yellow)",
  "guardrail:confidence": "var(--purple)",
  "generate:llm": "var(--text-dim)",
  "guardrail:grounding": "var(--purple)",
  "stt:sarvam": "var(--text-dim)",
};

export function TimingWaterfall({ timing }: { timing: TimingBreakdown }) {
  if (!timing.stages.length) return null;

  const counted = timing.stages.filter((s) => s.counted_in_retrieval_budget);
  const excluded = timing.stages.filter((s) => !s.counted_in_retrieval_budget);

  const retrievalMs = counted.reduce((sum, s) => sum + s.duration_ms, 0);
  const totalMs = timing.stages.reduce((sum, s) => sum + s.duration_ms, 0);
  const scaleMax = Math.max(...timing.stages.map((s) => s.duration_ms), 1);
  const withinBudget = retrievalMs < 200;

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
          Latency breakdown
        </span>
        <div className="flex items-center gap-3 text-[11px]">
          <span
            className="tabular font-medium"
            style={{ color: withinBudget ? "var(--green)" : "var(--red)" }}
            title="Sum of stages inside the retrieval budget"
          >
            retrieval {retrievalMs.toFixed(1)}ms
          </span>
          <span className="tabular text-[var(--text-dim)]">
            total {totalMs.toFixed(0)}ms
          </span>
        </div>
      </div>

      <div className="space-y-1.5">
        {timing.stages.map((stage, index) => {
          const widthPct = Math.max(1.5, (stage.duration_ms / scaleMax) * 100);
          const color = STAGE_COLORS[stage.stage] ?? "var(--text-dim)";
          const isCounted = stage.counted_in_retrieval_budget;

          return (
            <div key={`${stage.stage}-${index}`} className="group flex items-center gap-2">
              <span
                className="w-[132px] shrink-0 truncate text-[11px]"
                style={{ color: isCounted ? "var(--text-muted)" : "var(--text-dim)" }}
                title={stage.stage}
              >
                {STAGE_LABELS[stage.stage] ?? stage.stage}
              </span>

              <div className="relative h-4 flex-1 overflow-hidden rounded-sm bg-[var(--bg-input)]">
                <div
                  className="h-full rounded-sm transition-all duration-300"
                  style={{
                    width: `${widthPct}%`,
                    background: isCounted
                      ? color
                      : // Hatched fill marks "measured but not claimed".
                        `repeating-linear-gradient(45deg, ${color}, ${color} 3px, transparent 3px, transparent 6px)`,
                    border: isCounted ? "none" : `1px solid ${color}`,
                    opacity: isCounted ? 0.85 : 0.55,
                  }}
                />
              </div>

              <span
                className="tabular w-[62px] shrink-0 text-right text-[11px]"
                style={{ color: isCounted ? "var(--text)" : "var(--text-dim)" }}
              >
                {stage.duration_ms.toFixed(1)}ms
              </span>
            </div>
          );
        })}
      </div>

      {excluded.length > 0 && (
        <p className="border-t border-[var(--border)] pt-2 text-[10.5px] leading-relaxed text-[var(--text-dim)]">
          <span className="inline-block h-2 w-4 translate-y-[1px] rounded-sm border border-[var(--text-dim)] align-middle" />{" "}
          Hatched stages are third-party network calls (LLM generation, speech
          services) measured and reported but excluded from the sub-200ms
          retrieval budget.
        </p>
      )}
    </div>
  );
}
