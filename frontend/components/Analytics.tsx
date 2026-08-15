"use client";

/**
 * Latency analytics — the P50/P70/P100 figures the brief asks for, measured
 * live across every query run in this session rather than from a single
 * best-case run.
 */

import { useCallback, useEffect, useState } from "react";
import * as api from "@/lib/api";
import type { LatencySummary } from "@/lib/types";

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

const EXCLUDED_STAGES = new Set([
  "generate:llm",
  "guardrail:grounding",
  "stt:sarvam",
]);

export function Analytics() {
  const [summary, setSummary] = useState<LatencySummary | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setSummary(await api.latencyAnalytics());
    } catch {
      setSummary(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  const budget = summary?.budget_compliance;
  const hasSamples = (summary?.samples ?? 0) > 0;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-[var(--border)] px-5 py-3">
        <h2 className="text-[14px] font-medium text-[var(--text)]">Latency analytics</h2>
        <span className="tabular text-[11.5px] text-[var(--text-dim)]">
          {summary?.samples ?? 0} queries
        </span>
        <span className="flex-1" />
        <button
          type="button"
          onClick={() => void api.resetAnalytics().then(refresh)}
          className="rounded-[var(--radius)] border border-[var(--border)] px-2 py-1 text-[11px] text-[var(--text-dim)] hover:text-[var(--text-muted)]"
        >
          reset
        </button>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
        {!hasSamples ? (
          <p className="py-10 text-center text-[12.5px] text-[var(--text-dim)]">
            {loading ? "Loading…" : "Run some queries to collect latency data."}
          </p>
        ) : (
          <>
            {budget && (
              <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-raised)] p-4">
                <div className="flex items-baseline gap-3">
                  <span
                    className="tabular text-[26px] font-medium leading-none"
                    style={{
                      color:
                        budget.percentage >= 95
                          ? "var(--green)"
                          : budget.percentage >= 80
                            ? "var(--yellow)"
                            : "var(--red)",
                    }}
                  >
                    {budget.percentage.toFixed(1)}%
                  </span>
                  <div>
                    <p className="text-[12.5px] text-[var(--text)]">
                      of queries under {budget.threshold_ms}ms
                    </p>
                    <p className="tabular text-[11px] text-[var(--text-dim)]">
                      {budget.within_budget} / {budget.total} · {budget.measures}
                    </p>
                  </div>
                </div>
              </div>
            )}

            <PercentileTable
              title="Retrieval pipeline (the <200ms claim)"
              rows={[summary!.retrieval]}
              highlight
            />

            <PercentileTable
              title="Per-stage breakdown"
              rows={summary!.stages}
              labels={STAGE_LABELS}
            />

            <PercentileTable
              title="End to end (includes third-party calls)"
              rows={[summary!.total]}
            />

            <div className="grid grid-cols-2 gap-3">
              <CountBox title="Outcomes" counts={summary!.status_counts} />
              <CountBox title="Languages" counts={summary!.languages ?? {}} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function PercentileTable({
  title,
  rows,
  labels,
  highlight = false,
}: {
  title: string;
  rows: LatencySummary["stages"];
  labels?: Record<string, string>;
  highlight?: boolean;
}) {
  if (!rows.length) return null;

  return (
    <div className="space-y-1.5">
      <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
        {title}
      </span>
      <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)]">
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-dim)]">
              <th className="px-2.5 py-1.5 text-left font-medium">stage</th>
              {["P50", "P70", "P90", "P95", "P100", "mean"].map((header) => (
                <th key={header} className="px-2 py-1.5 text-right font-medium">
                  {header}
                </th>
              ))}
              <th className="px-2.5 py-1.5 text-right font-medium">n</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const excluded = EXCLUDED_STAGES.has(row.stage);
              return (
                <tr
                  key={row.stage}
                  className="border-b border-[var(--border)] last:border-0"
                >
                  <td
                    className="px-2.5 py-1.5"
                    style={{
                      color: excluded ? "var(--text-dim)" : "var(--text-muted)",
                    }}
                  >
                    {labels?.[row.stage] ?? row.stage}
                    {excluded && (
                      <span
                        className="ml-1.5 text-[9px] uppercase"
                        title="Third-party network call, excluded from the retrieval budget"
                      >
                        ext
                      </span>
                    )}
                  </td>
                  {[row.p50, row.p70, row.p90, row.p95, row.p100, row.mean].map(
                    (value, index) => (
                      <td
                        key={index}
                        className="tabular px-2 py-1.5 text-right"
                        style={{
                          color:
                            highlight && value < 200
                              ? "var(--green)"
                              : excluded
                                ? "var(--text-dim)"
                                : "var(--text)",
                        }}
                      >
                        {value.toFixed(1)}
                      </td>
                    ),
                  )}
                  <td className="tabular px-2.5 py-1.5 text-right text-[var(--text-dim)]">
                    {row.samples}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CountBox({
  title,
  counts,
}: {
  title: string;
  counts: Record<string, number>;
}) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;

  return (
    <div className="space-y-1.5">
      <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
        {title}
      </span>
      <div className="space-y-0.5 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-2.5 py-2">
        {entries.map(([key, count]) => (
          <div key={key} className="flex items-center justify-between text-[11.5px]">
            <span className="truncate text-[var(--text-muted)]">
              {key.replace(/_/g, " ")}
            </span>
            <span className="tabular text-[var(--text-dim)]">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
