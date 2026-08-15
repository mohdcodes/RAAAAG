"use client";

/**
 * Metrics dashboard.
 *
 * Everything the brief asks to see measured: per-stage P50/P70/P90/P95/P100,
 * budget compliance, retrieval accuracy against the dataset's own relevance
 * labels, and index composition. Summary figures first, detail below.
 */

import { useCallback, useEffect, useState } from "react";
import * as api from "@/lib/api";
import type { DatasetStats, LatencySummary } from "@/lib/types";

const LABELS: Record<string, string> = {
  "guardrail:input": "guardrails in",
  "embed:query": "embed",
  "search:vector": "search",
  "rerank:cross_encoder": "rerank",
  "guardrail:confidence": "confidence",
  "generate:llm": "generate",
  "guardrail:grounding": "grounding",
  "stt:sarvam": "speech",
  retrieval_ms: "retrieval",
  total_ms: "end to end",
};

const EXTERNAL = new Set(["generate:llm", "guardrail:grounding", "stt:sarvam"]);

interface EvalReport {
  metrics?: {
    recall_at_1: number;
    recall_at_5: number;
    recall_at_10: number;
    mrr_at_10: number;
    ndcg_at_10: number;
    queries_evaluated: number;
  };
}

export function Metrics() {
  const [latency, setLatency] = useState<LatencySummary | null>(null);
  const [dataset, setDataset] = useState<DatasetStats | null>(null);
  const [evaluation, setEvaluation] = useState<EvalReport | null>(null);

  const refresh = useCallback(async () => {
    const [l, d] = await Promise.allSettled([
      api.latencyAnalytics(),
      api.datasetStats(),
    ]);
    if (l.status === "fulfilled") setLatency(l.value);
    if (d.status === "fulfilled") setDataset(d.value);

    try {
      const strategies = await api.strategies();
      const benchmark = strategies.benchmark as { results?: Array<EvalReport> } | null;
      if (benchmark?.results?.length) setEvaluation(benchmark.results[0]);
    } catch {
      /* benchmark not run yet */
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, [refresh]);

  const budget = latency?.budget_compliance;
  const samples = latency?.samples ?? 0;
  const accuracy = evaluation?.metrics;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6 px-5 py-6">
        {/* Headline figures */}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Big
            value={latency?.retrieval.p50?.toFixed(0) ?? "—"}
            unit="ms"
            label="retrieval p50"
            tone={
              (latency?.retrieval.p50 ?? 0) < 200 && samples > 0 ? "ok" : "plain"
            }
          />
          <Big
            value={budget ? budget.percentage.toFixed(0) : "—"}
            unit="%"
            label="under 200ms"
            tone={budget && budget.percentage >= 95 ? "ok" : "plain"}
          />
          <Big
            value={(dataset?.collection.points_count ?? 0).toLocaleString()}
            label="vectors"
          />
          <Big value={String(samples)} label="queries" />
        </div>

        {samples === 0 && (
          <p className="rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)] px-3 py-2.5 text-[12.5px] text-[var(--cream-faint)]">
            Run a few queries to populate these.
          </p>
        )}

        {/* Latency percentiles */}
        {samples > 0 && latency && (
          <Section title="Latency" note={`${samples} queries`}>
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="border-b border-[var(--line)] text-[var(--cream-faint)]">
                  <th className="py-1.5 pr-2 text-left font-normal">stage</th>
                  {["p50", "p70", "p90", "p95", "p100", "mean"].map((h) => (
                    <th key={h} className="px-2 py-1.5 text-right font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[latency.retrieval, ...latency.stages, latency.total].map((row, i) => {
                  const external = EXTERNAL.has(row.stage);
                  const isSummary =
                    row.stage === "retrieval_ms" || row.stage === "total_ms";
                  return (
                    <tr
                      key={`${row.stage}-${i}`}
                      className="border-b border-[var(--line)] last:border-0"
                      style={{ background: isSummary ? "var(--sunken)" : undefined }}
                    >
                      <td
                        className="py-1.5 pr-2"
                        style={{
                          color: external ? "var(--cream-faint)" : "var(--cream-soft)",
                          fontWeight: isSummary ? 500 : 400,
                        }}
                      >
                        {LABELS[row.stage] ?? row.stage}
                        {external && (
                          <span className="ml-1.5 text-[9.5px] uppercase">ext</span>
                        )}
                      </td>
                      {[row.p50, row.p70, row.p90, row.p95, row.p100, row.mean].map(
                        (v, j) => (
                          <td
                            key={j}
                            className="num px-2 py-1.5 text-right"
                            style={{
                              color:
                                row.stage === "retrieval_ms" && v < 200
                                  ? "var(--ok)"
                                  : external
                                    ? "var(--cream-faint)"
                                    : "var(--cream)",
                            }}
                          >
                            {v.toFixed(1)}
                          </td>
                        ),
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-2 text-[11px] text-[var(--cream-faint)]">
              ext = third-party network call, excluded from the retrieval budget
            </p>
          </Section>
        )}

        {/* Retrieval accuracy */}
        <Section
          title="Accuracy"
          note={accuracy ? `${accuracy.queries_evaluated} labelled queries` : undefined}
        >
          {accuracy ? (
            <div className="grid grid-cols-3 gap-3 md:grid-cols-5">
              <Metric label="recall@1" value={accuracy.recall_at_1} />
              <Metric label="recall@5" value={accuracy.recall_at_5} />
              <Metric label="recall@10" value={accuracy.recall_at_10} />
              <Metric label="mrr@10" value={accuracy.mrr_at_10} />
              <Metric label="ndcg@10" value={accuracy.ndcg_at_10} />
            </div>
          ) : (
            <p className="text-[12.5px] text-[var(--cream-faint)]">
              Run{" "}
              <code className="num text-[var(--gold)]">
                python scripts/evaluate.py
              </code>{" "}
              to measure against the dataset&rsquo;s relevance labels.
            </p>
          )}
        </Section>

        {/* Outcomes and index composition */}
        <div className="grid gap-4 md:grid-cols-2">
          {latency && Object.keys(latency.status_counts).length > 0 && (
            <Section title="Outcomes">
              <Bars counts={latency.status_counts} />
            </Section>
          )}
          {latency?.languages && Object.keys(latency.languages).length > 0 && (
            <Section title="Languages">
              <Bars counts={latency.languages} />
            </Section>
          )}
        </div>

        {dataset && (
          <Section title="Index">
            <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[12.5px] md:grid-cols-3">
              <Row k="backend" v={String(dataset.collection.backend ?? "faiss")} />
              <Row k="vectors" v={(dataset.collection.points_count ?? 0).toLocaleString()} />
              <Row k="dimensions" v={String(dataset.collection.dimension ?? "—")} />
              <Row
                k="quantization"
                v={dataset.collection.binary_quantization ? "binary (32×)" : "none"}
              />
              <Row k="dataset" v={dataset.dataset.id} />
              <Row
                k="upstream"
                v={`${(dataset.dataset.total_rows_upstream / 1e6).toFixed(1)}M rows`}
              />
            </dl>
          </Section>
        )}
      </div>
    </div>
  );
}

function Big({
  value,
  unit,
  label,
  tone = "plain",
}: {
  value: string;
  unit?: string;
  label: string;
  tone?: "ok" | "plain";
}) {
  return (
    <div className="rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)] px-3.5 py-3">
      <div className="flex items-baseline gap-1">
        <span
          className="num text-[26px] leading-none"
          style={{ color: tone === "ok" ? "var(--ok)" : "var(--cream)" }}
        >
          {value}
        </span>
        {unit && (
          <span className="num text-[13px] text-[var(--cream-faint)]">{unit}</span>
        )}
      </div>
      <div className="mt-1.5 text-[11px] text-[var(--cream-faint)]">{label}</div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-[var(--r-sm)] bg-[var(--sunken)] px-2.5 py-2">
      <div className="num text-[17px] text-[var(--cream)]">{value.toFixed(3)}</div>
      <div className="num text-[10px] text-[var(--cream-faint)]">{label}</div>
    </div>
  );
}

function Section({
  title,
  note,
  children,
}: {
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-2">
      <div className="flex items-baseline gap-2">
        <h3 className="text-[13px] font-medium text-[var(--cream)]">{title}</h3>
        {note && <span className="num text-[11px] text-[var(--cream-faint)]">{note}</span>}
      </div>
      <div className="overflow-x-auto rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)] p-3">
        {children}
      </div>
    </section>
  );
}

function Bars({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, n]) => n), 1);

  return (
    <div className="space-y-1.5">
      {entries.map(([key, n]) => (
        <div key={key} className="flex items-center gap-2 text-[12px]">
          <span className="w-28 shrink-0 truncate text-[var(--cream-soft)]">
            {key.replace(/_/g, " ")}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--sunken)]">
            <div
              className="h-full rounded-full bg-[var(--gold)]"
              style={{ width: `${(n / max) * 100}%` }}
            />
          </div>
          <span className="num w-8 shrink-0 text-right text-[var(--cream-faint)]">
            {n}
          </span>
        </div>
      ))}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-[var(--line)] py-1 last:border-0">
      <dt className="text-[var(--cream-faint)]">{k}</dt>
      <dd className="num truncate text-[var(--cream-soft)]">{v}</dd>
    </div>
  );
}
