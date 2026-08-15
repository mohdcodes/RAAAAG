"use client";

/**
 * Chunking strategy registry and benchmark comparison.
 *
 * The brief asks for a chunking approach with real thought behind it. This tab
 * shows the strategies implemented and, once the benchmark has been run, their
 * measured Recall/MRR/nDCG side by side — so the choice is visibly a result
 * rather than an assertion.
 */

import { useEffect, useState } from "react";
import * as api from "@/lib/api";
import type { BenchmarkRow, StrategyInfo } from "@/lib/types";

interface BenchmarkFile {
  language?: string;
  passages?: number;
  eval_queries?: number;
  embedding_model?: string;
  winner?: string;
  caveat?: string;
  results?: BenchmarkRow[];
}

export function Strategies() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [active, setActive] = useState<string>("");
  const [benchmark, setBenchmark] = useState<BenchmarkFile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .strategies()
      .then((data) => {
        setStrategies(data.strategies);
        setActive(data.active);
        setBenchmark(data.benchmark as BenchmarkFile | null);
      })
      .catch(() => undefined)
      .finally(() => setLoading(false));
  }, []);

  const results = benchmark?.results ?? [];

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-[var(--border)] px-5 py-3">
        <h2 className="text-[14px] font-medium text-[var(--text)]">
          Chunking strategies
        </h2>
        <p className="mt-0.5 text-[11.5px] text-[var(--text-dim)]">
          {strategies.length} strategies implemented · active:{" "}
          <span className="text-[var(--accent)]">{active}</span>
        </p>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="shimmer h-10 rounded-[var(--radius)]" />
            ))}
          </div>
        ) : (
          <>
            {results.length > 0 ? (
              <BenchmarkTable benchmark={benchmark!} results={results} />
            ) : (
              <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-3 py-2.5">
                <p className="text-[12px] text-[var(--text-muted)]">
                  No benchmark results yet.
                </p>
                <p className="mt-1 text-[11px] text-[var(--text-dim)]">
                  Run{" "}
                  <code className="tabular text-[var(--accent)]">
                    python scripts/benchmark_chunking.py --language hi
                  </code>{" "}
                  to compare every strategy on the same labelled queries and
                  populate this table.
                </p>
              </div>
            )}

            <div className="space-y-1.5">
              <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
                Implemented
              </span>
              {strategies.map((strategy) => (
                <div
                  key={strategy.name}
                  className="rounded-[var(--radius)] border bg-[var(--bg-input)] px-3 py-2"
                  style={{
                    borderColor:
                      strategy.name === active
                        ? "var(--accent-dim)"
                        : "var(--border)",
                  }}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="tabular text-[12px] font-medium"
                      style={{
                        color:
                          strategy.name === active
                            ? "var(--accent)"
                            : "var(--text)",
                      }}
                    >
                      {strategy.name}
                    </span>
                    {strategy.name === active && (
                      <span className="rounded-sm bg-[var(--accent-dim)] px-1.5 py-0.5 text-[9px] uppercase text-[var(--accent)]">
                        active
                      </span>
                    )}
                    {!strategy.requires_own_embeddings && (
                      <span
                        className="rounded-sm bg-[var(--bg-overlay)] px-1.5 py-0.5 text-[9px] uppercase text-[var(--text-dim)]"
                        title="Reuses another strategy's embeddings — no extra forward pass"
                      >
                        shared vectors
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-[11.5px] leading-relaxed text-[var(--text-dim)]">
                    {strategy.description}
                  </p>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function BenchmarkTable({
  benchmark,
  results,
}: {
  benchmark: BenchmarkFile;
  results: BenchmarkRow[];
}) {
  const best = Math.max(...results.map((r) => r.metrics.recall_at_10));

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
          Benchmark results
        </span>
        <span className="tabular text-[10.5px] text-[var(--text-dim)]">
          {benchmark.eval_queries} queries · {benchmark.language} ·{" "}
          {benchmark.embedding_model?.split("/").pop()}
        </span>
      </div>

      <div className="overflow-x-auto rounded-[var(--radius)] border border-[var(--border)]">
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-dim)]">
              <th className="px-2.5 py-1.5 text-left font-medium">strategy</th>
              {["R@1", "R@5", "R@10", "MRR@10", "nDCG@10", "chunks", "P50 ms"].map(
                (header) => (
                  <th key={header} className="px-2 py-1.5 text-right font-medium">
                    {header}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {results.map((row) => {
              const isWinner = row.metrics.recall_at_10 === best;
              return (
                <tr
                  key={row.strategy}
                  className="border-b border-[var(--border)] last:border-0"
                  style={{ background: isWinner ? "#3fb95010" : undefined }}
                >
                  <td
                    className="tabular px-2.5 py-1.5"
                    style={{ color: isWinner ? "var(--green)" : "var(--text-muted)" }}
                  >
                    {row.strategy}
                  </td>
                  <td className="tabular px-2 py-1.5 text-right text-[var(--text-muted)]">
                    {row.metrics.recall_at_1.toFixed(3)}
                  </td>
                  <td className="tabular px-2 py-1.5 text-right text-[var(--text-muted)]">
                    {row.metrics.recall_at_5.toFixed(3)}
                  </td>
                  <td
                    className="tabular px-2 py-1.5 text-right font-medium"
                    style={{ color: isWinner ? "var(--green)" : "var(--text)" }}
                  >
                    {row.metrics.recall_at_10.toFixed(3)}
                  </td>
                  <td className="tabular px-2 py-1.5 text-right text-[var(--text-muted)]">
                    {row.metrics.mrr_at_10.toFixed(3)}
                  </td>
                  <td className="tabular px-2 py-1.5 text-right text-[var(--text-muted)]">
                    {row.metrics.ndcg_at_10.toFixed(3)}
                  </td>
                  <td className="tabular px-2 py-1.5 text-right text-[var(--text-dim)]">
                    {row.chunk_count.toLocaleString()}
                  </td>
                  <td className="tabular px-2 py-1.5 text-right text-[var(--text-dim)]">
                    {row.latency?.total?.p50?.toFixed(1) ?? "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {benchmark.caveat && (
        <p className="text-[10.5px] leading-relaxed text-[var(--text-dim)]">
          {benchmark.caveat}
        </p>
      )}
    </div>
  );
}
