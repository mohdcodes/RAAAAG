"use client";

/**
 * Retrieved context inspector.
 *
 * Shows exactly what the vector search returned and what was handed to the
 * LLM — including the distinction between the narrow text that was *embedded*
 * and the wider parent context that was *retrieved*, which is the whole point
 * of the small-to-big chunking strategies.
 */

import { useState } from "react";
import type { RetrievedChunk } from "@/lib/types";

export function RetrievedChunks({
  chunks,
  citedIds = [],
}: {
  chunks: RetrievedChunk[];
  citedIds?: string[];
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [showAll, setShowAll] = useState(false);

  if (!chunks.length) return null;

  const cited = new Set(citedIds);
  const visible = showAll ? chunks : chunks.slice(0, 5);

  const toggle = (id: string) =>
    setExpanded((current) => {
      const next = new Set(current);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
          Retrieved context
        </span>
        <span className="tabular text-[10.5px] text-[var(--text-dim)]">
          {chunks.length} chunks · {cited.size} cited
        </span>
      </div>

      <div className="space-y-1.5">
        {visible.map((item, index) => {
          const { chunk } = item;
          const isOpen = expanded.has(chunk.chunk_id);
          const isCited = cited.has(chunk.chunk_id);
          const score = item.rerank_score ?? item.fused_score;
          const hasWiderContext =
            chunk.context_text && chunk.context_text !== chunk.text;

          return (
            <div
              key={chunk.chunk_id}
              className="overflow-hidden rounded-[var(--radius)] border bg-[var(--bg-input)] transition-colors"
              style={{
                borderColor: isCited ? "var(--accent-dim)" : "var(--border)",
              }}
            >
              <button
                type="button"
                onClick={() => toggle(chunk.chunk_id)}
                className="flex w-full items-start gap-2 px-2.5 py-2 text-left hover:bg-[var(--bg-overlay)]"
              >
                <span
                  className="tabular mt-[1px] flex h-4 w-4 shrink-0 items-center justify-center rounded-sm text-[10px] font-bold"
                  style={{
                    background: isCited ? "var(--accent-dim)" : "var(--bg-overlay)",
                    color: isCited ? "var(--accent)" : "var(--text-dim)",
                  }}
                  title={isCited ? "Cited in the answer" : "Retrieved, not cited"}
                >
                  {index + 1}
                </span>

                <p
                  className={`flex-1 text-[12px] leading-relaxed text-[var(--text-muted)] ${
                    isOpen ? "" : "line-clamp-2"
                  } ${chunk.metadata.language !== "en" ? "indic" : ""}`}
                >
                  {chunk.text}
                </p>

                <div className="flex shrink-0 flex-col items-end gap-0.5">
                  <span
                    className="tabular text-[11px] font-medium"
                    style={{
                      color:
                        score > 0.7
                          ? "var(--green)"
                          : score > 0.4
                            ? "var(--yellow)"
                            : "var(--text-dim)",
                    }}
                  >
                    {score.toFixed(3)}
                  </span>
                  <span className="text-[9.5px] uppercase text-[var(--text-dim)]">
                    {chunk.metadata.language}
                  </span>
                </div>
              </button>

              {isOpen && (
                <div className="animate-slide-up space-y-2 border-t border-[var(--border)] px-2.5 py-2">
                  {hasWiderContext && (
                    <div>
                      <span className="text-[10px] uppercase tracking-wide text-[var(--text-dim)]">
                        Parent context sent to the model
                      </span>
                      <p
                        className={`mt-1 text-[11.5px] leading-relaxed text-[var(--text-muted)] ${
                          chunk.metadata.language !== "en" ? "indic" : ""
                        }`}
                      >
                        {chunk.context_text}
                      </p>
                    </div>
                  )}

                  <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[10.5px]">
                    <dt className="text-[var(--text-dim)]">strategy</dt>
                    <dd className="text-[var(--text-muted)]">{chunk.metadata.strategy}</dd>

                    <dt className="text-[var(--text-dim)]">doc hash</dt>
                    <dd className="tabular truncate text-[var(--text-muted)]">
                      {chunk.metadata.doc_hash}
                    </dd>

                    <dt className="text-[var(--text-dim)]">dense / rerank</dt>
                    <dd className="tabular text-[var(--text-muted)]">
                      {item.dense_score.toFixed(4)}
                      {item.rerank_score !== null &&
                        ` → ${item.rerank_score.toFixed(4)}`}
                    </dd>

                    <dt className="text-[var(--text-dim)]">chars</dt>
                    <dd className="tabular text-[var(--text-muted)]">
                      {chunk.metadata.char_count}
                    </dd>

                    {chunk.metadata.is_selected && (
                      <>
                        <dt className="text-[var(--text-dim)]">label</dt>
                        <dd style={{ color: "var(--green)" }}>
                          is_selected (ground truth)
                        </dd>
                      </>
                    )}
                  </dl>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {chunks.length > 5 && (
        <button
          type="button"
          onClick={() => setShowAll(!showAll)}
          className="w-full rounded-[var(--radius)] border border-[var(--border)] py-1 text-[11px] text-[var(--text-dim)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-muted)]"
        >
          {showAll ? "Show fewer" : `Show all ${chunks.length} chunks`}
        </button>
      )}
    </div>
  );
}
