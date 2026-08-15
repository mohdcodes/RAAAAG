"use client";

/**
 * Dataset browser — the simple preview the brief asks for.
 *
 * Pages through indexed chunks with language and strategy filters, and shows
 * the dataset's real provenance figures alongside so a reviewer can see what
 * fraction of the upstream corpus is actually indexed.
 */

import { useCallback, useEffect, useState } from "react";
import * as api from "@/lib/api";
import type { DatasetPreviewRow, DatasetStats } from "@/lib/types";

export function DatasetPreview() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [rows, setRows] = useState<DatasetPreviewRow[]>([]);
  const [language, setLanguage] = useState<string>("");
  const [offsets, setOffsets] = useState<(string | null)[]>([null]);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.datasetStats().then(setStats).catch(() => undefined);
  }, []);

  const load = useCallback(
    async (pageIndex: number, offset: string | null, lang: string) => {
      setLoading(true);
      setError(null);
      try {
        const result = await api.datasetPreview({
          limit: 25,
          offset,
          language: lang || null,
        });
        setRows(result.rows);
        setOffsets((current) => {
          const next = [...current];
          next[pageIndex + 1] = result.next_offset;
          return next;
        });
        setPage(pageIndex);
      } catch (caught) {
        setError(
          caught instanceof api.ApiError
            ? caught.message
            : "Could not load dataset preview.",
        );
        setRows([]);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    void load(0, null, language);
  }, [language, load]);

  const collection = stats?.collection;
  const indexed = collection?.points_count ?? 0;

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-[var(--border)] px-5 py-3">
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
          <h2 className="text-[14px] font-medium text-[var(--text)]">Dataset</h2>
          <a
            href={stats?.dataset.url}
            target="_blank"
            rel="noreferrer"
            className="text-[11.5px] text-[var(--accent)] hover:underline"
          >
            {stats?.dataset.id ?? "ai4bharat/MSMARCO-XI"} ↗
          </a>
          <span className="flex-1" />
          <Stat label="indexed chunks" value={indexed.toLocaleString()} />
          <Stat
            label="upstream rows"
            value={(stats?.dataset.total_rows_upstream ?? 0).toLocaleString()}
          />
          <Stat
            label="upstream size"
            value={`${(((stats?.dataset.upstream_size_bytes ?? 0) / 1e9) || 0).toFixed(1)} GB`}
          />
        </div>

        {stats?.dataset.note && (
          <p className="mt-1.5 text-[10.5px] leading-relaxed text-[var(--text-dim)]">
            {stats.dataset.note}
          </p>
        )}

        <div className="mt-2.5 flex items-center gap-2">
          <select
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
            className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-2 py-1 text-[11.5px] text-[var(--text-muted)] outline-none focus:border-[var(--accent)]"
          >
            <option value="">all languages</option>
            <option value="en">English</option>
            {stats?.languages.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name} ({lang.native_name})
              </option>
            ))}
          </select>

          <span className="flex-1" />

          <button
            type="button"
            disabled={page === 0 || loading}
            onClick={() => void load(page - 1, offsets[page - 1] ?? null, language)}
            className="rounded-[var(--radius)] border border-[var(--border)] px-2 py-1 text-[11px] text-[var(--text-muted)] disabled:opacity-30"
          >
            ← prev
          </button>
          <span className="tabular text-[11px] text-[var(--text-dim)]">
            page {page + 1}
          </span>
          <button
            type="button"
            disabled={!offsets[page + 1] || loading}
            onClick={() => void load(page + 1, offsets[page + 1] ?? null, language)}
            className="rounded-[var(--radius)] border border-[var(--border)] px-2 py-1 text-[11px] text-[var(--text-muted)] disabled:opacity-30"
          >
            next →
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-3">
        {error ? (
          <div className="rounded-[var(--radius)] border border-[#d2992244] bg-[#d2992211] px-3 py-2 text-[12px] text-[var(--yellow)]">
            {error}
          </div>
        ) : loading ? (
          <div className="space-y-2">
            {Array.from({ length: 8 }).map((_, index) => (
              <div key={index} className="shimmer h-12 rounded-[var(--radius)]" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <p className="py-8 text-center text-[12.5px] text-[var(--text-dim)]">
            No indexed chunks yet. Run the ingest script to populate the index.
          </p>
        ) : (
          <div className="space-y-1.5">
            {rows.map((row, index) => (
              <div
                key={String(row.chunk_id ?? index)}
                className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-3 py-2"
              >
                <div className="mb-1 flex flex-wrap items-center gap-2 text-[10px]">
                  <span className="rounded-sm bg-[var(--bg-overlay)] px-1.5 py-0.5 uppercase text-[var(--text-dim)]">
                    {String(row.language ?? "?")}
                  </span>
                  <span className="text-[var(--text-dim)]">
                    {String(row.strategy ?? "")}
                  </span>
                  {row.query_type ? (
                    <span className="text-[var(--text-dim)]">
                      {String(row.query_type)}
                    </span>
                  ) : null}
                  {row.is_selected ? (
                    <span
                      className="rounded-sm px-1.5 py-0.5"
                      style={{ background: "#3fb95022", color: "var(--green)" }}
                      title="Marked relevant by MS MARCO's is_selected label"
                    >
                      ground truth
                    </span>
                  ) : null}
                  <span className="flex-1" />
                  <span className="tabular text-[var(--text-dim)]">
                    {String(row.doc_hash ?? "").slice(0, 12)}
                  </span>
                </div>
                <p
                  className={`text-[12px] leading-relaxed text-[var(--text-muted)] ${
                    row.language !== "en" ? "indic" : ""
                  }`}
                >
                  {String(row.text ?? "")}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="tabular text-[13px] font-medium text-[var(--text)]">{value}</span>
      <span className="text-[10px] uppercase tracking-wide text-[var(--text-dim)]">
        {label}
      </span>
    </div>
  );
}
