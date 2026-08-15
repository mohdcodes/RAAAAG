"use client";

/**
 * Dataset browser.
 *
 * Pages through indexed chunks with a language filter, and shows the upstream
 * dataset's real size alongside what is actually indexed — so the sample's
 * scope is visible rather than implied.
 */

import { useCallback, useEffect, useState } from "react";
import * as api from "@/lib/api";
import type { DatasetPreviewRow, DatasetStats } from "@/lib/types";

export function DatasetPreview() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [rows, setRows] = useState<DatasetPreviewRow[]>([]);
  const [language, setLanguage] = useState("");
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
        const result = await api.datasetPreview({ limit: 25, offset, language: lang || null });
        setRows(result.rows);
        setOffsets((current) => {
          const next = [...current];
          next[pageIndex + 1] = result.next_offset;
          return next;
        });
        setPage(pageIndex);
      } catch (caught) {
        setError(
          caught instanceof api.ApiError ? caught.message : "Could not load preview.",
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

  const indexed = stats?.collection.points_count ?? 0;

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-[var(--line)] px-5 py-3">
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
          <a
            href={stats?.dataset.url}
            target="_blank"
            rel="noreferrer"
            className="text-[13px] text-[var(--gold)] hover:underline"
          >
            {stats?.dataset.id ?? "ai4bharat/MSMARCO-XI"} ↗
          </a>
          <span className="flex-1" />
          <Stat value={indexed.toLocaleString()} label="indexed" />
          <Stat
            value={`${((stats?.dataset.total_rows_upstream ?? 0) / 1e6).toFixed(1)}M`}
            label="upstream rows"
          />
          <Stat
            value={`${((stats?.dataset.upstream_size_bytes ?? 0) / 1e9).toFixed(0)}GB`}
            label="upstream size"
          />
        </div>

        <div className="mt-3 flex items-center gap-2">
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="rounded-[var(--r-sm)] border border-[var(--line)] bg-[var(--sunken)] px-2 py-1 text-[12px] text-[var(--cream-soft)] outline-none focus:border-[var(--gold)]"
          >
            <option value="">all languages</option>
            <option value="en">English</option>
            {stats?.languages.map((l) => (
              <option key={l.code} value={l.code}>
                {l.name} ({l.native_name})
              </option>
            ))}
          </select>

          <span className="flex-1" />

          <button
            type="button"
            disabled={page === 0 || loading}
            onClick={() => void load(page - 1, offsets[page - 1] ?? null, language)}
            className="rounded-[var(--r-pill)] border border-[var(--line)] px-2.5 py-0.5 text-[11.5px] text-[var(--cream-soft)] disabled:opacity-25"
          >
            prev
          </button>
          <span className="num text-[11.5px] text-[var(--cream-faint)]">{page + 1}</span>
          <button
            type="button"
            disabled={!offsets[page + 1] || loading}
            onClick={() => void load(page + 1, offsets[page + 1] ?? null, language)}
            className="rounded-[var(--r-pill)] border border-[var(--line)] px-2.5 py-0.5 text-[11.5px] text-[var(--cream-soft)] disabled:opacity-25"
          >
            next
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-3">
        {error ? (
          <div
            className="rounded-[var(--r-sm)] border border-[var(--warn)] px-3 py-2 text-[12px] text-[var(--warn)]"
            style={{ background: "rgba(227,178,60,0.1)" }}
          >
            {error}
          </div>
        ) : loading ? (
          <div className="space-y-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="skeleton h-14 rounded-[var(--r-sm)]" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <p className="py-10 text-center text-[12.5px] text-[var(--cream-faint)]">
            No chunks indexed yet.
          </p>
        ) : (
          <div className="space-y-2">
            {rows.map((row, i) => (
              <div
                key={String(row.chunk_id ?? i)}
                className="rounded-[var(--r-sm)] border border-[var(--line)] bg-[var(--panel)] px-3 py-2.5"
              >
                <div className="mb-1.5 flex flex-wrap items-center gap-2">
                  <span className="eyebrow">{String(row.language ?? "?")}</span>
                  <span className="eyebrow">{String(row.strategy ?? "")}</span>
                  {row.is_selected ? (
                    <span
                      className="num rounded-[var(--r-pill)] px-2 py-0.5 text-[9.5px] uppercase tracking-wider"
                      style={{ background: "rgba(126,201,143,0.15)", color: "var(--ok)" }}
                      title="Marked relevant by the dataset's is_selected label"
                    >
                      ground truth
                    </span>
                  ) : null}
                  <span className="flex-1" />
                  <span className="num text-[10px] text-[var(--cream-faint)]">
                    {String(row.doc_hash ?? "").slice(0, 10)}
                  </span>
                </div>
                <p
                  className={`text-[12.5px] leading-relaxed text-[var(--cream-soft)] ${
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

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="num text-[14px] text-[var(--cream)]">{value}</span>
      <span className="eyebrow">{label}</span>
    </div>
  );
}
