"use client";

/**
 * One answered query.
 *
 * Two tabs, because the two halves of RAG answer different questions:
 * "Retrieved" shows what the index actually returned, "Answer" shows what the
 * model made of it. Each carries its own elapsed time, so a slow query can be
 * attributed to the right half at a glance.
 */

import { useEffect, useRef, useState } from "react";
import { speak } from "@/lib/api";
import type { AnswerResponse, AnswerStatus } from "@/lib/types";

type Tab = "retrieved" | "answer" | "checks";

const STATUS: Record<AnswerStatus, { text: string; tone: "ok" | "warn" | "bad" }> = {
  answered: { text: "answered", tone: "ok" },
  refused_unsafe: { text: "blocked", tone: "bad" },
  refused_off_topic: { text: "out of scope", tone: "warn" },
  refused_low_confidence: { text: "no match", tone: "warn" },
  refused_ungrounded: { text: "ungrounded", tone: "bad" },
  degraded_extractive: { text: "degraded", tone: "warn" },
  error: { text: "error", tone: "bad" },
};

const TONE = {
  ok: { fg: "var(--ok)", bg: "var(--ok-soft)" },
  warn: { fg: "var(--warn)", bg: "var(--warn-soft)" },
  bad: { fg: "var(--bad)", bg: "var(--bad-soft)" },
};

export function Result({ response }: { response: AnswerResponse }) {
  const [tab, setTab] = useState<Tab>("answer");

  const stages = response.timing.stages;
  const retrievalMs = stages
    .filter((s) => s.counted_in_retrieval_budget)
    .reduce((sum, s) => sum + s.duration_ms, 0);
  const genMs = stages.find((s) => s.stage === "generate:llm")?.duration_ms ?? 0;

  const status = STATUS[response.status];
  const tone = TONE[status.tone];
  const blocked = response.guardrails.filter((g) => g.verdict === "block").length;

  return (
    <div className="rise overflow-hidden rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)]">
      {/* Tabs carry their own timing — that is the whole point of the split. */}
      <div className="flex items-center gap-1 border-b border-[var(--line)] px-2">
        <Tab
          active={tab === "retrieved"}
          onClick={() => setTab("retrieved")}
          label="Retrieved"
          badge={`${retrievalMs.toFixed(0)}ms`}
          count={response.retrieved.length}
        />
        <Tab
          active={tab === "answer"}
          onClick={() => setTab("answer")}
          label="Answer"
          badge={genMs > 0 ? `${genMs.toFixed(0)}ms` : undefined}
        />
        <Tab
          active={tab === "checks"}
          onClick={() => setTab("checks")}
          label="Checks"
          count={response.guardrails.length}
          alert={blocked > 0}
        />

        <span className="flex-1" />

        <span
          className="num rounded-full px-2 py-0.5 text-[10.5px]"
          style={{ background: tone.bg, color: tone.fg }}
        >
          {status.text}
        </span>
      </div>

      <div className="p-3">
        {tab === "answer" && <AnswerTab response={response} />}
        {tab === "retrieved" && <RetrievedTab response={response} />}
        {tab === "checks" && <ChecksTab response={response} />}
      </div>

      <Timeline stages={stages} retrievalMs={retrievalMs} />
    </div>
  );
}

function Tab({
  active,
  onClick,
  label,
  badge,
  count,
  alert,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  badge?: string;
  count?: number;
  alert?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="relative flex items-center gap-1.5 px-2.5 py-2 text-[12.5px] transition-colors"
      style={{ color: active ? "var(--ink)" : "var(--ink-faint)" }}
    >
      {label}
      {count !== undefined && (
        <span className="num text-[10.5px] text-[var(--ink-faint)]">{count}</span>
      )}
      {badge && (
        <span
          className="num text-[10.5px]"
          style={{ color: active ? "var(--accent)" : "var(--ink-faint)" }}
        >
          {badge}
        </span>
      )}
      {alert && (
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--bad)]" />
      )}
      {active && (
        <span className="absolute inset-x-1.5 -bottom-px h-0.5 rounded-full bg-[var(--accent)]" />
      )}
    </button>
  );
}

function AnswerTab({ response }: { response: AnswerResponse }) {
  const [audio, setAudio] = useState<"idle" | "load" | "play">("idle");
  const player = useRef<HTMLAudioElement | null>(null);
  const url = useRef<string | null>(null);

  useEffect(
    () => () => {
      player.current?.pause();
      if (url.current) URL.revokeObjectURL(url.current);
    },
    [],
  );

  const play = async () => {
    if (audio === "play") {
      player.current?.pause();
      setAudio("idle");
      return;
    }
    setAudio("load");
    try {
      const out = await speak(response.answer, response.detected_language);
      if (url.current) URL.revokeObjectURL(url.current);
      url.current = out.url;
      const el = new Audio(out.url);
      player.current = el;
      el.onended = () => setAudio("idle");
      await el.play();
      setAudio("play");
    } catch {
      setAudio("idle");
    }
  };

  return (
    <div className="space-y-3">
      <p className="indic text-[14px] leading-relaxed text-[var(--ink)]">
        {response.answer}
      </p>

      <div className="flex flex-wrap items-center gap-1.5">
        {response.citations.map((c) => (
          <span
            key={c.chunk_id}
            title={c.text}
            className="num cursor-help rounded-[var(--r-sm)] bg-[var(--accent-soft)] px-1.5 py-0.5 text-[10.5px] text-[var(--accent)]"
          >
            [{c.marker}] {c.score.toFixed(2)}
          </span>
        ))}

        <span className="flex-1" />

        {response.provider_used && (
          <span className="num text-[10.5px] text-[var(--ink-faint)]">
            {response.provider_used}
          </span>
        )}

        <button
          type="button"
          onClick={play}
          className="flex items-center gap-1 rounded-[var(--r-sm)] border border-[var(--line)] px-2 py-0.5 text-[11px] text-[var(--ink-soft)] transition-colors hover:border-[var(--accent)] hover:text-[var(--accent)]"
        >
          {audio === "load" ? (
            <span className="turn inline-block h-2.5 w-2.5 rounded-full border border-current border-t-transparent" />
          ) : (
            "♪"
          )}
          {audio === "play" ? "Stop" : "Listen"}
        </button>
      </div>

      {response.warnings.map((w, i) => (
        <p key={i} className="text-[11px] text-[var(--warn)]">
          {w}
        </p>
      ))}
    </div>
  );
}

function RetrievedTab({ response }: { response: AnswerResponse }) {
  const cited = new Set(response.citations.map((c) => c.chunk_id));

  if (!response.retrieved.length) {
    return <Empty>Nothing retrieved</Empty>;
  }

  return (
    <ol className="space-y-1.5">
      {response.retrieved.map((item, i) => {
        const score = item.rerank_score ?? item.fused_score;
        const isCited = cited.has(item.chunk.chunk_id);
        return (
          <li
            key={item.chunk.chunk_id}
            className="flex gap-2.5 rounded-[var(--r-sm)] border px-2.5 py-2"
            style={{
              borderColor: isCited ? "var(--accent)" : "var(--line)",
              background: isCited ? "var(--accent-soft)" : "var(--sunken)",
            }}
          >
            <span className="num w-4 shrink-0 pt-px text-[11px] text-[var(--ink-faint)]">
              {i + 1}
            </span>
            <p
              className={`flex-1 text-[12.5px] leading-relaxed text-[var(--ink-soft)] ${
                item.chunk.metadata.language !== "en" ? "indic" : ""
              }`}
            >
              {/* Small-to-big strategies embed a narrow span but retrieve the
                  wider parent; show what the model actually received. */}
              {item.chunk.context_text ?? item.chunk.text}
            </p>
            <div className="shrink-0 text-right">
              <div
                className="num text-[12px]"
                style={{
                  color:
                    score > 0.7
                      ? "var(--ok)"
                      : score > 0.4
                        ? "var(--warn)"
                        : "var(--ink-faint)",
                }}
              >
                {score.toFixed(3)}
              </div>
              <div className="num text-[9.5px] uppercase text-[var(--ink-faint)]">
                {item.chunk.metadata.language}
              </div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function ChecksTab({ response }: { response: AnswerResponse }) {
  if (!response.guardrails.length) return <Empty>No checks ran</Empty>;

  return (
    <div className="space-y-1.5">
      {response.guardrails.map((g) => {
        const tone =
          g.verdict === "pass" ? TONE.ok : g.verdict === "warn" ? TONE.warn : TONE.bad;
        return (
          <div
            key={g.stage}
            className="flex items-start gap-2.5 rounded-[var(--r-sm)] bg-[var(--sunken)] px-2.5 py-2"
          >
            <span
              className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ background: tone.fg }}
            />
            <div className="flex-1">
              <div className="flex items-baseline gap-2">
                <span className="text-[12.5px] text-[var(--ink)]">
                  {g.stage.replace(/_/g, " ")}
                </span>
                {g.score !== null && (
                  <span className="num text-[11px]" style={{ color: tone.fg }}>
                    {g.score.toFixed(3)}
                    {g.threshold !== null && (
                      <span className="text-[var(--ink-faint)]">
                        {" "}
                        / {g.threshold.toFixed(2)}
                      </span>
                    )}
                  </span>
                )}
                <span className="flex-1" />
                <span className="num text-[10px] text-[var(--ink-faint)]">
                  {g.duration_ms.toFixed(1)}ms
                </span>
              </div>
              <p className="mt-0.5 text-[11.5px] leading-snug text-[var(--ink-soft)]">
                {g.reason}
              </p>
            </div>
          </div>
        );
      })}

      {response.grounding_claims.length > 0 && (
        <div className="space-y-1 pt-1">
          {response.grounding_claims.map((claim, i) => (
            <div key={i} className="flex items-start gap-2 px-2.5 text-[11.5px]">
              <span
                className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ background: claim.supported ? "var(--ok)" : "var(--bad)" }}
              />
              <p className="indic flex-1 text-[var(--ink-soft)]">{claim.claim}</p>
              <span className="num text-[10px] text-[var(--ink-faint)]">
                {claim.confidence.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Proportional stage bar: one row, no labels until hover. */
function Timeline({
  stages,
  retrievalMs,
}: {
  stages: AnswerResponse["timing"]["stages"];
  retrievalMs: number;
}) {
  const total = stages.reduce((sum, s) => sum + s.duration_ms, 0);
  if (!total) return null;

  const COLORS: Record<string, string> = {
    "guardrail:input": "var(--ink-faint)",
    "embed:query": "var(--accent)",
    "search:vector": "var(--ok)",
    "rerank:cross_encoder": "var(--warn)",
    "guardrail:confidence": "var(--ink-faint)",
    "generate:llm": "var(--line-firm)",
    "guardrail:grounding": "var(--ink-faint)",
    "stt:sarvam": "var(--line-firm)",
  };

  return (
    <div className="flex items-center gap-2 border-t border-[var(--line)] px-3 py-2">
      <div className="flex h-1.5 flex-1 gap-px overflow-hidden rounded-full">
        {stages
          .filter((s) => s.duration_ms > 0)
          .map((s, i) => (
            <div
              key={i}
              title={`${s.stage} — ${s.duration_ms.toFixed(1)}ms`}
              style={{
                width: `${(s.duration_ms / total) * 100}%`,
                background: COLORS[s.stage] ?? "var(--line-firm)",
                opacity: s.counted_in_retrieval_budget ? 1 : 0.35,
              }}
            />
          ))}
      </div>
      <span
        className="num shrink-0 text-[11px]"
        style={{ color: retrievalMs < 200 ? "var(--ok)" : "var(--warn)" }}
      >
        {retrievalMs.toFixed(0)}ms
      </span>
      <span className="num shrink-0 text-[11px] text-[var(--ink-faint)]">
        {total.toFixed(0)}ms total
      </span>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <p className="py-6 text-center text-[12.5px] text-[var(--ink-faint)]">{children}</p>
  );
}
