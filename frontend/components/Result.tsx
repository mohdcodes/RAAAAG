"use client";

/**
 * One answered query.
 *
 * Opens on Retrieved, not Answer: in a RAG system the retrieved context is the
 * evidence and the answer is the claim, so the evidence is what you check
 * first. Each tab carries its own elapsed time, and a metric strip sits under
 * every result so per-query numbers are visible without leaving the chat.
 */

import { useEffect, useRef, useState } from "react";
import { speak } from "@/lib/api";
import type { AnswerResponse, AnswerStatus } from "@/lib/types";

type Tab = "retrieved" | "answer" | "checks";

const STATUS: Record<AnswerStatus, { text: string; tone: string }> = {
  answered: { text: "answered", tone: "var(--ok)" },
  refused_unsafe: { text: "blocked", tone: "var(--bad)" },
  refused_off_topic: { text: "out of scope", tone: "var(--warn)" },
  refused_low_confidence: { text: "no match", tone: "var(--warn)" },
  refused_ungrounded: { text: "ungrounded", tone: "var(--bad)" },
  degraded_extractive: { text: "degraded", tone: "var(--warn)" },
  error: { text: "error", tone: "var(--bad)" },
};

const STAGE_COLOR: Record<string, string> = {
  "guardrail:input": "var(--cream-faint)",
  "embed:query": "var(--gold)",
  "search:vector": "var(--ok)",
  "rerank:cross_encoder": "var(--pink)",
  "guardrail:confidence": "var(--cream-faint)",
  "generate:llm": "var(--line-firm)",
  "guardrail:grounding": "var(--cream-faint)",
  "stt:sarvam": "var(--line-firm)",
};

export function Result({ response }: { response: AnswerResponse }) {
  const [tab, setTab] = useState<Tab>("retrieved");

  const stages = response.timing.stages;
  const retrievalMs = stages
    .filter((s) => s.counted_in_retrieval_budget)
    .reduce((sum, s) => sum + s.duration_ms, 0);
  const embedMs = stages.find((s) => s.stage === "embed:query")?.duration_ms ?? 0;
  const searchMs = stages.find((s) => s.stage === "search:vector")?.duration_ms ?? 0;
  const genMs = stages.find((s) => s.stage === "generate:llm")?.duration_ms ?? 0;
  const sttMs = stages.find((s) => s.stage === "stt:sarvam")?.duration_ms ?? 0;
  const totalMs = stages.reduce((sum, s) => sum + s.duration_ms, 0);

  const status = STATUS[response.status];
  const blocked = response.guardrails.some((g) => g.verdict === "block");

  return (
    <div className="rise overflow-hidden rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)]">
      <div className="flex items-center gap-1 border-b border-[var(--line)] px-2">
        <TabButton
          active={tab === "retrieved"}
          onClick={() => setTab("retrieved")}
          label="Retrieved"
          count={response.retrieved.length}
          ms={retrievalMs}
        />
        <TabButton
          active={tab === "answer"}
          onClick={() => setTab("answer")}
          label="Answer"
          ms={genMs || undefined}
        />
        <TabButton
          active={tab === "checks"}
          onClick={() => setTab("checks")}
          label="Checks"
          count={response.guardrails.length}
          alert={blocked}
        />
        <span className="flex-1" />
        <span
          className="num rounded-[var(--r-pill)] px-2.5 py-0.5 text-[10px] uppercase tracking-wider"
          style={{ color: status.tone, background: `${status.tone}1f` }}
        >
          {status.text}
        </span>
      </div>

      <div className="p-3.5">
        {tab === "retrieved" && <Retrieved response={response} />}
        {tab === "answer" && <Answer response={response} />}
        {tab === "checks" && <Checks response={response} />}
      </div>

      {/* Per-query metrics — visible on every result, not hidden in a tab. */}
      <div className="border-t border-[var(--line)] px-3.5 py-2.5">
        <div className="mb-2 flex h-1.5 gap-px overflow-hidden rounded-[var(--r-pill)]">
          {stages
            .filter((s) => s.duration_ms > 0)
            .map((s, i) => (
              <div
                key={i}
                title={`${s.stage} — ${s.duration_ms.toFixed(1)}ms`}
                style={{
                  width: `${(s.duration_ms / totalMs) * 100}%`,
                  background: STAGE_COLOR[s.stage] ?? "var(--line-firm)",
                  opacity: s.counted_in_retrieval_budget ? 1 : 0.3,
                }}
              />
            ))}
        </div>

        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <Stat
            label="retrieval"
            value={retrievalMs.toFixed(0)}
            unit="ms"
            tone={retrievalMs < 200 ? "var(--ok)" : "var(--warn)"}
            strong
          />
          {sttMs > 0 && <Stat label="speech" value={sttMs.toFixed(0)} unit="ms" />}
          <Stat label="embed" value={embedMs.toFixed(0)} unit="ms" />
          <Stat label="search" value={searchMs.toFixed(1)} unit="ms" />
          <Stat label="llm" value={genMs.toFixed(0)} unit="ms" dim />
          <Stat label="chunks" value={String(response.retrieved.length)} />
          {response.retrieved[0] && (
            <Stat
              label="top score"
              value={(
                response.retrieved[0].rerank_score ?? response.retrieved[0].fused_score
              ).toFixed(3)}
            />
          )}
          <span className="flex-1" />
          <Stat label="total" value={totalMs.toFixed(0)} unit="ms" dim />
        </div>
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  label,
  count,
  ms,
  alert,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count?: number;
  ms?: number;
  alert?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="relative flex items-center gap-1.5 px-3 py-2.5 text-[13px] transition-colors"
      style={{ color: active ? "var(--cream)" : "var(--cream-faint)" }}
    >
      {label}
      {count !== undefined && (
        <span className="num text-[10.5px] opacity-70">{count}</span>
      )}
      {ms !== undefined && (
        <span
          className="num text-[10.5px]"
          style={{ color: active ? "var(--gold)" : "var(--cream-faint)" }}
        >
          {ms.toFixed(0)}ms
        </span>
      )}
      {alert && <span className="h-1.5 w-1.5 rounded-full bg-[var(--bad)]" />}
      {active && (
        <span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-[var(--gold)]" />
      )}
    </button>
  );
}

function Stat({
  label,
  value,
  unit,
  tone,
  strong,
  dim,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: string;
  strong?: boolean;
  dim?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-1">
      <span
        className="num"
        style={{
          fontSize: strong ? "15px" : "12.5px",
          color: tone ?? (dim ? "var(--cream-faint)" : "var(--cream)"),
        }}
      >
        {value}
        {unit && <span className="text-[10px] opacity-60">{unit}</span>}
      </span>
      <span className="eyebrow">{label}</span>
    </div>
  );
}

function Retrieved({ response }: { response: AnswerResponse }) {
  const cited = new Set(response.citations.map((c) => c.chunk_id));

  if (!response.retrieved.length) {
    return (
      <p className="py-6 text-center text-[12.5px] text-[var(--cream-faint)]">
        Nothing retrieved
      </p>
    );
  }

  return (
    <ol className="space-y-2">
      {response.retrieved.map((item, i) => {
        const score = item.rerank_score ?? item.fused_score;
        const isCited = cited.has(item.chunk.chunk_id);
        return (
          <li
            key={item.chunk.chunk_id}
            className="flex gap-3 rounded-[var(--r-sm)] border px-3 py-2.5"
            style={{
              borderColor: isCited ? "var(--gold)" : "var(--line)",
              background: isCited ? "rgba(227,178,60,0.07)" : "var(--sunken)",
            }}
          >
            <span className="num w-4 shrink-0 pt-0.5 text-[11px] text-[var(--cream-faint)]">
              {i + 1}
            </span>
            <p
              className={`flex-1 text-[12.5px] leading-relaxed text-[var(--cream-soft)] ${
                item.chunk.metadata.language !== "en" ? "indic" : ""
              }`}
            >
              {item.chunk.context_text ?? item.chunk.text}
            </p>
            <div className="shrink-0 text-right">
              <div
                className="num text-[12.5px]"
                style={{
                  color:
                    score > 0.7
                      ? "var(--ok)"
                      : score > 0.4
                        ? "var(--gold)"
                        : "var(--cream-faint)",
                }}
              >
                {score.toFixed(3)}
              </div>
              <div className="eyebrow mt-0.5">{item.chunk.metadata.language}</div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function Answer({ response }: { response: AnswerResponse }) {
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
      <p className="indic text-[14.5px] leading-relaxed text-[var(--cream)]">
        {response.answer}
      </p>

      <div className="flex flex-wrap items-center gap-1.5">
        {response.citations.map((c) => (
          <span
            key={c.chunk_id}
            title={c.text}
            className="num cursor-help rounded-[var(--r-pill)] border border-[var(--line-firm)] px-2 py-0.5 text-[10.5px] text-[var(--gold)]"
          >
            [{c.marker}] {c.score.toFixed(2)}
          </span>
        ))}
        <span className="flex-1" />
        {response.provider_used && (
          <span className="eyebrow">{response.provider_used}</span>
        )}
        <button
          type="button"
          onClick={play}
          className="flex items-center gap-1.5 rounded-[var(--r-pill)] border border-[var(--line-firm)] px-2.5 py-1 text-[11.5px] text-[var(--cream-soft)] transition-colors hover:border-[var(--gold)] hover:text-[var(--gold)]"
        >
          {audio === "load" ? (
            <span className="turn inline-block h-2.5 w-2.5 rounded-full border border-current border-t-transparent" />
          ) : (
            <span>♪</span>
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

function Checks({ response }: { response: AnswerResponse }) {
  if (!response.guardrails.length) {
    return (
      <p className="py-6 text-center text-[12.5px] text-[var(--cream-faint)]">
        No checks ran
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {response.guardrails.map((g) => {
        const tone =
          g.verdict === "pass"
            ? "var(--ok)"
            : g.verdict === "warn"
              ? "var(--warn)"
              : "var(--bad)";
        return (
          <div
            key={g.stage}
            className="flex items-start gap-2.5 rounded-[var(--r-sm)] bg-[var(--sunken)] px-3 py-2"
          >
            <span
              className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ background: tone }}
            />
            <div className="flex-1">
              <div className="flex items-baseline gap-2">
                <span className="text-[12.5px] text-[var(--cream)]">
                  {g.stage.replace(/_/g, " ")}
                </span>
                {g.score !== null && (
                  <span className="num text-[11px]" style={{ color: tone }}>
                    {g.score.toFixed(3)}
                    {g.threshold !== null && (
                      <span className="text-[var(--cream-faint)]">
                        {" / "}
                        {g.threshold.toFixed(2)}
                      </span>
                    )}
                  </span>
                )}
                <span className="flex-1" />
                <span className="num text-[10px] text-[var(--cream-faint)]">
                  {g.duration_ms.toFixed(1)}ms
                </span>
              </div>
              <p className="mt-0.5 text-[11.5px] leading-snug text-[var(--cream-soft)]">
                {g.reason}
              </p>
            </div>
          </div>
        );
      })}

      {response.grounding_claims.length > 0 && (
        <div className="space-y-1 border-t border-[var(--line)] pt-2">
          <span className="eyebrow">claim verification</span>
          {response.grounding_claims.map((claim, i) => (
            <div key={i} className="flex items-start gap-2 text-[11.5px]">
              <span
                className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ background: claim.supported ? "var(--ok)" : "var(--bad)" }}
              />
              <p className="indic flex-1 text-[var(--cream-soft)]">{claim.claim}</p>
              <span className="num text-[10px] text-[var(--cream-faint)]">
                {claim.confidence.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
