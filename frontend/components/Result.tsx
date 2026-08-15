"use client";

/**
 * One answered query, laid out in pipeline order.
 *
 * Retrieval sits above generation because that is the order the work happened
 * in and the order the evidence should be read in: the passages are what the
 * system found, the answer is what it claimed from them. Tabs hid that
 * relationship; stacking makes it the structure of the page.
 *
 * The retrieval figure is the server's own instrumentation, not a wall-clock
 * measurement from the browser — network and rendering time are real but they
 * are not what the sub-200ms budget covers.
 */

import { useEffect, useRef, useState } from "react";
import { speak } from "@/lib/api";
import type { AnswerResponse, AnswerStatus } from "@/lib/types";

const STATUS: Record<AnswerStatus, { text: string; tone: string }> = {
  answered: { text: "answered", tone: "var(--ok)" },
  refused_unsafe: { text: "blocked", tone: "var(--bad)" },
  refused_off_topic: { text: "out of scope", tone: "var(--warn)" },
  refused_low_confidence: { text: "no match", tone: "var(--warn)" },
  refused_ungrounded: { text: "ungrounded", tone: "var(--bad)" },
  degraded_extractive: { text: "degraded", tone: "var(--warn)" },
  error: { text: "error", tone: "var(--bad)" },
};

export function Result({ response }: { response: AnswerResponse }) {
  const stages = response.timing.stages;
  const ms = (name: string) =>
    stages.find((s) => s.stage === name)?.duration_ms ?? 0;

  const retrievalMs = stages
    .filter((s) => s.counted_in_retrieval_budget)
    .reduce((sum, s) => sum + s.duration_ms, 0);
  const genMs = ms("generate:llm");
  const status = STATUS[response.status];
  const withinBudget = retrievalMs < 200;

  return (
    <div className="space-y-2.5">
      {/* ── 1. RETRIEVAL ─────────────────────────────────────────── */}
      <section className="rise overflow-hidden rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)]">
        <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-[var(--line)] px-3.5 py-2">
          <span className="text-[12.5px] font-medium text-[var(--cream)]">
            Retrieved
          </span>
          <span className="num text-[11px] text-[var(--cream-faint)]">
            {response.retrieved.length} chunks
          </span>

          <span className="flex-1" />

          <span className="num text-[11px] text-[var(--cream-faint)]">
            embed {ms("embed:query").toFixed(0)}
          </span>
          <span className="num text-[11px] text-[var(--cream-faint)]">
            search {ms("search:vector").toFixed(1)}
          </span>
          {ms("rerank:cross_encoder") > 0 && (
            <span className="num text-[11px] text-[var(--cream-faint)]">
              rerank {ms("rerank:cross_encoder").toFixed(0)}
            </span>
          )}

          {/* The headline number: server-measured retrieval against the bar. */}
          <span
            className="num rounded-[var(--r-pill)] px-2.5 py-0.5 text-[12px] font-medium"
            style={{
              color: withinBudget ? "var(--ok)" : "var(--warn)",
              background: withinBudget
                ? "rgba(126,201,143,0.14)"
                : "rgba(227,178,60,0.14)",
            }}
            title="Guardrails + embedding + search + rerank, measured server-side"
          >
            {retrievalMs.toFixed(0)}ms {withinBudget ? "✓" : ""} / 200
          </span>
        </header>

        <div className="max-h-[340px] overflow-y-auto p-3">
          <Passages response={response} />
        </div>
      </section>

      {/* ── 2. GENERATION ────────────────────────────────────────── */}
      <section className="rise overflow-hidden rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)]">
        <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-[var(--line)] px-3.5 py-2">
          <span className="text-[12.5px] font-medium text-[var(--cream)]">
            Answer
          </span>
          {response.provider_used && (
            <span className="eyebrow">{response.provider_used}</span>
          )}

          <span className="flex-1" />

          {genMs > 0 && (
            <span
              className="num text-[11px] text-[var(--cream-faint)]"
              title="Third-party LLM call — reported, but outside the retrieval budget"
            >
              {genMs.toFixed(0)}ms · external
            </span>
          )}
          <span
            className="num rounded-[var(--r-pill)] px-2 py-0.5 text-[10px] uppercase tracking-wider"
            style={{ color: status.tone, background: `${status.tone}1f` }}
          >
            {status.text}
          </span>
        </header>

        <div className="p-3.5">
          <Answer response={response} />
        </div>
      </section>

      {/* ── 3. CHECKS ────────────────────────────────────────────── */}
      <Checks response={response} />
    </div>
  );
}

function Passages({ response }: { response: AnswerResponse }) {
  const cited = new Set(response.citations.map((c) => c.chunk_id));

  if (!response.retrieved.length) {
    return (
      <p className="py-4 text-center text-[12.5px] text-[var(--cream-faint)]">
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
  const [open, setOpen] = useState(false);
  if (!response.guardrails.length) return null;

  const blocked = response.guardrails.filter((g) => g.verdict === "block").length;
  const warned = response.guardrails.filter((g) => g.verdict === "warn").length;
  const totalMs = response.timing.stages.reduce((s, x) => s + x.duration_ms, 0);

  return (
    <div className="overflow-hidden rounded-[var(--r)] border border-[var(--line)]">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2.5 px-3.5 py-2 text-left transition-colors hover:bg-[var(--panel)]"
      >
        <span
          className="text-[11px] text-[var(--cream-faint)] transition-transform"
          style={{ transform: open ? "rotate(90deg)" : "none" }}
        >
          ▸
        </span>
        <span className="text-[12px] text-[var(--cream-soft)]">
          {response.guardrails.length} checks
        </span>
        {blocked > 0 && (
          <span className="num text-[11px] text-[var(--bad)]">{blocked} blocked</span>
        )}
        {warned > 0 && (
          <span className="num text-[11px] text-[var(--warn)]">{warned} warn</span>
        )}
        {blocked === 0 && warned === 0 && (
          <span className="num text-[11px] text-[var(--ok)]">all passed</span>
        )}
        <span className="flex-1" />
        <span className="num text-[11px] text-[var(--cream-faint)]">
          {totalMs.toFixed(0)}ms end to end
        </span>
      </button>

      {open && (
        <div className="space-y-2 border-t border-[var(--line)] bg-[var(--panel)] p-3">
          {response.guardrails.map((g) => {
            const tone =
              g.verdict === "pass"
                ? "var(--ok)"
                : g.verdict === "warn"
                  ? "var(--warn)"
                  : "var(--bad)";
            return (
              <div key={g.stage} className="flex items-start gap-2.5">
                <span
                  className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ background: tone }}
                />
                <div className="flex-1">
                  <div className="flex items-baseline gap-2">
                    <span className="text-[12px] text-[var(--cream)]">
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
                    style={{
                      background: claim.supported ? "var(--ok)" : "var(--bad)",
                    }}
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
      )}
    </div>
  );
}
