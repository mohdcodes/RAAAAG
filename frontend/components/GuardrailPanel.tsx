"use client";

/**
 * Guardrail verdicts.
 *
 * The brief asks the system to show it knows when *not* to answer, so each
 * layer's verdict, score, threshold and reasoning is rendered inspectably —
 * a refusal should be legible as a decision with evidence, not an error.
 */

import { useState } from "react";
import type { GroundingClaim, GuardrailResult, GuardrailVerdict } from "@/lib/types";

const STAGE_LABELS: Record<string, string> = {
  input_safety: "Input safety",
  off_topic: "Topic scope",
  retrieval_confidence: "Retrieval confidence",
  grounding: "Answer grounding",
};

const VERDICT_STYLE: Record<GuardrailVerdict, { color: string; label: string; icon: string }> = {
  pass: { color: "var(--green)", label: "PASS", icon: "✓" },
  warn: { color: "var(--yellow)", label: "WARN", icon: "!" },
  block: { color: "var(--red)", label: "BLOCK", icon: "✕" },
};

export function GuardrailPanel({
  guardrails,
  claims = [],
}: {
  guardrails: GuardrailResult[];
  claims?: GroundingClaim[];
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  if (!guardrails.length) return null;

  return (
    <div className="space-y-2">
      <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
        Guardrails
      </span>

      <div className="space-y-1">
        {guardrails.map((guard) => {
          const style = VERDICT_STYLE[guard.verdict];
          const isOpen = expanded === guard.stage;
          const hasDetail =
            Object.keys(guard.details ?? {}).length > 0 || guard.score !== null;

          return (
            <div
              key={guard.stage}
              className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)]"
            >
              <button
                type="button"
                onClick={() => hasDetail && setExpanded(isOpen ? null : guard.stage)}
                className={`flex w-full items-center gap-2.5 px-2.5 py-1.5 text-left transition-colors ${
                  hasDetail ? "hover:bg-[var(--bg-overlay)]" : "cursor-default"
                }`}
              >
                <span
                  className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-bold"
                  style={{ background: `${style.color}22`, color: style.color }}
                >
                  {style.icon}
                </span>

                <span className="flex-1 truncate text-[12px] text-[var(--text-muted)]">
                  {STAGE_LABELS[guard.stage] ?? guard.stage}
                </span>

                {guard.score !== null && (
                  <span className="tabular text-[11px]" style={{ color: style.color }}>
                    {guard.score.toFixed(3)}
                    {guard.threshold !== null && (
                      <span className="text-[var(--text-dim)]">
                        {" "}
                        / {guard.threshold.toFixed(2)}
                      </span>
                    )}
                  </span>
                )}

                <span
                  className="tabular w-[42px] shrink-0 text-right text-[10px] font-medium"
                  style={{ color: style.color }}
                >
                  {style.label}
                </span>

                <span className="tabular w-[46px] shrink-0 text-right text-[10px] text-[var(--text-dim)]">
                  {guard.duration_ms.toFixed(1)}ms
                </span>
              </button>

              {isOpen && (
                <div className="animate-slide-up space-y-2 border-t border-[var(--border)] px-2.5 py-2">
                  <p className="text-[11.5px] leading-relaxed text-[var(--text-muted)]">
                    {guard.reason}
                  </p>

                  {Object.keys(guard.details ?? {}).length > 0 && (
                    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[10.5px]">
                      {Object.entries(guard.details).map(([key, value]) => (
                        <div key={key} className="contents">
                          <dt className="text-[var(--text-dim)]">{key}</dt>
                          <dd className="tabular truncate text-[var(--text-muted)]">
                            {typeof value === "object"
                              ? JSON.stringify(value)
                              : String(value)}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {claims.length > 0 && <GroundingClaims claims={claims} />}
    </div>
  );
}

function GroundingClaims({ claims }: { claims: GroundingClaim[] }) {
  const supported = claims.filter((c) => c.supported).length;

  return (
    <div className="space-y-1 pt-1">
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-dim)]">
          Claim verification
        </span>
        <span className="tabular text-[10.5px] text-[var(--text-dim)]">
          {supported}/{claims.length} traced to context
        </span>
      </div>

      {claims.map((claim, index) => (
        <div
          key={index}
          className="flex items-start gap-2 rounded-[var(--radius)] border px-2 py-1.5"
          style={{
            borderColor: claim.supported ? "var(--border)" : "#f8514944",
            background: claim.supported ? "var(--bg-input)" : "#f8514911",
          }}
        >
          <span
            className="mt-[3px] h-1.5 w-1.5 shrink-0 rounded-full"
            style={{ background: claim.supported ? "var(--green)" : "var(--red)" }}
          />
          <p className="flex-1 text-[11.5px] leading-relaxed text-[var(--text-muted)] indic">
            {claim.claim}
          </p>
          <span className="tabular shrink-0 text-[10px] text-[var(--text-dim)]">
            {claim.confidence.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}
