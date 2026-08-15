"use client";

import { useCallback, useEffect, useState } from "react";
import * as api from "@/lib/api";
import type { HealthResponse } from "@/lib/types";
import { Analytics } from "@/components/Analytics";
import { ChatPanel } from "@/components/ChatPanel";
import { DatasetPreview } from "@/components/DatasetPreview";
import { Strategies } from "@/components/Strategies";

type Tab = "chat" | "dataset" | "analytics" | "strategies";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "chat", label: "Chat" },
  { id: "dataset", label: "Dataset" },
  { id: "analytics", label: "Latency" },
  { id: "strategies", label: "Chunking" },
];

export default function Home() {
  const [tab, setTab] = useState<Tab>("chat");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setUnreachable(false);
    } catch {
      setUnreachable(true);
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
    const timer = setInterval(refreshHealth, 15_000);
    return () => clearInterval(timer);
  }, [refreshHealth]);

  return (
    <div className="flex h-screen flex-col">
      <header className="flex shrink-0 items-center gap-4 border-b border-[var(--border)] bg-[var(--bg-raised)] px-5 py-2.5">
        <div className="flex items-baseline gap-2">
          <h1 className="text-[14.5px] font-medium tracking-tight text-[var(--text)]">
            Voice<span className="text-[var(--accent)]">RAG</span>
          </h1>
          <span className="text-[10.5px] text-[var(--text-dim)]">
            MSMARCO-XI · 14 languages
          </span>
        </div>

        <nav className="flex gap-0.5">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              onClick={() => setTab(entry.id)}
              className="rounded-[var(--radius)] px-2.5 py-1 text-[12px] transition-colors"
              style={{
                background: tab === entry.id ? "var(--bg-overlay)" : "transparent",
                color: tab === entry.id ? "var(--text)" : "var(--text-dim)",
              }}
            >
              {entry.label}
            </button>
          ))}
        </nav>

        <span className="flex-1" />

        <HealthIndicator health={health} unreachable={unreachable} />
      </header>

      <main className="min-h-0 flex-1">
        {tab === "chat" && <ChatPanel health={health} />}
        {tab === "dataset" && <DatasetPreview />}
        {tab === "analytics" && <Analytics />}
        {tab === "strategies" && <Strategies />}
      </main>
    </div>
  );
}

function HealthIndicator({
  health,
  unreachable,
}: {
  health: HealthResponse | null;
  unreachable: boolean;
}) {
  const [open, setOpen] = useState(false);

  if (unreachable) {
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-[var(--red)]">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--red)]" />
        API unreachable
      </span>
    );
  }

  if (!health) {
    return <span className="text-[11px] text-[var(--text-dim)]">connecting…</span>;
  }

  const vectorOk = Boolean(health.components.vector_store.target_collection_present);
  const genOk = health.components.generation.any_available;
  const sttOk = health.components.voice.stt_available;
  const allOk = vectorOk && genOk && sttOk;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 rounded-[var(--radius)] px-2 py-1 text-[11px] text-[var(--text-dim)] transition-colors hover:bg-[var(--bg-overlay)]"
      >
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{
            background: allOk
              ? "var(--green)"
              : genOk
                ? "var(--yellow)"
                : "var(--red)",
          }}
        />
        {allOk ? "ready" : "degraded"}
      </button>

      {open && (
        <div className="animate-slide-up absolute right-0 top-full z-20 mt-1.5 w-64 space-y-1 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-overlay)] p-2 shadow-xl">
          <Row label="Vector index" ok={vectorOk} detail={vectorOk ? "loaded" : "not built"} />
          <Row
            label="Generation"
            ok={genOk}
            detail={health.components.generation.providers
              .filter((p) => p.configured)
              .map((p) => p.provider)
              .join(", ") || "no keys"}
          />
          <Row
            label="Voice (Sarvam)"
            ok={sttOk}
            detail={sttOk ? "enabled" : "no SARVAM_API_KEY"}
          />
          <Row
            label="Guardrails"
            ok={health.components.guardrails.enabled}
            detail={`threshold ${health.components.guardrails.confidence_threshold}`}
          />
          <p className="border-t border-[var(--border)] pt-1.5 text-[10px] text-[var(--text-dim)]">
            {String(health.config.embedding_model ?? "")}
          </p>
        </div>
      )}
    </div>
  );
}

function Row({
  label,
  ok,
  detail,
}: {
  label: string;
  ok: boolean;
  detail: string;
}) {
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span
        className="h-1.5 w-1.5 shrink-0 rounded-full"
        style={{ background: ok ? "var(--green)" : "var(--text-dim)" }}
      />
      <span className="text-[var(--text-muted)]">{label}</span>
      <span className="flex-1" />
      <span className="truncate text-[10px] text-[var(--text-dim)]">{detail}</span>
    </div>
  );
}
