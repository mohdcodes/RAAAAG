"use client";

import { useCallback, useEffect, useState } from "react";
import * as api from "@/lib/api";
import type { HealthResponse } from "@/lib/types";
import { Chat } from "@/components/Chat";
import { DatasetPreview } from "@/components/DatasetPreview";
import { Metrics } from "@/components/Metrics";

type Tab = "chat" | "metrics" | "data";

export default function Home() {
  const [tab, setTab] = useState<Tab>("chat");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [down, setDown] = useState(false);

  const check = useCallback(async () => {
    try {
      setHealth(await api.health());
      setDown(false);
    } catch {
      setDown(true);
    }
  }, []);

  useEffect(() => {
    void check();
    const timer = setInterval(check, 15_000);
    return () => clearInterval(timer);
  }, [check]);

  const ok =
    !down &&
    Boolean(health?.components.vector_store.target_collection_present) &&
    Boolean(health?.components.generation.any_available);

  return (
    <div className="flex h-screen flex-col">
      <header className="flex shrink-0 items-center gap-5 border-b border-[var(--line)] bg-[var(--panel)] px-5 py-2.5">
        <span className="text-[14px] font-medium tracking-tight text-[var(--ink)]">
          Voice<span className="text-[var(--accent)]">RAG</span>
        </span>

        <nav className="flex gap-0.5">
          {(
            [
              ["chat", "Chat"],
              ["metrics", "Metrics"],
              ["data", "Data"],
            ] as Array<[Tab, string]>
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className="rounded-[var(--r-sm)] px-2.5 py-1 text-[13px] transition-colors"
              style={{
                background: tab === id ? "var(--sunken)" : "transparent",
                color: tab === id ? "var(--ink)" : "var(--ink-faint)",
              }}
            >
              {label}
            </button>
          ))}
        </nav>

        <span className="flex-1" />

        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{ background: ok ? "var(--ok)" : down ? "var(--bad)" : "var(--warn)" }}
          title={down ? "API unreachable" : ok ? "ready" : "degraded"}
        />
      </header>

      <main className="min-h-0 flex-1">
        {tab === "chat" && <Chat health={health} />}
        {tab === "metrics" && <Metrics />}
        {tab === "data" && <DatasetPreview />}
      </main>
    </div>
  );
}
