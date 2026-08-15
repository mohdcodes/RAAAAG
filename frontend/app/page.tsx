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
      <header className="flex shrink-0 items-center gap-5 border-b border-[var(--line)] px-5 py-3">
        <div className="flex items-baseline gap-2.5">
          <span className="text-[16px] font-bold tracking-tight text-[var(--cream)]">
            RAAAAAG<span className="text-[var(--gold)]">&nbsp;!!</span>
          </span>
          <span className="eyebrow hidden sm:inline">Hacker House Goa</span>
        </div>

        <nav className="flex gap-1">
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
              className="rounded-[var(--r-pill)] px-3 py-1 text-[13px] transition-colors"
              style={{
                background: tab === id ? "var(--forest)" : "transparent",
                color: tab === id ? "var(--cream)" : "var(--cream-faint)",
              }}
            >
              {label}
            </button>
          ))}
        </nav>

        <span className="flex-1" />

        <span className="eyebrow hidden md:inline">by BrBik</span>
        <span
          className="h-2 w-2 rounded-full"
          style={{
            background: ok ? "var(--ok)" : down ? "var(--bad)" : "var(--warn)",
          }}
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
