"use client";

/**
 * The chat surface: composer, transcript, and retrieval controls.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "@/lib/api";
import type { ChatMessage, HealthResponse, RetrievalScope } from "@/lib/types";
import { Message } from "./Message";
import { VoiceInput } from "./VoiceInput";

const SAMPLE_QUERIES = [
  { text: "what is a corporation?", lang: "en" },
  { text: "निगम क्या है?", lang: "hi" },
  { text: "why did rachel carson write an obligation to endure", lang: "en" },
  { text: "प्रकाश संश्लेषण कैसे काम करता है?", lang: "hi" },
];

export function ChatPanel({ health }: { health: HealthResponse | null }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [scope, setScope] = useState<RetrievalScope>("all");
  const [topK, setTopK] = useState(10);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const addMessage = useCallback((message: ChatMessage) => {
    setMessages((current) => [...current, message]);
  }, []);

  const replaceMessage = useCallback((id: string, patch: Partial<ChatMessage>) => {
    setMessages((current) =>
      current.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    );
  }, []);

  const runQuery = useCallback(
    async (text: string, transcription?: ChatMessage["transcription"]) => {
      const trimmed = text.trim();
      if (!trimmed || busy) return;

      setBusy(true);
      setInput("");

      addMessage({
        id: `u-${Date.now()}`,
        role: "user",
        text: trimmed,
        transcription,
        timestamp: Date.now(),
      });

      const pendingId = `a-${Date.now()}`;
      addMessage({
        id: pendingId,
        role: "assistant",
        text: "",
        isLoading: true,
        timestamp: Date.now(),
      });

      try {
        const response = await api.query(trimmed, {
          scope,
          topK,
          language: transcription?.language_code ?? null,
        });
        replaceMessage(pendingId, { isLoading: false, response, text: response.answer });
      } catch (error) {
        replaceMessage(pendingId, {
          isLoading: false,
          error:
            error instanceof api.ApiError
              ? error.message
              : `Request failed: ${String(error)}`,
        });
      } finally {
        setBusy(false);
        inputRef.current?.focus();
      }
    },
    [addMessage, busy, replaceMessage, scope, topK],
  );

  const handleRecorded = useCallback(
    async (audio: Blob) => {
      if (busy) return;
      setBusy(true);

      const pendingId = `a-${Date.now()}`;
      addMessage({
        id: `u-${Date.now()}`,
        role: "user",
        text: "🎤 transcribing…",
        timestamp: Date.now(),
      });
      addMessage({
        id: pendingId,
        role: "assistant",
        text: "",
        isLoading: true,
        timestamp: Date.now(),
      });

      try {
        // One round trip: transcribe and answer server-side.
        const { transcription, response } = await api.voiceAsk(audio, { scope, topK });
        setMessages((current) =>
          current.map((m) => {
            if (m.text === "🎤 transcribing…" && m.role === "user") {
              return { ...m, text: transcription.text, transcription };
            }
            if (m.id === pendingId) {
              return { ...m, isLoading: false, response, text: response.answer };
            }
            return m;
          }),
        );
      } catch (error) {
        replaceMessage(pendingId, {
          isLoading: false,
          error:
            error instanceof api.ApiError
              ? error.message
              : `Voice request failed: ${String(error)}`,
        });
      } finally {
        setBusy(false);
      }
    },
    [addMessage, busy, replaceMessage, scope, topK],
  );

  const sttAvailable = health?.components.voice.stt_available ?? false;
  const indexEmpty = !health?.components.vector_store.target_collection_present;

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
        {messages.length === 0 ? (
          <EmptyState
            onPick={runQuery}
            indexEmpty={indexEmpty}
            sttAvailable={sttAvailable}
          />
        ) : (
          messages.map((message) => <Message key={message.id} message={message} />)
        )}
      </div>

      <div className="border-t border-[var(--border)] bg-[var(--bg-raised)] px-5 py-3">
        <div className="mb-2 flex items-center gap-3 text-[11px]">
          <label className="flex items-center gap-1.5 text-[var(--text-dim)]">
            scope
            <select
              value={scope}
              onChange={(event) => setScope(event.target.value as RetrievalScope)}
              className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-1.5 py-0.5 text-[var(--text-muted)] outline-none focus:border-[var(--accent)]"
            >
              <option value="all">all languages</option>
              <option value="same">same language</option>
              <option value="english">english only</option>
            </select>
          </label>

          <label className="flex items-center gap-1.5 text-[var(--text-dim)]">
            top-k
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(event) =>
                setTopK(Math.min(50, Math.max(1, Number(event.target.value) || 10)))
              }
              className="tabular w-12 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-1.5 py-0.5 text-[var(--text-muted)] outline-none focus:border-[var(--accent)]"
            />
          </label>

          <span className="flex-1" />

          {messages.length > 0 && (
            <button
              type="button"
              onClick={() => setMessages([])}
              className="text-[var(--text-dim)] transition-colors hover:text-[var(--text-muted)]"
            >
              clear
            </button>
          )}
        </div>

        <div className="flex items-end gap-2">
          <VoiceInput
            onRecorded={handleRecorded}
            disabled={busy}
            sttAvailable={sttAvailable}
          />

          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void runQuery(input);
              }
            }}
            rows={1}
            disabled={busy}
            placeholder="Ask a question in any of 14 languages…"
            className="indic max-h-32 min-h-[36px] flex-1 resize-none rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-3 py-2 text-[13px] text-[var(--text)] outline-none transition-colors placeholder:text-[var(--text-dim)] focus:border-[var(--accent)] disabled:opacity-50"
          />

          <button
            type="button"
            onClick={() => void runQuery(input)}
            disabled={busy || !input.trim()}
            className="flex h-9 items-center rounded-[var(--radius)] bg-[var(--accent)] px-4 text-[12.5px] font-medium text-[#0b0f14] transition-colors hover:bg-[var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-30"
          >
            {busy ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

function EmptyState({
  onPick,
  indexEmpty,
  sttAvailable,
}: {
  onPick: (text: string) => void;
  indexEmpty: boolean;
  sttAvailable: boolean;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-5 px-6 text-center">
      <div>
        <h2 className="text-[17px] font-medium text-[var(--text)]">
          Cross-lingual voice retrieval
        </h2>
        <p className="mt-1.5 max-w-md text-[12.5px] leading-relaxed text-[var(--text-dim)]">
          Ask in any of 14 Indic languages or English. Queries retrieve across
          every language in the index, and answers come back in the language you
          asked in — grounded in retrieved passages, with every pipeline stage
          measurable below.
        </p>
      </div>

      {(indexEmpty || !sttAvailable) && (
        <div className="w-full max-w-md space-y-1 rounded-[var(--radius)] border border-[#d2992244] bg-[#d2992211] px-3 py-2 text-left">
          {indexEmpty && (
            <p className="text-[11.5px] text-[var(--yellow)]">
              No vector index found — run{" "}
              <code className="tabular">python scripts/ingest.py</code> to build one.
            </p>
          )}
          {!sttAvailable && (
            <p className="text-[11.5px] text-[var(--yellow)]">
              Voice disabled — add <code className="tabular">SARVAM_API_KEY</code> to{" "}
              <code className="tabular">backend/.env</code>.
            </p>
          )}
        </div>
      )}

      <div className="grid w-full max-w-md grid-cols-2 gap-1.5">
        {SAMPLE_QUERIES.map((sample) => (
          <button
            key={sample.text}
            type="button"
            onClick={() => onPick(sample.text)}
            className={`rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] px-2.5 py-2 text-left text-[12px] text-[var(--text-muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text)] ${
              sample.lang !== "en" ? "indic" : ""
            }`}
          >
            {sample.text}
          </button>
        ))}
      </div>
    </div>
  );
}
