"use client";

/**
 * Chat surface. Transcript scrolls, composer stays pinned to the bottom.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "@/lib/api";
import type { ChatMessage, HealthResponse } from "@/lib/types";
import { Result } from "./Result";
import { VoiceInput } from "./VoiceInput";

const EXAMPLES = [
  "what is a corporation?",
  "how does photosynthesis work",
  "निगम क्या है?",
  "मधुमेह का कारण क्या है?",
];

export function Chat({ health }: { health: HealthResponse | null }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const scroller = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scroller.current?.scrollTo({
      top: scroller.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const ask = useCallback(
    async (text: string, transcription?: ChatMessage["transcription"]) => {
      const q = text.trim();
      if (!q || busy) return;

      setBusy(true);
      setInput("");
      const pendingId = `a${Date.now()}`;

      setMessages((m) => [
        ...m,
        {
          id: `u${Date.now()}`,
          role: "user",
          text: q,
          transcription,
          timestamp: Date.now(),
        },
        {
          id: pendingId,
          role: "assistant",
          text: "",
          isLoading: true,
          timestamp: Date.now(),
        },
      ]);

      try {
        const response = await api.query(q, {
          language: transcription?.language_code ?? null,
        });
        setMessages((m) =>
          m.map((x) =>
            x.id === pendingId
              ? { ...x, isLoading: false, response, text: response.answer }
              : x,
          ),
        );
      } catch (error) {
        setMessages((m) =>
          m.map((x) =>
            x.id === pendingId
              ? {
                  ...x,
                  isLoading: false,
                  error:
                    error instanceof api.ApiError ? error.message : String(error),
                }
              : x,
          ),
        );
      } finally {
        setBusy(false);
        box.current?.focus();
      }
    },
    [busy],
  );

  const onRecorded = useCallback(
    async (audio: Blob) => {
      if (busy) return;
      setBusy(true);
      const userId = `u${Date.now()}`;
      const pendingId = `a${Date.now()}`;

      setMessages((m) => [
        ...m,
        { id: userId, role: "user", text: "…", timestamp: Date.now() },
        {
          id: pendingId,
          role: "assistant",
          text: "",
          isLoading: true,
          timestamp: Date.now(),
        },
      ]);

      try {
        const { transcription, response } = await api.voiceAsk(audio);
        setMessages((m) =>
          m.map((x) => {
            if (x.id === userId)
              return { ...x, text: transcription.text, transcription };
            if (x.id === pendingId)
              return { ...x, isLoading: false, response, text: response.answer };
            return x;
          }),
        );
      } catch (error) {
        setMessages((m) =>
          m.map((x) =>
            x.id === pendingId
              ? {
                  ...x,
                  isLoading: false,
                  error:
                    error instanceof api.ApiError ? error.message : String(error),
                }
              : x,
          ),
        );
      } finally {
        setBusy(false);
      }
    },
    [busy],
  );

  const stt = health?.components.voice.stt_available ?? false;
  const ready = Boolean(health?.components.vector_store.target_collection_present);

  return (
    <div className="flex h-full flex-col">
      <div ref={scroller} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-5 py-6">
          {messages.length === 0 ? (
            <Start onPick={ask} ready={ready} />
          ) : (
            <div className="space-y-5">
              {messages.map((m) =>
                m.role === "user" ? (
                  <Question key={m.id} message={m} />
                ) : (
                  <Reply key={m.id} message={m} />
                ),
              )}
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-[var(--line)]">
        <div className="mx-auto flex max-w-3xl items-end gap-2 px-5 py-3">
          <VoiceInput onRecorded={onRecorded} disabled={busy} sttAvailable={stt} />

          <textarea
            ref={box}
            rows={1}
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void ask(input);
              }
            }}
            placeholder="Ask anything"
            className="indic max-h-40 min-h-[40px] flex-1 resize-none rounded-[var(--r)] border border-[var(--line)] bg-[var(--sunken)] px-3.5 py-2.5 text-[14px] text-[var(--cream)] outline-none transition-colors placeholder:text-[var(--cream-faint)] focus:border-[var(--gold)] disabled:opacity-60"
          />

          <button
            type="button"
            onClick={() => void ask(input)}
            disabled={busy || !input.trim()}
            className="h-[40px] rounded-[var(--r)] bg-[var(--gold)] px-5 text-[13px] font-medium text-[var(--forest-deep)] transition-opacity disabled:opacity-25"
          >
            {busy ? "…" : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Question({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-[var(--r)] rounded-br-[3px] border border-[var(--line-firm)] bg-[var(--forest)] px-3.5 py-2">
        <p className="indic text-[14px] text-[var(--cream)]">{message.text}</p>
        {message.transcription && (
          <p className="eyebrow mt-1">
            voice · {message.transcription.language_code} ·{" "}
            {message.transcription.duration_ms.toFixed(0)}ms
          </p>
        )}
      </div>
    </div>
  );
}

function Reply({ message }: { message: ChatMessage }) {
  if (message.isLoading) {
    return (
      <div className="space-y-2 rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)] p-3.5">
        <div className="skeleton h-3 w-2/3 rounded" />
        <div className="skeleton h-3 w-1/3 rounded" />
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="rounded-[var(--r)] border border-[var(--bad)] px-3.5 py-2.5"
           style={{ background: "rgba(232,85,127,0.1)" }}>
        <p className="text-[12.5px] text-[var(--bad)]">{message.error}</p>
      </div>
    );
  }

  return message.response ? <Result response={message.response} /> : null;
}

function Start({ onPick, ready }: { onPick: (q: string) => void; ready: boolean }) {
  return (
    <div className="flex flex-col items-center gap-7 pt-20">
      <div className="text-center">
        <h2
          className="font-bold tracking-tight text-[var(--cream)]"
          style={{ fontSize: "clamp(28px, 4.4vw, 46px)", letterSpacing: "-0.02em" }}
        >
          RAAAAAG&nbsp;<span className="text-[var(--gold)]">!!</span>
        </h2>
        <p className="eyebrow mt-2">voice retrieval · 14 languages</p>
      </div>

      {!ready && (
        <p className="num rounded-[var(--r-pill)] border border-[var(--warn)] px-3 py-1 text-[11px] text-[var(--warn)]">
          index not built
        </p>
      )}

      <div className="grid w-full max-w-lg grid-cols-2 gap-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="indic rounded-[var(--r)] border border-[var(--line)] bg-[var(--panel)] px-3.5 py-3 text-left text-[13px] text-[var(--cream-soft)] transition-colors hover:border-[var(--gold)] hover:text-[var(--cream)]"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
