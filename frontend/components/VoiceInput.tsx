"use client";

/**
 * Push-to-talk with a live waveform.
 *
 * Recording is a modal state, so it takes over the composer rather than
 * sitting as a small icon — a live microphone needs unambiguous feedback and
 * an obvious way out.
 *
 * Speech is Sarvam end to end: saarika for transcription, bulbul for playback.
 */

import { useRecorder } from "@/lib/useRecorder";

export function VoiceInput({
  onRecorded,
  disabled = false,
  sttAvailable = true,
}: {
  onRecorded: (audio: Blob, durationMs: number) => void;
  disabled?: boolean;
  sttAvailable?: boolean;
}) {
  const recorder = useRecorder(onRecorded);
  const seconds = (recorder.durationMs / 1000).toFixed(1);
  const nearLimit = recorder.durationMs / recorder.maxDurationMs > 0.8;

  if (!recorder.isSupported) {
    return (
      <button
        type="button"
        disabled
        title="This browser cannot record audio"
        className="flex h-10 w-10 items-center justify-center rounded-[var(--r)] border border-[var(--line)] text-[var(--cream-faint)] opacity-50"
      >
        <MicIcon />
      </button>
    );
  }

  if (!recorder.isRecording) {
    return (
      <div className="relative">
        <button
          type="button"
          onClick={recorder.start}
          disabled={disabled || !sttAvailable}
          title={
            sttAvailable
              ? "Speak your question"
              : "Voice needs SARVAM_API_KEY in backend/.env"
          }
          className="flex h-10 w-10 items-center justify-center rounded-[var(--r)] border border-[var(--line)] bg-[var(--sunken)] text-[var(--cream-soft)] transition-colors hover:border-[var(--gold)] hover:text-[var(--gold)] disabled:cursor-not-allowed disabled:opacity-35"
        >
          <MicIcon />
        </button>
        {recorder.error && (
          <p
            className="absolute bottom-full left-0 mb-2 w-60 rounded-[var(--r-sm)] border border-[var(--bad)] px-2 py-1 text-[11px] text-[var(--bad)]"
            style={{ background: "rgba(232,85,127,0.12)" }}
          >
            {recorder.error}
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-1 items-center gap-3 rounded-[var(--r)] border border-[var(--gold)] bg-[var(--sunken)] px-3 py-2">
      <span className="pulse h-2.5 w-2.5 shrink-0 rounded-full bg-[var(--pink)]" />

      <div className="flex h-8 flex-1 items-center gap-[2px]">
        {recorder.levels.map((level, i) => (
          <div
            key={i}
            className="flex-1 rounded-full transition-[height] duration-75"
            style={{
              height: `${Math.max(3, level * 100)}%`,
              background:
                level > 0.55
                  ? "var(--gold)"
                  : level > 0.2
                    ? "rgba(227,178,60,0.5)"
                    : "var(--line-firm)",
            }}
          />
        ))}
      </div>

      <span className="num shrink-0 text-[12px] text-[var(--cream-soft)]">
        {seconds}s
        {nearLimit && (
          <span className="ml-1 text-[var(--warn)]">
            /{(recorder.maxDurationMs / 1000).toFixed(0)}
          </span>
        )}
      </span>

      <button
        type="button"
        onClick={recorder.cancel}
        className="shrink-0 rounded-[var(--r-sm)] px-2 py-1 text-[11px] text-[var(--cream-faint)] transition-colors hover:text-[var(--bad)]"
      >
        Cancel
      </button>

      <button
        type="button"
        onClick={recorder.stop}
        className="shrink-0 rounded-[var(--r-pill)] bg-[var(--gold)] px-3.5 py-1 text-[11.5px] font-medium text-[var(--forest-deep)]"
      >
        Send
      </button>
    </div>
  );
}

function MicIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
    </svg>
  );
}
