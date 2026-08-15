"use client";

/**
 * Push-to-talk control with a live waveform.
 *
 * Recording is a modal state, so it takes over the composer entirely rather
 * than sitting as a small icon — the user needs unambiguous feedback that the
 * microphone is live, and an obvious way out.
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
  const progress = recorder.durationMs / recorder.maxDurationMs;

  if (!recorder.isSupported) {
    return (
      <button
        type="button"
        disabled
        title="This browser does not support audio recording"
        className="flex h-9 w-9 items-center justify-center rounded-[var(--radius)] border border-[var(--border)] text-[var(--text-dim)] opacity-50"
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
              ? "Hold to speak your question"
              : "Voice input needs SARVAM_API_KEY in backend/.env"
          }
          className="flex h-9 w-9 items-center justify-center rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-input)] text-[var(--text-muted)] transition-all hover:border-[var(--accent)] hover:text-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-[var(--border)] disabled:hover:text-[var(--text-muted)]"
        >
          <MicIcon />
        </button>
        {recorder.error && (
          <p className="absolute bottom-full left-0 mb-1.5 w-64 rounded-[var(--radius)] border border-[#f8514944] bg-[#f8514911] px-2 py-1 text-[11px] text-[var(--red)]">
            {recorder.error}
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-1 items-center gap-3 rounded-[var(--radius)] border border-[var(--accent-dim)] bg-[var(--bg-input)] px-3 py-2">
      <span className="animate-pulse-record h-2.5 w-2.5 shrink-0 rounded-full bg-[var(--red)]" />

      <div className="flex h-8 flex-1 items-center gap-[2px]">
        {recorder.levels.map((level, index) => (
          <div
            key={index}
            className="flex-1 rounded-full transition-[height] duration-75"
            style={{
              height: `${Math.max(3, level * 100)}%`,
              background:
                level > 0.55
                  ? "var(--accent)"
                  : level > 0.2
                    ? "var(--accent-dim)"
                    : "var(--border-strong)",
            }}
          />
        ))}
      </div>

      <span className="tabular shrink-0 text-[12px] text-[var(--text-muted)]">
        {seconds}s
        {progress > 0.8 && (
          <span className="ml-1 text-[var(--yellow)]">
            / {(recorder.maxDurationMs / 1000).toFixed(0)}s
          </span>
        )}
      </span>

      <button
        type="button"
        onClick={recorder.cancel}
        title="Discard recording"
        className="shrink-0 rounded-[var(--radius)] px-2 py-1 text-[11px] text-[var(--text-dim)] transition-colors hover:bg-[var(--bg-overlay)] hover:text-[var(--red)]"
      >
        Cancel
      </button>

      <button
        type="button"
        onClick={recorder.stop}
        className="shrink-0 rounded-[var(--radius)] bg-[var(--accent)] px-3 py-1 text-[11px] font-medium text-[#0b0f14] transition-colors hover:bg-[var(--accent-hover)]"
      >
        Send
      </button>
    </div>
  );
}

function MicIcon() {
  return (
    <svg
      width="15"
      height="15"
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
