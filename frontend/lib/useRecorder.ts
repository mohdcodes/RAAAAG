"use client";

/**
 * Push-to-talk recorder with live waveform.
 *
 * MediaRecorder gives us the encoded audio; a parallel AnalyserNode gives us
 * amplitude data for the visualiser. They read the same MediaStream, so the
 * waveform reflects exactly what is being recorded.
 *
 * Every acquired resource (stream tracks, AudioContext, rAF loop, timer) is
 * released on stop and on unmount — a leaked microphone track leaves the
 * browser's recording indicator on, which users reasonably find alarming.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface RecorderState {
  isRecording: boolean;
  /** Normalized 0-1 amplitude samples for the visualiser. */
  levels: number[];
  durationMs: number;
  error: string | null;
  isSupported: boolean;
}

const BAR_COUNT = 48;
const MAX_DURATION_MS = 30_000;

export function useRecorder(onComplete: (audio: Blob, durationMs: number) => void) {
  const [state, setState] = useState<RecorderState>({
    isRecording: false,
    levels: new Array(BAR_COUNT).fill(0),
    durationMs: 0,
    error: null,
    isSupported: true,
  });

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number>(0);
  const levelsRef = useRef<number[]>(new Array(BAR_COUNT).fill(0));
  const onCompleteRef = useRef(onComplete);

  // Keep the callback fresh without making it a dependency of stop().
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    const supported =
      typeof window !== "undefined" &&
      typeof navigator !== "undefined" &&
      !!navigator.mediaDevices?.getUserMedia &&
      typeof MediaRecorder !== "undefined";
    if (!supported) {
      setState((s) => ({
        ...s,
        isSupported: false,
        error: "This browser does not support audio recording.",
      }));
    }
  }, []);

  const cleanup = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (audioContextRef.current?.state !== "closed") {
      void audioContextRef.current?.close();
    }
    audioContextRef.current = null;
    analyserRef.current = null;
    recorderRef.current = null;
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const tick = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const buffer = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(buffer);

    // RMS of the waveform, scaled for visual range. Byte data is centred on
    // 128, so subtract that before squaring.
    let sumSquares = 0;
    for (let i = 0; i < buffer.length; i += 1) {
      const deviation = (buffer[i] - 128) / 128;
      sumSquares += deviation * deviation;
    }
    const rms = Math.sqrt(sumSquares / buffer.length);
    const level = Math.min(1, rms * 3.2);

    levelsRef.current = [...levelsRef.current.slice(1), level];
    const elapsed = performance.now() - startedAtRef.current;

    setState((s) => ({ ...s, levels: levelsRef.current, durationMs: elapsed }));

    if (elapsed >= MAX_DURATION_MS) {
      stop();
      return;
    }
    rafRef.current = requestAnimationFrame(tick);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = useCallback(async () => {
    if (!state.isSupported || recorderRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const context = new AudioContext();
      audioContextRef.current = context;
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.7;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Prefer Opus, but fall back — Safari does not support webm/opus.
      const mimeType = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
      ].find((type) => MediaRecorder.isTypeSupported(type));

      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType, audioBitsPerSecond: 64_000 } : undefined,
      );
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };

      recorder.onstop = () => {
        const durationMs = performance.now() - startedAtRef.current;
        const blob = new Blob(chunksRef.current, {
          type: mimeType ?? "audio/webm",
        });
        cleanup();
        levelsRef.current = new Array(BAR_COUNT).fill(0);
        setState({
          isRecording: false,
          levels: levelsRef.current,
          durationMs: 0,
          error: null,
          isSupported: true,
        });
        // Ignore blips too short to contain speech.
        if (blob.size > 1000 && durationMs > 300) {
          onCompleteRef.current(blob, durationMs);
        }
      };

      recorderRef.current = recorder;
      startedAtRef.current = performance.now();
      recorder.start(100);

      setState((s) => ({ ...s, isRecording: true, error: null, durationMs: 0 }));
      rafRef.current = requestAnimationFrame(tick);
    } catch (error) {
      const message =
        error instanceof DOMException && error.name === "NotAllowedError"
          ? "Microphone permission denied. Allow access and try again."
          : `Could not start recording: ${String(error)}`;
      cleanup();
      setState((s) => ({ ...s, isRecording: false, error: message }));
    }
  }, [cleanup, state.isSupported, tick]);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop(); // onstop performs cleanup and fires the callback
    } else {
      cleanup();
      setState((s) => ({ ...s, isRecording: false }));
    }
  }, [cleanup]);

  const cancel = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder) recorder.onstop = null; // suppress the completion callback
    if (recorder && recorder.state !== "inactive") recorder.stop();
    cleanup();
    levelsRef.current = new Array(BAR_COUNT).fill(0);
    setState({
      isRecording: false,
      levels: levelsRef.current,
      durationMs: 0,
      error: null,
      isSupported: true,
    });
  }, [cleanup]);

  return { ...state, start, stop, cancel, maxDurationMs: MAX_DURATION_MS };
}
