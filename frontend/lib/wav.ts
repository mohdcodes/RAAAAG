/**
 * WebM/Opus -> WAV conversion.
 *
 * MediaRecorder produces WebM/Opus in Chrome and Firefox, but Sarvam's STT
 * accepts only MP3 and WAV. Rather than ship an MP3 encoder, decode the
 * recording with the Web Audio API — which handles whatever the browser
 * produced — and re-encode as 16-bit PCM WAV, a format simple enough to write
 * by hand in a few dozen lines.
 *
 * Output is mono at 16 kHz: speech models expect that rate, and downsampling
 * cuts the upload roughly 6x versus 48 kHz stereo with no accuracy cost.
 */

const TARGET_SAMPLE_RATE = 16_000;

export async function toWav(blob: Blob): Promise<Blob> {
  const buffer = await blob.arrayBuffer();

  // Safari still needs the webkit-prefixed constructor.
  const Ctor: typeof AudioContext =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext;

  const context = new Ctor();
  try {
    const decoded = await context.decodeAudioData(buffer);
    const mono = downmix(decoded);
    const resampled =
      decoded.sampleRate === TARGET_SAMPLE_RATE
        ? mono
        : resample(mono, decoded.sampleRate, TARGET_SAMPLE_RATE);
    return encodeWav(resampled, TARGET_SAMPLE_RATE);
  } finally {
    void context.close();
  }
}

/** Average all channels into one. Speech STT gains nothing from stereo. */
function downmix(audio: AudioBuffer): Float32Array {
  if (audio.numberOfChannels === 1) return audio.getChannelData(0);

  const out = new Float32Array(audio.length);
  for (let ch = 0; ch < audio.numberOfChannels; ch += 1) {
    const data = audio.getChannelData(ch);
    for (let i = 0; i < audio.length; i += 1) out[i] += data[i];
  }
  for (let i = 0; i < out.length; i += 1) out[i] /= audio.numberOfChannels;
  return out;
}

/**
 * Linear-interpolation resampling.
 *
 * Not as clean as a windowed-sinc filter, but the aliasing it introduces sits
 * well above the frequencies that carry speech, and it costs a fraction of the
 * time — which matters when this runs between the user releasing the button
 * and the request going out.
 */
function resample(input: Float32Array, from: number, to: number): Float32Array {
  const ratio = from / to;
  const length = Math.round(input.length / ratio);
  const out = new Float32Array(length);

  for (let i = 0; i < length; i += 1) {
    const position = i * ratio;
    const index = Math.floor(position);
    const fraction = position - index;
    const a = input[index] ?? 0;
    const b = input[index + 1] ?? a;
    out[i] = a + (b - a) * fraction;
  }
  return out;
}

/** 16-bit PCM WAV: 44-byte RIFF header followed by samples. */
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  const writeText = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  const dataBytes = samples.length * bytesPerSample;

  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true); // file size minus the first 8 bytes
  writeText(8, "WAVE");

  writeText(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // 1 = PCM, uncompressed
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample

  writeText(36, "data");
  view.setUint32(40, dataBytes, true);

  // Float [-1, 1] -> signed 16-bit. Clamping first prevents a sample slightly
  // over 1.0 from wrapping to a loud negative click.
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}
