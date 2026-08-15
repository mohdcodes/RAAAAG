import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VoiceRAG — MSMARCO-XI",
  description:
    "Voice-enabled cross-lingual retrieval-augmented generation over " +
    "ai4bharat/MSMARCO-XI, with per-stage latency instrumentation and " +
    "four-layer guardrails.",
};

export const viewport: Viewport = {
  themeColor: "#0b0f14",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
