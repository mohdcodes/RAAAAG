import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAAAAAG !!",
  description:
    "Voice retrieval over MSMARCO-XI across 14 Indic languages. " +
    "Developed by BrBik for Hacker House Goa.",
};

export const viewport: Viewport = {
  themeColor: "#0a3327",
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
