import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits a self-contained server bundle so the Docker runtime stage does not
  // need the full node_modules tree.
  output: "standalone",
  reactStrictMode: true,
};

export default nextConfig;
