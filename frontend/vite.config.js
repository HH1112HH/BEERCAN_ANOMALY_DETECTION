import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const proxyPaths = [
  "/status",
  "/history",
  "/video",
  "/control",
  "/conveyor",
  "/trigger",
  "/reset",
];

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_DEV_API_TARGET || "http://127.0.0.1:5004";
  const proxy = Object.fromEntries(
    proxyPaths.map((path) => [path, { target, changeOrigin: true }]),
  );

  return {
    plugins: [react(), tailwindcss()],
    server: { host: "0.0.0.0", port: 5173, strictPort: true, proxy },
    preview: { host: "0.0.0.0", port: 4173, strictPort: true },
  };
});
