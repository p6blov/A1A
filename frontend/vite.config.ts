import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend calls /api/* and Vite proxies those requests to the FastAPI
// backend during development, so there is no CORS or hard-coded host to manage.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
