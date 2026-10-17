#!/usr/bin/env bash
# Start both the FastAPI backend and the Vite frontend for local development.
#
#   ./start.sh
#
# Backend:  http://localhost:8000  (API at /api/*)
# Frontend: http://localhost:5173  (proxies /api -> backend)
#
# Ctrl+C stops both. Neither server persists chat data to disk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- Python / backend deps ---
if [ ! -d ".venv" ]; then
  echo "==> Creating virtualenv (.venv)"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> Installing backend dependencies"
pip install -q -r requirements.txt

# --- Frontend deps ---
if [ ! -d "frontend/node_modules" ]; then
  echo "==> Installing frontend dependencies"
  (cd frontend && npm install)
fi

# --- Launch both, clean up on exit ---
pids=()
cleanup() {
  echo ""
  echo "==> Shutting down"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "==> Starting backend on http://localhost:8000"
# --no-access-log: uvicorn's access log records the client IP on every request;
# disabling it keeps user IPs out of stdout/logs (duck.ai-style privacy).
uvicorn app.api.server:app --port 8000 --no-access-log &
pids+=($!)

echo "==> Starting frontend on http://localhost:5173"
(cd frontend && npm run dev) &
pids+=($!)

echo "==> Both servers running. Open http://localhost:5173  (Ctrl+C to stop)"
wait
