# Multi-Agent AI Orchestrator

**Proprietary and confidential. All rights reserved.**
This source code is not licensed for use, copying, modification, or distribution.

A web application that puts one question to several AI models at the same time and
returns a single merged answer. Python/FastAPI backend, React frontend.
Conversation history stays in the browser; the backend keeps no state.

---

## Requirements

| | Version | Notes |
|---|---|---|
| Python | 3.12 or newer | |
| Node.js | 20 or newer | ships with npm |
| OpenRouter API key | — | free tier is sufficient — [openrouter.ai/keys](https://openrouter.ai/keys) |

---

## Setup

```bash
cp .env.example .env
```

Open `.env` and set `OPENROUTER_API_KEY`. That is the only required value.

---

## Running

```bash
./start.sh
```

The first run creates the Python virtual environment, installs backend
dependencies, and runs `npm install`. It then starts the backend on
`http://localhost:8000` and the frontend on `http://localhost:5173`.
Open the frontend URL in a browser. `Ctrl+C` stops both.

A single-prompt command-line entry point runs the same pipeline without the server:

```bash
python -m app.main "your prompt here"
```

---

## Tests

```bash
pytest -q                  # backend
cd frontend && npm test    # frontend
```

Both suites mock all network calls, so no API key is needed to run them.
Please run both before opening a pull request.

---

## Layout

| Path | Contents |
|---|---|
| `app/` | FastAPI backend and orchestration |
| `frontend/` | React client (Vite) |
| `tests/` | Backend test suite |

---

## Configuration

Every setting is read from `.env`. See `.env.example` for the full list and the
built-in defaults; only `OPENROUTER_API_KEY` has to be set.

---

## Contributing

- Run both test suites before opening a pull request.
- Never commit `.env`, an API key, or any other credential.
- Build output (`frontend/dist/`), virtual environments, and local editor or
  tooling directories stay out of the repository — see `.gitignore`.
- Design documents and internal notes are maintained outside this repository.
