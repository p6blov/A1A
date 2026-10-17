"""Query-by-query request logging, shared by the API server and the CLI.

Design for multi-LLM fan-out (M2): every log line is tagged with a per-query
correlation id so that when one query expands into N parallel provider calls,
their interleaved lines can still be grouped by eye or by grep.

Ownership of the lines is split to match where the data actually lives:
  * the entry boundary (server/CLI) owns the start + aggregate lines because it
    is the only place that measures true wall-clock around the whole graph
    (for parallel branches wall-clock is the *max* latency, not the sum);
  * the provider node owns the per-provider line, so each future fan-out branch
    logs its own result as it resolves.

Everything here is metadata only — no IPs, no prompt/response content — matching
the privacy posture of the rest of the app (uvicorn --no-access-log,
strip_client_ip). Add content logging only behind an explicit dev-only flag.
"""
from __future__ import annotations

import logging
from uuid import uuid4

logger = logging.getLogger("app.telemetry")
logger.setLevel(logging.INFO)


def new_request_id() -> str:
    """Short, readable correlation id, e.g. "q4f2a1"."""
    return "q" + uuid4().hex[:5]


def attach_to_uvicorn() -> None:
    """Route telemetry lines through uvicorn's stdout handler.

    uvicorn does not configure the root logger, so a freshly-created named logger
    would have nowhere to emit. Borrow uvicorn's handler so request lines appear
    alongside the server's own output. Idempotent.
    """
    if logger.handlers:
        return
    for h in logging.getLogger("uvicorn").handlers:
        logger.addHandler(h)
    logger.propagate = False


def _tokens(n: int | None) -> str:
    return str(n) if n is not None else "n/a"


def _ms(latency: int | None) -> str:
    return f"{latency} ms" if latency is not None else "n/a ms"


# Field separator for the metadata segments on each line. Kept as a single
# constant so every log line stays visually aligned and is easy to split when
# parsing the terminal output by eye or with `cut -d'|'`.
_SEP = "  |  "


def log_start(rid: str, task: str) -> None:
    logger.info("[%s]%squery%stask=%s", rid, _SEP, _SEP, task)


def log_provider(rid: str, resp: dict) -> None:
    """One line per provider response. Reads a ProviderResponse-shaped dict."""
    who = f"{resp.get('provider')}/{resp.get('model')}"
    if resp.get("error"):
        fields = [f"[{rid}]", who, "ERR", resp["error"], _ms(resp.get("latency_ms"))]
    else:
        fields = [
            f"[{rid}]",
            who,
            "OK",
            f"in {_tokens(resp.get('prompt_tokens'))}",
            f"out {_tokens(resp.get('completion_tokens'))}",
            _ms(resp.get("latency_ms")),
        ]
    logger.info(_SEP.join(fields))


def log_aggregate(rid: str, responses: list[dict], wall_ms: int) -> None:
    """Per-query summary: success/failure counts, wall-clock, total tokens.

    wall_ms is measured around the whole graph, so for parallel fan-out it
    reflects the slowest branch (+ overhead), not the sum of branch latencies.
    """
    ok = sum(1 for r in responses if not r.get("error"))
    err = len(responses) - ok
    total_tokens = sum(
        (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
        for r in responses
    )
    fields = [
        f"[{rid}]",
        "done",
        f"{ok} ok / {err} err",
        f"wall {wall_ms} ms",
        f"Σ {total_tokens} tok",
    ]
    logger.info(_SEP.join(fields))
