"""Single-shot CLI entrypoint.

Usage:
    python -m app.main "Your prompt here"
    python -m app.main "Your prompt" --task research

Runs the same graph as the server for a one-message conversation and prints a
single JSON object (per-agent responses, summarizer run, final answer, errors).
No data is persisted to disk; logging is intentionally minimal.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

from app.config.settings import get_settings
from app.graph.builder import build_graph
from app.utils import telemetry
from app.utils.redact import redact


def _setup_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


async def run(prompt: str, task: str) -> dict:
    # Redaction happens at the entry boundary (see app/api/server.py for the same
    # pattern). For the single-prompt CLI, that is just the prompt itself.
    if get_settings().enable_redaction:
        prompt = redact(prompt)
    graph = build_graph()
    rid = telemetry.new_request_id()
    telemetry.log_start(rid, task)

    initial = {
        "messages": [{"role": "user", "content": prompt}],
        "task": task,
        "responses": [],
        "errors": [],
        "meta": {"request_id": rid},
    }
    started = time.perf_counter()
    final = await graph.ainvoke(initial)
    wall_ms = int((time.perf_counter() - started) * 1000)
    summarizer_run = final.get("summarizer_run")
    # The aggregate is the per-query cost line, so the synthesis call counts too.
    all_calls = (final.get("responses") or []) + ([summarizer_run] if summarizer_run else [])
    telemetry.log_aggregate(rid, all_calls, wall_ms)
    return {
        "task": task,
        "responses": final.get("responses", []),
        "summarizer": final.get("summarizer_run"),
        "final_answer": final.get("final_answer", ""),
        "errors": final.get("errors", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent orchestrator CLI")
    parser.add_argument("prompt", help="User prompt to send to the model")
    parser.add_argument("--task", default="default", help="Task type from tasks.yaml")
    args = parser.parse_args()

    settings = get_settings()
    _setup_logging(quiet=settings.quiet_logs)

    result = asyncio.run(run(args.prompt, args.task))
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
