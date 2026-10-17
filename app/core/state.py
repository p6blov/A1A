"""Typed state shared across the LangGraph.

The `responses` field uses an `add`-reducer so that when M2 introduces parallel
branches, each branch can append its result and LangGraph will merge them
automatically — without changing this file.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ChatMessage(TypedDict):
    """One conversation turn, provider-agnostic. Each provider adapter maps the
    role to its vendor's wire format (e.g. "assistant" → Gemini's "model")."""

    role: str  # "user" | "assistant"
    content: str


class ProviderResponse(TypedDict, total=False):
    provider: str  # the *agent* name from the task config, not the vendor
    model: str
    content: str
    error: str | None
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None


class OrchestratorState(TypedDict, total=False):
    # Inputs. `messages` is the structured conversation; PII redaction of the
    # newest user message happens at the entry boundary (API/CLI) before the
    # graph is invoked.
    messages: list[ChatMessage]
    task: str                     # e.g. "default", "research" — looked up in tasks.yaml

    # Per-request line-up override (the UI's agent panel). When set, these
    # replace the task's agents from tasks.yaml for this run; `task` still names
    # the fallback. Entries mirror `AgentSpec` fields: name/provider/model/system.
    # Validated at the entry boundary (app/api/server.py) — the graph trusts them.
    agent_specs: list[dict[str, Any]] | None
    # Same idea for the synthesis step: {provider, model, system}.
    summarizer_cfg: dict[str, Any] | None

    # Fan-out outputs (one entry per agent). Reducer = list concat.
    responses: Annotated[list[ProviderResponse], operator.add]

    # M3 synthesis. `summary` is set when the summarize node ran and produced an
    # answer; `summarizer_run` records that call (provider, model, latency,
    # tokens) so the API can surface it in the trace alongside the agent runs.
    summary: str | None
    summarizer_run: ProviderResponse | None

    # Final answer: the synthesis when available, else the first successful
    # response in task-config order.
    final_answer: str | None

    # Diagnostics
    errors: Annotated[list[str], operator.add]
    meta: dict[str, Any]
