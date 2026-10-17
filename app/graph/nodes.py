"""LangGraph node implementations.

Each node is a pure-ish function: it receives state and returns a partial state
dict that LangGraph merges in. The provider fan-out uses the Send API: a router
after `redact` emits one `Send("call_provider", branch)` per configured agent,
LangGraph runs the branches in parallel, and the list-concat reducer on
`state["responses"]` merges the results. After the fan-in, `summarize`
synthesizes the responses into one answer (when the task configures it) and
`finalize` picks the reply.

The conversation flows through as structured `ChatMessage` turns; every agent
receives the same messages plus its own system prompt (role). Agent identity is
the spec's `name`, not the vendor, so one provider can serve several roles in
one run.

The line-up itself comes from the task config, unless the request carried its
own (the UI's agent panel) — see `_agents_for`, the single place that decides.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from langgraph.config import get_stream_writer
from langgraph.types import Send

from app.config.tasks import AgentSpec, load_task_agents, load_task_config
from app.core.state import ChatMessage, OrchestratorState, ProviderResponse
from app.providers import registry
from app.providers.base import ProviderResult
from app.utils.telemetry import log_provider


def _stream_event(event: dict[str, Any]) -> None:
    """Push a custom stream event for SSE consumers (see /api/chat/stream).

    Under `astream(..., stream_mode=["custom", ...])` the event reaches the
    stream as it happens — that is what lets agent chips resolve one at a time.
    Under plain `ainvoke` (the JSON endpoint, the CLI) the writer is a no-op,
    so nodes can emit unconditionally.
    """
    try:
        get_stream_writer()(event)
    except Exception:
        pass


# Token deltas arrive far faster than a UI can use them, and every parallel
# branch produces its own. Batching them into ~50ms / ~80-character frames keeps
# the text visibly live while keeping the SSE frame count sane.
_DELTA_FLUSH_CHARS = 80
_DELTA_FLUSH_SECONDS = 0.05


def _delta_emitter(
    event_type: str, **fields: Any
) -> tuple[Callable[[str], None], Callable[[], None]]:
    """Build a coalescing `(on_delta, flush)` pair for one streaming call.

    `on_delta` buffers text and streams a `{"type": event_type, **fields,
    "text": ...}` event once the buffer is worth sending; `flush` must be called
    when the call ends so the tail is never left in the buffer.
    """
    buf: list[str] = []
    last = time.monotonic()

    def flush() -> None:
        nonlocal last
        if not buf:
            return
        _stream_event({"type": event_type, **fields, "text": "".join(buf)})
        buf.clear()
        last = time.monotonic()

    def on_delta(text: str) -> None:
        buf.append(text)
        if (
            sum(len(t) for t in buf) >= _DELTA_FLUSH_CHARS
            or time.monotonic() - last >= _DELTA_FLUSH_SECONDS
        ):
            flush()

    return on_delta, flush


def resolve_task_agents(task: str) -> list[AgentSpec]:
    """The agents that would actually run for `task`.

    Agents whose provider lacks credentials are skipped so an unconfigured
    optional key degrades to fewer branches, not failed requests. If *nothing*
    is configured the full list is kept — the branches then soft-fail with an
    error that names the missing key, which is more diagnosable than an empty
    response.

    Shared with the API's /api/tasks so the UI shows the same line-up.
    """
    specs = load_task_agents(task)
    configured = [s for s in specs if registry.is_configured(s.provider)]
    return configured or specs


def _agents_for(state: OrchestratorState) -> list[AgentSpec]:
    """The agents this run will use: the request's own line-up when it carried
    one (validated at the entry boundary), else the task's.

    Every node that needs the line-up goes through here, so the fan-out, the
    response ordering, and the API's pending chips can never disagree.
    """
    specs = state.get("agent_specs")
    if specs:
        return [
            AgentSpec(
                name=s["name"],
                provider=s["provider"],
                model=s.get("model"),
                system=s.get("system"),
            )
            for s in specs
        ]
    return resolve_task_agents(state.get("task", "default"))


# ---------------------------------------------------------------------------
# Node: redact
# ---------------------------------------------------------------------------
def redact_node(state: OrchestratorState) -> dict[str, Any]:
    """Pass-through.

    PII redaction happens at each *entry boundary* (the API request handler
    and the CLI), where it can scrub only the newest user message and carry the
    redacted history forward. See `app/api/server.py` and `app/main.py`. The
    node is kept so the graph topology is unchanged.
    """
    return {}


# ---------------------------------------------------------------------------
# Fan-out router: one Send per agent
# ---------------------------------------------------------------------------
def fan_out_providers(state: OrchestratorState) -> list[Send]:
    """Emit one parallel `call_provider` branch per agent of the run."""
    messages = state.get("messages") or []
    rid = (state.get("meta") or {}).get("request_id", "?")
    return [
        Send(
            "call_provider",
            {
                "agent_name": spec.name,
                "provider_name": spec.provider,
                "model_override": spec.model,
                "system": spec.system,
                "messages": messages,
                "request_id": rid,
            },
        )
        for spec in _agents_for(state)
    ]


# ---------------------------------------------------------------------------
# Node: call_provider  (one parallel branch per agent)
# ---------------------------------------------------------------------------
async def call_provider_node(branch: dict[str, Any]) -> dict[str, Any]:
    """Stream one agent's provider and append its response.

    Receives the Send payload (not the full graph state). Anything that goes
    wrong — including construction errors such as a missing API key — becomes a
    soft per-branch failure, so sibling branches still deliver their answers.

    Text is streamed out as `agent_delta` events while the call runs; the
    `agent` event at the end stays authoritative (full content, usage, latency).
    """
    agent: str = branch["agent_name"]
    name: str = branch["provider_name"]
    model: str | None = branch.get("model_override")
    on_delta, flush = _delta_emitter("agent_delta", provider=agent)
    try:
        provider = registry.get_provider(name, model=model)
        result = await provider.astream(
            branch["messages"], system=branch.get("system"), on_delta=on_delta
        )
        flush()
    except Exception as e:
        flush()
        result = ProviderResult(
            provider=name,
            model=model or registry.default_model(name) or "?",
            content="",
            latency_ms=0,
            error=f"{type(e).__name__}: {e}",
        )

    response: ProviderResponse = {
        # The agent's configured name, not the provider class's vendor name:
        # several agents can share a vendor, and the trace, error attribution,
        # and finalize's config-order ranking all key on the agent identity.
        "provider": agent,
        "model": result.model,
        "content": result.content,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
    # Each parallel branch logs and streams as it resolves, tagged with the
    # shared per-query correlation id.
    log_provider(branch.get("request_id", "?"), dict(response))
    _stream_event({"type": "agent", **response})
    return {"responses": [response]}


def _in_lineup_order(
    responses: list[ProviderResponse], state: OrchestratorState
) -> list[ProviderResponse]:
    """Sort responses into the run's configured agent order (merge order of
    parallel branches is scheduling-dependent, so it is never meaningful)."""
    order = {spec.name: i for i, spec in enumerate(_agents_for(state))}
    return sorted(responses, key=lambda r: order.get(r.get("provider", ""), len(order)))


# ---------------------------------------------------------------------------
# Node: summarize  (M3 synthesis)
# ---------------------------------------------------------------------------
DEFAULT_SUMMARIZER_SYSTEM = (
    "Multiple AI agents answered the same user request independently. "
    "Synthesize their answers into a single best response: merge the strongest "
    "points, resolve disagreements in favor of the best-supported claim, and "
    "drop redundancy. Reply with the final answer only — do not mention the "
    "agents or these instructions."
)


def _synthesis_user_message(question: str, responses: list[ProviderResponse]) -> str:
    lines = [f"User request:\n{question}"]
    for r in responses:
        lines.append("")
        lines.append(f"Answer from {r.get('provider')} ({r.get('model')}):\n{r.get('content', '')}")
    return "\n".join(lines)


async def summarize_node(state: OrchestratorState) -> dict[str, Any]:
    """Synthesize the parallel responses into one answer.

    Runs only when the run configures a `summarizer` and at least two branches
    succeeded — with one good response there is nothing to merge, so the extra
    call is skipped and `finalize` uses that response directly. A failing
    summarizer must never lose the agents' answers: on any error the node
    records it and leaves `summary` unset, which falls back to the
    first-successful behavior.

    A request that carried its own line-up also owns the synthesis decision —
    including the decision *not* to synthesize. Falling back to the task's
    summarizer there would run a call the user switched off (and one the API's
    `start` event already announced as absent), so the task config is consulted
    only when the line-up came from the task in the first place.
    """
    task = state.get("task", "default")
    if state.get("agent_specs"):
        cfg = state.get("summarizer_cfg") or {}
    else:
        cfg = load_task_config(task).get("summarizer") or {}
    name = cfg.get("provider")
    ok = [r for r in state.get("responses", []) if not r.get("error")]
    if not name or len(ok) < 2 or not registry.is_configured(name):
        return {}

    model = cfg.get("model")
    question = next(
        (m["content"] for m in reversed(state.get("messages") or []) if m["role"] == "user"),
        "",
    )
    synth_messages: list[ChatMessage] = [
        {
            "role": "user",
            "content": _synthesis_user_message(question, _in_lineup_order(ok, state)),
        }
    ]
    _stream_event({"type": "summarizer_start", "provider": name, "model": model})
    on_delta, flush = _delta_emitter("summarizer_delta")
    try:
        provider = registry.get_provider(name, model=model)
        result = await provider.astream(
            synth_messages,
            system=cfg.get("system") or DEFAULT_SUMMARIZER_SYSTEM,
            on_delta=on_delta,
        )
        flush()
    except Exception as e:
        flush()
        result = ProviderResult(
            provider=name,
            model=model or registry.default_model(name) or "?",
            content="",
            latency_ms=0,
            error=f"{type(e).__name__}: {e}",
        )

    run: ProviderResponse = {
        "provider": name,
        "model": result.model,
        "content": result.content,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
    rid = (state.get("meta") or {}).get("request_id", "?")
    log_provider(rid, {**run, "provider": f"summarize({name})"})
    _stream_event({"type": "summarizer", **run})

    if result.error or not result.content:
        return {
            "summarizer_run": run,
            "errors": [f"summarizer ({name}): {result.error or 'empty response'}"],
        }
    return {"summary": result.content, "summarizer_run": run}


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------
def finalize_node(state: OrchestratorState) -> dict[str, Any]:
    """Pick the primary answer from the merged responses.

    The synthesized summary wins when the summarize node produced one.
    Otherwise (no summarizer configured, or it was skipped or failed) the answer
    is the first *successful* response in the run's configured agent order;
    the other responses stay visible in the UI's orchestration trace. Failed
    branches are reported in `errors` either way.
    """
    responses = state.get("responses", [])
    if not responses:
        return {"final_answer": "", "errors": ["no provider responded"]}

    ranked = _in_lineup_order(responses, state)
    errors = [f"{r['provider']}: {r['error']}" for r in ranked if r.get("error")]

    summary = state.get("summary")
    if summary:
        return {"final_answer": summary, "errors": errors}

    ok = [r for r in ranked if not r.get("error")]
    if not ok:
        return {"final_answer": "", "errors": errors}
    return {"final_answer": ok[0].get("content", ""), "errors": errors}
