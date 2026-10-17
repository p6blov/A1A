"""FastAPI server that wraps the LangGraph orchestrator for the chat UI.

This is a thin HTTP layer over `build_graph()` — it does not change the graph.
The frontend sends the full conversation (held only in the browser's memory),
the server passes it into the graph as structured messages, and returns the
assistant's reply. Nothing is persisted to disk.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config.models import free_models
from app.config.settings import get_settings
from app.config.tasks import AgentSpec, load_task_config, load_tasks
from app.graph.builder import build_graph
from app.graph.nodes import resolve_task_agents
from app.providers import registry
from app.utils.redact import redact
from app.utils import telemetry

# Headers a reverse proxy / CDN (nginx, Cloudflare, load balancers) injects to
# carry the original client IP. We strip them so the user's IP never reaches
# application code, logs, or the upstream model provider.
_CLIENT_IP_HEADERS = (
    "x-forwarded-for",
    "x-real-ip",
    "x-client-ip",
    "cf-connecting-ip",
    "true-client-ip",
    "forwarded",
    "x-forwarded-host",
)


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AgentOverride(BaseModel):
    """One agent of a client-supplied line-up (the UI's agent panel).

    Sent per request, so who runs — and under which system prompt — is an
    on-device setting rather than server config. Validated by
    `_resolve_overrides` before it reaches the graph.
    """

    name: str | None = None
    provider: str = "openrouter"
    model: str | None = None
    system: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    task: str = "default"
    # When present, this line-up replaces the task's agents from tasks.yaml for
    # this turn only. `task` still selects the fallback if it does not survive
    # validation.
    agents: list[AgentOverride] | None = None
    # Likewise for the synthesis step; None means "use the task's summarizer".
    summarizer: AgentOverride | None = None


class AgentResult(BaseModel):
    """One agent's (provider's) contribution to a turn. Mirrors the graph's
    ProviderResponse so the UI can render the orchestration trace: who ran,
    what they said, how long it took."""

    provider: str
    model: str | None = None
    content: str = ""
    error: str | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatResponse(BaseModel):
    reply: str
    provider: str | None = None
    model: str | None = None
    error: str | None = None
    # Every agent that contributed to this turn (one today; N after fan-out).
    # `reply` stays the single synthesized answer.
    agents: list[AgentResult] = Field(default_factory=list)
    # The synthesis step, when the task's summarizer ran (whether it succeeded
    # or soft-failed) — so the UI can show it in the trace alongside the agents.
    summarizer: AgentResult | None = None
    # The newest user message after PII redaction. The client replaces its stored
    # copy with this so the conversation history carries forward already-scrubbed
    # (None when redaction is disabled).
    redacted_user_message: str | None = None


class TaskAgentInfo(BaseModel):
    name: str  # the agent's role name (defaults to the provider in config)
    provider: str
    model: str | None = None


class TaskInfo(BaseModel):
    """A task type from tasks.yaml with its resolved agent line-up, so the UI
    can offer a task picker and show which agents are about to run."""

    name: str
    agents: list[TaskAgentInfo]
    # Set when the task synthesizes its answers — lets the UI pre-render a
    # pending summarizer chip.
    summarizer: TaskAgentInfo | None = None


class FreeModelInfo(BaseModel):
    """One model the agent panel may offer. Free-tier only, by construction —
    see app/config/models.py for how the catalog is narrowed."""

    id: str
    name: str
    lab: str
    context_length: int | None = None
    max_completion_tokens: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Compile the graph once at startup and reuse it across requests.
    app.state.graph = build_graph()
    # Route telemetry through uvicorn's stdout handler (it runs with
    # --no-access-log, so this is what replaces the lost "200 OK" lines).
    telemetry.attach_to_uvicorn()
    yield


app = FastAPI(title="Multi-agent orchestrator", lifespan=lifespan)


@app.middleware("http")
async def strip_client_ip(request: Request, call_next):
    """Drop client-IP-bearing headers and blank out the connection's source IP.

    duck.ai-style privacy: the server that terminates the user's connection
    cannot avoid *receiving* their IP at the network layer, but it can refuse to
    propagate it. This middleware removes any proxy-forwarded IP headers and
    clears `request.client` so no downstream handler, log line, or error report
    can read the user's address. The model provider already only ever sees this
    server's IP (the relay makes a fresh outbound connection).
    """
    headers = [
        (k, v)
        for (k, v) in request.scope.get("headers", [])
        if k.decode("latin-1").lower() not in _CLIENT_IP_HEADERS
    ]
    request.scope["headers"] = headers
    request.scope["client"] = None  # request.client -> None for all handlers
    return await call_next(request)


# The Vite dev server proxies /api to this server, so requests are same-origin
# in development. CORS is kept permissive for local use only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    """Liveness plus the one setup question worth answering: are any providers
    actually usable? Every task now runs on a single key, so "the server is up
    but nothing can answer" is the failure people hit, and it is otherwise only
    visible as a per-branch error after spending a turn."""
    settings = get_settings()
    configured = [p for p in registry.available_providers() if registry.is_configured(p)]
    return {
        "status": "ok",
        "provider": settings.default_provider,
        "model": settings.default_model,
        "configured_providers": configured,
        # False means every branch will soft-fail with "…_API_KEY is not set".
        "default_provider_configured": registry.is_configured(settings.default_provider),
    }


def _agent_info(name: str, provider: str, model: str | None) -> TaskAgentInfo:
    """Describe one agent to the UI with its model made concrete."""
    return TaskAgentInfo(
        name=name, provider=provider, model=model or registry.default_model(provider)
    )


def _summarizer_info(cfg: dict | None) -> TaskAgentInfo | None:
    """The summarizer chip for a `{provider, model, system}` block, or None when
    the block is absent or its provider has no credentials."""
    cfg = cfg or {}
    provider = cfg.get("provider")
    if not provider or not registry.is_configured(provider):
        return None
    return _agent_info("summarizer", provider, cfg.get("model"))


def _task_info(name: str, cfg: dict | None) -> TaskInfo:
    """Resolve one task's line-up, exactly as the graph's router would run it:
    unconfigured providers filtered out, model overrides made concrete."""
    return TaskInfo(
        name=name,
        agents=[
            _agent_info(s.name, s.provider, s.model) for s in resolve_task_agents(name)
        ],
        summarizer=_summarizer_info((cfg or {}).get("summarizer")),
    )


@app.get("/api/tasks", response_model=list[TaskInfo])
async def tasks() -> list[TaskInfo]:
    """Describe the configured task types and the agents each one routes to,
    so the UI can offer a task picker and pre-render pending chips."""
    return [_task_info(name, cfg) for name, cfg in load_tasks().items()]


@app.get("/api/models", response_model=list[FreeModelInfo])
async def models() -> list[FreeModelInfo]:
    """The free models the agent panel may offer, live from OpenRouter's public
    catalog (cached, with a checked-in fallback). Never fails hard."""
    return [FreeModelInfo(**vars(m)) for m in await free_models()]


# A pasted-in "system prompt" that is really a document would blow past the
# model's context before the conversation even starts. Cap it at the boundary.
MAX_SYSTEM_PROMPT_CHARS = 4000


def _resolve_overrides(
    req: ChatRequest,
) -> tuple[list[AgentSpec] | None, dict | None]:
    """Validate a client-supplied line-up into specs the graph can run.

    Everything untrustworthy is handled here rather than deeper in: unknown
    providers are dropped, duplicate agent names are suffixed because the whole
    trace keys on them, oversized system prompts are truncated, and the roster
    is capped at `max_custom_agents` so one request cannot fan out arbitrarily
    many upstream calls.

    Unconfigured providers are dropped only if *something else* is configured —
    the same degradation `resolve_task_agents` applies to tasks.yaml. Keeping
    them when nothing is configured makes the branches soft-fail with an error
    naming the missing key, which beats silently answering from a preset the
    user did not choose.

    Returns `(None, None)` when nothing usable was supplied, which leaves the
    turn on its tasks.yaml preset — an empty fan-out is never an outcome.
    """
    if not req.agents:
        return None, None

    settings = get_settings()
    specs: list[AgentSpec] = []
    seen: set[str] = set()
    for i, a in enumerate(req.agents[: settings.max_custom_agents]):
        if a.provider not in registry.available_providers():
            continue
        name = (a.name or "").strip() or a.model or a.provider
        while name in seen:
            name = f"{name}-{i}"
        seen.add(name)
        system = (a.system or "").strip() or None
        specs.append(
            AgentSpec(
                name=name,
                provider=a.provider,
                model=(a.model or "").strip() or None,
                system=system[:MAX_SYSTEM_PROMPT_CHARS] if system else None,
            )
        )
    specs = [s for s in specs if registry.is_configured(s.provider)] or specs
    if not specs:
        return None, None

    summarizer = None
    s = req.summarizer
    if s and registry.is_configured(s.provider):
        system = (s.system or "").strip() or None
        summarizer = {
            "provider": s.provider,
            "model": (s.model or "").strip() or None,
            "system": system[:MAX_SYSTEM_PROMPT_CHARS] if system else None,
        }
    return specs, summarizer


def _prepare_request(req: ChatRequest) -> tuple[dict, str | None, str]:
    """Shared request prep for both chat endpoints: newest-message PII
    redaction, the client's agent line-up (when it sent one), initial graph
    state, and the telemetry correlation id."""
    settings = get_settings()
    messages = list(req.messages)

    # Redact PII from the newest user message only. Earlier turns were already
    # scrubbed when they were the newest message and are carried forward in
    # redacted form by the client, so the whole history stays clean without
    # re-scrubbing it — and assistant replies are never altered.
    redacted_user_message: str | None = None
    if settings.enable_redaction and messages[-1].role == "user":
        cleaned = redact(messages[-1].content)
        messages[-1] = Message(role="user", content=cleaned)
        redacted_user_message = cleaned

    rid = telemetry.new_request_id()
    telemetry.log_start(rid, req.task)

    specs, summarizer_cfg = _resolve_overrides(req)
    initial = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "task": req.task,
        # None on both leaves the graph on the task's tasks.yaml line-up.
        "agent_specs": [vars(s) for s in specs] if specs else None,
        "summarizer_cfg": summarizer_cfg,
        "responses": [],
        "errors": [],
        "meta": {"request_id": rid},
    }
    return initial, redacted_user_message, rid


def _lineup_info(req: ChatRequest, initial: dict) -> TaskInfo:
    """The line-up this turn will actually run, for the UI's pending chips.

    A client roster that survived validation wins; otherwise it is the task's
    own line-up. Reading it back off the prepared state (rather than re-deriving
    it from the request) guarantees the chips match the branches."""
    specs = initial.get("agent_specs")
    if not specs:
        return _task_info(req.task, load_task_config(req.task))
    return TaskInfo(
        name=req.task,
        agents=[_agent_info(s["name"], s["provider"], s["model"]) for s in specs],
        summarizer=_summarizer_info(initial.get("summarizer_cfg")),
    )


def _log_aggregate(rid: str, final: dict, wall_ms: int) -> None:
    """The aggregate is the per-query cost line, so the synthesis call counts too."""
    responses = final.get("responses") or []
    summarizer_run = final.get("summarizer_run")
    telemetry.log_aggregate(
        rid, responses + ([summarizer_run] if summarizer_run else []), wall_ms
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        return ChatResponse(reply="", error="no messages provided")

    initial, redacted_user_message, rid = _prepare_request(req)
    started = time.perf_counter()
    final = await app.state.graph.ainvoke(initial)
    _log_aggregate(rid, final, int((time.perf_counter() - started) * 1000))

    responses = final.get("responses") or []
    first = responses[0] if responses else {}
    errors = final.get("errors") or []
    summarizer_run = final.get("summarizer_run")

    return ChatResponse(
        reply=final.get("final_answer") or "",
        provider=first.get("provider"),
        model=first.get("model"),
        error=errors[0] if errors else None,
        agents=[AgentResult(**r) for r in responses],
        summarizer=AgentResult(**summarizer_run) if summarizer_run else None,
        redacted_user_message=redacted_user_message,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE variant of /api/chat: text and agent chips stream as they resolve.

    Event order: `start` (resolved line-up + redacted newest message) →
    `agent_delta` × M interleaved across branches → `agent` × N as each
    parallel branch finishes → `summarizer_start` / `summarizer_delta` ×  M /
    `summarizer` when the task synthesizes → `final` (reply, errors) → `done`.
    The `*_delta` events carry incremental text for live rendering; the `agent`,
    `summarizer`, and `final` events remain authoritative (full content, token
    usage, latency) and the client reconciles against them.

    All of these are pushed by the graph nodes through LangGraph's custom stream
    writer, so their timing is the branches' real timing. If the client
    disconnects (the UI's Stop button), this generator is cancelled and stops
    forwarding events — branches already in flight may still finish upstream,
    so a stopped turn costs what it had already committed to.
    """

    async def gen():
        try:
            if not req.messages:
                yield _sse("error", {"detail": "no messages provided"})
                return

            initial, redacted_user_message, rid = _prepare_request(req)
            info = _lineup_info(req, initial)
            yield _sse(
                "start",
                {
                    "task": req.task,
                    "agents": [a.model_dump() for a in info.agents],
                    "summarizer": info.summarizer.model_dump() if info.summarizer else None,
                    "redacted_user_message": redacted_user_message,
                },
            )

            started = time.perf_counter()
            final: dict = {}
            try:
                async for mode, chunk in app.state.graph.astream(
                    initial, stream_mode=["custom", "values"]
                ):
                    if mode == "custom":
                        payload = dict(chunk)
                        yield _sse(payload.pop("type", "event"), payload)
                    else:
                        final = chunk  # keep the latest full state; last one wins
            except asyncio.CancelledError:
                # Client hung up (Stop button). Still record what the turn cost
                # before letting the cancellation unwind — a stopped turn is
                # billed the same as a finished one.
                _log_aggregate(rid, final, int((time.perf_counter() - started) * 1000))
                raise
            _log_aggregate(rid, final, int((time.perf_counter() - started) * 1000))

            errors = final.get("errors") or []
            yield _sse("final", {"reply": final.get("final_answer") or "", "errors": errors})
            yield _sse("done", {})
        except Exception as e:  # surface mid-stream failures to the client
            yield _sse("error", {"detail": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        # No buffering between us and the browser: chip timing is the feature.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
