"""SSE streaming tests.

Verifies that POST /api/chat/stream:
  * Emits `start` (resolved line-up + redacted newest message) first, then one
    `agent` event per branch, `summarizer_start`/`summarizer` when the task
    synthesizes, and `final` + `done` last.
  * Streams a failing branch's `agent` event with its error, skips the
    summarizer when only one branch succeeded, and still falls back to the
    first successful answer in `final`.
  * Emits no summarizer events for tasks without a summarizer.
  * Streams token deltas (`agent_delta` / `summarizer_delta`) *before* the
    corresponding end event, and that they reconstruct its content exactly.
  * Reports a client-supplied line-up (the UI's agent panel) in `start` and
    runs it instead of the task's.

Fakes are instant, so tests assert event *order*, not wall-clock timing. The
per-branch timing itself comes from LangGraph's custom stream writer, which
fires inside the node as each branch resolves.

Every task now routes through the one `openrouter` provider, so the fake is
registered once and keys its answer on the *model* — the maps below mirror
tasks.yaml and are what tell the branches apart.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.providers import registry
from app.providers.base import LLMProvider, ProviderResult

SYNTH_MARKER = "Multiple AI agents answered"

# agent name -> model, per tasks.yaml.
DEFAULT_AGENTS = {
    "gpt-oss": "openai/gpt-oss-20b:free",
    "nemotron": "nvidia/nemotron-3-super-120b-a12b:free",
}
RESEARCH_AGENTS = {
    "researcher": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "skeptic": "google/gemma-4-31b-it:free",
}
SUMMARIZER_MODEL = "nvidia/nemotron-nano-9b-v2:free"


def answer_for(agent: str, agents: dict[str, str]) -> str:
    """What the fake replies for one agent of a line-up."""
    return f"answer from {agents[agent]}"


class FakeProvider(LLMProvider):
    """Blocking fake: implements only `acomplete`, so it also exercises the
    base class's `astream` fallback for providers that cannot stream.

    `failing` names the models whose branch call should soft-fail.
    """

    def __init__(self, model: str | None = None, *, failing: set[str] | None = None):
        super().__init__(model=model or "fake-model")
        self.name = "openrouter"
        self._failing = failing or set()

    def _answer(self, system: str | None) -> str:
        is_synth = bool(system) and SYNTH_MARKER in system
        return "merged answer" if is_synth else f"answer from {self.model}"

    async def acomplete(self, messages, *, system=None, **kwargs) -> ProviderResult:
        if self.model in self._failing:
            return ProviderResult(
                provider=self.name, model=self.model, content="",
                latency_ms=1, error="simulated failure",
            )
        return ProviderResult(
            provider=self.name, model=self.model,
            content=self._answer(system),
            latency_ms=1,
        )


class StreamingFakeProvider(FakeProvider):
    """Emits its answer one word at a time through `on_delta`.

    The pause between words exceeds the node's coalescing window, so each word
    becomes its own SSE frame — that is what makes the delta assertions below
    test the batching path rather than a single end-of-call flush.
    """

    async def astream(self, messages, *, system=None, on_delta=None, **kwargs):
        if self.model in self._failing:
            return await self.acomplete(messages, system=system, **kwargs)
        answer = self._answer(system)
        for i, word in enumerate(answer.split(" ")):
            if on_delta:
                on_delta(word if i == 0 else f" {word}")
            await asyncio.sleep(0.06)
        return ProviderResult(
            provider=self.name, model=self.model, content=answer, latency_ms=1
        )


def _register(cls=FakeProvider, **kwargs) -> None:
    registry.register("openrouter", lambda m: cls(m, **kwargs))


@pytest.fixture(autouse=True)
def _restore_registry():
    original = registry._REGISTRY.copy()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)


def _events(payload: dict) -> list[tuple[str, dict]]:
    """POST the stream endpoint and parse the SSE frames into (event, data)."""
    events: list[tuple[str, dict]] = []
    with TestClient(server.app) as client:
        with client.stream("POST", "/api/chat/stream", json=payload) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            name, data_lines = None, []
            for line in r.iter_lines():
                if line == "":
                    if data_lines:
                        events.append(
                            (name or "message", json.loads("\n".join(data_lines)))
                        )
                    name, data_lines = None, []
                elif line.startswith("event:"):
                    name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
    return events


def _first(events, name: str) -> dict:
    return next(d for n, d in events if n == name)


def test_research_stream_event_sequence():
    _register()

    events = _events(
        {"messages": [{"role": "user", "content": "reach me at alice@example.com"}],
         "task": "research"}
    )
    names = [n for n, _ in events]

    assert names[0] == "start"
    assert names[-2:] == ["final", "done"]
    assert names.count("agent") == 2
    # The synthesis happens strictly after the last branch resolved.
    assert names.index("summarizer_start") > max(
        i for i, n in enumerate(names) if n == "agent"
    )
    assert names.index("summarizer") > names.index("summarizer_start")

    start = _first(events, "start")
    assert [a["name"] for a in start["agents"]] == ["researcher", "skeptic"]
    assert start["summarizer"]["model"] == SUMMARIZER_MODEL
    # Boundary redaction is reported up front so the client can store it.
    assert start["redacted_user_message"] == "reach me at [EMAIL]"

    assert {d["provider"] for n, d in events if n == "agent"} == set(RESEARCH_AGENTS)
    assert _first(events, "final")["reply"] == "merged answer"


def test_failing_branch_streams_error_and_final_falls_back():
    _register(failing={RESEARCH_AGENTS["skeptic"]})

    events = _events(
        {"messages": [{"role": "user", "content": "hi"}], "task": "research"}
    )
    names = [n for n, _ in events]

    agent_events = {d["provider"]: d for n, d in events if n == "agent"}
    assert agent_events["skeptic"]["error"] == "simulated failure"
    assert agent_events["researcher"]["error"] is None
    # Only one success → the summarizer is skipped, no events for it.
    assert "summarizer_start" not in names and "summarizer" not in names
    final = _first(events, "final")
    assert final["reply"] == answer_for("researcher", RESEARCH_AGENTS)
    assert any("skeptic" in e for e in final["errors"])


def test_deltas_stream_before_each_agent_resolves():
    _register(StreamingFakeProvider)

    events = _events(
        {"messages": [{"role": "user", "content": "hi"}], "task": "default"}
    )
    names = [n for n, _ in events]

    for agent in DEFAULT_AGENTS:
        deltas = [
            i for i, (n, d) in enumerate(events)
            if n == "agent_delta" and d["provider"] == agent
        ]
        end = next(i for i, (n, d) in enumerate(events)
                   if n == "agent" and d["provider"] == agent)
        # Text arrives progressively, and all of it lands before the chip resolves.
        assert len(deltas) > 1
        assert max(deltas) < end
        # The deltas reconstruct exactly what the authoritative event carries.
        streamed = "".join(events[i][1]["text"] for i in deltas)
        assert streamed == events[end][1]["content"] == answer_for(agent, DEFAULT_AGENTS)

    assert names.index("agent_delta") < names.index("agent")


def test_summarizer_deltas_stream_between_its_start_and_end():
    _register(StreamingFakeProvider)

    events = _events(
        {"messages": [{"role": "user", "content": "hi"}], "task": "research"}
    )
    names = [n for n, _ in events]

    deltas = [i for i, n in enumerate(names) if n == "summarizer_delta"]
    assert deltas, "synthesis should stream too"
    assert min(deltas) > names.index("summarizer_start")
    assert max(deltas) < names.index("summarizer")
    streamed = "".join(events[i][1]["text"] for i in deltas)
    assert streamed == _first(events, "summarizer")["content"] == "merged answer"
    assert _first(events, "final")["reply"] == "merged answer"


def test_non_streaming_provider_still_delivers_its_text():
    """A provider that implements only `acomplete` falls back to the base
    class's `astream`: one delta with the whole answer, then the usual events."""
    _register()

    events = _events(
        {"messages": [{"role": "user", "content": "hi"}], "task": "default"}
    )
    names = [n for n, _ in events]

    assert names.count("agent") == 2
    for agent in DEFAULT_AGENTS:
        deltas = [d for n, d in events if n == "agent_delta" and d["provider"] == agent]
        assert [d["text"] for d in deltas] == [answer_for(agent, DEFAULT_AGENTS)]
    assert names[-2:] == ["final", "done"]
    assert _first(events, "final")["reply"] == answer_for("gpt-oss", DEFAULT_AGENTS)


def test_failing_branch_streams_no_text():
    _register(StreamingFakeProvider, failing={DEFAULT_AGENTS["nemotron"]})

    events = _events(
        {"messages": [{"role": "user", "content": "hi"}], "task": "default"}
    )

    providers = {d["provider"] for n, d in events if n == "agent_delta"}
    assert providers == {"gpt-oss"}
    assert _first(events, "final")["reply"] == answer_for("gpt-oss", DEFAULT_AGENTS)


def test_task_without_summarizer_emits_no_summarizer_events():
    _register()

    events = _events(
        {"messages": [{"role": "user", "content": "hi"}], "task": "default"}
    )
    names = [n for n, _ in events]

    assert names.count("agent") == 2
    assert "summarizer_start" not in names and "summarizer" not in names
    assert _first(events, "final")["reply"] == answer_for("gpt-oss", DEFAULT_AGENTS)


def test_custom_lineup_is_reported_in_start_and_run():
    """The agent panel's roster replaces the task's, and `start` announces it so
    the UI's pending chips match the branches that actually run."""
    _register()

    events = _events(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "task": "research",  # would otherwise run researcher/skeptic + synthesis
            "agents": [
                {"name": "brief", "provider": "openrouter", "model": "model-a",
                 "system": "Answer in one sentence."},
                {"name": "detailed", "provider": "openrouter", "model": "model-b"},
            ],
        }
    )
    names = [n for n, _ in events]

    start = _first(events, "start")
    assert [a["name"] for a in start["agents"]] == ["brief", "detailed"]
    assert [a["model"] for a in start["agents"]] == ["model-a", "model-b"]
    # No summarizer was requested, so the task's must not sneak back in.
    assert start["summarizer"] is None
    assert "summarizer_start" not in names

    assert {d["provider"] for n, d in events if n == "agent"} == {"brief", "detailed"}
    assert _first(events, "final")["reply"] == "answer from model-a"
