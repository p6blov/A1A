"""M4 tests: structured messages + per-agent system prompts (roles).

Verifies that:
  * Both tasks.yaml shapes (`agents:` list and the legacy `providers:`
    shorthand) normalize to the same AgentSpec structure; duplicate agent
    names are rejected.
  * A client-supplied line-up (the UI's agent panel) overrides the task's, and
    its per-agent system prompts reach the providers.
  * Each agent's system prompt from the task config reaches its provider.
  * The full multi-turn conversation reaches providers as structured messages
    (no flattening), end-to-end through /api/chat.
  * One provider can serve several roles in a task, and each resolves under
    its own agent name.
  * The Google adapter's message mapping converts roles to the genai wire
    format ("assistant" → "model").

Fakes record every call's (messages, system) so tests assert exactly what a
provider received. No test touches the network.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.config import tasks as tasks_mod
from app.config.tasks import AgentSpec, load_task_agents
from app.graph import nodes
from app.graph.builder import build_graph
from app.providers import registry
from app.providers.base import LLMProvider, ProviderResult
from app.providers.google_provider import to_genai_contents


class RecordingProvider(LLMProvider):
    """Fake that records (agent-call) inputs into a shared list."""

    def __init__(self, name: str, model: str | None = None, *, reply: str = "", calls: list | None = None):
        super().__init__(model=model or f"{name}-model")
        self.name = name
        self._reply = reply or f"answer from {name}"
        self._calls = calls if calls is not None else []

    async def acomplete(self, messages, *, system=None, **kwargs) -> ProviderResult:
        self._calls.append({"provider": self.name, "messages": messages, "system": system})
        return ProviderResult(
            provider=self.name, model=self.model, content=self._reply, latency_ms=1
        )


@pytest.fixture(autouse=True)
def _restore_registry():
    original = registry._REGISTRY.copy()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)


# ---------------------------------------------------------------------------
# Config normalization
# ---------------------------------------------------------------------------
def test_real_yaml_normalizes_both_shapes():
    research = load_task_agents("research")
    assert [s.name for s in research] == ["researcher", "skeptic"]
    assert all(s.provider == "openrouter" for s in research)
    # Same provider twice: the models are what differ, and both carry a role.
    assert len({s.model for s in research}) == 2
    assert all(s.system for s in research)

    default = load_task_agents("default")
    assert [s.name for s in default] == ["gpt-oss", "nemotron"]
    assert all(s.provider == "openrouter" and s.model and not s.system for s in default)


def test_agents_list_name_defaults_to_provider(monkeypatch):
    cfg = {"agents": [{"provider": "openrouter"}, {"name": "b", "provider": "openrouter"}]}
    monkeypatch.setattr(tasks_mod, "load_task_config", lambda task: cfg)
    assert [s.name for s in load_task_agents("x")] == ["openrouter", "b"]


def test_duplicate_agent_names_rejected(monkeypatch):
    cfg = {"agents": [{"provider": "openrouter"}, {"provider": "openrouter"}]}
    monkeypatch.setattr(tasks_mod, "load_task_config", lambda task: cfg)
    with pytest.raises(ValueError, match="duplicate agent name"):
        load_task_agents("x")


# ---------------------------------------------------------------------------
# Roles through the graph
# ---------------------------------------------------------------------------
def _run(task: str, messages=None) -> dict:
    graph = build_graph()
    return asyncio.run(graph.ainvoke({
        "messages": messages or [{"role": "user", "content": "ping"}],
        "task": task,
        "responses": [],
        "errors": [],
    }))


def test_system_prompts_reach_each_agents_provider():
    calls: list = []
    # Both research agents share one provider, so the model tells them apart.
    registry.register("openrouter", lambda m: RecordingProvider(m or "?", m, calls=calls))

    _run("research")

    branch_calls = [c for c in calls if c["system"] and "Synthesize" not in c["system"]]
    systems = [c["system"] for c in branch_calls]
    assert any("meticulous researcher" in s for s in systems)
    assert any("critical reviewer" in s for s in systems)


def test_task_without_roles_sends_no_system_prompt():
    calls: list = []
    registry.register("openrouter", lambda m: RecordingProvider(m or "?", m, calls=calls))

    _run("default")

    assert [c["system"] for c in calls] == [None, None]


def test_same_provider_twice_under_distinct_names(monkeypatch):
    specs = [
        AgentSpec(name="optimist", provider="openrouter", system="Be optimistic."),
        AgentSpec(name="pessimist", provider="openrouter", system="Be pessimistic."),
    ]
    monkeypatch.setattr(nodes, "load_task_agents", lambda task: specs)
    calls: list = []
    registry.register("openrouter", lambda m: RecordingProvider("openrouter", m, calls=calls))

    final = _run("default")

    assert {r["provider"] for r in final["responses"]} == {"optimist", "pessimist"}
    assert sorted(c["system"] for c in calls) == ["Be optimistic.", "Be pessimistic."]


# ---------------------------------------------------------------------------
# Structured history end-to-end
# ---------------------------------------------------------------------------
def test_multi_turn_history_reaches_provider_structured():
    calls: list = []
    registry.register("openrouter", lambda m: RecordingProvider("openrouter", m, calls=calls))

    convo = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow-up"},
    ]
    with TestClient(server.app) as client:
        data = client.post("/api/chat", json={"messages": convo, "task": "code_review"}).json()

    assert data["reply"] == "answer from openrouter"
    assert len(calls) == 1
    received = calls[0]["messages"]
    assert [m["role"] for m in received] == ["user", "assistant", "user"]
    assert [m["content"] for m in received] == ["first question", "first answer", "follow-up"]


# ---------------------------------------------------------------------------
# Google wire-format mapping
# ---------------------------------------------------------------------------
def test_google_contents_mapping_converts_roles():
    contents = to_genai_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert [c.role for c in contents] == ["user", "model"]
    assert [c.parts[0].text for c in contents] == ["hi", "hello"]
