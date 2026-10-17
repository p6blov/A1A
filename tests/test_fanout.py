"""M2 fan-out tests.

Verifies that:
  * The graph fans out to every agent of the task in parallel and the reducer
    merges all responses.
  * The final answer is the first *successful* response in task-config order.
  * A failing branch never breaks its siblings.
  * Unconfigured providers are skipped by the router (and by /api/tasks).
  * When nothing is configured, branches soft-fail with a diagnosable error.

The default task fans out to two OpenRouter models under one provider, so the
fake here is keyed by *model* rather than by provider name: that is what tells
the branches apart now. `DEFAULT_LINEUP` mirrors tasks.yaml — update it there
and here together.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.config.tasks import AgentSpec
from app.graph import nodes
from app.graph.builder import build_graph
from app.providers import registry
from app.providers.base import LLMProvider, ProviderResult

# (agent name, model) pairs of tasks.yaml's `default` task, in config order.
DEFAULT_LINEUP = [
    ("gpt-oss", "openai/gpt-oss-20b:free"),
    ("nemotron", "nvidia/nemotron-3-super-120b-a12b:free"),
]


class FakeProvider(LLMProvider):
    """Answers "answer from <model>", or fails when its model is in `failing`."""

    def __init__(self, name: str, model: str | None = None, *, failing: set[str] | None = None):
        super().__init__(model=model or f"{name}-model")
        self.name = name
        self._failing = failing or set()

    async def acomplete(self, messages, *, system=None, **kwargs) -> ProviderResult:
        if self.model in self._failing:
            return ProviderResult(
                provider=self.name, model=self.model, content="",
                latency_ms=1, error="simulated failure",
            )
        return ProviderResult(
            provider=self.name, model=self.model,
            content=f"answer from {self.model}", latency_ms=1,
        )


def _register_openrouter(*, failing: set[str] | None = None) -> None:
    registry.register("openrouter", lambda m: FakeProvider("openrouter", m, failing=failing))


@pytest.fixture(autouse=True)
def _restore_registry():
    original = registry._REGISTRY.copy()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)


def _run(task: str = "default") -> dict:
    graph = build_graph()
    return asyncio.run(graph.ainvoke(
        {"messages": [{"role": "user", "content": "ping"}], "task": task, "responses": [], "errors": []}
    ))


def test_fans_out_to_all_configured_providers():
    _register_openrouter()

    final = _run()

    assert {r["provider"] for r in final["responses"]} == {n for n, _ in DEFAULT_LINEUP}
    # Primary = first in tasks.yaml order, regardless of branch merge order.
    assert final["final_answer"] == f"answer from {DEFAULT_LINEUP[0][1]}"
    assert not final.get("errors")


def test_failing_branch_does_not_break_siblings():
    first_name, first_model = DEFAULT_LINEUP[0]
    _register_openrouter(failing={first_model})

    final = _run()

    assert len(final["responses"]) == 2
    # The first agent failed, so the next successful one supplies the answer...
    assert final["final_answer"] == f"answer from {DEFAULT_LINEUP[1][1]}"
    # ...and the failure is still reported, attributed to the agent.
    assert any(f"{first_name}: simulated failure" in e for e in final["errors"])


def _stub_lineup(monkeypatch, specs: list[AgentSpec]) -> None:
    """Pin the task line-up so provider-availability tests do not depend on what
    tasks.yaml happens to declare. Patched on the nodes module because that is
    where `resolve_task_agents` — and therefore /api/tasks — reads it."""
    monkeypatch.setattr(nodes, "load_task_agents", lambda task: list(specs))


MIXED_LINEUP = [
    AgentSpec(name="primary", provider="openrouter", model="openai/gpt-oss-20b:free"),
    AgentSpec(name="secondary", provider="google", model="gemini-x"),
]


def test_unconfigured_provider_is_skipped(monkeypatch):
    _stub_lineup(monkeypatch, MIXED_LINEUP)
    _register_openrouter()
    registry.register(
        "google", lambda m: FakeProvider("google", m), is_configured=lambda: False
    )

    final = _run()

    assert [r["provider"] for r in final["responses"]] == ["primary"]
    assert final["final_answer"] == "answer from openai/gpt-oss-20b:free"


def test_nothing_configured_soft_fails_with_named_error(monkeypatch):
    _stub_lineup(monkeypatch, MIXED_LINEUP)

    def _raise(_m):
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    registry.register("openrouter", _raise, is_configured=lambda: False)
    registry.register("google", _raise, is_configured=lambda: False)

    final = _run()

    # The full line-up is kept so the errors name what is missing.
    assert len(final["responses"]) == 2
    assert final["final_answer"] == ""
    assert any("OPENROUTER_API_KEY" in e for e in final["errors"])


def test_api_tasks_reflects_availability(monkeypatch):
    _stub_lineup(monkeypatch, MIXED_LINEUP)
    _register_openrouter()
    registry.register(
        "google", lambda m: FakeProvider("google", m), is_configured=lambda: False
    )
    with TestClient(server.app) as client:
        data = client.get("/api/tasks").json()

    default = next(t for t in data if t["name"] == "default")
    assert [a["name"] for a in default["agents"]] == ["primary"]


def test_api_chat_returns_all_agents():
    _register_openrouter()
    with TestClient(server.app) as client:
        data = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        ).json()

    assert data["reply"] == f"answer from {DEFAULT_LINEUP[0][1]}"
    assert {a["provider"] for a in data["agents"]} == {n for n, _ in DEFAULT_LINEUP}
