"""Milestone 1 tests.

Verifies that:
  * The LangGraph runs end-to-end with a fake provider (no network calls).
  * Swapping providers requires only a registry change (no code edits to graph).
  * Provider errors surface as `errors` in the final state instead of raising.
"""
from __future__ import annotations

import asyncio

import pytest

from app.graph.builder import build_graph
from app.providers.base import LLMProvider, ProviderResult
from app.providers import registry


# tasks.yaml's single-agent task. These tests want one branch, so they route
# through it rather than the two-agent `default`.
SINGLE_AGENT_TASK = "code_review"
SINGLE_AGENT_NAME = "reviewer"


class FakeProvider(LLMProvider):
    name = "openrouter"  # the provider every task in tasks.yaml routes to

    def __init__(self, model: str | None = None, *, reply: str = "hello", fail: bool = False):
        super().__init__(model=model or "fake-model")
        self._reply = reply
        self._fail = fail

    async def acomplete(self, messages, *, system=None, **kwargs) -> ProviderResult:
        if self._fail:
            return ProviderResult(
                provider=self.name, model=self.model, content="",
                latency_ms=1, error="simulated failure",
            )
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return ProviderResult(
            provider=self.name, model=self.model,
            content=f"{self._reply}::{last_user}", latency_ms=1,
        )


@pytest.fixture(autouse=True)
def _restore_registry():
    """Each test starts with a clean registry override and restores after, so a
    fake never leaks into another test — or, worse, a real provider into this
    one."""
    original = registry._REGISTRY.copy()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)


def test_graph_runs_end_to_end_with_fake_provider():
    registry.register("openrouter", lambda model: FakeProvider(model, reply="ok"))

    graph = build_graph()
    final = asyncio.run(graph.ainvoke(
        {"messages": [{"role": "user", "content": "ping"}], "task": SINGLE_AGENT_TASK, "responses": [], "errors": []}
    ))

    assert final["final_answer"] == "ok::ping"
    assert len(final["responses"]) == 1
    assert final["responses"][0]["provider"] == SINGLE_AGENT_NAME
    assert not final.get("errors")


def test_provider_error_does_not_crash_graph():
    registry.register("openrouter", lambda model: FakeProvider(model, fail=True))

    graph = build_graph()
    final = asyncio.run(graph.ainvoke(
        {"messages": [{"role": "user", "content": "ping"}], "task": SINGLE_AGENT_TASK, "responses": [], "errors": []}
    ))

    assert final["final_answer"] == ""
    assert final["errors"] and "simulated failure" in final["errors"][0]


def test_provider_swap_is_config_only():
    """Demonstrates the abstraction: registering a different factory under the
    same name changes behavior without touching the graph or nodes."""
    registry.register("openrouter", lambda model: FakeProvider(model, reply="v1"))
    graph = build_graph()
    r1 = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "x"}], "task": SINGLE_AGENT_TASK, "responses": [], "errors": []}))

    registry.register("openrouter", lambda model: FakeProvider(model, reply="v2"))
    graph = build_graph()
    r2 = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "x"}], "task": SINGLE_AGENT_TASK, "responses": [], "errors": []}))

    assert r1["final_answer"].startswith("v1::")
    assert r2["final_answer"].startswith("v2::")
