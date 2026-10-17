"""API tests for the orchestration surface consumed by the UI.

Verifies that:
  * POST /api/chat returns the full per-agent results (`agents`), not just the
    synthesized reply — the UI renders these as the orchestration trace.
  * Agent failures surface per-agent (`agents[i].error`) as well as top-level.
  * GET /api/tasks lists every task type with its resolved agent line-up.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.config import settings as settings_mod
from app.providers import registry
from app.providers.base import LLMProvider, ProviderResult


# tasks.yaml's single-agent task — these tests assert on one agent's fields, so
# they route through it rather than the two-agent `default`.
SINGLE_AGENT_TASK = "code_review"
SINGLE_AGENT_NAME = "reviewer"


class FakeProvider(LLMProvider):
    name = "openrouter"  # the provider every task in tasks.yaml routes to

    def __init__(self, model: str | None = None, *, fail: bool = False):
        super().__init__(model=model or "fake-model")
        self._fail = fail

    async def acomplete(self, messages, *, system=None, **kwargs) -> ProviderResult:
        if self._fail:
            return ProviderResult(
                provider=self.name, model=self.model, content="",
                latency_ms=3, error="simulated failure",
            )
        return ProviderResult(
            provider=self.name, model=self.model, content="the answer",
            latency_ms=7, prompt_tokens=11, completion_tokens=5,
        )


@pytest.fixture(autouse=True)
def _registry():
    original = registry._REGISTRY.copy()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)
    settings_mod.get_settings.cache_clear()


def test_chat_returns_per_agent_results():
    registry.register("openrouter", lambda model: FakeProvider(model))
    with TestClient(server.app) as client:
        data = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "task": SINGLE_AGENT_TASK,
            },
        ).json()

    assert data["reply"] == "the answer"
    assert len(data["agents"]) == 1
    agent = data["agents"][0]
    assert agent["provider"] == SINGLE_AGENT_NAME
    assert agent["content"] == "the answer"
    assert agent["error"] is None
    assert agent["latency_ms"] == 7
    assert agent["prompt_tokens"] == 11
    assert agent["completion_tokens"] == 5


def test_chat_agent_failure_is_visible_per_agent():
    registry.register("openrouter", lambda model: FakeProvider(model, fail=True))
    with TestClient(server.app) as client:
        data = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "task": SINGLE_AGENT_TASK,
            },
        ).json()

    assert data["reply"] == ""
    assert "simulated failure" in (data["error"] or "")
    assert len(data["agents"]) == 1
    assert "simulated failure" in data["agents"][0]["error"]


def test_tasks_endpoint_lists_resolved_line_up():
    with TestClient(server.app) as client:
        data = client.get("/api/tasks").json()

    by_name = {t["name"]: t for t in data}
    assert "default" in by_name
    default = by_name["default"]
    assert default["agents"][0]["provider"] == "openrouter"
    # Model overrides from tasks.yaml resolve to a concrete model id.
    assert default["agents"][0]["model"]
