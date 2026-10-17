"""Per-request agent line-up (the UI's agent panel).

The panel sends its roster on every chat request, so `_resolve_overrides` is a
trust boundary: it decides what a browser can make this server do upstream.
These tests pin that contract.

  * A valid roster replaces the task's line-up, names and all.
  * Each agent's system prompt reaches its own branch.
  * The roster is capped at `max_custom_agents` — one request cannot fan out
    arbitrarily many paid calls.
  * Duplicate agent names are made unique, because the whole trace, the error
    attribution, and finalize's ranking key on them.
  * Oversized system prompts are truncated rather than forwarded.
  * Unknown providers are dropped; a roster that survives to nothing falls back
    to the task preset instead of producing an empty fan-out.
  * When *nothing* is configured the roster is kept, so the branches soft-fail
    with an error naming the missing key.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.config import settings as settings_mod
from app.providers import registry
from app.providers.base import LLMProvider, ProviderResult


class RecordingProvider(LLMProvider):
    def __init__(self, model: str | None = None, *, calls: list):
        super().__init__(model=model or "fake-model")
        self.name = "openrouter"
        self._calls = calls

    async def acomplete(self, messages, *, system=None, **kwargs) -> ProviderResult:
        self._calls.append({"model": self.model, "system": system})
        return ProviderResult(
            provider=self.name, model=self.model,
            content=f"answer from {self.model}", latency_ms=1,
        )


@pytest.fixture
def calls():
    return []


@pytest.fixture(autouse=True)
def _registry(calls):
    original = registry._REGISTRY.copy()
    registry.register("openrouter", lambda m: RecordingProvider(m, calls=calls))
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)
    settings_mod.get_settings.cache_clear()


def _post(**body) -> dict:
    with TestClient(server.app) as client:
        return client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}], **body},
        ).json()


def _agent(name: str, model: str, system: str | None = None) -> dict:
    spec = {"name": name, "provider": "openrouter", "model": model}
    if system is not None:
        spec["system"] = system
    return spec


def test_roster_replaces_the_tasks_line_up(calls):
    data = _post(
        task="research",  # would run researcher/skeptic + a summarizer
        agents=[_agent("brief", "model-a"), _agent("detailed", "model-b")],
    )

    assert {a["provider"] for a in data["agents"]} == {"brief", "detailed"}
    assert {c["model"] for c in calls} == {"model-a", "model-b"}
    assert data["reply"] == "answer from model-a"  # first in roster order


def test_each_agents_system_prompt_reaches_its_branch(calls):
    _post(
        agents=[
            _agent("optimist", "model-a", "Be optimistic."),
            _agent("pessimist", "model-b", "Be pessimistic."),
        ]
    )

    assert {c["model"]: c["system"] for c in calls} == {
        "model-a": "Be optimistic.",
        "model-b": "Be pessimistic.",
    }


def test_roster_is_capped_at_max_custom_agents(calls):
    cap = settings_mod.get_settings().max_custom_agents

    data = _post(agents=[_agent(f"a{i}", f"model-{i}") for i in range(cap + 4)])

    assert len(data["agents"]) == cap
    assert len(calls) == cap


def test_duplicate_agent_names_are_made_unique(calls):
    # The trace, error attribution, and ranking all key on the agent name, so
    # two agents may not share one.
    data = _post(agents=[_agent("twin", "model-a"), _agent("twin", "model-b")])

    names = [a["provider"] for a in data["agents"]]
    assert len(set(names)) == 2
    assert "twin" in names
    assert len(calls) == 2


def test_blank_agent_name_falls_back_to_the_model(calls):
    data = _post(agents=[{"provider": "openrouter", "model": "model-a"}])

    assert [a["provider"] for a in data["agents"]] == ["model-a"]


def test_oversized_system_prompt_is_truncated(calls):
    limit = server.MAX_SYSTEM_PROMPT_CHARS

    _post(agents=[_agent("verbose", "model-a", "x" * (limit + 500))])

    assert len(calls[0]["system"]) == limit


def test_unknown_provider_is_dropped(calls):
    data = _post(
        agents=[_agent("real", "model-a"), {"name": "fake", "provider": "nope", "model": "m"}]
    )

    assert [a["provider"] for a in data["agents"]] == ["real"]


def test_roster_of_only_unknown_providers_falls_back_to_the_task(calls):
    # An empty fan-out is never an acceptable outcome, so the preset takes over.
    data = _post(
        task="code_review",
        agents=[{"name": "fake", "provider": "nope", "model": "m"}],
    )

    assert [a["provider"] for a in data["agents"]] == ["reviewer"]


def test_empty_roster_leaves_the_task_in_charge(calls):
    data = _post(task="code_review", agents=[])

    assert [a["provider"] for a in data["agents"]] == ["reviewer"]


def test_unconfigured_provider_is_kept_so_the_error_names_the_key():
    def _raise(_m):
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    registry.register("openrouter", _raise, is_configured=lambda: False)

    data = _post(agents=[_agent("mine", "model-a")])

    # Better a branch that soft-fails with a diagnosable error than a silent
    # answer from a preset the user did not choose.
    assert [a["provider"] for a in data["agents"]] == ["mine"]
    assert "OPENROUTER_API_KEY" in data["agents"][0]["error"]


def test_roster_summarizer_synthesizes(calls):
    data = _post(
        task="default",  # has no summarizer of its own
        agents=[_agent("a", "model-a"), _agent("b", "model-b")],
        summarizer={"provider": "openrouter", "model": "model-c"},
    )

    assert data["summarizer"]["model"] == "model-c"
    assert "model-c" in {c["model"] for c in calls}


def test_summarizer_without_a_roster_is_ignored(calls):
    """The summarizer field only means something alongside a line-up; on its own
    it must not bolt synthesis onto a task that does not synthesize."""
    data = _post(task="default", summarizer={"provider": "openrouter", "model": "model-c"})

    assert data["summarizer"] is None
    assert "model-c" not in {c["model"] for c in calls}
