"""M3 summarizer tests.

Verifies that:
  * A task with a `summarizer` block gets a synthesized final answer; the raw
    per-agent responses are retained and the summarizer run is reported.
  * A failing summarizer falls back to the first successful response — it can
    never lose the agents' answers.
  * The summarizer is skipped (no wasted call) when fewer than two branches
    succeeded, when the task has no summarizer, or when the summarizer's
    provider is unconfigured.
  * A client-supplied summarizer (the UI's agent panel) overrides the task's.
  * Branch responses carry the *agent* name, not the provider class's vendor
    name — several agents share one vendor now that the line-up is all
    OpenRouter, and finalize's config-order ranking depends on telling them
    apart.
  * /api/chat surfaces the summarizer run alongside the agents.

Fixtures replace providers with fakes so no test touches the network. One fake
is registered for `openrouter` and keys its behavior on the *model*, since that
is what distinguishes the branches. A synthesis call is recognized by the
summarizer's system prompt.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.graph import nodes
from app.graph.builder import build_graph
from app.providers import registry
from app.providers.base import LLMProvider, ProviderResult

SYNTH_MARKER = "Multiple AI agents answered"

# tasks.yaml's `research` line-up, in config order, plus its summarizer.
RESEARCHER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SKEPTIC_MODEL = "google/gemma-4-31b-it:free"
SUMMARIZER_MODEL = "nvidia/nemotron-nano-9b-v2:free"
# tasks.yaml's `default` line-up, first agent.
DEFAULT_FIRST_MODEL = "openai/gpt-oss-20b:free"


class FakeProvider(LLMProvider):
    """Fake with separate behavior for branch calls vs. the synthesis call.

    Appends ("branch"|"synth", model) to the shared `calls` list so tests can
    assert exactly which calls happened. `failing` names the models whose
    branch call should soft-fail.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        failing: set[str] | None = None,
        synth_reply: str = "merged answer",
        synth_fail: bool = False,
        calls: list | None = None,
    ):
        super().__init__(model=model or "fake-model")
        self.name = "openrouter"
        self._failing = failing or set()
        self._synth_reply = synth_reply
        self._synth_fail = synth_fail
        self._calls = calls if calls is not None else []

    async def acomplete(self, messages, *, system=None, **kwargs) -> ProviderResult:
        is_synth = bool(system) and SYNTH_MARKER in system
        self._calls.append(("synth" if is_synth else "branch", self.model))
        if self._synth_fail if is_synth else self.model in self._failing:
            return ProviderResult(
                provider=self.name, model=self.model, content="",
                latency_ms=1, error="simulated failure",
            )
        return ProviderResult(
            provider=self.name, model=self.model,
            content=self._synth_reply if is_synth else f"answer from {self.model}",
            latency_ms=1,
        )


def _register(**kwargs) -> None:
    registry.register("openrouter", lambda m: FakeProvider(m, **kwargs))


@pytest.fixture(autouse=True)
def _restore_registry():
    original = registry._REGISTRY.copy()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)


def _run(task: str = "research", **state) -> dict:
    graph = build_graph()
    return asyncio.run(graph.ainvoke(
        {"messages": [{"role": "user", "content": "ping"}], "task": task,
         "responses": [], "errors": [], **state}
    ))


def test_research_task_synthesizes():
    calls: list = []
    _register(calls=calls)

    final = _run("research")

    # research's summarizer merged the two answers...
    assert final["final_answer"] == "merged answer"
    assert final["summary"] == "merged answer"
    assert calls.count(("synth", SUMMARIZER_MODEL)) == 1
    # ...and the raw responses (under their role names) plus the summarizer
    # run are all reported.
    assert {r["provider"] for r in final["responses"]} == {"researcher", "skeptic"}
    assert final["summarizer_run"]["model"] == SUMMARIZER_MODEL
    assert not final.get("errors")


def test_failing_summarizer_falls_back_to_first_successful():
    _register(synth_fail=True)

    final = _run("research")

    assert final["final_answer"] == f"answer from {RESEARCHER_MODEL}"
    assert final.get("summary") is None
    assert final["summarizer_run"]["error"] == "simulated failure"
    assert any("summarizer" in e for e in final["errors"])


def test_single_successful_branch_skips_summarizer():
    calls: list = []
    _register(failing={SKEPTIC_MODEL}, calls=calls)

    final = _run("research")

    # One good answer: nothing to merge, no extra call spent.
    assert final["final_answer"] == f"answer from {RESEARCHER_MODEL}"
    assert final.get("summarizer_run") is None
    assert all(kind == "branch" for kind, _ in calls)


def test_task_without_summarizer_keeps_first_successful():
    calls: list = []
    _register(calls=calls)

    final = _run("default")

    assert final["final_answer"] == f"answer from {DEFAULT_FIRST_MODEL}"
    assert final.get("summary") is None
    assert all(kind == "branch" for kind, _ in calls)


def test_unconfigured_summarizer_provider_is_skipped(monkeypatch):
    cfg = {"summarizer": {"provider": "paid", "model": "paid-model"}}
    monkeypatch.setattr(nodes, "load_task_config", lambda task: cfg)
    _register()

    def _explode(_m):
        raise AssertionError("unconfigured summarizer must not be constructed")

    registry.register("paid", _explode, is_configured=lambda: False)

    final = _run("research")

    assert final["final_answer"] == f"answer from {RESEARCHER_MODEL}"
    assert final.get("summarizer_run") is None


CUSTOM_LINEUP = [
    {"name": "optimist", "provider": "openrouter", "model": "model-a",
     "system": "Be optimistic."},
    {"name": "pessimist", "provider": "openrouter", "model": "model-b",
     "system": "Be pessimistic."},
]


def test_request_summarizer_overrides_the_tasks():
    """The agent panel's synthesis settings win over tasks.yaml for that turn."""
    calls: list = []
    _register(calls=calls)

    final = _run(
        "research",
        agent_specs=CUSTOM_LINEUP,
        summarizer_cfg={"provider": "openrouter", "model": "chosen-by-the-user"},
    )

    assert final["final_answer"] == "merged answer"
    assert final["summarizer_run"]["model"] == "chosen-by-the-user"
    assert calls.count(("synth", SUMMARIZER_MODEL)) == 0


def test_custom_lineup_without_a_summarizer_suppresses_the_tasks():
    """Switching synthesis off in the panel must actually switch it off, even on
    a task that configures a summarizer — otherwise the user pays for a call
    they declined (and one the API's `start` event said would not happen)."""
    calls: list = []
    _register(calls=calls)

    final = _run("research", agent_specs=CUSTOM_LINEUP, summarizer_cfg=None)

    assert final.get("summarizer_run") is None
    assert all(kind == "branch" for kind, _ in calls)
    assert final["final_answer"] == "answer from model-a"


def test_request_summarizer_applies_to_a_custom_lineup():
    """A roster from the panel synthesizes even on a task that never would."""
    _register()

    final = _run(
        "default",
        agent_specs=CUSTOM_LINEUP,
        summarizer_cfg={"provider": "openrouter", "model": "model-c", "system": None},
    )

    assert {r["provider"] for r in final["responses"]} == {"optimist", "pessimist"}
    assert final["final_answer"] == "merged answer"
    assert final["summarizer_run"]["model"] == "model-c"


def test_responses_carry_agent_name_not_vendor_name():
    # Every agent self-reports the vendor name "openrouter"; the trace and the
    # config-order ranking must still key on the agent name from tasks.yaml.
    _register()

    final = _run("default")

    assert {r["provider"] for r in final["responses"]} == {"gpt-oss", "nemotron"}
    assert all(r["model"] != r["provider"] for r in final["responses"])
    # First successful in task-config order = the `gpt-oss` agent.
    assert final["final_answer"] == f"answer from {DEFAULT_FIRST_MODEL}"


def test_api_chat_returns_summarizer():
    _register()
    with TestClient(server.app) as client:
        data = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "task": "research"},
        ).json()

    assert data["reply"] == "merged answer"
    assert {a["provider"] for a in data["agents"]} == {"researcher", "skeptic"}
    assert data["summarizer"]["model"] == SUMMARIZER_MODEL
    assert data["summarizer"]["error"] is None
