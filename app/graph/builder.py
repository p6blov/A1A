"""LangGraph assembly.

Current graph (M3 — parallel fan-out + synthesis):

    START → redact → [call_provider × N in parallel] → summarize → finalize → END

The provider list is chosen per-request from the task config, so the fan-out
is a conditional edge emitting `Send` branches rather than static edges: the
graph is compiled once at startup and still routes dynamically. `summarize`
runs once after every branch has appended to `state["responses"]`; it calls
the task's configured summarizer to merge the answers, or passes through when
the task has none (then `finalize` answers with the first successful response).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.core.state import OrchestratorState
from app.graph.nodes import (
    call_provider_node,
    fan_out_providers,
    finalize_node,
    redact_node,
    summarize_node,
)


def build_graph():
    g: StateGraph = StateGraph(OrchestratorState)
    g.add_node("redact", redact_node)
    g.add_node("call_provider", call_provider_node)
    g.add_node("summarize", summarize_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "redact")
    # Dynamic fan-out: fan_out_providers returns one Send per configured agent.
    g.add_conditional_edges("redact", fan_out_providers, ["call_provider"])
    g.add_edge("call_provider", "summarize")
    g.add_edge("summarize", "finalize")
    g.add_edge("finalize", END)

    return g.compile()
