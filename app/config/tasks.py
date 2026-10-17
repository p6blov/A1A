"""Task configuration loading (tasks.yaml).

Shared by the graph nodes (to route a request) and the API (to describe the
available tasks and their agent line-up to the frontend). The parsed file is
cached and invalidated on mtime, so edits to tasks.yaml are picked up without
a restart but steady-state requests never touch the disk.

A task's line-up normalizes to a list of `AgentSpec` regardless of which of
the two config shapes declared it:

  * `agents:` — a list of {name?, provider, model?, system?} entries. Named
    roles; the same provider may appear under several names.
  * `providers:` + `model_overrides:` — the legacy shorthand; each provider
    becomes an agent named after it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TASKS_PATH = Path(__file__).resolve().parent / "tasks.yaml"


@dataclass(frozen=True)
class AgentSpec:
    """One agent of a task: which provider/model runs it and under which role.

    `name` is the agent's identity everywhere downstream — trace chips, error
    attribution, telemetry, and finalize's config-order ranking all key on it.
    """

    name: str
    provider: str
    model: str | None = None
    system: str | None = None

_cache: tuple[float, dict[str, Any]] | None = None


def load_tasks() -> dict[str, Any]:
    """Return the full task table, keyed by task name."""
    global _cache
    mtime = TASKS_PATH.stat().st_mtime
    if _cache is None or _cache[0] != mtime:
        with TASKS_PATH.open("r", encoding="utf-8") as f:
            _cache = (mtime, yaml.safe_load(f) or {})
    return _cache[1]


def load_task_config(task: str) -> dict[str, Any]:
    """Look up one task definition; fall back to `default` if unknown."""
    cfg = load_tasks()
    return cfg.get(task) or cfg.get("default") or {}


def load_task_agents(task: str) -> list[AgentSpec]:
    """The task's declared agent line-up, normalized (no availability filtering
    — that is the router's concern, see `resolve_task_agents`)."""
    cfg = load_task_config(task)
    if cfg.get("agents"):
        specs = [
            AgentSpec(
                name=a.get("name") or a["provider"],
                provider=a["provider"],
                model=a.get("model"),
                system=a.get("system"),
            )
            for a in cfg["agents"]
        ]
    else:
        overrides: dict[str, str] = cfg.get("model_overrides") or {}
        specs = [
            AgentSpec(name=p, provider=p, model=overrides.get(p))
            for p in (cfg.get("providers") or ["google"])
        ]

    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"task {task!r}: duplicate agent name(s) {dupes} — give each agent "
            "a unique `name` (it defaults to the provider)"
        )
    return specs
