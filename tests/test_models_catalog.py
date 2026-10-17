"""Free-model catalog tests (app/config/models.py + GET /api/models).

The catalog is what the UI's agent picker offers, so the filter is a spending
and policy control, not a cosmetic one. These tests pin it:

  * paid models never appear, whatever else they look like;
  * only the allowed labs appear;
  * non-chat free models (classifiers, audio) are excluded;
  * the hour-long cache means a second call makes no second request;
  * every failure mode — network error, HTTP error, unrecognizable payload —
    falls back to the checked-in `models.yaml` rather than returning nothing.

No test touches the network: `_fetch_live`'s HTTP call is stubbed.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.config import models as models_mod


def _entry(model_id: str, *, free: bool = True, outputs=("text",)) -> dict:
    """One raw OpenRouter catalog entry, shaped like the live API's."""
    return {
        "id": model_id if free else model_id.replace(":free", ""),
        "name": f"name of {model_id}",
        "context_length": 131072,
        "architecture": {"input_modalities": ["text"], "output_modalities": list(outputs)},
        "top_provider": {"max_completion_tokens": 32768},
    }


CATALOG = [
    _entry("openai/gpt-oss-20b:free"),
    _entry("google/gemma-4-31b-it:free"),
    _entry("nvidia/nemotron-nano-9b-v2:free"),
    # Free, allowed lab, but a classifier rather than a chat model.
    _entry("nvidia/nemotron-3.5-content-safety:free"),
    # Free and US, but audio out — cannot answer a chat turn.
    _entry("google/lyria-3-pro-preview:free", outputs=("audio",)),
    # Free, chat, but not an allowed lab.
    _entry("inclusionai/ling-3.0-flash:free"),
    _entry("cohere/north-mini-code:free"),
    # Allowed lab and chat, but paid.
    _entry("openai/gpt-5-mini:free", free=False),
]


@pytest.fixture(autouse=True)
def _clear_cache():
    models_mod.reset_cache()
    yield
    models_mod.reset_cache()


def _stub_fetch(monkeypatch, entries: list[dict], counter: list[int] | None = None):
    async def _fake_fetch():
        if counter is not None:
            counter.append(1)
        return [models_mod._to_model(e) for e in entries if models_mod._is_offerable(e)]

    monkeypatch.setattr(models_mod, "_fetch_live", _fake_fetch)


def test_catalog_keeps_only_free_chat_models_from_allowed_labs(monkeypatch):
    _stub_fetch(monkeypatch, CATALOG)

    ids = [m.id for m in asyncio.run(models_mod.free_models())]

    assert set(ids) == {
        "openai/gpt-oss-20b:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-nano-9b-v2:free",
    }
    # Nothing paid, ever — the picker cannot become a bill.
    assert all(i.endswith(":free") for i in ids)


def test_catalog_is_grouped_by_lab_in_a_stable_order(monkeypatch):
    _stub_fetch(monkeypatch, CATALOG)

    labs = [m.lab for m in asyncio.run(models_mod.free_models())]

    assert labs == ["openai", "google", "nvidia"]


def test_catalog_is_cached_between_calls(monkeypatch):
    calls: list[int] = []
    _stub_fetch(monkeypatch, CATALOG, counter=calls)

    asyncio.run(models_mod.free_models())
    asyncio.run(models_mod.free_models())

    assert len(calls) == 1


@pytest.mark.parametrize(
    "boom",
    [
        ConnectionError("offline"),
        RuntimeError("500 from openrouter"),
        KeyError("data"),  # payload shape changed
    ],
)
def test_fetch_failures_fall_back_to_the_snapshot(monkeypatch, boom):
    async def _fail():
        raise boom

    monkeypatch.setattr(models_mod, "_fetch_live", _fail)

    got = asyncio.run(models_mod.free_models())

    # The picker degrades to a slightly stale list, never an empty one.
    assert got
    assert got == models_mod._load_fallback() or {m.id for m in got} == {
        m.id for m in models_mod._load_fallback()
    }


def test_empty_live_result_falls_back_too(monkeypatch):
    """A successful fetch that filters down to nothing means the catalog shape
    changed under us — treat it like a failure, not like "no free models"."""
    _stub_fetch(monkeypatch, [])

    assert asyncio.run(models_mod.free_models())


def test_snapshot_passes_the_same_filter():
    """The checked-in fallback must obey the rules the live filter enforces,
    or an outage would quietly offer something the picker never should."""
    for m in models_mod._load_fallback():
        assert m.id.endswith(":free")
        assert m.lab in models_mod.ALLOWED_LABS
        assert m.id not in models_mod.EXCLUDED_IDS


def test_api_models_endpoint_serves_the_catalog(monkeypatch):
    _stub_fetch(monkeypatch, CATALOG)

    with TestClient(server.app) as client:
        data = client.get("/api/models").json()

    assert [m["id"] for m in data] == [
        "openai/gpt-oss-20b:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-nano-9b-v2:free",
    ]
    assert data[0]["lab"] == "openai"
    assert data[0]["context_length"] == 131072
