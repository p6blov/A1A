"""The free-model catalog offered by the UI's agent picker.

OpenRouter's free line-up rotates — models appear, get promoted to paid, or are
retired — so a hardcoded list goes stale. This module fetches the live catalog
from OpenRouter's public `/models` endpoint (no API key required), narrows it to
the models this app is willing to run, and caches the result.

The narrowing is deliberate policy, not a technical limit:

  * `:free` only — a picker that can silently select a paid model is a bill
    waiting to happen.
  * `ALLOWED_LABS` — only models from US labs (OpenAI, Google, NVIDIA).
  * `EXCLUDED_IDS` — free models that are not general-purpose chat models
    (classifiers, audio) and would only confuse the picker.

Any failure falls back to `models.yaml`, a checked-in snapshot of the same
list, so the picker degrades to a slightly stale catalog rather than an empty
one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.config.settings import get_settings

FALLBACK_PATH = Path(__file__).resolve().parent / "models.yaml"

# Model ids are "<lab>/<model>"; the prefix is the lab that made it.
ALLOWED_LABS = ("openai", "google", "nvidia")

# Free, from an allowed lab, text-out — but not something you would put in a
# chat line-up. Kept explicit so the reason survives the next catalog refresh.
EXCLUDED_IDS = frozenset(
    {
        "nvidia/nemotron-3.5-content-safety:free",  # moderation classifier
    }
)

CACHE_TTL_SECONDS = 3600
_FETCH_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class FreeModel:
    id: str
    name: str
    lab: str
    context_length: int | None = None
    max_completion_tokens: int | None = None


# (fetched_at_monotonic, models). Cleared by `reset_cache()` in tests.
_cache: tuple[float, list[FreeModel]] | None = None


def _lab_of(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else ""


def _is_offerable(entry: dict[str, Any]) -> bool:
    """Whether one raw OpenRouter catalog entry belongs in the picker."""
    model_id = entry.get("id") or ""
    if not model_id.endswith(":free") or model_id in EXCLUDED_IDS:
        return False
    if _lab_of(model_id) not in ALLOWED_LABS:
        return False
    # Music/image generators are free and text-in but cannot answer a chat turn.
    outputs = (entry.get("architecture") or {}).get("output_modalities") or []
    return "text" in outputs


def _to_model(entry: dict[str, Any]) -> FreeModel:
    model_id = entry["id"]
    top = entry.get("top_provider") or {}
    return FreeModel(
        id=model_id,
        name=entry.get("name") or model_id,
        lab=_lab_of(model_id),
        context_length=entry.get("context_length"),
        max_completion_tokens=top.get("max_completion_tokens"),
    )


def _load_fallback() -> list[FreeModel]:
    """The checked-in snapshot, used whenever the live fetch cannot be trusted."""
    try:
        with FALLBACK_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return [FreeModel(**m) for m in (raw.get("models") or [])]
    except Exception:
        # A broken snapshot must not take the endpoint down with it.
        return []


async def _fetch_live() -> list[FreeModel]:
    settings = get_settings()
    url = f"{settings.openrouter_base_url.rstrip('/')}/models"
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        entries = resp.json()["data"]
    return [_to_model(e) for e in entries if _is_offerable(e)]


async def free_models() -> list[FreeModel]:
    """The offerable free models, newest-first by lab then id, cached for an hour.

    Never raises: a failed fetch (and a fetch that returns nothing after
    filtering, which means the catalog shape changed) falls back to
    `models.yaml`. The fallback result is cached too, so one outage does not
    turn every request into a doomed round-trip.
    """
    global _cache
    if _cache is not None and time.monotonic() - _cache[0] < CACHE_TTL_SECONDS:
        return _cache[1]

    try:
        models = await _fetch_live()
    except Exception:
        models = []
    if not models:
        models = _load_fallback()

    models.sort(key=lambda m: (ALLOWED_LABS.index(m.lab) if m.lab in ALLOWED_LABS else 99, m.id))
    _cache = (time.monotonic(), models)
    # Hand out a copy: the cached list outlives the request, and a caller that
    # sorted or filtered it in place would corrupt it for the next hour.
    return list(models)


def reset_cache() -> None:
    """Drop the cached catalog (tests, and a future manual refresh endpoint)."""
    global _cache
    _cache = None
