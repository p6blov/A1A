"""Provider registry.

Mapping from short name (as used in tasks.yaml) to a factory that builds the
provider, plus per-provider metadata:

  * `is_configured` — whether the credentials for this provider are present.
    The fan-out router skips unconfigured providers so a missing optional key
    degrades to fewer branches instead of failing requests, and /api/tasks
    reports the same line-up to the UI.
  * `default_model` — the model used when tasks.yaml has no override, so the
    UI can display the concrete model an agent would run.

Tests can override entries here to inject fakes without monkey-patching the
rest of the app; `register(name, factory)` marks the entry as configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.config.settings import get_settings
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import LLMProvider
from app.providers.google_provider import GoogleProvider
from app.providers.openai_compat import OpenAICompatProvider

# Factory signature: (model_override: str | None) -> LLMProvider
ProviderFactory = Callable[[str | None], LLMProvider]


@dataclass
class ProviderEntry:
    factory: ProviderFactory
    is_configured: Callable[[], bool] = field(default=lambda: True)
    default_model: Callable[[], str | None] = field(default=lambda: None)


def _make_openrouter(model: str | None) -> OpenAICompatProvider:
    """OpenRouter (openrouter.ai), which speaks the OpenAI wire protocol.

    One key fronts models from several labs, so the model id — not the provider
    name — decides which lab answers. A task or a client-supplied roster picks
    the id; unset falls back to `OPENROUTER_MODEL`.
    """
    s = get_settings()
    return OpenAICompatProvider(
        name="openrouter",
        base_url=s.openrouter_base_url,
        api_key=s.openrouter_api_key,
        model=model or s.openrouter_model,
        timeout_seconds=s.request_timeout_seconds,
        max_retries=s.max_retries,
        default_headers={
            "HTTP-Referer": s.openrouter_site_url,
            "X-Title": s.openrouter_app_title,
        },
    )


def _make_openai(model: str | None) -> OpenAICompatProvider:
    """OpenAI, direct (api.openai.com) via the official openai SDK."""
    s = get_settings()
    return OpenAICompatProvider(
        name="openai",
        base_url=s.openai_base_url,
        api_key=s.openai_api_key,
        model=model or s.openai_model,
        timeout_seconds=s.request_timeout_seconds,
        max_retries=s.max_retries,
    )


_REGISTRY: dict[str, ProviderEntry] = {
    # The default line-up runs entirely here. Unlike the direct adapters, one
    # entry is enough for a multi-model fan-out: a task can list `openrouter`
    # several times under different agent names with different `model`s.
    "openrouter": ProviderEntry(
        factory=_make_openrouter,
        is_configured=lambda: bool(get_settings().openrouter_api_key),
        default_model=lambda: get_settings().openrouter_model,
    ),
    "google": ProviderEntry(
        factory=lambda model: GoogleProvider(model=model),
        is_configured=lambda: bool(get_settings().google_api_key),
        default_model=lambda: get_settings().google_model,
    ),
    # A second Gemini model on the same key — a free fan-out that does not go
    # through OpenRouter.
    "google-flash": ProviderEntry(
        factory=lambda model: GoogleProvider(
            model=model or get_settings().google_secondary_model
        ),
        is_configured=lambda: bool(get_settings().google_api_key),
        default_model=lambda: get_settings().google_secondary_model,
    ),
    "openai": ProviderEntry(
        factory=_make_openai,
        is_configured=lambda: bool(get_settings().openai_api_key),
        default_model=lambda: get_settings().openai_model,
    ),
    "anthropic": ProviderEntry(
        factory=lambda model: AnthropicProvider(model=model),
        is_configured=lambda: bool(get_settings().anthropic_api_key),
        default_model=lambda: get_settings().anthropic_model,
    ),
}


def register(
    name: str,
    factory: ProviderFactory,
    *,
    is_configured: Callable[[], bool] | None = None,
    default_model: Callable[[], str | None] | None = None,
) -> None:
    """Register or override a provider factory. Used by tests and new providers.
    Without explicit metadata the entry counts as configured."""
    _REGISTRY[name] = ProviderEntry(
        factory=factory,
        is_configured=is_configured or (lambda: True),
        default_model=default_model or (lambda: None),
    )


def get_provider(name: str, model: str | None = None) -> LLMProvider:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown provider: {name!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name].factory(model)


def is_configured(name: str) -> bool:
    entry = _REGISTRY.get(name)
    return entry is not None and entry.is_configured()


def default_model(name: str) -> str | None:
    entry = _REGISTRY.get(name)
    return entry.default_model() if entry else None


def available_providers() -> list[str]:
    return sorted(_REGISTRY)
