"""API-boundary redaction tests.

Verifies that POST /api/chat scrubs PII from *only the newest* user message,
returns the redacted text to the client, and leaves prior (already-redacted)
history untouched. A fake provider echoes the prompt back so we can assert
exactly what the model would have received.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.config import settings as settings_mod
from app.providers import registry
from app.providers.base import LLMProvider, ProviderResult


class EchoProvider(LLMProvider):
    name = "openrouter"

    def __init__(self, model: str | None = None):
        super().__init__(model=model or "fake-model")

    async def acomplete(self, messages, *, system=None, **kwargs) -> ProviderResult:
        # Echo the conversation so the test can inspect what the model received.
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return ProviderResult(
            provider=self.name, model=self.model, content=text, latency_ms=1
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Force redaction on and a clean registry, restore both afterwards."""
    monkeypatch.setenv("ENABLE_REDACTION", "true")
    settings_mod.get_settings.cache_clear()
    original = registry._REGISTRY.copy()
    registry.register("openrouter", lambda model: EchoProvider(model))
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original)
    settings_mod.get_settings.cache_clear()


def _post(messages):
    with TestClient(server.app) as client:
        # code_review runs a single agent, so `reply` is one clean echo.
        return client.post(
            "/api/chat", json={"messages": messages, "task": "code_review"}
        ).json()


def test_only_newest_user_message_is_redacted():
    data = _post(
        [
            {"role": "user", "content": "earlier i mentioned [EMAIL]"},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "my real email is alice@example.com"},
        ]
    )
    # Client gets the scrubbed newest message back to carry forward.
    assert data["redacted_user_message"] == "my real email is [EMAIL]"
    # The model never saw the raw address...
    assert "alice@example.com" not in data["reply"]
    assert "[EMAIL]" in data["reply"]
    # ...and the already-redacted history was passed through verbatim.
    assert "earlier i mentioned [EMAIL]" in data["reply"]


def test_assistant_content_is_not_redacted():
    # PII-looking text in an assistant turn must survive (we only scrub the
    # newest user message).
    data = _post(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "example: support@acme.com"},
            {"role": "user", "content": "thanks"},
        ]
    )
    assert "support@acme.com" in data["reply"]
