"""Google Gemini provider implementation.

Wraps the official async client (`google-genai`) and applies a retry policy. On
permanent failure it returns a `ProviderResult` with `error` set instead of
raising — the graph treats that as a soft failure so other providers (M2+) can
still succeed.

`astream` uses `generate_content_stream`, whose chunks carry cumulative usage
metadata, so the token counts come from the last chunk seen.
"""
from __future__ import annotations

import time
from typing import Callable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config.settings import get_settings
from app.core.state import ChatMessage
from app.providers.base import LLMProvider, ProviderResult, StreamInterrupted
from app.utils.retry import with_retries

# Gemini's wire format calls the assistant role "model". Unknown roles fall
# back to "user" rather than erroring — the boundary validates roles.
_ROLE_MAP = {"user": "user", "assistant": "model"}


def to_genai_contents(messages: list[ChatMessage]) -> list[types.Content]:
    """Map provider-agnostic messages to google-genai `Content` objects."""
    return [
        types.Content(
            role=_ROLE_MAP.get(m["role"], "user"),
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]


class GoogleProvider(LLMProvider):
    name = "google"

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        super().__init__(model=model or settings.google_model)
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add a key."
            )
        self._client = genai.Client(api_key=settings.google_api_key)
        self._timeout_ms = settings.request_timeout_seconds * 1000
        self._max_retries = settings.max_retries

    def _config(self, system: str | None, kwargs: dict) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=kwargs.get("temperature", 0.2),
            system_instruction=system,
            http_options=types.HttpOptions(timeout=self._timeout_ms),
        )

    async def acomplete(
        self, messages: list[ChatMessage], *, system: str | None = None, **kwargs
    ) -> ProviderResult:
        start = time.perf_counter()

        async def _call():
            return await self._client.aio.models.generate_content(
                model=self.model,
                contents=to_genai_contents(messages),
                config=self._config(system, kwargs),
            )

        try:
            resp = await with_retries(
                _call,
                max_attempts=self._max_retries,
                exceptions=(genai_errors.APIError,),
            )
            # usage_metadata is present on a successful response; guard each field
            # since the API may omit counts for some models/responses.
            usage = getattr(resp, "usage_metadata", None)
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content=resp.text or "",
                latency_ms=int((time.perf_counter() - start) * 1000),
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
            )
        except Exception as e:  # permanent failure → soft error
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content="",
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )

    async def astream(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        **kwargs,
    ) -> ProviderResult:
        start = time.perf_counter()
        streamed: list[str] = []

        async def _call():
            # A retry after the first token would replay text the client has
            # already rendered, so anything that fails mid-stream is re-raised
            # as the non-retryable StreamInterrupted.
            emitted_before = len(streamed)
            usage = None
            try:
                # The aio variant is a coroutine that resolves to the chunk
                # iterator, hence the await inside the `async for`.
                chunks = await self._client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=to_genai_contents(messages),
                    config=self._config(system, kwargs),
                )
                async for chunk in chunks:
                    # Usage is cumulative; the last chunk carries the totals.
                    if getattr(chunk, "usage_metadata", None):
                        usage = chunk.usage_metadata
                    text = chunk.text
                    if text:
                        streamed.append(text)
                        if on_delta:
                            on_delta(text)
            except Exception as e:
                if len(streamed) > emitted_before:
                    raise StreamInterrupted(f"{type(e).__name__}: {e}") from e
                raise
            return usage

        try:
            usage = await with_retries(
                _call,
                max_attempts=self._max_retries,
                exceptions=(genai_errors.APIError,),
            )
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content="".join(streamed),
                latency_ms=int((time.perf_counter() - start) * 1000),
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
            )
        except Exception as e:  # permanent failure → soft error, partial text kept
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content="".join(streamed),
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
