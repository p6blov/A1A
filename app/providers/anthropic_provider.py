"""Anthropic Claude provider implementation.

Uses the official `anthropic` async SDK directly (first-party API, own key) —
not the OpenAI-compatible adapter, since Anthropic's native Messages API is the
supported surface. Like the other providers, permanent failures return a
`ProviderResult` with `error` set instead of raising, so one failing branch
never kills the fan-out.

`astream` uses the SDK's `messages.stream()` helper: it yields text deltas as
they arrive and still accumulates the full message, so stop reason and token
usage come from `get_final_message()` exactly as in the blocking path.
"""
from __future__ import annotations

import time
from typing import Callable

import anthropic

from app.config.settings import get_settings
from app.core.state import ChatMessage
from app.providers.base import LLMProvider, ProviderResult, StreamInterrupted
from app.utils.retry import with_retries

# Transient failures worth retrying; 4xx errors (bad key, bad model id) are
# permanent and fall straight through to the soft-error path.
_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        super().__init__(model=model or settings.anthropic_model)
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example)."
            )
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=float(settings.request_timeout_seconds),
            max_retries=0,  # retries are handled by with_retries below
        )
        self._max_retries = settings.max_retries

    def _wire(self, messages: list[ChatMessage], system: str | None) -> tuple[list, dict]:
        return (
            [{"role": m["role"], "content": m["content"]} for m in messages],
            {"system": system} if system else {},
        )

    async def acomplete(
        self, messages: list[ChatMessage], *, system: str | None = None, **kwargs
    ) -> ProviderResult:
        start = time.perf_counter()
        wire, extra = self._wire(messages, system)

        async def _call():
            # Note: current Opus models reject temperature/top_p — prompt-steer
            # instead, so no sampling params are passed here.
            return await self._client.messages.create(
                model=self.model,
                max_tokens=16000,
                messages=wire,
                **extra,
            )

        try:
            resp = await with_retries(
                _call, max_attempts=self._max_retries, exceptions=_RETRYABLE
            )
            if resp.stop_reason == "refusal":
                return ProviderResult(
                    provider=self.name,
                    model=self.model,
                    content="",
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    error="refusal: the model declined this request",
                )
            content = "".join(b.text for b in resp.content if b.type == "text")
            usage = getattr(resp, "usage", None)
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=int((time.perf_counter() - start) * 1000),
                prompt_tokens=getattr(usage, "input_tokens", None),
                completion_tokens=getattr(usage, "output_tokens", None),
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
        wire, extra = self._wire(messages, system)
        streamed: list[str] = []

        async def _call():
            # A retry after the first token would replay text the client has
            # already rendered, so anything that fails mid-stream is re-raised
            # as the non-retryable StreamInterrupted.
            emitted_before = len(streamed)
            try:
                async with self._client.messages.stream(
                    model=self.model,
                    max_tokens=16000,
                    messages=wire,
                    **extra,
                ) as stream:
                    async for text in stream.text_stream:
                        streamed.append(text)
                        if on_delta:
                            on_delta(text)
                    return await stream.get_final_message()
            except Exception as e:
                if len(streamed) > emitted_before:
                    raise StreamInterrupted(f"{type(e).__name__}: {e}") from e
                raise

        try:
            resp = await with_retries(
                _call, max_attempts=self._max_retries, exceptions=_RETRYABLE
            )
            if resp.stop_reason == "refusal":
                return ProviderResult(
                    provider=self.name,
                    model=self.model,
                    content="",
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    error="refusal: the model declined this request",
                )
            content = "".join(b.text for b in resp.content if b.type == "text")
            usage = getattr(resp, "usage", None)
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=int((time.perf_counter() - start) * 1000),
                prompt_tokens=getattr(usage, "input_tokens", None),
                completion_tokens=getattr(usage, "output_tokens", None),
            )
        except Exception as e:  # permanent failure → soft error, partial text kept
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content="".join(streamed),
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
