"""Generic provider for any OpenAI-compatible chat-completions endpoint.

OpenRouter is the primary target today and OpenAI's own API the fallback; many
other vendors speak the same protocol, so one class covers them all: the
registry decides the name, base URL, key, default model, and any vendor-
specific headers. Swapping vendors — or moving from a free model to a paid one
— is a settings change, never a code change.

Like GoogleProvider, permanent failures return a `ProviderResult` with `error`
set instead of raising, so one failing branch never kills the fan-out.

`astream` sets `stream=True` plus `stream_options={"include_usage": True}` —
without that option a streamed chat-completion reports no token usage at all.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import openai

from app.core.state import ChatMessage
from app.providers.base import (
    LLMProvider,
    ProviderResult,
    StreamInterrupted,
    UpstreamError,
)
from app.utils.retry import with_retries

# Transient failures worth retrying; 4xx errors (bad key, bad model id) are
# permanent and fall straight through to the soft-error path.
_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _format_error(err: object) -> str | None:
    """Render a vendor `error` value as one readable line."""
    if not err:
        return None
    if isinstance(err, dict):
        message = err.get("message") or str(err)
        code = err.get("code")
        return f"{message} (code {code})" if code is not None else str(message)
    return str(err)


def _payload_error(obj: object) -> str | None:
    """Extract a vendor error delivered inside a 200 response, if any.

    OpenRouter answers HTTP 200 with `{"error": {"code", "message"}}` (and no
    `choices`) when the lab behind the model refuses the call — upstream rate
    limits, capacity, a cold model. The SDK does not treat that as an error: it
    parses a `ChatCompletion` with `choices=None` and keeps the error object in
    `model_extra`, so without this check the branch looks like a successful
    empty answer. Free models make this the *likely* failure mode, not an
    exotic one, since the daily quota is small.
    """
    err = getattr(obj, "error", None)
    if err is None:
        extra = getattr(obj, "model_extra", None) or {}
        err = extra.get("error")
    return _format_error(err)


async def _unstreamed_body_error(stream: object) -> str | None:
    """Recover the error from a stream request the vendor answered without a stream.

    When OpenRouter fails *before* the stream opens (the daily free-model quota
    is the common one) it replies HTTP 200 with a plain JSON error body instead
    of `text/event-stream`. The SDK does not raise; it hands back an
    `AsyncStream` that simply yields nothing, so the branch would otherwise be
    reported as an empty answer and the actual reason — the one telling the user
    to wait or add credits — would be lost.

    Reading the body is safe precisely because it is not a stream: iterating it
    produces no chunks either way.
    """
    resp = getattr(stream, "response", None)
    if resp is None or "text/event-stream" in resp.headers.get("content-type", ""):
        return None
    try:
        payload = json.loads(await resp.aread())
    except Exception:
        return None  # unparseable body: fall through to the empty-response path
    if not isinstance(payload, dict):
        return None
    return _format_error(payload.get("error"))


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_seconds: int,
        max_retries: int,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model=model)
        self.name = name
        if not api_key:
            raise RuntimeError(
                f"{name.upper()}_API_KEY is not set. Add it to .env (see .env.example)."
            )
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
            # Vendor-specific extras (e.g. OpenRouter's attribution headers).
            # Sent on every request; never carries anything about the end user.
            default_headers=default_headers,
        )
        self._max_retries = max_retries

    def _wire(self, messages: list[ChatMessage], system: str | None) -> list[dict]:
        # Chat-completions expresses the system prompt as a leading message.
        return ([{"role": "system", "content": system}] if system else []) + [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

    async def acomplete(
        self, messages: list[ChatMessage], *, system: str | None = None, **kwargs
    ) -> ProviderResult:
        start = time.perf_counter()
        wire = self._wire(messages, system)

        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=wire,
                temperature=kwargs.get("temperature", 0.2),
            )

        try:
            resp = await with_retries(
                _call, max_attempts=self._max_retries, exceptions=_RETRYABLE
            )
            vendor_error = _payload_error(resp)
            if vendor_error:
                raise UpstreamError(vendor_error)
            usage = getattr(resp, "usage", None)
            content = resp.choices[0].message.content if resp.choices else ""
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content=content or "",
                latency_ms=int((time.perf_counter() - start) * 1000),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
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
        wire = self._wire(messages, system)
        streamed: list[str] = []

        async def _call():
            # A retry after the first token would replay text the client has
            # already rendered, so anything that fails mid-stream is re-raised
            # as the non-retryable StreamInterrupted.
            emitted_before = len(streamed)
            usage = None
            try:
                stream = await self._client.chat.completions.create(
                    model=self.model,
                    messages=wire,
                    temperature=kwargs.get("temperature", 0.2),
                    stream=True,
                    # Streamed responses omit usage unless it is asked for; a
                    # vendor that ignores the option simply leaves it None.
                    stream_options={"include_usage": True},
                )
                # A stream request answered with a plain body never yields a
                # chunk, so check for that before the loop silently ends.
                prestream_error = await _unstreamed_body_error(stream)
                if prestream_error:
                    raise UpstreamError(prestream_error)
                async for chunk in stream:
                    # A stream can carry the same in-band error object as a
                    # non-streamed reply. The SDK raises on most of those, but
                    # not all shapes, so check before trusting the chunk.
                    chunk_error = _payload_error(chunk)
                    if chunk_error:
                        raise UpstreamError(chunk_error)
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if not chunk.choices:
                        continue  # the usage-only final chunk carries no choice
                    text = chunk.choices[0].delta.content
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
                _call, max_attempts=self._max_retries, exceptions=_RETRYABLE
            )
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content="".join(streamed),
                latency_ms=int((time.perf_counter() - start) * 1000),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )
        except Exception as e:  # permanent failure → soft error, partial text kept
            return ProviderResult(
                provider=self.name,
                model=self.model,
                content="".join(streamed),
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
