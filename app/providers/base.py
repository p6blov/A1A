"""Abstract base class for LLM providers.

Every concrete provider (OpenAI, Anthropic, local Ollama, …) implements the same
`acomplete` coroutine over provider-agnostic `ChatMessage` turns plus an
optional system prompt; each adapter maps them to its vendor's wire format.
The graph only ever talks to this interface — swapping providers is a registry
change, not a code change.

`astream` is the token-streaming twin of `acomplete`: same arguments, same
`ProviderResult`, plus an `on_delta` callback invoked with each text chunk as it
arrives. The default implementation falls back to `acomplete`, so a provider
that has not implemented streaming degrades to a single late "delta" instead of
breaking.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from app.core.state import ChatMessage


class StreamInterrupted(Exception):
    """A stream failed *after* emitting text.

    Retrying would replay the tokens the client already rendered, so providers
    wrap mid-stream failures in this type: it is deliberately absent from every
    provider's `_RETRYABLE` tuple, which makes `with_retries` re-raise it
    immediately onto the usual soft-error path.
    """


class UpstreamError(Exception):
    """The vendor reported a failure inside an otherwise-successful response.

    Not every provider signals failure with a transport error. OpenRouter, for
    one, answers HTTP 200 with an `error` object in the body when the lab behind
    the model rejects the call (rate limits, model cold-start, capacity), and
    Gemini can return a candidate with no text and a `finish_reason` of SAFETY.
    Adapters raise this so those land on the same soft-error path as a 500,
    instead of being mistaken for an empty-but-successful answer.

    Like `StreamInterrupted`, it is absent from every `_RETRYABLE` tuple: the
    upstream already made its decision, so retrying just spends quota.
    """


# What `error` is set to when a provider returns no text and gives no reason.
# The graph ranks a successful-but-empty response ahead of a sibling's real
# answer, so "no content" has to count as a failure, not as an empty success.
EMPTY_RESPONSE_ERROR = "provider returned an empty response"


@dataclass
class ProviderResult:
    provider: str
    model: str
    content: str
    latency_ms: int
    error: str | None = None
    # Token usage reported by the provider, when available (None if the provider
    # did not return usage metadata or the call failed before a response).
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        """Enforce the one invariant every adapter must not forget: a result
        with no text is a failed result.

        `finalize` answers with the first response that has no `error`, so an
        empty string with `error=None` outranks a sibling that actually
        answered — the user sees "(empty response)" while a good answer sits
        unused in the trace. Normalizing here covers every adapter, including
        ones added later, rather than relying on each to remember.
        """
        if self.error is None and not (self.content or "").strip():
            self.error = EMPTY_RESPONSE_ERROR


class LLMProvider(ABC):
    name: str  # short identifier, e.g. "openai"

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def acomplete(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        **kwargs,
    ) -> ProviderResult:
        """Return a single completion for the conversation `messages`.

        `system` is the agent's system prompt (role), when the task config
        assigns one. Implementations must NOT raise on transient errors — wrap
        with retry and return a `ProviderResult` whose `error` field is set on
        permanent failure. This keeps one failing provider from killing the
        whole fan-out.
        """
        raise NotImplementedError

    async def astream(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        **kwargs,
    ) -> ProviderResult:
        """Stream a completion, calling `on_delta` with each text chunk.

        Returns the same `ProviderResult` as `acomplete` — the deltas are a
        side channel for live rendering, not the source of truth. The same
        soft-error contract applies: never raise, set `error` instead. When a
        stream dies partway, implementations return the partial text *and* the
        error, so the client keeps what it already showed.

        This default just delegates to `acomplete` and emits the whole answer
        as one chunk, which is what a non-streaming provider should look like
        to the graph.
        """
        result = await self.acomplete(messages, system=system, **kwargs)
        if on_delta and result.content:
            on_delta(result.content)
        return result
