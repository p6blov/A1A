"""Tenacity-based async retry helper.

Used to wrap provider calls so transient network/5xx errors don't take down the
whole graph. Permanent failures bubble up as a returned error string, never as a
raised exception escaping the node.
"""
from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Run `fn` with exponential backoff. Re-raises the last exception on failure."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    ):
        with attempt:
            return await fn()
    # Unreachable, but satisfies the type checker.
    raise RuntimeError("with_retries exited without returning")
