"""Wire-level behavior of the OpenAI-compatible adapter against OpenRouter.

OpenRouter is not quite the OpenAI API, and the differences are exactly the
cases that silently produced blank answers before:

  * It answers **HTTP 200 with an `error` object** when the lab behind the
    model refuses the call. The SDK does not raise; it parses a
    `ChatCompletion` with `choices=None` and keeps the error in `model_extra`.
    On free models — 50 requests/day — this is the *expected* failure mode.
  * A response can carry no text at all (a reasoning model that emitted only
    `reasoning`, a filtered completion). With `error=None` that outranks a
    sibling's real answer in `finalize`, so the user sees "(empty response)"
    while a good answer sits unused in the trace.

Both must land on the soft-error path. These tests use stand-ins shaped like
the SDK's parsed objects rather than real sockets, so they stay fast.
"""
from __future__ import annotations

import pytest

import asyncio
import json

from app.providers.base import EMPTY_RESPONSE_ERROR, ProviderResult
from app.providers.openai_compat import _payload_error, _unstreamed_body_error


class FakeCompletion:
    """Shaped like the SDK's ChatCompletion: unknown top-level keys land in
    `model_extra`, and `choices` is None when the vendor sent none."""

    def __init__(self, *, choices=None, extra: dict | None = None):
        self.choices = choices
        self.model_extra = extra or {}
        if extra and "error" in extra:
            self.error = extra["error"]


# ---------------------------------------------------------------------------
# In-band vendor errors
# ---------------------------------------------------------------------------
def test_error_object_in_a_200_response_is_extracted():
    resp = FakeCompletion(extra={"error": {"code": 429, "message": "rate limited upstream"}})

    assert _payload_error(resp) == "rate limited upstream (code 429)"


def test_error_alongside_an_empty_choices_list_is_still_found():
    resp = FakeCompletion(choices=[], extra={"error": {"code": 502, "message": "upstream died"}})

    assert _payload_error(resp) == "upstream died (code 502)"


def test_error_without_a_code_keeps_the_message():
    assert _payload_error(FakeCompletion(extra={"error": {"message": "no endpoints"}})) == (
        "no endpoints"
    )


def test_non_dict_error_is_stringified():
    assert _payload_error(FakeCompletion(extra={"error": "plain string failure"})) == (
        "plain string failure"
    )


def test_a_healthy_response_reports_no_error():
    assert _payload_error(FakeCompletion(choices=[object()])) is None
    # Absent `model_extra` entirely (older/other SDK objects) must not blow up.
    assert _payload_error(object()) is None


def test_empty_error_value_is_not_treated_as_an_error():
    # `{"error": null}` appears in some healthy payloads; it means "no error".
    assert _payload_error(FakeCompletion(choices=[object()], extra={"error": None})) is None


# ---------------------------------------------------------------------------
# A stream request answered without a stream
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, content_type: str, body: bytes):
        self.headers = {"content-type": content_type}
        self._body = body

    async def aread(self) -> bytes:
        return self._body


class FakeStream:
    """Shaped like the SDK's AsyncStream, which exposes the raw httpx response."""

    def __init__(self, response=None):
        self.response = response


def _unstreamed(content_type: str, payload) -> str | None:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return asyncio.run(_unstreamed_body_error(FakeStream(FakeResponse(content_type, body))))


def test_json_error_answering_a_stream_request_is_recovered():
    """The daily free-model quota fails *before* the stream opens: HTTP 200 with
    a JSON body. The SDK yields no chunks and does not raise, so without this
    the user would see "empty response" instead of "come back tomorrow"."""
    got = _unstreamed(
        "application/json",
        {"error": {"code": 429, "message": "free-models-per-day exceeded"}},
    )

    assert got == "free-models-per-day exceeded (code 429)"


def test_a_real_event_stream_is_left_untouched():
    # Reading the body here would consume the stream before it is iterated.
    assert _unstreamed("text/event-stream", {"error": {"message": "ignored"}}) is None


def test_a_json_body_without_an_error_is_not_invented():
    assert _unstreamed("application/json", {"choices": []}) is None


def test_an_unparseable_body_falls_through_quietly():
    # Not JSON at all (an HTML error page from a proxy, say) — the
    # empty-response guard still catches the branch.
    assert _unstreamed("text/html", b"<html>502 Bad Gateway</html>") is None


def test_a_stream_object_without_a_response_is_handled():
    assert asyncio.run(_unstreamed_body_error(FakeStream())) is None
    assert asyncio.run(_unstreamed_body_error(object())) is None


# ---------------------------------------------------------------------------
# The empty-response invariant (every adapter, not just this one)
# ---------------------------------------------------------------------------
def test_blank_content_without_an_error_becomes_an_error():
    r = ProviderResult(provider="p", model="m", content="", latency_ms=1)

    assert r.error == EMPTY_RESPONSE_ERROR


def test_whitespace_only_content_counts_as_empty():
    r = ProviderResult(provider="p", model="m", content="  \n\t ", latency_ms=1)

    assert r.error == EMPTY_RESPONSE_ERROR


def test_a_real_answer_is_left_alone():
    r = ProviderResult(provider="p", model="m", content="the answer", latency_ms=1)

    assert r.error is None


def test_an_existing_error_is_never_overwritten():
    """A stream that died after emitting nothing already has the useful reason;
    the generic message must not replace it."""
    r = ProviderResult(
        provider="p", model="m", content="", latency_ms=1, error="APITimeoutError: timed out"
    )

    assert r.error == "APITimeoutError: timed out"


def test_partial_text_from_an_interrupted_stream_keeps_its_error():
    r = ProviderResult(
        provider="p", model="m", content="half an ans", latency_ms=1,
        error="StreamInterrupted: connection reset",
    )

    assert r.content == "half an ans"
    assert r.error == "StreamInterrupted: connection reset"


# ---------------------------------------------------------------------------
# What the fan-out does with it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("blank", ["", "   "])
def test_an_empty_branch_no_longer_outranks_a_real_answer(blank):
    """The bug this guards: finalize answers with the first response that has no
    `error`, so a blank success from agent #1 hid agent #2's real answer."""
    from app.graph.nodes import finalize_node

    empty = ProviderResult(provider="first", model="m", content=blank, latency_ms=1)
    good = ProviderResult(provider="second", model="m", content="real answer", latency_ms=1)
    state = {
        "task": "default",
        "agent_specs": [
            {"name": "first", "provider": "openrouter", "model": "m"},
            {"name": "second", "provider": "openrouter", "model": "m"},
        ],
        "responses": [
            {"provider": "first", "model": "m", "content": empty.content, "error": empty.error},
            {"provider": "second", "model": "m", "content": good.content, "error": good.error},
        ],
    }

    out = finalize_node(state)

    assert out["final_answer"] == "real answer"
    assert any("first" in e for e in out["errors"])
