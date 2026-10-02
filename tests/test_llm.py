"""
Tests for friday.llm — the LLM client wrapper.

Covers the four explicit scenarios requested:
  1. query() with ensemble_type uses run_async (not asyncio.run)
  2. stream() interrupt_event stops generation
  3. rollback_last_user called on failed query
  4. vision_query returns LLMResponse

Plus auxiliary coverage (query happy/error paths, history bookkeeping,
setup/teardown lifecycle, LLMResponse / LLMStatus semantics,
ConversationHistory) to comfortably exceed 60% on the file.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APITimeoutError

from friday.llm import (
    ConversationHistory,
    LLMBrain,
    LLMResponse,
    LLMStatus,
    Message,
    Role,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_chunk(content):
    """Build a streaming chunk that yields `content` as a delta."""
    chunk = MagicMock()
    chunk.usage = None
    choice = MagicMock()
    choice.delta.content = content
    chunk.choices = [choice]
    return chunk


def make_usage_chunk(total_tokens):
    """Final streaming chunk carrying usage info (no choices, no content)."""
    chunk = MagicMock()
    chunk.usage = MagicMock()
    chunk.usage.total_tokens = total_tokens
    chunk.choices = []
    return chunk


def make_completion(text: str, total_tokens: int | None = 100):
    """Build a non-streaming completion response mock."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = text
    if total_tokens is None:
        completion.usage = None
    else:
        completion.usage = MagicMock()
        completion.usage.total_tokens = total_tokens
    return completion


def make_brain(client=None) -> LLMBrain:
    """Build an LLMBrain ready to call query/stream without invoking setup()."""
    return LLMBrain(shared_client=client or MagicMock())


# ── 1. query() with ensemble_type uses run_async (not asyncio.run) ───────────


@patch("friday.llm.run_async")
def test_query_with_ensemble_uses_run_async(mock_run_async):
    """
    When ensemble_type is set, query() must dispatch through run_async()
    and not through asyncio.run(). The observable signature: run_async
    is called once with a coroutine, and the returned text comes from
    the awaited ensemble result.
    """

    def _resolve(coro):
        # Close the coroutine the production code passed us so we don't
        # leak a "never awaited" warning; return a fixed value as if
        # the inner coroutine resolved to it.
        if hasattr(coro, "close"):
            coro.close()
        return "ensemble result"

    mock_run_async.side_effect = _resolve

    brain = make_brain()

    result = brain.query("hello", ensemble_type="coding")

    # run_async was the dispatcher
    assert mock_run_async.call_count == 1
    # The result text came from the (awaited) coroutine
    assert isinstance(result, LLMResponse)
    assert result.text == "ensemble result"
    assert result.status is LLMStatus.SUCCESS
    # History was updated: the user message AND assistant response
    assert brain.history.turn_count == 2
    assert brain.history.last_user_message == "hello"
    assert brain.history.last_assistant_message == "ensemble result"


@patch("friday.llm.run_async")
def test_query_with_ensemble_rollback_on_exception(mock_run_async):
    """
    If the ensemble call raises, the user message added before the call
    must be rolled back so history doesn't accumulate ghost turns.
    """

    def _boom(coro):
        if hasattr(coro, "close"):
            coro.close()
        raise RuntimeError("ensemble blew up")

    mock_run_async.side_effect = _boom

    brain = make_brain()

    result = brain.query("hello", ensemble_type="coding")

    assert result.status is LLMStatus.API_ERROR
    assert result.text == ""
    # rollback_last_user removed the user turn -> history is empty
    assert brain.history.turn_count == 0


# ── 2. stream() interrupt_event stops generation ─────────────────────────────


def test_stream_interrupt_set_before_stream_stops_immediately():
    """
    If interrupt_event is already set when stream() runs, the streaming
    loop must break out at the first chunk check and on_chunk must never
    fire.
    """
    interrupt = threading.Event()
    interrupt.set()

    client = MagicMock()
    client.chat.completions.create.return_value = iter(
        [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    )

    brain = make_brain(client)
    chunks: list[str] = []
    result = brain.stream("hi", on_chunk=chunks.append, interrupt_event=interrupt)

    assert chunks == []
    assert result.text == ""
    # No content was streamed, so it's an empty-response status.
    assert result.status is LLMStatus.EMPTY_RESPONSE


def test_stream_interrupt_mid_stream_returns_partial_text():
    """
    If the interrupt event is set during on_chunk, the loop must break
    on the next iteration and return the partial text accumulated so far.
    """
    interrupt = threading.Event()

    client = MagicMock()
    client.chat.completions.create.return_value = iter(
        [make_chunk("alpha"), make_chunk("beta"), make_chunk("gamma")]
    )

    def on_chunk(text: str) -> None:
        if text == "alpha":
            interrupt.set()

    brain = make_brain(client)
    result = brain.stream("hi", on_chunk=on_chunk, interrupt_event=interrupt)

    # First chunk delivered, interrupt observed, loop exited
    assert result.text == "alpha"
    assert result.status is LLMStatus.SUCCESS
    # The interrupted (empty) response was NOT rolled back — we got a
    # real successful stream that just happened to end early.
    assert brain.history.last_user_message == "hi"
    assert brain.history.last_assistant_message == "alpha"


def test_stream_without_interrupt_runs_to_completion():
    """Sanity: no interrupt event -> all chunks accumulate, SUCCESS returned."""
    client = MagicMock()
    client.chat.completions.create.return_value = iter(
        [make_chunk("hello"), make_chunk(" "), make_chunk("world")]
    )

    brain = make_brain(client)
    chunks: list[str] = []
    result = brain.stream("hi", on_chunk=chunks.append)

    assert result.text == "hello world"
    assert result.status is LLMStatus.SUCCESS
    assert chunks == ["hello", " ", "world"]


# ── 3. rollback_last_user called on failed query ─────────────────────────────


def test_query_rollback_on_non_retryable_api_error():
    """
    A non-retryable API error on every attempt must leave history empty
    (the user turn added at the start of query() is rolled back).
    """
    client = MagicMock()
    # Non-retryable: APIStatusError that isn't 429
    error = MagicMock()
    error.status_code = 500
    error.message = "boom"
    # Make the create call raise APIStatusError via a side_effect.
    from openai import APIStatusError as RealAPIStatusError

    def _raise(*_args, **_kwargs):
        raise RealAPIStatusError(
            "boom", response=MagicMock(), body={"error": "boom"}
        )

    client.chat.completions.create.side_effect = _raise

    brain = make_brain(client)
    result = brain.query("hi", retries=1)

    assert result.status is LLMStatus.API_ERROR
    # User turn was rolled back -> history is empty (system prompt excluded)
    assert brain.history.turn_count == 0


def test_query_rollback_on_retryable_then_non_retryable():
    """
    Even after a transient retry, the final non-retryable failure still
    triggers rollback.
    """
    call_count = {"n": 0}

    def _maybe_timeout(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise APITimeoutError("transient")
        # Second attempt: also fails non-retryably
        from openai import APIStatusError as RealAPIStatusError

        raise RealAPIStatusError(
            "fatal", response=MagicMock(), body={"error": "fatal"}
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = _maybe_timeout

    brain = make_brain(client)
    result = brain.query("hi", retries=2)

    assert call_count["n"] == 2
    assert result.status is LLMStatus.API_ERROR
    assert brain.history.turn_count == 0


def test_query_no_rollback_on_successful_response():
    """
    On a successful response, the user AND assistant messages are both
    persisted in history (no rollback).
    """
    client = MagicMock()
    client.chat.completions.create.return_value = make_completion("hi back", 42)

    brain = make_brain(client)
    result = brain.query("hello", retries=0)

    assert result.status is LLMStatus.SUCCESS
    assert result.tokens_used == 42
    assert brain.history.turn_count == 2
    assert brain.history.last_user_message == "hello"
    assert brain.history.last_assistant_message == "hi back"
    assert brain.total_tokens_used == 42


def test_query_rollback_via_ensemble_path():
    """
    Ensemble failure also triggers rollback_last_user.
    """

    def _raise_conn(coro):
        if hasattr(coro, "close"):
            coro.close()
        raise ConnectionError("net down")

    with patch("friday.llm.run_async", side_effect=_raise_conn):
        brain = make_brain()
        result = brain.query("hi", ensemble_type="reasoning")

    assert result.status is LLMStatus.API_ERROR
    assert brain.history.turn_count == 0


# ── 4. vision_query returns LLMResponse ──────────────────────────────────────


def test_vision_query_returns_llm_response():
    """
    vision_query must return an LLMResponse carrying the model's text and
    SUCCESS status, and the payload sent to the client must contain the
    base64 image as a data URL.
    """
    client = MagicMock()
    client.chat.completions.create.return_value = make_completion(
        "I see a cat.", total_tokens=200
    )

    brain = make_brain(client)
    result = brain.vision_query("What is in this image?", base64_image="AAAABBBB")

    assert isinstance(result, LLMResponse)
    assert result.text == "I see a cat."
    assert result.status is LLMStatus.SUCCESS
    assert result.tokens_used == 200

    # Inspect the payload sent to the client
    client.chat.completions.create.assert_called_once()
    kwargs = client.chat.completions.create.call_args.kwargs
    messages = kwargs["messages"]
    assert len(messages) == 2  # system + user

    user_msg = messages[1]
    assert user_msg["role"] == "user"
    parts = user_msg["content"]
    assert isinstance(parts, list)
    # The first part is the text prompt, the second is the image
    assert parts[0]["type"] == "text"
    assert parts[0]["text"] == "What is in this image?"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1]["image_url"]["url"].endswith("AAAABBBB")


def test_vision_query_without_client_returns_api_error():
    """
    Without an injected client, vision_query must return a non-success
    LLMResponse rather than raising.
    """
    brain = LLMBrain()  # No shared client -> _client is None
    result = brain.vision_query("anything", base64_image="xx")

    assert result.status is LLMStatus.API_ERROR
    assert result.text == ""


# ── Auxiliary: query() lifecycle & error paths ───────────────────────────────


def test_query_before_setup_returns_context_error():
    """query() with no client (not ready) returns CONTEXT_ERROR."""
    brain = LLMBrain()  # No client injected
    result = brain.query("anything")

    assert result.status is LLMStatus.CONTEXT_ERROR
    assert result.text == ""


def test_query_empty_prompt_returns_empty_response():
    """An empty / whitespace prompt returns EMPTY_RESPONSE without an API call."""
    client = MagicMock()
    brain = make_brain(client)

    result = brain.query("   ")

    assert result.status is LLMStatus.EMPTY_RESPONSE
    assert result.text == ""
    client.chat.completions.create.assert_not_called()
    # Empty prompt was never added to history either.
    assert brain.history.turn_count == 0


def test_query_timeout_is_retryable_then_fails():
    """Timeouts are retryable — verify retries happen then the final timeout wins."""
    client = MagicMock()
    client.chat.completions.create.side_effect = APITimeoutError("slow")

    brain = make_brain(client)
    result = brain.query("hi", retries=1)

    # initial + 1 retry = 2 calls
    assert client.chat.completions.create.call_count == 2
    assert result.status is LLMStatus.TIMEOUT
    assert brain.history.turn_count == 0  # rolled back


def test_query_connection_error_classified_correctly():
    """APIConnectionError surfaces as LLMStatus.CONNECTION_ERROR."""
    client = MagicMock()
    client.chat.completions.create.side_effect = APIConnectionError(request=None)

    brain = make_brain(client)
    result = brain.query("hi", retries=0)

    assert result.status is LLMStatus.CONNECTION_ERROR


def test_query_rate_limited_429_is_retryable():
    """HTTP 429 is a transient failure and is retried."""
    # Simulate 429 via a real APIStatusError
    from openai import APIStatusError as RealAPIStatusError

    call_count = {"n": 0}

    def _maybe_429(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RealAPIStatusError(
                "rate limited",
                response=MagicMock(status_code=429),
                body={"error": "rate_limited"},
            )
        return make_completion("ok", 10)

    client = MagicMock()
    client.chat.completions.create.side_effect = _maybe_429

    brain = make_brain(client)
    result = brain.query("hi", retries=1)

    assert call_count["n"] == 2
    assert result.status is LLMStatus.SUCCESS
    # Successful retry -> both turns persisted
    assert brain.history.turn_count == 2


def test_query_add_to_history_false_does_not_persist():
    """add_to_history=False: query() must not touch history at all."""
    client = MagicMock()
    client.chat.completions.create.return_value = make_completion("ok")

    brain = make_brain(client)
    result = brain.query("hi", add_to_history=False)

    assert result.status is LLMStatus.SUCCESS
    assert brain.history.turn_count == 0


def test_query_unexpected_exception_classified_as_api_error():
    """A bare Exception from the client becomes LLMStatus.API_ERROR."""
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("weird")

    brain = make_brain(client)
    result = brain.query("hi", retries=0)

    assert result.status is LLMStatus.API_ERROR
    assert brain.history.turn_count == 0


# ── Auxiliary: setup() / teardown() ───────────────────────────────────────────


def test_setup_with_injected_client_is_immediately_ready():
    """A brain constructed with a shared client is ready and skips setup()."""
    client = MagicMock()
    brain = LLMBrain(shared_client=client)

    assert brain.is_ready is True
    # Calling setup() again is a no-op success.
    assert brain.setup() is True
    # The injected client was not overwritten.
    assert brain._client is client


def test_setup_creates_client_when_none_injected():
    """
    When no client is injected, setup() must construct an OpenAI client
    using the configured credentials.
    """
    brain = LLMBrain()
    assert brain.is_ready is False

    with patch("friday.llm.OpenAI") as mock_openai:
        ok = brain.setup()

    assert ok is True
    assert brain.is_ready is True
    mock_openai.assert_called_once()


def test_setup_returns_false_when_client_construction_fails():
    """If the OpenAI client raises during construction, setup() returns False."""
    brain = LLMBrain()

    with patch("friday.llm.OpenAI", side_effect=RuntimeError("bad config")):
        ok = brain.setup()

    assert ok is False
    assert brain.is_ready is False


def test_teardown_clears_history_and_disables_brain():
    """teardown() must clear history and flip is_ready off."""
    client = MagicMock()
    client.chat.completions.create.return_value = make_completion("ok")
    brain = make_brain(client)
    brain.query("hi")
    assert brain.history.turn_count == 2

    brain.teardown()

    assert brain.history.turn_count == 0
    assert brain.is_ready is False


def test_teardown_does_not_close_injected_client():
    """
    When a client was injected, teardown() must NOT close it (the caller
    owns it).
    """
    client = MagicMock()
    brain = LLMBrain(shared_client=client)
    brain.teardown()
    # Client reference is preserved (we don't own it).
    assert brain._client is client


# ── Auxiliary: LLMResponse / LLMStatus / Message / ConversationHistory ──────


def test_llm_response_success_property_and_str():
    ok = LLMResponse(text="hello", status=LLMStatus.SUCCESS)
    assert ok.success is True
    assert str(ok) == "hello"

    fail = LLMResponse(text="", status=LLMStatus.TIMEOUT)
    assert fail.success is False
    assert "TIMEOUT" in str(fail)


def test_llm_status_retryable_classification():
    assert LLMStatus.TIMEOUT.is_retryable is True
    assert LLMStatus.CONNECTION_ERROR.is_retryable is True
    assert LLMStatus.RATE_LIMITED.is_retryable is True
    assert LLMStatus.API_ERROR.is_retryable is False
    assert LLMStatus.EMPTY_RESPONSE.is_retryable is False
    assert LLMStatus.CONTEXT_ERROR.is_retryable is False
    assert LLMStatus.SUCCESS.is_retryable is False


def test_message_to_api_dict():
    m = Message(role=Role.USER, content="hi")
    assert m.to_api_dict() == {"role": "user", "content": "hi"}


def test_conversation_history_rollback_skips_trailing_system():
    """
    rollback_last_user must remove the last user message AND any
    messages that came after it (including a system-injected context
    snapshot that followed the last user turn).
    """
    h = ConversationHistory(max_turns=10)
    h.add_user("first question")
    h.add_assistant("first answer")
    h.add_user("second question")
    h.inject_context("snapshot after second question")

    assert h.turn_count == 4
    h.rollback_last_user()
    # The second user + the system injection that trailed it are gone.
    # "first answer" (assistant) stays.
    assert h.last_user_message == "first question"
    assert h.last_assistant_message == "first answer"
    assert h.turn_count == 2


def test_conversation_history_set_system_prompt_clears_history():
    """set_system_prompt() must clear the conversation history."""
    h = ConversationHistory(max_turns=10)
    h.add_user("hello")
    h.add_assistant("hi back")
    assert h.turn_count == 2

    h.set_system_prompt("a totally different persona")
    assert h.turn_count == 0


def test_conversation_history_trims_to_max_turns():
    h = ConversationHistory(max_turns=2)
    h.add_user("a")
    h.add_assistant("b")
    h.add_user("c")
    h.add_assistant("d")
    # Only the last 2 turns remain
    assert h.turn_count == 2
    assert h.last_user_message == "c"


def test_conversation_history_build_payload_prepends_system_prompt():
    h = ConversationHistory(max_turns=10)
    h.add_user("hi")
    payload = h.build_payload()
    assert payload[0]["role"] == "system"
    assert payload[1:] == [{"role": "user", "content": "hi"}]


def test_conversation_history_clear_preserves_system_prompt():
    h = ConversationHistory(max_turns=10)
    h.add_user("hi")
    h.clear()
    assert h.turn_count == 0
    # System prompt still appears in subsequent payload
    assert h.build_payload()[0]["role"] == "system"


def test_conversation_history_last_user_and_assistant():
    h = ConversationHistory(max_turns=10)
    h.add_user("u1")
    h.add_assistant("a1")
    h.add_user("u2")
    assert h.last_user_message == "u2"
    assert h.last_assistant_message == "a1"


def test_conversation_history_inject_context_appears_in_payload():
    h = ConversationHistory(max_turns=10)
    h.add_user("u")
    h.inject_context("ctx")
    payload = h.build_payload()
    # ctx sits between system and user
    assert [m["role"] for m in payload] == ["system", "user", "system"]


# ── Auxiliary: stream() retry / rollback edge cases ───────────────────────────


def test_stream_empty_prompt_returns_empty_response():
    client = MagicMock()
    brain = make_brain(client)
    result = brain.stream("   ")
    assert result.status is LLMStatus.EMPTY_RESPONSE
    client.chat.completions.create.assert_not_called()


def test_stream_before_setup_returns_context_error():
    brain = LLMBrain()  # no client
    result = brain.stream("hi")
    assert result.status is LLMStatus.CONTEXT_ERROR


def test_stream_timeout_triggers_rollback():
    client = MagicMock()
    client.chat.completions.create.side_effect = APITimeoutError("slow")

    brain = make_brain(client)
    result = brain.stream("hi", retries=0)

    assert result.status is LLMStatus.TIMEOUT
    assert brain.history.turn_count == 0


def test_stream_connection_error_rolls_back():
    client = MagicMock()
    client.chat.completions.create.side_effect = APIConnectionError(request=None)

    brain = make_brain(client)
    result = brain.stream("hi", retries=0)

    assert result.status is LLMStatus.CONNECTION_ERROR
    assert brain.history.turn_count == 0


def test_stream_records_token_usage_on_success():
    client = MagicMock()
    client.chat.completions.create.return_value = iter(
        [make_chunk("hi"), make_usage_chunk(77)]
    )

    brain = make_brain(client)
    result = brain.stream("hello")

    assert result.status is LLMStatus.SUCCESS
    assert result.tokens_used == 77
    assert brain.total_tokens_used == 77
