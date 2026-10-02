"""
F.R.I.D.A.Y. LLM Brain Module
Manages all communication with the NVIDIA Nemotron language model via the
OpenAI-compatible API. Handles conversation history, context windowing,
streaming responses, thread safety, and graceful error recovery.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from friday import config
from friday.async_runner import run_async
from friday.personality import get_system_prompt

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ── Message Types ─────────────────────────────────────────────────────────────


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """
    Immutable representation of a single conversation turn.

    Attributes:
        role:    Who authored the message.
        content: The message text.
        ts:      Unix timestamp of creation.
    """

    role: Role
    content: str
    ts: float = field(default_factory=time.time)

    def to_api_dict(self) -> dict[str, str]:
        """Serialise to the format expected by the OpenAI-compatible API."""
        return {"role": self.role.value, "content": self.content}


# ── Response Types ────────────────────────────────────────────────────────────


class LLMStatus(Enum):
    SUCCESS = auto()
    EMPTY_RESPONSE = auto()
    TIMEOUT = auto()
    CONNECTION_ERROR = auto()
    API_ERROR = auto()
    RATE_LIMITED = auto()
    CONTEXT_ERROR = auto()

    @property
    def is_retryable(self) -> bool:
        """True for transient failures that are worth retrying."""
        return self in (
            LLMStatus.TIMEOUT,
            LLMStatus.CONNECTION_ERROR,
            LLMStatus.RATE_LIMITED,
        )


@dataclass(frozen=True)
class LLMResponse:
    """
    Immutable result returned by every LLM query.

    Attributes:
        text:        The assistant's response text, empty string on failure.
        status:      Outcome classification.
        duration:    Wall-clock time taken for the full response in seconds.
        tokens_used: Approximate token count if reported by the API.
    """

    text: str
    status: LLMStatus
    duration: float = 0.0
    tokens_used: int | None = None

    @property
    def success(self) -> bool:
        return self.status == LLMStatus.SUCCESS

    def __str__(self) -> str:
        return self.text if self.success else f"[{self.status.name}]"


# ── Conversation History ──────────────────────────────────────────────────────


class ConversationHistory:
    """
    Manages the sliding-window message history sent to the LLM.

    The system prompt is always prepended and never counted against the
    history window. The oldest user/assistant pairs are dropped first
    when the window limit is reached. All mutations are thread-safe.
    """

    def __init__(self, max_turns: int = config.storage.max_history) -> None:
        self._max_turns = max_turns
        self._messages: list[Message] = []
        self._system_msg = Message(role=Role.SYSTEM, content=get_system_prompt())
        self._lock = threading.Lock()

    # ── System prompt ─────────────────────────────────────────────────────────

    def set_system_prompt(self, prompt: str) -> None:
        """Replace the system prompt. Clears history to avoid context leaks."""
        with self._lock:
            self._system_msg = Message(role=Role.SYSTEM, content=prompt)
            self._messages.clear()
        logger.debug("[History] System prompt updated; history cleared.")

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        """Append a user message and trim if necessary."""
        with self._lock:
            self._messages.append(Message(role=Role.USER, content=content.strip()))
            self._trim()

    def add_assistant(self, content: str) -> None:
        """Append an assistant message and trim if necessary."""
        with self._lock:
            self._messages.append(Message(role=Role.ASSISTANT, content=content.strip()))
            self._trim()

    def rollback_last_user(self) -> None:
        """
        Remove the most recent USER message (and any trailing SYSTEM injections
        that followed it). Called when a query fails so history stays consistent.
        """
        with self._lock:
            # Walk backwards and drop everything from the last USER turn onward.
            for i in range(len(self._messages) - 1, -1, -1):
                if self._messages[i].role == Role.USER:
                    del self._messages[i:]
                    logger.debug("[History] Rolled back unanswered user turn.")
                    return

    def inject_context(self, context: str) -> None:
        """
        Insert a transient system-level context note before the next turn.
        Useful for injecting screen state, time, or memory recall.
        """
        with self._lock:
            self._messages.append(Message(role=Role.SYSTEM, content=context))

    def clear(self) -> None:
        """Erase all conversation history. System prompt is preserved."""
        with self._lock:
            self._messages.clear()
        logger.debug("[History] Cleared.")

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def build_payload(self) -> list[dict[str, str]]:
        """
        Construct the full message list for an API call.

        Returns a consistent snapshot — safe to use after releasing the lock.
        """
        with self._lock:
            payload = [self._system_msg.to_api_dict()]
            payload.extend(m.to_api_dict() for m in self._messages)
            return payload

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        """Number of messages in history (excludes system prompt)."""
        with self._lock:
            return len(self._messages)

    @property
    def last_user_message(self) -> str | None:
        with self._lock:
            for msg in reversed(self._messages):
                if msg.role == Role.USER:
                    return msg.content
        return None

    @property
    def last_assistant_message(self) -> str | None:
        with self._lock:
            for msg in reversed(self._messages):
                if msg.role == Role.ASSISTANT:
                    return msg.content
        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _trim(self) -> None:
        """
        Drop the oldest messages when history exceeds the window limit.
        Must be called while holding self._lock.
        """
        while len(self._messages) > self._max_turns:
            self._messages.pop(0)


# ── LLM Client ────────────────────────────────────────────────────────────────


class LLMBrain:
    """
    Public interface for all LLM interactions.

    Owns a ConversationHistory instance and exposes both blocking and
    streaming query methods. Handles retry logic, error classification,
    and token tracking. All public methods are thread-safe.
    """

    def __init__(self, shared_client: OpenAI | None = None) -> None:
        # ── NEW: Accept injected client for shared-socket pattern ─────────
        self._client: OpenAI | None = shared_client
        self._owns_client: bool = shared_client is None  # Track ownership for teardown
        self._history: ConversationHistory = ConversationHistory()
        self._ready: bool = shared_client is not None  # Ready immediately if client injected
        self._total_tokens: int = 0
        self._token_lock: threading.Lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise the OpenAI-compatible client pointed at NVIDIA's endpoint.

        Returns:
            True if the client was created successfully.
        """
        # ── NEW: Skip if client was injected ───────────────────────────────
        if self._client is not None:
            if not self._ready:
                self._ready = True
                logger.info(
                    "[LLM] Using shared client. Model: %s | Endpoint: %s",
                    config.llm.model,
                    config.llm.base_url,
                )
            return True

        try:
            self._client = OpenAI(
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
                timeout=config.llm.timeout,
                max_retries=1,
            )
            self._owns_client = True
            self._ready = True
            logger.info(
                "[LLM] Client ready. Model: %s | Endpoint: %s",
                config.llm.model,
                config.llm.base_url,
            )
            return True
        except Exception as exc:
            logger.error("[LLM] Client initialisation failed: %s", exc)
            return False

    def teardown(self) -> None:
        """Release resources and reset state."""
        self._history.clear()
        # ── NEW: Only close client if we own it ───────────────────────────
        if self._owns_client:
            self._client = None
        self._ready = False

    # ── Core Query ────────────────────────────────────────────────────────────

    def query(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        retries: int = 1,
        add_to_history: bool = True,
        ensemble_type: str | None = None,
    ) -> LLMResponse:
        """
        Send a prompt to the LLM and return the complete response.

        Args:
            prompt:         The user's input text.
            temperature:    Override the config temperature (None = use config).
            max_tokens:     Override the config max_tokens (None = use config).
            retries:        Number of retry attempts on transient failures.
            add_to_history: Whether to persist this exchange in history.

        Returns:
            LLMResponse with the outcome and response text.
        """
        if not self._ready or self._client is None:
            logger.error("[LLM] query() called before setup().")
            return LLMResponse(text="", status=LLMStatus.CONTEXT_ERROR)

        if not prompt.strip():
            return LLMResponse(text="", status=LLMStatus.EMPTY_RESPONSE)

        if model is None:
            if len(prompt.split()) < 15:
                model = "qwen2.5:1.5b-instruct-q4_K_M"
            else:
                model = "qwen2.5:3b-instruct-q4_K_M"

        if add_to_history:
            self._history.add_user(prompt)
            payload = self._history.build_payload()
        else:
            payload = self._history.build_payload()
            payload.append({"role": "user", "content": prompt.strip()})
        if ensemble_type is None and len(prompt.split()) < 20:
            if payload and payload[0]["role"] == "system":
                payload[0] = {
                    "role": "system",
                    "content": payload[0]["content"] + "\nRespond in 1-2 sentences maximum. Be direct."
                }
        start = time.perf_counter()

        if ensemble_type:
            from friday.ensemble import EnsembleRunner

            try:
                result_text = run_async(
                    EnsembleRunner.run_ensemble(
                        ensemble_type=ensemble_type,
                        messages=payload,
                        original_prompt=prompt,
                        temperature=temperature
                        if temperature is not None
                        else config.llm.temperature,
                        max_tokens=max_tokens if max_tokens is not None else config.llm.max_tokens,
                    )
                )
                if add_to_history:
                    self._history.add_assistant(result_text)
                duration = time.perf_counter() - start
                return LLMResponse(
                    text=result_text,
                    status=LLMStatus.SUCCESS,
                    duration=duration,
                )
            except Exception as e:
                logger.error(f"[LLM] Ensemble query failed: {e}")
                if add_to_history:
                    self._history.rollback_last_user()
                return LLMResponse(text="", status=LLMStatus.API_ERROR)

        last_response = LLMResponse(text="", status=LLMStatus.CONNECTION_ERROR)

        for attempt in range(1, retries + 2):  # +2: initial try + N retries
            last_response = self._call_api(
                payload,
                temperature=temperature if temperature is not None else config.llm.temperature,
                max_tokens=max_tokens if max_tokens is not None else config.llm.max_tokens,
                model=model if model is not None else config.llm.model,
            )

            if last_response.success:
                if add_to_history:
                    self._history.add_assistant(last_response.text)
                self._add_tokens(last_response.tokens_used)
                duration = time.perf_counter() - start
                logger.info(
                    "[LLM] Response in %.2fs (%d tokens): '%s…'",
                    duration,
                    last_response.tokens_used or 0,
                    last_response.text[:60],
                )
                return LLMResponse(
                    text=last_response.text,
                    status=LLMStatus.SUCCESS,
                    duration=duration,
                    tokens_used=last_response.tokens_used,
                )

            if not last_response.status.is_retryable:
                # Non-retryable: API error, empty response, context error
                break

            if attempt <= retries:
                wait = 1.5**attempt
                logger.warning(
                    "[LLM] Transient failure (%s) — retrying in %.1fs (%d/%d).",
                    last_response.status.name,
                    wait,
                    attempt,
                    retries,
                )
                time.sleep(wait)

        # All attempts failed — roll back the user message we added
        if add_to_history:
            self._history.rollback_last_user()

        logger.error("[LLM] All attempts failed. Final status: %s", last_response.status.name)
        return last_response

    def vision_query(
        self,
        prompt: str,
        base64_image: str,
        temperature: float = 0.5,
        max_tokens: int = 500,
        model: str | None = None,
    ) -> LLMResponse:
        """
        Send a vision query (text + image) to the LLM.
        Note: This does NOT add to history to avoid bloating context with images.
        """
        if not self._client:
            return LLMResponse(text="", status=LLMStatus.API_ERROR)

        messages = [
            {"role": "system", "content": "You are a specialized visual analysis agent."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]

        return self._call_api(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model or "qwen2.5:3b-instruct-q4_K_M",
        )

    def stream(
        self,
        prompt: str,
        *,
        on_chunk: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        retries: int = 1,
        add_to_history: bool = True,
        interrupt_event: threading.Event | None = None,
        ensemble_type: str | None = None,
    ) -> LLMResponse:
        """
        Stream the LLM response, firing a callback for each token chunk.

        Unlike the old generator design, this returns a standard LLMResponse
        so callers get a reliable result regardless of error handling flow.
        Token chunks are delivered via the ``on_chunk`` callback as they arrive.

        Args:
            prompt:         The user's input text.
            on_chunk:       Called with each text chunk as it streams in.
                            Defaults to a no-op if not provided.
            temperature:    Override config temperature (None = use config).
            max_tokens:     Override config max_tokens (None = use config).
            retries:        Number of retry attempts on stream failures.
            add_to_history: Persist exchange in history on success.

        Returns:
            Complete LLMResponse after the stream is exhausted.
        """
        if not self._ready or self._client is None:
            logger.error("[LLM] stream() called before setup().")
            return LLMResponse(text="", status=LLMStatus.CONTEXT_ERROR)

        if not prompt.strip():
            return LLMResponse(text="", status=LLMStatus.EMPTY_RESPONSE)

        if model is None:
            if len(prompt.split()) < 15:
                model = "qwen2.5:1.5b-instruct-q4_K_M"
            else:
                model = "qwen2.5:3b-instruct-q4_K_M"

        _on_chunk = on_chunk or (lambda _: None)

        if add_to_history:
            self._history.add_user(prompt)
            payload = self._history.build_payload()
        else:
            payload = self._history.build_payload()
            payload.append({"role": "user", "content": prompt.strip()})
        if ensemble_type is None and len(prompt.split()) < 20:
            if payload and payload[0]["role"] == "system":
                payload[0] = {
                    "role": "system",
                    "content": payload[0]["content"] + "\nRespond in 1-2 sentences maximum. Be direct."
                }
        start = time.perf_counter()

        if ensemble_type:
            from friday.ensemble import EnsembleRunner

            try:
                result_text = run_async(
                    EnsembleRunner.run_ensemble(
                        ensemble_type=ensemble_type,
                        messages=payload,
                        original_prompt=prompt,
                        temperature=temperature
                        if temperature is not None
                        else config.llm.temperature,
                        max_tokens=max_tokens if max_tokens is not None else config.llm.max_tokens,
                    )
                )

                # Stream the result back in chunks
                chunk_size = 20
                for i in range(0, len(result_text), chunk_size):
                    if interrupt_event and interrupt_event.is_set():
                        break
                    chunk = result_text[i : i + chunk_size]
                    _on_chunk(chunk)
                    time.sleep(0.01)

                if add_to_history:
                    self._history.add_assistant(result_text)
                duration = time.perf_counter() - start
                return LLMResponse(
                    text=result_text,
                    status=LLMStatus.SUCCESS,
                    duration=duration,
                )
            except Exception as e:
                logger.error(f"[LLM] Ensemble stream failed: {e}")
                if add_to_history:
                    self._history.rollback_last_user()
                return LLMResponse(text="", status=LLMStatus.API_ERROR)

        for attempt in range(1, retries + 2):
            result = self._stream_once(
                temperature=temperature if temperature is not None else config.llm.temperature,
                max_tokens=max_tokens if max_tokens is not None else config.llm.max_tokens,
                model=model if model is not None else config.llm.model,
                on_chunk=_on_chunk,
                interrupt_event=interrupt_event,
                payload=payload,
            )

            if result.success:
                if add_to_history:
                    self._history.add_assistant(result.text)
                logger.info(
                    "[LLM/Stream] %.2fs | '%s…'",
                    result.duration,
                    result.text[:60],
                )
                return result

            if not result.status.is_retryable:
                break

            if attempt <= retries:
                wait = 1.5**attempt
                logger.warning(
                    "[LLM/Stream] Transient failure (%s) — retrying in %.1fs (%d/%d).",
                    result.status.name,
                    wait,
                    attempt,
                    retries,
                )
                time.sleep(wait)

        if add_to_history:
            self._history.rollback_last_user()

        logger.error("[LLM/Stream] All attempts failed. Status: %s", result.status.name)
        return result

    # ── Context Management ────────────────────────────────────────────────────

    def inject_context(self, context: str) -> None:
        """
        Insert a system-level context note before the next LLM call.
        Use for screen state, memory recall, or environmental data.
        """
        self._history.inject_context(context)

    def clear_history(self) -> None:
        """Reset conversation history while keeping the client active."""
        self._history.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """
        Override the system prompt for specialised modes.
        Clears conversation history to avoid context contamination.

        Args:
            prompt: New system prompt string.
        """
        self._history.set_system_prompt(prompt)

    # ── Swarm Intelligence ────────────────────────────────────────────────────

    def classify_intent(self, prompt: str) -> dict[str, Any]:
        """
        Analyze the user's prompt to determine if it's a casual chat or a deep task
        requiring background delegation.

        Returns:
            Dict containing 'type', 'specialist', and 'acknowledgment'.
        """
        system_msg = (
            "You are the F.R.I.D.A.Y. Intent Classifier. Analyze the user's request "
            "and determine if it requires background processing (deep_task) or "
            "can be handled as a real-time conversation (casual_chat).\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "type": "casual_chat" | "deep_task",\n'
            '  "specialist": "CODER" | "SECURITY" | "RESEARCH" | "BROWSER" | null,\n'
            '  "acknowledgment": "A short verbal confirmation for the user"\n'
            "}"
        )

        # Use a non-history-polluting query for classification
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]

        try:
            # Use the active NIM nano model for fast local routing decisions.
            routing_model = "qwen2.5:1.5b-instruct-q4_K_M"
            resp = self._call_api(messages, temperature=0.1, max_tokens=200, model=routing_model)
            if resp.success:
                # Basic cleaning in case the LLM includes markdown backticks
                cleaned = resp.text.strip().replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
        except Exception as e:
            logger.error(f"[LLM] Classification failed: {e}")

        return {"type": "casual_chat", "specialist": None, "acknowledgment": "I'm on it, Sir."}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def history(self) -> ConversationHistory:
        return self._history

    @property
    def total_tokens_used(self) -> int:
        with self._token_lock:
            return self._total_tokens

    @property
    def turn_count(self) -> int:
        return self._history.turn_count

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: str = "qwen2.5:3b-instruct-q4_K_M",
    ) -> LLMResponse:
        """
        Execute a single non-streaming API call and classify the response.

        Args:
            messages:    Full message payload including system prompt.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.

        Returns:
            LLMResponse with classified status.
        """
        try:
            completion = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            text = (completion.choices[0].message.content or "").strip()

            if not text:
                logger.warning("[LLM] API returned empty content.")
                return LLMResponse(text="", status=LLMStatus.EMPTY_RESPONSE)

            tokens = completion.usage.total_tokens if completion.usage else None

            return LLMResponse(text=text, status=LLMStatus.SUCCESS, tokens_used=tokens)

        except APITimeoutError:
            logger.warning("[LLM] Request timed out after %ds.", config.llm.timeout)
            return LLMResponse(text="", status=LLMStatus.TIMEOUT)

        except APIConnectionError as exc:
            logger.warning("[LLM] Connection error: %s", exc)
            return LLMResponse(text="", status=LLMStatus.CONNECTION_ERROR)

        except APIStatusError as exc:
            if exc.status_code == 429:
                logger.warning("[LLM] Rate limited (429).")
                return LLMResponse(text="", status=LLMStatus.RATE_LIMITED)
            logger.error("[LLM] API error %d: %s", exc.status_code, exc.message)
            return LLMResponse(text="", status=LLMStatus.API_ERROR)

        except Exception as exc:
            logger.exception("[LLM] Unexpected error: %s", exc)
            return LLMResponse(text="", status=LLMStatus.API_ERROR)

    def _stream_once(
        self,
        temperature: float,
        max_tokens: int,
        on_chunk: Callable[[str], None],
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        interrupt_event: threading.Event | None = None,
        payload: list[dict[str, str]] | None = None,
    ) -> LLMResponse:
        """
        Execute a single streaming API call.

        Fires on_chunk for every delta token received. Builds the full
        response text internally and returns it in an LLMResponse.

        Args:
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.
            on_chunk:    Callback invoked per text chunk.

        Returns:
            LLMResponse with the full accumulated text and status.
        """
        if payload is None:
            payload = self._history.build_payload()
        full_text = ""
        start = time.perf_counter()

        try:
            stream_iter = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},  # Request token counts
            )

            total_tokens: int | None = None

            for chunk in stream_iter:
                if interrupt_event and interrupt_event.is_set():
                    logger.info("[LLM/Stream] Generation interrupted by user.")
                    break

                # Final chunk carries usage when stream_options include_usage is set
                if chunk.usage:
                    total_tokens = chunk.usage.total_tokens

                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_text += delta
                    on_chunk(delta)

        except APITimeoutError as exc:
            logger.warning("[LLM/Stream] Request timed out: %s", exc)
            return LLMResponse(
                text=full_text,
                status=LLMStatus.TIMEOUT,
                duration=time.perf_counter() - start,
            )

        except APIConnectionError as exc:
            logger.warning("[LLM/Stream] Connection error: %s", exc)
            return LLMResponse(
                text=full_text,
                status=LLMStatus.CONNECTION_ERROR,
                duration=time.perf_counter() - start,
            )

        except APIStatusError as exc:
            if exc.status_code == 429:
                logger.warning("[LLM/Stream] Rate limited (429).")
                return LLMResponse(
                    text=full_text,
                    status=LLMStatus.RATE_LIMITED,
                    duration=time.perf_counter() - start,
                )
            logger.error("[LLM/Stream] API error %d: %s", exc.status_code, exc.message)
            return LLMResponse(
                text=full_text,
                status=LLMStatus.API_ERROR,
                duration=time.perf_counter() - start,
            )

        except Exception as exc:
            logger.exception("[LLM/Stream] Unexpected error: %s", exc)
            return LLMResponse(
                text=full_text,
                status=LLMStatus.API_ERROR,
                duration=time.perf_counter() - start,
            )

        duration = time.perf_counter() - start

        if not full_text.strip():
            return LLMResponse(text="", status=LLMStatus.EMPTY_RESPONSE, duration=duration)

        if total_tokens is not None:
            self._add_tokens(total_tokens)

        return LLMResponse(
            text=full_text,
            status=LLMStatus.SUCCESS,
            duration=duration,
            tokens_used=total_tokens,
        )

    def _add_tokens(self, count: int | None) -> None:
        """Thread-safe token counter increment."""
        if count:
            with self._token_lock:
                self._total_tokens += count
