"""
Tests for friday.async_runner — the safe-from-anywhere coroutine runner.

Covers the four explicit scenarios requested:
  1. run_async from a thread with no running loop
  2. run_async from a thread with a running loop
  3. run_async timeout raises TimeoutError
  4. run_async_fire_and_forget submits without blocking

Plus auxiliary coverage (executor initialisation, no-timeout path, error
propagation, _ensure_executor caching) so we comfortably exceed 85%.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from friday import async_runner
from friday.async_runner import (
    _ensure_executor,
    run_async,
    run_async_fire_and_forget,
)


# ── 1. run_async from a thread with no running loop ───────────────────────────


def test_run_async_from_sync_context_executes_coroutine():
    """
    Calling run_async() from a plain sync thread (no event loop running) must
    successfully run the coroutine via asyncio.run() and return the result.
    """

    async def _add(a: int, b: int) -> int:
        return a + b

    # No event loop in this thread — safe to use asyncio.run() under the hood.
    result = run_async(_add(2, 3))

    assert result == 5


def test_run_async_from_sync_context_propagates_exceptions():
    """
    Exceptions raised inside the coroutine must surface to the caller in
    the sync context.
    """

    async def _boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_async(_boom())


def test_run_async_from_sync_context_with_args_returns_value():
    """A coroutine that awaits asyncio.sleep still resolves correctly."""

    async def _slow_return(value: str) -> str:
        await asyncio.sleep(0.01)
        return value

    assert run_async(_slow_return("done")) == "done"


# ── 2. run_async from a thread with a running loop ────────────────────────────


@pytest.mark.asyncio
async def test_run_async_from_async_context_uses_executor():
    """
    Calling run_async() from within an async function (a loop is running)
    must submit the coroutine to the executor and return the result.
    """

    async def _double(n: int) -> int:
        return n * 2

    # Inside the pytest-asyncio-managed loop, run_async must NOT call
    # asyncio.run() — that would raise "asyncio.run() cannot be called from
    # a running event loop". The executor path is the only valid one.
    result = run_async(_double(21))

    assert result == 42


@pytest.mark.asyncio
async def test_run_async_from_async_context_propagates_exceptions():
    """Exceptions from the inner coroutine must surface to the outer async caller."""

    async def _explode() -> None:
        raise RuntimeError("inner failure")

    with pytest.raises(RuntimeError, match="inner failure"):
        run_async(_explode())


# ── 3. run_async timeout raises TimeoutError ──────────────────────────────────


def test_run_async_sync_timeout_raises_timeout_error():
    """
    A coroutine that exceeds the timeout must cause run_async() to raise
    TimeoutError, not asyncio.TimeoutError.
    """

    async def _slow() -> str:
        await asyncio.sleep(1.0)
        return "never"

    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_async(_slow(), timeout=0.05)
    elapsed = time.perf_counter() - started

    # Sanity: we should have given up well before the 1.0s sleep would have
    # resolved. Allow generous slack for slow CI machines.
    assert elapsed < 0.8, f"timeout took {elapsed:.3f}s — should have aborted early"


@pytest.mark.asyncio
async def test_run_async_async_timeout_raises_timeout_error():
    """
    TimeoutError must also be raised on the executor path (from inside a
    running loop).
    """

    async def _slow() -> str:
        await asyncio.sleep(1.0)
        return "never"

    with pytest.raises(TimeoutError):
        run_async(_slow(), timeout=0.05)


# ── 4. run_async_fire_and_forget submits without blocking ─────────────────────


def test_fire_and_forget_returns_immediately():
    """
    fire_and_forget must return None right away — it must not wait for the
    coroutine to complete.
    """

    async def _noop() -> None:
        return None

    started = time.perf_counter()
    rv = run_async_fire_and_forget(_noop())
    elapsed = time.perf_counter() - started

    assert rv is None
    assert elapsed < 0.05, f"fire_and_forget blocked for {elapsed:.3f}s"


def test_fire_and_forget_actually_runs_coroutine():
    """
    The submitted coroutine must eventually run on the executor loop.
    We use a threading.Event as a thread-safe completion signal.
    """
    done = threading.Event()

    async def _mark_done() -> None:
        done.set()

    run_async_fire_and_forget(_mark_done())

    assert done.wait(timeout=2.0), "fire-and-forget coroutine never executed"


def test_fire_and_forget_does_not_propagate_exceptions():
    """
    If the background coroutine raises, fire_and_forget must swallow it
    (it runs on the executor loop and there's no caller to surface to).
    """

    async def _explode() -> None:
        raise ValueError("background explosion")

    # Should NOT raise to the caller.
    run_async_fire_and_forget(_explode())

    # Give the executor a moment to attempt to run and fail silently.
    time.sleep(0.1)


# ── Auxiliary: _ensure_executor caching and thread-safety ────────────────────


def test_ensure_executor_returns_same_loop_on_repeat_calls():
    """
    _ensure_executor must cache the loop; subsequent calls return the same
    object (so the executor thread is started exactly once per process).
    """
    loop_a = _ensure_executor()
    loop_b = _ensure_executor()

    assert loop_a is loop_b
    assert not loop_a.is_closed()


def test_executor_thread_is_daemon():
    """
    The executor thread must be a daemon so it never blocks interpreter
    shutdown.
    """
    _ensure_executor()
    assert async_runner._executor_thread is not None
    assert async_runner._executor_thread.is_alive()
    assert async_runner._executor_thread.daemon is True
    assert "friday-async-executor" in async_runner._executor_thread.name


def test_executor_thread_named_distinctively():
    """The thread should carry a recognisable name for debugging."""
    _ensure_executor()
    assert async_runner._executor_thread is not None
    assert async_runner._executor_thread.name == "friday-async-executor"


def test_run_async_under_lock_initialises_executor_once():
    """
    Repeated run_async() invocations from the sync path must reuse the
    cached executor loop — the lock-protected _ensure_executor path
    must not leak threads.
    """
    _ensure_executor()  # pre-warm
    thread_before = async_runner._executor_thread

    async def _noop() -> None:
        return None

    run_async(_noop())
    run_async(_noop())
    run_async(_noop())

    assert async_runner._executor_thread is thread_before
