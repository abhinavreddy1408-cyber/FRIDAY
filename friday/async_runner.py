"""
F.R.I.D.A.Y. — Async Runner Utility
─────────────────────────────────────
Safely runs coroutines from any context:
  - From a thread with no event loop → creates one
  - From a thread with a running loop → uses a dedicated executor thread
  - Never calls asyncio.run() inside a running loop (prevents RuntimeError)

Import and use run_async() everywhere instead of asyncio.run().
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

T = TypeVar("T")

# One dedicated executor thread that owns a permanent event loop.
# All coroutines submitted from sync threads run here.
_executor_lock = threading.Lock()
_executor_loop: asyncio.AbstractEventLoop | None = None
_executor_thread: threading.Thread | None = None


def _ensure_executor() -> asyncio.AbstractEventLoop:
    """
    Ensure the background executor loop is running.
    Called once; subsequent calls return the cached loop.
    """
    global _executor_loop, _executor_thread

    with _executor_lock:
        if _executor_loop is not None and not _executor_loop.is_closed():
            return _executor_loop

        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(
            target=_run,
            daemon=True,
            name="friday-async-executor",
        )
        thread.start()

        _executor_loop = loop
        _executor_thread = thread
        logger.debug("[AsyncRunner] Executor loop started.")
        return loop


def run_async(coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    """
    Run a coroutine from any context — sync thread, async thread, or main.

    Args:
        coro:    The coroutine to run.
        timeout: Optional timeout in seconds. Raises TimeoutError on breach.

    Returns:
        The coroutine's return value.

    Raises:
        TimeoutError:   If the coroutine exceeds the timeout.
        Exception:      Any exception raised inside the coroutine.
    """
    try:
        # If there's already a running loop in this thread, use executor
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        # We're inside an async context — submit to the dedicated executor loop
        executor_loop = _ensure_executor()
        future = asyncio.run_coroutine_threadsafe(coro, executor_loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"run_async: coroutine exceeded {timeout}s timeout."
            ) from exc
    else:
        # No running loop in this thread — safe to use asyncio.run()
        if timeout is not None:
            async def _with_timeout() -> T:
                return await asyncio.wait_for(coro, timeout=timeout)
            return asyncio.run(_with_timeout())
        return asyncio.run(coro)


def run_async_fire_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
    """
    Submit a coroutine to the executor loop without waiting for the result.
    Use for non-critical background tasks (logging, metrics, etc).
    """
    loop = _ensure_executor()
    asyncio.run_coroutine_threadsafe(coro, loop)
