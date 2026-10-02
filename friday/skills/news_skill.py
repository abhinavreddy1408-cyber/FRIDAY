"""
task_queue.py - Parallel Task Execution Engine for F.R.I.D.A.Y.

Manages prioritized, concurrent task execution with lifecycle tracking,
cancellation support, and resource-aware scheduling.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(order=True)
class Task:
    """Represents a single unit of work managed by the task queue."""

    priority: int
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    fn: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict[str, Any] = field(default_factory=dict, compare=False)
    timeout: float | None = field(default=None, compare=False)
    callback: Callable[[Any], None] | None = field(default=None, compare=False)
    error_handler: Callable[[Exception], None] | None = field(default=None, compare=False)

    # Runtime state
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    result: Any = field(default=None, compare=False)
    error: Exception | None = field(default=None, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    started_at: float | None = field(default=None, compare=False)
    completed_at: float | None = field(default=None, compare=False)
    future: Future | None = field(default=None, compare=False)

    @classmethod
    def create(
        cls,
        fn: Callable,
        *,
        name: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        callback: Callable[[Any], None] | None = None,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> Task:
        return cls(
            task_id=str(uuid.uuid4()),
            name=name or fn.__name__,
            priority=priority.value,
            fn=fn,
            args=args,
            kwargs=kwargs or {},
            timeout=timeout,
            callback=callback,
            error_handler=error_handler,
        )

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        }


@dataclass
class QueueStats:
    """Runtime statistics for the task queue."""

    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_cancelled: int = 0
    total_timeouts: int = 0
    active_workers: int = 0
    pending_count: int = 0


# ---------------------------------------------------------------------------
# Task Queue
# ---------------------------------------------------------------------------


class TaskQueue:
    """
    Thread-safe, priority-based parallel task executor.

    Features:
    - Priority scheduling (CRITICAL → BACKGROUND)
    - Per-task timeout enforcement
    - Lifecycle callbacks and error handlers
    - Graceful shutdown with drain support
    - Runtime statistics
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 100,
    ) -> None:
        self._max_workers = max_workers
        self._max_queue_size = max_queue_size

        self._tasks: dict[str, Task] = {}
        self._pending: list[Task] = []  # acts as a min-heap by priority
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="friday-task",
        )
        self._stats = QueueStats()
        self._shutdown = False

        # Dispatcher thread feeds tasks to the executor
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="friday-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()
        logger.info("TaskQueue started (workers=%d, queue_cap=%d)", max_workers, max_queue_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        fn: Callable,
        *,
        name: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        callback: Callable[[Any], None] | None = None,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> str | None:
        """
        Submit a callable for async execution.

        Returns the task_id string, or None if the queue is full / shut down.
        """
        if self._shutdown:
            logger.warning("TaskQueue is shut down; rejecting task '%s'", name or fn.__name__)
            return None

        with self._lock:
            if len(self._pending) >= self._max_queue_size:
                logger.warning("Task queue full; dropping task '%s'", name or fn.__name__)
                return None

            task = Task.create(
                fn,
                name=name,
                priority=priority,
                args=args,
                kwargs=kwargs,
                timeout=timeout,
                callback=callback,
                error_handler=error_handler,
            )
            self._tasks[task.task_id] = task
            self._stats.total_submitted += 1
            self._stats.pending_count += 1

            # Insert in priority order (lower value = higher priority)
            self._insert_sorted(task)
            self._not_empty.notify()

        logger.debug(
            "Task submitted: id=%s name='%s' priority=%s", task.task_id, task.name, priority.name
        )
        return task.task_id

    def cancel(self, task_id: str) -> bool:
        """
        Attempt to cancel a pending task.
        Running tasks cannot be cancelled.
        Returns True if cancellation succeeded.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status != TaskStatus.PENDING:
                return False

            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            with contextlib.suppress(ValueError):
                self._pending.remove(task)
            self._stats.total_cancelled += 1
            self._stats.pending_count = max(0, self._stats.pending_count - 1)

        logger.info("Task cancelled: id=%s name='%s'", task_id, task.name)
        return True

    def get_task(self, task_id: str) -> Task | None:
        """Return a snapshot of task state by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_stats(self) -> QueueStats:
        """Return a snapshot of current queue statistics."""
        with self._lock:
            self._stats.active_workers = self._count_running()
            self._stats.pending_count = len(self._pending)
            return QueueStats(**self._stats.__dict__)

    def get_active_tasks(self) -> list[Task]:
        """Return all non-terminal tasks."""
        with self._lock:
            return [t for t in self._tasks.values() if not t.is_terminal]

    def shutdown(self, wait: bool = True, timeout: float = 10.0) -> None:
        """Shut down the queue and executor gracefully."""
        logger.info("TaskQueue shutting down (wait=%s)…", wait)
        self._shutdown = True
        with self._not_empty:
            self._not_empty.notify_all()

        if wait:
            self._dispatcher.join(timeout=timeout)
            self._executor.shutdown(wait=True, cancel_futures=False)
        else:
            self._executor.shutdown(wait=False, cancel_futures=True)

        logger.info("TaskQueue shutdown complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_sorted(self, task: Task) -> None:
        """Insert task into _pending maintaining ascending priority order."""
        lo, hi = 0, len(self._pending)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending[mid].priority <= task.priority:
                lo = mid + 1
            else:
                hi = mid
        self._pending.insert(lo, task)

    def _dispatch_loop(self) -> None:
        """Pull tasks from the priority list and submit to ThreadPoolExecutor."""
        while not self._shutdown:
            with self._not_empty:
                while not self._pending and not self._shutdown:
                    self._not_empty.wait(timeout=1.0)
                if self._shutdown and not self._pending:
                    break
                if not self._pending:
                    continue
                task = self._pending.pop(0)
                self._stats.pending_count = max(0, self._stats.pending_count - 1)

            future = self._executor.submit(self._run_task, task)
            task.future = future

    def _run_task(self, task: Task) -> None:
        """Execute a single task with timeout and callback handling."""
        with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

        logger.debug("Task started: id=%s name='%s'", task.task_id, task.name)

        try:
            if task.timeout:
                result = self._run_with_timeout(task)
            else:
                result = task.fn(*task.args, **task.kwargs)

            with self._lock:
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                self._stats.total_completed += 1

            logger.debug(
                "Task completed: id=%s name='%s' elapsed=%.3fs",
                task.task_id,
                task.name,
                task.elapsed or 0,
            )

            if task.callback:
                self._safe_call(task.callback, result, label=f"callback for '{task.name}'")

        except TimeoutError:
            with self._lock:
                task.status = TaskStatus.TIMEOUT
                task.completed_at = time.time()
                self._stats.total_timeouts += 1
            logger.warning("Task timeout: id=%s name='%s'", task.task_id, task.name)
            if task.error_handler:
                self._safe_call(
                    task.error_handler,
                    TimeoutError(f"Task '{task.name}' timed out"),
                    label="error_handler",
                )

        except Exception as exc:
            with self._lock:
                task.error = exc
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                self._stats.total_failed += 1
            logger.error(
                "Task failed: id=%s name='%s' error=%s", task.task_id, task.name, exc, exc_info=True
            )
            if task.error_handler:
                self._safe_call(task.error_handler, exc, label="error_handler")

    def _run_with_timeout(self, task: Task) -> Any:
        """Run task.fn in a sub-thread; raise TimeoutError if it exceeds task.timeout."""
        result_holder: dict[str, Any] = {}
        exc_holder: dict[str, Exception] = {}

        def _target():
            try:
                result_holder["value"] = task.fn(*task.args, **task.kwargs)
            except Exception as e:
                exc_holder["error"] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=task.timeout)

        if t.is_alive():
            raise TimeoutError(f"Task '{task.name}' exceeded timeout of {task.timeout}s")
        if "error" in exc_holder:
            raise exc_holder["error"]
        return result_holder.get("value")

    @staticmethod
    def _safe_call(fn: Callable, *args, label: str = "") -> None:
        """Invoke a callback without propagating exceptions."""
        try:
            fn(*args)
        except Exception as exc:
            logger.error("Exception in %s: %s", label or fn.__name__, exc, exc_info=True)

    def _count_running(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> TaskQueue:
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()
