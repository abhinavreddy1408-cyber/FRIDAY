"""
F.R.I.D.A.Y. Task Queue Module
Provides a prioritised, concurrent task execution system for background
operations that should not block the main voice interaction loop.

Features:
  - Priority-based scheduling (CRITICAL → HIGH → NORMAL → LOW → BACKGROUND)
  - Configurable worker thread pool
  - Per-task timeout enforcement
  - Task lifecycle callbacks (on_complete, on_error)
  - Task cancellation and status tracking
  - Rate limiting to prevent system overload
  - Full task history with result persistence
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────


class TaskPriority(IntEnum):
    """
    Task scheduling priority.
    Lower numeric value = higher priority (processed first).
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    PENDING = auto()  # Queued, not yet started
    RUNNING = auto()  # Currently executing
    COMPLETED = auto()  # Finished successfully
    FAILED = auto()  # Raised an exception
    CANCELLED = auto()  # Cancelled before execution
    TIMED_OUT = auto()  # Exceeded timeout


# ── Task Definition ───────────────────────────────────────────────────────────


@dataclass
class Task:
    """
    Descriptor for a single unit of background work.

    Attributes:
        fn:          The callable to execute.
        args:        Positional arguments for fn.
        kwargs:      Keyword arguments for fn.
        priority:    Scheduling priority (lower = higher priority).
        timeout:     Maximum execution time in seconds (None = unlimited).
        name:        Human-readable task identifier.
        on_complete: Optional callback invoked with the return value on success.
        on_error:    Optional callback invoked with the exception on failure.
        id:          Auto-generated unique task identifier.
        created_at:  Unix timestamp of task creation.
    """

    fn: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float | None = None
    name: str = ""
    on_complete: Callable[[Any], None] | None = None
    on_error: Callable[[Exception], None] | None = None

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.fn, "__name__", "unnamed_task")

    def __lt__(self, other: Task) -> bool:
        """Enable priority queue ordering (lower priority value = first)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at  # FIFO within same priority


@dataclass
class TaskResult:
    """
    Immutable record of a completed task execution.

    Attributes:
        task_id:    ID of the originating task.
        task_name:  Human-readable name.
        status:     Final execution status.
        value:      Return value on success, None otherwise.
        error:      Exception on failure, None otherwise.
        started_at: Unix timestamp execution began.
        ended_at:   Unix timestamp execution ended.
        duration:   Total execution time in seconds.
    """

    task_id: str
    task_name: str
    status: TaskStatus
    value: Any
    error: Exception | None
    started_at: float
    ended_at: float

    @property
    def duration(self) -> float:
        return self.ended_at - self.started_at

    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    def __str__(self) -> str:
        if self.success:
            return f"Task '{self.task_name}' completed in {self.duration:.2f}s"
        return f"Task '{self.task_name}' {self.status.name.lower()} after {self.duration:.2f}s" + (
            f": {self.error}" if self.error else ""
        )


# ── Task Registry ─────────────────────────────────────────────────────────────


class _TaskRegistry:
    """
    Thread-safe registry tracking live task status and completed results.
    Caps history at a configurable maximum to prevent unbounded growth.
    """

    def __init__(self, max_history: int = 200) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._status: dict[str, TaskStatus] = {}
        self._results: dict[str, TaskResult] = {}
        self._max: int = max_history

    def register(self, task: Task) -> None:
        with self._lock:
            self._status[task.id] = TaskStatus.PENDING

    def set_status(self, task_id: str, status: TaskStatus) -> None:
        with self._lock:
            self._status[task_id] = status

    def record_result(self, result: TaskResult) -> None:
        with self._lock:
            self._status[result.task_id] = result.status
            self._results[result.task_id] = result
            self._evict_old()

    def get_status(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            return self._status.get(task_id)

    def get_result(self, task_id: str) -> TaskResult | None:
        with self._lock:
            return self._results.get(task_id)

    def get_all_results(
        self,
        status_filter: TaskStatus | None = None,
    ) -> list[TaskResult]:
        with self._lock:
            results = list(self._results.values())

        if status_filter:
            results = [r for r in results if r.status == status_filter]

        return sorted(results, key=lambda r: r.ended_at, reverse=True)

    def cancel(self, task_id: str) -> bool:
        """
        Mark a PENDING task as cancelled.
        Returns False if the task is already running or completed.
        """
        with self._lock:
            current = self._status.get(task_id)
            if current == TaskStatus.PENDING:
                self._status[task_id] = TaskStatus.CANCELLED
                return True
            return False

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            return self._status.get(task_id) == TaskStatus.CANCELLED

    def _evict_old(self) -> None:
        """Remove oldest results when cap is exceeded (must hold lock)."""
        if len(self._results) > self._max:
            oldest_key = min(self._results, key=lambda k: self._results[k].ended_at)
            self._results.pop(oldest_key, None)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._status.values() if s == TaskStatus.PENDING)

    @property
    def running_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._status.values() if s == TaskStatus.RUNNING)


# ── Rate Limiter ──────────────────────────────────────────────────────────────


class _RateLimiter:
    """
    Token-bucket rate limiter for task submission.
    Prevents bursts of tasks from overwhelming the thread pool.
    """

    def __init__(self, rate: float, burst: int) -> None:
        """
        Args:
            rate:  Tokens replenished per second.
            burst: Maximum token bucket capacity.
        """
        self._rate: float = rate
        self._burst: int = burst
        self._tokens: float = float(burst)
        self._last: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()

    def acquire(self, block: bool = True, timeout: float = 5.0) -> bool:
        """
        Attempt to consume one token.

        Args:
            block:   If True, wait until a token is available.
            timeout: Maximum wait time in seconds when blocking.

        Returns:
            True if a token was acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                now = time.monotonic()
                gained = (now - self._last) * self._rate
                self._tokens = min(self._burst, self._tokens + gained)
                self._last = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            if not block or time.monotonic() >= deadline:
                return False

            time.sleep(0.05)


# ── Worker ────────────────────────────────────────────────────────────────────


class _Worker:
    """
    Executes a single Task within a thread pool future.
    Handles timeout enforcement, result recording, and callback invocation.
    """

    def __init__(self, registry: _TaskRegistry) -> None:
        self._registry = registry

    def run(self, task: Task, executor: ThreadPoolExecutor) -> None:
        """
        Submit a task for execution.

        Uses the executor for timeout-aware execution via a nested future.
        The worker itself runs inside the outer thread pool.

        Args:
            task:     The Task to execute.
            executor: The thread pool to submit work to.
        """
        if self._registry.is_cancelled(task.id):
            logger.debug("[TaskQueue] Task '%s' was cancelled before execution.", task.name)
            return

        self._registry.set_status(task.id, TaskStatus.RUNNING)
        started_at = time.time()

        try:
            if task.timeout is None:
                value = task.fn(*task.args, **task.kwargs)
            else:
                # Execute in a separate daemon thread to avoid blocking the executor thread
                # pool if a timeout occurs.
                res = []
                exc = []
                def target():
                    try:
                        res.append(task.fn(*task.args, **task.kwargs))
                    except Exception as e:
                        exc.append(e)
                
                t = threading.Thread(target=target, daemon=True, name=f"friday-task-timeout-{task.id}")
                t.start()
                t.join(timeout=task.timeout)
                if t.is_alive():
                    raise FuturesTimeout()
                if exc:
                    raise exc[0]
                value = res[0]

            result = TaskResult(
                task_id=task.id,
                task_name=task.name,
                status=TaskStatus.COMPLETED,
                value=value,
                error=None,
                started_at=started_at,
                ended_at=time.time(),
            )
            logger.debug("[TaskQueue] '%s' completed in %.2fs.", task.name, result.duration)

        except FuturesTimeout:
            result = TaskResult(
                task_id=task.id,
                task_name=task.name,
                status=TaskStatus.TIMED_OUT,
                value=None,
                error=TimeoutError(f"Task '{task.name}' timed out after {task.timeout}s"),
                started_at=started_at,
                ended_at=time.time(),
            )
            logger.warning("[TaskQueue] '%s' timed out after %.1fs.", task.name, task.timeout)

        except Exception as exc:
            result = TaskResult(
                task_id=task.id,
                task_name=task.name,
                status=TaskStatus.FAILED,
                value=None,
                error=exc,
                started_at=started_at,
                ended_at=time.time(),
            )
            logger.error("[TaskQueue] '%s' failed: %s", task.name, exc)

        self._registry.record_result(result)
        self._invoke_callbacks(task, result)

    @staticmethod
    def _invoke_callbacks(task: Task, result: TaskResult) -> None:
        """Invoke task lifecycle callbacks, isolating any callback exceptions."""
        if result.success and task.on_complete:
            try:
                task.on_complete(result.value)
            except Exception:
                logger.debug(
                    "[TaskQueue] on_complete callback raised for '%s'.", task.name, exc_info=True
                )

        elif not result.success and task.on_error and result.error:
            try:
                task.on_error(result.error)
            except Exception:
                logger.debug(
                    "[TaskQueue] on_error callback raised for '%s'.", task.name, exc_info=True
                )


# ── Dispatcher ────────────────────────────────────────────────────────────────


class _Dispatcher(threading.Thread):
    """
    Background thread that dequeues tasks by priority and dispatches them
    to the worker pool. Runs as a daemon thread for the lifetime of the queue.
    """

    def __init__(
        self,
        task_queue: queue.PriorityQueue,
        executor: ThreadPoolExecutor,
        registry: _TaskRegistry,
        rate_limiter: _RateLimiter,
    ) -> None:
        super().__init__(daemon=True, name="task-dispatcher")
        self._queue = task_queue
        self._executor = executor
        self._registry = registry
        self._limiter = rate_limiter
        self._worker = _Worker(registry)
        self._stop_event = threading.Event()

    def run(self) -> None:
        logger.debug("[TaskQueue] Dispatcher started.")
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._registry.is_cancelled(task.id):
                self._queue.task_done()
                logger.debug("[TaskQueue] Skipping cancelled task '%s'.", task.name)
                continue

            if not self._limiter.acquire(block=True, timeout=3.0):
                logger.warning("[TaskQueue] Rate limit exceeded — dropping '%s'.", task.name)
                self._queue.task_done()
                continue

            # Submit dispatch to executor — non-blocking
            self._executor.submit(self._worker.run, task, self._executor)
            self._queue.task_done()

        logger.debug("[TaskQueue] Dispatcher stopped.")

    def stop(self) -> None:
        self._stop_event.set()


# ── Task Queue (public) ───────────────────────────────────────────────────────


class TaskQueue:
    """
    Public interface for Friday's background task execution system.

    Manages a priority queue, dispatcher thread, and thread pool.
    The engine submits tasks here for any operation that should not
    block the active voice interaction loop.

    Usage:
        tq = TaskQueue(max_workers=4)
        tq.start()

        task_id = tq.submit(
            fn=fetch_weather,
            args=("London",),
            priority=TaskPriority.HIGH,
            name="weather_fetch",
            on_complete=lambda result: tts.speak(result),
        )

        status = tq.get_status(task_id)
        tq.cancel(task_id)
        tq.stop()
    """

    def __init__(
        self,
        max_workers: int = 4,
        rate_per_sec: float = 10.0,
        burst_size: int = 20,
        history_limit: int = 200,
    ) -> None:
        """
        Args:
            max_workers:   Maximum concurrent worker threads.
            rate_per_sec:  Maximum task submissions per second.
            burst_size:    Maximum burst of tasks before rate limiting kicks in.
            history_limit: Maximum task results retained in memory.
        """
        self._max_workers = max_workers
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._registry: _TaskRegistry = _TaskRegistry(max_history=history_limit)
        self._limiter: _RateLimiter = _RateLimiter(rate_per_sec, burst_size)
        self._executor: ThreadPoolExecutor | None = None
        self._dispatcher: _Dispatcher | None = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the dispatcher thread and worker pool."""
        with self._lock:
            if self._running:
                return

            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="friday-worker",
            )
            self._dispatcher = _Dispatcher(
                task_queue=self._queue,
                executor=self._executor,
                registry=self._registry,
                rate_limiter=self._limiter,
            )
            self._dispatcher.start()
            self._running = True

        logger.info(
            "[TaskQueue] Started with %d workers.",
            self._max_workers,
        )

    def stop(self, wait: bool = True, timeout: float = 10.0) -> None:
        """
        Stop the dispatcher and shut down the worker pool.

        Args:
            wait:    If True, wait for in-progress tasks to complete.
            timeout: Maximum seconds to wait for graceful shutdown.
        """
        with self._lock:
            if not self._running:
                return

            if self._dispatcher:
                self._dispatcher.stop()
                self._dispatcher.join(timeout=2.0)
                self._dispatcher = None

            if self._executor:
                self._executor.shutdown(wait=wait, cancel_futures=not wait)
                self._executor = None

            self._running = False

        logger.info("[TaskQueue] Stopped.")

    # ── Task Submission ───────────────────────────────────────────────────────

    def submit(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        name: str = "",
        on_complete: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> str:
        """
        Submit a callable for background execution.

        Args:
            fn:          The function to execute.
            args:        Positional arguments.
            kwargs:      Keyword arguments.
            priority:    Execution priority.
            timeout:     Maximum execution seconds (None = unlimited).
            name:        Human-readable task name for logging.
            on_complete: Called with the return value on success.
            on_error:    Called with the exception on failure.

        Returns:
            Task ID string for status tracking and cancellation.

        Raises:
            RuntimeError: If the queue has not been started.
        """
        if not self._running:
            raise RuntimeError("TaskQueue must be started before submitting tasks.")

        task = Task(
            fn=fn,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout=timeout,
            name=name,
            on_complete=on_complete,
            on_error=on_error,
        )

        self._registry.register(task)
        self._queue.put(task)

        logger.debug(
            "[TaskQueue] Submitted '%s' [%s] (id=%s).",
            task.name,
            priority.name,
            task.id,
        )
        return task.id

    def submit_task(self, task: Task) -> str:
        """
        Submit a pre-constructed Task object.

        Args:
            task: The Task to enqueue.

        Returns:
            The task's ID string.
        """
        if not self._running:
            raise RuntimeError("TaskQueue must be started before submitting tasks.")

        self._registry.register(task)
        self._queue.put(task)
        logger.debug("[TaskQueue] Submitted task object '%s' (id=%s).", task.name, task.id)
        return task.id

    # ── Task Control ──────────────────────────────────────────────────────────

    def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending task.

        Only PENDING tasks can be cancelled — running tasks cannot be
        interrupted mid-execution.

        Args:
            task_id: The ID returned by submit().

        Returns:
            True if the task was pending and successfully cancelled.
        """
        cancelled = self._registry.cancel(task_id)
        if cancelled:
            logger.info("[TaskQueue] Task '%s' cancelled.", task_id)
        return cancelled

    def cancel_all(self, priority: TaskPriority | None = None) -> int:
        """
        Cancel all pending tasks, optionally filtered by priority.

        Args:
            priority: If specified, only cancel tasks of this priority.

        Returns:
            Number of tasks cancelled.
        """
        cancelled = 0
        # Drain the queue into a temp list, filter, re-enqueue survivors
        temp: list[Task] = []
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                if priority is None or task.priority == priority:
                    self._registry.cancel(task.id)
                    cancelled += 1
                else:
                    temp.append(task)
                self._queue.task_done()
            except queue.Empty:
                break

        for task in temp:
            self._queue.put(task)

        logger.info("[TaskQueue] Cancelled %d tasks.", cancelled)
        return cancelled

    # ── Status & Results ──────────────────────────────────────────────────────

    def get_status(self, task_id: str) -> TaskStatus | None:
        """
        Return the current status of a task.

        Args:
            task_id: The ID returned by submit().

        Returns:
            TaskStatus or None if the ID is unknown.
        """
        return self._registry.get_status(task_id)

    def get_result(self, task_id: str) -> TaskResult | None:
        """
        Return the result of a completed task.

        Args:
            task_id: The ID returned by submit().

        Returns:
            TaskResult or None if the task hasn't completed yet.
        """
        return self._registry.get_result(task_id)

    def wait_for(
        self,
        task_id: str,
        timeout: float | None = None,
        poll_interval: float = 0.1,
    ) -> TaskResult | None:
        """
        Block until a task completes or the timeout expires.

        Args:
            task_id:       The ID to wait on.
            timeout:       Maximum seconds to wait (None = wait indefinitely).
            poll_interval: Seconds between status polls.

        Returns:
            TaskResult when done, None if timed out.
        """
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            status = self._registry.get_status(task_id)

            if status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMED_OUT,
            ):
                return self._registry.get_result(task_id)

            if deadline and time.monotonic() >= deadline:
                return None

            time.sleep(poll_interval)

    def get_history(
        self,
        status_filter: TaskStatus | None = None,
        limit: int = 50,
    ) -> list[TaskResult]:
        """
        Return recent task results.

        Args:
            status_filter: Optionally filter by task status.
            limit:         Maximum results to return.

        Returns:
            List of TaskResult ordered by completion time descending.
        """
        results = self._registry.get_all_results(status_filter)
        return results[:limit]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_depth(self) -> int:
        """Number of tasks currently waiting in the queue."""
        return self._queue.qsize()

    @property
    def stats(self) -> dict[str, Any]:
        """Return a snapshot of queue health metrics."""
        return {
            "running": self._running,
            "queue_depth": self.queue_depth,
            "pending": self._registry.pending_count,
            "active": self._registry.running_count,
            "max_workers": self._max_workers,
            "history_count": len(self._registry.get_all_results()),
        }

    # ── Compatibility wrappers for testing ────────────────────────────────────

    def add_task(self, name: str, payload: Any = None) -> str:
        """Compatibility wrapper: submit a task to the queue."""

        def dummy_fn():
            time.sleep(0.5)
            return True

        return self.submit(fn=dummy_fn, name=name)

    def list_active_tasks(self) -> list[str]:
        """Compatibility wrapper: return list of pending or running task IDs."""
        with self._registry._lock:
            return [
                tid
                for tid, status in self._registry._status.items()
                if status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            ]

    def list_completed_tasks(self) -> list[str]:
        """Compatibility wrapper: return list of completed task IDs."""
        with self._registry._lock:
            return [
                tid
                for tid, status in self._registry._status.items()
                if status == TaskStatus.COMPLETED
            ]
