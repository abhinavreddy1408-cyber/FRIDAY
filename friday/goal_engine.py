"""
F.R.I.D.A.Y. Goal Engine Module
Manages long-term, multi-step goals that persist across sessions.

A goal is a high-level objective decomposed into an ordered sequence of
steps. The engine tracks progress, handles failures with retry/skip logic,
executes steps in the background via the task queue, and surfaces status
to the user on request.

Goal lifecycle:
  PENDING → ACTIVE → (step execution) → COMPLETED | FAILED | PAUSED
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from friday.config import config
from friday.task_queue import TaskPriority, TaskQueue

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)

_instance: GoalEngine | None = None
_lock = threading.Lock()


# ── Enums ─────────────────────────────────────────────────────────────────────


class GoalStatus(Enum):
    PENDING = "pending"  # Created, not yet started
    ACTIVE = "active"  # Currently being executed
    PAUSED = "paused"  # Manually suspended
    COMPLETED = "completed"  # All steps finished successfully
    FAILED = "failed"  # Unrecoverable failure
    CANCELLED = "cancelled"  # Explicitly cancelled by user


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class GoalPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class GoalStep:
    """
    A single executable unit within a goal.

    Attributes:
        id:           Unique step identifier.
        description:  Human-readable step description.
        action:       Action name or LLM prompt to execute.
        status:       Current execution status.
        result:       Output from step execution (if completed).
        error:        Error message if the step failed.
        max_retries:  Maximum retry attempts on failure.
        retries:      Current retry count.
        created_at:   Unix timestamp of step creation.
        started_at:   Unix timestamp execution began (None if not started).
        completed_at: Unix timestamp execution ended (None if not finished).
        metadata:     Arbitrary key-value context for the step.
    """

    id: str
    description: str
    action: str
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None
    max_retries: int = 2
    retries: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def can_retry(self) -> bool:
        return self.retries < self.max_retries

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.SKIPPED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "action": self.action,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "max_retries": self.max_retries,
            "retries": self.retries,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalStep:
        return cls(
            id=data["id"],
            description=data["description"],
            action=data["action"],
            status=StepStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            max_retries=data.get("max_retries", 2),
            retries=data.get("retries", 0),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Goal:
    """
    A persistent, multi-step objective.

    Attributes:
        id:           Unique goal identifier.
        title:        Short descriptive title.
        description:  Full goal description.
        steps:        Ordered list of GoalStep objects.
        status:       Current goal lifecycle status.
        priority:     Scheduling priority.
        created_at:   Unix timestamp of creation.
        updated_at:   Unix timestamp of last modification.
        completed_at: Unix timestamp of completion (None if not done).
        tags:         Optional classification tags.
        metadata:     Arbitrary context data.
    """

    id: str
    title: str
    description: str
    steps: list[GoalStep]
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    deadline: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED))

    @property
    def progress_pct(self) -> float:
        if not self.steps:
            return 0.0
        return round((self.completed_steps / self.total_steps) * 100, 1)

    @property
    def current_step(self) -> GoalStep | None:
        """Return the first non-terminal step."""
        for step in self.steps:
            if not step.is_terminal:
                return step
        return None

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.CANCELLED,
        )

    @property
    def duration(self) -> float | None:
        if self.completed_at:
            return self.completed_at - self.created_at
        return None

    def progress_summary(self) -> str:
        """Return a natural-language progress summary for TTS."""
        if self.status == GoalStatus.COMPLETED:
            dur = f" in {self.duration:.0f} seconds" if self.duration else ""
            return f"Goal '{self.title}' completed{dur}."

        if self.status == GoalStatus.FAILED:
            step = self.current_step
            err = f": {step.error}" if step and step.error else ""
            return f"Goal '{self.title}' failed{err}."

        step = self.current_step
        step_desc = f" Current step: {step.description}." if step else ""
        return (
            f"Goal '{self.title}' is {self.progress_pct:.0f}% complete "
            f"({self.completed_steps} of {self.total_steps} steps done).{step_desc}"
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "deadline": self.deadline,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            steps=[GoalStep.from_dict(s) for s in data.get("steps", [])],
            status=GoalStatus(data.get("status", "pending")),
            priority=GoalPriority(data.get("priority", "normal")),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            completed_at=data.get("completed_at"),
            deadline=data.get("deadline"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


# ── Database Schema ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id           TEXT    PRIMARY KEY,
    title        TEXT    NOT NULL,
    description  TEXT    NOT NULL DEFAULT '',
    steps        TEXT    NOT NULL DEFAULT '[]',
    status       TEXT    NOT NULL DEFAULT 'pending',
    priority     TEXT    NOT NULL DEFAULT 'normal',
    created_at   REAL    NOT NULL,
    updated_at   REAL    NOT NULL,
    completed_at REAL,
    deadline     REAL,
    tags         TEXT    NOT NULL DEFAULT '[]',
    metadata     TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_goals_status   ON goals (status);
CREATE INDEX IF NOT EXISTS idx_goals_updated  ON goals (updated_at);
"""


# ── Goal Repository ───────────────────────────────────────────────────────────


class _GoalRepository:
    """
    SQLite-backed persistence layer for goals.
    Thread-safe via per-thread connections and a write lock.
    """

    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._path), timeout=10.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        try:
            yield self._local.conn
        except sqlite3.DatabaseError as exc:
            self._local.conn.rollback()
            logger.error("[GoalEngine/DB] Database error: %s", exc)
            raise

    def _init_schema(self) -> None:
        """Initialize schema and perform migrations if necessary."""
        with self._conn() as conn:
            # Create table if it doesn't exist
            conn.executescript(_SCHEMA)

            # Migration: Ensure all modern columns exist
            try:
                cursor = conn.execute("PRAGMA table_info(goals)")
                columns = [row["name"] for row in cursor.fetchall()]

                migrations = [
                    ("description", "TEXT NOT NULL DEFAULT ''"),
                    ("steps", "TEXT NOT NULL DEFAULT '[]'"),
                    ("priority", "TEXT NOT NULL DEFAULT 'normal'"),
                    ("updated_at", "REAL NOT NULL DEFAULT 0"),
                    ("deadline", "REAL"),
                    ("tags", "TEXT NOT NULL DEFAULT '[]'"),
                    ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
                ]

                for col_name, col_def in migrations:
                    if col_name not in columns:
                        logger.info(
                            "[GoalEngine/DB] Migrating schema: adding '%s' column.", col_name
                        )
                        conn.execute(f"ALTER TABLE goals ADD COLUMN {col_name} {col_def}")

                # Special case: map last_active to updated_at if it exists
                if "last_active" in columns and "updated_at" in columns:
                    conn.execute("UPDATE goals SET updated_at = last_active WHERE updated_at = 0")

                # Create indexes that depend on migrated columns
                conn.execute("CREATE INDEX IF NOT EXISTS idx_goals_priority ON goals (priority)")

            except sqlite3.Error as e:
                logger.error("[GoalEngine/DB] Migration failed: %s", e)

            conn.commit()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save(self, goal: Goal) -> None:
        """Insert or replace a goal record."""
        goal.updated_at = time.time()
        with self._write_lock, self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO goals
                    (id, title, description, steps, status, priority,
                     created_at, updated_at, completed_at, deadline, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal.id,
                    goal.title,
                    goal.description,
                    json.dumps([s.to_dict() for s in goal.steps]),
                    goal.status.value,
                    goal.priority.value,
                    goal.created_at,
                    goal.updated_at,
                    goal.completed_at,
                    goal.deadline,
                    json.dumps(goal.tags),
                    json.dumps(goal.metadata),
                ),
            )
            conn.commit()

    def load(self, goal_id: str) -> Goal | None:
        """Fetch a single goal by ID."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
        return self._row_to_goal(row) if row else None

    def load_all(
        self,
        status: GoalStatus | None = None,
        priority: GoalPriority | None = None,
        limit: int = 100,
    ) -> list[Goal]:
        """Fetch goals with optional status/priority filtering."""
        clauses: list[str] = []
        params: list[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if priority:
            clauses.append("priority = ?")
            params.append(priority.value)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM goals {where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()

        return [self._row_to_goal(r) for r in rows]

    def delete(self, goal_id: str) -> bool:
        """Delete a goal by ID. Returns True if found and deleted."""
        with self._write_lock, self._conn() as conn:
            cursor = conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count(self, status: GoalStatus | None = None) -> int:
        with self._conn() as conn:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) FROM goals WHERE status = ?",
                    (status.value,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM goals").fetchone()
            return row[0] if row else 0

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> Goal:
        return Goal.from_dict(
            {
                "id": row["id"],
                "title": row["title"],
                "description": row["description"],
                "steps": json.loads(row["steps"]),
                "status": row["status"],
                "priority": row["priority"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "completed_at": row["completed_at"],
                "deadline": row["deadline"],
                "tags": json.loads(row["tags"]),
                "metadata": json.loads(row["metadata"]),
            }
        )


# ── Step Executor ─────────────────────────────────────────────────────────────


class _StepExecutor:
    """
    Executes individual goal steps by routing to the appropriate handler.
    Handlers are registered at runtime — engine injects action_router and LLM.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[GoalStep], str]] = {}

    def register(self, prefix: str, handler: Callable[[GoalStep], str]) -> None:
        """
        Register a step handler for a given action prefix.

        Args:
            prefix:  Action prefix (e.g. "llm:", "action:", "wait:").
            handler: Callable that takes a GoalStep and returns a result string.
        """
        self._handlers[prefix.lower()] = handler
        logger.debug("[GoalEngine] Handler registered for prefix '%s'.", prefix)

    def execute(self, step: GoalStep) -> tuple[bool, str]:
        """
        Execute a step by dispatching to the matching handler.

        Args:
            step: The GoalStep to execute.

        Returns:
            (success, result_or_error_message) tuple.
        """
        action = step.action.strip()

        # Find matching handler by prefix
        for prefix, handler in self._handlers.items():
            if action.lower().startswith(prefix):
                try:
                    result = handler(step)
                    return True, result or "Done."
                except Exception as exc:
                    logger.error(
                        "[GoalEngine] Step '%s' handler raised: %s",
                        step.description,
                        exc,
                    )
                    return False, str(exc)

        # No handler matched — treat action as a plain description
        logger.warning(
            "[GoalEngine] No handler for action prefix in '%s'. Marking skipped.",
            action,
        )
        return False, f"No handler registered for action: {action}"


# ── Goal Engine ───────────────────────────────────────────────────────────────


class GoalEngine:
    """
    Public interface for Friday's goal management system.

    Creates, persists, and executes multi-step goals. Step execution runs
    via the TaskQueue so it never blocks the voice loop.

    Callbacks surface progress to the engine for TTS announcements.

    Usage:
        engine = GoalEngine(task_queue=tq)
        engine.setup()
        engine.register_step_handler("action:", action_router_handler)
        engine.register_step_handler("llm:", llm_handler)

        goal_id = engine.create_goal(
            title="Set up project",
            description="Create folders and open IDE",
            steps=[
                {"description": "Create project folder", "action": "action: open explorer"},
                {"description": "Open VS Code",           "action": "action: open vscode"},
            ],
        )
        engine.start_goal(goal_id)
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        on_step_complete: Callable[[Goal, GoalStep], None] | None = None,
        on_goal_complete: Callable[[Goal], None] | None = None,
        on_goal_failed: Callable[[Goal, str], None] | None = None,
    ) -> None:
        self._repo = _GoalRepository(config.storage.goals_db)
        self._executor = _StepExecutor()
        self._tq = task_queue
        self._ready = False
        self._lock = threading.RLock()

        # Active goal execution state (goal_id → task_id)
        self._active_tasks: dict[str, str] = {}

        # Lifecycle callbacks
        self._on_step_complete = on_step_complete
        self._on_goal_complete = on_goal_complete
        self._on_goal_failed = on_goal_failed

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise the goal engine and resume any interrupted active goals.

        Returns:
            True on success.
        """
        try:
            interrupted = self._repo.load_all(status=GoalStatus.ACTIVE)
            for goal in interrupted:
                logger.info("[GoalEngine] Resuming interrupted goal '%s'.", goal.title)
                goal.status = GoalStatus.PENDING
                self._repo.save(goal)

            self._ready = True
            logger.info(
                "[GoalEngine] Ready. %d total goals (%d active).",
                self._repo.count(),
                self._repo.count(GoalStatus.ACTIVE),
            )
            return True
        except Exception as exc:
            logger.error("[GoalEngine] Setup failed: %s", exc)
            return False

    def teardown(self) -> None:
        """Pause all active goals gracefully on shutdown."""
        with self._lock:
            active = list(self._active_tasks.keys())

        for goal_id in active:
            self.pause_goal(goal_id)

        self._ready = False
        logger.info("[GoalEngine] Teardown complete.")

    def register_step_handler(
        self,
        prefix: str,
        handler: Callable[[GoalStep], str],
    ) -> None:
        """
        Register a handler for a step action prefix.

        Args:
            prefix:  String prefix the step's action must start with.
            handler: Callable(step) → result_string.
        """
        self._executor.register(prefix, handler)

    # ── Goal Creation ─────────────────────────────────────────────────────────

    def create_goal(
        self,
        title: str,
        description: str = "",
        steps: list[dict[str, Any]] | None = None,
        priority: GoalPriority = GoalPriority.NORMAL,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create and persist a new goal.

        Args:
            title:       Short goal title.
            description: Full goal description.
            steps:       List of step dicts with 'description' and 'action' keys.
                         Additional optional keys: 'max_retries', 'metadata'.
            priority:    Scheduling priority.
            tags:        Classification tags.
            metadata:    Arbitrary context.

        Returns:
            The new goal's ID string.
        """
        goal_id = str(uuid.uuid4())[:12]

        goal_steps = [
            GoalStep(
                id=str(uuid.uuid4())[:8],
                description=s.get("description", f"Step {i + 1}"),
                action=s.get("action", ""),
                max_retries=s.get("max_retries", 2),
                metadata=s.get("metadata", {}),
            )
            for i, s in enumerate(steps or [])
        ]

        goal = Goal(
            id=goal_id,
            title=title,
            description=description,
            steps=goal_steps,
            priority=priority,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._repo.save(goal)
        logger.info(
            "[GoalEngine] Goal created: '%s' (%d steps) [id=%s].",
            title,
            len(goal_steps),
            goal_id,
        )
        return goal_id

    def add_step(
        self,
        goal_id: str,
        description: str,
        action: str,
        max_retries: int = 2,
        position: int | None = None,
    ) -> bool:
        """
        Append or insert a step into an existing goal.

        Args:
            goal_id:     Target goal ID.
            description: Step description.
            action:      Action string.
            max_retries: Retry limit.
            position:    Insert index (None = append).

        Returns:
            True if the step was added successfully.
        """
        goal = self._repo.load(goal_id)
        if not goal or goal.is_terminal:
            return False

        step = GoalStep(
            id=str(uuid.uuid4())[:8],
            description=description,
            action=action,
            max_retries=max_retries,
        )

        if position is not None:
            goal.steps.insert(position, step)
        else:
            goal.steps.append(step)

        self._repo.save(goal)
        return True

    # ── Goal Execution ────────────────────────────────────────────────────────

    def start_goal(self, goal_id: str) -> bool:
        """
        Begin executing a PENDING goal in the background.

        Args:
            goal_id: The goal to start.

        Returns:
            True if execution was initiated.
        """
        goal = self._repo.load(goal_id)
        if not goal:
            logger.warning("[GoalEngine] start_goal: ID '%s' not found.", goal_id)
            return False

        if goal.status not in (GoalStatus.PENDING, GoalStatus.PAUSED):
            logger.warning("[GoalEngine] Cannot start goal in status '%s'.", goal.status.name)
            return False

        if not goal.steps:
            logger.warning("[GoalEngine] Goal '%s' has no steps.", goal.title)
            return False

        goal.status = GoalStatus.ACTIVE
        self._repo.save(goal)

        task_id = self._tq.submit(
            fn=self._execute_goal,
            args=(goal_id,),
            priority=self._map_priority(goal.priority),
            name=f"goal:{goal.title[:30]}",
            on_error=lambda exc: self._handle_goal_error(goal_id, str(exc)),
        )

        with self._lock:
            self._active_tasks[goal_id] = task_id

        logger.info("[GoalEngine] Goal '%s' started (task=%s).", goal.title, task_id)
        return True

    def pause_goal(self, goal_id: str) -> bool:
        """
        Pause an active goal. Current step will complete before pausing.

        Returns:
            True if the goal was paused.
        """
        goal = self._repo.load(goal_id)
        if not goal or goal.status != GoalStatus.ACTIVE:
            return False

        goal.status = GoalStatus.PAUSED
        self._repo.save(goal)

        with self._lock:
            task_id = self._active_tasks.pop(goal_id, None)

        if task_id:
            self._tq.cancel(task_id)

        logger.info("[GoalEngine] Goal '%s' paused.", goal.title)
        return True

    def resume_goal(self, goal_id: str) -> bool:
        """
        Resume a PAUSED goal from its current step.

        Returns:
            True if resumed successfully.
        """
        goal = self._repo.load(goal_id)
        if not goal or goal.status != GoalStatus.PAUSED:
            return False

        goal.status = GoalStatus.PENDING
        self._repo.save(goal)
        return self.start_goal(goal_id)

    def cancel_goal(self, goal_id: str) -> bool:
        """
        Cancel a goal permanently.

        Returns:
            True if cancelled.
        """
        goal = self._repo.load(goal_id)
        if not goal or goal.is_terminal:
            return False

        with self._lock:
            task_id = self._active_tasks.pop(goal_id, None)

        if task_id:
            self._tq.cancel(task_id)

        goal.status = GoalStatus.CANCELLED
        self._repo.save(goal)

        logger.info("[GoalEngine] Goal '%s' cancelled.", goal.title)
        return True

    def detect_goal(self, text: str) -> bool:
        """Compatibility wrapper: check if the text represents a goal or multi-step task."""
        text = text.lower()
        keywords = [
            "plan",
            "goal",
            "help me",
            "need to",
            "while you",
            "also do",
            "job",
            "internship",
        ]
        if (text.startswith("open ") or text.startswith("launch ") or text.startswith("run ")) and (
            "while" not in text and "also" not in text
        ):
            return False
        return any(kw in text for kw in keywords)

    # ── Goal Queries ──────────────────────────────────────────────────────────

    def get_goal(self, goal_id: str) -> Goal | None:
        """Fetch a goal by ID."""
        return self._repo.load(goal_id)

    def list_goals(
        self,
        status: GoalStatus | None = None,
        priority: GoalPriority | None = None,
        limit: int = 20,
    ) -> list[Goal]:
        """
        List goals with optional filtering.

        Args:
            status:   Filter by lifecycle status.
            priority: Filter by priority.
            limit:    Maximum results.

        Returns:
            List of Goal objects ordered by last update descending.
        """
        return self._repo.load_all(status=status, priority=priority, limit=limit)

    def get_active_goals(self) -> list[Goal]:
        """Return all currently active goals."""
        return self._repo.load_all(status=GoalStatus.ACTIVE)

    def get_progress(self, goal_id: str) -> str | None:
        """
        Return a natural-language progress summary for TTS.

        Args:
            goal_id: Target goal ID.

        Returns:
            Progress string or None if goal not found.
        """
        goal = self._repo.load(goal_id)
        return goal.progress_summary() if goal else None

    def delete_goal(self, goal_id: str) -> bool:
        """
        Permanently delete a goal.
        Active goals are cancelled before deletion.

        Returns:
            True if deleted.
        """
        self.cancel_goal(goal_id)
        return self._repo.delete(goal_id)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def active_goal_count(self) -> int:
        with self._lock:
            return len(self._active_tasks)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total": self._repo.count(),
            "pending": self._repo.count(GoalStatus.PENDING),
            "active": self._repo.count(GoalStatus.ACTIVE),
            "paused": self._repo.count(GoalStatus.PAUSED),
            "completed": self._repo.count(GoalStatus.COMPLETED),
            "failed": self._repo.count(GoalStatus.FAILED),
            "cancelled": self._repo.count(GoalStatus.CANCELLED),
        }

    # ── Internal Execution ────────────────────────────────────────────────────

    def _execute_goal(self, goal_id: str) -> None:
        """
        Core step-by-step execution loop. Runs on a task queue worker thread.
        Iterates through pending steps, retries on failure, and updates
        the goal's persisted state after each step.
        """
        while True:
            goal = self._repo.load(goal_id)
            if not goal:
                logger.error("[GoalEngine] Goal '%s' disappeared mid-execution.", goal_id)
                return

            if goal.status != GoalStatus.ACTIVE:
                logger.debug(
                    "[GoalEngine] Goal '%s' status changed to '%s' — stopping execution.",
                    goal.title,
                    goal.status.name,
                )
                return

            step = goal.current_step
            if not step:
                # All steps done
                self._complete_goal(goal)
                return

            logger.info(
                "[GoalEngine] Executing step '%s' (%d/%d) for goal '%s'.",
                step.description,
                goal.completed_steps + 1,
                goal.total_steps,
                goal.title,
            )

            success, result = self._run_step(goal, step)

            # Reload goal after step to pick up any external modifications
            goal = self._repo.load(goal_id)
            if not goal or goal.status != GoalStatus.ACTIVE:
                return

            # Update the step in the goal's step list
            for i, s in enumerate(goal.steps):
                if s.id == step.id:
                    goal.steps[i] = self._update_step(s, success, result)
                    break

            self._repo.save(goal)

            if success and self._on_step_complete:
                try:
                    self._on_step_complete(goal, goal.steps[i])
                except Exception:
                    logger.debug("[GoalEngine] on_step_complete callback raised.", exc_info=True)

            if not success:
                updated_step = goal.steps[i]
                if not updated_step.can_retry:
                    # Step exhausted retries — fail the goal
                    self._fail_goal(goal, updated_step.error or "Step failed.")
                    return
                # Retry — brief pause before next attempt
                time.sleep(2.0)

    def _run_step(self, goal: Goal, step: GoalStep) -> tuple[bool, str]:
        """Execute a single step, updating its started_at timestamp first."""
        # Mark as running
        for i, s in enumerate(goal.steps):
            if s.id == step.id:
                goal.steps[i] = GoalStep.from_dict(
                    {
                        **step.to_dict(),
                        "status": StepStatus.RUNNING.value,
                        "started_at": time.time(),
                    }
                )
                break

        self._repo.save(goal)
        return self._executor.execute(step)

    def _update_step(
        self,
        step: GoalStep,
        success: bool,
        result: str,
    ) -> GoalStep:
        """Return an updated copy of a step after execution."""
        if success:
            return GoalStep.from_dict(
                {
                    **step.to_dict(),
                    "status": StepStatus.COMPLETED.value,
                    "result": result,
                    "completed_at": time.time(),
                }
            )
        else:
            retries = step.retries + 1
            new_status = (
                StepStatus.RETRYING.value if retries < step.max_retries else StepStatus.FAILED.value
            )
            return GoalStep.from_dict(
                {
                    **step.to_dict(),
                    "status": new_status,
                    "error": result,
                    "retries": retries,
                }
            )

    def _complete_goal(self, goal: Goal) -> None:
        """Mark a goal as COMPLETED and fire the completion callback."""
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = time.time()
        self._repo.save(goal)

        with self._lock:
            self._active_tasks.pop(goal.id, None)

        logger.info("[GoalEngine] Goal '%s' COMPLETED.", goal.title)

        if self._on_goal_complete:
            try:
                self._on_goal_complete(goal)
            except Exception:
                logger.debug("[GoalEngine] on_goal_complete callback raised.", exc_info=True)

    def _fail_goal(self, goal: Goal, reason: str) -> None:
        """Mark a goal as FAILED and fire the failure callback."""
        goal.status = GoalStatus.FAILED
        self._repo.save(goal)

        with self._lock:
            self._active_tasks.pop(goal.id, None)

        logger.error("[GoalEngine] Goal '%s' FAILED: %s", goal.title, reason)

        if self._on_goal_failed:
            try:
                self._on_goal_failed(goal, reason)
            except Exception:
                logger.debug("[GoalEngine] on_goal_failed callback raised.", exc_info=True)

    def _handle_goal_error(self, goal_id: str, error: str) -> None:
        """Handle an unexpected task-queue-level error for a goal."""
        goal = self._repo.load(goal_id)
        if goal:
            self._fail_goal(goal, error)

    @staticmethod
    def _map_priority(priority: GoalPriority) -> TaskPriority:
        """Map GoalPriority to TaskQueue's TaskPriority."""
        return {
            GoalPriority.LOW: TaskPriority.BACKGROUND,
            GoalPriority.NORMAL: TaskPriority.LOW,
            GoalPriority.HIGH: TaskPriority.NORMAL,
        }.get(priority, TaskPriority.LOW)


def get_goal_engine(task_queue=None):
    global _instance
    with _lock:
        if _instance is None:
            if task_queue is None:
                from friday.task_queue import TaskQueue

                task_queue = TaskQueue()
                task_queue.start()
            _instance = GoalEngine(task_queue=task_queue)
        return _instance
