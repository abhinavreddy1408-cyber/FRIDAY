"""
F.R.I.D.A.Y. Learning & Personalisation Module
Observes user behaviour over time and builds a preference model that
makes Friday progressively more tailored to the individual user.

Learning tracks:
  - Command frequency and shortcuts
  - Preferred applications and workflows
  - Response style preferences (verbosity, tone)
  - Time-of-day usage patterns
  - Action success/failure rates for adaptive routing

All data is persisted to SQLite and never leaves the local machine.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

from friday.config import config

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────


class EventType(Enum):
    COMMAND_EXECUTED = auto()
    ACTION_SUCCESS = auto()
    ACTION_FAILURE = auto()
    APP_OPENED = auto()
    SEARCH_PERFORMED = auto()
    CORRECTION_MADE = auto()  # User rephrased or corrected Friday
    PREFERENCE_STATED = auto()  # Explicit "I prefer..." statement
    SHORTCUT_USED = auto()  # User used a learned shortcut


class PreferenceKey(Enum):
    """Typed keys for the preference store — prevents magic string bugs."""

    VERBOSITY = "verbosity"  # "concise" | "detailed"
    WAKE_SENSITIVITY = "wake_sensitivity"  # float
    DEFAULT_BROWSER = "default_browser"  # str
    DEFAULT_SEARCH = "default_search"  # "google" | "bing" | ...
    PREFERRED_APPS = "preferred_apps"  # list[str]
    RESPONSE_SPEED = "response_speed"  # "fast" | "thorough"
    GREETING_STYLE = "greeting_style"  # "formal" | "casual"
    CONFIRM_THRESHOLD = "confirm_threshold"  # float (0-1)


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CommandRecord:
    """
    Immutable record of a single command execution event.

    Attributes:
        command:   Normalised command text.
        action:    Resolved action name (empty if routed to LLM).
        success:   Whether the action completed without error.
        timestamp: Unix timestamp of execution.
        duration:  Execution time in seconds.
        metadata:  Optional context (app name, search query, etc.).
    """

    command: str
    action: str
    success: bool
    timestamp: float
    duration: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Shortcut:
    """
    A user-defined or learned command shortcut.

    Attributes:
        trigger:    The short phrase that activates this shortcut.
        expansion:  The full command it expands to.
        use_count:  How many times this shortcut has been triggered.
        created_at: Unix timestamp of creation.
    """

    trigger: str
    expansion: str
    use_count: int
    created_at: float


@dataclass
class UsagePattern:
    """
    Aggregated statistics for a specific command or action.

    Attributes:
        command:       The normalised command string.
        total_calls:   Total number of invocations.
        success_rate:  Fraction of calls that succeeded (0-1).
        avg_duration:  Mean execution duration in seconds.
        last_used:     Unix timestamp of most recent use.
        peak_hour:     Hour of day (0-23) most commonly used.
    """

    command: str
    total_calls: int
    success_rate: float
    avg_duration: float
    last_used: float
    peak_hour: int

    @property
    def is_frequent(self) -> bool:
        return self.total_calls >= 5

    @property
    def is_reliable(self) -> bool:
        return self.success_rate >= 0.85

    @property
    def last_used_ago(self) -> float:
        """Seconds since last use."""
        return time.time() - self.last_used


# ── Database Schema ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS command_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    command     TEXT    NOT NULL,
    action      TEXT    NOT NULL DEFAULT '',
    success     INTEGER NOT NULL DEFAULT 1,
    timestamp   REAL    NOT NULL,
    duration    REAL    NOT NULL DEFAULT 0.0,
    metadata    TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT    PRIMARY KEY,
    value       TEXT    NOT NULL,
    updated_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS shortcuts (
    trigger     TEXT    PRIMARY KEY,
    expansion   TEXT    NOT NULL,
    use_count   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,
    payload     TEXT    NOT NULL DEFAULT '{}',
    timestamp   REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_command   ON command_history (command);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON command_history (timestamp);
CREATE INDEX IF NOT EXISTS idx_history_action    ON command_history (action);
CREATE INDEX IF NOT EXISTS idx_events_type       ON events (event_type);
"""


# ── Learning Database ─────────────────────────────────────────────────────────


class _LearningDB:
    """
    Low-level SQLite access layer for the learning module.
    Thread-safe via per-thread connection caching and a write lock.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn

        try:
            yield self._local.conn
        except sqlite3.DatabaseError as exc:
            self._local.conn.rollback()
            logger.error("[Learning/DB] Database error: %s", exc)
            raise

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # ── Command History ───────────────────────────────────────────────────────

    def insert_command(self, record: CommandRecord) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO command_history
                    (command, action, success, timestamp, duration, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.command,
                    record.action,
                    int(record.success),
                    record.timestamp,
                    record.duration,
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()

    def fetch_command_stats(
        self,
        command: str,
    ) -> dict[str, Any] | None:
        """Return aggregate stats for a specific command string."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)          AS total,
                    AVG(success)      AS success_rate,
                    AVG(duration)     AS avg_duration,
                    MAX(timestamp)    AS last_used,
                    strftime('%H', datetime(timestamp, 'unixepoch')) AS hour
                FROM command_history
                WHERE command = ?
                """,
                (command,),
            ).fetchone()

            if not row or row["total"] == 0:
                return None

            return dict(row)

    def fetch_top_commands(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most frequently used commands."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    command,
                    COUNT(*)       AS total,
                    AVG(success)   AS success_rate,
                    AVG(duration)  AS avg_duration,
                    MAX(timestamp) AS last_used
                FROM command_history
                GROUP BY command
                ORDER BY total DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def fetch_recent_commands(self, hours: int = 24, limit: int = 50) -> list[dict[str, Any]]:
        """Return commands executed within the last N hours."""
        cutoff = time.time() - (hours * 3600)
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT command, action, success, timestamp, duration
                FROM command_history
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Preferences ───────────────────────────────────────────────────────────

    def get_preference(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_preference(self, key: str, value: str) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                               updated_at = excluded.updated_at
                """,
                (key, value, time.time()),
            )
            conn.commit()

    def get_all_preferences(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM preferences").fetchall()
            return {r["key"]: r["value"] for r in rows}

    # ── Shortcuts ─────────────────────────────────────────────────────────────

    def upsert_shortcut(self, trigger: str, expansion: str) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO shortcuts (trigger, expansion, use_count, created_at)
                VALUES (?, ?, 0, ?)
                ON CONFLICT(trigger) DO UPDATE SET expansion = excluded.expansion
                """,
                (trigger.lower(), expansion, time.time()),
            )
            conn.commit()

    def increment_shortcut(self, trigger: str) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "UPDATE shortcuts SET use_count = use_count + 1 WHERE trigger = ?",
                (trigger.lower(),),
            )
            conn.commit()

    def get_shortcut(self, trigger: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT expansion FROM shortcuts WHERE trigger = ?",
                (trigger.lower(),),
            ).fetchone()
            return row["expansion"] if row else None

    def get_all_shortcuts(self) -> list[Shortcut]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT trigger, expansion, use_count, created_at FROM shortcuts"
                " ORDER BY use_count DESC"
            ).fetchall()
            return [
                Shortcut(
                    trigger=r["trigger"],
                    expansion=r["expansion"],
                    use_count=r["use_count"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def delete_shortcut(self, trigger: str) -> bool:
        with self._write_lock, self._conn() as conn:
            cursor = conn.execute("DELETE FROM shortcuts WHERE trigger = ?", (trigger.lower(),))
            conn.commit()
            return cursor.rowcount > 0

    # ── Events ────────────────────────────────────────────────────────────────

    def log_event(self, event_type: EventType, payload: dict[str, Any]) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO events (event_type, payload, timestamp) VALUES (?, ?, ?)",
                (event_type.name, json.dumps(payload), time.time()),
            )
            conn.commit()

    def fetch_events(
        self,
        event_type: EventType | None = None,
        hours: int = 24,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        cutoff = time.time() - (hours * 3600)
        with self._conn() as conn:
            if event_type:
                rows = conn.execute(
                    """
                    SELECT event_type, payload, timestamp FROM events
                    WHERE event_type = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (event_type.name, cutoff, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT event_type, payload, timestamp FROM events
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (cutoff, limit),
                ).fetchall()

            return [dict(r) for r in rows]


# ── Pattern Analyser ──────────────────────────────────────────────────────────


class _PatternAnalyser:
    """
    Analyses command history to surface patterns and shortcut candidates.
    Runs in-memory on data fetched from the database.
    """

    # Minimum uses before a command is considered for auto-shortcut
    SHORTCUT_THRESHOLD = 5

    def find_shortcut_candidates(
        self,
        top_commands: list[dict[str, Any]],
        existing_shortcuts: list[Shortcut],
    ) -> list[tuple[str, str]]:
        """
        Identify frequent commands that don't yet have a shortcut.

        Args:
            top_commands:       Aggregated command frequency data.
            existing_shortcuts: Currently registered shortcuts.

        Returns:
            List of (full_command, suggested_trigger) tuples.
        """
        existing_triggers = {s.expansion.lower() for s in existing_shortcuts}
        candidates: list[tuple[str, str]] = []

        for row in top_commands:
            command = row.get("command", "")
            total = row.get("total", 0)

            if total < self.SHORTCUT_THRESHOLD:
                continue
            if command.lower() in existing_triggers:
                continue
            if len(command.split()) <= 2:
                continue  # Already short enough

            trigger = self._generate_trigger(command)
            if trigger:
                candidates.append((command, trigger))

        return candidates

    def detect_usage_peaks(
        self,
        recent: list[dict[str, Any]],
    ) -> dict[int, int]:
        """
        Count command frequency per hour of day.

        Args:
            recent: List of recent command records.

        Returns:
            Dict mapping hour (0-23) to command count.
        """
        import datetime

        hourly: dict[int, int] = defaultdict(int)
        for row in recent:
            ts = row.get("timestamp", 0.0)
            hour = datetime.datetime.fromtimestamp(ts).hour
            hourly[hour] += 1
        return dict(hourly)

    def compute_action_reliability(
        self,
        top_commands: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Return a success-rate map per command for adaptive routing.

        Args:
            top_commands: Aggregated command frequency data.

        Returns:
            Dict mapping command string to success rate (0-1).
        """
        return {
            row["command"]: float(row.get("success_rate", 1.0))
            for row in top_commands
            if row.get("total", 0) >= 3
        }

    @staticmethod
    def _generate_trigger(command: str) -> str | None:
        """
        Generate a short trigger phrase from a long command.
        Returns None if a sensible trigger cannot be derived.
        """
        words = command.lower().split()
        if not words:
            return None

        # Drop common filler words from the beginning
        fillers = {"open", "launch", "start", "run", "please", "can", "you"}
        significant = [w for w in words if w not in fillers]

        if not significant:
            return None

        # Use first 2 significant words as trigger
        trigger = " ".join(significant[:2])
        return trigger if len(trigger) >= 3 else None


# ── Learning Engine ───────────────────────────────────────────────────────────


class LearningEngine:
    """
    Public interface for Friday's personalisation and learning system.

    Observes every command execution, persists patterns, manages shortcuts,
    and exposes a preference API for other modules.

    Usage:
        engine = LearningEngine()
        engine.setup()

        engine.record_command("open chrome", action="open_app", success=True)
        expansion = engine.resolve_shortcut("chrome")
        verbosity = engine.get_preference(PreferenceKey.VERBOSITY, "concise")
    """

    def __init__(self) -> None:
        self._db = _LearningDB(config.storage.learning_db)
        self._analyser = _PatternAnalyser()
        self._cache: dict[str, Any] = {}  # In-memory preference cache
        self._ready = False

        # In-memory counters for current session (not persisted)
        self._session_commands: int = 0
        self._session_start: float = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise the learning engine and warm the preference cache.

        Returns:
            True — always succeeds (database errors are non-fatal).
        """
        try:
            self._cache = self._db.get_all_preferences()
            self._ready = True
            shortcuts = len(self._db.get_all_shortcuts())
            logger.info(
                "[Learning] Ready. %d preferences | %d shortcuts loaded.",
                len(self._cache),
                shortcuts,
            )
            return True
        except Exception as exc:
            logger.error("[Learning] Setup failed: %s", exc)
            self._ready = True  # Degrade gracefully
            return True

    def teardown(self) -> None:
        """Flush any pending state and mark as offline."""
        self._ready = False
        logger.debug("[Learning] Teardown complete.")

    # ── Command Recording ─────────────────────────────────────────────────────

    def record_command(
        self,
        command: str,
        action: str = "",
        success: bool = True,
        duration: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a command execution event for pattern analysis.

        Args:
            command:  Normalised transcript of the user's command.
            action:   The action name that handled it (empty if LLM).
            success:  Whether the action completed successfully.
            duration: Execution wall-clock time in seconds.
            metadata: Optional context (app name, query, etc.).
        """
        if not command.strip():
            return

        record = CommandRecord(
            command=command.strip().lower(),
            action=action,
            success=success,
            timestamp=time.time(),
            duration=duration,
            metadata=metadata or {},
        )

        self._session_commands += 1

        try:
            self._db.insert_command(record)
            self._db.log_event(
                EventType.COMMAND_EXECUTED,
                {
                    "command": record.command,
                    "action": record.action,
                    "success": record.success,
                },
            )
        except Exception as exc:
            logger.warning("[Learning] Failed to record command: %s", exc)

    def record_correction(self, original: str, corrected: str) -> None:
        """
        Record when the user rephrases or corrects a misunderstood command.
        Used to improve pattern matching over time.

        Args:
            original:  What Friday misunderstood.
            corrected: What the user actually meant.
        """
        try:
            self._db.log_event(
                EventType.CORRECTION_MADE,
                {"original": original, "corrected": corrected},
            )
            logger.debug("[Learning] Correction recorded: '%s' → '%s'", original, corrected)
        except Exception as exc:
            logger.warning("[Learning] Failed to record correction: %s", exc)

    # ── Shortcut Management ───────────────────────────────────────────────────

    def register_shortcut(self, trigger: str, expansion: str) -> bool:
        """
        Register a command shortcut (user-defined or auto-suggested).

        Args:
            trigger:   Short phrase to recognise.
            expansion: Full command it maps to.

        Returns:
            True if registered successfully.
        """
        if not trigger.strip() or not expansion.strip():
            return False

        try:
            self._db.upsert_shortcut(trigger.strip(), expansion.strip())
            self._db.log_event(
                EventType.SHORTCUT_USED,
                {"trigger": trigger, "expansion": expansion, "action": "registered"},
            )
            logger.info("[Learning] Shortcut registered: '%s' → '%s'", trigger, expansion)
            return True
        except Exception as exc:
            logger.error("[Learning] Shortcut registration failed: %s", exc)
            return False

    def resolve_shortcut(self, text: str) -> str | None:
        """
        Check whether the input text matches a known shortcut trigger.

        Args:
            text: The user's utterance (will be lowercased for matching).

        Returns:
            The expansion string if matched, None otherwise.
        """
        text = text.strip().lower()
        expansion = self._db.get_shortcut(text)

        if expansion:
            self._db.increment_shortcut(text)
            logger.debug("[Learning] Shortcut resolved: '%s' → '%s'", text, expansion)

        return expansion

    def delete_shortcut(self, trigger: str) -> bool:
        """
        Remove a shortcut by trigger phrase.

        Returns:
            True if found and deleted.
        """
        return self._db.delete_shortcut(trigger)

    def list_shortcuts(self) -> list[Shortcut]:
        """Return all registered shortcuts ordered by use count."""
        return self._db.get_all_shortcuts()

    def suggest_shortcuts(self) -> list[tuple[str, str]]:
        """
        Analyse command history and return shortcut suggestions.

        Returns:
            List of (full_command, suggested_trigger) pairs.
        """
        try:
            top = self._db.fetch_top_commands(limit=50)
            existing = self._db.get_all_shortcuts()
            return self._analyser.find_shortcut_candidates(top, existing)
        except Exception as exc:
            logger.warning("[Learning] Shortcut suggestion failed: %s", exc)
            return []

    # ── Preference Management ─────────────────────────────────────────────────

    def get_preference(
        self,
        key: PreferenceKey,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a stored user preference.

        Reads from the in-memory cache first; falls back to the database.

        Args:
            key:     The preference key.
            default: Value to return if the preference is not set.

        Returns:
            The stored value (deserialised from JSON) or the default.
        """
        raw = self._cache.get(key.value)
        if raw is None:
            raw = self._db.get_preference(key.value)
            if raw is not None:
                self._cache[key.value] = raw

        if raw is None:
            return default

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set_preference(self, key: PreferenceKey, value: Any) -> None:
        """
        Persist a user preference.

        Args:
            key:   The preference key.
            value: The value to store (will be JSON-serialised).
        """
        serialised = json.dumps(value)
        self._cache[key.value] = serialised

        try:
            self._db.set_preference(key.value, serialised)
            self._db.log_event(
                EventType.PREFERENCE_STATED,
                {"key": key.value, "value": str(value)},
            )
            logger.debug("[Learning] Preference set: %s = %s", key.name, value)
        except Exception as exc:
            logger.error("[Learning] Failed to persist preference: %s", exc)

    def get_all_preferences(self) -> dict[str, Any]:
        """Return all stored preferences as a deserialised dict."""
        raw = self._db.get_all_preferences()
        result: dict[str, Any] = {}
        for k, v in raw.items():
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = v
        return result

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_usage_pattern(self, command: str) -> UsagePattern | None:
        """
        Return aggregated usage statistics for a specific command.

        Args:
            command: The normalised command string.

        Returns:
            UsagePattern if the command has been used, None otherwise.
        """
        try:
            stats = self._db.fetch_command_stats(command.lower())
            if not stats:
                return None

            return UsagePattern(
                command=command,
                total_calls=int(stats.get("total", 0)),
                success_rate=float(stats.get("success_rate", 1.0)),
                avg_duration=float(stats.get("avg_duration", 0.0)),
                last_used=float(stats.get("last_used", 0.0)),
                peak_hour=int(stats.get("hour") or 0),
            )
        except Exception as exc:
            logger.warning("[Learning] Usage pattern fetch failed: %s", exc)
            return None

    def get_top_commands(self, limit: int = 10) -> list[UsagePattern]:
        """
        Return the most frequently used commands as UsagePattern objects.

        Args:
            limit: Maximum number of patterns to return.

        Returns:
            List of UsagePattern ordered by total calls descending.
        """
        try:
            rows = self._db.fetch_top_commands(limit=limit)
            return [
                UsagePattern(
                    command=r["command"],
                    total_calls=int(r.get("total", 0)),
                    success_rate=float(r.get("success_rate", 1.0)),
                    avg_duration=float(r.get("avg_duration", 0.0)),
                    last_used=float(r.get("last_used", 0.0)),
                    peak_hour=0,
                )
                for r in rows
            ]
        except Exception as exc:
            logger.warning("[Learning] Top commands fetch failed: %s", exc)
            return []

    def get_action_reliability(self) -> dict[str, float]:
        """
        Return a command → success-rate map for adaptive routing decisions.

        The engine can use this to decide whether to retry a low-reliability
        action with LLM assistance.

        Returns:
            Dict of command string to success rate (0.0-1.0).
        """
        try:
            top = self._db.fetch_top_commands(limit=100)
            return self._analyser.compute_action_reliability(top)
        except Exception as exc:
            logger.warning("[Learning] Reliability fetch failed: %s", exc)
            return {}

    def get_hourly_usage(self, hours: int = 168) -> dict[int, int]:
        """
        Return command frequency per hour of day over the last N hours.

        Args:
            hours: Lookback window in hours (default: 7 days).

        Returns:
            Dict mapping hour-of-day (0-23) to command count.
        """
        try:
            recent = self._db.fetch_recent_commands(hours=hours, limit=1000)
            return self._analyser.detect_usage_peaks(recent)
        except Exception as exc:
            logger.warning("[Learning] Hourly usage fetch failed: %s", exc)
            return {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def session_stats(self) -> dict[str, Any]:
        """Return current session statistics."""
        elapsed = time.time() - self._session_start
        return {
            "session_commands": self._session_commands,
            "session_duration": round(elapsed, 1),
            "commands_per_hour": round(self._session_commands / max(elapsed / 3600, 0.01), 1),
        }

    # ── Compatibility wrappers for testing ────────────────────────────────────

    def record_failure(self, action: str, details: dict[str, Any], error: str) -> None:
        """Compatibility wrapper: record a command/action failure."""
        if not hasattr(self, "_failures"):
            self._failures = defaultdict(int)
        key = (action, json.dumps(details, sort_keys=True))
        self._failures[key] += 1

    def get_failure_count(self, action: str, details: dict[str, Any]) -> int:
        """Compatibility wrapper: get count of failures for a specific action/details."""
        if not hasattr(self, "_failures"):
            self._failures = defaultdict(int)
        key = (action, json.dumps(details, sort_keys=True))
        return self._failures[key]

    def is_reliable(self, action: str, details: dict[str, Any]) -> bool:
        """Compatibility wrapper: return True if failure count is 0."""
        return self.get_failure_count(action, details) == 0

    def create_shortcut(self, trigger: str, expansion: str) -> bool:
        """Compatibility wrapper: register a shortcut."""
        return self.register_shortcut(trigger, expansion)

    def save_shortcut(self, trigger: str, expansion: str) -> bool:
        """Compatibility wrapper: register a shortcut."""
        return self.register_shortcut(trigger, expansion)

    def get_shortcut(self, trigger: str) -> str | None:
        """Fuzzy match the trigger against registered shortcuts."""
        import friday.config as cfg_mod

        threshold = getattr(
            cfg_mod, "SHORTCUT_THRESHOLD", getattr(config, "SHORTCUT_THRESHOLD", 0.85)
        )

        def levenshtein_dist(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_dist(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        # Get all shortcuts from DB
        try:
            shortcuts = self.list_shortcuts()
        except Exception:
            shortcuts = []

        best_match = None
        best_ratio = 0.0

        trigger_clean = trigger.strip().lower()
        trigger_no_punc = trigger_clean.rstrip("!?. ,")

        for sc in shortcuts:
            sc_trigger = getattr(sc, "trigger", "").strip().lower()
            if not sc_trigger:
                continue

            # calculate ratio using Levenshtein similarity relative to stored trigger length
            dist = levenshtein_dist(trigger_no_punc, sc_trigger)
            ratio = 1.0 - dist / len(sc_trigger) if len(sc_trigger) > 0 else 0.0

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = sc

        if best_ratio >= threshold and best_match is not None:
            return getattr(best_match, "expansion", getattr(best_match, "action", None))

        # Try direct lookup in case list failed or has no items
        try:
            return self._db.get_shortcut(trigger)
        except Exception:
            return None

    def track_command(self, command: str) -> None:
        """Non-blocking call to record_command."""
        threading.Thread(target=self.record_command, args=(command,), daemon=True).start()
