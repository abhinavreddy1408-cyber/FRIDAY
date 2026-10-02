"""
F.R.I.D.A.Y. Structured Memory Store (Phase 5: True Agent Intelligence)
=======================================================================
Handles persistent storage of structured data: GoalSessions, Shortcuts,
FailurePatterns, SuccessPatterns, Selectors, and Analytics.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from friday.config import config

logger = logging.getLogger("MemoryStore")

# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class StoredGoalSession:
    """Persistent storage format for goal sessions."""

    id: str
    goal: str
    strategy: Any  # Stored as JSON string in DB
    current_step: int
    completed_steps: Any  # Stored as JSON string in DB
    pending_steps: Any  # Stored as JSON string in DB
    context: Any  # Stored as JSON string in DB
    status: str
    created_at: float
    last_active: float
    total_steps: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> StoredGoalSession:
        """Convert a SQLite row into a StoredGoalSession with JSON decoding."""
        return cls(
            id=row["id"],
            goal=row["goal"],
            strategy=json.loads(row["strategy"]),
            current_step=row["current_step"],
            completed_steps=json.loads(row["completed_steps"]),
            pending_steps=json.loads(row["pending_steps"]),
            context=json.loads(row["context"]),
            status=row["status"],
            created_at=row["created_at"],
            last_active=row["last_active"],
            total_steps=row["total_steps"],
        )

    def to_storage_tuple(self) -> tuple:
        """Convert object to a tuple suitable for SQLite INSERT."""
        return (
            self.id,
            self.goal,
            json.dumps(self.strategy),
            self.current_step,
            json.dumps(self.completed_steps),
            json.dumps(self.pending_steps),
            json.dumps(self.context),
            self.status,
            self.created_at,
            self.last_active,
            self.total_steps,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "goal": self.goal,
            "strategy": self.strategy,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "pending_steps": self.pending_steps,
            "context": self.context,
            "status": self.status,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "total_steps": self.total_steps,
        }


@dataclass
class StoredShortcut:
    """Persistent storage format for shortcuts."""

    trigger: str
    action: str
    created_at: float
    usage_count: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> StoredShortcut:
        """Convert a SQLite row into a StoredShortcut."""
        return cls(
            trigger=row["trigger"],
            action=row["action"],
            created_at=row["created_at"],
            usage_count=row["usage_count"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return asdict(self)


# ── Main Memory Store ─────────────────────────────────────────────────────────


class MemoryStore:
    """
    Thread-safe SQLite storage for F.R.I.D.A.Y.'s persistent structured data.
    Uses WAL mode for concurrent read/write operations.
    """

    def __init__(self):
        self._logger = logging.getLogger("MemoryStore")
        self._local = threading.local()
        self._init_databases()
        self._logger.info("[MEMORY_STORE] Structured memory store initialized with WAL mode.")

    def _get_connection(self, db_path: str) -> sqlite3.Connection:
        """
        Returns a thread-local connection to the specified database.
        Prevents 'database is locked' errors in multi-threaded environment.
        """
        conn_key = f"conn_{db_path}"
        if not hasattr(self._local, conn_key) or getattr(self._local, conn_key) is None:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # WAL mode allows concurrent reads and writes
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            setattr(self._local, conn_key, conn)
        return getattr(self._local, conn_key)

    def _init_databases(self):
        """Initialize schemas for all structured stores."""
        # Ensure data directory exists
        config.storage.data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Goal Sessions Schema
        with self._get_connection(config.storage.goals_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goal_sessions (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    current_step INTEGER DEFAULT 0,
                    completed_steps TEXT,
                    pending_steps TEXT,
                    context TEXT,
                    status TEXT,
                    created_at REAL,
                    last_active REAL,
                    total_steps INTEGER
                )
            """)
            conn.commit()

        # 2. Shortcuts Schema (Using Assistant DB for general agent settings)
        with self._get_connection(config.storage.assistant_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shortcuts (
                    trigger TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

        self._logger.info("[MEMORY_STORE] All database schemas initialized.")

    # ── Goal Session Management ──────────────────────────────────────────────

    def save_goal_session(self, session: StoredGoalSession) -> bool:
        """
        Persist a goal session. Handles JSON conversion automatically.
        Returns True on success, False on failure.
        """
        try:
            with self._get_connection(config.storage.goals_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO goal_sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    session.to_storage_tuple(),
                )
                conn.commit()
                self._logger.debug(f"[MEMORY_STORE] Saved goal session: {session.id}")
                return True
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to save goal session {session.id}: {e}")
            return False

    def load_goal_session(self, session_id: str) -> StoredGoalSession | None:
        """Retrieve a goal session by ID."""
        try:
            with self._get_connection(config.storage.goals_db) as conn:
                row = conn.execute(
                    "SELECT * FROM goal_sessions WHERE id = ?", (session_id,)
                ).fetchone()
                return StoredGoalSession.from_row(row) if row else None
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to load goal session {session_id}: {e}")
            return None

    def get_active_sessions(self) -> list[StoredGoalSession]:
        """Retrieve all currently active goals."""
        try:
            with self._get_connection(config.storage.goals_db) as conn:
                rows = conn.execute(
                    "SELECT * FROM goal_sessions WHERE status = 'active' ORDER BY last_active DESC"
                ).fetchall()
                return [StoredGoalSession.from_row(r) for r in rows]
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to get active sessions: {e}")
            return []

    def get_session_by_goal(self, goal_query: str) -> StoredGoalSession | None:
        """Search for a session using a partial goal match."""
        try:
            with self._get_connection(config.storage.goals_db) as conn:
                row = conn.execute(
                    "SELECT * FROM goal_sessions WHERE goal LIKE ?", (f"%{goal_query}%",)
                ).fetchone()
                return StoredGoalSession.from_row(row) if row else None
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to search goal sessions: {e}")
            return None

    def delete_goal_session(self, session_id: str) -> bool:
        """Remove a goal session from storage."""
        try:
            with self._get_connection(config.storage.goals_db) as conn:
                conn.execute("DELETE FROM goal_sessions WHERE id = ?", (session_id,))
                conn.commit()
                self._logger.debug(f"[MEMORY_STORE] Deleted goal session: {session_id}")
                return True
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to delete goal session {session_id}: {e}")
            return False

    def update_goal_status(self, session_id: str, status: str) -> bool:
        """Update the status of a goal session."""
        try:
            with self._get_connection(config.storage.goals_db) as conn:
                conn.execute(
                    "UPDATE goal_sessions SET status = ?, last_active = ? WHERE id = ?",
                    (status, time.time(), session_id),
                )
                conn.commit()
                return True
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to update goal status: {e}")
            return False

    # ── Shortcut Management ──────────────────────────────────────────────────

    def save_shortcut(self, trigger: str, action: str) -> bool:
        """Create or update a shortcut."""
        try:
            now = time.time()
            with self._get_connection(config.storage.assistant_db) as conn:
                conn.execute(
                    """
                    INSERT INTO shortcuts (trigger, action, created_at, usage_count)
                    VALUES (?, ?, ?, 0)
                    ON CONFLICT(trigger) DO UPDATE SET action=excluded.action
                    """,
                    (trigger, action, now),
                )
                conn.commit()
                self._logger.debug(f"[MEMORY_STORE] Saved shortcut: {trigger}")
                return True
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to save shortcut {trigger}: {e}")
            return False

    def load_shortcut(self, trigger: str) -> StoredShortcut | None:
        """Retrieve a shortcut by its trigger phrase."""
        try:
            with self._get_connection(config.storage.assistant_db) as conn:
                row = conn.execute(
                    "SELECT * FROM shortcuts WHERE trigger = ?", (trigger,)
                ).fetchone()
                return StoredShortcut.from_row(row) if row else None
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to load shortcut {trigger}: {e}")
            return None

    def list_shortcuts(self) -> list[StoredShortcut]:
        """List all registered shortcuts, sorted by usage count."""
        try:
            with self._get_connection(config.storage.assistant_db) as conn:
                rows = conn.execute("SELECT * FROM shortcuts ORDER BY usage_count DESC").fetchall()
                return [StoredShortcut.from_row(r) for r in rows]
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to list shortcuts: {e}")
            return []

    def increment_shortcut_usage(self, trigger: str) -> bool:
        """Track how often a shortcut is used for future optimization."""
        try:
            with self._get_connection(config.storage.assistant_db) as conn:
                conn.execute(
                    "UPDATE shortcuts SET usage_count = usage_count + 1 WHERE trigger = ?",
                    (trigger,),
                )
                conn.commit()
                return True
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to increment shortcut usage: {e}")
            return False

    def delete_shortcut(self, trigger: str) -> bool:
        """Remove a shortcut from storage."""
        try:
            with self._get_connection(config.storage.assistant_db) as conn:
                conn.execute("DELETE FROM shortcuts WHERE trigger = ?", (trigger,))
                conn.commit()
                return True
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to delete shortcut {trigger}: {e}")
            return False

    # ── Analytics & Stats ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return storage statistics."""
        try:
            with self._get_connection(config.storage.goals_db) as conn:
                goal_count = conn.execute("SELECT COUNT(*) FROM goal_sessions").fetchone()[0]
                active_count = conn.execute(
                    "SELECT COUNT(*) FROM goal_sessions WHERE status = 'active'"
                ).fetchone()[0]

            with self._get_connection(config.storage.assistant_db) as conn:
                shortcut_count = conn.execute("SELECT COUNT(*) FROM shortcuts").fetchone()[0]

            return {
                "total_goals": goal_count,
                "active_goals": active_count,
                "total_shortcuts": shortcut_count,
            }
        except Exception as e:
            self._logger.error(f"[MEMORY_STORE] Failed to get stats: {e}")
            return {}


# ── Test Suite ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test MemoryStore functionality."""
    print("=" * 60)
    print("F.R.I.D.A.Y. MemoryStore Test Suite")
    print("=" * 60)

    # Temporary config override for testing
    TEST_DIR = Path("test_memory_store")
    TEST_DIR.mkdir(exist_ok=True)

    # Mock config paths for testing
    config.storage.goals_db = str(TEST_DIR / "goals.db")
    config.storage.assistant_db = str(TEST_DIR / "shortcuts.db")
    config.storage.data_dir = TEST_DIR

    store = MemoryStore()

    # ── Test 1: Goal Session with Complex Data ───────────────────────────────
    print("\n[TEST 1] Goal Session Storage...")
    session = StoredGoalSession(
        id="goal_001",
        goal="Research Quantum Computing",
        strategy=["Find papers", "Summarize key points", "Draft report"],
        current_step=1,
        completed_steps=["Find papers"],
        pending_steps=["Summarize key points", "Draft report"],
        context={"difficulty": "high", "source": "arXiv"},
        status="active",
        created_at=time.time(),
        last_active=time.time(),
        total_steps=3,
    )

    assert store.save_goal_session(session)
    loaded = store.load_goal_session("goal_001")
    assert loaded is not None
    assert loaded.goal == "Research Quantum Computing"
    assert isinstance(loaded.strategy, list)
    assert isinstance(loaded.context, dict)
    print("✅ Goal Session Storage Passed")

    # ── Test 2: Active Sessions Query ────────────────────────────────────────
    print("\n[TEST 2] Active Sessions Query...")
    active = store.get_active_sessions()
    assert len(active) >= 1
    assert any(s.id == "goal_001" for s in active)
    print("✅ Active Sessions Query Passed")

    # ── Test 3: Goal Status Update ───────────────────────────────────────────
    print("\n[TEST 3] Goal Status Update...")
    assert store.update_goal_status("goal_001", "completed")
    loaded = store.load_goal_session("goal_001")
    assert loaded.status == "completed"
    print("✅ Goal Status Update Passed")

    # ── Test 4: Shortcut Storage ─────────────────────────────────────────────
    print("\n[TEST 4] Shortcut Storage...")
    assert store.save_shortcut("hey friday lofi", "play_spotify_lofi_playlist")
    sc = store.load_shortcut("hey friday lofi")
    assert sc is not None
    assert sc.action == "play_spotify_lofi_playlist"
    print("✅ Shortcut Storage Passed")

    # ── Test 5: Shortcut Usage Tracking ──────────────────────────────────────
    print("\n[TEST 5] Shortcut Usage Tracking...")
    store.increment_shortcut_usage("hey friday lofi")
    sc = store.load_shortcut("hey friday lofi")
    assert sc.usage_count == 1
    print("✅ Shortcut Usage Tracking Passed")

    # ── Test 6: List All Shortcuts ───────────────────────────────────────────
    print("\n[TEST 6] List All Shortcuts...")
    shortcuts = store.list_shortcuts()
    assert len(shortcuts) >= 1
    print("✅ List All Shortcuts Passed")

    # ── Test 7: Storage Statistics ───────────────────────────────────────────
    print("\n[TEST 7] Storage Statistics...")
    stats = store.get_stats()
    assert "total_goals" in stats
    assert "active_goals" in stats
    assert "total_shortcuts" in stats
    print(f"   Stats: {stats}")
    print("✅ Storage Statistics Passed")

    # ── Test 8: Goal Deletion ────────────────────────────────────────────────
    print("\n[TEST 8] Goal Deletion...")
    assert store.delete_goal_session("goal_001")
    loaded = store.load_goal_session("goal_001")
    assert loaded is None
    print("✅ Goal Deletion Passed")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    shutil.rmtree(TEST_DIR)
    print("Cleanup completed. All MemoryStore tests passed! ✅")
    print("=" * 60)
