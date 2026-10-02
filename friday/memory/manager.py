"""
F.R.I.D.A.Y. Memory Manager
===========================
Tripartite memory architecture with Episodic, Semantic, and Working Memory layers.
Also provides unified interface for short-term (RAM), long-term (SQLite), and
vector-based (ChromaDB) memory with hybrid retrieval support.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from friday.config import config

os.environ.setdefault("TRANSFORMERS_CACHE", os.getenv("TRANSFORMERS_CACHE", "./models/transformers"))

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


# ── Embedding Cache ───────────────────────────────────────────────────────────

import hashlib
import collections

class EmbeddingCache:
    def __init__(self, max_size: int = 500) -> None:
        self._cache = collections.OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _hash(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def get(self, query: str) -> list[float] | None:
        key = self._hash(query)
        with self._lock:
            if key in self._cache:
                embedding, timestamp = self._cache[key]
                self._cache.move_to_end(key)
                if time.time() - timestamp < 3600:
                    self._hits += 1
                    logger.debug("[EmbeddingCache] Hit for: '%s'", query[:30])
                    return embedding
                else:
                    del self._cache[key]
            self._misses += 1
            logger.debug("[EmbeddingCache] Miss for: '%s'", query[:30])
            return None

    def set(self, query: str, embedding: list[float]) -> None:
        key = self._hash(query)
        with self._lock:
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            self._cache[key] = (embedding, time.time())
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            logger.debug("[EmbeddingCache] Cleared.")

_embedding_cache = EmbeddingCache()


# ── Memory Management Constants (Phase 7.5) ────────────────────────────────────

# If a new fact has a ChromaDB distance < 0.15 to an existing fact, skip saving.
SEMANTIC_DEDUP_THRESHOLD: float = 0.15

# Default age (in days) before long-term memories are deleted during compaction.
MEMORY_MAX_AGE_DAYS: int = 14


# ── Optional ChromaDB Setup ───────────────────────────────────────────────────

try:
    import chromadb

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("[Memory] chromadb not installed — semantic search disabled.")


# ── Data Models ───────────────────────────────────────────────────────────────


class MemoryTier(Enum):
    SHORT_TERM = auto()  # Volatile, fast, conversation window
    LONG_TERM = auto()  # Persistent, factual, indexed
    SEMANTIC = auto()  # Vector-based, conceptual, associative


class MemoryCategory(Enum):
    FACT = auto()
    PREFERENCE = auto()
    EVENT = auto()
    CONVERSATION = auto()
    GOAL = auto()
    SYSTEM = auto()
    WEB_RESEARCH = auto()


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    content: str
    category: MemoryCategory
    tier: MemoryTier
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    ttl: float | None = None

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    @classmethod
    def create(
        cls,
        content: str,
        category: MemoryCategory,
        tier: MemoryTier,
        metadata: dict | None = None,
        ttl: float | None = None,
    ) -> MemoryEntry:
        """Factory method to create a new memory entry."""
        return cls(
            id=str(uuid.uuid4()),
            content=content.strip(),
            category=category,
            tier=tier,
            timestamp=time.time(),
            metadata=metadata or {},
            ttl=ttl,
        )


def vitality_score(entry: MemoryEntry, recency_weight: float = 0.7, frequency_weight: float = 0.3) -> float:
    days_since_created = (time.time() - entry.timestamp) / 86400.0
    recency = 1.0 / (1.0 + days_since_created)
    access_count = int(entry.metadata.get("access_count", 1)) if entry.metadata else 1
    return (recency_weight * recency) + (frequency_weight * access_count)


# ── Short-Term Memory (RAM) ───────────────────────────────────────────────────


class ShortTermMemory:
    def __init__(self, max_size: int = config.storage.max_history) -> None:
        self._store: deque[MemoryEntry] = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def store(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._store.append(entry)

    def recall(self, category: MemoryCategory | None = None, limit: int = 10) -> list[MemoryEntry]:
        with self._lock:
            entries = [e for e in self._store if not e.is_expired]
            if category:
                entries = [e for e in entries if e.category == category]
            return entries[-limit:]

    def clear(self, category: MemoryCategory | None = None) -> int:
        with self._lock:
            if category is None:
                count = len(self._store)
                self._store.clear()
                return count

            before = len(self._store)
            remaining = [e for e in self._store if e.category != category]
            self._store.clear()
            self._store.extend(remaining)
            return before - len(remaining)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ── Long-Term Memory (SQLite) ────────────────────────────────────────────────-

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id        TEXT    PRIMARY KEY,
    content   TEXT    NOT NULL,
    category  TEXT    NOT NULL,
    timestamp REAL    NOT NULL,
    metadata  TEXT    NOT NULL DEFAULT '{}',
    ttl       REAL
);

CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id    TEXT    PRIMARY KEY,
    agent_type TEXT    NOT NULL,
    prompt     TEXT    NOT NULL,
    status     TEXT    NOT NULL,
    result     TEXT,
    created_at REAL    NOT NULL,
    updated_at REAL    NOT NULL
);
"""


class LongTermMemory:
    def __init__(self) -> None:
        self._db_path = config.storage.memory_db
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    @contextmanager
    def _assistant_connection(self) -> Generator[sqlite3.Connection, None, None]:
        if not hasattr(self._local, "assistant_conn") or self._local.assistant_conn is None:
            conn = sqlite3.connect(config.storage.assistant_db, timeout=10.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.row_factory = sqlite3.Row
            self._local.assistant_conn = conn
        try:
            yield self._local.assistant_conn
        except Exception as e:
            self._local.assistant_conn.rollback()
            logger.error(f"[LongTermMemory] Assistant database error: {e}")
            raise

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            # Logic: If we are accessing tasks, use assistant_db, else memory_db
            conn = sqlite3.connect(self._db_path, timeout=10.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")  # High-performance concurrent mode
            self._local.conn = conn
        try:
            yield self._local.conn
        except Exception as e:
            self._local.conn.rollback()
            logger.error(f"[LongTermMemory] Database error: {e}")
            raise

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def store(self, entry: MemoryEntry) -> None:
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.content,
                    entry.category.name,
                    entry.timestamp,
                    json.dumps(entry.metadata),
                    entry.ttl,
                ),
            )
            conn.commit()

    def recall(
        self, category: MemoryCategory | None = None, query: str | None = None, limit: int = 20
    ) -> list[MemoryEntry]:
        clauses, params = [], []
        if category:
            clauses.append("category = ?")
            params.append(category.name)
        if query:
            clauses.append("content LIKE ?")
            params.append(f"%{query}%")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories {where} ORDER BY timestamp DESC LIMIT ?", [*params, limit]
            ).fetchall()

        return [
            MemoryEntry(
                id=r["id"],
                content=r["content"],
                category=MemoryCategory[r["category"]],
                tier=MemoryTier.LONG_TERM,
                timestamp=r["timestamp"],
                metadata=json.loads(r["metadata"]),
                ttl=r["ttl"],
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # ── Task Management (Integrated into LTM for simplicity) ───────────────────

    def create_task(self, task_id: str, agent_type: str, prompt: str) -> None:
        now = time.time()
        with self._assistant_connection() as conn:
            conn.execute(
                "INSERT INTO agent_tasks (task_id, agent_type, prompt, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, agent_type, prompt, "IN_PROGRESS", now, now),
            )
            conn.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._assistant_connection() as conn:
            row = conn.execute("SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def update_task(self, task_id: str, status: str, result: str | None = None) -> None:
        now = time.time()
        with self._assistant_connection() as conn:
            if result is not None:
                conn.execute(
                    "UPDATE agent_tasks SET status=?, result=?, updated_at=? WHERE task_id=?",
                    (status, result, now, task_id),
                )
            else:
                conn.execute(
                    "UPDATE agent_tasks SET status=?, updated_at=? WHERE task_id=?",
                    (status, now, task_id),
                )
            conn.commit()


def _safe_category(name: str) -> MemoryCategory:
    """Safely parse a MemoryCategory name, defaulting to FACT on unknown values."""
    try:
        return MemoryCategory[name]
    except KeyError:
        return MemoryCategory.FACT


# ── Semantic Memory (ChromaDB) ────────────────────────────────────────────────


class SemanticMemory:
    def __init__(self) -> None:
        self._db_path = config.storage.chroma_db
        self._client = None
        self._collection = None
        self._available = False

    def setup(self) -> bool:
        if not _CHROMA_AVAILABLE:
            return False
        try:
            # Priority 5: Respect chroma_mode config for server-mode migration
            chroma_mode = getattr(config.storage, "chroma_mode", "persistent")
            if chroma_mode == "http":
                chroma_host = getattr(config.storage, "chroma_host", "localhost")
                chroma_port = getattr(config.storage, "chroma_port", 8000)
                self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
                logger.info("[SemanticMemory] Connected to ChromaDB server at %s:%d", chroma_host, chroma_port)
            else:
                self._client = chromadb.PersistentClient(path=self._db_path)

            # HNSW tuning metadata
            hnsw_m = getattr(config.storage, "chroma_hnsw_m", 16)
            hnsw_ef = getattr(config.storage, "chroma_hnsw_ef_search", 100)
            from chromadb.utils import embedding_functions

            ef = embedding_functions.DefaultEmbeddingFunction()
            self._collection = self._client.get_or_create_collection(
                name="friday_memory",
                embedding_function=ef,
                metadata={
                    "hnsw:M": hnsw_m,
                    "hnsw:search_ef": hnsw_ef,
                },
            )
            self._available = True
            return True
        except Exception as e:
            logger.error(f"[SemanticMemory] Setup failed: {e}")
            return False

    def store(self, entry: MemoryEntry) -> bool:
        if not self._available or self._collection is None:
            return False
        try:
            # Store metadata as strings (Chroma requirement)
            meta = {"category": entry.category.name, "timestamp": str(entry.timestamp)}
            meta.update({k: str(v) for k, v in entry.metadata.items()})

            self._collection.upsert(ids=[entry.id], documents=[entry.content], metadatas=[meta])
            return True
        except Exception as e:
            logger.error(f"[SemanticMemory] Store failed: {e}")
            return False

    def search(self, query: str, limit: int = 5) -> list[tuple[MemoryEntry, float]]:
        if not self._available or self._collection is None:
            return []
        try:
            # Check embedding cache
            embedding = _embedding_cache.get(query)
            if embedding is None:
                raw_emb = self._collection._embedding_function([query])[0]
                if hasattr(raw_emb, "tolist"):
                    embedding = raw_emb.tolist()
                else:
                    embedding = list(raw_emb)
                _embedding_cache.set(query, embedding)
            
            results = self._collection.query(query_embeddings=[embedding], n_results=limit)

            processed_results = []

            docs = results.get("documents")
            if not docs or not docs[0]:
                return []

            docs = docs[0]
            metas = results.get("metadatas")
            metas = metas[0] if metas and metas[0] else [{}] * len(docs)
            dists = results.get("distances")
            dists = dists[0] if dists and dists[0] else [0.0] * len(docs)
            ids = results.get("ids")
            ids = ids[0] if ids and ids[0] else ["unknown"] * len(docs)

            for i in range(len(docs)):
                doc = docs[i]
                meta = metas[i] if i < len(metas) else {}
                dist = dists[i] if i < len(dists) else 0.0
                entry_id = ids[i] if i < len(ids) else "unknown"

                # Reconstruct the MemoryEntry from semantic data
                entry = MemoryEntry(
                    id=entry_id,
                    content=doc,
                    category=_safe_category(meta.get("category", "FACT")),
                    tier=MemoryTier.SEMANTIC,
                    timestamp=float(meta.get("timestamp", time.time())),
                    metadata=meta,
                )
                processed_results.append((entry, dist))

            return processed_results
        except Exception as e:
            logger.error(f"[SemanticMemory] Search failed: {e}")
            return []


# ── Unified Memory Manager ────────────────────────────────────────────────────


class MemoryManager:
    def __init__(self) -> None:
        self._short = ShortTermMemory()
        self._long = LongTermMemory()
        self._semantic = SemanticMemory()
        self._working = WorkingMemory()
        self._episodic = EpisodicMemory()
        self._facts = SemanticFactStore()
        self._hygiene: MemoryHygiene | None = None
        self._ready = False

    def setup(self) -> bool:
        """Initialize subsystems."""
        self._semantic.setup()
        self._episodic.setup(self._semantic)
        self._facts.setup()
        self._hygiene = MemoryHygiene(self._long, self._semantic)
        self._pruned_last_run = 0
        self._ready = True

        # Background thread for hygiene running every 6 hours
        self._bg_thread = threading.Thread(
            target=self._run_hygiene_loop,
            daemon=True,
            name="memory-hygiene-worker"
        )
        self._bg_thread.start()

        return True

    def _run_hygiene_loop(self) -> None:
        """Runs the hygiene loop every 6 hours."""
        logger.info("[MemoryManager] Starting background memory hygiene thread.")
        # Run immediately on startup, then every 6 hours
        while self._ready:
            try:
                pruned = self._hygiene.prune(self, min_vitality=0.05)
                deduped = self._hygiene.deduplicate(self, similarity_threshold=0.95)
                self._hygiene.consolidate(self)
                self._pruned_last_run = pruned + deduped
                logger.info(
                    "[MemoryManager] Background hygiene run complete. Pruned: %d, Deduped: %d.",
                    pruned, deduped
                )
            except Exception as e:
                logger.error("[MemoryManager] Background hygiene loop error: %s", e)
            
            # Sleep 6 hours (checking ready flag every 10 seconds for clean stop)
            for _ in range(2160):  # 2160 * 10 seconds = 6 hours
                if not self._ready:
                    break
                time.sleep(10)

    def clear_embedding_cache(self) -> None:
        """Expose embedding cache clearing."""
        _embedding_cache.clear()

    def memory_stats(self) -> dict[str, Any]:
        """Return memory statistics including cache hit rate."""
        hits = _embedding_cache._hits
        misses = _embedding_cache._misses
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        return {
            "total_entries": self._long.count(),
            "pruned_last_run": getattr(self, "_pruned_last_run", 0),
            "cache_hit_rate": hit_rate,
        }

    def remember(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.FACT,
        tier: MemoryTier = MemoryTier.LONG_TERM,
        metadata: dict | None = None,
        semantic: bool = True,
    ) -> MemoryEntry:
        """
        The primary method to store information.
        If tier is LONG_TERM, it automatically mirrors the entry in SHORT_TERM for immediate recall.
        Now includes semantic deduplication to prevent storing near-identical facts.
        """
        # ── Semantic Deduplication ────────────────────────────────────────
        if semantic and self._semantic._available:
            try:
                existing = self._semantic.search(content, limit=1)
                if existing:
                    entry_check, dist = existing[0]
                    if dist < SEMANTIC_DEDUP_THRESHOLD:
                        logger.debug(
                            "[Memory] Dedup: Skipping save (dist=%.3f) for: '%s...'",
                            dist, content[:50],
                        )
                        return entry_check
            except Exception as e:
                logger.warning("[Memory] Dedup check failed: %s. Proceeding with save.", e)

        entry = MemoryEntry.create(content, category, tier, metadata)

        # ── Standard Storage Logic ────────────────────────────────────────
        # Note: Short-term capacity is natively handled by deque(maxlen=config.storage.max_history)
        if tier == MemoryTier.SHORT_TERM:
            self._short.store(entry)
        else:
            self._long.store(entry)
            self._short.store(entry)  # Mirror in short-term for conversational context

        if semantic:
            self._semantic.store(entry)

        return entry

    def remember_conversation(
        self,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Persist a conversation exchange to both short-term and long-term memory."""
        content = f"User: {user_text}\nFriday: {assistant_text}"
        self.remember(
            content=content,
            category=MemoryCategory.CONVERSATION,
            tier=MemoryTier.LONG_TERM,  # persists to SQLite + mirrors to short-term
            semantic=False,             # conversations don't need vector indexing
        )

    def build_context_summary(self, max_entries: int = 5) -> str:
        """
        Build a compact context string for LLM injection.
        Tries short-term first; falls back to long-term if session just started.
        """
        recent = self._short.recall(category=MemoryCategory.CONVERSATION, limit=max_entries)

        if not recent:
            # Session just started — pull from persistent long-term memory
            recent = self._long.recall(
                category=MemoryCategory.CONVERSATION,
                limit=max_entries,
            )

        if not recent:
            return ""

        lines = ["Recent context:"]
        for entry in recent:
            lines.append(f"  {entry.content}")

        return "\n".join(lines)

    def recall(
        self,
        category: MemoryCategory | None = None,
        limit: int = 10,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
    ) -> list[MemoryEntry]:
        """Retrieve memories from a specific tier."""
        if tier == MemoryTier.SHORT_TERM:
            return self._short.recall(category, limit)
        return self._long.recall(category, None, limit)

    def recall_relevant(self, query: str, limit: int = 5) -> list[tuple[MemoryEntry, float]]:
        """Perform a semantic vector search for the most conceptually relevant memories."""
        return self._semantic.search(query, limit)

    def compact_memory(self, max_age_days: int = MEMORY_MAX_AGE_DAYS) -> int:
        """
        Delete long-term and semantic memories older than max_age_days.
        Call this periodically (e.g., on startup or via a daily cron) to prevent bloat.

        Returns:
            Number of memories deleted.
        """
        cutoff = time.time() - (max_age_days * 86400)
        deleted_count = 0
        old_ids = []

        # 1. Find and delete old memories from SQLite
        try:
            with self._long._connection() as conn:
                rows = conn.execute(
                    "SELECT id FROM memories WHERE timestamp < ?", (cutoff,)
                ).fetchall()
                old_ids = [row["id"] for row in rows]

                if old_ids:
                    placeholders = ",".join("?" for _ in old_ids)
                    conn.execute(
                        f"DELETE FROM memories WHERE id IN ({placeholders})", old_ids
                    )
                    conn.commit()
                    deleted_count = len(old_ids)
        except Exception as e:
            logger.error("[Memory] SQLite compaction failed: %s", e)

        # 2. Delete corresponding vectors from ChromaDB
        if old_ids and self._semantic._available and self._semantic._collection:
            try:
                self._semantic._collection.delete(ids=old_ids)
            except Exception as e:
                logger.error("[Memory] ChromaDB compaction failed: %s", e)

        if deleted_count > 0:
            logger.info(
                "[Memory] Compaction: Deleted %d memories older than %d days.",
                deleted_count, max_age_days,
            )

        return deleted_count

    def export_memory(self, file_path: str | Path) -> bool:
        """
        Export all long-term memories from SQLite to a JSON file for backup.
        Semantic vectors are excluded (they can be regenerated from content if needed),
        but all metadata, timestamps, and content are preserved.
        """
        try:
            # Pull all records from SQLite (source of truth for full metadata)
            entries = self._long.recall(limit=100000)
            export_data = [
                {
                    "id": e.id,
                    "content": e.content,
                    "category": e.category.name,
                    "tier": e.tier.name,
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                    "ttl": e.ttl,
                }
                for e in entries
            ]

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            logger.info("[Memory] Exported %d memories to %s", len(export_data), path)
            return True
        except Exception as e:
            logger.error("[Memory] Export failed: %s", e)
            return False

    def import_memory(self, file_path: str | Path, overwrite: bool = False) -> int:
        """
        Import memories from a JSON file back into SQLite and ChromaDB.
        If overwrite=True, existing memories with the same ID will be replaced.

        Returns:
            Number of memories imported.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error("[Memory] Import file not found: %s", path)
                return 0

            with open(path, encoding="utf-8") as f:
                import_data = json.load(f)

            if not isinstance(import_data, list):
                logger.error("[Memory] Invalid import file format.")
                return 0

            count = 0
            for item in import_data:
                entry = MemoryEntry(
                    id=item.get("id", str(uuid.uuid4())),
                    content=item.get("content", ""),
                    category=_safe_category(item.get("category", "FACT")),
                    tier=MemoryTier[item.get("tier", "LONG_TERM")],
                    timestamp=item.get("timestamp", time.time()),
                    metadata=item.get("metadata", {}),
                    ttl=item.get("ttl"),
                )

                # Insert into SQLite
                self._long.store(entry)
                # Re-index into ChromaDB
                self._semantic.store(entry)
                count += 1

            logger.info("[Memory] Imported %d memories from %s", count, path)
            return count
        except Exception as e:
            logger.error("[Memory] Import failed: %s", e)
            return 0

    def clear_session(self) -> None:
        """Clear short-term memory (used when starting a new conversation)."""
        self._short.clear()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def working(self) -> "WorkingMemory":
        """Access the working memory (agent scratchpad)."""
        return self._working

    @property
    def episodic(self) -> "EpisodicMemory":
        """Access episodic memory (conversation turn storage)."""
        return self._episodic

    @property
    def facts(self) -> "SemanticFactStore":
        """Access the structured semantic fact store."""
        return self._facts

    @property
    def hygiene(self) -> "MemoryHygiene | None":
        """Access the memory hygiene system (available after setup)."""
        return self._hygiene

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "short_term": self._short.size,
            "long_term": self._long.count(),
            "semantic_enabled": self._semantic._available,
            "working_memory_active": self._working.has_state,
            "episodic_available": self._episodic.is_available,
            "facts_count": self._facts.count(),
        }


# ── Working Memory (Agent Scratchpad) ────────────────────────────────────────


class WorkingMemory:
    """
    Thread-safe scratchpad for the current agent loop state.
    Manages goals, plans, observations, and intermediate results
    for the active reasoning cycle.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._goal: str | None = None
        self._plan: list[str] = []
        self._observations: list[str] = []
        self._scratch: dict[str, Any] = {}

    def set_goal(self, goal: str) -> None:
        """Set the current task goal."""
        with self._lock:
            self._goal = goal.strip()

    def get_goal(self) -> str | None:
        """Return the current goal."""
        with self._lock:
            return self._goal

    def set_plan(self, steps: list[str]) -> None:
        """Set the execution plan as a list of steps."""
        with self._lock:
            self._plan = list(steps)

    def get_plan(self) -> list[str]:
        """Return the current plan steps."""
        with self._lock:
            return list(self._plan)

    def add_observation(self, observation: str) -> None:
        """Record an observation from the current reasoning cycle."""
        with self._lock:
            self._observations.append(observation.strip())

    def get_observations(self) -> list[str]:
        """Return all observations from the current cycle."""
        with self._lock:
            return list(self._observations)

    def set_scratch(self, key: str, value: Any) -> None:
        """Store an arbitrary key-value pair in the scratchpad."""
        with self._lock:
            self._scratch[key] = value

    def get_scratch(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the scratchpad."""
        with self._lock:
            return self._scratch.get(key, default)

    def compress(self, summarizer: Callable[[str], str] | None = None) -> str:
        """
        Compress observations into a summary and reset them.
        If a summarizer callable is provided, it's used to generate the summary.
        Otherwise, observations are joined with newlines.

        Returns:
            The compressed summary string.
        """
        with self._lock:
            if not self._observations:
                return ""

            raw = "\n".join(self._observations)
            summary = summarizer(raw) if summarizer else raw
            self._observations.clear()
            return summary

    def clear(self) -> None:
        """Reset all working memory state."""
        with self._lock:
            self._goal = None
            self._plan.clear()
            self._observations.clear()
            self._scratch.clear()

    @property
    def has_state(self) -> bool:
        """Return True if working memory has any active state."""
        with self._lock:
            return bool(self._goal or self._plan or self._observations or self._scratch)


# ── Episodic Memory (Conversation Turn Storage) ──────────────────────────────


class EpisodicMemory:
    """
    Stores specific past experiences and conversation turns with rich metadata.
    Supports contextual recall like "continue our previous thought" by storing
    turn indices, session IDs, and speaker information.
    """

    def __init__(self) -> None:
        self._semantic: SemanticMemory | None = None
        self._available = False

    def setup(self, semantic_memory: SemanticMemory) -> bool:
        """Link to the underlying semantic memory for vector search."""
        self._semantic = semantic_memory
        self._available = semantic_memory._available
        return self._available

    def store_turn(
        self,
        user_text: str,
        assistant_text: str,
        session_id: str = "default",
        turn_index: int = 0,
    ) -> bool:
        """
        Store a conversation turn with episodic metadata.

        Args:
            user_text:      What the user said.
            assistant_text:  What the assistant responded.
            session_id:     Session identifier for grouping turns.
            turn_index:     Turn number within the session.
        """
        if not self._available or not self._semantic:
            return False

        content = f"User: {user_text}\nAssistant: {assistant_text}"
        entry = MemoryEntry.create(
            content=content,
            category=MemoryCategory.CONVERSATION,
            tier=MemoryTier.SEMANTIC,
            metadata={
                "session_id": session_id,
                "turn_index": str(turn_index),
                "speaker": "dialogue",
                "memory_type": "episodic",
            },
        )
        return self._semantic.store(entry)

    def recall_recent(
        self,
        session_id: str = "default",
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """
        Recall recent turns from a specific session.
        Uses vector search with session metadata filtering.
        """
        if not self._available or not self._semantic:
            return []

        # Search for recent conversation context
        results = self._semantic.search(f"session:{session_id} recent conversation", limit=limit * 2)
        # Filter by session_id
        filtered = [
            entry for entry, _dist in results
            if entry.metadata.get("session_id") == session_id
        ]
        return filtered[:limit]

    def recall_similar(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Recall episodic memories similar to the query.
        Filters for episodic entries specifically.
        """
        if not self._available or not self._semantic:
            return []

        results = self._semantic.search(query, limit=limit * 2)
        episodic = [
            (entry, dist) for entry, dist in results
            if entry.metadata.get("memory_type") == "episodic"
               or entry.category == MemoryCategory.CONVERSATION
        ]
        return episodic[:limit]

    @property
    def is_available(self) -> bool:
        return self._available


# ── Structured Fact Store (Semantic Facts) ────────────────────────────────────


@dataclass(frozen=True)
class StructuredFact:
    """A structured fact with subject-predicate-value triple."""
    subject: str
    predicate: str
    value: str
    confidence: float
    source: str
    updated_at: float


_FACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_facts (
    subject    TEXT    NOT NULL,
    predicate  TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    confidence REAL    NOT NULL DEFAULT 0.5,
    source     TEXT    NOT NULL DEFAULT 'user',
    updated_at REAL    NOT NULL,
    PRIMARY KEY (subject, predicate)
);
"""


class SemanticFactStore:
    """
    Stores structured, agent-writable facts in a relational format.
    Uses subject-predicate-value triples with confidence scoring and
    automatic deduplication via upsert.
    """

    def __init__(self) -> None:
        self._db_path = config.storage.memory_db
        self._local = threading.local()
        self._ready = False

    def setup(self) -> bool:
        """Initialize the facts table."""
        try:
            with self._connection() as conn:
                conn.executescript(_FACTS_SCHEMA)
                conn.commit()
            self._ready = True
            return True
        except Exception as e:
            logger.error("[SemanticFactStore] Setup failed: %s", e)
            return False

    @contextmanager
    def _connection(self) -> "Generator[sqlite3.Connection, None, None]":
        if not hasattr(self._local, "fact_conn") or self._local.fact_conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            self._local.fact_conn = conn
        try:
            yield self._local.fact_conn
        except Exception as e:
            self._local.fact_conn.rollback()
            logger.error("[SemanticFactStore] Database error: %s", e)
            raise

    def store_fact(
        self,
        subject: str,
        predicate: str,
        value: str,
        confidence: float = 0.5,
        source: str = "user",
    ) -> bool:
        """
        Store or update a fact. If the same subject+predicate exists,
        updates only if the new confidence >= old confidence.
        """
        try:
            with self._connection() as conn:
                existing = conn.execute(
                    "SELECT confidence FROM semantic_facts WHERE subject=? AND predicate=?",
                    (subject.strip(), predicate.strip()),
                ).fetchone()

                if existing and existing["confidence"] > confidence:
                    logger.debug(
                        "[Facts] Skipping update: existing confidence %.2f > new %.2f for %s.%s",
                        existing["confidence"], confidence, subject, predicate,
                    )
                    return False

                conn.execute(
                    "INSERT OR REPLACE INTO semantic_facts VALUES (?, ?, ?, ?, ?, ?)",
                    (subject.strip(), predicate.strip(), value.strip(),
                     confidence, source, time.time()),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error("[Facts] Store failed: %s", e)
            return False

    def query_facts(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int = 10,
    ) -> list[StructuredFact]:
        """Query facts by subject and/or predicate."""
        clauses, params = [], []
        if subject:
            clauses.append("subject = ?")
            params.append(subject.strip())
        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate.strip())

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    f"SELECT * FROM semantic_facts {where} ORDER BY updated_at DESC LIMIT ?",
                    [*params, limit],
                ).fetchall()
            return [
                StructuredFact(
                    subject=r["subject"],
                    predicate=r["predicate"],
                    value=r["value"],
                    confidence=r["confidence"],
                    source=r["source"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]
        except Exception as e:
            logger.error("[Facts] Query failed: %s", e)
            return []

    def update_fact(
        self,
        subject: str,
        predicate: str,
        new_value: str,
        new_confidence: float | None = None,
    ) -> bool:
        """Update an existing fact's value and optionally its confidence."""
        try:
            with self._connection() as conn:
                if new_confidence is not None:
                    conn.execute(
                        "UPDATE semantic_facts SET value=?, confidence=?, updated_at=? "
                        "WHERE subject=? AND predicate=?",
                        (new_value.strip(), new_confidence, time.time(),
                         subject.strip(), predicate.strip()),
                    )
                else:
                    conn.execute(
                        "UPDATE semantic_facts SET value=?, updated_at=? "
                        "WHERE subject=? AND predicate=?",
                        (new_value.strip(), time.time(),
                         subject.strip(), predicate.strip()),
                    )
                conn.commit()
                return conn.total_changes > 0
        except Exception as e:
            logger.error("[Facts] Update failed: %s", e)
            return False

    def count(self) -> int:
        """Return the total number of stored facts."""
        if not self._ready:
            return 0
        try:
            with self._connection() as conn:
                return conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
        except Exception:
            return 0


# ── Memory Hygiene (Vitality Scoring + Pruning) ──────────────────────────────


class MemoryHygiene:
    """
    Maintains the health and integrity of the memory system.
    Calculates vitality scores for memories based on recency, access frequency,
    and importance. Prunes low-vitality entries and consolidates related facts.
    """

    def __init__(self, long_term: LongTermMemory, semantic: SemanticMemory) -> None:
        self._long_term = long_term
        self._semantic = semantic

    @staticmethod
    def calculate_vitality(
        entry: MemoryEntry,
        recency_weight: float = 0.4,
        importance_weight: float = 0.3,
        access_weight: float = 0.3,
    ) -> float:
        """
        Calculate a vitality score (0.0–1.0) for a memory entry.

        Factors:
          - Recency: How recently the memory was created (exponential decay).
          - Importance: Based on category (GOAL > PREFERENCE > FACT > CONVERSATION).
          - Access frequency: From metadata if tracked.
        """
        now = time.time()
        age_days = (now - entry.timestamp) / 86400.0

        # Recency score (exponential decay, half-life = 7 days)
        recency = 2.0 ** (-age_days / 7.0)

        # Importance score by category
        importance_map = {
            MemoryCategory.GOAL: 1.0,
            MemoryCategory.PREFERENCE: 0.9,
            MemoryCategory.SYSTEM: 0.8,
            MemoryCategory.FACT: 0.7,
            MemoryCategory.EVENT: 0.6,
            MemoryCategory.WEB_RESEARCH: 0.5,
            MemoryCategory.CONVERSATION: 0.3,
        }
        importance = importance_map.get(entry.category, 0.5)

        # Access frequency (from metadata)
        access_count = int(entry.metadata.get("access_count", "1"))
        access = min(1.0, access_count / 10.0)  # Normalize: 10+ accesses = max

        return (
            recency * recency_weight
            + importance * importance_weight
            + access * access_weight
        )

    def prune_low_vitality(self, threshold: float = 0.2) -> int:
        """
        Remove memories with vitality score below the threshold.

        Returns:
            Number of memories pruned.
        """
        entries = self._long_term.recall(limit=10000)
        to_delete: list[str] = []

        for entry in entries:
            vitality = self.calculate_vitality(entry)
            if vitality < threshold:
                to_delete.append(entry.id)

        if not to_delete:
            return 0

        try:
            with self._long_term._connection() as conn:
                placeholders = ",".join("?" for _ in to_delete)
                conn.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})",
                    to_delete,
                )
                conn.commit()
        except Exception as e:
            logger.error("[MemoryHygiene] Pruning failed: %s", e)
            return 0

        # Also clean from semantic memory
        if self._semantic._available and self._semantic._collection:
            try:
                self._semantic._collection.delete(ids=to_delete)
            except Exception as e:
                logger.error("[MemoryHygiene] Semantic pruning failed: %s", e)

        logger.info("[MemoryHygiene] Pruned %d low-vitality memories.", len(to_delete))
        return len(to_delete)

    def consolidate_related(self, similarity_threshold: float = 0.85) -> int:
        """
        Merge near-duplicate memories based on semantic similarity.
        Keeps the higher-vitality entry and removes the duplicate.

        Returns:
            Number of consolidations performed.
        """
        if not self._semantic._available:
            return 0

        entries = self._long_term.recall(limit=1000)
        consolidated = 0
        seen_ids: set[str] = set()

        for entry in entries:
            if entry.id in seen_ids:
                continue

            similar = self._semantic.search(entry.content, limit=3)
            for similar_entry, distance in similar:
                if similar_entry.id == entry.id or similar_entry.id in seen_ids:
                    continue

                # Lower distance = more similar (ChromaDB L2 distance)
                if distance < (1.0 - similarity_threshold):
                    # Keep the one with higher vitality
                    v1 = self.calculate_vitality(entry)
                    v2 = self.calculate_vitality(similar_entry)
                    loser_id = similar_entry.id if v1 >= v2 else entry.id

                    try:
                        with self._long_term._connection() as conn:
                            conn.execute("DELETE FROM memories WHERE id = ?", (loser_id,))
                            conn.commit()
                        if self._semantic._collection:
                            self._semantic._collection.delete(ids=[loser_id])
                    except Exception as e:
                        logger.error("[MemoryHygiene] Consolidation delete failed: %s", e)
                        continue

                    seen_ids.add(loser_id)
                    consolidated += 1

        if consolidated > 0:
            logger.info("[MemoryHygiene] Consolidated %d duplicate memories.", consolidated)
        return consolidated

    def prune(self, memory_manager: MemoryManager, min_vitality: float = 0.05) -> int:
        """
        Remove entries with vitality below threshold.
        Never remove entries newer than 24 hours.
        Never remove PREFERENCE or GOAL category entries.
        """
        entries = memory_manager._long.recall(limit=10000)
        to_delete: list[str] = []
        now = time.time()

        for entry in entries:
            # Never remove entries newer than 24 hours
            if now - entry.timestamp < 86400:
                continue
            # Never remove PREFERENCE or GOAL category entries
            if entry.category in (MemoryCategory.PREFERENCE, MemoryCategory.GOAL):
                continue

            score = vitality_score(entry)
            if score < min_vitality:
                to_delete.append(entry.id)

        if not to_delete:
            return 0

        # Remove from LongTermMemory
        try:
            with memory_manager._long._connection() as conn:
                placeholders = ",".join("?" for _ in to_delete)
                conn.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})",
                    to_delete,
                )
                conn.commit()
        except Exception as e:
            logger.error("[MemoryHygiene] prune failed: %s", e)
            return 0

        # Remove from SemanticMemory
        if memory_manager._semantic._available and memory_manager._semantic._collection:
            try:
                memory_manager._semantic._collection.delete(ids=to_delete)
            except Exception as e:
                logger.error("[MemoryHygiene] Semantic prune failed: %s", e)

        logger.info("[MemoryHygiene] Pruned %d low-vitality memories.", len(to_delete))
        return len(to_delete)

    def deduplicate(self, memory_manager: MemoryManager, similarity_threshold: float = 0.95) -> int:
        """
        Find pairs with cosine similarity > threshold.
        Keep higher vitality entry, remove lower.
        """
        if not memory_manager._semantic._available:
            return 0

        entries = memory_manager._long.recall(limit=1000)
        deduped = 0
        seen_ids: set[str] = set()

        for entry in entries:
            if entry.id in seen_ids:
                continue

            similar = memory_manager._semantic.search(entry.content, limit=3)
            for similar_entry, distance in similar:
                if similar_entry.id == entry.id or similar_entry.id in seen_ids:
                    continue

                # ChromaDB L2 or cosine distance threshold
                if distance < (1.0 - similarity_threshold):
                    # Keep the one with higher vitality
                    v1 = vitality_score(entry)
                    v2 = vitality_score(similar_entry)
                    loser_id = similar_entry.id if v1 >= v2 else entry.id

                    try:
                        with memory_manager._long._connection() as conn:
                            conn.execute("DELETE FROM memories WHERE id = ?", (loser_id,))
                            conn.commit()
                        if memory_manager._semantic._collection:
                            memory_manager._semantic._collection.delete(ids=[loser_id])
                    except Exception as e:
                        logger.error("[MemoryHygiene] deduplicate delete failed: %s", e)
                        continue

                    seen_ids.add(loser_id)
                    deduped += 1

        if deduped > 0:
            logger.info("[MemoryHygiene] Deduplicated %d memories.", deduped)
        return deduped

    def consolidate(self, memory_manager: MemoryManager) -> None:
        """
        Merge CONVERSATION entries older than 7 days
        into a single summary entry per day.
        """
        cutoff = time.time() - (7 * 86400)
        try:
            with memory_manager._long._connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE category = ? AND timestamp < ? ORDER BY timestamp ASC",
                    (MemoryCategory.CONVERSATION.name, cutoff),
                ).fetchall()
        except Exception as e:
            logger.error("[MemoryHygiene] consolidate fetch failed: %s", e)
            return

        if not rows:
            return

        import datetime
        grouped = {}
        for r in rows:
            dt = datetime.datetime.fromtimestamp(r["timestamp"])
            day_str = dt.strftime("%Y-%m-%d")
            if day_str not in grouped:
                grouped[day_str] = []
            grouped[day_str].append(r)

        for day_str, day_rows in grouped.items():
            if len(day_rows) <= 1:
                # No need to merge a single conversation entry
                continue

            # Merge contents
            merged_lines = [f"Daily Summary of conversations ({day_str}):"]
            for r in day_rows:
                merged_lines.append(r["content"])
            merged_content = "\n---\n".join(merged_lines)

            # Create new consolidated entry
            # Use the timestamp of the last entry of that day
            last_timestamp = day_rows[-1]["timestamp"]
            new_id = str(uuid.uuid4())
            new_entry = MemoryEntry(
                id=new_id,
                content=merged_content,
                category=MemoryCategory.CONVERSATION,
                tier=MemoryTier.LONG_TERM,
                timestamp=last_timestamp,
                metadata={"consolidated": "true", "day": day_str},
            )

            # Store the new entry and delete the old ones
            old_ids = [r["id"] for r in day_rows]
            try:
                with memory_manager._long._connection() as conn:
                    # Insert new entry
                    conn.execute(
                        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            new_entry.id,
                            new_entry.content,
                            new_entry.category.name,
                            new_entry.timestamp,
                            json.dumps(new_entry.metadata),
                            new_entry.ttl,
                        ),
                    )
                    # Delete old entries
                    placeholders = ",".join("?" for _ in old_ids)
                    conn.execute(
                        f"DELETE FROM memories WHERE id IN ({placeholders})",
                        old_ids,
                    )
                    conn.commit()

                # Also delete old entries from ChromaDB just in case
                if memory_manager._semantic._available and memory_manager._semantic._collection:
                    try:
                        memory_manager._semantic._collection.delete(ids=old_ids)
                    except Exception as e:
                        logger.error("[MemoryHygiene] ChromaDB delete in consolidate failed: %s", e)

            except Exception as e:
                logger.error("[MemoryHygiene] Consolidation failed for day %s: %s", day_str, e)
