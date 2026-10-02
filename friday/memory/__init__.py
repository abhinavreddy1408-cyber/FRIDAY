"""
F.R.I.D.A.Y. Memory Package
==========================
Centralized interface for short-term conversational context
and long-term semantic memory.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional  # noqa: F401

from .manager import MemoryCategory, MemoryEntry, MemoryManager, MemoryTier

_manager: MemoryManager | None = None
_lock = threading.Lock()


@dataclass
class VectorMemoryRecord:
    text: str
    embedding: list[float]
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


class VectorMemory:
    """Small cosine-similarity memory store backed by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model: object | None = None) -> None:
        self.model_name = model_name
        self._model = model
        self._records: list[VectorMemoryRecord] = []

    @property
    def records(self) -> list[VectorMemoryRecord]:
        return list(self._records)

    def store(self, text: str, tags: list[str] | None = None) -> VectorMemoryRecord:
        if not text or not text.strip():
            raise ValueError("VectorMemory.store requires non-empty text")
        embedding = self._embed(text)
        record = VectorMemoryRecord(text=text.strip(), embedding=embedding, tags=tags or [])
        self._records.append(record)
        return record

    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("VectorMemory.retrieve_relevant requires non-empty query")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self._records:
            return []
        query_embedding = self._embed(query)
        ranked = sorted(
            (
                {
                    "text": record.text,
                    "embedding": record.embedding,
                    "timestamp": record.timestamp,
                    "tags": record.tags,
                    "score": _cosine_similarity(query_embedding, record.embedding),
                }
                for record in self._records
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked[:top_k]

    def _embed(self, text: str) -> list[float]:
        model = self._get_model()
        vector = model.encode(text)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if vector and isinstance(vector[0], list):
            vector = vector[0]
        return [float(item) for item in vector]

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for VectorMemory embeddings"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def initialize_memory() -> MemoryManager:
    """Thread-safe initialization of the global MemoryManager singleton."""
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:  # Double-checked locking pattern
                _manager = MemoryManager()
                _manager.setup()
    return _manager


def get_memory() -> MemoryManager:
    """Fetch the active MemoryManager. Raises RuntimeError if uninitialized."""
    if _manager is None:
        raise RuntimeError(
            "MemoryManager accessed before initialization. "
            "Call initialize_memory() during system boot."
        )
    return _manager


__all__ = [
    "MemoryCategory",
    "MemoryEntry",
    "MemoryManager",
    "MemoryTier",
    "VectorMemory",
    "VectorMemoryRecord",
    "get_memory",
    "initialize_memory",
]
