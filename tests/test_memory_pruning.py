# mypy: ignore-errors
"""
Phase 7.5 Tests — Memory Pruning, Deduplication, and Export/Import
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from friday.memory.manager import (
    MemoryCategory,
    MemoryEntry,
    MemoryManager,
    MemoryTier,
)


@pytest.fixture
def manager():
    """Create a MemoryManager with mocked subsystems."""
    mgr = MemoryManager()
    mgr._short = MagicMock()
    mgr._long = MagicMock()
    mgr._semantic = MagicMock()
    return mgr


class TestSemanticDeduplication:
    """Verify near-duplicate memories are rejected before hitting SQLite/Chroma."""

    def test_skips_save_when_distance_below_threshold(self, manager):
        """Should return existing entry and NOT call _long.store if dist < 0.15."""
        fake_entry = MemoryEntry.create("Existing fact", MemoryCategory.FACT, MemoryTier.LONG_TERM)

        # Simulate ChromaDB finding a very close match (distance 0.05)
        manager._semantic._available = True
        manager._semantic.search.return_value = [(fake_entry, 0.05)]

        result = manager.remember("Existing fact", semantic=True)

        # Should return the existing entry
        assert result.id == fake_entry.id
        # Should NOT store in SQLite or Chroma
        manager._long.store.assert_not_called()
        manager._semantic.store.assert_not_called()
        manager._short.store.assert_not_called()

    def test_saves_when_distance_above_threshold(self, manager):
        """Should save normally if dist > 0.15."""
        fake_entry = MemoryEntry.create("Old fact", MemoryCategory.FACT, MemoryTier.LONG_TERM)

        # Simulate ChromaDB finding a distant match (distance 0.8)
        manager._semantic._available = True
        manager._semantic.search.return_value = [(fake_entry, 0.8)]

        manager.remember("Brand new fact", semantic=True)

        # Should proceed with normal save
        manager._long.store.assert_called_once()
        manager._semantic.store.assert_called_once()
        manager._short.store.assert_called_once()

    def test_saves_when_semantic_is_false(self, manager):
        """Deduplication should be bypassed if semantic=False."""
        manager._semantic._available = True
        manager._semantic.search.return_value = [(MagicMock(), 0.01)] # Close match

        manager.remember("Some fact", semantic=False)

        # Should save because semantic flag was False
        manager._long.store.assert_called_once()

    def test_saves_when_chromadb_unavailable(self, manager):
        """Deduplication should be bypassed if ChromaDB failed to setup."""
        manager._semantic._available = False

        manager.remember("Some fact", semantic=True)

        # search() shouldn't even be called, save should proceed
        manager._semantic.search.assert_not_called()
        manager._long.store.assert_called_once()


class TestMemoryCompaction:
    """Verify old memories are deleted from both SQLite and ChromaDB."""

    def test_deletes_old_memories_from_both_dbs(self, manager):
        time.time() - (15 * 86400) # 15 days ago

        # Mock SQLite returning two old IDs
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": "old_1"}, {"id": "old_2"}]
        mock_conn.execute.return_value = mock_cursor

        manager._long._connection.return_value.__enter__.return_value = mock_conn

        deleted = manager.compact_memory(max_age_days=14)

        assert deleted == 2
        # Verify SQLite DELETE was executed
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called_once()
        # Verify ChromaDB delete was called with the IDs
        manager._semantic._collection.delete.assert_called_once_with(ids=["old_1", "old_2"])

    def test_skips_deletion_if_no_old_memories(self, manager):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [] # No old rows
        mock_conn.execute.return_value = mock_cursor

        manager._long._connection.return_value.__enter__.return_value = mock_conn

        deleted = manager.compact_memory()

        assert deleted == 0
        manager._semantic._collection.delete.assert_not_called()


class TestExportImport:
    """Verify memories can be serialized and deserialized."""

    def test_export_writes_json_to_file(self, manager):
        fake_entry = MemoryEntry.create(
            content="Test fact",
            category=MemoryCategory.FACT,
            tier=MemoryTier.LONG_TERM,
            metadata={"source": "test"}
        )
        manager._long.recall.return_value = [fake_entry]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = manager.export_memory(temp_path)
            assert result is True

            with open(temp_path) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["content"] == "Test fact"
            assert data[0]["metadata"]["source"] == "test"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_import_reads_json_and_stores(self, manager):
        import_data = [
            {
                "id": "import_1",
                "content": "Imported fact",
                "category": "FACT",
                "tier": "LONG_TERM",
                "timestamp": time.time(),
                "metadata": {},
                "ttl": None
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(import_data, f)
            temp_path = f.name

        try:
            count = manager.import_memory(temp_path)
            assert count == 1

            # Verify it tried to store in both SQLite and Chroma
            manager._long.store.assert_called_once()
            manager._semantic.store.assert_called_once()

            # Verify the passed object is a MemoryEntry
            stored_entry = manager._long.store.call_args[0][0]
            assert isinstance(stored_entry, MemoryEntry)
            assert stored_entry.id == "import_1"
            assert stored_entry.content == "Imported fact"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_import_handles_missing_file(self, manager):
        count = manager.import_memory("/nonexistent/path.json")
        assert count == 0
