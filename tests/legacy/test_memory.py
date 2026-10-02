import importlib
from datetime import timezone

import pytest
from app.models import MessageRole


@pytest.mark.asyncio
async def test_memory_roundtrip(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("DB_URL", db_url)
    import app.config
    app.config.get_settings.cache_clear()  # type: ignore
    import app.memory.sqlite as mem
    importlib.reload(mem)

    await mem.init_db()
    convo = await mem.create_conversation("test")
    await mem.add_message(convo.id, MessageRole.user, "hello")
    msgs = await mem.get_recent_messages(convo.id, 5)
    assert msgs[-1].content == "hello"

@pytest.mark.asyncio
async def test_memory_search(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("DB_URL", db_url)
    import app.config
    app.config.get_settings.cache_clear()  # type: ignore
    import app.memory.sqlite as mem
    importlib.reload(mem)

    await mem.init_db()
    convo = await mem.create_conversation("search")
    await mem.add_message(convo.id, MessageRole.user, "find this needle")
    results = await mem.search_memories(convo.id, "needle")
    assert len(results) == 1


def test_timestamps_timezone_aware():
    from app.models import Conversation, MemoryRecord

    m = MemoryRecord(role="user", content="test")
    c = Conversation()
    assert m.created_at.tzinfo is not None
    assert m.created_at.tzinfo.utcoffset(m.created_at) == timezone.utc.utcoffset(m.created_at)
    assert c.created_at.tzinfo is not None


@pytest.mark.asyncio
async def test_memory_service_semantic_search(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("DB_URL", db_url)
    import app.config
    app.config.get_settings.cache_clear()  # type: ignore

    from app.memory.service import MemoryService

    service = MemoryService()
    await service.init()
    await service.store_fact("Python is great for assistants", tags=["lang"])
    await service.store_fact("Basketball is a sport", tags=["sports"])

    results = await service.semantic_search("assistants")
    assert len(results) >= 1
    assert any("assistant" in r.content.lower() or "python" in r.content.lower() for r in results)
