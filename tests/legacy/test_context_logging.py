import pytest
from app.logger import set_context
from app.memory.service import MemoryService
from loguru import logger


@pytest.mark.asyncio
async def test_context_propagates_to_memory_logs(tmp_path, monkeypatch):
    events = []
    logger.add(lambda msg: events.append(msg.record["extra"]))

    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    import app.config
    app.config.get_settings.cache_clear()  # type: ignore

    memory = MemoryService()
    await memory.init()
    set_context(session_id="sess-1", turn_id="turn-1")
    await memory.create_conversation("ctx")
    await memory.store_fact("ctx fact")

    assert any(e.get("session_id") == "sess-1" for e in events)
    assert any(e.get("turn_id") == "turn-1" for e in events)
