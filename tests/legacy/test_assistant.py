import importlib

import pytest
from app.llm.provider import MockProvider
from app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_assistant_with_mock(tmp_path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("DB_URL", db_url)
    import app.config
    app.config.get_settings.cache_clear()  # type: ignore
    import app.memory.sqlite as mem
    importlib.reload(mem)
    await mem.init_db()

    from app.core.assistant import Assistant
    assistant = Assistant(MockProvider(["hi there"]), ToolRegistry())
    convo_id, reply = await assistant.handle_user_message("hello")
    assert reply.content == "hi there" or reply.role.value == "tool"
    assert convo_id is not None
