import pytest
from app.core.assistant import Assistant
from app.llm.provider import LLMProvider, LLMResponse
from app.memory.service import MemoryService
from app.models import Message, MessageRole
from app.tools.registry import ToolRegistry


class ToolCallProvider(LLMProvider):
    async def complete(self, messages):
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content="hi there"),
            tool_calls=[{"name": "recall", "function": {"name": "recall", "arguments": {"query": "hi"}}}],
        )


@pytest.mark.asyncio
async def test_stream_includes_tool_events(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore
    memory = MemoryService()
    await memory.init()
    await memory.store_fact("hi memory", tags=["test"])
    tools = ToolRegistry(memory=memory, confirm_mutating=True)
    assistant = Assistant(ToolCallProvider(), tools, memory=memory, max_iterations=1)

    events = []
    async for ev in assistant.stream_user_message("hello"):
        events.append(ev)
    kinds = [e["type"] for e in events if isinstance(e, dict)]
    assert "tool_start" in kinds
    assert "tool_end" in kinds
    assert "done" in kinds
