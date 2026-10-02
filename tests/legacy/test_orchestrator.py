import pytest
from app.core.assistant import Assistant
from app.llm.provider import LLMProvider, LLMResponse
from app.memory.service import MemoryService
from app.models import Message, MessageRole
from app.tools.registry import ToolRegistry


class ToolMockProvider(LLMProvider):
    """First call emits a tool call, second call returns final answer."""

    def __init__(self):
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                message=Message(role=MessageRole.assistant, content="running tool"),
                tool_calls=[{"name": "recall", "function": {"name": "recall", "arguments": {"query": "hello"}}}],
            )
        return LLMResponse(message=Message(role=MessageRole.assistant, content="done"))


class VisionInspectProvider(LLMProvider):
    def __init__(self):
        self.seen_messages = None

    async def complete(self, messages):
        self.seen_messages = messages
        return LLMResponse(message=Message(role=MessageRole.assistant, content="vision done"))


@pytest.mark.asyncio
async def test_orchestrator_runs_tool_and_finishes(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore
    memory = MemoryService()
    await memory.init()
    await memory.store_fact("hello world", tags=["test"])
    tools = ToolRegistry(memory=memory)
    assistant = Assistant(ToolMockProvider(), tools, memory=memory, max_iterations=2)

    convo_id, reply = await assistant.handle_user_message("hi")
    assert reply.content == "done"
    assert convo_id is not None


@pytest.mark.asyncio
async def test_vision_intent_auto_attaches_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path}/vision.db")
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore
    memory = MemoryService()
    await memory.init()
    tools = ToolRegistry(memory=memory)
    provider = VisionInspectProvider()
    assistant = Assistant(provider, tools, memory=memory, max_iterations=1)

    fake_jpeg = b"\xff\xd8\xff\xdbdummy"
    import base64

    monkeypatch.setattr(
        tools,
        "_capture_screen_bytes",
        lambda max_width, quality: (fake_jpeg, 800, 500, "image/jpeg"),
    )
    _, reply = await assistant.handle_user_message("look at my screen")
    assert reply.content == "vision done"
    assert provider.seen_messages is not None
    user_message = provider.seen_messages[-1]
    assert isinstance(user_message.content, list)
    image_block = next(block for block in user_message.content if getattr(block, "type", "") == "image_url")
    assert image_block.image_url.url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_block.image_url.url.split(",", 1)[1]) == fake_jpeg
