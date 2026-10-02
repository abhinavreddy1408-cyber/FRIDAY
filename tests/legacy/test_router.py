import pytest
from app.core.assistant import Assistant
from app.llm.provider import LLMProvider, LLMResponse
from app.llm.router import HeuristicIntentRouter, IntentType, ModelSupervisor
from app.memory.service import MemoryService
from app.models import Message, MessageRole
from app.tools.registry import ToolRegistry


class SpyProvider(LLMProvider):
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        return LLMResponse(message=Message(role=MessageRole.assistant, content=self.text))


@pytest.mark.asyncio
async def test_heuristic_router_labels():
    router = HeuristicIntentRouter()
    assert await router.classify("hello there", []) == IntentType.chat_simple
    assert await router.classify("open app and list directory", []) == IntentType.tool_request
    assert await router.classify("write python tests for flask", []) == IntentType.coding_complex
    assert await router.classify("look at my screen and explain this error", []) == IntentType.vision_request
    assert await router.classify("what is this error on my display?", []) == IntentType.vision_request


def test_model_supervisor_selects_models():
    fast = SpyProvider("fast")
    heavy = SpyProvider("heavy")
    supervisor = ModelSupervisor(fast_provider=fast, heavy_provider=heavy)
    assert supervisor.select(IntentType.chat_simple) is fast
    assert supervisor.select(IntentType.tool_request) is heavy
    assert supervisor.select(IntentType.coding_complex) is heavy
    assert supervisor.select(IntentType.vision_request) is heavy


@pytest.mark.asyncio
async def test_assistant_uses_fast_for_simple_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path}/router_fast.db")
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore
    memory = MemoryService()
    await memory.init()
    tools = ToolRegistry(memory=memory, confirm_mutating=True)

    fast = SpyProvider("fast-answer")
    heavy = SpyProvider("heavy-answer")
    assistant = Assistant(
        provider=fast,
        tools=tools,
        memory=memory,
        intent_router=HeuristicIntentRouter(),
        model_supervisor=ModelSupervisor(fast_provider=fast, heavy_provider=heavy),
    )
    _, reply = await assistant.handle_user_message("hello")
    assert reply.content == "fast-answer"
    assert fast.calls == 1
    assert heavy.calls == 0


@pytest.mark.asyncio
async def test_assistant_uses_heavy_for_coding_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path}/router_heavy.db")
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore
    memory = MemoryService()
    await memory.init()
    tools = ToolRegistry(memory=memory, confirm_mutating=True)

    fast = SpyProvider("fast-answer")
    heavy = SpyProvider("heavy-answer")
    assistant = Assistant(
        provider=fast,
        tools=tools,
        memory=memory,
        intent_router=HeuristicIntentRouter(),
        model_supervisor=ModelSupervisor(fast_provider=fast, heavy_provider=heavy),
    )
    _, reply = await assistant.handle_user_message("write python code to parse json")
    assert reply.content == "heavy-answer"
    assert fast.calls == 0
    assert heavy.calls == 1


@pytest.mark.asyncio
async def test_assistant_uses_heavy_for_tool_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path}/router_tool.db")
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore
    memory = MemoryService()
    await memory.init()
    tools = ToolRegistry(memory=memory, confirm_mutating=True)

    fast = SpyProvider("fast-answer")
    heavy = SpyProvider("heavy-answer")
    assistant = Assistant(
        provider=fast,
        tools=tools,
        memory=memory,
        intent_router=HeuristicIntentRouter(),
        model_supervisor=ModelSupervisor(fast_provider=fast, heavy_provider=heavy),
    )
    _, reply = await assistant.handle_user_message("list directory please")
    assert reply.content == "heavy-answer"
    assert fast.calls == 0
    assert heavy.calls == 1
