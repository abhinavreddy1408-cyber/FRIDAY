import json
import subprocess
import sys
from types import SimpleNamespace

from friday.memory import VectorMemory
from friday.tools.tool_executor import ToolExecutor


class FakeEmbeddingModel:
    def encode(self, text):
        lowered = text.lower()
        return [
            1.0 if any(word in lowered for word in ("friday", "planner", "memory")) else 0.0,
            1.0 if any(word in lowered for word in ("calendar", "event", "meeting")) else 0.0,
            1.0,
        ]


class FakeDDGS:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, max_results):
        assert query == "friday assistant"
        assert max_results == 3
        return [
            {"title": "Friday", "href": "https://example.com/friday", "body": "Assistant"},
            {"title": "Planner", "href": "https://example.com/planner", "body": "ReAct"},
            {"title": "Memory", "href": "https://example.com/memory", "body": "Vector"},
            {"title": "Ignored", "href": "https://example.com/ignored", "body": "Fourth"},
        ][:max_results]


def test_web_search_returns_top_three_results(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckduckgo_search", SimpleNamespace(DDGS=FakeDDGS))

    result = ToolExecutor().execute("web_search", "friday assistant")

    assert result["success"] is True
    assert result["error"] is None
    assert result["result"] == [
        {"title": "Friday", "url": "https://example.com/friday", "snippet": "Assistant"},
        {"title": "Planner", "url": "https://example.com/planner", "snippet": "ReAct"},
        {"title": "Memory", "url": "https://example.com/memory", "snippet": "Vector"},
    ]


def test_code_exec_uses_subprocess_with_timeout(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="42\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ToolExecutor().execute("code_exec", "calculate 40 + 2")

    assert result["success"] is True
    assert result["result"] == "42"
    assert calls[0][1]["timeout"] == 10.0
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True
    assert calls[0][0][0][2] == "-c"


def test_code_exec_failure_returns_stderr(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ToolExecutor().execute("code_exec", "raise SystemExit(1)")

    assert result["success"] is False
    assert result["error"] == "boom"
    assert result["result"]["returncode"] == 1


def test_calendar_api_add_list_and_find_event(tmp_path):
    calendar_path = tmp_path / "calendar.json"
    executor = ToolExecutor(calendar_path=calendar_path)
    event = {"title": "Planning Meeting", "date": "2026-06-02", "time": "10:00"}

    added = executor.execute("calendar_api", json.dumps({"action": "add_event", "event": event}))
    listed = executor.execute("calendar_api", "list_events")
    found = executor.execute("calendar_api", json.dumps({"action": "find_event", "term": "planning"}))

    assert added["success"] is True
    assert added["result"]["id"] == "event-1"
    assert listed["result"][0]["title"] == "Planning Meeting"
    assert found["result"] == listed["result"]
    assert json.loads(calendar_path.read_text(encoding="utf-8"))["events"][0]["title"] == "Planning Meeting"


def test_calendar_api_invalid_action_fails_loudly(tmp_path):
    result = ToolExecutor(calendar_path=tmp_path / "calendar.json").execute(
        "calendar_api",
        "schedule meeting soon",
    )

    assert result["success"] is False
    assert "requires add_event" in result["error"]
    assert result["attempt"] == 2


def test_knowledge_base_retrieves_from_vector_memory():
    memory = VectorMemory(model=FakeEmbeddingModel())
    memory.store("Friday memory retrieves relevant planner notes", tags=["memory"])
    memory.store("Calendar events are stored locally", tags=["calendar"])

    result = ToolExecutor().execute("knowledge_base", "planner memory", memory=memory)

    assert result["success"] is True
    assert result["error"] is None
    assert result["result"][0] == "Friday memory retrieves relevant planner notes"
