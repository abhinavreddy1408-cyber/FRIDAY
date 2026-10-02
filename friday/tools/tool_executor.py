"""
Structured tool execution for autonomous reasoning and ReAct loops.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from friday.memory import VectorMemory


class ToolExecutionError(RuntimeError):
    """Raised when a requested tool is unknown or misconfigured."""


ToolResult = dict[str, Any]
ToolHandler = Callable[[str, Any | None], ToolResult]


@dataclass
class ToolExecutor:
    """Executes whitelisted tools and returns structured observations."""

    timeout_seconds: float = 10.0
    calendar_path: Path | str = Path("data/calendar.json")

    def execute(self, tool: str, query: str, *, memory: Any | None = None) -> ToolResult:
        canonical = _canonical_tool(tool)
        handlers: dict[str, ToolHandler] = {
            "web_search": self._web_search,
            "calendar_api": self._calendar_api,
            "code_exec": self._code_exec,
            "knowledge_base": self._knowledge_base,
        }
        handler = handlers.get(canonical)
        if handler is None:
            raise ToolExecutionError(f"Unknown tool: {tool}")

        last: ToolResult | None = None
        for attempt in range(2):
            result = handler(query, memory)
            if not _valid_result(result):
                raise ToolExecutionError(f"Tool {canonical} returned invalid result: {result!r}")
            result["attempt"] = attempt + 1
            last = result
            if result["success"]:
                return result
        if last is None:
            raise ToolExecutionError(f"Tool {canonical} produced no result")
        return last

    def _web_search(self, query: str, memory: Any | None = None) -> ToolResult:
        del memory
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=3):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("href") or item.get("url", ""),
                            "snippet": item.get("body") or item.get("snippet", ""),
                        }
                    )
            return _tool_result("web_search", query, results, bool(results), None if results else "No results found")
        except Exception as exc:
            return _tool_result("web_search", query, None, False, str(exc))

    def _calendar_api(self, query: str, memory: Any | None = None) -> ToolResult:
        del memory
        try:
            request = _parse_calendar_query(query)
            action = request["action"]
            calendar = self._read_calendar()

            if action == "add_event":
                event = request["event"]
                if not event.get("title"):
                    raise ValueError("add_event requires title")
                event.setdefault("id", _next_event_id(calendar["events"]))
                calendar["events"].append(event)
                self._write_calendar(calendar)
                return _tool_result("calendar_api", query, event, True)

            if action == "list_events":
                return _tool_result("calendar_api", query, calendar["events"], True)

            if action == "find_event":
                term = request["term"].strip().lower()
                if not term:
                    raise ValueError("find_event requires search text")
                matches = [
                    event
                    for event in calendar["events"]
                    if term in json.dumps(event, sort_keys=True).lower()
                ]
                return _tool_result("calendar_api", query, matches, True)

            raise ValueError(f"Unsupported calendar action: {action}")
        except Exception as exc:
            return _tool_result("calendar_api", query, None, False, str(exc))

    def _code_exec(self, query: str, memory: Any | None = None) -> ToolResult:
        del memory
        try:
            code = _extract_python_code(query)
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(Path.cwd()),
                check=False,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            if completed.returncode == 0:
                return _tool_result("code_exec", query, stdout, True, stderr or None)
            return _tool_result(
                "code_exec",
                query,
                {"stdout": stdout, "stderr": stderr, "returncode": completed.returncode},
                False,
                stderr or f"Process exited with code {completed.returncode}",
            )
        except Exception as exc:
            return _tool_result("code_exec", query, None, False, str(exc))

    def _knowledge_base(self, query: str, memory: Any | None = None) -> ToolResult:
        try:
            memory_backend = memory if memory is not None else VectorMemory()
            matches = memory_backend.retrieve_relevant(query, top_k=3)
            result = [match["text"] if isinstance(match, dict) and "text" in match else match for match in matches]
            return _tool_result("knowledge_base", query, result, True)
        except Exception as exc:
            return _tool_result("knowledge_base", query, None, False, str(exc))

    def _read_calendar(self) -> dict[str, list[dict[str, Any]]]:
        path = Path(self.calendar_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"events": []}, indent=2), encoding="utf-8")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("events"), list):
            raise ValueError("Calendar file must contain {'events': []}")
        return data

    def _write_calendar(self, calendar: dict[str, list[dict[str, Any]]]) -> None:
        path = Path(self.calendar_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(calendar, indent=2, sort_keys=True), encoding="utf-8")


def _canonical_tool(tool: str) -> str:
    mapping = {
        "calendar": "calendar_api",
        "code_execution": "code_exec",
        "memory_or_file_search": "knowledge_base",
        "memory": "knowledge_base",
        "kb": "knowledge_base",
    }
    token = tool.strip().lower()
    return mapping.get(token, token)


def _valid_result(result: ToolResult) -> bool:
    required = ("tool", "query", "result", "success", "error")
    return all(key in result for key in required) and isinstance(result["success"], bool)


def _tool_result(
    tool: str,
    query: str,
    result: Any,
    success: bool,
    error: str | None = None,
) -> ToolResult:
    return {"tool": tool, "query": query, "result": result, "success": success, "error": error}


def _extract_python_code(query: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", query, flags=re.S | re.I)
    if match:
        return match.group(1).strip()

    expression = query.strip()
    calc = re.search(r"(?:calculate|compute|what is|solve)\s+(.+)", expression, flags=re.I)
    if calc:
        expression = calc.group(1).strip()
    if re.fullmatch(r"[\d\s+\-*/().,%]+", expression):
        return f"print({expression.replace('%', '/100')})"
    return query


def _parse_calendar_query(query: str) -> dict[str, Any]:
    stripped = query.strip()
    if not stripped:
        raise ValueError("Calendar query cannot be empty")

    if stripped.startswith("{"):
        payload = json.loads(stripped)
        action = payload.get("action")
        if action == "add_event":
            event = payload.get("event")
            if not isinstance(event, dict):
                raise ValueError("add_event requires event object")
            return {"action": action, "event": event}
        if action == "list_events":
            return {"action": action}
        if action == "find_event":
            return {"action": action, "term": str(payload.get("term", ""))}
        raise ValueError("Calendar JSON requires action add_event, list_events, or find_event")

    lowered = stripped.lower()
    if lowered in {"list_events", "list events", "calendar list"}:
        return {"action": "list_events"}
    if lowered.startswith("find_event "):
        return {"action": "find_event", "term": stripped.split(" ", 1)[1]}
    if lowered.startswith("find event "):
        return {"action": "find_event", "term": stripped.split(" ", 2)[2]}
    raise ValueError("Calendar query requires add_event, list_events, or find_event")


def _next_event_id(events: list[dict[str, Any]]) -> str:
    existing = {str(event.get("id", "")) for event in events}
    index = len(events) + 1
    while f"event-{index}" in existing:
        index += 1
    return f"event-{index}"
