"""
Bounded ReAct loop: reason, act, observe, repeat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Callable

from friday.tools.tool_executor import ToolExecutor


Reasoner = Callable[[str, list[dict[str, Any]]], dict[str, Any]]


@dataclass
class ReActResult:
    final_answer: str
    iterations: int
    trace: list[dict[str, Any]] = field(default_factory=list)
    forced_final: bool = False


class ReActLoop:
    """Runs a bounded ReAct cycle with structured observations."""

    def __init__(
        self,
        tool_executor: ToolExecutor | None = None,
        reasoner: Reasoner | None = None,
        max_iterations: int = 5,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.tool_executor = tool_executor or ToolExecutor()
        self.reasoner = reasoner or self._default_reasoner
        self.max_iterations = max_iterations

    def run(self, prompt: str, *, memory: Any | None = None) -> ReActResult:
        if not prompt or not prompt.strip():
            raise ValueError("ReActLoop requires a non-empty prompt")

        observations: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        retry_count: dict[str, int] = {}

        for iteration in range(1, self.max_iterations + 1):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self.reason, prompt, observations)
            try:
                thought = future.result(timeout=30.0)
            except TimeoutError:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                return ReActResult(
                    final_answer=self._force_final(prompt, observations),
                    iterations=iteration,
                    trace=trace,
                    forced_final=True,
                )
            finally:
                if future.done():
                    executor.shutdown(wait=False, cancel_futures=True)
            trace.append({"phase": "reason", "iteration": iteration, "thought": thought})

            if thought.get("final") or not thought.get("tool"):
                return ReActResult(
                    final_answer=str(thought.get("answer") or thought.get("reason") or ""),
                    iterations=iteration,
                    trace=trace,
                    forced_final=False,
                )

            tool_name = str(thought["tool"])
            query = str(thought.get("query") or prompt)
            action = self.act(tool_name, query, memory=memory)
            trace.append({"phase": "act", "iteration": iteration, "action": action})
            observation = self.observe(action)
            observations.append(observation)
            trace.append({"phase": "observe", "iteration": iteration, "observation": observation})

            if not observation["success"]:
                retry_count[tool_name] = retry_count.get(tool_name, 0) + 1
                if retry_count[tool_name] > 1:
                    return ReActResult(
                        final_answer=f"Tool {tool_name} failed after retry: {observation['result']}",
                        iterations=iteration,
                        trace=trace,
                        forced_final=True,
                    )

        return ReActResult(
            final_answer=self._force_final(prompt, observations),
            iterations=self.max_iterations,
            trace=trace,
            forced_final=True,
        )

    def reason(self, prompt: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
        result = self.reasoner(prompt, observations)
        if not isinstance(result, dict):
            raise TypeError("reasoner must return a dict")
        return result

    def act(self, tool: str, query: str, *, memory: Any | None = None) -> dict[str, Any]:
        return self.tool_executor.execute(tool, query, memory=memory)

    def observe(self, tool_result: dict[str, Any]) -> dict[str, Any]:
        required = {"tool", "query", "result", "success"}
        if not required.issubset(tool_result):
            raise ValueError(f"Invalid tool observation: {tool_result!r}")
        return tool_result

    def _default_reasoner(self, prompt: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
        if observations:
            latest = observations[-1]
            if latest["success"]:
                return {
                    "final": True,
                    "answer": f"Observed {latest['tool']} result: {latest['result']}",
                }
            return {"final": False, "reason": "Retry failed tool once.", "tool": latest["tool"], "query": latest["query"]}

        lowered = prompt.lower()
        if any(token in lowered for token in ("latest", "current", "today", "weather", "news")):
            return {"final": False, "reason": "Need current data.", "tool": "web_search", "query": prompt}
        if any(token in lowered for token in ("remember", "recall", "previous")):
            return {"final": False, "reason": "Need memory lookup.", "tool": "knowledge_base", "query": prompt}
        if any(token in lowered for token in ("calculate", "compute")):
            return {"final": False, "reason": "Need computation.", "tool": "code_exec", "query": prompt}
        return {"final": True, "answer": prompt}

    @staticmethod
    def _force_final(prompt: str, observations: list[dict[str, Any]]) -> str:
        if not observations:
            return prompt
        successful = [item for item in observations if item.get("success")]
        if successful:
            return str(successful[-1]["result"])
        return f"Unable to complete with tools. Last observation: {observations[-1]['result']}"
