"""
Autonomous reasoning helpers for F.R.I.D.A.Y.

This module keeps the first autonomous layer deterministic and inspectable:
it infers intent signals, identifies missing information, pulls relevant
memory, and decides when Friday must ask before acting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_RISKY_ACTIONS = (
    "buy",
    "purchase",
    "order",
    "book",
    "reserve",
    "send",
    "email",
    "delete",
    "remove",
    "cancel",
    "shutdown",
    "restart",
    "reboot",
    "pay",
    "transfer",
)

_SCHEDULE_WORDS = ("meeting", "appointment", "call", "calendar", "schedule")
_VAGUE_TIME_WORDS = ("soon", "later", "sometime", "next week", "next month", "morning")
_VAGUE_TARGET_WORDS = ("it", "that", "this", "them", "there")


@dataclass(frozen=True)
class AutonomousReasoningResult:
    """Structured reasoning state added to ChainContext."""

    inferred_intent: str
    confidence: float
    assumptions: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    clarification_question: str | None = None
    requires_confirmation: bool = False
    risk_level: str = "low"
    relevant_memory: tuple[str, ...] = ()
    tool_hints: tuple[str, ...] = ()
    cot_trace: tuple[str, ...] = ()
    tool_observations: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_clarification(self) -> bool:
        return self.clarification_question is not None


class AutonomousReasoner:
    """
    Deterministic pre-routing reasoning layer.

    It deliberately does not execute tools or call LLMs. The output is used by
    the planner, execution prompt, and delivery layer to keep autonomous
    behavior bounded and debuggable.
    """

    def analyze(
        self,
        transcript: str,
        structured_input: Any = None,
        memory: Any = None,
    ) -> AutonomousReasoningResult:
        cleaned = _clean(transcript)
        category = getattr(structured_input, "task_category", "general") or "general"
        confidence = float(getattr(structured_input, "confidence", 0.5) or 0.5)

        missing = list(self._missing_information(cleaned, category))
        assumptions = list(self._assumptions(cleaned, category))
        risk_level, requires_confirmation = self._risk(cleaned, category)
        relevant_memory = self._recall_memory(cleaned, memory)
        tool_hints = self._tool_hints(cleaned, category)
        cot_trace = generate_cot_trace(
            f"{category}:requires_{'+'.join(tool_hints)}" if tool_hints else category,
            {
                "text": cleaned,
                "category": category,
                "missing_information": missing,
                "assumptions": assumptions,
                "risk_level": risk_level,
                "tool_hints": tool_hints,
            },
        )

        clarification = None
        schedule_missing = bool(missing) and any(word in cleaned for word in _SCHEDULE_WORDS)
        if missing and (schedule_missing or confidence < 0.82 or risk_level in {"medium", "high"}):
            clarification = self._clarification_question(cleaned, category, missing)
        elif requires_confirmation and self._is_external_action(cleaned):
            clarification = "Before I take that action, confirm the exact target and say whether I should proceed."

        inferred_intent = self._intent_label(cleaned, category, tool_hints)
        adjusted_confidence = max(0.0, min(1.0, confidence - (0.12 if missing else 0.0)))

        return AutonomousReasoningResult(
            inferred_intent=inferred_intent,
            confidence=round(adjusted_confidence, 3),
            assumptions=tuple(assumptions),
            missing_information=tuple(missing),
            clarification_question=clarification,
            requires_confirmation=requires_confirmation,
            risk_level=risk_level,
            relevant_memory=tuple(relevant_memory),
            tool_hints=tuple(tool_hints),
            cot_trace=tuple(cot_trace),
            notes=tuple(self._notes(category, missing, relevant_memory)),
        )

    def _missing_information(self, text: str, category: str) -> list[str]:
        missing: list[str] = []

        if any(word in text for word in _SCHEDULE_WORDS):
            if not _has_time_detail(text):
                missing.append("time")
            if not _has_participant_detail(text):
                missing.append("participants")

        if category == "productivity" and any(word in text for word in ("plan", "trip", "travel")):
            if not re.search(r"\b\d+\s*(day|days|week|weeks)\b", text):
                missing.append("duration")
            if not re.search(r"\$\d+|\bunder\s+\d+|\bbudget\b", text):
                missing.append("budget")

        if any(re.search(rf"\b{word}\b", text) for word in _VAGUE_TARGET_WORDS):
            if len(text.split()) <= 6:
                missing.append("target")

        if category == "document" and "improve" in text and len(text.split()) < 8:
            missing.append("source text")

        return _dedupe(missing)

    def _assumptions(self, text: str, category: str) -> list[str]:
        assumptions: list[str] = []
        if category in {"search", "reasoning"} and _needs_current_info(text):
            assumptions.append("The request may require current information.")
        if category == "coding" and "python" not in text and not re.search(r"\b(js|javascript|java|c\+\+|go|rust)\b", text):
            assumptions.append("Use Python unless the user specifies another language.")
        if category == "productivity":
            assumptions.append("Do not perform external actions until required details are clear.")
        return assumptions

    def _risk(self, text: str, category: str) -> tuple[str, bool]:
        has_risky_action = any(re.search(rf"\b{word}\b", text) for word in _RISKY_ACTIONS)
        drafting_only = bool(re.search(r"\b(write|draft|compose|help me write|prepare)\b", text))
        destructive = any(word in text for word in ("delete", "remove", "shutdown", "restart", "reboot", "pay", "transfer"))
        if destructive:
            return "high", True
        if has_risky_action and not drafting_only:
            return "medium", True
        return "low", False

    def _recall_memory(self, text: str, memory: Any) -> list[str]:
        if memory is None:
            return []
        snippets: list[str] = []
        try:
            for entry, distance in memory.recall_relevant(text, limit=3):
                if distance <= 0.55:
                    snippets.append(str(entry.content))
        except Exception:
            snippets = []
        if snippets:
            return snippets[:3]
        try:
            summary = memory.build_context_summary(max_entries=3)
            return [summary] if summary else []
        except Exception:
            return []

    def _tool_hints(self, text: str, category: str) -> list[str]:
        hints: list[str] = []
        if _needs_current_info(text):
            hints.append("web_search")
        if any(word in text for word in _SCHEDULE_WORDS):
            hints.append("calendar")
        if "email" in text or "send" in text:
            hints.append("email")
        if category == "coding":
            hints.append("code_execution")
        if category in {"retrieval", "file_analysis"}:
            hints.append("memory_or_file_search")
        return _dedupe(hints)

    def _intent_label(self, text: str, category: str, tool_hints: list[str]) -> str:
        if tool_hints:
            return f"{category}:requires_{'+'.join(tool_hints)}"
        return category

    def _clarification_question(self, text: str, category: str, missing: list[str]) -> str:
        if any(word in text for word in _SCHEDULE_WORDS):
            if set(missing) >= {"time", "participants"}:
                return "Who should I include, and what exact time should I use?"
            if "time" in missing:
                return "What exact time should I schedule it for?"
            if "participants" in missing:
                return "Who should I include?"
        if category == "document" and "source text" in missing:
            return "Send me the text you want improved, and I will revise it."
        if "target" in missing:
            return "What exactly should I apply that to?"
        return f"I need one detail before I continue: {', '.join(missing)}."

    def _notes(self, category: str, missing: list[str], relevant_memory: list[str]) -> list[str]:
        notes: list[str] = []
        if missing:
            notes.append("Clarification required before execution.")
        if relevant_memory:
            notes.append("Relevant memory is available for prompt grounding.")
        if category == "general":
            notes.append("Keep response brief unless the user asks for depth.")
        return notes

    @staticmethod
    def _is_external_action(text: str) -> bool:
        return bool(
            re.search(
                r"\b(send|email|buy|purchase|order|book|reserve|delete|remove|cancel|pay|transfer|shutdown|restart|reboot)\b",
                text,
            )
        )


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _has_time_detail(text: str) -> bool:
    if re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", text):
        return True
    if re.search(r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text):
        return not any(word in text for word in _VAGUE_TIME_WORDS)
    return False


def _has_participant_detail(text: str) -> bool:
    if re.search(r"\bwith\s+[\w@. -]{2,}", text):
        return True
    return bool(re.search(r"\b(team|everyone|me|myself)\b", text))


def _needs_current_info(text: str) -> bool:
    return bool(
        re.search(
            r"\b(latest|recent|today|current|now|news|weather|stock|price|score|forecast)\b",
            text,
        )
    )


def generate_cot_trace(intent: str, context: dict[str, Any]) -> list[str]:
    """
    Build concise execution-facing reasoning steps.

    The trace is deterministic, does not reveal hidden model internals, and each
    line is intended to be injected with a "thinking:" prefix.
    """
    if not intent or not isinstance(context, dict):
        raise ValueError("generate_cot_trace requires a non-empty intent and context dict")

    text = str(context.get("text", "")).strip()
    category = str(context.get("category") or intent.split(":", 1)[0] or "general")
    missing = [str(item) for item in context.get("missing_information", []) if str(item)]
    assumptions = [str(item) for item in context.get("assumptions", []) if str(item)]
    tools = [str(item) for item in context.get("tool_hints", []) if str(item)]
    risk = str(context.get("risk_level") or "low")

    steps = [
        f"thinking: identify the request as {category} intent",
        f"thinking: preserve the user's original goal: {text}" if text else "thinking: preserve the user's original goal",
    ]
    if missing:
        steps.append("thinking: ask for missing details before execution: " + ", ".join(missing))
    if assumptions:
        steps.append("thinking: use bounded assumptions: " + "; ".join(assumptions))
    if tools:
        steps.append("thinking: gather tool observations from: " + ", ".join(tools))
    if risk != "low":
        steps.append(f"thinking: apply {risk} risk guardrails before action")
    steps.append("thinking: verify the result against the original request before delivery")
    return steps
