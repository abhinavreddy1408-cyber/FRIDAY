"""
F.R.I.D.A.Y. Voice-to-Prompt Agent
Transforms raw voice transcripts into model-optimized structured instructions.
Operates under an 85ms execution SLA.
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TASK_CATEGORIES = (
    "coding",
    "reasoning",
    "search",
    "brainstorming",
    "document",
    "productivity",
    "media",
    "retrieval",
    "file_analysis",
    "general",
)


@dataclass
class VoiceToPromptResult:
    original_transcript: str
    cleaned_transcript: str
    intent: str
    target_model: str
    structured_prompt: str
    confidence: float
    entities: dict[str, Any]
    task_category: str = "reasoning"
    ensemble_type: str | None = "reasoning"
    target_agent: str | None = "REASONING"
    priority: int = 4
    needs_clarification: bool = False
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    validated: bool = True


class VoiceToPromptAgent:
    """
    Core Voice-to-Prompt Agent.
    Implements a five-stage processing pipeline:
      Stage 1: Transcript Cleanup (<10ms)
      Stage 2: Intent Extraction (<30ms)
      Stage 3: Task Classification (<20ms)
      Stage 4: Model-Specific Prompt Structuring (<15ms)
      Stage 5: Prompt Validation (<10ms)
    """

    def __init__(
        self,
        llm_classifier: Callable[[str], tuple[str, float, dict[str, Any]]] | None = None,
    ) -> None:
        self._llm_classifier = llm_classifier

    def process(self, transcript: str) -> VoiceToPromptResult:
        start_time = time.perf_counter()

        # --- Stage 1: Transcript Cleanup ---
        cleaned = self._cleanup_transcript(transcript)

        # --- Stage 2: Intent Extraction ---
        intent, confidence, entities = self._extract_intent(cleaned)
        if confidence < 0.7 and self._llm_classifier:
            intent, confidence, entities = self._llm_classifier(cleaned)

        # --- Stage 3: Task Classification ---
        target_model, priority = self._classify_task(intent)
        ensemble_type = self._ensemble_for_intent(intent)
        target_agent = self._agent_for_intent(intent)

        # --- Stage 4: Model-Specific Prompt Structuring ---
        structured = self._structure_prompt(cleaned, target_model, entities, intent)

        # --- Stage 5: Prompt Validation ---
        validated, validated_prompt = self._validate_prompt(structured, target_model)

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Log SLA warning if it exceeds 85ms
        if duration_ms > 85.0:
            logger.warning("[VoiceToPrompt] SLA breached: %.2fms", duration_ms)
        else:
            logger.debug("[VoiceToPrompt] Pipeline completed in %.2fms", duration_ms)

        return VoiceToPromptResult(
            original_transcript=transcript,
            cleaned_transcript=cleaned,
            intent=intent,
            target_model=target_model,
            structured_prompt=validated_prompt,
            confidence=confidence,
            entities=entities,
            task_category=self._task_category(intent),
            ensemble_type=ensemble_type,
            target_agent=target_agent,
            priority=priority,
            needs_clarification=(confidence < 0.75 or not validated),
            metadata={
                "mixed_categories": entities.get("categories", []),
                "model_family": target_model,
            },
            duration_ms=duration_ms,
            validated=validated,
        )

    def _cleanup_transcript(self, text: str) -> str:
        # Remove filler words
        fillers = [
            r"\b(um|uh|er|hmm|like|you know)\b",
        ]
        cleaned = text.strip()
        for f in fillers:
            cleaned = re.sub(f, "", cleaned, flags=re.IGNORECASE)

        # Normalize punctuation and whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Correct common STT errors
        replacements = {
            r"\bfriday\b": "FRIDAY",
            r"\bpython\b": "Python",
            r"\bjavascript\b": "JavaScript",
            r"\byoutube\b": "YouTube",
            r"\bvscode\b": "VSCode",
            r"\bgithub\b": "GitHub",
        }
        for pattern, replacement in replacements.items():
            cleaned = re.compile(pattern, re.IGNORECASE).sub(replacement, cleaned)

        return cleaned

    def _extract_intent(self, text: str) -> tuple[str, float, dict[str, Any]]:
        text_lower = text.lower()
        entities = {}

        detected_categories = self._detect_categories(text_lower)
        mixed_markers = (" and also ", " while you ", " then ", " also ")
        if len(detected_categories) > 1 and any(
            marker in f" {text_lower} " for marker in mixed_markers
        ):
            entities["categories"] = detected_categories[:3]
            return "mixed_task", 0.86, entities

        # Word-boundary based checks for instant local actions
        local_keywords = [
            "open",
            "launch",
            "start",
            "run",
            "close",
            "quit",
            "exit",
            "kill",
            "volume",
            "mute",
            "unmute",
            "play",
            "pause",
            "next",
            "previous",
            "skip",
            "screenshot",
            "battery",
            "cpu",
            "ram",
            "system",
        ]
        if any(re.search(rf"\b{w}\b", text_lower) for w in local_keywords):
            app_match = re.search(
                r"\b(open|launch|start|run|close|quit|exit|kill)\s+(\w+)", text_lower
            )
            if app_match:
                entities["action"] = app_match.group(1)
                entities["target"] = app_match.group(2)
            return "general", 0.90, entities

        # Check for search
        search_keywords = ["search", "google", "look up", "find"]
        if (
            any(re.search(rf"\b{w}\b", text_lower) for w in search_keywords)
            or "search for" in text_lower
        ):
            query_match = re.search(r"\b(search for|google|look up|find)\s+(.+)", text_lower)
            if query_match:
                entities["query"] = query_match.group(2)
            return "search", 0.88, entities

        # Check for coding
        coding_keywords = [
            "code",
            "program",
            "script",
            "function",
            "programming",
            "debug",
            "refactor",
            "compile",
            "python",
            "javascript",
            "coder",
            "coding",
            "develop",
        ]
        if any(re.search(rf"\b{w}\b", text_lower) for w in coding_keywords):
            return "coding", 0.85, entities

        # Check for brainstorming
        brainstorm_keywords = ["brainstorm", "ideas", "suggest", "creative", "brainstorming"]
        if any(re.search(rf"\b{w}\b", text_lower) for w in brainstorm_keywords):
            return "brainstorming", 0.85, entities

        # Check for document creation
        doc_keywords = ["pdf", "document", "docx", "powerpoint", "report", "summarize", "summary"]
        if any(re.search(rf"\b{w}\b", text_lower) for w in doc_keywords):
            return "document", 0.80, entities

        # Check for productivity or goal-oriented work
        productivity_keywords = [
            "apply",
            "job",
            "plan",
            "goal",
            "strategy",
            "accomplish",
            "achieve",
            "calendar",
            "email",
            "note",
            "task",
            "schedule",
        ]
        if (
            any(re.search(rf"\b{w}\b", text_lower) for w in productivity_keywords)
            or "help me" in text_lower
        ):
            return "productivity", 0.88, entities

        # Check for media generation
        media_keywords = ["generate", "image", "drawing", "draw", "picture", "photo"]
        if any(re.search(rf"\b{w}\b", text_lower) for w in media_keywords):
            return "media", 0.80, entities

        retrieval_keywords = ["retrieve", "recall", "remember", "memory", "previous conversation"]
        if any(re.search(rf"\b{w}\b", text_lower) for w in retrieval_keywords):
            return "retrieval", 0.82, entities

        file_keywords = [
            "analyze file",
            "analyse file",
            "read file",
            "extract from",
            "screen",
            "ocr",
        ]
        if any(keyword in text_lower for keyword in file_keywords):
            return "file_analysis", 0.85, entities

        return "reasoning", 0.70, entities

    def _detect_categories(self, text_lower: str) -> list[str]:
        categories: list[str] = []
        keyword_map = {
            "general": [
                "open",
                "launch",
                "start",
                "run",
                "close",
                "quit",
                "exit",
                "kill",
                "volume",
                "mute",
                "unmute",
                "play",
                "pause",
                "next",
                "previous",
                "skip",
                "battery",
                "cpu",
                "ram",
                "system",
            ],
            "search": ["search", "google", "look up", "find", "research"],
            "coding": [
                "code",
                "program",
                "script",
                "function",
                "programming",
                "debug",
                "refactor",
                "compile",
                "python",
                "javascript",
                "coder",
                "coding",
                "develop",
            ],
            "brainstorming": ["brainstorm", "ideas", "suggest", "creative", "brainstorming"],
            "document": ["pdf", "document", "docx", "powerpoint", "report", "summarize", "summary"],
            "productivity": ["calendar", "email", "note", "task", "schedule", "goal", "plan"],
            "media": ["generate image", "image", "drawing", "draw", "picture", "photo"],
            "retrieval": ["retrieve", "recall", "remember", "memory"],
            "file_analysis": [
                "analyze file",
                "analyse file",
                "read file",
                "extract from",
                "document analysis",
            ],
        }
        for category, keywords in keyword_map.items():
            if any(re.search(rf"\b{re.escape(keyword)}\b", text_lower) for keyword in keywords):
                categories.append(category)
        return categories

    def _classify_task(self, intent: str) -> tuple[str, int]:
        model_mapping = {
            "general": ("local", 1),
            "search": ("context_flagship", 2),
            "coding": ("coding_flagship", 3),
            "reasoning": ("reasoning_flagship", 4),
            "brainstorming": ("creative_flagship", 5),
            "document": ("context_flagship", 6),
            "productivity": ("context_flagship", 6),
            "media": ("local", 7),
            "retrieval": ("reasoning_70b", 8),
            "file_analysis": ("reasoning_70b", 8),
            "mixed_task": ("context_flagship", 10),
        }
        return model_mapping.get(intent, ("reasoning_flagship", 4))

    def _ensemble_for_intent(self, intent: str) -> str | None:
        ensemble_mapping = {
            "general": None,
            "search": "search",
            "coding": "coding",
            "reasoning": "reasoning",
            "brainstorming": "brainstorming",
            "document": "document",
            "productivity": "document",
            "media": None,
            "retrieval": "file_analysis",
            "file_analysis": "file_analysis",
            "mixed_task": "reasoning",
        }
        return ensemble_mapping.get(intent, "reasoning")

    def _agent_for_intent(self, intent: str) -> str | None:
        agent_mapping = {
            "search": "RESEARCH",
            "coding": "CODER",
            "reasoning": "REASONING",
            "brainstorming": "BRAINSTORMING",
            "document": "PRODUCTIVITY",
            "productivity": "PRODUCTIVITY",
            "media": "MEDIA",
            "retrieval": "RETRIEVAL",
            "file_analysis": "RETRIEVAL",
            "mixed_task": None,
            "general": None,
        }
        return agent_mapping.get(intent, "REASONING")

    def _task_category(self, intent: str) -> str:
        if intent == "mixed_task":
            return intent
        return intent if intent in TASK_CATEGORIES else "reasoning"

    def _structure_prompt(
        self,
        cleaned_text: str,
        target_model: str,
        entities: dict[str, Any],
        intent: str,
    ) -> str:
        if target_model in {
            "context_flagship",
            "reasoning_flagship",
            "reasoning_70b",
        }:
            return (
                f"[SYSTEM]\n"
                f"You are F.R.I.D.A.Y., an advanced agentic assistant.\n"
                f"[INSTRUCTION]\n"
                f"Handle the request: {cleaned_text}\n"
                f"[CONTEXT]\n"
                f"User intent: {intent}. Entities: {entities}\n"
                f"[CONSTRAINTS]\n"
                f"- Use the selected NVIDIA NIM ensemble for this task\n"
                f"- Respect privacy mode settings\n"
                f"[OUTPUT_FORMAT]\n"
                f"Provide a structured, helpful explanation."
            )
        elif target_model == "creative_flagship":
            return (
                f"system: You are F.R.I.D.A.Y.'s conversational partner. Be friendly, creative, and concise.\n"
                f"user: {cleaned_text}"
            )
        elif target_model == "coding_flagship":
            return (
                f"Task: {cleaned_text}\n"
                f"Instructions:\n"
                f"1. Break down the coding problem step-by-step.\n"
                f"2. Write clean Python code that runs within sandboxed limits.\n"
                f"3. Verify logic and handle potential exceptions."
            )
        else:  # local
            action = entities.get("action", "unknown")
            target = entities.get("target", "unknown")
            return f"Action: {action}\nTarget: {target}"

    def _validate_prompt(self, structured_prompt: str, target_model: str) -> tuple[bool, str]:
        if not structured_prompt:
            return False, "Prompt is empty."

        if target_model in {
            "context_flagship",
            "reasoning_flagship",
            "reasoning_70b",
        }:
            # Verify all sections exist, repair if missing
            sections = [
                "[SYSTEM]",
                "[INSTRUCTION]",
                "[CONTEXT]",
                "[CONSTRAINTS]",
                "[OUTPUT_FORMAT]",
            ]
            repaired_prompt = structured_prompt
            for section in sections:
                if section not in repaired_prompt:
                    repaired_prompt += f"\n\n{section}\nDefault context populated."
            return True, repaired_prompt

        return True, structured_prompt
