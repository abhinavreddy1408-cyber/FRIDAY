"""
F.R.I.D.A.Y. Action Planner Module
Bridges the gap between raw user intent and concrete execution decisions.

The planner sits between the engine's input (transcribed text) and its
output (action execution or LLM query). It is responsible for:

  - Intent classification (local action vs LLM vs goal vs system command)
  - Context-aware routing with confidence scoring
  - Conversation follow-up detection
  - Shortcut resolution before routing
  - Proactive suggestion generation
  - Command history for undo/retry support

The planner does NOT execute anything — it returns a typed decision that
the engine acts upon. This keeps routing logic fully testable.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from friday.config import config

if TYPE_CHECKING:
    from friday.learning import LearningEngine
    from friday.llm import LLMBrain

logger = logging.getLogger(__name__)


# ── Intent Classification ─────────────────────────────────────────────────────


class Intent(Enum):
    """
    Classified intent of a user utterance.

    Values determine which execution path the engine follows.
    """

    LOCAL_ACTION = auto()  # Handled by ActionRouter instantly
    LLM_QUERY = auto()  # Routed to LLM for reasoning/generation
    GOAL_CREATE = auto()  # User wants to plan a multi-step goal
    GOAL_MANAGE = auto()  # Query/control an existing goal
    SYSTEM_COMMAND = auto()  # Friday-internal command (stop, sleep, etc.)
    FOLLOW_UP = auto()  # Continuation of the current conversation
    CLARIFICATION = auto()  # Friday needs more information
    UNKNOWN = auto()  # Cannot be classified confidently


class RoutingDecision(Enum):
    """Concrete routing outcome returned to the engine."""

    EXECUTE_ACTION = auto()
    QUERY_LLM = auto()
    CREATE_GOAL = auto()
    MANAGE_GOAL = auto()
    HANDLE_SYSTEM = auto()
    ASK_CLARIFY = auto()
    DELEGATE = auto()
    IGNORE = auto()


# ── Decision Models ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlannerDecision:
    """
    Immutable routing decision returned by the planner.

    Attributes:
        routing:     What the engine should do with this input.
        intent:      Classified user intent.
        text:        The (possibly transformed) text to act on.
        confidence:  Classification confidence (0.0-1.0).
        context:     Additional data for the executor (e.g. goal_id, params).
        rationale:   Why this decision was made (for logging/debugging).
        duration:    Time taken to produce this decision in seconds.
    """

    routing: RoutingDecision
    intent: Intent
    text: str
    confidence: float
    context: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    duration: float = 0.0

    @property
    def is_actionable(self) -> bool:
        return self.routing != RoutingDecision.IGNORE

    @property
    def needs_llm(self) -> bool:
        return self.routing in (
            RoutingDecision.QUERY_LLM,
            RoutingDecision.CREATE_GOAL,
            RoutingDecision.DELEGATE,
        )

    def __str__(self) -> str:
        return (
            f"[{self.routing.name}] intent={self.intent.name} "
            f"conf={self.confidence:.0%} text='{self.text[:40]}'"
        )


@dataclass(frozen=True)
class PlanStep:
    """One Execution Chain agent activation."""

    agent: str
    task_category: str
    ensemble_type: str | None
    prompt: str
    dependencies: tuple[str, ...] = ()
    read_only: bool = True


@dataclass(frozen=True)
class ExecutionPlan:
    """
    Routing Chain output consumed by the Execution Chain.

    The plan enforces the PDF constraints: max three parallel agents, read-only
    agents may run in parallel, and mutating agents carry explicit dependencies.
    """

    steps: tuple[PlanStep, ...]
    max_concurrency: int = 3
    local_bypass: bool = False
    routing_models: tuple[str, ...] = ("ultrafast_8b", "action_planner")
    rationale: str = ""

    @property
    def has_agents(self) -> bool:
        return bool(self.steps)


@dataclass
class CommandRecord:
    """
    Record of a dispatched command for undo/retry tracking.

    Attributes:
        text:      Original command text.
        decision:  The planner decision that was made.
        timestamp: Unix timestamp of dispatch.
        result:    Outcome string (populated after execution).
    """

    text: str
    decision: PlannerDecision
    timestamp: float = field(default_factory=time.time)
    result: str = ""


# ── Intent Pattern Registry ───────────────────────────────────────────────────


@dataclass
class _IntentPattern:
    """
    A pattern-to-intent mapping with an associated confidence score.
    """

    pattern: re.Pattern
    intent: Intent
    routing: RoutingDecision
    confidence: float


def _compile_patterns(
    entries: list[tuple[str, Intent, RoutingDecision, float]],
) -> list[_IntentPattern]:
    """Compile raw pattern tuples into _IntentPattern objects."""
    return [
        _IntentPattern(
            pattern=re.compile(p, re.IGNORECASE),
            intent=i,
            routing=r,
            confidence=c,
        )
        for p, i, r, c in entries
    ]


# Pattern registry: (regex, intent, routing, confidence)
_PATTERNS: list[_IntentPattern] = _compile_patterns(
    [
        # ── System commands ──────────────────────────────────────────────────────
        (
            r"\b(stop|shut ?down friday|go ?(offline|to sleep)|exit|quit)\b",
            Intent.SYSTEM_COMMAND,
            RoutingDecision.HANDLE_SYSTEM,
            0.95,
        ),
        (
            r"\b(pause|resume|restart friday)\b",
            Intent.SYSTEM_COMMAND,
            RoutingDecision.HANDLE_SYSTEM,
            0.90,
        ),
        (
            r"\b(what can you do|help|list (commands?|actions?))\b",
            Intent.SYSTEM_COMMAND,
            RoutingDecision.HANDLE_SYSTEM,
            0.85,
        ),
        (
            r"\b(status|how are you|are you (there|online|working))\b",
            Intent.SYSTEM_COMMAND,
            RoutingDecision.HANDLE_SYSTEM,
            0.80,
        ),
        # ── Goal management ──────────────────────────────────────────────────────
        (
            r"\b(plan|create (a )?goal|set up a goal|help me (plan|do|accomplish|achieve))\b",
            Intent.GOAL_CREATE,
            RoutingDecision.CREATE_GOAL,
            0.88,
        ),
        (
            r"\b(goal (status|progress|update)|how('s| is) (my|the) goal)\b",
            Intent.GOAL_MANAGE,
            RoutingDecision.MANAGE_GOAL,
            0.88,
        ),
        (
            r"\b(pause|resume|cancel|stop) (the |my )?(current )?goal\b",
            Intent.GOAL_MANAGE,
            RoutingDecision.MANAGE_GOAL,
            0.90,
        ),
        (
            r"\b(list|show|what are) (my |the )?(goals?|plans?)\b",
            Intent.GOAL_MANAGE,
            RoutingDecision.MANAGE_GOAL,
            0.85,
        ),
        # ── Local actions ────────────────────────────────────────────────────────
        (
            r"\b(open|launch|start|run|close|quit|exit|kill)\s+\w+",
            Intent.LOCAL_ACTION,
            RoutingDecision.EXECUTE_ACTION,
            0.82,
        ),
        (
            r"\b(volume (up|down)|mute|unmute|play|pause|next|previous|skip)\b",
            Intent.LOCAL_ACTION,
            RoutingDecision.EXECUTE_ACTION,
            0.90,
        ),
        (
            r"\b(buy|purchase|recommend|which is better|best|"
            r"compare|review|price of|cost of|specs of|"
            r"should i get|worth buying|alternatives to)\b.+",
            Intent.LOCAL_ACTION,
            RoutingDecision.DELEGATE,
            0.87,
        ),
        (
            r"\b(buy|purchase|order)\s+.+",
            Intent.LOCAL_ACTION,
            RoutingDecision.DELEGATE,
            0.90,
        ),
        (
            r"\b(best|top|cheapest)\s+\w+\s+(for|under|below|above)\b.+",
            Intent.LOCAL_ACTION,
            RoutingDecision.DELEGATE,
            0.88,
        ),
        (
            r"\b(compare|review|price|specs?)\s+.+\b(of|for|under|vs|between)\b.+",
            Intent.LOCAL_ACTION,
            RoutingDecision.DELEGATE,
            0.87,
        ),
        (
            r"\b(recommend)\s+(?:a\s+)?(?:laptop|phone|tablet|pc|computer|headphone|monitor|camera|tv|watch)\b.+",
            Intent.LOCAL_ACTION,
            RoutingDecision.DELEGATE,
            0.88,
        ),
        (
            r"\b(what (time|day|date) is it|current time|today's date)\b",
            Intent.LOCAL_ACTION,
            RoutingDecision.EXECUTE_ACTION,
            0.92,
        ),
        (
            r"\b(battery|cpu|ram|memory|system) (status|usage|level|info)\b",
            Intent.LOCAL_ACTION,
            RoutingDecision.EXECUTE_ACTION,
            0.90,
        ),
        (
            r"\b(take a? screenshot|capture (the )?screen)\b",
            Intent.LOCAL_ACTION,
            RoutingDecision.EXECUTE_ACTION,
            0.93,
        ),
        (
            r"\b(shutdown|restart|reboot|sleep|hibernate|lock)\b",
            Intent.LOCAL_ACTION,
            RoutingDecision.EXECUTE_ACTION,
            0.88,
        ),
        (
            r"\b(play|search|open|find)\s+.+\s+on (youtube|spotify)\b",
            Intent.LOCAL_ACTION,
            RoutingDecision.EXECUTE_ACTION,
            0.90,
        )
    ]
)


# ── Context Tracker ───────────────────────────────────────────────────────────


class _ConversationContext:
    """
    Tracks the active conversation state for follow-up detection.

    A follow-up is when the user speaks again within CONVERSATION_TIMEOUT
    seconds without saying the wake word — the context window allows
    Friday to treat it as a continuation of the last exchange.
    """

    def __init__(self, timeout: int | None = None) -> None:
        self._timeout: int = timeout if timeout is not None else config.system.conv_timeout
        self._last_at: float = 0.0
        self._last_intent: Intent | None = None
        self._last_topic: str = ""
        self._turn_count: int = 0

    def record(self, text: str, intent: Intent) -> None:
        """Record a completed interaction turn."""
        self._last_at = time.time()
        self._last_intent = intent
        self._last_topic = self._extract_topic(text)
        self._turn_count += 1

    def is_follow_up(self) -> bool:
        """True if we are within the conversation timeout window."""
        if self._last_at == 0.0:
            return False
        return (time.time() - self._last_at) < self._timeout

    def reset(self) -> None:
        """Clear the conversation context (e.g. on wake word re-detection)."""
        self._last_at = 0.0
        self._last_intent = None
        self._last_topic = ""
        self._turn_count = 0

    @property
    def last_intent(self) -> Intent | None:
        return self._last_intent

    @property
    def last_topic(self) -> str:
        return self._last_topic

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def time_since_last(self) -> float:
        """Seconds since the last interaction."""
        if self._last_at == 0.0:
            return float("inf")
        return time.time() - self._last_at

    @staticmethod
    def _extract_topic(text: str) -> str:
        """
        Extract a short topic identifier from the command text.
        Used for context continuity in follow-up detection.
        """
        words = text.lower().split()
        fillers = {
            "can",
            "you",
            "please",
            "the",
            "a",
            "an",
            "it",
            "this",
            "that",
            "is",
            "are",
            "what",
            "how",
            "why",
            "when",
            "where",
        }
        topic_words = [w for w in words if w not in fillers]
        return " ".join(topic_words[:4])


# ── Proactive Suggestion Engine ───────────────────────────────────────────────


class _ProactiveSuggester:
    """
    Generates contextual suggestions based on usage patterns and current state.
    Suggestions are surfaced by the engine as optional follow-up prompts.
    """

    # Minimum time between suggestions (seconds) — prevents suggestion spam
    _COOLDOWN = 300

    def __init__(self) -> None:
        self._last_suggestion_at: float = 0.0

    def suggest(
        self,
        text: str,
        intent: Intent,
        learning: LearningEngine | None = None,
    ) -> str | None:
        """
        Generate an optional contextual suggestion after command execution.

        Args:
            text:     The executed command text.
            intent:   The classified intent.
            learning: LearningEngine for usage pattern access.

        Returns:
            A suggestion string for TTS, or None if no suggestion is warranted.
        """
        if not self._cooldown_passed():
            return None

        suggestion = self._suggest_from_pattern(text, intent) or self._suggest_from_learning(
            text, learning
        )

        if suggestion:
            self._last_suggestion_at = time.time()

        return suggestion

    def _cooldown_passed(self) -> bool:
        return (time.time() - self._last_suggestion_at) > self._COOLDOWN

    @staticmethod
    def _suggest_from_pattern(text: str, intent: Intent) -> str | None:
        """Return a hardcoded contextual suggestion based on the command."""
        text_lower = text.lower()

        suggestions: list[tuple[str, str]] = [
            ("open chrome", "Want me to search for something while Chrome loads?"),
            ("open vscode", "Should I open the last project you were working on?"),
            ("search for", "Want me to open one of the top results?"),
            ("screenshot", "Screenshot saved. Want me to open the screenshots folder?"),
            ("volume up", "Music too loud? I can set a specific volume level."),
            ("battery", "Battery is low. Want me to switch to power-save mode?"),
        ]

        for trigger, suggestion in suggestions:
            if trigger in text_lower:
                return suggestion

        return None

    @staticmethod
    def _suggest_from_learning(
        text: str,
        learning: LearningEngine | None,
    ) -> str | None:
        """
        Generate a shortcut suggestion if the command is frequently used
        and doesn't have a shortcut yet.
        """
        if not learning:
            return None

        candidates = learning.suggest_shortcuts()
        for full_command, trigger in candidates:
            if full_command in text.lower():
                return (
                    f"You use '{full_command}' often. "
                    f"Want me to register '{trigger}' as a shortcut?"
                )

        return None


_READ_ONLY_CATEGORIES = frozenset(
    {
        "search",
        "reasoning",
        "retrieval",
        "file_analysis",
    }
)

_AGENT_BY_CATEGORY = {
    "coding": "CODER",
    "reasoning": "REASONING",
    "search": "RESEARCH",
    "brainstorming": "BRAINSTORMING",
    "document": "PRODUCTIVITY",
    "productivity": "PRODUCTIVITY",
    "media": "MEDIA",
    "retrieval": "RETRIEVAL",
    "file_analysis": "RETRIEVAL",
    "general": "REASONING",
}

_ENSEMBLE_BY_CATEGORY = {
    "coding": "coding",
    "reasoning": "reasoning",
    "search": "search",
    "brainstorming": "brainstorming",
    "document": "document",
    "productivity": "document",
    "media": None,
    "retrieval": "file_analysis",
    "file_analysis": "file_analysis",
    "general": None,
}


def _agent_for_category(category: str) -> str:
    return _AGENT_BY_CATEGORY.get(category, "REASONING")


def _ensemble_for_category(category: str) -> str | None:
    return _ENSEMBLE_BY_CATEGORY.get(category, "reasoning")


# ── Planner ───────────────────────────────────────────────────────────────────


class ActionPlanner:
    """
    Public interface for intent classification and routing decisions.

    The engine calls `decide()` with every transcribed utterance and acts
    on the returned PlannerDecision. The planner maintains conversation
    context across multiple turns within the timeout window.

    Usage:
        planner = ActionPlanner(learning=learning_engine)

        decision = planner.decide("open chrome")
        # PlannerDecision(routing=EXECUTE_ACTION, intent=LOCAL_ACTION, ...)

        decision = planner.decide("and search for machine learning")
        # PlannerDecision(routing=QUERY_LLM, intent=FOLLOW_UP, ...)
    """

    # Minimum confidence to commit to a classification
    _CONFIDENCE_THRESHOLD = 0.60

    def __init__(
        self,
        learning: LearningEngine | None = None,
        llm: LLMBrain | None = None,
        timeout: int | None = None,
    ) -> None:
        self._learning = learning
        self._llm = llm
        self._context = _ConversationContext(timeout=timeout)
        self._suggester = _ProactiveSuggester()
        self._history: list[CommandRecord] = []
        self._max_history = 50

    def classify_intent(self, text: str) -> str:
        """Compatibility wrapper: classify intent into command, goal, question, or chitchat."""
        text = text.lower()
        if (
            "apply for internships" in text
            or "find a job" in text
            or "goal" in text
            or "plan" in text
        ):
            return "goal"
        if "what" in text or "how" in text or "why" in text or "weather" in text:
            return "question"
        if (
            "good morning" in text
            or "hello" in text
            or "good afternoon" in text
            or "hi" in text
            or "hey" in text
        ):
            return "chitchat"
        return "command"

    # ── Core Decision ─────────────────────────────────────────────────────────

    def decide(self, text: str) -> PlannerDecision:
        """
        Classify a user utterance and return an actionable routing decision.

        Processing pipeline:
          1. Normalise text
          2. Check for shortcut expansion
          3. Check for follow-up context
          4. Pattern-match intent
          5. Apply confidence threshold
          6. Record context and return decision

        Args:
            text: Normalised transcript from STT.

        Returns:
            PlannerDecision describing what the engine should do.
        """
        if not text or not text.strip():
            return PlannerDecision(
                routing=RoutingDecision.IGNORE,
                intent=Intent.UNKNOWN,
                text="",
                confidence=0.0,
                rationale="Empty input.",
            )

        # Step 0.5: Filter redundant wake-word-only input
        if text.strip().lower() == "friday":
            return PlannerDecision(
                routing=RoutingDecision.IGNORE,
                intent=Intent.UNKNOWN,
                text=text,
                confidence=1.0,
                rationale="Input is only the wake word. Ignoring to avoid redundant processing.",
            )

        start = time.perf_counter()
        text = self._normalise(text)

        # Step 1: Shortcut resolution
        expanded, was_shortcut = self._resolve_shortcut(text)
        if was_shortcut:
            logger.debug("[Planner] Shortcut expanded: '%s' → '%s'", text, expanded)
            text = expanded

        # Step 2: Follow-up detection
        if not was_shortcut and self._context.is_follow_up():
            decision = self._decide_follow_up(text, start)
            self._record(text, decision)
            return decision

        # Step 3: Pattern-based classification
        decision = self._classify(text, start)

        # Single-pass escalation — use ensemble_hint from Phase 1 TriageResult if available.
        # The engine passes this via context["ensemble_hint"] when delegating.
        if decision.routing == RoutingDecision.QUERY_LLM:
            ensemble_hint = decision.context.get("ensemble_hint")
            if ensemble_hint and ensemble_hint != "null":
                specialist = _agent_for_category(ensemble_hint)
                if specialist and specialist != "REASONING":
                    logger.debug(
                        "[Planner] Escalating to swarm via ensemble_hint: %s → %s",
                        ensemble_hint,
                        specialist,
                    )
                    decision = PlannerDecision(
                        routing=RoutingDecision.DELEGATE,
                        intent=decision.intent,
                        text=text,
                        confidence=0.85,
                        context={
                            "specialist": specialist,
                            "ensemble_hint": ensemble_hint,
                        },
                        rationale=f"Escalated via Phase1 ensemble_hint={ensemble_hint}.",
                        duration=time.perf_counter() - start,
                    )

        self._context.record(text, decision.intent)
        self._record(text, decision)

        logger.debug("[Planner] %s (%.3fs)", decision, time.perf_counter() - start)
        return decision

    def decide_with_ensemble_hint(
        self,
        text: str,
        ensemble_hint: str | None = None,
    ) -> PlannerDecision:
        """
        Classify input and inject Phase 1 ensemble_hint to skip the second LLM call.

        Args:
            text:          Raw user input.
            ensemble_hint: ensemble_type string from Phase1 TriageResult, or None.

        Returns:
            PlannerDecision with DELEGATE routing if hint resolves to a specialist,
            otherwise standard decide() result.
        """
        decision = self.decide(text)

        # If Phase 1 already identified a specialist, use it — no second LLM call
        if (
            decision.routing == RoutingDecision.QUERY_LLM
            and ensemble_hint
            and ensemble_hint not in ("null", "None", "")
        ):
            specialist = _agent_for_category(ensemble_hint)
            if specialist and specialist != "REASONING":
                return PlannerDecision(
                    routing=RoutingDecision.DELEGATE,
                    intent=decision.intent,
                    text=decision.text,
                    confidence=0.87,
                    context={
                        "specialist": specialist,
                        "ensemble_hint": ensemble_hint,
                    },
                    rationale=f"Phase1 hint={ensemble_hint} resolved to {specialist}. Second LLM call skipped.",
                    duration=decision.duration,
                )

        return decision

    def decide_with_suggestion(
        self,
        text: str,
    ) -> tuple[PlannerDecision, str | None]:
        """
        Classify the input and optionally generate a proactive suggestion.

        Args:
            text: Normalised transcript.

        Returns:
            Tuple of (PlannerDecision, optional suggestion string for TTS).
        """
        decision = self.decide(text)
        suggestion = self._suggester.suggest(text, decision.intent, self._learning)
        return decision, suggestion

    def create_execution_plan(
        self,
        structured_input: Any,
        decision: PlannerDecision | None = None,
    ) -> ExecutionPlan:
        """Build the PDF Routing Chain ExecutionPlan from structured input."""
        category = getattr(structured_input, "task_category", "reasoning")
        if decision and decision.routing in (
            RoutingDecision.EXECUTE_ACTION,
            RoutingDecision.HANDLE_SYSTEM,
            RoutingDecision.CREATE_GOAL,
            RoutingDecision.MANAGE_GOAL,
            RoutingDecision.ASK_CLARIFY,
            RoutingDecision.IGNORE,
        ):
            return ExecutionPlan(
                steps=(),
                local_bypass=True,
                rationale=f"Local bypass for {decision.routing.name}.",
            )

        categories = (
            list(
                dict.fromkeys(getattr(structured_input, "metadata", {}).get("mixed_categories", []))
            )
            if category == "mixed_task"
            else [category]
        )
        categories = categories[:3]

        steps: list[PlanStep] = []
        mutating_seen: list[str] = []
        for item in categories:
            agent = _agent_for_category(item)
            ensemble = _ensemble_for_category(item)
            read_only = item in _READ_ONLY_CATEGORIES
            dependencies = tuple(mutating_seen) if not read_only else ()
            prompt = getattr(
                structured_input,
                "structured_prompt",
                getattr(structured_input, "cleaned_transcript", ""),
            )
            steps.append(
                PlanStep(
                    agent=agent,
                    task_category=item,
                    ensemble_type=ensemble,
                    prompt=prompt,
                    dependencies=dependencies,
                    read_only=read_only,
                )
            )
            if not read_only:
                mutating_seen.append(agent)

        return ExecutionPlan(
            steps=tuple(steps),
            max_concurrency=min(3, max(1, len(steps))),
            local_bypass=False,
            rationale=f"Planned {len(steps)} agent(s) for {category}.",
        )

    # ── Context Management ────────────────────────────────────────────────────

    def notify_executed(self, text: str, result: str) -> None:
        """
        Notify the planner that the last decision was executed.
        Updates the command history with the execution result.

        Args:
            text:   The command that was executed.
            result: The execution outcome string.
        """
        if self._history:
            last = self._history[-1]
            if last.text == text:
                self._history[-1] = CommandRecord(
                    text=last.text,
                    decision=last.decision,
                    timestamp=last.timestamp,
                    result=result,
                )

    def reset_context(self) -> None:
        """
        Reset conversation context.
        Call this when a new wake-word activation begins.
        """
        self._context.reset()
        logger.debug("[Planner] Conversation context reset.")

    def get_last_command(self) -> CommandRecord | None:
        """Return the most recently dispatched command record."""
        return self._history[-1] if self._history else None

    def get_retry_decision(self) -> PlannerDecision | None:
        """
        Return a decision to retry the last executed command.

        Returns:
            The last PlannerDecision, or None if no history exists.
        """
        last = self.get_last_command()
        if last:
            logger.debug("[Planner] Retry: '%s'", last.text)
            return last.decision
        return None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def in_conversation(self) -> bool:
        """True if within the active conversation timeout window."""
        return self._context.is_follow_up()

    @property
    def conversation_turn(self) -> int:
        """Number of turns in the current conversation window."""
        return self._context.turn_count

    @property
    def context_topic(self) -> str:
        """Extracted topic of the current conversation."""
        return self._context.last_topic

    @property
    def time_since_last_command(self) -> float:
        """Seconds since the last command was processed."""
        return self._context.time_since_last

    # ── Internal ──────────────────────────────────────────────────────────────

    def _normalise(self, text: str) -> str:
        """
        Lowercase, strip, and collapse whitespace in the input.
        Remove common STT artefacts.
        """
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        # Remove common STT noise tokens
        text = re.sub(r"\b(um|uh|er|hmm|like|you know)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _resolve_shortcut(self, text: str) -> tuple[str, bool]:
        """
        Check if the text matches a registered shortcut trigger.

        Returns:
            (expanded_text, was_shortcut) tuple.
        """
        if not self._learning:
            return text, False

        expansion = self._learning.resolve_shortcut(text)
        if expansion:
            return expansion, True

        return text, False

    def _classify(self, text: str, start: float) -> PlannerDecision:
        """
        Run pattern matching and return the highest-confidence decision.

        Args:
            text:  Normalised input text.
            start: perf_counter timestamp for duration tracking.

        Returns:
            PlannerDecision from the best-matching pattern or fallback.
        """
        best_match: _IntentPattern | None = None
        best_confidence: float = 0.0

        for pattern in _PATTERNS:
            if pattern.pattern.search(text) and pattern.confidence > best_confidence:
                best_match = pattern
                best_confidence = pattern.confidence

        duration = time.perf_counter() - start

        if best_match and best_confidence >= self._CONFIDENCE_THRESHOLD:
            return PlannerDecision(
                routing=best_match.routing,
                intent=best_match.intent,
                text=text,
                confidence=best_confidence,
                rationale=f"Pattern matched: {best_match.pattern.pattern[:50]}",
                duration=duration,
            )

        # Below threshold — default to LLM for resolution
        return PlannerDecision(
            routing=RoutingDecision.QUERY_LLM,
            intent=Intent.LLM_QUERY,
            text=text,
            confidence=0.5,
            rationale="No high-confidence pattern match — escalating to LLM.",
            duration=duration,
        )

    def _decide_follow_up(self, text: str, start: float) -> PlannerDecision:
        """
        Classify a follow-up utterance within an active conversation window.

        Follow-ups are biased toward LLM_QUERY unless they clearly match
        a local action pattern at high confidence.

        Args:
            text:  Normalised follow-up text.
            start: perf_counter timestamp.

        Returns:
            PlannerDecision for the follow-up.
        """
        # Still attempt pattern matching — high-confidence local actions win
        for pattern in _PATTERNS:
            if (
                pattern.intent == Intent.LOCAL_ACTION
                and pattern.confidence >= 0.85
                and pattern.pattern.search(text)
            ):
                return PlannerDecision(
                    routing=RoutingDecision.EXECUTE_ACTION,
                    intent=Intent.LOCAL_ACTION,
                    text=text,
                    confidence=pattern.confidence,
                    rationale="High-confidence local action in follow-up window.",
                    duration=time.perf_counter() - start,
                )

        # Otherwise treat as conversational follow-up
        return PlannerDecision(
            routing=RoutingDecision.QUERY_LLM,
            intent=Intent.FOLLOW_UP,
            text=text,
            confidence=0.75,
            context={"follow_up": True, "topic": self._context.last_topic},
            rationale="Within conversation window — routing to LLM as follow-up.",
            duration=time.perf_counter() - start,
        )

    def _record(self, text: str, decision: PlannerDecision) -> None:
        """Append to command history, capping at max_history."""
        self._history.append(CommandRecord(text=text, decision=decision))
        if len(self._history) > self._max_history:
            self._history.pop(0)


# Alias for backward compatibility / testing
Planner = ActionPlanner
