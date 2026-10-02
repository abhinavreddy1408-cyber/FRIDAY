"""
F.R.I.D.A.Y. Goal Planner Module
Translates high-level natural language objectives into structured,
executable GoalStep sequences using the LLM as a reasoning engine.

The planner bridges the gap between "help me set up my Python project"
and a concrete, ordered list of actions the GoalEngine can execute.

Responsibilities:
  - Decompose natural language goals into ordered steps
  - Validate and sanitise LLM-generated step plans
  - Suggest appropriate action prefixes for each step
  - Estimate goal complexity and execution time
  - Refine existing goals when the user requests changes
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from friday.goal_engine import GoalPriority

if TYPE_CHECKING:
    from friday.llm import LLMBrain

logger = logging.getLogger(__name__)

_instance: GoalPlanner | None = None
_lock = threading.Lock()


# ── Enums & Constants ─────────────────────────────────────────────────────────


class PlannerStatus(Enum):
    SUCCESS = auto()
    LLM_UNAVAILABLE = auto()
    PARSE_FAILED = auto()
    EMPTY_PLAN = auto()
    VALIDATION_ERROR = auto()
    REFINEMENT_ERROR = auto()


class ComplexityTier(Enum):
    TRIVIAL = auto()  # 1-2 steps, < 1 min
    SIMPLE = auto()  # 3-5 steps, < 5 min
    MODERATE = auto()  # 6-10 steps, < 30 min
    COMPLEX = auto()  # 11-20 steps, < 2 hours
    EPIC = auto()  # 20+ steps, open-ended


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepPlan:
    """
    A single step as returned by the LLM planner before GoalStep creation.

    Attributes:
        description:  Human-readable step description.
        action:       Action string with prefix (e.g. "action:open chrome").
        rationale:    Why this step is needed (optional, for debugging).
        max_retries:  Suggested retry limit.
        estimated_s:  Estimated duration in seconds.
        depends_on:   Index of step this depends on (None = sequential).
    """

    description: str
    action: str
    rationale: str = ""
    max_retries: int = 2
    estimated_s: float = 5.0
    depends_on: int | None = None


@dataclass(frozen=True)
class GoalPlan:
    """
    Complete structured plan returned by the planner.

    Attributes:
        title:          Concise goal title.
        description:    Full goal description.
        steps:          Ordered list of StepPlan objects.
        priority:       Suggested execution priority.
        complexity:     Assessed complexity tier.
        total_estimate: Total estimated duration in seconds.
        status:         Planner outcome.
        raw_llm_output: Raw LLM response for debugging.
    """

    title: str
    description: str
    steps: list[StepPlan]
    priority: GoalPriority
    complexity: ComplexityTier
    total_estimate: float
    status: PlannerStatus
    raw_llm_output: str = ""

    @property
    def success(self) -> bool:
        return self.status == PlannerStatus.SUCCESS

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_goal_steps(self) -> list[dict[str, Any]]:
        """Convert StepPlan list to GoalEngine-compatible step dicts."""
        return [
            {
                "description": s.description,
                "action": s.action,
                "max_retries": s.max_retries,
                "metadata": {
                    "rationale": s.rationale,
                    "estimated_s": s.estimated_s,
                },
            }
            for s in self.steps
        ]

    def summary(self) -> str:
        """Return a natural-language plan summary for TTS."""
        if not self.success:
            return f"Planning failed: {self.status.name}."

        estimate = self._format_duration(self.total_estimate)
        return (
            f"Plan ready: '{self.title}'. "
            f"{self.step_count} steps, estimated {estimate}. "
            f"Complexity: {self.complexity.name.lower()}."
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        if seconds < 3600:
            return f"{seconds / 60:.0f} minutes"
        return f"{seconds / 3600:.1f} hours"


@dataclass
class RefinementResult:
    """
    Result of a plan refinement operation.

    Attributes:
        original_step_count: Steps before refinement.
        new_step_count:      Steps after refinement.
        added_steps:         Descriptions of added steps.
        removed_steps:       Descriptions of removed steps.
        plan:                The refined GoalPlan.
    """

    original_step_count: int
    new_step_count: int
    added_steps: list[str]
    removed_steps: list[str]
    plan: GoalPlan

    @property
    def changed(self) -> bool:
        return self.original_step_count != self.new_step_count

    def summary(self) -> str:
        if not self.changed:
            return "Plan refined with no structural changes."
        delta = self.new_step_count - self.original_step_count
        direction = "added" if delta > 0 else "removed"
        return (
            f"Plan updated: {abs(delta)} step{'s' if abs(delta) != 1 else ''} "
            f"{direction}. Now {self.new_step_count} steps total."
        )


# ── Prompt Builder ────────────────────────────────────────────────────────────


class _PromptBuilder:
    """
    Constructs LLM prompts for goal decomposition and refinement.
    All prompts are versioned here for easy iteration.
    """

    # Known action prefixes Friday can handle
    _ACTION_PREFIXES = (
        "action:",  # Routes to ActionRouter
        "llm:",  # Routes to LLM for a sub-query
        "wait:",  # Pause execution N seconds
        "confirm:",  # Ask user for confirmation before proceeding
        "notify:",  # Announce progress to user via TTS
    )

    _PREFIX_DOCS = "\n".join(
        [
            "  action:<command>  — Execute a local action (open app, search web, etc.)",
            "  llm:<prompt>      — Ask the AI to perform a reasoning or writing task",
            "  wait:<seconds>    — Pause for N seconds",
            "  confirm:<message> — Ask the user to confirm before continuing",
            "  notify:<message>  — Announce progress to the user",
        ]
    )

    def build_decomposition_prompt(self, objective: str) -> str:
        """
        Build the system + user prompt for goal decomposition.

        Args:
            objective: The user's natural language goal.

        Returns:
            Full prompt string ready for LLM submission.
        """
        return f"""You are a precise task planner for an AI assistant called Friday.

Your job is to decompose the user's objective into a structured JSON plan.

AVAILABLE ACTION PREFIXES:
{self._PREFIX_DOCS}

OUTPUT FORMAT — respond with ONLY valid JSON, no explanation:
{{
  "title": "Short goal title (max 60 chars)",
  "description": "One sentence describing the overall goal",
  "priority": "low|normal|high",
  "steps": [
    {{
      "description": "What this step does (human-readable)",
      "action": "action:open chrome",
      "rationale": "Why this step is needed",
      "max_retries": 2,
      "estimated_seconds": 5
    }}
  ]
}}

RULES:
- Decompose into 2-5 steps. Fewer is better when sufficient.
- Each step must have exactly one action with a valid prefix.
- Steps are executed sequentially in order.
- Use notify: steps to keep the user informed on long goals.
- Use confirm: before any destructive or irreversible action.
- estimated_seconds should be realistic (5-300 per step).
- Do not include steps the user didn't ask for.
- Do not explain yourself. Output JSON only.

USER OBJECTIVE:
{objective}"""

    def build_refinement_prompt(
        self,
        original_plan: GoalPlan,
        refinement_request: str,
    ) -> str:
        """
        Build a prompt to refine an existing plan based on user feedback.

        Args:
            original_plan:      The plan to refine.
            refinement_request: The user's modification request.

        Returns:
            Full refinement prompt string.
        """
        steps_json = json.dumps(
            [
                {
                    "description": s.description,
                    "action": s.action,
                    "rationale": s.rationale,
                }
                for s in original_plan.steps
            ],
            indent=2,
        )

        return f"""You are refining an existing task plan for Friday.

CURRENT PLAN: "{original_plan.title}"
STEPS:
{steps_json}

USER MODIFICATION REQUEST:
{refinement_request}

OUTPUT FORMAT — respond with ONLY valid JSON using the same schema as the original.
Apply the user's requested changes precisely. Do not add unrequested changes.
Maintain all existing steps that are not affected by the request.

Available action prefixes: action:, llm:, wait:, confirm:, notify:"""

    def build_complexity_prompt(self, objective: str) -> str:
        """
        Build a prompt to assess goal complexity before full decomposition.

        Args:
            objective: The user's natural language goal.

        Returns:
            Complexity assessment prompt.
        """
        return f"""Assess the complexity of the following task in one word.

Reply with ONLY one of: trivial, simple, moderate, complex, epic

Task: {objective}"""


# ── Response Parser ───────────────────────────────────────────────────────────


class _ResponseParser:
    """
    Extracts and validates structured GoalPlan data from LLM responses.
    Handles malformed JSON, missing fields, and invalid action prefixes.
    """

    _VALID_PRIORITIES = {"low", "normal", "high"}  # noqa: RUF012
    _VALID_PREFIXES = {"action:", "llm:", "wait:", "confirm:", "notify:"}  # noqa: RUF012

    def parse(self, raw: str, objective: str) -> GoalPlan:
        """
        Parse an LLM response into a GoalPlan.

        Args:
            raw:       Raw LLM response text.
            objective: Original user objective (for fallback title).

        Returns:
            GoalPlan with SUCCESS status on valid parse, error status otherwise.
        """
        data = self._extract_json(raw)
        if data is None:
            logger.warning("[Planner] JSON extraction failed.")
            return self._error_plan(PlannerStatus.PARSE_FAILED, raw)

        try:
            return self._build_plan(data, raw)
        except Exception as exc:
            logger.warning("[Planner] Plan construction failed: %s", exc)
            return self._error_plan(PlannerStatus.VALIDATION_ERROR, raw)

    def _extract_json(self, text: str) -> dict | None:
        """
        Extract a JSON object from text that may contain surrounding prose.
        Handles markdown code fences and raw JSON responses.
        """
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find the first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _build_plan(self, data: dict, raw: str) -> GoalPlan:
        """Construct and validate a GoalPlan from parsed JSON data."""
        title = str(data.get("title", "Unnamed Goal"))[:60]
        description = str(data.get("description", ""))
        priority = self._parse_priority(data.get("priority", "normal"))
        raw_steps = data.get("steps", [])

        if not raw_steps:
            return self._error_plan(PlannerStatus.EMPTY_PLAN, raw)

        steps: list[StepPlan] = []
        for i, s in enumerate(raw_steps):
            step = self._parse_step(s, index=i)
            if step:
                steps.append(step)

        if not steps:
            return self._error_plan(PlannerStatus.EMPTY_PLAN, raw)

        total_estimate = sum(s.estimated_s for s in steps)
        complexity = self._assess_complexity(len(steps), total_estimate)

        return GoalPlan(
            title=title,
            description=description,
            steps=steps,
            priority=priority,
            complexity=complexity,
            total_estimate=total_estimate,
            status=PlannerStatus.SUCCESS,
            raw_llm_output=raw,
        )

    def _parse_step(self, data: Any, index: int) -> StepPlan | None:
        """
        Parse and validate a single step dict.
        Returns None if the step is unrecoverable.
        """
        if not isinstance(data, dict):
            return None

        description = str(data.get("description", f"Step {index + 1}")).strip()
        action = str(data.get("action", "")).strip()

        if not action:
            logger.debug("[Planner] Step %d has no action — skipping.", index)
            return None

        # Validate action prefix
        action = self._sanitise_action(action)

        return StepPlan(
            description=description,
            action=action,
            rationale=str(data.get("rationale", "")),
            max_retries=max(0, min(int(data.get("max_retries", 2)), 5)),
            estimated_s=max(1.0, float(data.get("estimated_seconds", 5.0))),
        )

    def _sanitise_action(self, action: str) -> str:
        """
        Ensure an action string has a valid prefix.
        Falls back to 'action:' if the prefix is unrecognised.
        """
        for prefix in self._VALID_PREFIXES:
            if action.lower().startswith(prefix):
                return action

        # No valid prefix — assume it's a local action
        logger.debug("[Planner] Action '%s' has no valid prefix — prepending 'action:'.", action)
        return f"action:{action}"

    def _parse_priority(self, raw: str) -> GoalPriority:
        """Parse priority string to GoalPriority enum with fallback."""
        val = str(raw).lower().strip()
        if val not in self._VALID_PRIORITIES:
            return GoalPriority.NORMAL
        return GoalPriority(val)

    @staticmethod
    def _assess_complexity(step_count: int, total_seconds: float) -> ComplexityTier:
        """Derive complexity tier from step count and total estimated time."""
        if step_count <= 2 and total_seconds < 60:
            return ComplexityTier.TRIVIAL
        if step_count <= 5 and total_seconds < 300:
            return ComplexityTier.SIMPLE
        if step_count <= 10 and total_seconds < 1800:
            return ComplexityTier.MODERATE
        if step_count <= 20:
            return ComplexityTier.COMPLEX
        return ComplexityTier.EPIC

    @staticmethod
    def _error_plan(status: PlannerStatus, raw: str) -> GoalPlan:
        return GoalPlan(
            title="",
            description="",
            steps=[],
            priority=GoalPriority.NORMAL,
            complexity=ComplexityTier.TRIVIAL,
            total_estimate=0.0,
            status=status,
            raw_llm_output=raw,
        )


# ── Goal Planner ──────────────────────────────────────────────────────────────


class GoalPlanner:
    """
    Public interface for AI-assisted goal decomposition and refinement.

    Uses the LLM to convert natural language objectives into structured
    GoalPlan objects that the GoalEngine can execute directly.

    Usage:
        planner = GoalPlanner(llm=brain)
        plan = planner.plan("Help me set up a Python project in VS Code")

        if plan.success:
            print(plan.summary())
            goal_id = goal_engine.create_goal(
                title=plan.title,
                description=plan.description,
                steps=plan.to_goal_steps(),
                priority=plan.priority,
            )
            goal_engine.start_goal(goal_id)
    """

    def __init__(self, llm: LLMBrain) -> None:
        self._llm = llm
        self._builder = _PromptBuilder()
        self._parser = _ResponseParser()
        self._ready = False

        # Cache last plan for refinement
        self._last_plan: GoalPlan | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Verify the LLM is ready for planning.

        Returns:
            True if the planner is operational.
        """
        if not self._llm.is_ready:
            logger.error("[Planner] LLM is not ready — planner unavailable.")
            return False

        self._ready = True
        logger.info("[Planner] Ready.")
        return True

    # ── Core Planning ─────────────────────────────────────────────────────────

    def plan(
        self,
        objective: str,
        priority: GoalPriority | None = None,
        max_steps: int = 5,
    ) -> GoalPlan:
        """
        Decompose a natural language objective into an executable GoalPlan.

        Args:
            objective:  The user's high-level goal description.
            priority:   Override the LLM-suggested priority.
            max_steps:  Maximum steps allowed in the plan.

        Returns:
            GoalPlan with SUCCESS status on success, error status otherwise.
        """
        if not self._ready:
            return self._error_plan(PlannerStatus.LLM_UNAVAILABLE)

        objective = objective.strip()
        if not objective:
            return self._error_plan(PlannerStatus.EMPTY_PLAN)

        logger.info("[Planner] Planning objective: '%s'", objective[:80])
        start = time.perf_counter()

        prompt = self._builder.build_decomposition_prompt(objective)
        response = self._llm.query(
            prompt,
            temperature=0.3,  # Low temperature for structured output
            max_tokens=512,
            add_to_history=False,  # Planning shouldn't pollute conversation history
        )

        if not response.success:
            logger.warning("[Planner] LLM query failed: %s", response.status.name)
            return self._error_plan(PlannerStatus.LLM_UNAVAILABLE)

        plan = self._parser.parse(response.text, objective)
        if not plan.success:
            retry_prompt = prompt + "\nReturn a smaller valid JSON plan with no more than 5 steps."
            retry_response = self._llm.query(
                retry_prompt,
                temperature=0.1,
                max_tokens=512,
                add_to_history=False,
            )
            if retry_response.success:
                plan = self._parser.parse(retry_response.text, objective)

        # Apply caller-specified priority override
        if priority and plan.success:
            plan = GoalPlan(
                title=plan.title,
                description=plan.description,
                steps=plan.steps,
                priority=priority,
                complexity=plan.complexity,
                total_estimate=plan.total_estimate,
                status=plan.status,
                raw_llm_output=plan.raw_llm_output,
            )

        # Enforce step limit
        if plan.success and len(plan.steps) > max_steps:
            logger.warning(
                "[Planner] Plan has %d steps — truncating to %d.",
                len(plan.steps),
                max_steps,
            )
            plan = GoalPlan(
                title=plan.title,
                description=plan.description,
                steps=plan.steps[:max_steps],
                priority=plan.priority,
                complexity=plan.complexity,
                total_estimate=sum(s.estimated_s for s in plan.steps[:max_steps]),
                status=plan.status,
                raw_llm_output=plan.raw_llm_output,
            )

        elapsed = time.perf_counter() - start

        if plan.success:
            self._last_plan = plan
            logger.info(
                "[Planner] Plan ready: '%s' | %d steps | %.2fs",
                plan.title,
                plan.step_count,
                elapsed,
            )
        else:
            logger.warning(
                "[Planner] Planning failed (%s) in %.2fs.",
                plan.status.name,
                elapsed,
            )

        return plan

    def refine(
        self,
        refinement_request: str,
        base_plan: GoalPlan | None = None,
    ) -> RefinementResult:
        """
        Refine an existing plan based on the user's modification request.

        Args:
            refinement_request: Natural language description of the change.
            base_plan:          Plan to refine (defaults to last planned goal).

        Returns:
            RefinementResult with the updated plan and change summary.
        """
        target = base_plan or self._last_plan

        if not target or not target.success:
            empty = self._error_plan(PlannerStatus.REFINEMENT_ERROR)
            return RefinementResult(
                original_step_count=0,
                new_step_count=0,
                added_steps=[],
                removed_steps=[],
                plan=empty,
            )

        original_count = target.step_count
        original_descs = {s.description for s in target.steps}

        prompt = self._builder.build_refinement_prompt(target, refinement_request)
        response = self._llm.query(
            prompt,
            temperature=0.2,
            max_tokens=512,
            add_to_history=False,
        )

        if not response.success:
            return RefinementResult(
                original_step_count=original_count,
                new_step_count=original_count,
                added_steps=[],
                removed_steps=[],
                plan=self._error_plan(PlannerStatus.REFINEMENT_ERROR),
            )

        refined_plan = self._parser.parse(response.text, target.title)

        if not refined_plan.success:
            return RefinementResult(
                original_step_count=original_count,
                new_step_count=original_count,
                added_steps=[],
                removed_steps=[],
                plan=refined_plan,
            )

        new_descs = {s.description for s in refined_plan.steps}
        added = list(new_descs - original_descs)
        removed = list(original_descs - new_descs)

        self._last_plan = refined_plan

        return RefinementResult(
            original_step_count=original_count,
            new_step_count=refined_plan.step_count,
            added_steps=added,
            removed_steps=removed,
            plan=refined_plan,
        )

    def assess_complexity(self, objective: str) -> ComplexityTier:
        """
        Quickly assess the complexity of an objective without full decomposition.
        Useful for deciding whether to ask the user for confirmation before planning.

        Args:
            objective: The user's natural language goal.

        Returns:
            ComplexityTier assessment.
        """
        if not self._ready:
            return ComplexityTier.MODERATE

        prompt = self._builder.build_complexity_prompt(objective)
        response = self._llm.query(
            prompt,
            temperature=0.1,
            max_tokens=10,
            add_to_history=False,
        )

        if not response.success:
            return ComplexityTier.MODERATE

        word = response.text.strip().lower().split()[0] if response.text.strip() else ""

        mapping = {
            "trivial": ComplexityTier.TRIVIAL,
            "simple": ComplexityTier.SIMPLE,
            "moderate": ComplexityTier.MODERATE,
            "complex": ComplexityTier.COMPLEX,
            "epic": ComplexityTier.EPIC,
        }
        return mapping.get(word, ComplexityTier.MODERATE)

    def estimate_duration(self, plan: GoalPlan) -> str:
        """
        Return a human-readable duration estimate for a plan.

        Args:
            plan: The plan to estimate.

        Returns:
            Formatted string like "about 3 minutes" or "under 30 seconds".
        """
        secs = plan.total_estimate

        if secs < 30:
            return "under 30 seconds"
        if secs < 90:
            return "about a minute"
        if secs < 600:
            return f"about {secs / 60:.0f} minutes"
        if secs < 3600:
            return f"around {secs / 60:.0f} minutes"
        return f"approximately {secs / 3600:.1f} hours"

    def explain_plan(self, plan: GoalPlan) -> str:
        """
        Generate a natural-language walkthrough of a plan for TTS.

        Args:
            plan: The plan to explain.

        Returns:
            Multi-sentence explanation string.
        """
        if not plan.success:
            return "I wasn't able to create a plan for that."

        lines = [
            f"Here's the plan for '{plan.title}'.",
            f"It has {plan.step_count} steps and should take {self.estimate_duration(plan)}.",
        ]

        # Narrate up to first 5 steps verbally
        for i, step in enumerate(plan.steps[:5], 1):
            lines.append(f"Step {i}: {step.description}.")

        if plan.step_count > 5:
            lines.append(
                f"... and {plan.step_count - 5} more step{'s' if plan.step_count - 5 != 1 else ''}."
            )

        return " ".join(lines)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def last_plan(self) -> GoalPlan | None:
        """The most recently generated or refined plan."""
        return self._last_plan

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _error_plan(status: PlannerStatus) -> GoalPlan:
        return GoalPlan(
            title="",
            description="",
            steps=[],
            priority=GoalPriority.NORMAL,
            complexity=ComplexityTier.TRIVIAL,
            total_estimate=0.0,
            status=status,
        )


def get_goal_planner(llm=None):
    global _instance
    with _lock:
        if _instance is None:
            if llm is None:
                from friday.llm import LLMBrain

                llm = LLMBrain()
            _instance = GoalPlanner(llm=llm)
        return _instance
