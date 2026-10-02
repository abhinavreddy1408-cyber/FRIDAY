"""
Tests for friday.goal_planner — natural-language goal decomposition.

Covers the five explicit scenarios requested:
  1. plan() returns valid GoalPlan on clear input
  2. plan() handles vague input gracefully
  3. explain_plan() returns non-empty string
  4. to_goal_steps() returns correct step count
  5. plan() failure path returns success=False

Plus auxiliary coverage (refine, assess_complexity, estimate_duration,
explain_plan on failed plans, the singleton factory, error-plan path on
bad JSON, _ResponseParser edge cases) so we comfortably exceed 60%.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from friday.goal_engine import GoalPriority
from friday.goal_planner import (
    ComplexityTier,
    GoalPlan,
    GoalPlanner,
    PlannerStatus,
    RefinementResult,
    StepPlan,
    get_goal_planner,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_brain(response_text: str = "", success: bool = True):
    """
    Build a MagicMock stand-in for LLMBrain.

    The planner only reads .is_ready and calls .query(), so those two are
    all we need to stub.
    """
    brain = MagicMock()
    brain.is_ready = True
    response = MagicMock()
    response.text = response_text
    response.success = success
    response.status.name = "SUCCESS" if success else "API_ERROR"
    brain.query.return_value = response
    return brain


def valid_plan_json(
    title: str = "Open Chrome and search",
    n_steps: int = 3,
    priority: str = "high",
) -> str:
    """Return a realistic LLM-emitted JSON plan string."""
    steps = []
    for i in range(n_steps):
        steps.append(
            {
                "description": f"Step {i + 1} description",
                "action": f"action:do thing {i + 1}",
                "rationale": f"because {i + 1}",
                "max_retries": 2,
                "estimated_seconds": 5,
            }
        )
    payload = {
        "title": title,
        "description": "A complete, multi-step goal.",
        "priority": priority,
        "steps": steps,
    }
    return json.dumps(payload)


def make_planner(response_text: str = "", success: bool = True) -> GoalPlanner:
    """Build a ready-to-use GoalPlanner backed by a mocked LLM."""
    brain = make_brain(response_text=response_text, success=success)
    planner = GoalPlanner(llm=brain)
    assert planner.setup() is True
    return planner


# ── 1. plan() returns valid GoalPlan on clear input ───────────────────────────


def test_plan_returns_valid_goal_plan_on_clear_input():
    """Happy path: clear JSON, all fields parseable -> GoalPlan with SUCCESS."""
    planner = make_planner(response_text=valid_plan_json(n_steps=3))

    plan = planner.plan("Open Chrome and search for AI news")

    assert isinstance(plan, GoalPlan)
    assert plan.success is True
    assert plan.status is PlannerStatus.SUCCESS
    assert plan.title == "Open Chrome and search"
    assert plan.description == "A complete, multi-step goal."
    assert plan.priority is GoalPriority.HIGH
    assert plan.step_count == 3
    assert plan.total_estimate == 15.0  # 3 steps * 5s
    assert all(isinstance(s, StepPlan) for s in plan.steps)
    # Each step has a sanitised action
    assert all(s.action.startswith("action:") for s in plan.steps)
    # Last plan is cached for refinement
    assert planner.last_plan is plan


def test_plan_classifies_complexity_trivial_for_short_plan():
    planner = make_planner(response_text=valid_plan_json(n_steps=1))
    plan = planner.plan("quick task")
    assert plan.success
    assert plan.complexity is ComplexityTier.TRIVIAL


def test_plan_classifies_complexity_epic_for_huge_plan():
    """25+ steps -> EPIC tier."""
    planner = make_planner(response_text=valid_plan_json(n_steps=25))
    plan = planner.plan("epic task")
    assert plan.success
    assert plan.complexity is ComplexityTier.EPIC


def test_plan_strips_markdown_fences_from_llm_output():
    """The parser must handle ```json ... ``` wrapped responses."""
    raw = "```json\n" + valid_plan_json(n_steps=2) + "\n```"
    planner = make_planner(response_text=raw)

    plan = planner.plan("anything")

    assert plan.success
    assert plan.step_count == 2


def test_plan_priority_override_is_applied_to_successful_plan():
    planner = make_planner(
        response_text=valid_plan_json(n_steps=2, priority="low")
    )
    plan = planner.plan("low priority task", priority=GoalPriority.HIGH)
    assert plan.success
    assert plan.priority is GoalPriority.HIGH


def test_plan_truncates_to_max_steps():
    """If the LLM returns more steps than max_steps, the plan is truncated."""
    planner = make_planner(response_text=valid_plan_json(n_steps=10))
    plan = planner.plan("big plan", max_steps=4)
    assert plan.success
    assert plan.step_count == 4
    assert plan.total_estimate == 20.0  # 4 * 5s


# ── 2. plan() handles vague input gracefully ──────────────────────────────────


def test_plan_handles_empty_objective_without_calling_llm():
    """An empty / whitespace objective never reaches the LLM."""
    planner = make_planner(response_text=valid_plan_json())
    plan = planner.plan("   ")
    assert plan.success is False
    assert plan.status is PlannerStatus.EMPTY_PLAN
    assert plan.step_count == 0
    planner._llm.query.assert_not_called()


def test_plan_handles_garbage_llm_output():
    """LLM returns prose with no recoverable JSON -> PARSE_FAILED."""
    planner = make_planner(response_text="I am not sure what you mean.")
    plan = planner.plan("set up my dev environment")
    assert plan.success is False
    assert plan.status in (
        PlannerStatus.PARSE_FAILED,
        PlannerStatus.VALIDATION_ERROR,
        PlannerStatus.EMPTY_PLAN,
    )


def test_plan_handles_json_with_no_steps():
    """Valid JSON but no steps -> EMPTY_PLAN."""
    raw = json.dumps({"title": "x", "description": "y", "priority": "low", "steps": []})
    planner = make_planner(response_text=raw)
    plan = planner.plan("anything")
    assert plan.success is False
    assert plan.status is PlannerStatus.EMPTY_PLAN


def test_plan_handles_json_with_steps_missing_actions():
    """A step without an 'action' field is silently skipped (per parser)."""
    raw = json.dumps(
        {
            "title": "mixed",
            "description": "x",
            "steps": [
                {"description": "valid", "action": "action:go"},
                {"description": "broken — no action"},
            ],
        }
    )
    planner = make_planner(response_text=raw)
    plan = planner.plan("anything")
    # The valid step is kept, the broken one is dropped -> SUCCESS with 1 step.
    assert plan.success
    assert plan.step_count == 1
    assert plan.steps[0].description == "valid"


def test_plan_prepends_action_prefix_when_missing():
    """If a step's action has no recognised prefix, 'action:' is prepended."""
    raw = json.dumps(
        {
            "title": "x",
            "description": "y",
            "steps": [
                {"description": "raw command", "action": "open chrome"},
            ],
        }
    )
    planner = make_planner(response_text=raw)
    plan = planner.plan("anything")
    assert plan.success
    assert plan.steps[0].action == "action:open chrome"


def test_plan_clamps_extreme_max_retries():
    """LLM-supplied max_retries is clamped to [0, 5]."""
    raw = json.dumps(
        {
            "title": "x",
            "description": "y",
            "steps": [
                {
                    "description": "step",
                    "action": "action:go",
                    "max_retries": 99,
                }
            ],
        }
    )
    planner = make_planner(response_text=raw)
    plan = planner.plan("anything")
    assert plan.success
    assert plan.steps[0].max_retries == 5  # clamped from 99


def test_plan_falls_back_to_normal_for_invalid_priority():
    """Unknown priority value -> NORMAL."""
    raw = json.dumps(
        {
            "title": "x",
            "description": "y",
            "priority": "ULTRA_HIGH",
            "steps": [{"description": "s", "action": "action:go"}],
        }
    )
    planner = make_planner(response_text=raw)
    plan = planner.plan("anything")
    assert plan.success
    assert plan.priority is GoalPriority.NORMAL


def test_plan_normalises_unknown_objective_to_meaningful_title():
    """Missing title -> 'Unnamed Goal' fallback."""
    raw = json.dumps(
        {"description": "y", "steps": [{"description": "s", "action": "action:go"}]}
    )
    planner = make_planner(response_text=raw)
    plan = planner.plan("anything")
    assert plan.success
    assert plan.title == "Unnamed Goal"


# ── 3. explain_plan() returns non-empty string ────────────────────────────────


def test_explain_plan_returns_non_empty_string_for_successful_plan():
    planner = make_planner(response_text=valid_plan_json(n_steps=3))
    plan = planner.plan("explain this")
    assert plan.success

    explanation = planner.explain_plan(plan)

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "3 steps" in explanation
    # All step descriptions are narrated
    for step in plan.steps:
        assert step.description in explanation


def test_explain_plan_handles_long_plan_with_summary():
    """Plans are capped at five steps for the local planner."""
    planner = make_planner(response_text=valid_plan_json(n_steps=7))
    plan = planner.plan("long task")
    assert plan.success

    explanation = planner.explain_plan(plan)

    assert "It has 5 steps" in explanation
    # First 5 steps narrated explicitly
    assert "Step 1:" in explanation
    assert "Step 5:" in explanation
    assert "Step 6:" not in explanation
    assert "Step 7:" not in explanation


def test_explain_plan_returns_explanation_for_failed_plan():
    """Failed plans get a friendly fallback explanation."""
    planner = make_planner(response_text="not json", success=True)
    plan = planner.plan("anything")
    assert plan.success is False

    explanation = planner.explain_plan(plan)

    assert isinstance(explanation, str)
    assert "wasn't able to create a plan" in explanation


# ── 4. to_goal_steps() returns correct step count ─────────────────────────────


def test_to_goal_steps_returns_correct_count():
    planner = make_planner(response_text=valid_plan_json(n_steps=4))
    plan = planner.plan("convert me")
    assert plan.success
    assert plan.step_count == 4

    goal_steps = plan.to_goal_steps()

    assert isinstance(goal_steps, list)
    assert len(goal_steps) == 4
    for step in goal_steps:
        assert "description" in step
        assert "action" in step
        assert "max_retries" in step
        assert "metadata" in step
        assert "rationale" in step["metadata"]
        assert "estimated_s" in step["metadata"]


def test_to_goal_steps_returns_empty_list_for_failed_plan():
    planner = make_planner(response_text="", success=False)
    plan = planner.plan("anything")
    assert plan.success is False
    assert plan.to_goal_steps() == []


def test_goal_plan_step_count_property():
    plan = GoalPlan(
        title="x",
        description="y",
        steps=[StepPlan(description="a", action="action:go")],
        priority=GoalPriority.NORMAL,
        complexity=ComplexityTier.TRIVIAL,
        total_estimate=1.0,
        status=PlannerStatus.SUCCESS,
    )
    assert plan.step_count == 1
    assert plan.success is True


def test_goal_plan_summary_includes_title_and_step_count():
    plan = GoalPlan(
        title="My Goal",
        description="x",
        steps=[
            StepPlan(description="a", action="action:go", estimated_s=5.0),
            StepPlan(description="b", action="action:go2", estimated_s=5.0),
        ],
        priority=GoalPriority.NORMAL,
        complexity=ComplexityTier.TRIVIAL,
        total_estimate=10.0,
        status=PlannerStatus.SUCCESS,
    )
    summary = plan.summary()
    assert "My Goal" in summary
    assert "2 steps" in summary
    assert "trivial" in summary


def test_goal_plan_summary_for_failed_plan_mentions_status():
    plan = GoalPlan(
        title="",
        description="",
        steps=[],
        priority=GoalPriority.NORMAL,
        complexity=ComplexityTier.TRIVIAL,
        total_estimate=0.0,
        status=PlannerStatus.LLM_UNAVAILABLE,
    )
    summary = plan.summary()
    assert "Planning failed" in summary
    assert "LLM_UNAVAILABLE" in summary


# ── 5. plan() failure path returns success=False ──────────────────────────────


def test_plan_returns_success_false_when_llm_call_fails():
    """LLM query returns success=False -> LLM_UNAVAILABLE plan, success=False."""
    planner = make_planner(response_text="ignored", success=False)
    plan = planner.plan("anything")
    assert plan.success is False
    assert plan.status is PlannerStatus.LLM_UNAVAILABLE
    assert plan.step_count == 0
    # last_plan must NOT be updated for a failed plan
    assert planner.last_plan is None


def test_plan_returns_llm_unavailable_when_planner_not_ready():
    """Planner that was never setup() -> LLM_UNAVAILABLE without calling LLM."""
    brain = make_brain()
    planner = GoalPlanner(llm=brain)  # note: no setup() call
    plan = planner.plan("anything")
    assert plan.success is False
    assert plan.status is PlannerStatus.LLM_UNAVAILABLE
    brain.query.assert_not_called()


# ── refine() ──────────────────────────────────────────────────────────────────


def test_refine_with_no_base_plan_returns_empty_refinement():
    """No cached plan and no override -> REFINEMENT_ERROR result."""
    planner = make_planner()
    result = planner.refine("add a step")
    assert isinstance(result, RefinementResult)
    assert result.changed is False
    assert result.original_step_count == 0
    assert result.new_step_count == 0
    assert result.plan.success is False
    assert result.plan.status is PlannerStatus.REFINEMENT_ERROR


def test_refine_with_successful_response_reports_changes():
    """Successful refinement: a new step is added -> result.changed True."""
    base = make_planner(response_text=valid_plan_json(n_steps=2)).plan("base")
    assert base.success

    # Second brain mock returns a plan with 3 steps
    refined_payload = valid_plan_json(n_steps=3)
    new_brain = make_brain(response_text=refined_payload, success=True)

    from friday.goal_planner import GoalPlanner
    planner = GoalPlanner(llm=new_brain)
    planner.setup()
    # Inject the base plan so refine() uses it
    planner._last_plan = base

    result = planner.refine("add another step")

    assert isinstance(result, RefinementResult)
    assert result.changed
    assert result.original_step_count == 2
    assert result.new_step_count == 3
    assert result.plan.success
    assert planner.last_plan is result.plan


def test_refine_when_llm_fails_returns_error_result():
    base = make_planner().plan("base")
    new_brain = make_brain(response_text="", success=False)
    planner = GoalPlanner(llm=new_brain)
    planner.setup()
    planner._last_plan = base

    result = planner.refine("anything")

    assert result.plan.status is PlannerStatus.REFINEMENT_ERROR
    # Original step count is preserved on the result envelope
    assert result.original_step_count == base.step_count
    assert result.new_step_count == base.step_count


# ── assess_complexity() ───────────────────────────────────────────────────────


def test_assess_complexity_returns_tier():
    planner = make_planner(response_text="trivial")
    tier = planner.assess_complexity("simple task")
    assert tier is ComplexityTier.TRIVIAL


def test_assess_complexity_falls_back_to_moderate_on_unready():
    brain = make_brain()
    brain.is_ready = False
    planner = GoalPlanner(llm=brain)  # not setup()
    tier = planner.assess_complexity("anything")
    assert tier is ComplexityTier.MODERATE


def test_assess_complexity_falls_back_to_moderate_on_unknown_word():
    planner = make_planner(response_text="very-difficult-impossible")
    tier = planner.assess_complexity("anything")
    assert tier is ComplexityTier.MODERATE


# ── estimate_duration() ───────────────────────────────────────────────────────


def test_estimate_duration_for_short_plan():
    plan = GoalPlan(
        title="x",
        description="y",
        steps=[],
        priority=GoalPriority.NORMAL,
        complexity=ComplexityTier.TRIVIAL,
        total_estimate=10.0,  # < 30s
        status=PlannerStatus.SUCCESS,
    )
    planner = make_planner()
    assert "30 seconds" in planner.estimate_duration(plan)


def test_estimate_duration_for_hours_long_plan():
    plan = GoalPlan(
        title="x",
        description="y",
        steps=[],
        priority=GoalPriority.NORMAL,
        complexity=ComplexityTier.EPIC,
        total_estimate=7200.0,  # 2 hours
        status=PlannerStatus.SUCCESS,
    )
    planner = make_planner()
    assert "hours" in planner.estimate_duration(plan)


# ── setup() / properties / singleton ──────────────────────────────────────────


def test_setup_returns_false_when_llm_not_ready():
    brain = MagicMock()
    brain.is_ready = False
    planner = GoalPlanner(llm=brain)
    assert planner.setup() is False
    assert planner.is_ready is False


def test_is_ready_reflects_setup_state():
    planner = make_planner()
    assert planner.is_ready is True


def test_get_goal_planner_returns_singleton():
    """The factory returns the same instance on repeat calls."""
    # Reset the module-level singleton so the test is hermetic
    import friday.goal_planner as gp_mod
    gp_mod._instance = None

    brain1 = make_brain()
    brain2 = make_brain()

    a = get_goal_planner(llm=brain1)
    b = get_goal_planner(llm=brain2)  # should NOT replace the instance

    assert a is b
    assert a is gp_mod._instance
    # Second call must NOT swap the LLM
    assert a._llm is brain1
