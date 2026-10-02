import time
import json
import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from friday.goal_engine import (
    Goal,
    GoalStep,
    GoalStatus,
    StepStatus,
    GoalPriority,
    _GoalRepository,
    GoalEngine,
    get_goal_engine,
)
from friday.goal_planner import (
    GoalPlan,
    StepPlan,
    RefinementResult,
    ComplexityTier,
    PlannerStatus,
    GoalPlanner,
    get_goal_planner,
)
from friday.task_queue import TaskQueue, TaskPriority

# ── Goal Step & Goal Models ───────────────────────────────────────────────────

def test_goal_step_duration_none():
    step = GoalStep("s1", "desc", "action")
    assert step.duration is None
    step.started_at = time.time()
    assert step.duration is None
    step.completed_at = step.started_at + 10
    assert step.duration == 10

def test_goal_duration_none():
    goal = Goal("g1", "title", "desc", [])
    assert goal.duration is None
    goal.completed_at = goal.created_at + 100
    assert goal.duration == 100

def test_goal_progress_empty_steps():
    goal = Goal("g1", "title", "desc", [])
    assert goal.progress_pct == 0.0

def test_goal_step_from_dict_defaults():
    step = GoalStep.from_dict({
        "id": "s1",
        "description": "desc",
        "action": "act",
    })
    assert step.status == StepStatus.PENDING
    assert step.max_retries == 2
    assert step.retries == 0

# ── SQLite Migration & Database Gaps ──────────────────────────────────────────

def test_goal_repository_migration_missing_columns(tmp_path):
    db_file = str(tmp_path / "test_migration.db")
    
    # Pre-create table with older schema
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE goals (id TEXT PRIMARY KEY, title TEXT, status TEXT, updated_at REAL)")
    conn.commit()
    conn.close()

    # Instantiate repo — should alter table and add missing columns
    repo = _GoalRepository(db_file)
    
    # Check that modern columns exist now
    with repo._conn() as conn:
        cursor = conn.execute("PRAGMA table_info(goals)")
        columns = [row["name"] for row in cursor.fetchall()]
        assert "steps" in columns
        assert "priority" in columns
        assert "updated_at" in columns
        assert "tags" in columns

def test_goal_repository_database_exceptions(tmp_path):
    repo = _GoalRepository(str(tmp_path / "test_error.db"))
    
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.DatabaseError("DB read error")
    if hasattr(repo._local, "conn"):
        repo._local.conn = None
        
    with patch("sqlite3.connect", return_value=mock_conn):
        with pytest.raises(sqlite3.DatabaseError):
            repo.count()

# ── Goal Engine Callback Exceptions & Singleton ───────────────────────────────

def test_goal_engine_callback_exceptions(tmp_path):
    # Setup mocks
    tq = MagicMock(spec=TaskQueue)
    tq.submit.return_value = "task-1"
    
    on_step = MagicMock(side_effect=Exception("step callback crashed"))
    on_complete = MagicMock(side_effect=Exception("complete callback crashed"))
    on_failed = MagicMock(side_effect=Exception("failed callback crashed"))

    engine = GoalEngine(
        task_queue=tq,
        on_step_complete=on_step,
        on_goal_complete=on_complete,
        on_goal_failed=on_failed,
    )
    engine._repo = _GoalRepository(str(tmp_path / "test_callbacks.db"))
    engine.setup()
    
    # Create simple goal
    goal_id = engine.create_goal(
        "Callback Test",
        steps=[{"description": "step1", "action": "action:act"}],
    )
    
    # Call completed callback directly
    goal = engine.get_goal(goal_id)
    engine._complete_goal(goal)  # Should not raise exception
    
    # Call failed callback directly
    goal.status = GoalStatus.ACTIVE
    engine._fail_goal(goal, "some failure reason")  # Should not raise exception

    # Call run step callback
    engine.register_step_handler("action:", lambda step: "success")
    goal.status = GoalStatus.ACTIVE
    engine._repo.save(goal)
    engine._execute_goal(goal_id)  # Executes step and fires callback which raises exception, shouldn't crash

def test_goal_engine_singleton():
    with patch("friday.goal_engine._instance", None), \
         patch("friday.task_queue.TaskQueue.start") as mock_tq_start:
        engine = get_goal_engine()
        assert engine is not None
        assert engine._tq is not None
        # Subsequent get returns same instance
        assert get_goal_engine() is engine

def test_goal_engine_resume_active_goals_on_setup(tmp_path):
    repo = _GoalRepository(str(tmp_path / "test_resume.db"))
    goal = Goal("goal-active", "Active Title", "desc", [], status=GoalStatus.ACTIVE)
    repo.save(goal)

    engine = GoalEngine(task_queue=MagicMock())
    engine._repo = repo
    engine.setup()
    
    # The active goal should have been reset to PENDING status on setup
    loaded = engine.get_goal("goal-active")
    assert loaded.status == GoalStatus.PENDING

def test_goal_engine_step_handler_exceptions():
    engine = GoalEngine(task_queue=MagicMock())
    engine.register_step_handler("action:fail", lambda step: exec('raise Exception("handler failed")'))
    
    step = GoalStep("s1", "fail step", "action:fail")
    success, result = engine._executor.execute(step)
    assert success is False
    assert "handler failed" in result

# ── Goal Planner Gaps ─────────────────────────────────────────────────────────

def test_goal_plan_duration_hours():
    # seconds >= 3600
    plan = GoalPlan(
        title="Epic Goal",
        description="Epic Desc",
        priority=GoalPriority.NORMAL,
        complexity=ComplexityTier.EPIC,
        steps=[],
        total_estimate=7200,
        status=PlannerStatus.SUCCESS,
    )
    assert "2.0 hours" in plan.summary()

def test_refinement_result_summary_no_change():
    plan = GoalPlan(
        title="Goal",
        description="Desc",
        priority=GoalPriority.NORMAL,
        complexity=ComplexityTier.TRIVIAL,
        steps=[],
        total_estimate=0,
        status=PlannerStatus.SUCCESS,
    )
    res = RefinementResult(
        original_step_count=5,
        new_step_count=5,
        added_steps=[],
        removed_steps=[],
        plan=plan,
    )
    assert res.changed is False
    assert res.summary() == "Plan refined with no structural changes."

def test_goal_planner_complexity_mapping():
    brain = MagicMock()
    planner = GoalPlanner(llm=brain)
    planner._ready = True

    # Test all tiers of complexity
    from friday.goal_planner import _ResponseParser
    parser = _ResponseParser()
    assert parser._assess_complexity(1, 10) == ComplexityTier.TRIVIAL
    assert parser._assess_complexity(4, 150) == ComplexityTier.SIMPLE
    assert parser._assess_complexity(8, 800) == ComplexityTier.MODERATE
    assert parser._assess_complexity(15, 2000) == ComplexityTier.COMPLEX
    assert parser._assess_complexity(25, 5000) == ComplexityTier.EPIC

def test_goal_planner_estimate_duration():
    brain = MagicMock()
    planner = GoalPlanner(llm=brain)
    planner._ready = True

    def build_plan(secs):
        return GoalPlan(
            title="T", description="D", steps=[],
            priority=GoalPriority.NORMAL, complexity=ComplexityTier.SIMPLE,
            total_estimate=secs, status=PlannerStatus.SUCCESS
        )

    assert planner.estimate_duration(build_plan(20)) == "under 30 seconds"
    assert planner.estimate_duration(build_plan(60)) == "about a minute"
    assert planner.estimate_duration(build_plan(180)) == "about 3 minutes"
    assert planner.estimate_duration(build_plan(1200)) == "around 20 minutes"
    assert planner.estimate_duration(build_plan(7200)) == "approximately 2.0 hours"

def test_goal_planner_explain_plan_epic():
    brain = MagicMock()
    planner = GoalPlanner(llm=brain)
    planner._ready = True

    # Plan with 7 steps
    steps = [StepPlan(f"step {i}", "action:a") for i in range(7)]
    plan = GoalPlan(
        title="Many Steps", description="Desc", steps=steps,
        priority=GoalPriority.NORMAL, complexity=ComplexityTier.MODERATE,
        total_estimate=600, status=PlannerStatus.SUCCESS
    )
    explanation = planner.explain_plan(plan)
    assert "It has 7 steps" in explanation
    assert "and 2 more steps." in explanation

def test_goal_planner_parse_failures():
    brain = MagicMock()
    planner = GoalPlanner(llm=brain)
    planner._ready = True

    # Direct malformed JSON (not an object)
    plan1 = planner._parser.parse("[1, 2, 3]", "Objective")
    assert plan1.success is False
    assert plan1.status == PlannerStatus.VALIDATION_ERROR

    # JSON valid but schema invalid (missing steps)
    plan2 = planner._parser.parse('{"title": "Test"}', "Objective")
    assert plan2.success is False
    assert plan2.status == PlannerStatus.EMPTY_PLAN

    # malformed JSON inside text block
    plan3 = planner._parser.parse("Some random text with { incomplete json ", "Objective")
    assert plan3.success is False
    assert plan3.status == PlannerStatus.PARSE_FAILED

    # step list contains non-dict item
    plan4 = planner._parser.parse('{"title": "Test", "steps": ["invalid step"]}', "Objective")
    assert plan4.success is False
    assert plan4.status == PlannerStatus.EMPTY_PLAN

    # sanitise_action falls back to prepending action:
    parsed_step = planner._parser._parse_step({"description": "s", "action": "open chrome"}, 0)
    assert parsed_step.action == "action:open chrome"

def test_goal_planner_assess_complexity_llm_failure():
    brain = MagicMock()
    planner = GoalPlanner(llm=brain)
    planner._ready = True

    mock_resp = MagicMock()
    mock_resp.success = False
    brain.query.return_value = mock_resp
    assert planner.assess_complexity("Some epic task") == ComplexityTier.MODERATE

def test_goal_planner_refinement_error_flows():
    brain = MagicMock()
    planner = GoalPlanner(llm=brain)
    planner._ready = True

    # No last plan available
    res = planner.refine("add something")
    assert res.plan.status == PlannerStatus.REFINEMENT_ERROR

    # Query LLM fails
    planner._last_plan = GoalPlan(
        title="Last", description="D", steps=[],
        priority=GoalPriority.NORMAL, complexity=ComplexityTier.TRIVIAL,
        total_estimate=0, status=PlannerStatus.SUCCESS
    )
    mock_resp = MagicMock()
    mock_resp.success = False
    brain.query.return_value = mock_resp
    res2 = planner.refine("add step")
    assert res2.plan.status == PlannerStatus.REFINEMENT_ERROR

    # Parsing refined response fails
    mock_resp.success = True
    mock_resp.text = "invalid json"
    brain.query.return_value = mock_resp
    res3 = planner.refine("add step")
    assert res3.plan.status == PlannerStatus.PARSE_FAILED

def test_goal_planner_singleton():
    with patch("friday.goal_planner._instance", None):
        mock_brain = MagicMock()
        planner = get_goal_planner(llm=mock_brain)
        assert planner is not None
        assert get_goal_planner() is planner
