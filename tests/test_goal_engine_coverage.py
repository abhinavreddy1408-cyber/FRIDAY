import time

from friday.goal_engine import (
    Goal,
    GoalEngine,
    GoalPriority,
    GoalStatus,
    GoalStep,
    StepStatus,
    _GoalRepository,
    _StepExecutor,
)
from friday.task_queue import TaskPriority


class FakeTaskQueue:
    def __init__(self):
        self.submitted = []
        self.cancelled = []

    def submit(self, **kwargs):
        task_id = f"task-{len(self.submitted) + 1}"
        self.submitted.append((task_id, kwargs))
        return task_id

    def cancel(self, task_id):
        self.cancelled.append(task_id)
        return True


def test_goal_step_and_goal_serialization_and_summaries():
    started = time.time()
    step = GoalStep(
        id="s1",
        description="Collect inputs",
        action="test: collect",
        status=StepStatus.COMPLETED,
        result="ok",
        started_at=started,
        completed_at=started + 2,
        metadata={"kind": "setup"},
    )
    pending = GoalStep(id="s2", description="Run plan", action="test: run")
    goal = Goal(
        id="g1",
        title="Demo",
        description="Demo goal",
        steps=[step, pending],
        tags=["demo"],
        metadata={"owner": "friday"},
    )

    restored = Goal.from_dict(goal.to_dict())

    assert step.duration == 2
    assert step.can_retry is True
    assert step.is_terminal is True
    assert restored.steps[0].result == "ok"
    assert restored.current_step.id == "s2"
    assert restored.progress_pct == 50.0
    assert "50% complete" in restored.progress_summary()

    restored.status = GoalStatus.COMPLETED
    restored.completed_at = restored.created_at + 5
    assert "completed in 5 seconds" in restored.progress_summary()

    restored.status = GoalStatus.FAILED
    restored.steps[1].error = "blocked"
    assert "failed" in restored.progress_summary()


def test_goal_repository_crud_filters_and_delete(tmp_path):
    repo = _GoalRepository(str(tmp_path / "goals.db"))
    goal = Goal(
        id="goal-1",
        title="Persist",
        description="Stored goal",
        steps=[GoalStep(id="step-1", description="Do it", action="test: do")],
        priority=GoalPriority.HIGH,
        tags=["core"],
    )

    repo.save(goal)
    loaded = repo.load("goal-1")
    high = repo.load_all(priority=GoalPriority.HIGH)
    pending = repo.load_all(status=GoalStatus.PENDING)

    assert loaded.title == "Persist"
    assert high[0].id == "goal-1"
    assert pending[0].steps[0].description == "Do it"
    assert repo.count() == 1
    assert repo.count(GoalStatus.PENDING) == 1
    assert repo.delete("goal-1") is True
    assert repo.load("goal-1") is None
    assert repo.delete("missing") is False


def test_step_executor_dispatch_success_failure_and_missing_handler():
    executor = _StepExecutor()
    executor.register("ok:", lambda step: f"handled {step.description}")
    executor.register("empty:", lambda step: "")

    def fail(step):
        raise RuntimeError("handler failed")

    executor.register("fail:", fail)

    assert executor.execute(GoalStep("1", "step", "ok: run")) == (True, "handled step")
    assert executor.execute(GoalStep("2", "step", "empty: run")) == (True, "Done.")
    assert executor.execute(GoalStep("3", "step", "fail: run")) == (False, "handler failed")
    missing_success, missing_error = executor.execute(GoalStep("4", "step", "none: run"))
    assert missing_success is False
    assert "No handler registered" in missing_error


def test_goal_engine_create_query_pause_resume_cancel_delete_and_stats(tmp_path):
    tq = FakeTaskQueue()
    engine = GoalEngine(task_queue=tq)
    engine._repo = _GoalRepository(str(tmp_path / "engine.db"))

    assert engine.setup() is True
    goal_id = engine.create_goal(
        "Build",
        description="Build system",
        steps=[{"description": "First", "action": "ok: first", "metadata": {"n": 1}}],
        priority=GoalPriority.HIGH,
        tags=["build"],
        metadata={"source": "test"},
    )

    assert engine.is_ready is True
    assert engine.add_step(goal_id, "Inserted", "ok: inserted", position=0) is True
    assert engine.add_step("missing", "Nope", "ok: no") is False
    assert engine.get_goal(goal_id).steps[0].description == "Inserted"
    assert engine.list_goals(priority=GoalPriority.HIGH)[0].id == goal_id
    assert engine.get_progress(goal_id).startswith("Goal 'Build'")
    assert engine.get_progress("missing") is None
    assert engine.detect_goal("help me plan a project") is True
    assert engine.detect_goal("open calculator") is False
    assert engine.start_goal("missing") is False
    assert engine.start_goal(goal_id) is True
    assert tq.submitted[0][1]["priority"] == TaskPriority.NORMAL
    assert engine.active_goal_count == 1
    assert engine.pause_goal(goal_id) is True
    assert tq.cancelled == ["task-1"]
    assert engine.resume_goal(goal_id) is True
    assert engine.cancel_goal(goal_id) is True
    assert engine.cancel_goal(goal_id) is False
    assert engine.stats["cancelled"] == 1
    assert engine.delete_goal(goal_id) is True
    assert engine.delete_goal("missing") is False
    engine.teardown()
    assert engine.is_ready is False


def test_goal_engine_execute_goal_completion_and_failure_paths(tmp_path):
    completed = []
    failed = []
    stepped = []
    engine = GoalEngine(
        task_queue=FakeTaskQueue(),
        on_step_complete=lambda goal, step: stepped.append(step.description),
        on_goal_complete=completed.append,
        on_goal_failed=lambda goal, reason: failed.append(reason),
    )
    engine._repo = _GoalRepository(str(tmp_path / "execute.db"))
    engine.setup()
    engine.register_step_handler("ok:", lambda step: "done")
    engine.register_step_handler("bad:", lambda step: "still bad" if step.retries else "bad")

    good_id = engine.create_goal(
        "Good",
        steps=[
            {"description": "One", "action": "ok: one"},
            {"description": "Two", "action": "ok: two"},
        ],
    )
    good = engine.get_goal(good_id)
    good.status = GoalStatus.ACTIVE
    engine._repo.save(good)
    engine._active_tasks[good_id] = "task-good"

    engine._execute_goal(good_id)

    good = engine.get_goal(good_id)
    assert good.status == GoalStatus.COMPLETED
    assert [step.status for step in good.steps] == [StepStatus.COMPLETED, StepStatus.COMPLETED]
    assert stepped == ["One", "Two"]
    assert completed[0].id == good_id

    bad_id = engine.create_goal(
        "Bad",
        steps=[{"description": "Fail", "action": "missing: fail", "max_retries": 1}],
    )
    bad = engine.get_goal(bad_id)
    bad.status = GoalStatus.ACTIVE
    engine._repo.save(bad)
    engine._active_tasks[bad_id] = "task-bad"

    engine._execute_goal(bad_id)

    bad = engine.get_goal(bad_id)
    assert bad.status == GoalStatus.FAILED
    assert bad.steps[0].status == StepStatus.FAILED
    assert failed
    assert engine._map_priority(GoalPriority.LOW) == TaskPriority.BACKGROUND
    assert engine._map_priority(GoalPriority.NORMAL) == TaskPriority.LOW
    assert engine._map_priority(GoalPriority.HIGH) == TaskPriority.NORMAL
