from unittest.mock import MagicMock

from friday.engine import FridayEngine
from friday.planner import Intent, PlannerDecision, RoutingDecision
from friday.swarm import Specialist, TaskResult, TaskStatus
from friday.voice_to_prompt import VoiceToPromptResult


def test_engine_routes_structured_prompt_with_ensemble():
    engine = FridayEngine()
    engine._handle_llm_query = MagicMock(return_value="ok")
    voice_prompt = VoiceToPromptResult(
        original_transcript="write code",
        cleaned_transcript="write code",
        intent="coding",
        target_model="deepseek",
        structured_prompt="Task: write code",
        confidence=0.9,
        entities={},
        task_category="coding",
        ensemble_type="coding",
        target_agent="CODER",
        priority=3,
    )
    decision = PlannerDecision(
        routing=RoutingDecision.QUERY_LLM,
        intent=Intent.LLM_QUERY,
        text="write code",
        confidence=0.9,
    )

    assert engine._route_decision(decision, "write code", voice_prompt) == "ok"
    engine._handle_llm_query.assert_called_once_with(
        "Task: write code",
        ensemble_type="coding",
    )


def test_engine_resolves_specialist_names_and_labels():
    engine = FridayEngine()

    assert engine._resolve_specialist("CODER") is Specialist.CODER
    assert engine._resolve_specialist("CoderAgent") is Specialist.CODER
    assert engine._resolve_specialist("coding") is Specialist.CODER
    assert engine._resolve_specialist("file_analysis") is Specialist.RETRIEVAL
    assert engine._resolve_specialist("BROWSER") is Specialist.BROWSER
    assert engine._resolve_specialist("BrowserAgent") is Specialist.BROWSER
    assert engine._resolve_specialist("browser") is Specialist.BROWSER
    assert engine._resolve_specialist("browser_agent") is Specialist.BROWSER
    assert engine._resolve_specialist("browser_automation") is Specialist.BROWSER


def test_push_trigger_accepts_task_result_and_wakes_engine():
    engine = FridayEngine()
    result = TaskResult(task_id="1", status=TaskStatus.COMPLETED, output="done")

    engine.push_trigger(result)

    assert engine._trigger_queue.get_nowait() == "done"
    assert engine._wake_event.is_set()
