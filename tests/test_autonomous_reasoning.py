from unittest.mock import MagicMock

from friday.engine import FridayEngine
from friday.planner import PlanStep
from friday.reasoning import AutonomousReasoner
from friday.voice_to_prompt import VoiceToPromptAgent


def test_reasoner_asks_clarification_for_vague_meeting():
    structured = VoiceToPromptAgent().process("set up a team meeting next Thursday morning")
    result = AutonomousReasoner().analyze(
        "set up a team meeting next Thursday morning",
        structured_input=structured,
    )

    assert result.needs_clarification is True
    assert "time" in result.missing_information
    assert result.clarification_question == "What exact time should I schedule it for?"


def test_reasoner_uses_memory_context_when_available():
    structured = VoiceToPromptAgent().process("remember what we decided about the Friday planner")
    memory = MagicMock()
    memory.recall_relevant.return_value = []
    memory.build_context_summary.return_value = "Recent context:\n  User: Use a chain-linked planner."

    result = AutonomousReasoner().analyze(
        "remember what we decided about the Friday planner",
        structured_input=structured,
        memory=memory,
    )

    assert result.relevant_memory == ("Recent context:\n  User: Use a chain-linked planner.",)
    assert "Relevant memory is available for prompt grounding." in result.notes


def test_engine_short_circuits_to_clarification_before_routing():
    engine = FridayEngine()
    emotion = MagicMock()
    emotion.needs_acknowledgement = False
    engine._memory = MagicMock()
    engine._memory.recall_relevant.return_value = []
    engine._memory.build_context_summary.return_value = ""

    response = engine._process_command("set up a meeting soon", emotion)

    assert response == "Who should I include, and what exact time should I use?"
    assert engine._last_chain_context is not None
    assert engine._last_chain_context["execution_plan"] is None


def test_reasoning_context_is_added_to_execution_prompt():
    engine = FridayEngine()
    context = engine._create_chain_context("write a function to merge sorted lists", MagicMock())
    engine._run_perception_chain(context)
    engine._run_autonomous_reasoning_chain(context)
    step = PlanStep(
        agent="CODER",
        task_category="coding",
        ensemble_type="coding",
        prompt="Task: write code",
    )
    prompt = engine._plan_prompt(step, context)

    assert "[AUTONOMOUS_REASONING]" in prompt
    assert "Use Python unless the user specifies another language." in prompt
