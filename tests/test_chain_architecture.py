from unittest.mock import MagicMock

import pytest

from friday.engine import FridayEngine
from friday.planner import ActionPlanner
from friday.refinement import RefinementChain
from friday.safety import VerificationChain
from friday.voice_to_prompt import VoiceToPromptAgent


def test_routing_chain_builds_execution_plan_with_three_agent_cap():
    planner = ActionPlanner()
    structured = VoiceToPromptAgent().process(
        "search NVIDIA models and also write code then summarize a document"
    )

    plan = planner.create_execution_plan(structured)

    assert plan.max_concurrency == 3
    assert [step.task_category for step in plan.steps] == ["search", "coding", "document"]
    assert plan.steps[0].read_only is True
    assert plan.steps[1].read_only is False
    assert plan.steps[2].dependencies == ("CODER",)


def test_delivery_chain_voice_optimizes_and_uses_emotion_context():
    engine = FridayEngine()
    emotion = MagicMock()
    emotion.needs_acknowledgement = False
    context = engine._create_chain_context("explain API and RAM for 1,024 users", emotion)
    context["synthesized_response"] = "API uses RAM for 1,024 users."

    engine._run_delivery_chain(context)

    assert context["delivery_format"] == "A-P-I uses R-A-M for one thousand twenty-four users."


@pytest.mark.asyncio
async def test_refinement_chain_stops_when_improvement_below_rule():
    chain = RefinementChain()
    context = {"confidence": 0.8, "refinement_history": []}

    async def executor(candidate, iteration, tier):
        candidate = dict(candidate)
        candidate["confidence"] = 0.83
        return candidate

    result = await chain.refine(context, executor)

    assert result["confidence"] == 0.8
    assert result["refinement_history"][0].improvement == 0.03
    assert result["refinement_history"][0].accepted is False


@pytest.mark.asyncio
async def test_verification_chain_blocks_when_both_safety_models_flag(monkeypatch):
    async def fake_call(model_id, ensemble_name, messages, *args, **kwargs):
        if "nemoguard" in model_id or "llama-guard" in model_id:
            return "unsafe"
        # reasoning_70b and reward_340b both resolve to llama-3.1-70b now;
        # distinguish by ensemble_name instead of model_id.
        if ensemble_name == "reasoning":
            return "correct"
        if ensemble_name == "document":
            return "0.9"
        return None

    monkeypatch.setattr("friday.safety.call_model", fake_call)
    context = {
        "transcript": "test",
        "agent_results": [{"agent": "REASONING", "output": "result"}],
    }

    result = await VerificationChain().verify(context)

    assert result["safety_verdict"].verdict == "block"
    assert result["confidence"] == 0.5
