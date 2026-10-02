
import time
from unittest.mock import MagicMock, patch

from friday.llm import LLMBrain, LLMResponse, LLMStatus
from friday.memory.manager import LongTermMemory
from friday.planner import ActionPlanner, RoutingDecision
from friday.swarm import Specialist, SwarmOrchestrator


def test_swarm_delegation_flow():
    # Setup a dummy reporting callback
    reports = []
    def mock_report(msg):
        reports.append(msg)

    orchestrator = SwarmOrchestrator(engine_report_callback=mock_report)

    # Mock LLM in the agents
    with patch("friday.llm.LLMBrain.query") as mock_query, \
         patch("friday.llm.LLMBrain.setup") as mock_setup:

        mock_setup.return_value = True
        mock_query.return_value = LLMResponse(text="Success response", status=LLMStatus.SUCCESS)

        orchestrator.start()

        try:
            # Delegate a task
            prompt = "Write a hello world script in Python."
            task_id = orchestrator.delegate(Specialist.CODER, prompt)

            # Verify task exists in DB
            db = LongTermMemory()
            task = db.get_task(task_id)
            assert task is not None
            assert task["agent_type"] == "CoderAgent"

            # Wait for worker to pick up and finish
            timeout = 5
            start = time.time()
            while not reports and (time.time() - start) < timeout:
                time.sleep(0.5)

            # Verify result
            assert len(reports) > 0
            final_task = db.get_task(task_id)
            assert final_task is not None, "Task not found"
            assert final_task["status"] == "COMPLETED"

        finally:
            orchestrator.stop()

def test_intent_classification():
    brain = LLMBrain()

    with patch("friday.llm.LLMBrain._call_api") as mock_call:
        brain._ready = True
        brain._client = MagicMock()

        # Mocking deep task response
        mock_call.return_value = LLMResponse(
            text='{"type": "deep_task", "specialist": "CODER", "acknowledgment": "I will handle that, Sir."}',
            status=LLMStatus.SUCCESS
        )

        res = brain.classify_intent("Implement a complex system.")
        assert res["type"] == "deep_task"
        assert res["specialist"] == "CODER"

        # Mocking casual response
        mock_call.return_value = LLMResponse(
            text='{"type": "casual_chat", "specialist": null, "acknowledgment": "Hello!"}',
            status=LLMStatus.SUCCESS
        )

        res = brain.classify_intent("Hi!")
        assert res["type"] == "casual_chat"

def test_planner_delegation():
    # Verify the planner actually decides to delegate
    planner = ActionPlanner()

    # Simulate a low-confidence or unknown command with ensemble_hint
    decision = planner.decide_with_ensemble_hint("write a complex python script for me", ensemble_hint="coding")

    assert decision.routing == RoutingDecision.DELEGATE
    assert decision.context["specialist"] == "CODER"
    assert "ensemble_hint" in decision.context
