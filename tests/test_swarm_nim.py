
from unittest.mock import MagicMock, patch

from friday.llm import LLMBrain
from friday.swarm import (
    BrowserAgent,
    CoderAgent,
    MediaAgent,
    ResearchAgent,
    SecurityAgent,
    Specialist,
    SwarmOrchestrator,
    TaskPayload,
)


def test_specialist_model_assignment():
    """Verify that each specialist uses its exact assigned NIM model."""
    mock_mem = MagicMock()
    coder = CoderAgent(lambda x: None, mock_mem)
    security = SecurityAgent(lambda x: None, mock_mem)
    research = ResearchAgent(lambda x: None, mock_mem)

    assert coder._model == "qwen2.5-coder:3b-instruct-q4_K_M"
    assert research._model == "qwen2.5:3b-instruct-q4_K_M"
    assert security._model == "qwen2.5:3b-instruct-q4_K_M"

def test_router_model_override():
    """Verify that LLMBrain.classify_intent overrides the model to 8b."""
    brain = LLMBrain()
    # Mock the client so it doesn't try to connect
    brain._client = MagicMock()
    brain._ready = True

    with patch.object(brain, '_call_api') as mock_call:
        mock_call.return_value.success = True
        mock_call.return_value.text = '{"type": "deep_task", "specialist": "CODER", "acknowledgment": "Ack"}'

        brain.classify_intent("How do I build a PQC key exchange?")

        # Check if model override was passed
        _args, kwargs = mock_call.call_args
        assert kwargs.get("model") == "qwen2.5:1.5b-instruct-q4_K_M"


@patch("friday.swarm.LongTermMemory")
def test_swarm_delegation_flow(mock_memory):
    """Verify that delegation correctly packages history and updates DB."""
    mock_db = mock_memory.return_value
    with patch("friday.swarm.SwarmBus") as mock_bus:
        orch = SwarmOrchestrator(lambda x: None)
        history = [{"role": "user", "content": "hi"}]
        task_id = orch.delegate(Specialist.CODER, "write a script", history=history)

        assert task_id is not None
        # Verify db record creation
        mock_db.create_task.assert_called_once()
        # Verify bus push
        mock_bus.return_value.put.assert_called_once()
        payload = mock_bus.return_value.put.call_args[0][0]
        assert payload.prompt == "write a script"
        assert list(payload.history) == history

@patch("friday.swarm.LongTermMemory")
def test_agent_status_updates(mock_memory):
    """Verify agent updates status to IN_PROGRESS and COMPLETED."""
    mock_db = mock_memory.return_value
    agent = CoderAgent(lambda x: None, mock_db)
    payload = TaskPayload(task_id="test_id", specialist=Specialist.CODER, prompt="test")

    # Mock specialized LLM setup
    agent._llm._ready = True
    agent._llm._client = MagicMock()

    with patch.object(agent._llm, 'query') as mock_query:
        mock_query.return_value.text = "done"
        mock_query.return_value.status.name = "SUCCESS"
        mock_query.return_value.success = True

        agent.run_task(payload)

        # Check DB updates (success status is implicit in update_task call logic)
        mock_db.update_task.assert_any_call("test_id", "IN_PROGRESS")
        mock_db.update_task.assert_any_call("test_id", "COMPLETED", "done")


def test_media_agent_execute():
    """Verify that MediaAgent executes task by calling media_skill.generate_image."""
    mock_db = MagicMock()
    agent = MediaAgent(lambda x: None, mock_db)
    payload = TaskPayload(task_id="media_test", specialist=Specialist.MEDIA, prompt="A beautiful painting")

    with patch("friday.skills.media_skill.generate_image") as mock_generate:
        res = agent.execute(payload)
        mock_generate.assert_called_once_with("A beautiful painting")
        assert "Image generation initiated" in res


def test_browser_agent_model_and_execute():
    """Verify that BrowserAgent is correctly assigned the correct model and calls the LLM directly on execute."""
    mock_db = MagicMock()
    agent = BrowserAgent(lambda x: None, mock_db)
    assert agent._model == "qwen2.5:3b-instruct-q4_K_M"

    payload = TaskPayload(task_id="browser_test", specialist=Specialist.BROWSER, prompt="test action")

    agent._llm._ready = True
    agent._llm._client = MagicMock()

    with patch.object(agent._llm, 'query') as mock_query:
        mock_query.return_value.text = "Browser execution result"
        mock_query.return_value.status.name = "SUCCESS"
        mock_query.return_value.success = True

        res = agent.execute(payload)
        mock_query.assert_called_once_with("test action", temperature=None, ensemble_type="screen_interaction")
        assert res == "Browser execution result"
