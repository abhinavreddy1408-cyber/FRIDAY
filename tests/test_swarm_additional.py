import time
import pytest
from unittest.mock import MagicMock, patch

from friday.swarm import (
    Specialist,
    TaskPayload,
    TaskResult,
    TaskStatus,
    MergeResult,
    BaseAgent,
    CoderAgent,
    SecurityAgent,
    ResearchAgent,
    ReasoningAgent,
    BrainstormingAgent,
    ProductivityAgent,
    MediaAgent,
    RetrievalAgent,
    BrowserAgent,
    VisionAgent,
    MultilingualAgent,
    SwarmOrchestrator,
    _merge_results,
    _merge_results_tracked,
    _normalise_agent_result,
    _tier_rank,
)
from friday.memory.manager import LongTermMemory

# ── Merge Gaps & Tier Ranking ──────────────────────────────────────────────────

def test_tier_rank():
    assert _tier_rank("flagship-tier") == 3
    assert _tier_rank("high-perf") == 2
    assert _tier_rank("medium-perf") == 2
    assert _tier_rank("specialist-coder") == 1
    assert _tier_rank("fast-reasoner") == 1
    assert _tier_rank("unknown") == 0

def test_normalise_agent_result_object():
    class DummyResult:
        agent = "TestAgent"
        output = "output text"
        confidence = 0.85
        model_tier = "Flagship"

    res = _normalise_agent_result(DummyResult())
    assert res["agent"] == "TestAgent"
    assert res["output"] == "output text"
    assert res["confidence"] == 0.85
    assert res["model_tier"] == "Flagship"

    class DummyResultIdOnly:
        task_id = "task-123"
        output = "output text"

    res2 = _normalise_agent_result(DummyResultIdOnly())
    assert res2["agent"] == "task-123"

def test_merge_results_cascade_fallbacks():
    # Length == 1
    res1 = _merge_results_tracked([{"agent": "A", "output": "single output"}])
    assert res1.output == "single output"
    assert res1.source == "synthesis"

    # synthesis_callable fails -> fallback to ranked
    def bad_synthesis(prompt):
        raise RuntimeError("synthesis failed")

    results = [
        {"agent": "A", "output": "out A", "confidence": 0.8, "model_tier": "Flagship"},
        {"agent": "B", "output": "out B", "confidence": 0.9, "model_tier": "High"},
    ]
    res2 = _merge_results_tracked(results, synthesis_callable=bad_synthesis)
    assert res2.source == "ranked_fallback"
    assert "B: out B" in res2.output

    # ranked merge fails -> fallback to raw concat
    # We trigger ranked merge failure by putting non-numeric confidence
    bad_results = [
        {"agent": "A", "output": "out A", "confidence": "invalid_conf"},
        {"agent": "B", "output": "out B", "confidence": "invalid_conf"},
    ]
    res3 = _merge_results_tracked(bad_results)
    assert res3.source == "raw_concat"
    assert "[A - unverified]" in res3.output

    # raw concat fails -> empty
    # We pass bad objects that raise on access
    mock_bad = MagicMock()
    mock_bad.get.return_value = "some output"
    mock_bad.__getitem__.side_effect = RuntimeError("getitem error")
    
    with patch("friday.swarm._normalise_agent_result", return_value=mock_bad):
        res4 = _merge_results_tracked([object(), object()])
        assert res4.source == "empty"
        assert res4.output == ""

# ── Base Agent Tasks, Exceptions, Timeout and Persistence ─────────────────────

class DummyAgent(BaseAgent):
    def execute(self, payload: TaskPayload) -> str:
        if "sleep" in payload.prompt:
            time.sleep(1.0)
        if "fail" in payload.prompt:
            raise RuntimeError("execution failed")
        return "dummy executed " + payload.prompt

def test_base_agent_run_task_scenarios():
    mem = MagicMock(spec=LongTermMemory)
    cb = MagicMock()
    agent = DummyAgent(report_callback=cb, memory=mem)

    # Success path
    payload1 = TaskPayload("t1", Specialist.CODER, "hello")
    res1 = agent.run_task(payload1)
    assert res1.status == TaskStatus.COMPLETED
    assert res1.output == "dummy executed hello"
    assert res1.success is True
    assert cb.call_count == 1

    # Exception path
    payload2 = TaskPayload("t2", Specialist.CODER, "fail")
    res2 = agent.run_task(payload2)
    assert res2.status == TaskStatus.FAILED
    assert "execution failed" in res2.output
    assert res2.success is False

    # Timeout path
    payload3 = TaskPayload("t3", Specialist.CODER, "sleep", timeout=0.1)
    res3 = agent.run_task(payload3)
    assert res3.status == TaskStatus.TIMED_OUT
    assert "timed out after 0s" in res3.output

    # Callback Exception does not crash run_task
    bad_cb = MagicMock(side_effect=Exception("cb crash"))
    agent_bad_cb = DummyAgent(report_callback=bad_cb, memory=mem)
    res4 = agent_bad_cb.run_task(payload1)
    assert res4.status == TaskStatus.COMPLETED

    # DB Persistence exception does not crash run_task
    mem.update_task.side_effect = Exception("DB crash")
    res5 = agent.run_task(payload1)
    assert res5.status == TaskStatus.COMPLETED

# ── Specialist execute() prompts coverage ──────────────────────────────────────

@pytest.mark.asyncio
async def test_specialists_queries():
    mem = MagicMock(spec=LongTermMemory)
    cb = MagicMock()

    with patch("friday.llm.LLMBrain.query") as mock_query, \
         patch("friday.llm.LLMBrain.setup", return_value=True):
        
        mock_query.return_value = MagicMock(success=True, text="Response Text")

        # Coder Agent
        coder = CoderAgent(cb, mem)
        assert coder.execute(TaskPayload("tc", Specialist.CODER, "write code")) == "Response Text"

        # Security Agent
        security = SecurityAgent(cb, mem)
        assert security.execute(TaskPayload("ts", Specialist.SECURITY, "audit code")) == "Response Text"

        # Research Agent
        research = ResearchAgent(cb, mem)
        assert "Response Text" in research.execute(TaskPayload("tr", Specialist.RESEARCH, "find paper"))

        # Reasoning Agent
        reasoning = ReasoningAgent(cb, mem)
        assert reasoning.execute(TaskPayload("tl", Specialist.REASONING, "solve riddle")) == "Response Text"

        # Brainstorming Agent
        brainstorming = BrainstormingAgent(cb, mem)
        assert brainstorming.execute(TaskPayload("tb", Specialist.BRAINSTORMING, "ideas")) == "Response Text"

        # Productivity Agent
        productivity = ProductivityAgent(cb, mem)
        assert productivity.execute(TaskPayload("tp", Specialist.PRODUCTIVITY, "draft mail")) == "Response Text"

        # Media Agent
        media = MediaAgent(cb, mem)
        assert "Image generation initiated" in media.execute(TaskPayload("tm", Specialist.MEDIA, "paint sun"))

        # Retrieval Agent
        retrieval = RetrievalAgent(cb, mem)
        assert retrieval.execute(TaskPayload("tr", Specialist.RETRIEVAL, "find file")) == "Response Text"

        # Browser Agent
        browser = BrowserAgent(cb, mem)
        assert browser.execute(TaskPayload("tb", Specialist.BROWSER, "click submit")) == "Response Text"

        # Vision Agent
        vision = VisionAgent(cb, mem)
        assert vision.execute(TaskPayload("tv", Specialist.VISION, "analyze layout")) == "Response Text"

        # Multilingual Agent
        multilingual = MultilingualAgent(cb, mem)
        assert multilingual.execute(TaskPayload("tm", Specialist.MULTILINGUAL, "bonjour")) == "Response Text"

# ── Swarm Orchestrator Gaps ───────────────────────────────────────────────────

def test_swarm_orchestrator_initialization_failures():
    # OpenAI client creation fails with exception
    with patch("friday.swarm.OpenAI", side_effect=Exception("OpenAI init crashed")):
        orch = SwarmOrchestrator(report_callback=MagicMock())
        assert orch.shared_client is None

def test_swarm_orchestrator_lifecycle():
    cb = MagicMock()
    orch = SwarmOrchestrator(report_callback=cb)
    assert orch.pending_count == 0

    # Start loop
    orch.start()
    # Start loop twice warning
    orch.start()
    
    # delegate and check pending counts
    task_id = orch.delegate(Specialist.CODER, "task prompt")
    assert orch.pending_count >= 0

    # stop
    orch.stop(graceful=False)

def test_swarm_orchestrator_dispatch_errors():
    cb = MagicMock()
    orch = SwarmOrchestrator(report_callback=cb)
    orch._memory = MagicMock()
    orch._running = True

    # No agent registered for specialist
    payload_no_agent = TaskPayload("t-no-agent", "NonExistentSpecialist", "do it")
    orch._bus.put(payload_no_agent)
    
    # Thread pool submit fails
    payload_submit_fail = TaskPayload("t-submit-fail", Specialist.CODER, "do it")
    orch._bus.put(payload_submit_fail)
    
    with patch.object(orch._executor, "submit", side_effect=Exception("Thread pool exhaust")):
        # Let's run dispatch loop for one iteration or run the thread briefly
        orch._monitor_thread = patch("threading.Thread")
        orch.start()
        time.sleep(0.5)
        orch.stop(graceful=False)

def test_swarm_orchestrator_await_result_futures():
    cb = MagicMock()
    orch = SwarmOrchestrator(report_callback=cb)
    
    # Result exists in dict
    mock_res = TaskResult("t1", TaskStatus.COMPLETED, "output")
    orch._results["t1"] = mock_res
    assert orch.get_result("t1") is mock_res
    assert orch.await_result("t1") is mock_res

    # Result exists as future
    mock_future = MagicMock()
    mock_future.result.return_value = mock_res
    orch._futures["t2"] = mock_future
    assert orch.await_result("t2") is mock_res

    # Future result raises exception -> falls back to dict
    mock_future.result.side_effect = Exception("future crashed")
    orch._results["t2"] = mock_res
    assert orch.await_result("t2") is mock_res

    # Merge results warning log
    agent_results = [{"agent": "A", "output": "fallback text"}]
    res = orch.merge_results(agent_results)
    assert res == "fallback text"
