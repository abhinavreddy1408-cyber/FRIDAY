# mypy: ignore-errors
"""
Phase 7 Tests
Covers: shared LLMBrain client, ensemble updates, agent fixes, multilingual, merge hardening
"""

from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI

# ── Block 1: Shared Client Tests ──────────────────────────────────────────


class TestSharedLLMClient:
    """Verify shared OpenAI client injection across swarm specialists."""

    def test_swarm_creates_single_client(self):
        """SwarmOrchestrator should create exactly one OpenAI client."""
        from friday.swarm import SwarmOrchestrator

        mock_cb = MagicMock()
        with patch("friday.swarm.OpenAI") as mock_openai, patch("friday.swarm.LongTermMemory"):
            swarm = SwarmOrchestrator(report_callback=mock_cb)

        assert hasattr(swarm, 'shared_client')
        assert swarm.shared_client is not None
        mock_openai.assert_called_once()

    def test_all_agents_receive_same_client(self):
        """Every specialist should share the same OpenAI client instance."""
        from friday.swarm import SwarmOrchestrator

        mock_cb = MagicMock()
        mock_client = MagicMock(spec=OpenAI)

        with patch("friday.swarm.LongTermMemory"):
            swarm = SwarmOrchestrator(
                report_callback=mock_cb,
                shared_client=mock_client,
            )

        for name, agent in swarm._agents.items():
            assert agent._llm._client is mock_client, \
                f"{name} has different client instance"

    def test_injected_client_is_used(self):
        """If a client is injected, it should be used by all agents."""
        from friday.swarm import SwarmOrchestrator

        mock_cb = MagicMock()
        injected_client = MagicMock(spec=OpenAI)

        with patch("friday.swarm.LongTermMemory"):
            swarm = SwarmOrchestrator(
                report_callback=mock_cb,
                shared_client=injected_client,
            )

        assert swarm.shared_client is injected_client

    def test_llm_brain_accepts_shared_client(self):
        """LLMBrain should accept an injected OpenAI client."""
        from friday.llm import LLMBrain

        mock_client = MagicMock(spec=OpenAI)
        brain = LLMBrain(shared_client=mock_client)

        assert brain._client is mock_client
        assert brain._owns_client is False
        assert brain.is_ready is True

    def test_llm_brain_setup_skips_if_client_injected(self):
        """setup() should not overwrite an injected client."""
        from friday.llm import LLMBrain

        mock_client = MagicMock(spec=OpenAI)
        brain = LLMBrain(shared_client=mock_client)

        assert brain.setup() is True
        assert brain._client is mock_client

    def test_agents_have_isolated_histories(self):
        """Agents sharing a client must NOT share conversation histories."""
        from friday.swarm import SwarmOrchestrator

        mock_cb = MagicMock()
        mock_client = MagicMock(spec=OpenAI)

        with patch("friday.swarm.LongTermMemory"):
            swarm = SwarmOrchestrator(
                report_callback=mock_cb,
                shared_client=mock_client,
            )

        agent_a = swarm._agents[next(iter(swarm._agents.keys()))]
        agent_b = swarm._agents[list(swarm._agents.keys())[1]]

        assert agent_a._llm.history is not agent_b._llm.history


# ── Block 2: Ensemble Tests ─────────────────────────────────────────────────


class TestEnsembleTimeouts:
    """Verify ensemble timeout configuration and enforcement."""

    def test_brainstorming_has_strict_timeout(self):
        from friday.ensemble import get_ensemble_timeout
        assert get_ensemble_timeout("brainstorming") == 15.0

    def test_ensemble_timeout_defaults_to_20(self):
        from friday.ensemble import get_ensemble_timeout
        assert get_ensemble_timeout("unknown_ensemble") == 20.0

    def test_all_ensembles_have_timeouts(self):
        from friday.ensemble import ENSEMBLE_REGISTRY, get_ensemble_timeout
        for name in ENSEMBLE_REGISTRY:
            assert get_ensemble_timeout(name) > 0, f"{name} has no timeout"


class TestDeepContextGating:
    """Verify deep_context is only used for long inputs."""

    def test_short_input_skips_deep_context(self):
        from friday.ensemble import should_use_deep_context
        assert should_use_deep_context("What is 2+2?") is False
        assert should_use_deep_context("") is False

    def test_long_input_uses_deep_context(self):
        from friday.ensemble import should_use_deep_context
        long_text = "word " * 500  # ~2000 chars
        assert should_use_deep_context(long_text) is True

    def test_boundary_case(self):
        from friday.ensemble import should_use_deep_context
        below = "word " * 350
        assert should_use_deep_context(below) is False
        above = "word " * 450
        assert should_use_deep_context(above) is True


# ── Block 3: Agent Model Tests ──────────────────────────────────────────────


class TestProductivityAgentModel:
    def test_productivity_uses_document_ensemble(self):
        from friday.llm import LLMResponse, LLMStatus
        from friday.swarm import ProductivityAgent, Specialist, TaskPayload

        captured_ensemble = []
        def mock_query(prompt, *, temperature=None, ensemble_type=None):
            captured_ensemble.append(ensemble_type)
            return LLMResponse(text="mock response", status=LLMStatus.SUCCESS)

        agent = ProductivityAgent(MagicMock(), MagicMock(), shared_client=MagicMock(spec=OpenAI))
        agent._llm.query = mock_query

        payload = TaskPayload(task_id="t1", specialist=Specialist.PRODUCTIVITY, prompt="Write a summary")
        agent.execute(payload)

        assert captured_ensemble[0] == "document"


class TestBrowserAgentEnsemble:
    def test_browser_uses_screen_interaction_ensemble(self):
        from friday.llm import LLMResponse, LLMStatus
        from friday.swarm import BrowserAgent, Specialist, TaskPayload

        captured_ensemble = []
        def mock_query(prompt, *, temperature=None, ensemble_type=None):
            captured_ensemble.append(ensemble_type)
            return LLMResponse(text="mock response", status=LLMStatus.SUCCESS)

        agent = BrowserAgent(MagicMock(), MagicMock(), shared_client=MagicMock(spec=OpenAI))
        agent._llm.query = mock_query

        payload = TaskPayload(task_id="t2", specialist=Specialist.BROWSER, prompt="Navigate to google.com")
        agent.execute(payload)

        assert captured_ensemble[0] == "screen_interaction"


# ── Block 4: Multilingual Tests ─────────────────────────────────────────────


class TestMultilingualRouting:
    def test_route_decision_has_multilingual(self):
        from friday.phase1 import RouteDecision
        assert hasattr(RouteDecision, 'MULTILINGUAL')
        assert RouteDecision.MULTILINGUAL == "MultilingualAgent"

    @pytest.mark.asyncio
    async def test_multilingual_routing_by_phase1_gate(self):
        from friday.phase1 import Phase1Gate, RouteDecision

        # Mock safety filter to return safe
        with patch("friday.phase1.Phase1Gate.check_input_safety", return_value=True):
            # Mock ultrafast_8b response
            mock_resp = '{"route": "MultilingualAgent", "confidence": "high", "ensemble_hint": "multilingual"}'
            with patch("friday.phase1.call_model", return_value=mock_resp):
                triage = await Phase1Gate.route("Bonjour, comment ça va?")
                assert triage.route == RouteDecision.MULTILINGUAL
                assert triage.ensemble_hint == "multilingual"


class TestMultilingualAgent:
    def test_multilingual_agent_execute(self):
        from friday.swarm import MultilingualAgent, Specialist, TaskPayload

        mock_cb = MagicMock()
        mock_mem = MagicMock()
        mock_client = MagicMock(spec=OpenAI)

        agent = MultilingualAgent(mock_cb, mock_mem, shared_client=mock_client)

        with patch.object(agent, "_query", return_value="Bonjour!") as mock_query:
            payload = TaskPayload(task_id="t3", specialist=Specialist.MULTILINGUAL, prompt="Comment ça va?")
            res = agent.execute(payload)
            assert res == "Bonjour!"
            mock_query.assert_called_once_with("Comment ça va?", ensemble_type="multilingual")


class TestMultilingualEnsemble:
    @pytest.mark.asyncio
    async def test_multilingual_ensemble_success(self):
        from friday.ensemble import EnsembleRunner

        with patch("friday.ensemble.call_model") as mock_call, \
             patch("friday.ensemble.check_safety", return_value=True):
            mock_call.return_value = "Hola"
            res = await EnsembleRunner.run_ensemble(
                ensemble_type="multilingual",
                messages=[{"role": "user", "content": "Hola"}],
                original_prompt="Hola",
                temperature=0.7,
                max_tokens=100
            )
            assert res == "Hola"
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_multilingual_ensemble_fallback(self):
        from friday.ensemble import EnsembleRunner

        call_count = 0
        async def mock_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # primary fails
            if call_count == 2:
                return "Fallback response"
            return None

        with patch("friday.ensemble.call_model", side_effect=mock_call_model), \
             patch("friday.ensemble.check_safety", return_value=True):
            res = await EnsembleRunner.run_ensemble(
                ensemble_type="multilingual",
                messages=[{"role": "user", "content": "Hola"}],
                original_prompt="Hola",
                temperature=0.7,
                max_tokens=100
            )
            assert res == "Fallback response"
            assert call_count == 2


# ── Block 5: Merge Hardening Tests ──────────────────────────────────────────


class TestMergeHardening:
    def test_merge_results_tracked_synthesis(self):
        from friday.swarm import _merge_results_tracked

        agent_results = [{"agent": "CoderAgent", "output": "Output A", "confidence": 0.9, "model_tier": "flagship"}]
        res = _merge_results_tracked(agent_results)
        assert res.output == "Output A"
        assert res.source == "synthesis"

    def test_merge_results_tracked_ranked(self):
        from friday.swarm import _merge_results_tracked

        agent_results = [
            {"agent": "CoderAgent", "output": "Output A", "confidence": 0.9, "model_tier": "flagship"},
            {"agent": "SecurityAgent", "output": "Output B", "confidence": 0.8, "model_tier": "medium"}
        ]
        res = _merge_results_tracked(agent_results)
        assert "CoderAgent: Output A" in res.output
        assert "SecurityAgent: Output B" in res.output
        assert res.source == "ranked_fallback"

    def test_merge_results_tracked_raw_concat(self):
        from friday.swarm import _merge_results_tracked

        agent_results = [
            {"agent": "CoderAgent", "output": "Output A", "confidence": 0.9, "model_tier": "flagship"},
            {"agent": "SecurityAgent", "output": "Output B", "confidence": 0.8, "model_tier": "medium"}
        ]
        # To trigger raw concat, let's mock sorted to raise an exception
        with patch("friday.swarm.sorted", side_effect=RuntimeError("Mock sort failure")):
            res = _merge_results_tracked(agent_results)
            assert "[CoderAgent - unverified]" in res.output
            assert res.source == "raw_concat"

    def test_merge_results_tracked_empty(self):
        from friday.swarm import _merge_results_tracked

        res = _merge_results_tracked([])
        assert res.output == ""
        assert res.source == "empty"
