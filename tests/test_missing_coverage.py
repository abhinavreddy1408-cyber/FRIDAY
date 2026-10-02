# mypy: ignore-errors
"""
Comprehensive tests to cover remaining gaps and push test coverage toward 100%.
Targets:
1. friday/emotion.py (lexical matching, history window, transformer pipeline mock, wrapper checks)
2. friday/logger.py (color/plain formatters, JSON formatting, exceptions, session stats, handlers)
3. friday/refinement.py (chain accepted/rejected loops, prompt builders, iteration tier resolve)
4. friday/react_loop.py (react loop execution, bounds validation, tool failure limits, default reasoner)
"""

import os
import sys
import time
import logging
import sqlite3
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from friday.personality import EmotionalTone
from friday.emotion import (
    EmotionResult,
    _ToneHistory,
    _LexicalDetector,
    _TransformerDetector,
    EmotionDetector,
    EmotionState,
)
from friday.logger import (
    _ColourFormatter,
    _JSONFormatter,
    _SessionStats,
    _StatsHandler,
    get_logger,
    log_session_summary,
    shutdown_logging,
)
from friday.refinement import (
    RefinementStep,
    RefinementChain,
    build_refinement_prompt,
    _tier_for_iteration,
)
from friday.react_loop import (
    ReActResult,
    ReActLoop,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. EMOTION MODULE TESTS (friday/emotion.py)
# ──────────────────────────────────────────────────────────────────────────────

class TestEmotionModule:
    def test_emotion_result(self):
        res = EmotionResult(
            tone=EmotionalTone.FRUSTRATED,
            confidence=0.8,
            signals=["ugh"],
            duration=0.01,
            raw_text="ugh"
        )
        assert res.is_neutral is False
        assert res.needs_acknowledgement is True
        assert str(res) == "FRUSTRATED (80%)"

        res_neutral = EmotionResult(
            tone=EmotionalTone.NEUTRAL,
            confidence=0.0,
            signals=[],
            duration=0.0,
            raw_text=""
        )
        assert res_neutral.is_neutral is True
        assert res_neutral.needs_acknowledgement is False

    def test_tone_history(self):
        hist = _ToneHistory(_window=3, _min_repeat=2)
        assert hist.dominant() == EmotionalTone.NEUTRAL

        hist.push(EmotionalTone.HAPPY)
        assert hist.dominant() == EmotionalTone.NEUTRAL

        hist.push(EmotionalTone.HAPPY)
        assert hist.dominant() == EmotionalTone.HAPPY

        hist.push(EmotionalTone.FRUSTRATED)
        # Window holds [HAPPY, HAPPY, FRUSTRATED] -> dominant is HAPPY (count 2)
        assert hist.dominant() == EmotionalTone.HAPPY

        hist.push(EmotionalTone.FRUSTRATED)
        # Window holds [HAPPY, FRUSTRATED, FRUSTRATED] -> dominant is FRUSTRATED (count 2)
        assert hist.dominant() == EmotionalTone.FRUSTRATED

        hist.clear()
        assert hist.dominant() == EmotionalTone.NEUTRAL

    def test_lexical_detector(self):
        detector = _LexicalDetector()
        
        # Neutral
        tone, conf, sigs = detector.detect("This is neutral.")
        assert tone == EmotionalTone.NEUTRAL
        assert conf == 0.0

        # Frustrated
        tone, conf, sigs = detector.detect("Damn it, this keeps happening!")
        assert tone == EmotionalTone.FRUSTRATED
        assert conf > 0.0
        assert len(sigs) > 0

        # Urgent
        tone, conf, sigs = detector.detect("Please hurry up, asap!")
        assert tone == EmotionalTone.URGENT
        assert conf > 0.0

        # Happy
        tone, conf, sigs = detector.detect("Thank you, amazing job!")
        assert tone == EmotionalTone.HAPPY
        assert conf > 0.0

        # Confused
        tone, conf, sigs = detector.detect("Wait, what does that mean? I don't follow.")
        assert tone == EmotionalTone.CONFUSED
        assert conf > 0.0

    @patch("friday.emotion.hf_pipeline")
    def test_transformer_detector_setup_and_detect(self, mock_pipeline):
        # Test setup success
        detector = _TransformerDetector()
        assert detector.is_ready is False

        # Mock imports available
        with patch("friday.emotion._TRANSFORMERS_AVAILABLE", True):
            mock_pipe = MagicMock()
            mock_pipeline.return_value = mock_pipe
            
            assert detector.setup() is True
            assert detector.is_ready is True

            # Test detect successful
            mock_pipe.return_value = [[{"label": "joy", "score": 0.95}]]
            tone, conf = detector.detect("happy text")
            assert tone == EmotionalTone.HAPPY
            assert conf == 0.95

            # Test detect failure cases
            mock_pipe.return_value = []
            assert detector.detect("fail") is None

            mock_pipe.side_effect = Exception("Inference error")
            assert detector.detect("fail") is None

    def test_transformer_detector_setup_no_transformers(self):
        detector = _TransformerDetector()
        with patch("friday.emotion._TRANSFORMERS_AVAILABLE", False):
            assert detector.setup() is False
            assert detector.detect("text") is None

    @patch("friday.emotion._TransformerDetector")
    def test_emotion_detector_integration(self, mock_transformer_cls):
        mock_trans = MagicMock()
        mock_transformer_cls.return_value = mock_trans
        mock_trans.is_ready = True
        mock_trans.setup.return_value = True

        detector = EmotionDetector(use_transformer=True)
        assert detector.setup() is True
        assert detector.is_ready is True
        assert detector.transformer_active is True

        # Test lexical high confidence (no transformer called)
        # First analysis is smoothed to NEUTRAL because there is only one history entry
        res1 = detector.analyse("asap hurry quick")
        assert res1.tone == EmotionalTone.NEUTRAL

        # Second analysis keeps URGENT because we now have two URGENT entries in history
        res2 = detector.analyse("asap hurry quick")
        assert res2.tone == EmotionalTone.URGENT
        assert res2.confidence > 0.45
        mock_trans.detect.assert_not_called()

        # Test lexical low confidence -> triggers transformer
        # We search a query with very low lexical signal
        mock_trans.detect.return_value = (EmotionalTone.HAPPY, 0.8)
        res_low = detector.analyse("low signal text")
        # History smoothing might override it because history was URGENT.
        # Let's reset history first
        detector.reset_history()
        
        # Analyse again. Since history is empty, dominant will return NEUTRAL.
        # If smoothed (NEUTRAL) != final_tone (HAPPY), confidence is reduced
        res_low = detector.analyse("low signal text")
        assert res_low.tone == EmotionalTone.NEUTRAL  # smoothed override

        # Check stats
        stats = detector.stats
        assert stats["total_analyses"] == 4
        assert stats["transformer_active"] is True

        detector.teardown()
        assert detector.is_ready is False

    def test_detect_emotion_compatibility_wrapper(self):
        detector = EmotionDetector(use_transformer=False)
        
        # Test tired hours
        assert detector.detect_emotion("hello", hour=23).tone == "tired"
        assert detector.detect_emotion("hello", hour=4).tone == "tired"
        
        # Test tired keyword
        assert detector.detect_emotion("I am so tired", hour=12).tone == "tired"
        
        # Test urgent keywords
        assert detector.detect_emotion("do it now asap", hour=12).tone == "urgent"

        # Test frustrated keywords
        assert detector.detect_emotion("it is broken ugh", hour=12).tone == "frustrated"

        # Test casual keywords
        assert detector.detect_emotion("hey there how are you", hour=12).tone == "casual"

        # Test default
        assert detector.detect_emotion("random request", hour=12).tone == "casual"


# ──────────────────────────────────────────────────────────────────────────────
# 2. LOGGER MODULE TESTS (friday/logger.py)
# ──────────────────────────────────────────────────────────────────────────────

class TestLoggerModule:
    def _create_record(self, msg="Test log message", level=logging.INFO, exc_info=None):
        return logging.LogRecord(
            name="friday.test_module",
            level=level,
            pathname="test_module.py",
            lineno=42,
            msg=msg,
            args=(),
            exc_info=exc_info
        )

    def test_colour_formatter_plain(self):
        formatter = _ColourFormatter()
        record = self._create_record()
        
        # Patch USE_COLOUR to False
        with patch("friday.logger._USE_COLOUR", False):
            formatted = formatter.format(record)
            assert "INFO" in formatted
            assert "friday.test_module" in formatted
            assert "Test log message" in formatted

    def test_colour_formatter_color(self):
        formatter = _ColourFormatter()
        record = self._create_record(level=logging.WARNING)
        
        # Patch USE_COLOUR to True
        with patch("friday.logger._USE_COLOUR", True):
            formatted = formatter.format(record)
            assert "WARNING" in formatted
            # Contains ANSI escape codes (e.g. \033[)
            assert "\033[" in formatted

    def test_colour_formatter_with_exception(self):
        formatter = _ColourFormatter()
        try:
            raise ValueError("Test error exception")
        except ValueError:
            record = self._create_record(exc_info=sys.exc_info())

        with patch("friday.logger._USE_COLOUR", False):
            formatted = formatter.format(record)
            assert "ValueError: Test error exception" in formatted
            assert "│" in formatted

        with patch("friday.logger._USE_COLOUR", True):
            formatted = formatter.format(record)
            assert "ValueError: Test error exception" in formatted

    def test_json_formatter(self):
        formatter = _JSONFormatter()
        record = self._create_record(msg="JSON test")
        formatted = formatter.format(record)
        import json
        data = json.loads(formatted)
        assert data["level"] == "INFO"
        assert data["logger"] == "friday.test_module"
        assert data["msg"] == "JSON test"
        assert data["file"] == "test_module.py"
        assert data["line"] == 42
        assert data["exc"] is None

        # JSON formatter with exception
        try:
            raise RuntimeError("JSON Runtime Exception")
        except RuntimeError:
            record_exc = self._create_record(exc_info=sys.exc_info())
        formatted_exc = formatter.format(record_exc)
        data_exc = json.loads(formatted_exc)
        assert data_exc["exc"] is not None
        assert "RuntimeError" in data_exc["exc"]

    def test_session_stats(self):
        stats = _SessionStats()
        assert stats.snapshot()["INFO"] == 0
        stats.increment("INFO")
        assert stats.snapshot()["INFO"] == 1
        assert stats.elapsed_seconds >= 0

    def test_stats_handler(self):
        stats_handler = _StatsHandler()
        record = self._create_record(level=logging.ERROR)
        
        from friday.logger import _session
        initial = _session.snapshot()["ERROR"]
        stats_handler.emit(record)
        assert _session.snapshot()["ERROR"] == initial + 1

    def test_get_logger(self):
        log = get_logger("my_module")
        assert log.name == "friday.my_module"

        log_duped = get_logger("friday.already_prefixed")
        assert log_duped.name == "friday.already_prefixed"

    def test_log_session_summary_and_shutdown(self):
        # Simply calling to cover statements. Mock root logger.
        mock_root = MagicMock()
        with patch("logging.getLogger", return_value=mock_root):
            log_session_summary()
            mock_root.info.assert_called_once()

        # Test shutdown logging
        with patch("logging.shutdown") as mock_shutdown:
            shutdown_logging()
            mock_shutdown.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# 3. REFINEMENT MODULE TESTS (friday/refinement.py)
# ──────────────────────────────────────────────────────────────────────────────

class TestRefinementModule:
    def test_refinement_step(self):
        step = RefinementStep(
            iteration=1,
            previous_confidence=0.7,
            new_confidence=0.8,
            improvement=0.1,
            model_tier="medium",
            accepted=True
        )
        assert step.iteration == 1
        assert step.accepted is True

    @pytest.mark.asyncio
    async def test_refinement_chain_already_sufficient(self):
        chain = RefinementChain(accept_threshold=0.8)
        context = {"confidence": 0.85}
        
        async def mock_executor(ctx, it, tier):
            return {"confidence": 0.9}

        res = await chain.refine(context, mock_executor)
        assert res["confidence"] == 0.85
        assert res.get("refinement_history") == []

    @pytest.mark.asyncio
    async def test_refinement_chain_rejected_improvement(self):
        chain = RefinementChain(accept_threshold=0.85, min_improvement=0.05, max_iterations=3)
        context = {"confidence": 0.7, "transcript": "test"}
        
        # Iteration 1 returns confidence 0.72 (improvement 0.02 < 0.05 -> rejected)
        async def mock_executor(ctx, it, tier):
            return {"confidence": 0.72}

        res = await chain.refine(context, mock_executor)
        history = res["refinement_history"]
        assert len(history) == 1
        assert history[0].accepted is False
        assert res["confidence"] == 0.7  # Value remains unchanged because candidate was rejected

    @pytest.mark.asyncio
    async def test_refinement_chain_accepted_loops(self):
        chain = RefinementChain(accept_threshold=0.9, min_improvement=0.05, max_iterations=2)
        context = {"confidence": 0.7, "transcript": "test"}
        
        # Iteration 1 returns 0.8 (accepted), Iteration 2 returns 0.95 (accepted -> meets threshold)
        scores = [0.8, 0.95]
        call_count = 0
        
        async def mock_executor(ctx, it, tier):
            nonlocal call_count
            val = scores[call_count]
            call_count += 1
            return {"confidence": val, "result_val": val}

        res = await chain.refine(context, mock_executor)
        assert len(res["refinement_history"]) == 2
        assert res["confidence"] == 0.95
        assert res["result_val"] == 0.95

    def test_build_refinement_prompt(self):
        # Context with dict results
        context = {
            "transcript": "Original user transcript",
            "agent_results": [{"agent": "AgentA", "output": "Output claims."}],
            "verification_report": MagicMock(critique="Needs clarification on X.")
        }
        prompt = build_refinement_prompt(context)
        assert "Original user transcript" in prompt
        assert "Output claims." in prompt
        assert "Needs clarification on X." in prompt

        # Context with object results
        mock_result = MagicMock()
        mock_result.output = "Object output"
        context_obj = {
            "structured_input": MagicMock(cleaned_transcript="Cleaned transcript text"),
            "agent_results": [mock_result],
            "verification_report": MagicMock(critique="Critique")
        }
        prompt_obj = build_refinement_prompt(context_obj)
        assert "Cleaned transcript text" in prompt_obj
        assert "Object output" in prompt_obj

    def test_tier_for_iteration(self):
        assert _tier_for_iteration(1) == "medium"
        assert _tier_for_iteration(2) == "flagship"
        assert _tier_for_iteration(3) == "flagship"


# ──────────────────────────────────────────────────────────────────────────────
# 4. REACT LOOP TESTS (friday/react_loop.py)
# ──────────────────────────────────────────────────────────────────────────────

class TestReactLoopModule:
    def test_react_loop_invalid_iterations(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            ReActLoop(max_iterations=0)

    def test_react_loop_empty_prompt(self):
        loop = ReActLoop()
        with pytest.raises(ValueError, match="requires a non-empty prompt"):
            loop.run("")

    def test_react_loop_invalid_reasoner_type(self):
        # Reasoner must return a dict
        def bad_reasoner(prompt, observations):
            return "not a dict"
            
        loop = ReActLoop(reasoner=bad_reasoner)
        with pytest.raises(TypeError, match="reasoner must return a dict"):
            loop.run("test prompt")

    def test_react_loop_double_tool_failure_limit(self):
        # Mock reasoner to return a tool that fails
        def reasoner(prompt, observations):
            return {"final": False, "tool": "broken_tool", "query": "query"}

        mock_executor = MagicMock()
        mock_executor.execute.return_value = {
            "tool": "broken_tool",
            "query": "query",
            "result": "Failure error details",
            "success": False
        }

        loop = ReActLoop(tool_executor=mock_executor, reasoner=reasoner, max_iterations=3)
        result = loop.run("test prompt")
        assert result.forced_final is True
        # Fails after second retry (iteration 2)
        assert result.iterations == 2
        assert "failed after retry" in result.final_answer

    def test_react_loop_invalid_observation_validation(self):
        # Executor returns a dict missing required keys
        mock_executor = MagicMock()
        mock_executor.execute.return_value = {"missing_success": True}
        
        def reasoner(prompt, observations):
            return {"final": False, "tool": "tool", "query": "q"}

        loop = ReActLoop(tool_executor=mock_executor, reasoner=reasoner)
        with pytest.raises(ValueError, match="Invalid tool observation"):
            loop.run("test prompt")

    def test_default_reasoner_routing(self):
        loop = ReActLoop()
        
        # Test "latest" weather routing
        res = loop.reason("what is today's weather?", [])
        assert res["tool"] == "web_search"
        
        # Test "remember" memory routing
        res_mem = loop.reason("what did I say previously?", [])
        assert res_mem["tool"] == "knowledge_base"

        # Test "calculate" code routing
        res_calc = loop.reason("calculate 4 * 10", [])
        assert res_calc["tool"] == "code_exec"
        
        # Test default fallback
        res_def = loop.reason("who is the president?", [])
        assert res_def["final"] is True
        assert res_def["answer"] == "who is the president?"

    def test_react_loop_successful_run(self):
        # Iteration 1: reasoner requests tool "some_tool", executor succeeds.
        # Iteration 2: reasoner sees the successful observation and returns final=True with answer.
        def reasoner(prompt, observations):
            if len(observations) == 0:
                return {"final": False, "tool": "some_tool", "query": "q"}
            else:
                return {"final": True, "answer": f"Got: {observations[0]['result']}"}

        mock_executor = MagicMock()
        mock_executor.execute.return_value = {
            "tool": "some_tool",
            "query": "q",
            "result": "success data",
            "success": True
        }

        loop = ReActLoop(tool_executor=mock_executor, reasoner=reasoner)
        result = loop.run("test prompt")
        assert result.final_answer == "Got: success data"
        assert result.iterations == 2
        assert result.forced_final is False

    def test_react_loop_reaches_max_iterations(self):
        # Always requests another tool, never final
        def reasoner(prompt, observations):
            return {"final": False, "tool": "always_tool", "query": "q"}
        
        mock_executor = MagicMock()
        mock_executor.execute.return_value = {
            "tool": "always_tool",
            "query": "q",
            "result": "some result",
            "success": True
        }

        loop = ReActLoop(tool_executor=mock_executor, reasoner=reasoner, max_iterations=3)
        result = loop.run("test prompt")
        assert result.iterations == 3
        assert result.forced_final is True
        assert result.final_answer == "some result"

    def test_react_loop_force_final_static(self):
        # Test all branches of staticmethod _force_final
        assert ReActLoop._force_final("prompt", []) == "prompt"
        assert ReActLoop._force_final("prompt", [{"success": False, "result": "error"}]) == "Unable to complete with tools. Last observation: error"
        assert ReActLoop._force_final("prompt", [{"success": True, "result": "good"}]) == "good"

    def test_default_reasoner_with_observations(self):
        loop = ReActLoop()
        # With successful observation
        res_success = loop.reason("prompt", [{"tool": "web_search", "result": "sunny", "success": True}])
        assert res_success["final"] is True
        assert "sunny" in res_success["answer"]

        # With failed observation
        res_fail = loop.reason("prompt", [{"tool": "web_search", "query": "weather", "result": "network error", "success": False}])
        assert res_fail["final"] is False
        assert res_fail["tool"] == "web_search"
        assert res_fail["query"] == "weather"


# ──────────────────────────────────────────────────────────────────────────────
# 5. ADDITIONAL LOGGER TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestLoggerGaps:
    def test_setup_logging(self, tmp_path):
        import friday.logger
        from friday.logger import setup_logging
        # Save setup state
        old_setup_done = friday.logger._setup_done
        friday.logger._setup_done = False
        try:
            # Setup logging with temp log directory
            setup_logging(console_level="DEBUG", log_dir=tmp_path)
            # Call again to cover early return line 254
            setup_logging(console_level="DEBUG", log_dir=tmp_path)
            # Verify file created
            assert (tmp_path / "friday.log").exists()
            assert (tmp_path / "errors.log").exists()
        finally:
            # Cleanup handlers to avoid polluting other tests
            root = logging.getLogger("friday")
            for h in list(root.handlers):
                root.removeHandler(h)
                h.close()
            # Restore state
            friday.logger._setup_done = old_setup_done


# ──────────────────────────────────────────────────────────────────────────────
# 6. ADDITIONAL REFINEMENT TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestRefinementGaps:
    @pytest.mark.asyncio
    async def test_refinement_chain_natural_exit(self):
        # Starts at 0.5, accept threshold is 0.9.
        # Iteration 1: 0.6 (improvement 0.1 >= 0.05 -> accepted)
        # Iteration 2: 0.7 (improvement 0.1 >= 0.05 -> accepted)
        # Loop finishes (max_iterations=2) without meeting accept_threshold (0.7 < 0.9).
        # Should exit loop naturally and return context.
        chain = RefinementChain(accept_threshold=0.9, min_improvement=0.05, max_iterations=2)
        context = {"confidence": 0.5}
        scores = [0.6, 0.7]
        call_count = 0
        async def mock_executor(ctx, it, tier):
            nonlocal call_count
            val = scores[call_count]
            call_count += 1
            return {"confidence": val}
        res = await chain.refine(context, mock_executor)
        assert res["confidence"] == 0.7
        assert len(res["refinement_history"]) == 2
        assert res["refinement_history"][0].accepted is True
        assert res["refinement_history"][1].accepted is True

    def test_build_refinement_prompt_empty_outputs(self):
        # Context with empty outputs
        context_empty = {
            "agent_results": [{"output": ""}, MagicMock(output="   ")]
        }
        prompt_empty = build_refinement_prompt(context_empty)
        assert prompt_empty is not None


# ──────────────────────────────────────────────────────────────────────────────
# 7. SAFETY MODULE GAPS
# ──────────────────────────────────────────────────────────────────────────────

from friday.safety import SafetyCheck, VerificationChain, ToolOutputValidator, GroundingVerifier, ValidationResult

class TestSafetyModuleGaps:
    def test_safety_check_passed(self):
        sc_pass = SafetyCheck(verdict="pass", filter_flagged=False, classifier_flagged=False)
        assert sc_pass.passed is True
        sc_block = SafetyCheck(verdict="block", filter_flagged=True, classifier_flagged=True)
        assert sc_block.passed is False

    @pytest.mark.asyncio
    @patch("friday.safety.call_model")
    async def test_safety_filter_empty_and_unavailable(self, mock_call_model):
        chain = VerificationChain()
        
        # Test empty output
        sc_empty = await chain._safety_filter("   ")
        assert sc_empty.verdict == "block"
        assert sc_empty.reason == "empty output"

        # Test model unavailable (returns None)
        mock_call_model.return_value = None
        sc_unavail = await chain._safety_filter("hello")
        assert sc_unavail.verdict == "block"
        assert sc_unavail.reason == "safety model unavailable"

        # Test one model flagged, other not
        mock_call_model.side_effect = ["unsafe", "safe"]
        sc_partial = await chain._safety_filter("hello")
        assert sc_partial.verdict == "review"
        assert sc_partial.reason == "one safety model flagged output"

    @pytest.mark.asyncio
    @patch("friday.safety.call_model")
    async def test_cross_verify_and_quality_score_unavailable(self, mock_call_model):
        chain = VerificationChain()
        mock_call_model.return_value = None
        
        ok, reason = await chain._cross_verify("input", "output")
        assert ok is False
        assert "unavailable" in reason

        score, critique = await chain._quality_score("input", "output")
        assert score == 0.0
        assert "unavailable" in critique

    def test_flagged_helper(self):
        import re
        from friday.safety import _flagged
        assert _flagged(None) is None
        assert _flagged("unsafe message") is True
        assert _flagged("this is potentially bad") is True  # does not contain "safe" word boundaries
        assert _flagged("safe response") is False

    def test_original_input_helper(self):
        from friday.safety import _original_input
        # Mock structured input object
        mock_struct = MagicMock()
        mock_struct.cleaned_transcript = "cleaned input"
        assert _original_input({"structured_input": mock_struct}) == "cleaned input"

    def test_format_agent_outputs_non_dict(self):
        from friday.safety import _format_agent_outputs
        mock_res = MagicMock()
        del mock_res.agent
        mock_res.task_id = "task_123"
        mock_res.output = "some output"
        # Also test with agent attribute
        mock_res_2 = MagicMock()
        mock_res_2.agent = "agent_abc"
        mock_res_2.output = "other output"
        # Test fallback to task_id when both are missing
        mock_res_fallback = MagicMock()
        del mock_res_fallback.agent
        del mock_res_fallback.task_id
        mock_res_fallback.output = "fallback output"
        # Test with empty output
        mock_res_empty = MagicMock()
        mock_res_empty.output = ""

        formatted = _format_agent_outputs([mock_res, mock_res_2, mock_res_fallback, mock_res_empty])
        assert "task_123: some output" in formatted
        assert "agent_abc: other output" in formatted
        assert "agent: fallback output" in formatted

    def test_validation_schema_branches(self):
        validator = ToolOutputValidator()
        # Test expected_schema is None
        res_no_schema = validator.validate("my_tool", None, {"key": "val"})
        assert res_no_schema.valid is True

        # Test type mismatch
        res_type_err = validator.validate("my_tool", {"type": "integer"}, "not an int")
        assert res_type_err.valid is False
        assert "expected type 'integer'" in res_type_err.errors[0]

        # Test missing required keys
        res_missing_keys = validator.validate("my_tool", {"type": "dict", "required_keys": ["needed"]}, {"other": "val"})
        assert res_missing_keys.valid is False
        assert "missing required keys" in res_missing_keys.errors[0]

        # Test redact keys
        res_redact = validator.validate("my_tool", {"type": "dict", "redact_keys": ["secret"]}, {"secret": "123", "public": "abc"})
        assert res_redact.valid is True
        assert res_redact.sanitized_output["secret"] == "[REDACTED]"
        assert res_redact.sanitized_output["public"] == "abc"


# ──────────────────────────────────────────────────────────────────────────────
# 8. CONFIG MODULE GAPS
# ──────────────────────────────────────────────────────────────────────────────

from friday.config import _get_env, _cast_bool, ConfigError, FridayConfig, StorageConfig

class TestConfigModuleGaps:
    def test_get_env_required_missing(self):
        with pytest.raises(ConfigError, match="Required environment variable"):
            _get_env("NON_EXISTENT_REQUIRED_KEY", required=True)

    def test_cast_bool(self):
        assert _cast_bool("yes") is True
        assert _cast_bool("1") is True
        assert _cast_bool("true") is True
        assert _cast_bool("on") is True
        assert _cast_bool("no") is False
        assert _cast_bool("0") is False
        assert _cast_bool("false") is False
        assert _cast_bool(True) is True
        assert _cast_bool(False) is False

    def test_silenced_showwarning(self):
        from friday.config import _silenced_showwarning
        with patch("friday.config._original_showwarning") as mock_orig:
            # Silenced warnings should return early
            _silenced_showwarning("msg", UserWarning, "file.py", 10)
            mock_orig.assert_not_called()
            
            _silenced_showwarning("msg", DeprecationWarning, "file.py", 10)
            mock_orig.assert_not_called()
            
            # Non-silenced warnings should call the original warning handler
            _silenced_showwarning("msg", RuntimeWarning, "file.py", 10)
            mock_orig.assert_called_once_with("msg", RuntimeWarning, "file.py", 10, None, None)

    def test_config_startup_failure(self):
        # If FridayConfig encounters ConfigError during initialization, it should print and call sys.exit(1)
        with patch("friday.config.LLMConfig", side_effect=ConfigError("Simulated LLM Config error")):
            with patch("sys.exit") as mock_exit:
                FridayConfig()
                mock_exit.assert_called_once_with(1)

    def test_config_online_models_import(self):
        import importlib
        import os
        import sys
        
        # Save env and status
        orig_online = os.environ.get("FRIDAY_ONLINE_MODELS")
        os.environ["FRIDAY_ONLINE_MODELS"] = "1"
        try:
            importlib.reload(sys.modules["friday.config"])
        finally:
            if orig_online is not None:
                os.environ["FRIDAY_ONLINE_MODELS"] = orig_online
            else:
                os.environ.pop("FRIDAY_ONLINE_MODELS", None)
            # reload config again to restore default offline state
            importlib.reload(sys.modules["friday.config"])


# ──────────────────────────────────────────────────────────────────────────────
# 9. MEMORY MANAGER GAPS
# ──────────────────────────────────────────────────────────────────────────────

from friday.memory.manager import SemanticMemory, MemoryEntry, MemoryCategory, MemoryTier, MemoryHygiene, SemanticFactStore, StructuredFact

class TestMemoryManagerGaps:
    def test_semantic_memory_uninitialized_and_exceptions(self):
        # When uninitialized
        mem = SemanticMemory()
        assert mem.store(MemoryEntry(id="1", content="c", category=MemoryCategory.FACT, tier=MemoryTier.SEMANTIC, timestamp=1.0)) is False
        assert mem.search("query") == []

        # Setup with client throwing exception
        with patch("chromadb.PersistentClient", side_effect=Exception("Chroma init error")):
            assert mem.setup() is False

        # Query/Upsert throwing exception when active
        mem._available = True
        mock_col = MagicMock()
        mem._collection = mock_col
        mock_col.upsert.side_effect = Exception("Chroma upsert error")
        mock_col.query.side_effect = Exception("Chroma query error")

        assert mem.store(MemoryEntry(id="1", content="c", category=MemoryCategory.FACT, tier=MemoryTier.SEMANTIC, timestamp=1.0)) is False
        assert mem.search("query") == []

    def test_memory_hygiene_vitality_and_pruning(self):
        # MemoryHygiene constructor takes long_term and semantic memories
        hygiene = MemoryHygiene(MagicMock(), MagicMock())
        
        # Test vitality calculations on various category & access patterns
        entry_fact = MemoryEntry(
            id="fact1",
            content="Water freezes at 0 degrees Celsius.",
            category=MemoryCategory.FACT,
            tier=MemoryTier.SEMANTIC,
            timestamp=time.time() - 10000,
            metadata={"importance": "0.9", "access_count": "5"}
        )
        # Recency is high, importance is high, access is high -> vitality should be high
        vit = MemoryHygiene.calculate_vitality(entry_fact)
        assert vit > 0.5

        # Old entry with low importance/access
        entry_old = MemoryEntry(
            id="old1",
            content="Old text.",
            category=MemoryCategory.CONVERSATION,
            tier=MemoryTier.SEMANTIC,
            timestamp=time.time() - 10 * 24 * 3600, # 10 days old
            metadata={"importance": "0.1", "access_count": "0"}
        )
        vit_old = MemoryHygiene.calculate_vitality(entry_old)
        assert vit_old < vit

    def test_semantic_fact_store_conflict(self):
        # Test sqlite fact store conflict resolution
        store = SemanticFactStore()
        store._db_path = ":memory:"
        store.setup()
        
        # Store initial fact
        store.store_fact("sky", "color", "blue", 0.9, "user")
        facts = store.query_facts("sky", "color")
        assert len(facts) == 1
        assert facts[0].value == "blue"

        # Update fact with higher confidence -> should update
        store.store_fact("sky", "color", "clear blue", 0.95, "system")
        facts2 = store.query_facts("sky", "color")
        assert len(facts2) == 1
        assert facts2[0].value == "clear blue"

        # Update fact with lower confidence -> should NOT update
        store.store_fact("sky", "color", "green", 0.2, "noise")
        facts3 = store.query_facts("sky", "color")
        assert len(facts3) == 1
        assert facts3[0].value == "clear blue"

    def test_semantic_fact_store_unready_and_exceptions(self):
        store = SemanticFactStore()
        store._db_path = ":memory:"
        # unready count
        assert store.count() == 0
        
        store.setup()
        
        # connection yield exception
        with pytest.raises(RuntimeError):
            with store._connection() as conn:
                raise RuntimeError("yield error")
                
        # query exception
        with patch.object(store, "_connection", side_effect=Exception("query fail")):
            assert store.query_facts("sky", "color") == []
            assert store.update_fact("sky", "color", "green") is False
            assert store.count() == 0

        # Query only by subject or only by predicate branches
        store.store_fact("sky", "color", "blue", 0.9, "user")
        facts_sub = store.query_facts(subject="sky")
        assert len(facts_sub) == 1
        facts_pred = store.query_facts(predicate="color")
        assert len(facts_pred) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 10. ENSEMBLE MODULE GAPS
# ──────────────────────────────────────────────────────────────────────────────

from friday.ensemble import EmbeddingCache, BM25Index, CrossEncoderReranker

class TestEnsembleGaps:
    def test_embedding_cache_size_and_clear(self):
        cache = EmbeddingCache(max_size=2)
        assert cache.size == 0
        cache.put("q1", [0.1, 0.2])
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_bm25_index_import_error(self):
        # We simulate rank_bm25 not being installed
        idx = BM25Index()
        with patch.dict("sys.modules", {"rank_bm25": None}):
            res = idx.index(["doc"], ["id"])
            assert res is False

    @patch("transformers.AutoModelForSequenceClassification")
    @patch("transformers.AutoTokenizer")
    def test_cross_encoder_reranker_failures(self, mock_tokenizer, mock_model_cls):
        # Cross encoder setup failures should fall back to identity mapping
        mock_model_cls.from_pretrained.side_effect = Exception("Model download failed")
        
        reranker = CrossEncoderReranker()
        assert reranker.is_available is False
        
        # rerank returns identity ranks when not ready
        ranks = reranker.rerank("query", ["doc1", "doc2"], top_k=2)
        assert reranker.is_available is False
        assert ranks == [(0, 1.0), (1, 0.99)]


# ──────────────────────────────────────────────────────────────────────────────
# 11. MEMORY PACKAGE GAPS (friday/memory/__init__.py)
# ──────────────────────────────────────────────────────────────────────────────

class TestMemoryPackageGaps:
    def test_global_memory_manager_singleton(self):
        from friday.memory import initialize_memory, get_memory
        import friday.memory
        
        # Reset global manager
        old_manager = friday.memory._manager
        friday.memory._manager = None
        try:
            # get_memory raises RuntimeError before initialization
            with pytest.raises(RuntimeError, match="MemoryManager accessed before initialization"):
                get_memory()
                
            # Initialize
            mgr = initialize_memory()
            assert mgr is not None
            assert get_memory() is mgr
            
            # Initialize again (returns early from _manager is not None check)
            mgr2 = initialize_memory()
            assert mgr2 is mgr

            # Test double-checked lock branch inside lock
            friday.memory._manager = None
            class MockLock:
                def __enter__(self):
                    friday.memory._manager = "concurrently-set"
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            with patch("friday.memory._lock", MockLock()):
                mgr3 = initialize_memory()
                assert mgr3 == "concurrently-set"
        finally:
            # Restore
            friday.memory._manager = old_manager

    def test_vector_memory(self):
        from friday.memory import VectorMemory, VectorMemoryRecord, _cosine_similarity
        
        # Mock sentence_transformers model returning standard list (covers False branches of tolist() and list-of-lists check)
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.5, 0.5]
        
        vmem = VectorMemory(model=mock_model)
        assert vmem.records == []
        
        # Test store raises on empty
        with pytest.raises(ValueError, match="requires non-empty text"):
            vmem.store("")
        with pytest.raises(ValueError, match="requires non-empty text"):
            vmem.store("   ")

        # Test store success
        record1 = vmem.store("hello", tags=["t1"])
        assert record1.text == "hello"
        assert record1.tags == ["t1"]
        assert record1.embedding == [0.5, 0.5]
        assert len(vmem.records) == 1

        # Test store with a custom Vector type having tolist() method returning list of lists (covers True branches)
        class MockVector:
            def tolist(self):
                return [[0.6, 0.6]]
        mock_model.encode.return_value = MockVector()
        record2 = vmem.store("world", tags=["t2"])
        assert record2.text == "world"
        assert record2.embedding == [0.6, 0.6]

        # Retrieve relevant raises on empty query
        with pytest.raises(ValueError, match="requires non-empty query"):
            vmem.retrieve_relevant("")
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            vmem.retrieve_relevant("hello", top_k=0)

        # Retrieve relevant success
        results = vmem.retrieve_relevant("hello", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "world"
        assert results[0]["score"] > 0.99

        # Retrieve relevant when no records
        vmem_empty = VectorMemory(model=mock_model)
        assert vmem_empty.retrieve_relevant("query") == []

        # Test _cosine_similarity edge cases
        # dimension mismatch
        with pytest.raises(ValueError, match="Embedding dimensions do not match"):
            _cosine_similarity([0.1], [0.1, 0.2])
        # zero norm
        assert _cosine_similarity([0.0, 0.0], [0.1, 0.2]) == 0.0

        # Test SentenceTransformer import error fallback
        vmem_no_model = VectorMemory()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match="sentence-transformers is required"):
                vmem_no_model._get_model()

        # Test lazy-loading of SentenceTransformer success path
        # Inject mock sentence_transformers into sys.modules
        mock_st = MagicMock()
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.encode.return_value = [0.5, 0.5]
        mock_st.SentenceTransformer.return_value = mock_transformer_instance
        
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            vmem_lazy = VectorMemory()
            vmem_lazy.store("hello")
            assert vmem_lazy._model is mock_transformer_instance


# ── ADDITIONAL MODULE GAPS (Task 4) ───────────────────────────────────────────

import importlib
import httpx
from friday.ensemble import get_ensemble_config, assert_not_batch_only, resolve_canonical_model, call_canonical_model, resolve_api_key, check_safety, EnsembleRunner, EmbeddingCache, CrossEncoderReranker
from friday.config import config

class TestEmotionGapsAdditional:
    def test_transformers_import_error(self):
        with patch.dict("sys.modules", {"transformers": None}):
            if "friday.emotion" in sys.modules:
                importlib.reload(sys.modules["friday.emotion"])
            from friday.emotion import _TRANSFORMERS_AVAILABLE
            assert _TRANSFORMERS_AVAILABLE is False
        # Restore normal state
        importlib.reload(sys.modules["friday.emotion"])

    def test_transformer_detector_setup_exception(self):
        from friday.emotion import _TransformerDetector
        with patch("friday.emotion.hf_pipeline", side_effect=Exception("HF pipeline setup failed")):
            detector = _TransformerDetector()
            assert detector.setup(blocking=True) is False
            assert detector.is_ready is False

    def test_action_emotion_detector_no_transformer(self):
        from friday.emotion import EmotionDetector
        with patch("friday.emotion._TRANSFORMERS_AVAILABLE", False):
            detector = EmotionDetector(use_transformer=False)
            assert detector._transformer is None
            assert detector.setup() is True
            assert detector.transformer_active is False

    def test_emotion_detector_empty_and_focused(self):
        from friday.emotion import EmotionDetector
        detector = EmotionDetector()
        
        # Empty input
        res1 = detector.detect_emotion("")
        assert res1.tone == "casual"

        # Wrapper Focused tone tests
        state1 = detector.detect_emotion("write python code")
        assert state1.tone == "focused"
        state2 = detector.detect_emotion("run a script")
        assert state2.tone == "focused"

    def test_emotion_transformer_thresholds_and_overrides(self):
        from friday.emotion import EmotionDetector
        detector = EmotionDetector()
        
        mock_tf = MagicMock()
        detector._transformer = mock_tf
        detector._ready = True

        # Case 1: transformer returns None
        mock_tf.detect.return_value = None
        res1 = detector.analyse("hello")
        assert res1.tone == EmotionalTone.NEUTRAL

        # Case 2: tf_conf <= lex_conf
        mock_tf.detect.return_value = (EmotionalTone.HAPPY, 0.2)
        detector.reset_history()
        detector._history.push(EmotionalTone.FRUSTRATED)
        detector._history.push(EmotionalTone.FRUSTRATED)
        res2 = detector.analyse("!!")
        assert res2.tone == EmotionalTone.FRUSTRATED
        assert res2.confidence == 0.3

        # Case 3: tf_conf > lex_conf
        mock_tf.detect.return_value = (EmotionalTone.HAPPY, 0.8)
        detector.reset_history()
        detector._history.push(EmotionalTone.HAPPY)
        detector._history.push(EmotionalTone.HAPPY)
        res3 = detector.analyse("hello")
        assert res3.tone == EmotionalTone.HAPPY
        assert res3.confidence == 0.8


class TestEnsembleGapsAdditional:
    def test_ensemble_get_config_keyerror(self):
        with pytest.raises(KeyError, match="Unknown ensemble 'unknown'"):
            get_ensemble_config("unknown")

    def test_assert_not_batch_only(self):
        assert_not_batch_only("reward_340b")

    def test_resolve_canonical_model_keyerror(self):
        with pytest.raises(KeyError, match="Unknown canonical model 'unknown'"):
            resolve_canonical_model("unknown")

    @pytest.mark.asyncio
    async def test_call_canonical_model_rate_limiting_and_fallback(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 429
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        
        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("asyncio.sleep", AsyncMock()):
            res = await call_canonical_model("fast_coder_32b", "coding", [{"role": "user", "content": "hi"}])
            assert res is None

    def test_resolve_api_key_edges(self, monkeypatch):
        # selected exists, exact env var missing
        monkeypatch.delenv("FRIDAY_NIM_KEY_CODING_FAST_CODER", raising=False)
        key = resolve_api_key("coding", "nvidia/llama-3.1-nemotron-51b-instruct", role="fast_coder")
        assert key == config.llm.api_key

        key2 = resolve_api_key("coding", "qwen2.5-coder:3b-instruct-q4_K_M")
        assert key2 == config.llm.api_key

    @pytest.mark.asyncio
    async def test_check_safety_empty_and_ambiguous(self):
        assert await check_safety("") is False
        assert await check_safety("   ") is False

        with patch("friday.ensemble.call_canonical_model", AsyncMock(return_value="unknown verdict")):
            assert await check_safety("some input") is True

    @pytest.mark.asyncio
    async def test_run_coding_ensemble_failures(self):
        """Both primary and fallback fail — returns error message."""
        with patch("friday.ensemble.call_model", AsyncMock(return_value=None)):
            res = await EnsembleRunner._run_coding_ensemble([{"role": "user", "content": "write code"}], "write code", 0.7, 1024)
            assert "all coding models failed" in res

    @pytest.mark.asyncio
    async def test_run_search_ensemble_variants(self):
        # 1. No research responses
        with patch("friday.ensemble.call_model", AsyncMock(return_value=None)):
            res1 = await EnsembleRunner._run_search_ensemble([{"role": "user", "content": "search query"}], 0.7, 1024)
            assert res1 == "No research results available."

        # 2. Single research response
        with patch("friday.ensemble.call_model", AsyncMock(side_effect=["result A", None, None])):
            res2 = await EnsembleRunner._run_search_ensemble([{"role": "user", "content": "search query"}], 0.7, 1024)
            assert res2 == "result A"

        # 3. Long context deep search trigger
        long_prompt = "user query " * 1500
        with patch("friday.ensemble.call_model", AsyncMock(return_value="deep response")), \
             patch("friday.ensemble.should_use_deep_context", return_value=True):
            res3 = await EnsembleRunner._run_search_ensemble([{"role": "user", "content": long_prompt}], 0.7, 1024)
            assert "deep response" in res3

    @pytest.mark.asyncio
    async def test_run_brainstorming_ensemble_exceptions_and_timeouts(self):
        """Both primary and fallback fail — returns error message."""
        with patch("friday.ensemble.call_model", AsyncMock(return_value=None)):
            res1 = await EnsembleRunner._run_brainstorming_ensemble([{"role": "user", "content": "ideas"}], 0.7, 1024)
            assert "No ideas generated" in res1

        # Primary fails, fallback succeeds
        call_results = [None, "fallback ideas"]
        with patch("friday.ensemble.call_model", AsyncMock(side_effect=call_results)):
            res2 = await EnsembleRunner._run_brainstorming_ensemble([{"role": "user", "content": "ideas"}], 0.7, 1024)
            assert res2 == "fallback ideas"

        # Primary succeeds
        with patch("friday.ensemble.call_model", AsyncMock(return_value="primary ideas")):
            res3 = await EnsembleRunner._run_brainstorming_ensemble([{"role": "user", "content": "ideas"}], 0.7, 1024)
            assert res3 == "primary ideas"

    @pytest.mark.asyncio
    async def test_run_document_and_file_analysis_failures(self):
        # Document draft fail
        with patch("friday.ensemble.call_model", AsyncMock(return_value=None)):
            assert await EnsembleRunner._run_document_ensemble([], 0.7, 1024) == "Draft generation failed."

        # Single document draft success
        with patch("friday.ensemble.call_model", AsyncMock(side_effect=["draft A", None])):
            assert await EnsembleRunner._run_document_ensemble([], 0.7, 1024) == "draft A"

        # File extraction fail
        with patch("friday.ensemble.call_model", AsyncMock(return_value=None)):
            assert await EnsembleRunner._run_file_analysis_ensemble([], 0.7, 1024) == "File analysis failed to extract information."

        # Single file extraction success
        with patch("friday.ensemble.call_model", AsyncMock(side_effect=["extraction A", None, None])):
            assert await EnsembleRunner._run_file_analysis_ensemble([], 0.7, 1024) == "extraction A"

    @pytest.mark.asyncio
    async def test_run_screen_interaction_vision_fails(self):
        with patch("friday.ensemble.call_model", AsyncMock(side_effect=["fast result", None])):
            res = await EnsembleRunner._run_screen_ensemble([], 0.7, 1024)
            assert res == "fast result"

    @pytest.mark.asyncio
    async def test_run_cascade_fallback_tiers(self):
        """Cascade fallback delegates to _call_primary_fallback('cascade_medium')."""
        # Primary fails, fallback succeeds
        with patch("friday.ensemble.call_model", AsyncMock(side_effect=[None, "fallback response"])):
            res = await EnsembleRunner._run_cascade_fallback([], 0.7, 1024)
            assert res == "fallback response"

        # All fail
        with patch("friday.ensemble.call_model", AsyncMock(return_value=None)):
            res2 = await EnsembleRunner._run_cascade_fallback([], 0.7, 1024)
            assert res2 == "All models in the cascade failed to respond."

    @pytest.mark.asyncio
    async def test_run_multilingual_ensemble_edges(self):
        """Multilingual delegates to _call_primary_fallback('multilingual')."""
        # Primary fails, fallback succeeds
        with patch("friday.ensemble.call_model", AsyncMock(side_effect=[None, "french response"])):
            res1 = await EnsembleRunner._run_multilingual_ensemble([], 0.7, 1024)
            assert res1 == "french response"

        # All fail
        with patch("friday.ensemble.call_model", AsyncMock(return_value=None)):
            res2 = await EnsembleRunner._run_multilingual_ensemble([], 0.7, 1024)
            assert "unable to process your request in that language" in res2

    def test_embedding_cache_mru_and_capacity(self):
        cache = EmbeddingCache(max_size=3)
        cache.put("q1", [1.0])
        cache.put("q2", [2.0])
        cache.put("q3", [3.0])

        assert cache.get("q1") == [1.0]

        # Put q4, should evict q2
        cache.put("q4", [4.0])
        assert cache.get("q2") is None
        assert cache.get("q1") == [1.0]

        # Put existing key updates it
        cache.put("q1", [1.1])
        assert cache.size == 3

    def test_cross_encoder_cache_loaded(self):
        reranker = CrossEncoderReranker()
        reranker._load_attempted = True
        reranker._available = True
        assert reranker._ensure_loaded() is True

    def test_cross_encoder_rerank_single_doc(self):
        mock_torch = MagicMock()
        mock_torch.no_grad = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            reranker = CrossEncoderReranker()
            reranker._available = True
            reranker._load_attempted = True
            reranker._tokenizer = MagicMock()
            reranker._model = MagicMock()
            
            mock_logits = MagicMock()
            mock_scores = MagicMock()
            mock_scores.dim.return_value = 1
            mock_val = MagicMock()
            mock_val.__float__.return_value = 1.5
            mock_scores.__getitem__.return_value = mock_val
            mock_logits.squeeze.return_value = mock_scores
            reranker._model.return_value.logits = mock_logits
            
            res = reranker.rerank("query", ["doc1"], top_k=1)
            assert len(res) == 1
            assert res[0][0] == 0


class TestLLMGaps:
    def test_llm_setup_edges(self):
        from friday.llm import LLMBrain
        brain = LLMBrain()
        mock_config = MagicMock()
        mock_config.llm.api_key = ""
        mock_config.llm.base_url = "http://test"
        mock_config.llm.timeout = 10
        with patch("friday.llm.config", mock_config), \
             patch("friday.llm.OpenAI", side_effect=Exception("API Key missing")):
            assert brain.setup() is False
        brain.teardown()
        assert brain.is_ready is False

    @patch("openai.OpenAI")
    def test_llm_query_and_token_caching(self, mock_openai_cls):
        from friday.llm import LLMBrain, LLMStatus, LLMResponse
        brain = LLMBrain()
        brain._ready = True
        mock_client = MagicMock()
        brain._client = mock_client
        
        # Test API Error fallback
        mock_client.chat.completions.create.side_effect = Exception("OpenAI API Down")
        resp = brain.query("hello")
        assert resp.status == LLMStatus.API_ERROR

        # Test retry logic with transient failures
        call_responses = [
            LLMResponse(text="", status=LLMStatus.TIMEOUT),
            LLMResponse(text="Success response", status=LLMStatus.SUCCESS, tokens_used=10)
        ]
        with patch.object(brain, "_call_api", side_effect=call_responses) as mock_call, \
             patch("time.sleep", return_value=None) as mock_sleep:
            resp_retry = brain.query("test retry", retries=2)
            assert resp_retry.status == LLMStatus.SUCCESS
            assert resp_retry.text == "Success response"
            assert mock_call.call_count == 2
            mock_sleep.assert_called_once()


class TestLearningGaps:
    def test_learning_gaps(self):
        from friday.learning import LearningEngine
        mock_config = MagicMock()
        mock_config.storage.learning_db = ":memory:"
        with patch("friday.learning.config", mock_config):
            engine = LearningEngine()
            engine.setup()
            assert engine.delete_shortcut("nonexistent") is False


class TestStreamingGaps:
    def test_streaming_pipeline_start_idempotent(self):
        from friday.streaming import StreamingResponsePipeline
        mock_tts = MagicMock()
        pipe = StreamingResponsePipeline(tts=mock_tts)
        pipe.start()
        assert pipe.is_running is True
        
        pipe.start()
        assert pipe.is_running is True
        pipe.interrupt()
        pipe.wait_until_done()

    def test_streaming_pipeline_finish_before_start(self):
        from friday.streaming import StreamingResponsePipeline
        pipe = StreamingResponsePipeline(tts=MagicMock())
        pipe.finish()

    def test_streaming_pipeline_push_token_edges(self):
        from friday.streaming import StreamingResponsePipeline
        pipe = StreamingResponsePipeline(tts=MagicMock())
        pipe.start()
        pipe.interrupt()
        pipe.push_token("token")
        assert pipe._buffer == ""
        pipe.wait_until_done()

