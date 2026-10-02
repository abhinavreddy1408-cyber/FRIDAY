"""
Tests for the evaluation harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from friday.eval.harness import EVAL_PROMPTS, run_eval
from friday.react_loop import ReActResult


def test_eval_harness() -> None:
    """Verifies that the evaluation harness runs correctly using mocked ReActLoops."""

    def mock_run(prompt: str, *args: Any, **kwargs: Any) -> ReActResult:
        if "Italy" in prompt:
            # completeness=1.0, tool_usage=1.0, iterations_ok=1.0 -> avg_score = 1.0
            return ReActResult(
                final_answer="Italy trip planned.",
                iterations=3,
                trace=[
                    {
                        "phase": "reason",
                        "iteration": 1,
                        "thought": {"final": False, "tool": "web_search", "query": "Italy"},
                    },
                    {
                        "phase": "act",
                        "iteration": 1,
                        "action": {
                            "tool": "web_search",
                            "query": "Italy",
                            "result": "hotels",
                            "success": True,
                            "error": None,
                        },
                    },
                    {
                        "phase": "reason",
                        "iteration": 2,
                        "thought": {"final": True, "answer": "Italy trip planned."},
                    },
                ],
            )
        elif "ethics" in prompt:
            # completeness=1.0, tool_usage=0.5, iterations_ok=1.0 -> avg_score = 0.83
            return ReActResult(
                final_answer="Ethics summary.",
                iterations=1,
                trace=[
                    {
                        "phase": "reason",
                        "iteration": 1,
                        "thought": {"final": True, "answer": "Ethics summary."},
                    }
                ],
            )
        elif "meeting" in prompt:
            # completeness=0.0, tool_usage=0.5, iterations_ok=1.0 -> avg_score = 0.5
            return ReActResult(
                final_answer="",
                iterations=1,
                trace=[
                    {
                        "phase": "reason",
                        "iteration": 1,
                        "thought": {"final": True, "answer": ""},
                    }
                ],
            )
        elif "cat" in prompt:
            # Simulate a runtime error/exception
            raise RuntimeError("Simulation error")
        else:
            raise ValueError(f"Unexpected prompt in test: {prompt}")

    with patch("friday.eval.harness.ReActLoop.run", side_effect=mock_run):
        results_file = Path("friday/eval/results.json")
        if results_file.exists():
            results_file.unlink()

        results = run_eval()

        # Test: results.json is created after harness runs
        assert results_file.exists()

        # Read the file to ensure it's valid JSON
        with open(results_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)

        assert file_data == results

        # Test: timestamp and overall_avg are present
        assert "timestamp" in results
        assert "overall_avg" in results
        assert "runs" in results
        assert len(results["runs"]) == len(EVAL_PROMPTS)

        expected_scores = {
            "Italy": {"completeness": 1.0, "tool_usage": 1.0, "iterations_ok": 1.0, "avg_score": 1.0},
            "ethics": {"completeness": 1.0, "tool_usage": 0.5, "iterations_ok": 1.0, "avg_score": 0.83},
            "meeting": {"completeness": 0.0, "tool_usage": 0.5, "iterations_ok": 1.0, "avg_score": 0.50},
            "cat": {"completeness": 0.0, "tool_usage": 0.5, "iterations_ok": 0.0, "avg_score": 0.17},
        }

        for run in results["runs"]:
            # Test: each run contains all required keys
            for key in [
                "prompt",
                "scores",
                "avg_score",
                "cot_trace",
                "tool_calls",
                "final_answer",
                "iterations",
            ]:
                assert key in run

            prompt = run["prompt"]
            found = False
            for substr, expected in expected_scores.items():
                if substr in prompt:
                    found = True
                    # Test: avg_score is correctly calculated
                    assert run["scores"]["completeness"] == expected["completeness"]
                    assert run["scores"]["tool_usage"] == expected["tool_usage"]
                    assert run["scores"]["iterations_ok"] == expected["iterations_ok"]
                    assert run["avg_score"] == expected["avg_score"]
                    break
            assert found

        # Test: overall_avg is mean of all run avg_scores
        # (1.0 + 0.83 + 0.50 + 0.17) / 4 = 2.5 / 4 = 0.625 -> rounds to 0.62
        assert results["overall_avg"] == 0.62

        # Clean up
        results_file.unlink()
