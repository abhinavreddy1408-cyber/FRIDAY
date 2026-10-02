"""
Lightweight evaluation harness for F.R.I.D.A.Y.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from friday.react_loop import ReActLoop

EVAL_PROMPTS = [
    "Plan a 5-day trip to Italy next summer on a $2000 budget.",
    "Summarize recent developments in AI ethics.",
    "Set up a team meeting next Thursday morning.",
    "Improve the writing: The cat sat. It was big. It ate food.",
]


def run_eval() -> dict[str, Any]:
    """Runs evaluation on EVAL_PROMPTS, calculates scores, and saves results.json."""
    runs = []
    
    for prompt in EVAL_PROMPTS:
        try:
            loop = ReActLoop()
            result = loop.run(prompt)
            
            final_answer = result.final_answer
            iterations = result.iterations
            trace = result.trace
            
            cot_trace = [t["thought"] for t in trace if t.get("phase") == "reason"]
            tool_calls = [t["action"] for t in trace if t.get("phase") == "act"]
            
            completeness = 1.0 if (final_answer and str(final_answer).strip()) else 0.0
            tool_usage = 1.0 if len(tool_calls) >= 1 else 0.5
            iterations_ok = 1.0 if iterations <= 5 else 0.0
        except Exception as exc:
            final_answer = f"Error: {exc}"
            iterations = 0
            cot_trace = []
            tool_calls = []
            completeness = 0.0
            tool_usage = 0.5
            iterations_ok = 0.0
            
        scores = {
            "completeness": completeness,
            "tool_usage": tool_usage,
            "iterations_ok": iterations_ok,
        }
        
        avg_score = round(sum(scores.values()) / len(scores), 2)
        
        runs.append({
            "prompt": prompt,
            "scores": scores,
            "avg_score": avg_score,
            "cot_trace": cot_trace,
            "tool_calls": tool_calls,
            "final_answer": final_answer,
            "iterations": iterations,
        })
        
    overall_avg = round(sum(run["avg_score"] for run in runs) / len(runs), 2) if runs else 0.0
    
    results = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runs": runs,
        "overall_avg": overall_avg,
    }
    
    harness_dir = Path(__file__).resolve().parent
    results_path = harness_dir / "results.json"
    
    # Save results.json
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        
    # Print summary table
    print("=" * 90)
    print(f"{'F.R.I.D.A.Y. Evaluation Summary':^90}")
    print("=" * 90)
    print(f"{'Prompt':<50} | {'Comp':<5} | {'Tool':<5} | {'IterOK':<6} | {'Avg':<5} | {'Iters':<5}")
    print("-" * 90)
    for run in runs:
        short_prompt = run["prompt"]
        if len(short_prompt) > 47:
            short_prompt = short_prompt[:44] + "..."
        s = run["scores"]
        print(
            f"{short_prompt:<50} | {s['completeness']:<5.1f} | {s['tool_usage']:<5.1f} | "
            f"{s['iterations_ok']:<6.1f} | {run['avg_score']:<5.2f} | {run['iterations']:<5}"
        )
    print("-" * 90)
    print(f"{'Overall Average:':<73} {overall_avg:<5.2f}")
    print("=" * 90)
    
    return results


if __name__ == "__main__":
    run_eval()
