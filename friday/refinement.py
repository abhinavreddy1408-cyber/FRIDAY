"""
F.R.I.D.A.Y. Refinement Chain.

Implements the PDF-specified confidence threshold, +0.05 improvement rule,
maximum three iterations, and full-history ChainContext updates.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RefinementStep:
    iteration: int
    previous_confidence: float
    new_confidence: float
    improvement: float
    model_tier: str
    accepted: bool


RefinementExecutor = Callable[[dict[str, Any], int, str], Awaitable[dict[str, Any]]]


class RefinementChain:
    """Chain 5: iterative improvement with residual ChainContext access."""

    def __init__(
        self,
        accept_threshold: float = 0.85,
        min_improvement: float = 0.05,
        max_iterations: int = 3,
    ) -> None:
        self.accept_threshold = accept_threshold
        self.min_improvement = min_improvement
        self.max_iterations = max_iterations

    async def refine(
        self,
        context: dict[str, Any],
        executor: RefinementExecutor,
    ) -> dict[str, Any]:
        history: list[RefinementStep] = context.setdefault("refinement_history", [])
        best_confidence = float(context.get("confidence", 0.0) or 0.0)

        if best_confidence >= self.accept_threshold:
            return context

        for iteration in range(1, self.max_iterations + 1):
            previous_confidence = best_confidence
            tier = _tier_for_iteration(iteration)
            context["refinement_instruction"] = build_refinement_prompt(context)
            context["refinement_model_tier"] = tier

            candidate = await executor(context, iteration, tier)
            new_confidence = float(candidate.get("confidence", 0.0) or 0.0)
            improvement = round(new_confidence - previous_confidence, 4)
            accepted = improvement >= self.min_improvement

            step = RefinementStep(
                iteration=iteration,
                previous_confidence=round(previous_confidence, 4),
                new_confidence=round(new_confidence, 4),
                improvement=improvement,
                model_tier=tier,
                accepted=accepted,
            )
            history.append(step)

            if not accepted:
                break

            context.update(candidate)
            best_confidence = new_confidence

            if best_confidence >= self.accept_threshold:
                break

        return context


def build_refinement_prompt(context: dict[str, Any]) -> str:
    """Build the residual-context improvement prompt for the next iteration."""
    structured = context.get("structured_input")
    original = (
        structured.cleaned_transcript
        if structured is not None and hasattr(structured, "cleaned_transcript")
        else str(context.get("transcript", ""))
    )
    outputs = []
    for result in context.get("agent_results", []):
        if isinstance(result, dict):
            output = result.get("output", "")
        else:
            output = getattr(result, "output", "")
        if str(output).strip():
            outputs.append(str(output))
    critique = getattr(context.get("verification_report"), "critique", "")
    return (
        "Improve the current answer by addressing the verification critique.\n\n"
        f"Original input:\n{original}\n\n"
        f"Current output:\n{chr(10).join(outputs)}\n\n"
        f"Verification critique:\n{critique}"
    )


def _tier_for_iteration(iteration: int) -> str:
    if iteration <= 1:
        return "medium"
    return "flagship"
