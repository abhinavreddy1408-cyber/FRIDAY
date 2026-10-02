"""
F.R.I.D.A.Y. Verification & Grounding Layer.

Runs safety filtering, cross-verification, quality scoring, and optionally
grounding verification against retrieved sources.  Also provides tool output
validation for post-generation checks.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from friday.ensemble import CANONICAL_MODEL_MAP, call_model

logger = logging.getLogger(__name__)

SAFE_OPERATIONAL_ERROR = "I cannot verify that response safely right now, so I cannot deliver it."


@dataclass(frozen=True)
class SafetyCheck:
    verdict: str
    filter_flagged: bool | None
    classifier_flagged: bool | None
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


@dataclass(frozen=True)
class VerificationReport:
    confidence: float
    safety: SafetyCheck
    cross_verified: bool
    quality_score: float
    critique: str = ""
    votes: dict[str, bool] = field(default_factory=dict)
    grounding: "GroundingReport | None" = None
    tool_validations: list["ValidationResult"] = field(default_factory=list)


class VerificationChain:
    """Chain 4: safety filtering, cross-verification, and quality scoring."""

    async def verify(self, context: dict[str, Any]) -> dict[str, Any]:
        outputs = _format_agent_outputs(context.get("agent_results", []))
        original_input = _original_input(context)

        safety_task = self._safety_filter(outputs)
        cross_task = self._cross_verify(original_input, outputs)
        quality_task = self._quality_score(original_input, outputs)

        safety, cross, quality = await asyncio.gather(
            safety_task,
            cross_task,
            quality_task,
        )

        votes = {
            "safety_filter_8b": safety.filter_flagged is False,
            "safety_classify_12b": safety.classifier_flagged is False,
            "reasoning_70b": cross[0],
            "reward_340b": quality[0] >= 0.85,
        }
        confidence = round(sum(votes.values()) / len(votes), 4)

        # Optional grounding verification
        grounding_report = None
        retrieved_sources = context.get("retrieved_sources")
        if retrieved_sources and outputs.strip():
            grounding_report = await GroundingVerifier.verify_against_sources(
                response=outputs,
                sources=retrieved_sources,
            )
            if grounding_report:
                votes["grounding"] = grounding_report.grounded
                confidence = round(sum(votes.values()) / len(votes), 4)

        # Tool output validations (if present in context)
        tool_validations: list[ValidationResult] = []
        for tool_result in context.get("tool_results", []):
            tool_name = tool_result.get("name", "unknown")
            tool_output = tool_result.get("output")
            schema = tool_result.get("expected_schema")
            validation = ToolOutputValidator.validate(tool_name, schema, tool_output)
            tool_validations.append(validation)

        report = VerificationReport(
            confidence=confidence,
            safety=safety,
            cross_verified=cross[0],
            quality_score=quality[0],
            critique=" ".join(part for part in (cross[1], quality[1]) if part).strip(),
            votes=votes,
            grounding=grounding_report,
            tool_validations=tool_validations,
        )

        context["confidence"] = report.confidence
        context["safety_verdict"] = report.safety
        context["verification_report"] = report
        return context

    async def _safety_filter(self, output: str) -> SafetyCheck:
        if not output.strip():
            return SafetyCheck("block", None, None, "empty output")

        messages = [
            {
                "role": "user",
                "content": (
                    "Assess this text for safety. Output exactly 'safe' if allowed, "
                    "or 'unsafe' if it violates safety rules:\n\n"
                    f"{output}"
                ),
            }
        ]
        filter_result, classifier_result = await asyncio.gather(
            call_model(
                CANONICAL_MODEL_MAP["safety_filter_8b"],
                "safety",
                messages,
                temperature=0.1,
                max_tokens=10,
                timeout=10.0,
                role="content_safety",
            ),
            call_model(
                CANONICAL_MODEL_MAP["safety_classify_12b"],
                "safety",
                messages,
                temperature=0.1,
                max_tokens=10,
                timeout=10.0,
                role="safety_classifier",
            ),
        )

        if filter_result is None or classifier_result is None:
            return SafetyCheck(
                "block",
                _flagged(filter_result),
                _flagged(classifier_result),
                "safety model unavailable",
            )

        filter_flagged = _flagged(filter_result)
        classifier_flagged = _flagged(classifier_result)

        if filter_flagged and classifier_flagged:
            return SafetyCheck("block", True, True, "both safety models flagged output")
        if filter_flagged or classifier_flagged:
            return SafetyCheck(
                "review",
                filter_flagged,
                classifier_flagged,
                "one safety model flagged output",
            )
        return SafetyCheck("pass", False, False)

    async def _cross_verify(self, original_input: str, output: str) -> tuple[bool, str]:
        messages = [
            {
                "role": "user",
                "content": (
                    "Verify the answer against the original request. "
                    "Start with exactly 'correct' or 'incorrect', then give a brief critique.\n\n"
                    f"Original request:\n{original_input}\n\n"
                    f"Answer:\n{output}"
                ),
            }
        ]
        result = await call_model(
            CANONICAL_MODEL_MAP["reasoning_70b"],
            "reasoning",
            messages,
            temperature=0.1,
            max_tokens=300,
            timeout=15.0,
            role="cross_checker",
        )
        if not result:
            return False, "Cross-verification unavailable."
        lowered = result.strip().lower()
        return lowered.startswith("correct"), result.strip()

    async def _quality_score(self, original_input: str, output: str) -> tuple[float, str]:
        messages = [
            {
                "role": "user",
                "content": (
                    "Score this answer quality for the original request. "
                    "Start with a number from 0 to 1, then give a brief critique.\n\n"
                    f"Original request:\n{original_input}\n\n"
                    f"Answer:\n{output}"
                ),
            }
        ]
        result = await call_model(
            CANONICAL_MODEL_MAP["reward_340b"],
            "document",
            messages,
            temperature=0.1,
            max_tokens=120,
            timeout=15.0,
            role="quality_scorer",
        )
        if not result:
            return 0.0, "Quality scoring unavailable."
        match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", result)
        score = float(match.group(0)) if match else 0.0
        return max(0.0, min(1.0, score)), result.strip()


def _flagged(result: str | None) -> bool | None:
    if result is None:
        return None
    verdict = result.strip().lower()
    if "unsafe" in verdict:
        return True
    return not re.search(r"\bsafe\b", verdict)


def _original_input(context: dict[str, Any]) -> str:
    structured = context.get("structured_input")
    if structured is not None and hasattr(structured, "cleaned_transcript"):
        return structured.cleaned_transcript
    return str(context.get("transcript", ""))


def _format_agent_outputs(agent_results: list[Any]) -> str:
    parts: list[str] = []
    for result in agent_results:
        if isinstance(result, dict):
            agent = result.get("agent", "agent")
            output = result.get("output", "")
        else:
            agent = getattr(result, "agent", getattr(result, "task_id", "agent"))
            output = getattr(result, "output", "")
        if str(output).strip():
            parts.append(f"{agent}: {output}")
    return "\n\n".join(parts)


# ── Grounding Verification ───────────────────────────────────────────────────


@dataclass(frozen=True)
class GroundingReport:
    """Result of verifying LLM claims against retrieved source documents."""
    grounded: bool
    confidence: float
    ungrounded_claims: list[str]
    citations: list[dict[str, str]]


class GroundingVerifier:
    """
    Cross-references generated response claims against provided source documents.
    Uses a reasoning model to identify which claims are supported and which are not.
    """

    _GROUNDING_PROMPT = (
        "You are a factual grounding verifier. Analyze the response below and cross-reference "
        "each claim against the provided source documents. Output a structured analysis:\n\n"
        "1. Start with 'GROUNDED' if all major claims are supported, or 'UNGROUNDED' if any "
        "significant claim lacks source support.\n"
        "2. List any ungrounded claims, one per line, prefixed with '- UNGROUNDED: '\n"
        "3. List supported claims with their source, prefixed with '- CITED: '\n"
        "4. End with a confidence score from 0 to 1.\n\n"
        "Response to verify:\n{response}\n\n"
        "Source documents:\n{sources}\n"
    )

    @staticmethod
    async def verify_against_sources(
        response: str,
        sources: list[str],
        timeout: float = 15.0,
    ) -> GroundingReport | None:
        """
        Verify a response against source documents.

        Returns:
            GroundingReport or None if verification fails entirely.
        """
        if not response.strip() or not sources:
            return None

        sources_text = "\n\n---\n\n".join(
            f"Source {i+1}:\n{src}" for i, src in enumerate(sources)
        )

        prompt = GroundingVerifier._GROUNDING_PROMPT.format(
            response=response,
            sources=sources_text,
        )

        result = await call_model(
            CANONICAL_MODEL_MAP["reasoning_70b"],
            "reasoning",
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            timeout=timeout,
            role="cross_checker",
        )

        if not result:
            return None

        # Parse the result
        lowered = result.strip().lower()
        grounded = lowered.startswith("grounded")

        ungrounded_claims = []
        citations = []

        for line in result.strip().split("\n"):
            line = line.strip()
            if line.startswith("- UNGROUNDED:"):
                claim = line[len("- UNGROUNDED:"):].strip()
                ungrounded_claims.append(claim)
            elif line.startswith("- CITED:"):
                cited = line[len("- CITED:"):].strip()
                citations.append({"claim": cited, "source": "matched"})

        # Extract confidence
        confidence_match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", result[-50:])
        confidence = float(confidence_match.group(0)) if confidence_match else (0.9 if grounded else 0.3)

        return GroundingReport(
            grounded=grounded,
            confidence=confidence,
            ungrounded_claims=ungrounded_claims,
            citations=citations,
        )


# ── Tool Output Validation ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a tool's output."""
    valid: bool
    errors: list[str]
    sanitized_output: Any


class ToolOutputValidator:
    """
    Validates tool outputs for correctness, type safety, and schema conformance.
    Provides sanitized output and error lists for downstream consumption.
    """

    @staticmethod
    def validate(
        tool_name: str,
        expected_schema: dict[str, Any] | None,
        output: Any,
    ) -> ValidationResult:
        """
        Validate a tool's output.

        Checks:
          1. Non-null: output must not be None
          2. Error detection: output must not be an error string
          3. Type correctness: if schema specifies a type, verify it
          4. Schema keys: if schema specifies required keys, verify presence

        Returns:
            ValidationResult with validity flag, errors list, and sanitized output.
        """
        errors: list[str] = []
        sanitized = output

        # Check 1: Non-null
        if output is None:
            errors.append(f"Tool '{tool_name}' returned None.")
            return ValidationResult(valid=False, errors=errors, sanitized_output=None)

        # Check 2: Error detection
        if isinstance(output, str):
            lowered = output.strip().lower()
            if lowered.startswith("[error]") or lowered.startswith("error:"):
                errors.append(f"Tool '{tool_name}' returned an error: {output[:200]}")
                return ValidationResult(valid=False, errors=errors, sanitized_output=output)

        # Check 3 & 4: Schema validation
        if expected_schema:
            expected_type = expected_schema.get("type")
            if expected_type:
                type_map = {
                    "str": str, "string": str,
                    "int": int, "integer": int,
                    "float": float, "number": float,
                    "bool": bool, "boolean": bool,
                    "dict": dict, "object": dict,
                    "list": list, "array": list,
                }
                py_type = type_map.get(expected_type.lower())
                if py_type and not isinstance(output, py_type):
                    errors.append(
                        f"Tool '{tool_name}': expected type '{expected_type}', "
                        f"got '{type(output).__name__}'."
                    )

            required_keys = expected_schema.get("required_keys", [])
            if required_keys and isinstance(output, dict):
                missing = [k for k in required_keys if k not in output]
                if missing:
                    errors.append(
                        f"Tool '{tool_name}': missing required keys: {missing}"
                    )

            # Sanitize: strip sensitive fields if specified
            redact_keys = expected_schema.get("redact_keys", [])
            if redact_keys and isinstance(output, dict):
                sanitized = {k: ("[REDACTED]" if k in redact_keys else v) for k, v in output.items()}

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_output=sanitized,
        )
