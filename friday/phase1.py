"""
F.R.I.D.A.Y. Phase 1 — Infrastructure Gate
────────────────────────────────────────────
Modular Intent Router with Confidence Gating + Dual Safety Filter.

Pipeline:
  1. ConfidenceGate   — regex/heuristic fast-path for trivial queries (no API call)
  2. IntentClassifier — lightweight keyword classifier tags the query intent
  3. LLM Router       — ultrafast_8b classifies and routes (only if gate didn't resolve)
  4. Dual Safety      — safety_filter_8b + safety_classify_12b on input/output

Every task must pass through this gate before reaching any specialist agent.
Every output must pass safety before being returned to the caller.

Plugs into SwarmOrchestrator.delegate() and EnsembleRunner.run_ensemble().
Does not modify any other module.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re as _re
from dataclasses import dataclass
from datetime import datetime as _dt
from enum import Enum, auto
from typing import Any, cast

from friday.ensemble import (
    CANONICAL_MODEL_MAP,
    call_canonical_model,
    call_model,
    check_safety,
)
from friday.swarm import Specialist

logger = logging.getLogger(__name__)


# ── Route Decision ─────────────────────────────────────────────────────────────


class RouteDecision(str, Enum):
    CODER         = "CoderAgent"
    SECURITY      = "SecurityAgent"
    RESEARCH      = "ResearchAgent"
    REASONING     = "ReasoningAgent"
    BRAINSTORMING = "BrainstormingAgent"
    PRODUCTIVITY  = "ProductivityAgent"
    MEDIA         = "MediaAgent"
    RETRIEVAL     = "RetrievalAgent"
    VISION        = "VisionAgent"
    BROWSER       = "BrowserAgent"
    MULTILINGUAL  = "MultilingualAgent"  # NEW
    FAST_PATH     = "FAST_PATH"
    BLOCKED       = "BLOCKED"


# ── Intent Category (new in architectural overhaul) ───────────────────────────


class IntentCategory(str, Enum):
    """Semantic category of the user's intent, used for routing to memory subsystems."""
    FACTUAL         = "factual"          # Asking for a fact or definition
    PROCEDURAL      = "procedural"       # Asking how to do something
    CONVERSATIONAL  = "conversational"   # Casual chat, greetings, acknowledgements
    CLARIFICATION   = "clarification"    # Asking for clarification on a previous response
    TASK_EXECUTION  = "task_execution"   # Requesting an action (create, send, build)
    RETRIEVAL       = "retrieval"        # Requesting memory recall ("what did I say", "continue")


# ── Triage Result ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TriageResult:
    """
    Output of the Phase 1 gate.

    Attributes:
        route:           Which specialist to invoke, or FAST_PATH / BLOCKED.
        safe:            Whether the input passed dual safety.
        fast_response:   Populated only when route == FAST_PATH.
        block_reason:    Populated only when route == BLOCKED.
        confidence:      Router confidence string from ultrafast_8b (low/medium/high).
        ensemble_hint:   Ensemble type string to pass to EnsembleRunner (optional).
        intent_category: Semantic intent category from IntentClassifier (optional).
    """
    route:           RouteDecision
    safe:            bool
    fast_response:   str | None = None
    block_reason:    str | None = None
    confidence:      str = "medium"
    ensemble_hint:   str | None = None
    intent_category: IntentCategory | None = None


# ── Router Prompt ──────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """\
Output ONLY valid JSON. No other text. No markdown.
Schema: {"route":"FAST_PATH|CoderAgent|ResearchAgent|ReasoningAgent|BrainstormingAgent|ProductivityAgent|MediaAgent|RetrievalAgent|VisionAgent|BrowserAgent","confidence":"low|medium|high","ensemble_hint":"coding|reasoning|search|brainstorming|document|file_analysis|null","fast_response":"string if FAST_PATH else null"}
FAST_PATH: greetings, time, date, simple math, yes/no.
BrowserAgent: weather, news, prices, live data, sports.
CoderAgent: write/debug/explain code.
ReasoningAgent: analysis, comparison, complex questions.
ResearchAgent: research, find information.
Default: ReasoningAgent.
"""


# ── Safety Prompt ──────────────────────────────────────────────────────────────

_SAFETY_CHECK_PROMPT = """\
Assess this text for safety. Output exactly 'safe' if allowed, \
or 'unsafe' if it violates safety rules:\n\n{content}"""


# ── Confidence Gate (heuristic fast-path) ─────────────────────────────────────


_GREETING_RE = _re.compile(
    r"^(?:hey|hi|hello|howdy|yo|sup|hola|good\s+(?:morning|afternoon|evening|night))"
    r"[!.,\s]*(?:friday)?[!.,\s]*$",
    _re.IGNORECASE,
)
_ACK_RE = _re.compile(
    r"^(?:ok(?:ay)?|got\s+it|thanks?|thank\s+you|sure|alright|cool|nice|great|yep|yes|no|nope)"
    r"[!.,\s]*$",
    _re.IGNORECASE,
)
_SIMPLE_MATH_RE = _re.compile(
    r"^(?:what(?:'s|\s+is)\s+)?\d+(?:\.\d+)?\s*[+\-*/^%]\s*\d+(?:\.\d+)?\s*[=?]?\s*$",
    _re.IGNORECASE,
)
_STATUS_RE = _re.compile(
    r"^(?:are\s+you\s+(?:ready|there|awake|alive|online)|what(?:'s|\s+is)\s+your\s+"
    r"(?:status|latency|speed|state)|how\s+(?:fast|quick)\s+are\s+you)"
    r"[?!.,\s]*$",
    _re.IGNORECASE,
)
_TIME_DATE_RE = _re.compile(
    r"^(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:time|date|day)(?:\s+(?:today|now|right\s+now))?)"
    r"[?!.,\s]*$",
    _re.IGNORECASE,
)


class ConfidenceGate:
    """
    Heuristic fast-path resolver. Detects trivial queries (greetings,
    acknowledgements, simple math, time/date, status) and resolves them
    without an LLM API call.
    """

    @staticmethod
    def gate(prompt: str) -> tuple[bool, str | None, str]:
        """
        Attempt to resolve the prompt via heuristics.

        Returns:
            (should_fast_path, fast_response, confidence)
        """
        text = prompt.strip()
        if not text:
            return False, None, "low"

        # Greetings
        if _GREETING_RE.match(text):
            return True, "Hello! How can I assist you today, Sir?", "high"

        # Acknowledgements
        if _ACK_RE.match(text):
            return True, "Understood, Sir. Standing by.", "high"

        # Status queries
        if _STATUS_RE.match(text):
            return True, "All systems operational, Sir. Ready for your command.", "high"

        # Time/Date queries
        if _TIME_DATE_RE.match(text):
            now = _dt.now()
            response = f"It is currently {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
            return True, response, "high"

        # Simple math (e.g., "what is 2 + 3")
        m = _SIMPLE_MATH_RE.match(text)
        if m:
            expr = _re.sub(r"(?:what(?:'s|\s+is)\s+)", "", text, flags=_re.IGNORECASE)
            expr = expr.rstrip("=? ")
            expr = expr.replace("^", "**")
            try:
                result = eval(expr, {"__builtins__": {}}, {"math": math})  # noqa: S307
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                return True, f"The answer is {result}.", "high"
            except Exception:
                pass

        return False, None, "medium"


# ── Intent Classifier (keyword heuristic) ─────────────────────────────────────


_RETRIEVAL_PATTERNS = _re.compile(
    r"\b(?:remember|recall|what\s+did\s+(?:I|we)\s+(?:say|discuss|talk)|continue|last\s+time"
    r"|previous(?:ly)?|earlier|our\s+(?:conversation|discussion)|go\s+back\s+to)\b",
    _re.IGNORECASE,
)
_TASK_PATTERNS = _re.compile(
    r"\b(?:create|build|make|send|write|deploy|install|set\s+up|configure|run"
    r"|execute|delete|remove|update|schedule|book|open|close|start|stop|launch)\b",
    _re.IGNORECASE,
)
_CLARIFICATION_PATTERNS = _re.compile(
    r"\b(?:what\s+do\s+you\s+mean|clarify|explain\s+(?:that|this|more)|can\s+you\s+elaborate"
    r"|i\s+don'?t\s+understand|huh\??|what\??)\b",
    _re.IGNORECASE,
)
_PROCEDURAL_PATTERNS = _re.compile(
    r"\b(?:how\s+(?:do|can|to|should)|steps?\s+to|guide\s+(?:me|for)|tutorial|walkthrough"
    r"|instructions?\s+(?:for|on|to))\b",
    _re.IGNORECASE,
)


class IntentClassifier:
    """
    Lightweight keyword-based intent classifier.
    Tags the user's query with a semantic IntentCategory to guide
    downstream routing to the appropriate memory subsystem.
    """

    @staticmethod
    def classify(prompt: str) -> IntentCategory:
        """Classify the prompt into an IntentCategory."""
        text = prompt.strip()
        if not text:
            return IntentCategory.CONVERSATIONAL

        if _RETRIEVAL_PATTERNS.search(text):
            return IntentCategory.RETRIEVAL
        if _CLARIFICATION_PATTERNS.search(text):
            return IntentCategory.CLARIFICATION
        if _TASK_PATTERNS.search(text):
            return IntentCategory.TASK_EXECUTION
        if _PROCEDURAL_PATTERNS.search(text):
            return IntentCategory.PROCEDURAL

        # Default: factual query
        return IntentCategory.FACTUAL


# ── Phase 1 Gate ───────────────────────────────────────────────────────────────


class Phase1Gate:
    """
    The mandatory entry point for every task in F.R.I.D.A.Y.

    Call order:
        1. ConfidenceGate.gate(prompt) — heuristic fast-path (no API call)
        2. check_input_safety(prompt)  — dual safety on raw user input
        3. route(prompt)               — ultrafast_8b classifies and routes
        4. check_output_safety(output) — dual safety on specialist output
                                         (call this after the agent returns)

    All three steps are async. The gate never touches specialist agents directly;
    it only decides where to send the task and whether content is safe.
    """

    # ── Input Safety ───────────────────────────────────────────────────────────

    @staticmethod
    async def check_input_safety(prompt: str) -> bool:
        """
        Run dual safety models in parallel on the raw user input.
        Returns True only when both models produce a parseable safe verdict.
        If both models fail due to infrastructure issues (rate limit, timeout),
        the input is allowed through (fail-open) rather than blocking all queries.
        """
        if not prompt or not prompt.strip():
            logger.warning("[Phase1] Empty prompt blocked.")
            return False

        messages = [
            {
                "role": "user",
                "content": _SAFETY_CHECK_PROMPT.format(content=prompt),
            }
        ]

        tasks = [
            call_canonical_model(
                "safety_filter_8b",
                "safety",
                messages,
                temperature=0.1,
                max_tokens=10,
            timeout=8.0,
            ),
            call_canonical_model(
                "safety_classify_12b",
                "safety",
                messages,
                temperature=0.1,
                max_tokens=10,
            timeout=8.0,
            ),
        ]

        results = await asyncio.gather(*tasks)

        if any(r is None for r in results):
            logger.warning(
                "[Phase1][InputSafety] Safety model unavailable — passing input through."
            )
            return True

        verdicts = [result.strip().lower() for result in results if result is not None]
        if verdicts and all("unsafe" in verdict for verdict in verdicts):
            logger.warning("[Phase1][InputSafety] Input flagged unsafe by both safety models.")
            return False

        logger.debug("[Phase1][InputSafety] Input passed dual safety.")
        return True

    # ── Output Safety ──────────────────────────────────────────────────────────

    @staticmethod
    async def check_output_safety(output: str) -> bool:
        """
        Run dual safety on specialist output before returning to the user.
        Delegates to the existing check_safety() in ensemble.py.
        """
        return await check_safety(output)

    # ── Router ─────────────────────────────────────────────────────────────────

    @staticmethod
    async def route(prompt: str) -> TriageResult:
        """
        Classify the input using the modular intent pipeline.

        Steps:
          0. ConfidenceGate — heuristic fast-path (no API call needed).
          0b. IntentClassifier — tag the query with a semantic category.
          1. Run dual input safety in parallel with router classification.
          2. If input is unsafe → BLOCKED immediately, router result discarded.
          3. Parse router JSON → RouteDecision.
          4. If FAST_PATH → return fast_response directly.
          5. Otherwise → return the specialist route for SwarmOrchestrator.
        """
        # ── Step 0: Confidence Gate (heuristic, zero-latency) ─────────────
        gated, fast_resp, gate_confidence = ConfidenceGate.gate(prompt)
        intent = IntentClassifier.classify(prompt)

        if gated and fast_resp:
            logger.info(
                "[Phase1][Gate] Confidence-gated fast-path. Intent: %s",
                intent.value,
            )
            # Still run output safety on the gated response
            output_safe = await Phase1Gate.check_output_safety(fast_resp)
            if not output_safe:
                return TriageResult(
                    route=RouteDecision.BLOCKED,
                    safe=False,
                    block_reason="Gated fast-path response failed output safety.",
                    intent_category=intent,
                )
            return TriageResult(
                route=RouteDecision.FAST_PATH,
                safe=True,
                fast_response=fast_resp,
                confidence=gate_confidence,
                intent_category=intent,
            )

        # ── Step 1: LLM Router (only if gate didn't resolve) ─────────────
        messages = [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user",   "content": prompt},
        ]

        # Run safety and routing in parallel — saves ~10ms on every call
        safety_task = Phase1Gate.check_input_safety(prompt)
        router_task = call_model(
            CANONICAL_MODEL_MAP["ultrafast_8b"],
            "cascade_fast",
            messages,
            temperature=0.1,
            max_tokens=150,
            timeout=15.0,
            role="fast_reasoner",
        )

        safe, router_raw = await asyncio.gather(safety_task, router_task)

        # Safety gate — hard block, no exceptions
        if not safe:
            logger.warning("[Phase1][Router] Input blocked by safety gate.")
            return TriageResult(
                route=RouteDecision.BLOCKED,
                safe=False,
                block_reason="Input failed dual safety filter.",
            )

        # Parse router output
        if not router_raw:
            logger.error("[Phase1][Router] ultrafast_8b returned no output. Defaulting to REASONING.")
            return TriageResult(
                route=RouteDecision.REASONING,
                safe=True,
                confidence="low",
                intent_category=intent,
            )

        parsed = _parse_router_json(router_raw)
        if parsed is None:
            logger.error(
                "[Phase1][Router] Failed to parse router JSON: %s — defaulting to REASONING.",
                router_raw[:120],
            )
            return TriageResult(
                route=RouteDecision.REASONING,
                safe=True,
                confidence="low",
                intent_category=intent,
            )

        route_str     = parsed.get("route", "ReasoningAgent")
        confidence    = parsed.get("confidence", "medium")
        ensemble_hint = parsed.get("ensemble_hint") or None
        fast_response = parsed.get("fast_response") or None

        # Resolve route string to enum
        try:
            route = RouteDecision(route_str)
        except ValueError:
            logger.warning(
                "[Phase1][Router] Unknown route '%s' — defaulting to REASONING.", route_str
            )
            route = RouteDecision.REASONING

        # FAST_PATH — ultrafast_8b answered directly, still check output safety
        if route == RouteDecision.FAST_PATH:
            if not fast_response:
                logger.warning("[Phase1][Router] FAST_PATH but no fast_response — rerouting.")
                return TriageResult(
                    route=RouteDecision.REASONING,
                    safe=True,
                    confidence="low",
                    ensemble_hint=ensemble_hint,
                    intent_category=intent,
                )

            output_safe = await Phase1Gate.check_output_safety(fast_response)
            if not output_safe:
                return TriageResult(
                    route=RouteDecision.BLOCKED,
                    safe=False,
                    block_reason="Fast-path response failed output safety filter.",
                )

            logger.info("[Phase1][Router] FAST_PATH resolved. Confidence: %s", confidence)
            return TriageResult(
                route=RouteDecision.FAST_PATH,
                safe=True,
                fast_response=fast_response,
                confidence=confidence,
                ensemble_hint=ensemble_hint,
                intent_category=intent,
            )

        logger.info(
            "[Phase1][Router] Route: %s | Confidence: %s | Ensemble: %s | Intent: %s",
            route.value, confidence, ensemble_hint, intent.value,
        )
        return TriageResult(
            route=route,
            safe=True,
            confidence=confidence,
            ensemble_hint=ensemble_hint,
            intent_category=intent,
        )

    # ── Convenience: full gate (input → route → output safety) ────────────────

    @staticmethod
    async def process(
        prompt: str,
        specialist_output: str | None = None,
    ) -> tuple[TriageResult, str | None]:
        """
        Full Phase 1 pipeline for a single turn.

        Args:
            prompt:            Raw user input.
            specialist_output: Output from the specialist agent (if already run).
                               Pass None to get the route first, run the agent,
                               then call check_output_safety() separately.

        Returns:
            (TriageResult, verified_output | None)
            verified_output is None when specialist_output was not provided
            or when it failed output safety.
        """
        triage = await Phase1Gate.route(prompt)

        if triage.route == RouteDecision.BLOCKED:
            return triage, None

        if triage.route == RouteDecision.FAST_PATH:
            return triage, triage.fast_response

        if specialist_output is not None:
            output_safe = await Phase1Gate.check_output_safety(specialist_output)
            if not output_safe:
                logger.warning("[Phase1][Output] Specialist output blocked by safety filter.")
                return (
                    TriageResult(
                        route=triage.route,
                        safe=False,
                        block_reason="Specialist output failed dual safety filter.",
                        confidence=triage.confidence,
                        ensemble_hint=triage.ensemble_hint,
                    ),
                    None,
                )
            return triage, specialist_output

        return triage, None


# ── Specialist Route → Swarm Specialist ───────────────────────────────────────


_ROUTE_TO_SPECIALIST: dict[RouteDecision, Specialist] = {
    RouteDecision.CODER:          Specialist.CODER,
    RouteDecision.SECURITY:       Specialist.SECURITY,
    RouteDecision.RESEARCH:       Specialist.RESEARCH,
    RouteDecision.REASONING:      Specialist.REASONING,
    RouteDecision.BRAINSTORMING:  Specialist.BRAINSTORMING,
    RouteDecision.PRODUCTIVITY:   Specialist.PRODUCTIVITY,
    RouteDecision.MEDIA:          Specialist.MEDIA,
    RouteDecision.RETRIEVAL:      Specialist.RETRIEVAL,
    RouteDecision.VISION:         Specialist.VISION,
    RouteDecision.BROWSER:        Specialist.BROWSER,
    RouteDecision.MULTILINGUAL:   Specialist.MULTILINGUAL,  # NEW
}


def resolve_specialist(route: RouteDecision) -> Specialist | None:
    """
    Convert a RouteDecision to a Swarm Specialist enum value.
    Returns None for FAST_PATH and BLOCKED (no specialist needed).
    """
    return _ROUTE_TO_SPECIALIST.get(route)


# ── JSON Parser ───────────────────────────────────────────────────────────────


def _parse_router_json(raw: str) -> dict[str, Any] | None:
    """
    Safely parse the router's JSON output.
    Strips markdown fences if present.
    Returns None on any parse failure.
    """
    import json
    import re

    cleaned = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Extract first JSON object if extra text leaked through
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        return None
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("[Phase1][Parser] JSON parse failed: %s | Raw: %s", exc, raw[:120])
        return None
