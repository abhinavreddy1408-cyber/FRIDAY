"""
Tests for friday.phase1 — Phase 1 Gate (Router + Dual Safety).

Covers the six explicit scenarios requested:
  1. BLOCKED route when safety models return "unsafe"
  2. FAST_PATH route with fast_response populated
  3. Valid specialist route (e.g. BrowserAgent)
  4. Router JSON parse failure -> defaults to REASONING
  5. Input safety fail-open when both models return None
  6. Output safety blocking a specialist response

Plus auxiliary coverage for the JSON parser, resolve_specialist(), and
the FAST_PATH / no-response reroute branch so we comfortably exceed 80%.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from friday.phase1 import (
    ConfidenceGate,
    Phase1Gate,
    RouteDecision,
    TriageResult,
    _parse_router_json,
    resolve_specialist,
)
from friday.swarm import Specialist


# ── Helper builders ────────────────────────────────────────────────────────────


def _safety_mock(filter_verdict: str | None, classify_verdict: str | None) -> AsyncMock:
    """
    Build a mock for call_canonical_model that returns the supplied verdicts
    for the two safety models (in the order they're called by the gate).
    """
    side_effects = [filter_verdict, classify_verdict]
    return AsyncMock(side_effect=side_effects)


def _json_router_payload(**overrides) -> str:
    """Build a router JSON string the way the production model would emit it."""
    payload = {
        "route": "ReasoningAgent",
        "confidence": "medium",
        "ensemble_hint": None,
        "fast_response": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


# ── 1. BLOCKED route when safety models return "unsafe" ───────────────────────


@pytest.mark.asyncio
async def test_check_input_safety_one_unsafe_passes():
    """A single unsafe verdict does not block under the dual-confirmation policy."""
    mock_canonical = _safety_mock("unsafe", "safe")

    with patch("friday.phase1.call_canonical_model", mock_canonical):
        result = await Phase1Gate.check_input_safety("drop the table now")

    assert result is True
    assert mock_canonical.call_count == 2


@pytest.mark.asyncio
async def test_check_input_safety_other_model_unsafe_passes():
    """Either model alone cannot block under the dual-confirmation policy."""
    mock_canonical = _safety_mock("safe", "this is unsafe content")

    with patch("friday.phase1.call_canonical_model", mock_canonical):
        result = await Phase1Gate.check_input_safety("anything bad")

    assert result is True


@pytest.mark.asyncio
async def test_route_returns_blocked_when_safety_fails():
    """
    End-to-end: route() must short-circuit to BLOCKED when input safety
    flags the prompt, even if the router would have produced a valid verdict.
    """
    mock_canonical = _safety_mock("unsafe", "unsafe")
    mock_call_model = AsyncMock(return_value=_json_router_payload(route="BrowserAgent"))

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        result = await Phase1Gate.route("harm me please")

    assert isinstance(result, TriageResult)
    assert result.route is RouteDecision.BLOCKED
    assert result.safe is False
    assert result.block_reason is not None
    assert "safety" in result.block_reason.lower()
    # Router result is discarded once safety blocks — the spec doesn't forbid
    # the call but its output must be ignored.
    assert result.fast_response is None


@pytest.mark.asyncio
async def test_check_input_safety_rejects_empty_prompt():
    """Empty / whitespace prompts are blocked without calling any safety model."""
    mock_canonical = AsyncMock()

    with patch("friday.phase1.call_canonical_model", mock_canonical):
        ok = await Phase1Gate.check_input_safety("   ")
        bad = await Phase1Gate.check_input_safety("")

    assert ok is False
    assert bad is False
    assert mock_canonical.call_count == 0


# ── 2. FAST_PATH route with fast_response populated ───────────────────────────


@pytest.mark.asyncio
async def test_route_fast_path_returns_fast_response():
    """
    Router says FAST_PATH with a populated fast_response and output safety
    passes -> TriageResult.route == FAST_PATH and fast_response is set.
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(
            route="FAST_PATH",
            confidence="high",
            fast_response="Hello! I'm Friday.",
        )
    )
    mock_check_safety = AsyncMock(return_value=True)

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
        patch("friday.phase1.check_safety", mock_check_safety),
        patch.object(ConfidenceGate, "gate", return_value=(False, None, "medium")),
    ):
        result = await Phase1Gate.route("hey")

    assert result.route is RouteDecision.FAST_PATH
    assert result.safe is True
    assert result.fast_response == "Hello! I'm Friday."
    assert result.confidence == "high"
    assert result.block_reason is None


@pytest.mark.asyncio
async def test_route_fast_path_blocks_when_output_safety_fails():
    """
    FAST_PATH with a fast_response that fails output safety -> BLOCKED.
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(
            route="FAST_PATH",
            fast_response="some answer",
        )
    )
    mock_check_safety = AsyncMock(return_value=False)

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
        patch("friday.phase1.check_safety", mock_check_safety),
        patch.object(ConfidenceGate, "gate", return_value=(False, None, "medium")),
    ):
        result = await Phase1Gate.route("hi")

    assert result.route is RouteDecision.BLOCKED
    assert result.safe is False
    assert "fast-path" in (result.block_reason or "").lower()
    # The fast_response is NOT echoed back when output safety blocks it.
    assert result.fast_response is None


@pytest.mark.asyncio
async def test_route_fast_path_without_response_reroutes_to_reasoning():
    """
    Router says FAST_PATH but provides no fast_response -> reroute to
    REASONING rather than handing an empty answer to the caller.
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(
            route="FAST_PATH",
            confidence="low",
            fast_response=None,
        )
    )

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        result = await Phase1Gate.route("greetings")

    assert result.route is RouteDecision.REASONING
    assert result.confidence == "low"
    assert result.safe is True
    assert result.fast_response is None


# ── 3. Valid specialist route (e.g. BrowserAgent) ────────────────────────────


@pytest.mark.asyncio
async def test_route_returns_browser_specialist():
    """
    Router says BrowserAgent -> TriageResult.route == BROWSER, safe, and
    resolve_specialist() maps it to Specialist.BROWSER.
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(
            route="BrowserAgent",
            confidence="high",
            ensemble_hint="browser",
        )
    )

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        result = await Phase1Gate.route("what is the weather in Paris today")

    assert result.route is RouteDecision.BROWSER
    assert result.safe is True
    assert result.confidence == "high"
    assert result.ensemble_hint == "browser"
    assert result.fast_response is None
    assert result.block_reason is None

    assert resolve_specialist(result.route) is Specialist.BROWSER


@pytest.mark.asyncio
async def test_route_returns_coder_specialist():
    """CoderAgent -> RouteDecision.CODER + Specialist.CODER."""
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(route="CoderAgent", confidence="high")
    )

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        result = await Phase1Gate.route("write a quicksort in python")

    assert result.route is RouteDecision.CODER
    assert resolve_specialist(result.route) is Specialist.CODER


@pytest.mark.asyncio
async def test_route_returns_multilingual_specialist():
    """MultilingualAgent -> RouteDecision.MULTILINGUAL."""
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(
            route="MultilingualAgent", confidence="medium", ensemble_hint="multilingual"
        )
    )

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        result = await Phase1Gate.route("Bonjour, comment ça va?")

    assert result.route is RouteDecision.MULTILINGUAL
    assert resolve_specialist(result.route) is Specialist.MULTILINGUAL


# ── 4. Router JSON parse failure -> defaults to REASONING ─────────────────────


@pytest.mark.asyncio
async def test_route_invalid_json_defaults_to_reasoning():
    """
    Router returns garbage that is not parseable JSON -> TriageResult.route
    is REASONING with confidence='low'.
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(return_value="this is not json at all {{ broken")

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        result = await Phase1Gate.route("anything")

    assert result.route is RouteDecision.REASONING
    assert result.safe is True
    assert result.confidence == "low"
    assert result.fast_response is None


@pytest.mark.asyncio
async def test_route_empty_router_output_defaults_to_reasoning():
    """Empty router response (None / '') -> REASONING, low confidence."""
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(return_value="")

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        result = await Phase1Gate.route("anything")

    assert result.route is RouteDecision.REASONING
    assert result.confidence == "low"


@pytest.mark.asyncio
async def test_route_unknown_route_string_defaults_to_reasoning():
    """An unrecognised route value is treated like a parse failure -> REASONING."""
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(route="TotallyMadeUpAgent")
    )

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        result = await Phase1Gate.route("anything")

    assert result.route is RouteDecision.REASONING
    assert result.confidence == "medium"  # parsed, just an unknown value


# ── 5. Input safety fail-open when both models return None ───────────────────


@pytest.mark.asyncio
async def test_check_input_safety_fail_open_when_both_models_unavailable():
    """
    Both safety models return None (infrastructure failure) -> input is
    allowed through (fail-open) rather than blocking every prompt.
    """
    mock_canonical = AsyncMock(side_effect=[None, None])

    with patch("friday.phase1.call_canonical_model", mock_canonical):
        result = await Phase1Gate.check_input_safety("hello world")

    assert result is True
    assert mock_canonical.call_count == 2


@pytest.mark.asyncio
async def test_check_input_safety_one_model_unavailable_one_safe_passes():
    """
    If one model returns None but the other says 'safe', the input passes
    (the failed model is skipped, the surviving safe verdict wins).
    """
    mock_canonical = AsyncMock(side_effect=[None, "safe"])

    with patch("friday.phase1.call_canonical_model", mock_canonical):
        result = await Phase1Gate.check_input_safety("hi")

    assert result is True


@pytest.mark.asyncio
async def test_check_input_safety_one_model_unavailable_one_unsafe_passes():
    """
    If one model is unavailable, fail open regardless of the other verdict.
    """
    mock_canonical = AsyncMock(side_effect=[None, "unsafe output"])

    with patch("friday.phase1.call_canonical_model", mock_canonical):
        result = await Phase1Gate.check_input_safety("bad thing")

    assert result is True


@pytest.mark.asyncio
async def test_check_input_safety_ambiguous_verdict_passes():
    """
    A verdict that does not contain 'safe' or 'unsafe' is treated as
    ambiguous and the input passes unless both models explicitly say unsafe.
    """
    mock_canonical = _safety_mock("maybe", "safe")

    with patch("friday.phase1.call_canonical_model", mock_canonical):
        result = await Phase1Gate.check_input_safety("hello")

    assert result is True


# ── 6. Output safety blocking a specialist response ───────────────────────────


@pytest.mark.asyncio
async def test_process_blocks_specialist_output_when_safety_fails():
    """
    Full process() pipeline: when the specialist returns output that
    fails the output safety filter, the returned tuple's verified_output
    is None and a new TriageResult with safe=False is emitted.
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(route="CoderAgent", confidence="high")
    )
    mock_check_safety = AsyncMock(return_value=False)

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
        patch("friday.phase1.check_safety", mock_check_safety),
    ):
        triage, verified = await Phase1Gate.process(
            "write code", specialist_output="rm -rf /"
        )

    assert triage.route is RouteDecision.CODER  # routing decision is preserved
    assert triage.safe is False
    assert "output" in (triage.block_reason or "").lower()
    assert verified is None
    # Output safety was checked exactly once.
    assert mock_check_safety.call_count == 1


@pytest.mark.asyncio
async def test_process_returns_specialist_output_when_safety_passes():
    """
    Happy path through process(): output safety passes -> the specialist
    output is returned unchanged along with the original triage decision.
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(route="CoderAgent", confidence="high")
    )
    mock_check_safety = AsyncMock(return_value=True)

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
        patch("friday.phase1.check_safety", mock_check_safety),
    ):
        triage, verified = await Phase1Gate.process(
            "write code", specialist_output="def hello(): return 1"
        )

    assert triage.route is RouteDecision.CODER
    assert triage.safe is True
    assert verified == "def hello(): return 1"


@pytest.mark.asyncio
async def test_process_no_specialist_output_returns_none_payload():
    """
    process() called without a specialist_output -> second tuple element
    is None, regardless of route.
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(route="ReasoningAgent", confidence="medium")
    )

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
    ):
        triage, verified = await Phase1Gate.process("think about it")

    assert triage.route is RouteDecision.REASONING
    assert verified is None


@pytest.mark.asyncio
async def test_process_short_circuits_when_route_is_blocked():
    """
    process() with a BLOCKED triage -> (triage, None) without ever
    running output safety on a missing specialist output.
    """
    mock_canonical = _safety_mock("unsafe", "unsafe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(route="CoderAgent")
    )
    mock_check_safety = AsyncMock(return_value=True)

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
        patch("friday.phase1.check_safety", mock_check_safety),
    ):
        triage, verified = await Phase1Gate.process("bad prompt", specialist_output="x")

    assert triage.route is RouteDecision.BLOCKED
    assert verified is None
    # Output safety should NOT have run — the triage is already terminal.
    assert mock_check_safety.call_count == 0


@pytest.mark.asyncio
async def test_process_fast_path_returns_router_response_directly():
    """
    process() with a FAST_PATH route -> the second tuple element is the
    fast_response itself, with no separate output-safety call (the route
    function already verified it).
    """
    mock_canonical = _safety_mock("safe", "safe")
    mock_call_model = AsyncMock(
        return_value=_json_router_payload(
            route="FAST_PATH", confidence="high", fast_response="Hi there!"
        )
    )
    mock_check_safety = AsyncMock(return_value=True)

    with (
        patch("friday.phase1.call_canonical_model", mock_canonical),
        patch("friday.phase1.call_model", mock_call_model),
        patch("friday.phase1.check_safety", mock_check_safety),
        patch.object(ConfidenceGate, "gate", return_value=(False, None, "medium")),
    ):
        triage, verified = await Phase1Gate.process("hello")

    assert triage.route is RouteDecision.FAST_PATH
    assert verified == "Hi there!"


# ── resolve_specialist mapping ────────────────────────────────────────────────


def test_resolve_specialist_returns_none_for_terminal_routes():
    """FAST_PATH and BLOCKED have no specialist — resolve returns None."""
    assert resolve_specialist(RouteDecision.FAST_PATH) is None
    assert resolve_specialist(RouteDecision.BLOCKED) is None


def test_resolve_specialist_maps_every_specialist_route():
    """Every non-terminal RouteDecision maps to a distinct Specialist."""
    seen: set[Specialist] = set()
    for decision in RouteDecision:
        if decision in (RouteDecision.FAST_PATH, RouteDecision.BLOCKED):
            continue
        spec = resolve_specialist(decision)
        assert spec is not None
        seen.add(spec)
    # Eleven real specialists (everything except the two terminal routes).
    assert len(seen) == 11


# ── _parse_router_json edge cases ─────────────────────────────────────────────


def test_parse_router_json_strips_markdown_fences():
    raw = "```json\n" + _json_router_payload(route="CoderAgent") + "\n```"
    parsed = _parse_router_json(raw)
    assert parsed is not None
    assert parsed["route"] == "CoderAgent"


def test_parse_router_json_strips_plain_fences():
    raw = "```\n" + _json_router_payload(route="BrowserAgent") + "\n```"
    parsed = _parse_router_json(raw)
    assert parsed is not None
    assert parsed["route"] == "BrowserAgent"


def test_parse_router_json_extracts_embedded_object():
    raw = (
        "Here is the answer you asked for: "
        + _json_router_payload(route="VisionAgent")
        + " (end)"
    )
    parsed = _parse_router_json(raw)
    assert parsed is not None
    assert parsed["route"] == "VisionAgent"


def test_parse_router_json_returns_none_for_garbage():
    assert _parse_router_json("") is None
    assert _parse_router_json("not json at all") is None
    assert _parse_router_json("[1, 2, 3]") is None  # not a dict
    assert _parse_router_json("{not valid json}") is None
