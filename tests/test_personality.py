"""
Tests for friday.personality — defines Friday's system prompt, voice lines, and response variation logic.
"""

from __future__ import annotations

import pytest
from friday.personality import (
    EmotionalTone,
    ResponseCategory,
    _history,
    build_tone_aware_response,
    format_error_response,
    get_proactive_prefix,
    get_system_prompt,
    get_voice_line,
)


def test_get_voice_line_returns_non_empty_string_for_every_category():
    """get_voice_line() returns a non-empty string for every ResponseCategory."""
    for category in ResponseCategory:
        line = get_voice_line(category)
        assert isinstance(line, str)
        assert len(line) > 0


def test_get_voice_line_does_not_repeat_same_line_twice_in_row():
    """get_voice_line() does not repeat the same line twice in a row."""
    # We use a category with multiple lines to ensure selection is diverse and avoids repeating
    category = ResponseCategory.ACKNOWLEDGEMENT
    
    first = get_voice_line(category)
    second = get_voice_line(category)
    third = get_voice_line(category)
    
    # Check that adjacent items are not equal
    assert first != second
    assert second != third


def test_get_voice_line_exhausts_pool_and_resets():
    """get_voice_line() resets available lines when pool is exhausted (covers line 322)."""
    # Temporarily set max_history high so history is not popped, allowing pool exhaustion.
    original_max = _history.max_history
    _history.max_history = 10
    try:
        # ResponseCategory.DENIAL has 4 lines. Calling it 6 times exhausts the pool and triggers reset.
        lines = [get_voice_line(ResponseCategory.DENIAL) for _ in range(6)]
        assert all(isinstance(l, str) and len(l) > 0 for l in lines)
    finally:
        _history.max_history = original_max


def test_build_tone_aware_response_with_neutral_returns_base_unchanged():
    """build_tone_aware_response() with NEUTRAL returns base response unchanged."""
    base = "Hello, user."
    response = build_tone_aware_response(base, EmotionalTone.NEUTRAL)
    assert response == base


def test_build_tone_aware_response_with_non_neutral_adds_prefix():
    """build_tone_aware_response() with non-NEUTRAL tone prepends a tone prefix."""
    base = "I am ready."
    
    # Test for HAPPY tone
    happy_response = build_tone_aware_response(base, EmotionalTone.HAPPY)
    assert happy_response != base
    assert happy_response.endswith(base)
    assert any(happy_response.startswith(prefix) for prefix in ["Absolutely.", "Happy to.", "Of course."])
    
    # Test for URGENT tone
    urgent_response = build_tone_aware_response(base, EmotionalTone.URGENT)
    assert urgent_response != base
    assert urgent_response.endswith(base)
    assert any(urgent_response.startswith(prefix) for prefix in ["On it immediately.", "Moving fast on this.", "Prioritising now."])


def test_build_tone_aware_response_unsupported_tone():
    """build_tone_aware_response() returns base unchanged if tone has no prefixes (covers line 382)."""
    base = "Nothing changed."
    # None is not in _TONE_PREFIXES, so it has no prefixes
    response = build_tone_aware_response(base, None)  # type: ignore
    assert response == base


def test_get_system_prompt_returns_non_empty_string():
    """get_system_prompt() returns a non-empty system prompt string."""
    prompt = get_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "F.R.I.D.A.Y." in prompt


def test_format_error_response_returns_non_empty_string():
    """format_error_response() returns a non-empty string."""
    error = ValueError("Test error")
    response = format_error_response(error)
    assert isinstance(response, str)
    assert len(response) > 0


def test_get_proactive_prefix_returns_non_empty_string():
    """get_proactive_prefix() returns a non-empty prefix from proactive category."""
    prefix = get_proactive_prefix()
    assert isinstance(prefix, str)
    assert len(prefix) > 0


def test_get_voice_line_with_prefix_override():
    """get_voice_line() with prefix_override prepends the override string."""
    line = get_voice_line(ResponseCategory.ACKNOWLEDGEMENT, prefix_override="Wait,")
    assert line.startswith("Wait,")


def test_get_voice_line_with_tone_prefix():
    """get_voice_line() with non-NEUTRAL tone returns response with tone prefix."""
    line = get_voice_line(ResponseCategory.ACKNOWLEDGEMENT, tone=EmotionalTone.HAPPY)
    assert any(line.startswith(prefix) for prefix in ["Absolutely.", "Happy to.", "Of course."])
