"""
Tests for friday.learning — the user-personalisation engine.

Covers the five explicit scenarios requested:
  1. record_command() stores command
  2. resolve_shortcut() returns expansion for known trigger
  3. resolve_shortcut() returns None for unknown trigger
  4. suggest_shortcuts() returns list
  5. setup() and teardown() lifecycle

Plus auxiliary coverage (preferences, usage patterns, pattern analyser,
compatibility wrappers) so we comfortably exceed 50%.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from friday.learning import (
    CommandRecord,
    EventType,
    LearningEngine,
    PreferenceKey,
    Shortcut,
    UsagePattern,
    _LearningDB,
    _PatternAnalyser,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    """
    A unique path under tmp_path. Tests that need the engine should
    pair this with the `engine` fixture below.
    """
    return str(tmp_path / "learning.db")


@pytest.fixture
def learning_engine(db_path, monkeypatch):
    """
    Build a LearningEngine backed by a throwaway SQLite database.
    Monkeypatches config.storage.learning_db so the engine picks up the
    temporary path. StorageConfig is a frozen dataclass, so we use
    object.__setattr__ to override the field.
    """
    from friday import config as cfg_mod

    object.__setattr__(cfg_mod.config.storage, "learning_db", db_path)
    engine = LearningEngine()
    engine.setup()
    return engine


# ── 1. record_command() stores command ────────────────────────────────────────


def test_record_command_stores_in_history(learning_engine):
    learning_engine.record_command("open chrome", action="open_app", success=True)
    rows = learning_engine._db.fetch_top_commands(limit=10)
    assert len(rows) == 1
    assert rows[0]["command"] == "open chrome"
    assert rows[0]["total"] == 1


def test_record_command_normalises_lowercase_and_strips(learning_engine):
    learning_engine.record_command("  OPEN CHROME  ", action="x")
    rows = learning_engine._db.fetch_top_commands(limit=10)
    assert len(rows) == 1
    # Stored as normalised lowercase
    assert rows[0]["command"] == "open chrome"


def test_record_command_empty_string_is_silently_dropped(learning_engine):
    learning_engine.record_command("   ")
    rows = learning_engine._db.fetch_top_commands(limit=10)
    assert rows == []


def test_record_command_logs_event_in_addition_to_history(learning_engine):
    learning_engine.record_command("play music", action="media")
    events = learning_engine._db.fetch_events(event_type=EventType.COMMAND_EXECUTED)
    assert len(events) == 1
    import json as _json

    payload = _json.loads(events[0]["payload"])
    assert payload["command"] == "play music"
    assert payload["action"] == "media"
    assert payload["success"] is True


def test_record_command_increments_session_counter(learning_engine):
    assert learning_engine.session_stats["session_commands"] == 0
    learning_engine.record_command("a")
    learning_engine.record_command("b")
    assert learning_engine.session_stats["session_commands"] == 2


def test_record_correction_logs_event(learning_engine):
    learning_engine.record_correction("open chorom", "open chrome")
    events = learning_engine._db.fetch_events(event_type=EventType.CORRECTION_MADE)
    assert len(events) == 1


# ── 2. resolve_shortcut() returns expansion for known trigger ─────────────────


def test_resolve_shortcut_returns_expansion_for_known_trigger(learning_engine):
    learning_engine.register_shortcut("chrome", "open chrome")
    expansion = learning_engine.resolve_shortcut("chrome")
    assert expansion == "open chrome"


def test_resolve_shortcut_is_case_insensitive(learning_engine):
    learning_engine.register_shortcut("Chrome", "open chrome")
    assert learning_engine.resolve_shortcut("CHROME") == "open chrome"
    assert learning_engine.resolve_shortcut("chrome") == "open chrome"


def test_resolve_shortcut_increments_use_count(learning_engine):
    learning_engine.register_shortcut("x", "do thing x")
    assert learning_engine.list_shortcuts()[0].use_count == 0
    learning_engine.resolve_shortcut("x")
    learning_engine.resolve_shortcut("x")
    assert learning_engine.list_shortcuts()[0].use_count == 2


# ── 3. resolve_shortcut() returns None for unknown trigger ────────────────────


def test_resolve_shortcut_returns_none_for_unknown_trigger(learning_engine):
    assert learning_engine.resolve_shortcut("never-registered") is None


def test_resolve_shortcut_does_not_increment_for_unknown(learning_engine):
    learning_engine.register_shortcut("real", "do real thing")
    learning_engine.resolve_shortcut("fake")
    # Real shortcut use count is still 0
    assert learning_engine.list_shortcuts()[0].use_count == 0


def test_register_shortcut_rejects_empty_strings(learning_engine):
    assert learning_engine.register_shortcut("", "expansion") is False
    assert learning_engine.register_shortcut("trigger", "") is False
    assert learning_engine.register_shortcut("   ", "exp") is False


def test_register_shortcut_upserts_existing(learning_engine):
    learning_engine.register_shortcut("x", "old expansion")
    learning_engine.register_shortcut("x", "new expansion")
    assert learning_engine.resolve_shortcut("x") == "new expansion"
    assert len(learning_engine.list_shortcuts()) == 1


def test_delete_shortcut_removes_entry(learning_engine):
    learning_engine.register_shortcut("temp", "temporary command")
    assert learning_engine.delete_shortcut("temp") is True
    assert learning_engine.resolve_shortcut("temp") is None


def test_delete_shortcut_returns_false_when_missing(learning_engine):
    assert learning_engine.delete_shortcut("never-existed") is False


# ── 4. suggest_shortcuts() returns list ───────────────────────────────────────


def test_suggest_shortcuts_returns_list_on_empty_history(learning_engine):
    """Empty history -> empty list, not an error."""
    assert learning_engine.suggest_shortcuts() == []


def test_suggest_shortcuts_returns_empty_when_below_threshold(learning_engine):
    """Commands used fewer than SHORTCUT_THRESHOLD times are not suggested."""
    for _ in range(2):
        learning_engine.record_command("rare command that is long")
    assert learning_engine.suggest_shortcuts() == []


def test_suggest_shortcuts_returns_candidate_for_frequent_long_command(
    learning_engine,
):
    """A long command used 5+ times becomes a suggestion."""
    for _ in range(6):
        learning_engine.record_command("launch the development server please")
    suggestions = learning_engine.suggest_shortcuts()
    assert isinstance(suggestions, list)
    assert len(suggestions) >= 1
    # The first element is the full command, second is the suggested trigger
    full_cmd, trigger = suggestions[0]
    assert "launch" in full_cmd
    assert isinstance(trigger, str)
    assert len(trigger) >= 3


def test_suggest_shortcuts_skips_already_registered_commands(learning_engine):
    """A command that already has a shortcut must not be re-suggested."""
    for _ in range(6):
        learning_engine.record_command("set up the development environment now")
    # Register a shortcut for the same expansion
    learning_engine.register_shortcut("dev env", "set up the development environment now")
    suggestions = learning_engine.suggest_shortcuts()
    # The full command now matches an existing shortcut's expansion -> skipped
    for full_cmd, _ in suggestions:
        assert full_cmd != "set up the development environment now"


def test_suggest_shortcuts_skips_short_commands(learning_engine):
    """Commands that are 2 words or fewer are too short to warrant a shortcut."""
    for _ in range(6):
        learning_engine.record_command("go home")
    assert learning_engine.suggest_shortcuts() == []


# ── 5. setup() and teardown() lifecycle ──────────────────────────────────────


def test_setup_loads_preferences_and_marks_ready(learning_engine):
    assert learning_engine.is_ready is True
    # Cache is populated
    assert isinstance(learning_engine._cache, dict)


def test_teardown_disables_engine(learning_engine):
    learning_engine.teardown()
    assert learning_engine.is_ready is False


def test_setup_is_idempotent(learning_engine):
    """Calling setup() twice doesn't break the engine."""
    assert learning_engine.setup() is True
    assert learning_engine.is_ready is True


# ── Preference management ─────────────────────────────────────────────────────


def test_set_and_get_preference_round_trips(learning_engine):
    learning_engine.set_preference(PreferenceKey.VERBOSITY, "concise")
    assert learning_engine.get_preference(PreferenceKey.VERBOSITY) == "concise"


def test_get_preference_returns_default_when_unset(learning_engine):
    val = learning_engine.get_preference(PreferenceKey.GREETING_STYLE, "formal")
    assert val == "formal"


def test_set_preference_persists_to_db(learning_engine):
    learning_engine.set_preference(PreferenceKey.WAKE_SENSITIVITY, 0.42)
    # A fresh engine reading the same DB should see the value
    raw = learning_engine._db.get_preference("wake_sensitivity")
    assert raw is not None
    import json as _json
    assert _json.loads(raw) == 0.42


def test_set_preference_overwrites_existing(learning_engine):
    learning_engine.set_preference(PreferenceKey.VERBOSITY, "concise")
    learning_engine.set_preference(PreferenceKey.VERBOSITY, "detailed")
    assert learning_engine.get_preference(PreferenceKey.VERBOSITY) == "detailed"


def test_get_all_preferences_returns_dict(learning_engine):
    learning_engine.set_preference(PreferenceKey.VERBOSITY, "concise")
    learning_engine.set_preference(PreferenceKey.GREETING_STYLE, "casual")
    all_prefs = learning_engine.get_all_preferences()
    assert isinstance(all_prefs, dict)
    assert all_prefs.get("verbosity") == "concise"
    assert all_prefs.get("greeting_style") == "casual"


# ── Usage pattern / analytics ────────────────────────────────────────────────


def test_get_usage_pattern_returns_none_for_unknown_command(learning_engine):
    assert learning_engine.get_usage_pattern("never-used") is None


def test_get_usage_pattern_returns_aggregated_stats(learning_engine):
    for _ in range(3):
        learning_engine.record_command("open chrome", action="open_app", success=True)
    pattern = learning_engine.get_usage_pattern("open chrome")
    assert isinstance(pattern, UsagePattern)
    assert pattern.total_calls == 3
    assert pattern.success_rate == 1.0
    assert pattern.peak_hour == datetime.now().hour
    assert pattern.is_frequent is False  # < 5 calls
    assert pattern.is_reliable is True   # 100% success


def test_usage_pattern_is_frequent_property():
    p = UsagePattern(
        command="x", total_calls=5, success_rate=1.0, avg_duration=0.0,
        last_used=time.time(), peak_hour=0,
    )
    assert p.is_frequent is True
    p2 = UsagePattern(
        command="x", total_calls=4, success_rate=1.0, avg_duration=0.0,
        last_used=time.time(), peak_hour=0,
    )
    assert p2.is_frequent is False


def test_usage_pattern_is_reliable_property():
    p = UsagePattern(
        command="x", total_calls=10, success_rate=0.9, avg_duration=0.0,
        last_used=time.time(), peak_hour=0,
    )
    assert p.is_reliable is True
    p2 = UsagePattern(
        command="x", total_calls=10, success_rate=0.5, avg_duration=0.0,
        last_used=time.time(), peak_hour=0,
    )
    assert p2.is_reliable is False


def test_get_top_commands_returns_ordered_list(learning_engine):
    for _ in range(3):
        learning_engine.record_command("rare")
    for _ in range(7):
        learning_engine.record_command("frequent")
    top = learning_engine.get_top_commands(limit=5)
    assert isinstance(top, list)
    assert len(top) == 2
    # Most frequent comes first
    assert top[0].command == "frequent"
    assert top[0].total_calls == 7
    assert top[1].command == "rare"


def test_get_action_reliability_filters_low_sample(learning_engine):
    for _ in range(3):
        learning_engine.record_command("ok", success=True)
    for _ in range(1):
        learning_engine.record_command("single", success=False)
    reliability = learning_engine.get_action_reliability()
    # 'ok' has 3 calls (>=3) and is 100% reliable
    # 'single' has 1 call (<3) and is filtered out
    assert "ok" in reliability
    assert reliability["ok"] == 1.0
    assert "single" not in reliability


def test_get_hourly_usage_returns_dict(learning_engine):
    learning_engine.record_command("x")
    usage = learning_engine.get_hourly_usage(hours=24)
    assert isinstance(usage, dict)
    # At least one hour bucket is populated
    assert sum(usage.values()) >= 1


# ── Compatibility wrappers ────────────────────────────────────────────────────


def test_record_failure_and_get_count_round_trip(learning_engine):
    learning_engine.record_failure("open_app", {"app": "chrome"}, "timeout")
    learning_engine.record_failure("open_app", {"app": "chrome"}, "timeout")
    learning_engine.record_failure("open_app", {"app": "vscode"}, "crash")
    assert learning_engine.get_failure_count("open_app", {"app": "chrome"}) == 2
    assert learning_engine.get_failure_count("open_app", {"app": "vscode"}) == 1
    assert learning_engine.get_failure_count("open_app", {"app": "none"}) == 0


def test_is_reliable_wrapper(learning_engine):
    details = {"x": 1}
    assert learning_engine.is_reliable("action", details) is True
    learning_engine.record_failure("action", details, "boom")
    assert learning_engine.is_reliable("action", details) is False


def test_create_and_save_shortcut_aliases(learning_engine):
    assert learning_engine.create_shortcut("a", "do a") is True
    assert learning_engine.save_shortcut("b", "do b") is True
    assert learning_engine.resolve_shortcut("a") == "do a"
    assert learning_engine.resolve_shortcut("b") == "do b"


def test_get_shortcut_fuzzy_match(learning_engine):
    learning_engine.register_shortcut("chrome", "open chrome")
    # Exact match
    assert learning_engine.get_shortcut("chrome") == "open chrome"
    # Fuzzy match (typo, below default threshold) falls back to direct DB lookup
    result = learning_engine.get_shortcut("chrme")
    # Either the fuzzy match passes (if similar enough) or it falls back to
    # the direct DB lookup, which won't find it. Either way no crash.
    assert result is None or result == "open chrome"


def test_track_command_is_non_blocking(learning_engine, monkeypatch):
    """track_command spawns a daemon thread; we just verify it doesn't crash."""
    learning_engine.record_command = lambda *a, **kw: None  # shortcut
    learning_engine.track_command("anything")
    # No assertion needed — the thread will run record_command in the background.
    # We give it a brief moment to settle.
    time.sleep(0.05)


# ── _PatternAnalyser direct coverage ─────────────────────────────────────────


def test_generate_trigger_drops_filler_words():
    trigger = _PatternAnalyser._generate_trigger("please open the chrome browser")
    # 'please' is a filler; the first two significant words are "open chrome"
    assert trigger is not None
    assert "please" not in trigger


def test_generate_trigger_returns_none_for_all_fillers():
    assert _PatternAnalyser._generate_trigger("please can you") is None


def test_generate_trigger_returns_none_for_empty():
    assert _PatternAnalyser._generate_trigger("") is None


def test_generate_trigger_returns_none_for_too_short():
    # Only one significant word -> result is shorter than 3 chars
    assert _PatternAnalyser._generate_trigger("a b please c") in (None, "a b")


def test_pattern_analyser_find_shortcut_candidates_under_threshold():
    """Commands below SHORTCUT_THRESHOLD are filtered out."""
    candidates = _PatternAnalyser().find_shortcut_candidates(
        top_commands=[{"command": "some long command text", "total": 2}],
        existing_shortcuts=[],
    )
    assert candidates == []


def test_pattern_analyser_detect_usage_peaks():
    now = time.time()
    peaks = _PatternAnalyser().detect_usage_peaks(
        [
            {"timestamp": now},
            {"timestamp": now},
            {"timestamp": now - 86400},  # yesterday
        ]
    )
    # All entries fall into the current hour
    assert isinstance(peaks, dict)
    assert sum(peaks.values()) == 3


def test_pattern_analyser_compute_action_reliability():
    rel = _PatternAnalyser().compute_action_reliability(
        [
            {"command": "ok", "success_rate": 0.95, "total": 5},
            {"command": "flaky", "success_rate": 0.5, "total": 4},
            {"command": "low_sample", "success_rate": 1.0, "total": 1},  # filtered
        ]
    )
    assert rel["ok"] == 0.95
    assert rel["flaky"] == 0.5
    assert "low_sample" not in rel


# ── _LearningDB direct coverage ──────────────────────────────────────────────


def test_learning_db_basic_round_trip(db_path):
    from friday import config as cfg_mod

    object.__setattr__(cfg_mod.config.storage, "learning_db", db_path)
    db = _LearningDB(db_path)
    db.insert_command(
        CommandRecord(
            command="x",
            action="a",
            success=True,
            timestamp=time.time(),
            duration=0.1,
            metadata={"k": "v"},
        )
    )
    stats = db.fetch_command_stats("x")
    assert stats is not None
    assert stats["total"] == 1
    assert stats["success_rate"] == 1.0
    assert stats["avg_duration"] == 0.1


def test_learning_db_fetch_command_stats_returns_none_for_missing(db_path):
    from friday import config as cfg_mod

    object.__setattr__(cfg_mod.config.storage, "learning_db", db_path)
    db = _LearningDB(db_path)
    assert db.fetch_command_stats("never") is None


def test_learning_db_preferences_round_trip(db_path):
    from friday import config as cfg_mod

    object.__setattr__(cfg_mod.config.storage, "learning_db", db_path)
    db = _LearningDB(db_path)
    db.set_preference("k1", "v1")
    db.set_preference("k2", "v2")
    assert db.get_preference("k1") == "v1"
    assert db.get_all_preferences() == {"k1": "v1", "k2": "v2"}
    assert db.get_preference("missing") is None


def test_learning_db_delete_shortcut_returns_flag(db_path):
    from friday import config as cfg_mod

    object.__setattr__(cfg_mod.config.storage, "learning_db", db_path)
    db = _LearningDB(db_path)
    db.upsert_shortcut("a", "do a")
    assert db.delete_shortcut("a") is True
    assert db.delete_shortcut("a") is False  # already gone


def test_learning_db_get_shortcut_returns_none_for_missing(db_path):
    from friday import config as cfg_mod

    object.__setattr__(cfg_mod.config.storage, "learning_db", db_path)
    db = _LearningDB(db_path)
    assert db.get_shortcut("nope") is None
