"""
F.R.I.D.A.Y. — Preferences & Learning Module
==============================================
Tracks, learns, and adapts to user preferences over time.

This module is the "personality memory" of F.R.I.D.A.Y. — it remembers
choices the user makes, frequencies of their habits, and adapts tool
behaviour to match learned patterns without needing explicit instructions.

Learned preferences include
---------------------------
• Favourite apps & websites (top-N by usage frequency)
• Preferred music service / genre / playlist defaults
• Default search engine, browser, news sources
• Communication style (formal vs casual responses)
• Preferred times of day for certain activities (e.g. "news at 8 AM")
• Custom aliases ("my email" → gmail.com, "the boss" → a specific contact)
• Tool parameter defaults (e.g. default volume level, default PDF style)

Storage
-------
JSON file at `~/Documents/FRIDAY_Output/preferences.json` — human-readable,
git-diffable, easy to inspect/edit manually. Atomic writes prevent corruption.

Public API
----------
    init_preferences()                      → load from disk
    get(key, default=None)                  → read a preference
    set(key, value)                         → write a preference
    record_usage(category, item)            → increment a usage counter
    top_items(category, n=5)                → most-used items in a category
    learn_alias(alias, target)              → add a custom shortcut
    resolve_alias(alias)                    → expand alias → real value
    get_style()                             → "formal" | "casual"
    set_style(style)                        → override style
    suggest_default(category)               → learned default or None
    export_preferences(path)                → dump to file
    reset_preferences(confirm=False)        → wipe (requires explicit True)
    get_preferences_stats()                 → diagnostics dict
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PreferencesConfig:
    # Storage location (same folder pattern as debug_execution.py)
    storage_dir: Path = Path.home() / "Documents" / "FRIDAY_Output"
    filename: str = "preferences.json"
    backup_filename: str = "preferences.backup.json"

    # How many usages before we call it a "learned default"
    learning_threshold: int = 3

    # Usage categories tracked by default
    default_categories: tuple = (
        "apps",  # opened applications
        "websites",  # visited URLs / domains
        "music_service",  # Spotify, YouTube Music, etc.
        "music_genre",  # lofi, classical, rock, ...
        "search_engine",  # google, duckduckgo, ...
        "news_source",  # BBC, NYT, ...
        "commands",  # raw command phrases (for autocomplete later)
    )

    # Max items kept per category (evict least-used beyond this)
    max_items_per_category: int = 50

    # Auto-save debounce (seconds) — batch rapid writes
    autosave_debounce_s: float = 2.0

    @property
    def storage_path(self) -> Path:
        return self.storage_dir / self.filename

    @property
    def backup_path(self) -> Path:
        return self.storage_dir / self.backup_filename


CONFIG = PreferencesConfig()


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger("friday.preferences")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s [PREFS] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)


# ─────────────────────────────────────────────────────────────
# Default schema — bootstrapped on first run
# ─────────────────────────────────────────────────────────────
DEFAULT_PREFERENCES: dict[str, Any] = {
    "meta": {
        "version": 1,
        "created_at": None,
        "updated_at": None,
        "total_writes": 0,
    },
    "settings": {
        "style": "formal",  # "formal" | "casual"
        "wake_word": "friday",
        "tts_voice": "default",
        "default_volume": 50,
        "conversation_timeout_s": 15,
        "verbose_responses": False,
    },
    "usage": {
        # category → {item: count}
        cat: {}
        for cat in CONFIG.default_categories
    },
    "aliases": {
        # "my email" → "https://mail.google.com"
    },
    "defaults": {
        # learned defaults, e.g. "music_service": "spotify"
    },
    "habits": {
        # category → [{"item": ..., "hour": ..., "count": ...}]
        # e.g. {"news_check": [{"item":"BBC","hour":8,"count":5}]}
    },
    "history": [
        # recent (item, category, timestamp) tuples — capped
    ],
}


# ─────────────────────────────────────────────────────────────
# Runtime state
# ─────────────────────────────────────────────────────────────
@dataclass
class PreferencesState:
    data: dict = field(default_factory=dict)
    loaded: bool = False
    dirty: bool = False
    last_save_ts: float = 0.0
    pending_save_timer: threading.Timer | None = None


_state = PreferencesState()
_lock = threading.RLock()


# ─────────────────────────────────────────────────────────────
# Persistence (atomic writes)
# ─────────────────────────────────────────────────────────────
def _atomic_write(path: Path, content: str) -> None:
    """Write to temp file then rename — prevents half-written corruption."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _load_from_disk() -> dict:
    """Load preferences, merging with defaults to fill in missing keys."""
    if not CONFIG.storage_path.exists():
        logger.info("No existing preferences — starting fresh.")
        data = _deep_copy(DEFAULT_PREFERENCES)
        data["meta"]["created_at"] = datetime.now().isoformat(timespec="seconds")
        return data

    try:
        raw = CONFIG.storage_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        # Merge to ensure new keys added in later versions are present
        merged = _deep_merge(_deep_copy(DEFAULT_PREFERENCES), data)
        logger.info(f"Preferences loaded from {CONFIG.storage_path}")
        return merged
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Corrupted preferences file — restoring from backup. ({e})")
        if CONFIG.backup_path.exists():
            try:
                raw = CONFIG.backup_path.read_text(encoding="utf-8")
                return _deep_merge(_deep_copy(DEFAULT_PREFERENCES), json.loads(raw))
            except Exception as e2:
                logger.error(f"Backup also corrupted: {e2}")
        logger.warning("Falling back to defaults.")
        return _deep_copy(DEFAULT_PREFERENCES)


def _save_to_disk() -> bool:
    """Serialize and atomically persist. Keeps a backup of the previous version."""
    with _lock:
        if not _state.dirty:
            return True
        try:
            CONFIG.storage_dir.mkdir(parents=True, exist_ok=True)

            # Back up the existing file first
            if CONFIG.storage_path.exists():
                try:
                    CONFIG.backup_path.write_bytes(CONFIG.storage_path.read_bytes())
                except Exception as e:
                    logger.warning(f"Backup failed (continuing): {e}")

            _state.data["meta"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _state.data["meta"]["total_writes"] = _state.data["meta"].get("total_writes", 0) + 1

            content = json.dumps(_state.data, indent=2, ensure_ascii=False)
            _atomic_write(CONFIG.storage_path, content)
            _state.dirty = False
            _state.last_save_ts = time.time()
            logger.debug(f"Preferences saved ({len(content)} bytes).")
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False


def _schedule_save() -> None:
    """Debounced autosave — avoids hammering disk during rapid updates."""
    with _lock:
        _state.dirty = True
        if _state.pending_save_timer is not None:
            _state.pending_save_timer.cancel()
        timer = threading.Timer(CONFIG.autosave_debounce_s, _save_to_disk)
        timer.daemon = True
        _state.pending_save_timer = timer
        timer.start()


def _deep_copy(obj):
    return json.loads(json.dumps(obj))


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Overlay wins on conflicts."""
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ─────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────
def init_preferences() -> None:
    """Load preferences from disk. Safe to call multiple times."""
    with _lock:
        if _state.loaded:
            return
        _state.data = _load_from_disk()
        _state.loaded = True
        _state.dirty = False
        logger.info(
            f"Preferences initialized "
            f"(style={_state.data['settings'].get('style')}, "
            f"{sum(len(v) for v in _state.data['usage'].values())} usage records)"
        )


def _ensure_loaded() -> None:
    if not _state.loaded:
        init_preferences()


# ─────────────────────────────────────────────────────────────
# Generic get/set (dotted key support: "settings.style")
# ─────────────────────────────────────────────────────────────
def get(key: str, default: Any = None) -> Any:
    """Retrieve a preference by dotted path. E.g. get('settings.style')."""
    _ensure_loaded()
    with _lock:
        node = _state.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def set(key: str, value: Any) -> None:
    """Set a preference by dotted path. Creates intermediate dicts as needed."""
    _ensure_loaded()
    with _lock:
        node = _state.data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        old = node.get(parts[-1])
        node[parts[-1]] = value
        logger.info(f"set {key}: {old!r} → {value!r}")
        _schedule_save()


# ─────────────────────────────────────────────────────────────
# Usage tracking & learning
# ─────────────────────────────────────────────────────────────
def record_usage(category: str, item: str) -> None:
    """
    Increment the usage counter for an item in a category.
    Triggers default-learning once the threshold is reached.
    """
    if not category or not item:
        return
    _ensure_loaded()
    item = str(item).strip().lower()
    with _lock:
        usage = _state.data["usage"].setdefault(category, {})
        usage[item] = usage.get(item, 0) + 1

        # Evict least-used if over cap
        if len(usage) > CONFIG.max_items_per_category:
            least = min(usage.items(), key=lambda kv: kv[1])
            usage.pop(least[0], None)
            logger.debug(f"Evicted '{least[0]}' from {category}")

        # Track habit timing (what hour did this happen?)
        hour = datetime.now().hour
        habits = _state.data["habits"].setdefault(category, [])
        existing = next(
            (h for h in habits if h["item"] == item and h["hour"] == hour),
            None,
        )
        if existing:
            existing["count"] += 1
        else:
            habits.append({"item": item, "hour": hour, "count": 1})
            # Cap habit list size
            if len(habits) > 200:
                habits.sort(key=lambda h: h["count"], reverse=True)
                _state.data["habits"][category] = habits[:100]

        # Append to rolling history
        history = _state.data.setdefault("history", [])
        history.append(
            {
                "category": category,
                "item": item,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if len(history) > 500:
            _state.data["history"] = history[-300:]

        # Auto-learn a default once threshold is crossed
        if (
            usage[item] >= CONFIG.learning_threshold
            and _state.data["defaults"].get(category) != item
        ):
            # Only promote if it's the current leader
            leader = max(usage.items(), key=lambda kv: kv[1])[0]
            if leader == item:
                _state.data["defaults"][category] = item
                logger.info(f"🎓 Learned default for {category}: {item!r}")

    _schedule_save()


def top_items(category: str, n: int = 5) -> list[tuple[str, int]]:
    """Return the N most-used items in a category, sorted by frequency."""
    _ensure_loaded()
    with _lock:
        usage = _state.data["usage"].get(category, {})
        return Counter(usage).most_common(n)


def suggest_default(category: str) -> str | None:
    """Return the learned default for a category, or None if not yet learned."""
    _ensure_loaded()
    with _lock:
        return _state.data["defaults"].get(category)


def preferred_hour_item(category: str, hour: int | None = None) -> str | None:
    """
    Return the item the user tends to use at this hour
    (e.g. news_check at 8 AM → 'BBC'). Useful for proactive suggestions.
    """
    _ensure_loaded()
    if hour is None:
        hour = datetime.now().hour
    with _lock:
        habits = _state.data["habits"].get(category, [])
        at_hour = [h for h in habits if h["hour"] == hour]
        if not at_hour:
            return None
        best = max(at_hour, key=lambda h: h["count"])
        return best["item"] if best["count"] >= CONFIG.learning_threshold else None


# ─────────────────────────────────────────────────────────────
# Aliases (custom shortcuts)
# ─────────────────────────────────────────────────────────────
def learn_alias(alias: str, target: str) -> None:
    """Teach F.R.I.D.A.Y. a shortcut: 'my email' → 'https://mail.google.com'."""
    if not alias or not target:
        return
    _ensure_loaded()
    with _lock:
        _state.data["aliases"][alias.strip().lower()] = target
        logger.info(f"Learned alias: {alias!r} → {target!r}")
    _schedule_save()


def resolve_alias(alias: str) -> str | None:
    """Expand an alias to its target, or None if unknown."""
    if not alias:
        return None
    _ensure_loaded()
    with _lock:
        return _state.data["aliases"].get(alias.strip().lower())


def forget_alias(alias: str) -> bool:
    """Remove a previously learned alias. True if removed."""
    _ensure_loaded()
    with _lock:
        removed = _state.data["aliases"].pop(alias.strip().lower(), None)
    if removed is not None:
        logger.info(f"Forgot alias: {alias!r}")
        _schedule_save()
        return True
    return False


def all_aliases() -> dict[str, str]:
    _ensure_loaded()
    with _lock:
        return dict(_state.data["aliases"])


# ─────────────────────────────────────────────────────────────
# Communication style (formal vs casual)
# ─────────────────────────────────────────────────────────────
_VALID_STYLES = {"formal", "casual", "neutral"}


def get_style() -> str:
    return get("settings.style", "formal")


def set_style(style: str) -> bool:
    if style not in _VALID_STYLES:
        logger.warning(f"Unknown style {style!r}; expected one of {_VALID_STYLES}")
        return False
    set("settings.style", style)
    return True


# ─────────────────────────────────────────────────────────────
# Export / reset / stats
# ─────────────────────────────────────────────────────────────
def export_preferences(path: Path | None = None) -> Path:
    """Dump preferences to a given path (or a timestamped file)."""
    _ensure_loaded()
    if path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CONFIG.storage_dir / f"preferences_export_{ts}.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        path.write_text(
            json.dumps(_state.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    logger.info(f"Preferences exported to {path}")
    return path


def reset_preferences(confirm: bool = False) -> bool:
    """Wipe all preferences. Must pass confirm=True to actually do it."""
    if not confirm:
        logger.warning("reset_preferences() called without confirm=True — ignored.")
        return False
    with _lock:
        _state.data = _deep_copy(DEFAULT_PREFERENCES)
        _state.data["meta"]["created_at"] = datetime.now().isoformat(timespec="seconds")
        _state.dirty = True
    _save_to_disk()
    logger.warning("🧹 Preferences reset to defaults.")
    return True


def flush() -> bool:
    """Force-save any pending changes immediately (bypass debounce)."""
    with _lock:
        if _state.pending_save_timer is not None:
            _state.pending_save_timer.cancel()
            _state.pending_save_timer = None
        _state.dirty = True
    return _save_to_disk()


def get_preferences_stats() -> dict:
    """Runtime diagnostics — useful for debug UI / tray tooltip."""
    _ensure_loaded()
    with _lock:
        usage_counts = {c: len(v) for c, v in _state.data["usage"].items()}
        return {
            "loaded": _state.loaded,
            "dirty": _state.dirty,
            "path": str(CONFIG.storage_path),
            "style": get_style(),
            "total_aliases": len(_state.data["aliases"]),
            "total_defaults": len(_state.data["defaults"]),
            "usage_counts": usage_counts,
            "history_entries": len(_state.data.get("history", [])),
            "total_writes": _state.data["meta"].get("total_writes", 0),
            "last_updated": _state.data["meta"].get("updated_at"),
        }


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running preferences self-test...\n")
    init_preferences()

    # Simulate usage
    for _ in range(4):
        record_usage("apps", "chrome")
    for _ in range(2):
        record_usage("apps", "notepad")
    for _ in range(5):
        record_usage("music_service", "spotify")

    learn_alias("my email", "https://mail.google.com")
    set_style("casual")

    print(f"  Top apps         : {top_items('apps')}")
    print(f"  Top music        : {top_items('music_service')}")
    print(f"  Default music    : {suggest_default('music_service')}")
    print(f"  Alias 'my email' : {resolve_alias('my email')}")
    print(f"  Style            : {get_style()}")

    print("\nStats:")
    for k, v in get_preferences_stats().items():
        print(f"  {k:.<25} {v}")

    flush()
    print(f"\nSaved to {CONFIG.storage_path}\n")
