"""
F.R.I.D.A.Y. — Permission & Risk Assessment
=============================================
Centralised risk scoring for every tool operation.

Determines whether a tool call can auto-execute, needs a verbal confirmation,
or requires a secure passphrase ("Confirm Friday") before proceeding.

Design goals
------------
1. **Single source of truth** for tool risk levels (TOOL_RISK_MAP).
2. **Context-aware escalation** — e.g. `file_write` is HIGH by default but
   becomes CRITICAL if the target is a system path. `code_run` escalates if
   the script length or duration exceeds safe bounds.
3. **Flexible confirmation matching** — handles natural phrasing like
   "yeah okay", "go ahead please", "absolutely not".
4. **Audit trail** — every permission decision is logged for later review.
5. **Rate limiting** — blocks runaway loops that spam the same high-risk tool.

Public API
----------
    assess_risk(tool_name, args)           → RiskLevel
    needs_confirmation(risk)               → bool
    needs_secure_confirm(risk)             → bool
    classify_response(text)                → "confirm" | "secure_confirm"
                                             | "deny" | "unknown"
    register_tool_risk(name, level)        → plug in custom tools
    record_execution(tool_name)            → rate-limit bookkeeping
    is_rate_limited(tool_name)             → bool
    get_permission_stats()                 → diagnostics dict
"""

from __future__ import annotations

import enum
import logging
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Risk Levels
# ─────────────────────────────────────────────────────────────
class RiskLevel(enum.Enum):
    LOW = "low"  # Auto-execute (open URL, get time, volume)
    MEDIUM = "medium"  # Verbal confirm ("Shall I proceed, Sir?")
    HIGH = "high"  # Verbal confirm + audit log
    CRITICAL = "critical"  # Secure phrase required ("Confirm Friday")

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]

    def escalate(self, steps: int = 1) -> RiskLevel:
        """Bump risk by N levels, capped at CRITICAL."""
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        idx = min(order.index(self) + steps, len(order) - 1)
        return order[idx]


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PermissionConfig:
    # Rate limiting: max invocations of the same tool per window
    rate_limit_window_s: float = 10.0
    rate_limit_max_calls: int = 5
    # Critical tool extra cooldown (prevents rapid destructive operations)
    critical_cooldown_s: float = 3.0
    # History for audit/stats
    history_size: int = 100
    # Fuzzy matching threshold (0.0-1.0)
    fuzzy_threshold: float = 0.80

    # System paths considered sensitive — writing here escalates risk
    sensitive_path_patterns: tuple = (
        r"^[A-Z]:[\\/]Windows[\\/]",
        r"^[A-Z]:[\\/]Program Files",
        r"^/etc/",
        r"^/usr/",
        r"^/bin/",
        r"^/boot/",
        r"^/System/",
        r"\.ssh[\\/]",
        r"\.env$",
        r"registry",
    )

    # Code execution thresholds
    code_long_threshold_chars: int = 2000
    code_long_timeout_s: int = 10


CONFIG = PermissionConfig()


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger("friday.permissions")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s [PERM] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)


# ─────────────────────────────────────────────────────────────
# Tool → Risk mapping (single source of truth)
# ─────────────────────────────────────────────────────────────
TOOL_RISK_MAP: dict[str, RiskLevel] = {
    # ── LOW ─ safe, read-only, trivially reversible
    "open_url": RiskLevel.LOW,
    "open_app": RiskLevel.LOW,
    "search_web": RiskLevel.LOW,
    "web_search": RiskLevel.LOW,
    "youtube_play": RiskLevel.LOW,
    "google_search": RiskLevel.LOW,
    "browser_navigate": RiskLevel.LOW,
    "browser_read_page": RiskLevel.LOW,
    "get_time": RiskLevel.LOW,
    "get_date": RiskLevel.LOW,
    "volume_up": RiskLevel.LOW,
    "volume_down": RiskLevel.LOW,
    "volume_mute": RiskLevel.LOW,
    "media_play": RiskLevel.LOW,
    "media_next": RiskLevel.LOW,
    "media_prev": RiskLevel.LOW,
    "file_exists": RiskLevel.LOW,
    "file_list": RiskLevel.LOW,
    "file_read": RiskLevel.LOW,
    "file_search": RiskLevel.LOW,
    "file_get_info": RiskLevel.LOW,
    "desktop_list_windows": RiskLevel.LOW,
    "none": RiskLevel.LOW,
    # ── MEDIUM ─ intrusive but non-destructive
    "focus_window": RiskLevel.MEDIUM,
    "read_screen_text": RiskLevel.MEDIUM,
    "desktop_focus_window": RiskLevel.MEDIUM,
    "desktop_read_window": RiskLevel.MEDIUM,
    "browser_open": RiskLevel.MEDIUM,
    "browser_screenshot": RiskLevel.MEDIUM,
    "browser_go_back": RiskLevel.MEDIUM,
    # ── HIGH ─ modifies state (types, clicks, moves files)
    "click": RiskLevel.HIGH,
    "type_text": RiskLevel.HIGH,
    "close_app": RiskLevel.HIGH,
    "browser_click": RiskLevel.HIGH,
    "browser_type": RiskLevel.HIGH,
    "browser_close_context": RiskLevel.HIGH,
    "desktop_open_app": RiskLevel.HIGH,
    "desktop_type_text": RiskLevel.HIGH,
    "desktop_press_key": RiskLevel.HIGH,
    "file_write": RiskLevel.HIGH,
    "file_move": RiskLevel.HIGH,
    "file_create_folder": RiskLevel.HIGH,
    # ── CRITICAL ─ destructive / irreversible
    "file_delete": RiskLevel.CRITICAL,
    "code_run": RiskLevel.CRITICAL,
    "shell_exec": RiskLevel.CRITICAL,
    "system_shutdown": RiskLevel.CRITICAL,
    "system_restart": RiskLevel.CRITICAL,
}


def register_tool_risk(name: str, level: RiskLevel) -> None:
    """Register or override the risk level for a tool (used by plugins)."""
    TOOL_RISK_MAP[name] = level
    logger.info(f"Registered tool risk: {name} → {level.value}")


# ─────────────────────────────────────────────────────────────
# Confirmation phrase matching
# ─────────────────────────────────────────────────────────────
# Base phrases — exact & partial matching both supported
CONFIRM_PHRASES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "okay",
        "ok",
        "go ahead",
        "proceed",
        "do it",
        "go for it",
        "affirmative",
        "confirmed",
        "approved",
        "do that",
        "please do",
        "fine",
        "alright",
        "of course",
        "absolutely",
    }
)

SECURE_CONFIRM_PHRASES = frozenset(
    {
        "confirm friday",
        "proceed friday",
        "authorize",
        "authorise",
        "confirm proceed",
        "execute",
        "authorise friday",
        "authorize friday",
        "i authorize this",
        "i authorise this",
        "confirm and proceed",
    }
)

DENY_PHRASES = frozenset(
    {
        "no",
        "nope",
        "nah",
        "cancel",
        "stop",
        "abort",
        "halt",
        "don't",
        "do not",
        "negative",
        "never mind",
        "nevermind",
        "not now",
        "wait",
        "hold on",
        "absolutely not",
        "forget it",
    }
)

# Filler words stripped before matching
_FILLER_WORDS = re.compile(
    r"\b(please|friday|thanks|thank you|um|uh|like|just|maybe|i think)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase, strip fillers/punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = _FILLER_WORDS.sub("", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_ratio(a: str, b: str) -> float:
    """Simple char-overlap ratio — no external deps."""
    if not a or not b:
        return 0.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def _phrase_hits(text: str, phrases: frozenset[str]) -> bool:
    """True if any phrase matches exactly, as substring, or fuzzily."""
    if not text:
        return False
    # Exact / substring match
    for p in phrases:
        if text == p or f" {p} " in f" {text} ":
            return True
    # Fuzzy match as last resort (only for short inputs to avoid false positives)
    if len(text.split()) <= 4:
        for p in phrases:
            if _fuzzy_ratio(text, p) >= CONFIG.fuzzy_threshold:
                return True
    return False


def classify_response(text: str) -> str:
    """
    Classify a user's voice response to a permission prompt.
    Priority: secure_confirm > deny > confirm > unknown
    (Deny beats confirm so "no, go ahead" — ambiguous — is treated as deny.)

    Returns: 'confirm' | 'secure_confirm' | 'deny' | 'unknown'
    """
    if not text:
        return "unknown"

    normalized = _normalize(text)

    if _phrase_hits(normalized, SECURE_CONFIRM_PHRASES):
        return "secure_confirm"
    if _phrase_hits(normalized, DENY_PHRASES):
        return "deny"
    if _phrase_hits(normalized, CONFIRM_PHRASES):
        return "confirm"
    return "unknown"


# ─────────────────────────────────────────────────────────────
# Context-aware risk escalation
# ─────────────────────────────────────────────────────────────
_SENSITIVE_RE = re.compile("|".join(CONFIG.sensitive_path_patterns), re.IGNORECASE)


def _is_sensitive_path(path: str) -> bool:
    if not path:
        return False
    return bool(_SENSITIVE_RE.search(path))


def _escalate_for_context(tool_name: str, args: dict, base: RiskLevel) -> RiskLevel:
    """Bump risk based on argument contents."""
    if not args:
        return base

    # File operations targeting sensitive paths → escalate
    if tool_name in ("file_write", "file_move", "file_delete"):
        path = str(args.get("path") or args.get("target") or args.get("src") or "")
        if _is_sensitive_path(path):
            logger.warning(f"Sensitive path detected → escalating {tool_name}")
            return base.escalate(1)

    # Large/long-running code → escalate beyond CRITICAL's bar (still CRITICAL,
    # but flagged in audit so the engine can add extra warnings)
    if tool_name in ("code_run", "shell_exec"):
        code = str(args.get("code") or args.get("command") or "")
        if len(code) > CONFIG.code_long_threshold_chars:
            logger.warning(f"Large code payload ({len(code)} chars) for {tool_name}")
        timeout = args.get("timeout", 0)
        try:
            if float(timeout) > CONFIG.code_long_timeout_s:
                logger.warning(f"Long timeout ({timeout}s) for {tool_name}")
        except (TypeError, ValueError):
            pass

    # URL schemes other than http(s) → escalate open_url
    if tool_name in ("open_url", "browser_open", "browser_navigate"):
        url = str(args.get("url", ""))
        if url and not url.lower().startswith(("http://", "https://", "about:")):
            logger.warning(f"Unusual URL scheme → escalating: {url[:60]}")
            return base.escalate(1)

    return base


# ─────────────────────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────────────────────
@dataclass
class PermissionState:
    recent_calls: dict = field(default_factory=lambda: defaultdict(deque))
    last_critical_ts: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=CONFIG.history_size))
    total_assessments: int = 0
    total_blocked: int = 0


_state = PermissionState()
_lock = threading.Lock()


def record_execution(tool_name: str) -> None:
    """Track that a tool just ran — used by is_rate_limited()."""
    now = time.time()
    with _lock:
        dq = _state.recent_calls[tool_name]
        dq.append(now)
        # Prune old entries
        while dq and now - dq[0] > CONFIG.rate_limit_window_s:
            dq.popleft()
        # Track critical-tool cadence
        risk = TOOL_RISK_MAP.get(tool_name, RiskLevel.HIGH)
        if risk == RiskLevel.CRITICAL:
            _state.last_critical_ts = now


def is_rate_limited(tool_name: str) -> bool:
    """True if the tool has been invoked too many times recently."""
    now = time.time()
    with _lock:
        dq = _state.recent_calls.get(tool_name)
        if dq:
            # Clean expired entries
            while dq and now - dq[0] > CONFIG.rate_limit_window_s:
                dq.popleft()
            if len(dq) >= CONFIG.rate_limit_max_calls:
                logger.warning(
                    f"🚦 Rate limit hit: {tool_name} "
                    f"({len(dq)} calls in {CONFIG.rate_limit_window_s}s)"
                )
                return True

        # Critical tools also have a short absolute cooldown
        risk = TOOL_RISK_MAP.get(tool_name, RiskLevel.HIGH)
        if risk == RiskLevel.CRITICAL:
            since = now - _state.last_critical_ts
            if 0 < since < CONFIG.critical_cooldown_s:
                logger.warning(
                    f"⏳ Critical cooldown active ({since:.1f}s < {CONFIG.critical_cooldown_s}s)"
                )
                return True

    return False


# ─────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────
def assess_risk(tool_name: str, args: dict | None = None) -> RiskLevel:
    """
    Return the risk level for a given tool call.

    • Unknown tools default to HIGH (fail-safe, not fail-open).
    • Context-aware escalation examines args for sensitive paths, unusual
      URL schemes, large code payloads, etc.
    """
    base = TOOL_RISK_MAP.get(tool_name, RiskLevel.HIGH)
    risk = _escalate_for_context(tool_name, args or {}, base)

    with _lock:
        _state.total_assessments += 1
        _state.history.append(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "tool": tool_name,
                "base": base.value,
                "effective": risk.value,
                "escalated": risk != base,
            }
        )

    if risk != base:
        logger.info(f"Risk for {tool_name}: {base.value} → {risk.value} (escalated)")
    else:
        logger.debug(f"Risk for {tool_name}: {risk.value}")

    return risk


def needs_confirmation(risk: RiskLevel) -> bool:
    """True if the risk level requires at least a verbal 'yes'."""
    return risk.rank >= RiskLevel.MEDIUM.rank


def needs_secure_confirm(risk: RiskLevel) -> bool:
    """True if the risk requires the secure passphrase ('Confirm Friday')."""
    return risk == RiskLevel.CRITICAL


# ─────────────────────────────────────────────────────────────
# Stats / diagnostics
# ─────────────────────────────────────────────────────────────
def get_permission_stats() -> dict:
    """Runtime metrics for debug UI / tray tooltip."""
    with _lock:
        hist = list(_state.history)
    escalations = sum(1 for h in hist if h.get("escalated"))
    return {
        "total_assessments": _state.total_assessments,
        "total_blocked": _state.total_blocked,
        "escalations": escalations,
        "registered_tools": len(TOOL_RISK_MAP),
        "recent": hist[-10:],
    }


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running permissions self-test...\n")

    cases = [
        ("open_url", {"url": "https://youtube.com"}),
        ("open_url", {"url": "file:///etc/passwd"}),
        ("file_write", {"path": "/tmp/notes.txt"}),
        ("file_write", {"path": "C:\\Windows\\System32\\evil.dll"}),
        ("file_delete", {"path": "/tmp/junk.log"}),
        ("code_run", {"code": "print('hi')"}),
        ("code_run", {"code": "x" * 3000, "timeout": 60}),
        ("unknown_tool", {}),
    ]
    for tool, args in cases:
        r = assess_risk(tool, args)
        print(f"  {tool:20s} {args!s:55.55s} → {r.value}")

    print("\nResponse classification:")
    for phrase in [
        "yes please",
        "yeah go ahead friday",
        "absolutely not",
        "confirm friday",
        "nope cancel that",
        "um, okay",
        "i'm not sure",
    ]:
        print(f"  {phrase!r:35s} → {classify_response(phrase)}")

    print("\nDone.")
