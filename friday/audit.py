"""
F.R.I.D.A.Y. Action Audit Log
"""

from __future__ import annotations

import datetime
import gzip
import json
import logging
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditConfig:
    audit_dir: Path = Path(os.path.expanduser("~")) / ".friday" / "logs"
    log_filename: str = "action_audit.log"
    jsonl_filename: str = "action_audit.jsonl"
    max_file_size_mb: int = 10
    max_rotations: int = 5
    max_arg_length: int = 200
    max_result_length: int = 300
    memory_buffer_size: int = 100
    compress_rotated: bool = True

    @property
    def log_path(self) -> Path:
        return self.audit_dir / self.log_filename

    @property
    def jsonl_path(self) -> Path:
        return self.audit_dir / self.jsonl_filename


CONFIG = AuditConfig()

logger = logging.getLogger("friday.audit")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s [AUDIT] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)


# Patterns in argument keys that should always be redacted
_SENSITIVE_KEYS = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|auth|credential|private)",
    re.IGNORECASE,
)

# Tool-specific arg redaction rules (key → max visible chars)
_TOOL_REDACTION_RULES: dict[str, dict[str, int]] = {
    "type_text": {"text": 20},
    "desktop_type_text": {"text": 20},
    "browser_type": {"text": 20},
    "code_run": {"code": 50},
    "shell_exec": {"command": 50},
    "file_write": {"content": 30},
}


@dataclass
class AuditState:
    recent: deque = field(default_factory=lambda: deque(maxlen=CONFIG.memory_buffer_size))
    total_logged: int = 0
    total_failed: int = 0
    total_verified: int = 0


_state = AuditState()
_lock = threading.Lock()


def _ensure_log_files() -> None:
    CONFIG.audit_dir.mkdir(parents=True, exist_ok=True)
    if not CONFIG.log_path.exists():
        CONFIG.log_path.write_text("=== F.R.I.D.A.Y. AUDIT LOG ===\n", encoding="utf-8")
    if not CONFIG.jsonl_path.exists():
        CONFIG.jsonl_path.touch()


def _rotate_if_needed(path: Path) -> None:
    """Rotate file if it exceeds max size. Optionally gzip old copies."""
    try:
        if not path.exists():
            return
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < CONFIG.max_file_size_mb:
            return

        # Shift existing rotations: .1 → .2, .2 → .3, etc.
        for i in range(CONFIG.max_rotations - 1, 0, -1):
            old = path.with_suffix(path.suffix + f".{i}")
            new = path.with_suffix(path.suffix + f".{i + 1}")
            if CONFIG.compress_rotated:
                old = old.with_suffix(old.suffix + ".gz")
                new = new.with_suffix(new.suffix + ".gz")
            if old.exists():
                if new.exists():
                    new.unlink()
                old.rename(new)

        rotated = path.with_suffix(path.suffix + ".1")
        path.rename(rotated)

        if CONFIG.compress_rotated:
            with (
                open(rotated, "rb") as f_in,
                gzip.open(rotated.with_suffix(rotated.suffix + ".gz"), "wb") as f_out,
            ):
                f_out.writelines(f_in)
            rotated.unlink()

        logger.info(f"Rotated {path.name} ({size_mb:.1f}MB)")
    except Exception as e:
        logger.warning(f"Rotation failed for {path}: {e}")


def _redact_value(value: Any, max_len: int) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"... [+{len(value) - max_len} chars]"
    return value


def _sanitize_args(tool_name: str, args: dict) -> dict:
    """Redact sensitive/long fields before logging."""
    if not args:
        return {}
    safe: dict = {}
    tool_rules = _TOOL_REDACTION_RULES.get(tool_name, {})

    for k, v in args.items():
        if _SENSITIVE_KEYS.search(str(k)):
            safe[k] = "[REDACTED]"
            continue
        if k in tool_rules:
            safe[k] = _redact_value(v, tool_rules[k])
            continue
        if isinstance(v, dict):
            safe[k] = _sanitize_args(tool_name, v)
        elif isinstance(v, str):
            safe[k] = _redact_value(v, CONFIG.max_arg_length)
        else:
            safe[k] = v
    return safe


def _shorten(text: str, max_len: int) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ").replace("\r", " ").strip()
    if len(t) > max_len:
        return t[:max_len] + "..."
    return t


def log_action(
    risk_level: str,
    tool_name: str,
    args: dict,
    result: str,
    verified: bool,
    user: str | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
) -> None:
    """Append a tool execution entry to the audit log (human + JSONL)."""
    _ensure_log_files()

    with _lock:
        _rotate_if_needed(CONFIG.log_path)
        _rotate_if_needed(CONFIG.jsonl_path)

        now = datetime.datetime.now()
        timestamp = now.isoformat(sep=" ", timespec="seconds")
        safe_args = _sanitize_args(tool_name, args or {})
        short_result = _shorten(str(result or ""), CONFIG.max_result_length)
        status = "VERIFIED" if verified else "FAILED/UNVERIFIED"

        # Human-readable line
        human_line = (
            f"[{timestamp}] [RISK:{risk_level.upper()}] "
            f"Action: {tool_name}({json.dumps(safe_args, ensure_ascii=False)}) "
            f"-> {status} -> {short_result}"
        )
        if duration_ms is not None:
            human_line += f" ({duration_ms:.0f}ms)"
        if error:
            human_line += f" | ERR: {_shorten(error, 100)}"
        human_line += "\n"

        # Structured JSONL entry
        entry = {
            "ts": now.isoformat(timespec="seconds"),
            "risk": risk_level.lower(),
            "tool": tool_name,
            "args": safe_args,
            "result": short_result,
            "verified": verified,
            "status": status,
            "user": user,
            "duration_ms": duration_ms,
            "error": error,
        }

        try:
            with open(CONFIG.log_path, "a", encoding="utf-8") as f:
                f.write(human_line)
            with open(CONFIG.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            _state.recent.append(entry)
            _state.total_logged += 1
            if verified:
                _state.total_verified += 1
            else:
                _state.total_failed += 1
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


def get_recent_history(lines: int = 10) -> str:
    """Return the last N lines of the human-readable audit log."""
    if not CONFIG.log_path.exists():
        return "No audit log found."
    try:
        with open(CONFIG.log_path, encoding="utf-8") as f:
            content = f.readlines()
        return "".join(content[-lines:])
    except Exception as e:
        return f"Error reading audit log: {e}"


def get_recent_entries(n: int = 10) -> list[dict]:
    """Return the last N structured entries (in-memory, fast)."""
    with _lock:
        return list(_state.recent)[-n:]


def query(
    tool_name: str | None = None,
    risk_level: str | None = None,
    verified: bool | None = None,
    since: datetime.datetime | None = None,
    limit: int = 100,
) -> list[dict]:
    """Search JSONL audit log with optional filters."""
    if not CONFIG.jsonl_path.exists():
        return []

    results: list[dict] = []
    try:
        with open(CONFIG.jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if tool_name and e.get("tool") != tool_name:
                    continue
                if risk_level and e.get("risk") != risk_level.lower():
                    continue
                if verified is not None and e.get("verified") is not verified:
                    continue
                if since:
                    try:
                        ts = datetime.datetime.fromisoformat(e.get("ts", ""))
                        if ts < since:
                            continue
                    except ValueError:
                        continue
                results.append(e)
    except Exception as ex:
        logger.error(f"Query failed: {ex}")

    return results[-limit:]


def clear_log(confirm: bool = False) -> bool:
    """Wipe the audit logs (requires explicit confirm=True)."""
    if not confirm:
        logger.warning("clear_log() called without confirm=True — ignored.")
        return False
    with _lock:
        try:
            if CONFIG.log_path.exists():
                CONFIG.log_path.unlink()
            if CONFIG.jsonl_path.exists():
                CONFIG.jsonl_path.unlink()
            _state.recent.clear()
            _state.total_logged = 0
            _state.total_failed = 0
            _state.total_verified = 0
            _ensure_log_files()
            logger.warning("🧹 Audit logs cleared.")
            return True
        except Exception as e:
            logger.error(f"clear_log failed: {e}")
            return False


def get_audit_stats() -> dict:
    """Runtime metrics for debug UI / tray tooltip."""
    with _lock:
        size_mb = (
            CONFIG.log_path.stat().st_size / (1024 * 1024) if CONFIG.log_path.exists() else 0.0
        )
        return {
            "log_path": str(CONFIG.log_path),
            "jsonl_path": str(CONFIG.jsonl_path),
            "size_mb": round(size_mb, 2),
            "total_logged": _state.total_logged,
            "total_verified": _state.total_verified,
            "total_failed": _state.total_failed,
            "in_memory_entries": len(_state.recent),
        }


if __name__ == "__main__":
    print("Testing audit module...\n")

    log_action("low", "get_time", {}, "12:34 PM", True, duration_ms=5)
    log_action(
        "high",
        "type_text",
        {"text": "Dear Sir, " + "x" * 300, "password": "hunter2"},
        "typed 300 chars",
        True,
        duration_ms=420,
    )
    log_action(
        "critical",
        "file_delete",
        {"path": "/tmp/old.txt"},
        "permission denied",
        False,
        error="PermissionError: [Errno 13]",
    )
    log_action("low", "open_url", {"url": "https://youtube.com"}, "opened", True, duration_ms=88)

    print(get_recent_history(5))
    print("\nQuery critical only:")
    for e in query(risk_level="critical"):
        print(f"  {e['ts']} {e['tool']} verified={e['verified']}")
    print("\nStats:", get_audit_stats())
