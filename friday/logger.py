"""
F.R.I.D.A.Y.  —  Logging Module
─────────────────────────────────────────────────────────────────────────────
Provides a single, project-wide logging setup with:

  • Colour-coded console output  (auto-disabled when stdout is not a TTY)
  • Rotating file handler        (logs/friday.log  - 5 MB x 5 backups)
  • Structured JSON session log  (logs/session_<timestamp>.jsonl)
  • Separate crash/error log     (logs/errors.log  — errors and above only)
  • Per-module logger factory    (get_logger())
  • Session summary on shutdown  (log_session_summary())

Usage
─────
    # In any friday module:
    from friday.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("Low-level detail.")
    logger.info("Normal operation.")
    logger.warning("Something unexpected but recoverable.")
    logger.error("Something failed.")
    logger.critical("System is going down.")

    # At process shutdown:
    from friday.logger import log_session_summary, shutdown_logging
    log_session_summary()
    shutdown_logging()
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Root directory for all log files.
LOG_DIR: Path = Path(__file__).parent.parent / "logs"

# Maximum size of friday.log before rotation (bytes).
LOG_FILE_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB

# Number of rotated backup files to keep.
LOG_FILE_BACKUPS: int = 5

# Console log level (can be overridden via FRIDAY_LOG_LEVEL env var).
_DEFAULT_CONSOLE_LEVEL: str = "INFO"

# File log level (always DEBUG so full detail is captured on disk).
_FILE_LEVEL: int = logging.DEBUG

# Name of the root logger for the project.
_ROOT_LOGGER_NAME: str = "friday"

# ── ANSI colour palette ───────────────────────────────────────────────────────

_USE_COLOUR: bool = (sys.stdout.isatty() and sys.platform != "win32") or (
    sys.platform == "win32" and os.environ.get("TERM") == "xterm"
)

_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG: "\033[38;5;244m",  # grey
    logging.INFO: "\033[38;5;117m",  # sky blue
    logging.WARNING: "\033[38;5;214m",  # amber
    logging.ERROR: "\033[38;5;203m",  # red-orange
    logging.CRITICAL: "\033[38;5;196m",  # bright red
}

_MODULE_COLOUR: str = "\033[38;5;141m"  # lavender  — for module name
_TIME_COLOUR: str = "\033[38;5;244m"  # grey      — for timestamp
_RESET: str = "\033[0m"
_BOLD: str = "\033[1m"


# ── Colour formatter ──────────────────────────────────────────────────────────


class _ColourFormatter(logging.Formatter):
    """
    Console formatter that colour-codes by level and highlights the
    module name.  Automatically degrades to plain text on non-TTY outputs.

    Output format (colour)
        12:04:33  INFO      friday.llm       Response in 1.2s
        12:04:34  WARNING   friday.wakeword  Vosk not installed

    Output format (plain)
        2025-04-22T12:04:33  INFO      friday.llm       Response in 1.2s
    """

    _PLAIN_FMT = "{asctime}  {levelname:<9} {name:<28} {message}"
    _COLOR_FMT = "{time_str}  {level_str} {name_str:<28} {message}"

    def format(self, record: logging.LogRecord) -> str:
        # Let the base class handle exception formatting
        record.message = record.getMessage()
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        if not _USE_COLOUR:
            asctime = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            base = self._PLAIN_FMT.format(
                asctime=asctime,
                levelname=record.levelname,
                name=record.name,
                message=record.message,
            )
        else:
            level_col = _LEVEL_COLOURS.get(record.levelno, "")
            time_str = (
                _TIME_COLOUR + datetime.fromtimestamp(record.created).strftime("%H:%M:%S") + _RESET
            )
            level_str = f"{level_col}{_BOLD}{record.levelname:<9}{_RESET}"
            name_str = f"{_MODULE_COLOUR}{record.name}{_RESET}"
            base = self._COLOR_FMT.format(
                time_str=time_str,
                level_str=level_str,
                name_str=name_str,
                message=record.message,
            )

        if record.exc_text:
            # Indent exception lines for visual separation
            exc_lines = record.exc_text.splitlines()
            indent = "  │  "
            if _USE_COLOUR:
                exc_block = "\n".join(
                    f"{_LEVEL_COLOURS.get(record.levelno, '')}{indent}{line}{_RESET}"
                    for line in exc_lines
                )
            else:
                exc_block = "\n".join(f"{indent}{line}" for line in exc_lines)
            base = f"{base}\n{exc_block}"

        return base


# ── JSON formatter (session log) ──────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    """
    Emits one JSON object per line (JSONL) for structured log ingestion.

    Each record::

        {
          "ts":      "2025-04-22T12:04:33.412Z",
          "level":   "INFO",
          "logger":  "friday.llm",
          "msg":     "Response in 1.2s",
          "file":    "llm.py",
          "line":    247,
          "exc":     null
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        exc_text: str | None = None
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)

        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S."
            )
            + f"{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
            "exc": exc_text,
        }
        return json.dumps(entry, ensure_ascii=False)


# ── Session statistics ────────────────────────────────────────────────────────


class _SessionStats:
    """Thread-safe counter for per-level log totals within a session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }
        self.started_at: float = time.time()

    def increment(self, levelname: str) -> None:
        with self._lock:
            self._counts[levelname] = self._counts.get(levelname, 0) + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at


_session = _SessionStats()


class _StatsHandler(logging.Handler):
    """Lightweight handler that updates session statistics on every record."""

    def emit(self, record: logging.LogRecord) -> None:
        _session.increment(record.levelname)


# ── Setup ─────────────────────────────────────────────────────────────────────

_setup_done = False
_setup_lock = threading.Lock()


def setup_logging(
    console_level: str | None = None,
    log_dir: Path | None = None,
) -> None:
    """
    Configure the project-wide logging stack.

    Should be called **once** at process startup (e.g. in ``main.py`` or
    ``cli/main.py``).  Subsequent calls are silently ignored.

    Args:
        console_level: Override the console log level.
                       Defaults to the ``FRIDAY_LOG_LEVEL`` env var, then INFO.
        log_dir:       Override the directory for log files.
                       Defaults to ``<project_root>/logs``.
    """
    global _setup_done

    with _setup_lock:
        if _setup_done:
            return
        _setup_done = True

    # Resolve parameters
    effective_log_dir = log_dir or LOG_DIR
    effective_log_dir.mkdir(parents=True, exist_ok=True)

    raw_level = (
        console_level or os.environ.get("FRIDAY_LOG_LEVEL", _DEFAULT_CONSOLE_LEVEL)
    ).upper()
    console_level_int = getattr(logging, raw_level, logging.INFO)

    # Session log filename encodes the startup timestamp
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = effective_log_dir / f"session_{session_ts}.jsonl"

    # ── Root "friday" logger ───────────────────────────────────────────────
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False  # don't bubble to the Python root logger

    # 1. Colour console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level_int)
    console_handler.setFormatter(_ColourFormatter())
    root.addHandler(console_handler)

    # 2. Rotating main file (DEBUG and above — full detail)
    file_handler = logging.handlers.RotatingFileHandler(
        effective_log_dir / "friday.log",
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setLevel(_FILE_LEVEL)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-9s  %(name)-28s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    # 3. Error-only log (WARNING and above — for quick incident review)
    error_handler = logging.handlers.RotatingFileHandler(
        effective_log_dir / "errors.log",
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUPS,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-9s  %(name)-28s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(error_handler)

    # 4. JSONL session log (all levels)
    json_handler = logging.FileHandler(session_path, encoding="utf-8")
    json_handler.setLevel(logging.DEBUG)
    json_handler.setFormatter(_JSONFormatter())
    root.addHandler(json_handler)

    # 5. In-memory stats counter
    root.addHandler(_StatsHandler())

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "vosk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info(
        "Logging initialised — console: %s | log dir: %s | session: %s",
        raw_level,
        effective_log_dir,
        session_path.name,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger scoped under the 'friday' root.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.

    Example::

        logger = get_logger(__name__)
        logger.info("Module ready.")
    """
    # Normalise: strip package prefix duplication (friday.friday.llm → friday.llm)
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def log_session_summary() -> None:
    """
    Emit a session summary to INFO.  Call this just before process exit
    so the final stats are captured in both the console and log files.

    Output example::

        ─────────────────────────────────────────
        F.R.I.D.A.Y. Session Summary
          Uptime  :  00h 14m 32s
          DEBUG   :  1 024
          INFO    :  248
          WARNING :  3
          ERROR   :  0
          CRITICAL:  0
        ─────────────────────────────────────────
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    counts = _session.snapshot()
    elapsed = _session.elapsed_seconds

    hours, rem = divmod(int(elapsed), 3600)
    mins, sec = divmod(rem, 60)
    uptime_str = f"{hours:02d}h {mins:02d}m {sec:02d}s"

    sep = "─" * 41
    lines = [
        sep,
        "  F.R.I.D.A.Y. Session Summary",
        f"  Uptime   :  {uptime_str}",
    ]
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        lines.append(f"  {level:<9}:  {counts.get(level, 0):,}")
    lines.append(sep)

    root.info("\n" + "\n".join(lines))


def shutdown_logging() -> None:
    """
    Flush and close all log handlers gracefully.
    Should be called at the very end of the process, after
    ``log_session_summary()``.
    """
    logging.getLogger(_ROOT_LOGGER_NAME).info("[Logger] Shutting down log handlers.")
    logging.shutdown()
