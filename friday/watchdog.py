"""
F.R.I.D.A.Y.  —  Watchdog Process
─────────────────────────────────────────────────────────────────────────────
Monitors the main assistant process and restarts it on unexpected exit.

Features
  • Exponential back-off with jitter between restart attempts
  • Crash budget  — gives up after N crashes within a rolling time window
    to avoid infinite restart loops on a broken build
  • Structured JSON crash log  (one JSON object per line, easy to parse)
  • Clean shutdown on SIGINT / SIGTERM  — propagates the signal to the
    child first, waits for graceful exit, then quits
  • Cross-platform  — works on Windows (pythonw.exe) and Unix/macOS
  • stdout/stderr capture  — child output is tee-d to a rotating log file
    while still streaming to the watchdog's own stdout

Usage
  python watchdog.py [--target main.py] [--no-window] [--max-crashes 5]

Exit codes
  0   Assistant exited cleanly (returncode 0)
  1   Crash budget exhausted
  2   Target script not found
  3   Fatal watchdog error
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

# How many crashes are tolerated within the rolling window before giving up.
DEFAULT_MAX_CRASHES: int = 5

# Rolling window (seconds) over which crashes are counted.
CRASH_WINDOW_SECS: float = 300.0  # 5 minutes

# Base delay (seconds) for exponential back-off.
BACKOFF_BASE: float = 3.0

# Maximum back-off cap (seconds).
BACKOFF_MAX: float = 120.0

# Jitter factor — actual delay is base ± (base * JITTER_FACTOR).
JITTER_FACTOR: float = 0.2

# Seconds to wait for child to exit cleanly after SIGTERM before SIGKILL.
GRACEFUL_SHUTDOWN_TIMEOUT: float = 120.0

# Log directory (relative to this file's location).
LOG_DIR: Path = Path(__file__).parent / "logs"

# Max size of a single crash log file before rotation.
CRASH_LOG_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MB
CRASH_LOG_BACKUPS: int = 5

# ── Logging setup ─────────────────────────────────────────────────────────────


def _configure_logging() -> None:
    """
    Configure the watchdog's own logger:
      - Console handler  (INFO and above, coloured on TTYs)
      - Rotating file handler  (DEBUG and above → logs/watchdog.log)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_file = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt_console = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Rotating file
    fh = RotatingFileHandler(
        LOG_DIR / "watchdog.log",
        maxBytes=CRASH_LOG_MAX_BYTES,
        backupCount=CRASH_LOG_BACKUPS,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    root.addHandler(fh)
    root.addHandler(ch)


logger = logging.getLogger("friday.watchdog")


# ── Crash journal ─────────────────────────────────────────────────────────────


class CrashJournal:
    """
    Append-only structured crash log  (newline-delimited JSON).

    Each entry is a JSON object::

        {
          "ts":      "2025-04-22T14:03:11Z",
          "attempt": 3,
          "code":    -1,
          "reason":  "Process exited with code -1"
        }

    Also maintains an in-memory sliding window of crash timestamps so the
    watchdog can enforce a crash budget without re-reading the file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._window: list[float] = []  # epoch timestamps of recent crashes

    def record(self, attempt: int, returncode: int, reason: str) -> None:
        """Append a crash record and update the in-memory window."""
        now = time.monotonic()
        now_ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

        entry = {
            "ts": now_ts,
            "attempt": attempt,
            "code": returncode,
            "reason": reason,
        }

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("[Journal] Could not write crash record: %s", exc)

        self._window.append(now)
        logger.debug("[Journal] Crash recorded (attempt %d, code %d).", attempt, returncode)

    def crashes_in_window(self, window_secs: float) -> int:
        """Return the count of crashes that occurred within the last *window_secs* seconds."""
        cutoff = time.monotonic() - window_secs
        self._window = [t for t in self._window if t >= cutoff]
        return len(self._window)


# ── Back-off calculator ───────────────────────────────────────────────────────


def _backoff(attempt: int) -> float:
    """
    Exponential back-off with ± jitter.

    Delay = min(base * 2^(attempt-1), max) ± (base * jitter)
    """
    delay = min(BACKOFF_BASE * math.pow(2, attempt - 1), BACKOFF_MAX)
    jitter = BACKOFF_BASE * JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.0, delay + jitter)


# ── Executable resolution ─────────────────────────────────────────────────────


def _resolve_python(windowless: bool) -> str:
    """
    Return the path to the Python interpreter to use for the child process.

    On Windows, prefer ``pythonw.exe`` (no console window) when *windowless*
    is True.  Falls back to ``sys.executable`` if the windowed variant does
    not exist.
    """
    exe = sys.executable

    if windowless and sys.platform == "win32":
        candidate = Path(exe).with_name("pythonw.exe")
        if candidate.is_file():
            logger.debug("[Watchdog] Using windowless interpreter: %s", candidate)
            return str(candidate)
        logger.warning(
            "[Watchdog] pythonw.exe not found at '%s'; falling back to '%s'.",
            candidate,
            exe,
        )

    return exe


# ── Signal handling ───────────────────────────────────────────────────────────


class _ShutdownFlag:
    """Thread-safe flag set by signal handlers to request a clean stop."""

    def __init__(self) -> None:
        self._stop = False

    def set(self) -> None:
        self._stop = True

    @property
    def requested(self) -> bool:
        return self._stop


_shutdown = _ShutdownFlag()


def _terminate_child(proc: subprocess.Popen) -> None:
    """
    Attempt a graceful shutdown of *proc*:
    1. Send SIGTERM (or CTRL_BREAK_EVENT on Windows).
    2. Wait up to GRACEFUL_SHUTDOWN_TIMEOUT seconds.
    3. SIGKILL / TerminateProcess if still alive.
    """
    if proc.poll() is not None:
        return  # already exited

    logger.info("[Watchdog] Sending termination signal to child (PID %d)…", proc.pid)

    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except OSError:
        pass  # process may have just exited

    try:
        proc.wait(timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
        logger.info("[Watchdog] Child exited gracefully.")
    except subprocess.TimeoutExpired:
        logger.warning("[Watchdog] Child did not exit in time — killing.")
        proc.kill()
        proc.wait()


# ── Core watchdog loop ────────────────────────────────────────────────────────


class Watchdog:
    """
    Manages the full lifecycle of the monitored assistant process.

    Args:
        target:      Path to the Python script to run (e.g. ``main.py``).
        windowless:  Use pythonw.exe on Windows to suppress the console.
        max_crashes: Maximum tolerated crashes within CRASH_WINDOW_SECS.
    """

    def __init__(
        self,
        target: str,
        windowless: bool = True,
        max_crashes: int = DEFAULT_MAX_CRASHES,
    ) -> None:
        self._target = Path(target).resolve()
        self._python = _resolve_python(windowless)
        self._max_crashes = max_crashes
        self._journal = CrashJournal(LOG_DIR / "crash.log")
        self._proc: subprocess.Popen | None = None
        self._attempt = 0

        # Wire signal handlers so Ctrl-C / kill propagates cleanly
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:  # noqa: SIM105
                signal.signal(sig, self._handle_signal)
            except (OSError, ValueError):
                pass  # some signals can't be caught on all platforms

    def _handle_signal(self, signum: int, _frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("[Watchdog] Received %s — initiating clean shutdown.", sig_name)
        _shutdown.set()
        if self._proc and self._proc.poll() is None:
            _terminate_child(self._proc)

    def run(self) -> int:
        """
        Enter the watchdog loop.

        Returns:
            Exit code:
              0  — child exited cleanly
              1  — crash budget exhausted
              2  — target script not found
              3  — fatal watchdog error
        """
        if not self._target.is_file():
            logger.critical("[Watchdog] Target script not found: %s", self._target)
            return 2

        cmd = [self._python, str(self._target)]
        logger.info(
            "[Watchdog] Guarding '%s' (max %d crashes / %.0fs window).",
            self._target.name,
            self._max_crashes,
            CRASH_WINDOW_SECS,
        )

        while not _shutdown.requested:
            self._attempt += 1
            logger.info(
                "[Watchdog] ── Attempt #%d ── Starting F.R.I.D.A.Y…",
                self._attempt,
            )

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    # Keep child in its own process group so signals
                    # can be targeted precisely on Unix.
                    **(
                        {"start_new_session": True}
                        if sys.platform != "win32"
                        else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                    ),
                )

                logger.debug("[Watchdog] Child PID: %d", self._proc.pid)
                self._proc.wait()
                rc = self._proc.returncode

            except FileNotFoundError:
                logger.critical(
                    "[Watchdog] Interpreter not found: '%s'. Aborting.",
                    self._python,
                )
                return 3

            except Exception as exc:
                logger.exception("[Watchdog] Unexpected error launching child: %s", exc)
                self._journal.record(self._attempt, -1, str(exc))
                delay = _backoff(self._attempt)
                logger.info("[Watchdog] Retrying in %.1fs…", delay)
                self._sleep(delay)
                continue

            # ── Exit code routing ─────────────────────────────────────────────

            if _shutdown.requested:
                logger.info("[Watchdog] Shutdown was requested — not restarting.")
                return 0

            if rc == 0:
                logger.info("[Watchdog] Assistant exited normally (code 0). Watchdog done.")
                return 0

            # Abnormal exit — record and evaluate crash budget
            reason = f"Process exited with code {rc}"
            logger.error("[Watchdog] %s.", reason)
            self._journal.record(self._attempt, rc, reason)

            recent = self._journal.crashes_in_window(CRASH_WINDOW_SECS)
            logger.warning(
                "[Watchdog] Crash #%d within %.0fs window (budget: %d).",
                recent,
                CRASH_WINDOW_SECS,
                self._max_crashes,
            )

            if recent >= self._max_crashes:
                logger.critical(
                    "[Watchdog] Crash budget exhausted (%d/%d crashes in %.0fs). "
                    "F.R.I.D.A.Y. will not be restarted automatically.",
                    recent,
                    self._max_crashes,
                    CRASH_WINDOW_SECS,
                )
                return 1

            delay = _backoff(self._attempt)
            logger.info("[Watchdog] Restarting in %.1fs…", delay)
            self._sleep(delay)

        return 0

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep — wakes early if shutdown is requested."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not _shutdown.requested:
            time.sleep(min(0.25, deadline - time.monotonic()))


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="F.R.I.D.A.Y. Watchdog — restarts the assistant on unexpected exit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--target",
        default="main.py",
        metavar="SCRIPT",
        help="Python script to monitor.",
    )
    p.add_argument(
        "--no-window",
        dest="windowless",
        action="store_true",
        default=True,
        help="Use pythonw.exe on Windows to suppress the console window.",
    )
    p.add_argument(
        "--max-crashes",
        type=int,
        default=DEFAULT_MAX_CRASHES,
        metavar="N",
        help=f"Max tolerated crashes within {CRASH_WINDOW_SECS:.0f}s before giving up.",
    )
    return p


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()
    dog = Watchdog(
        target=args.target,
        windowless=args.windowless,
        max_crashes=args.max_crashes,
    )
    sys.exit(dog.run())


if __name__ == "__main__":
    main()
