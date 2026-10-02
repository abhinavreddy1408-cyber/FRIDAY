"""
F.R.I.D.A.Y. — Female Replacement Intelligent Digital Assistant Youth
Entry point for the F.R.I.D.A.Y. voice assistant system.

Responsibilities:
  - Configure the logging pipeline
  - Validate the runtime environment
  - Boot the engine headless (console-only)
  - Handle clean process-level shutdown on signals and exceptions

Usage:
    python main.py          # With console (development)
    pythonw main.py         # Silent background (production)
"""

from __future__ import annotations

# ── Pre-import dependency gate ────────────────────────────────────────────────
# Runs before ANY other import so the user gets a clear message instead of a
# cryptic traceback when SpeechRecognition or PyAudio is missing.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_missing: list[str] = []
try:
    import speech_recognition  # noqa: F401
except ImportError:
    _missing.append("SpeechRecognition")
try:
    import pyaudio  # noqa: F401
except ImportError:
    _missing.append("PyAudio")

if _missing:
    _project_root = _Path(__file__).resolve().parent
    _expected_venv = _project_root / ".venv"
    _expected_python = _expected_venv / "Scripts" / "python.exe"

    print()
    print("=" * 62)
    print("  F.R.I.D.A.Y. — Environment Check FAILED")
    print("=" * 62)
    print()
    print(f"  Missing packages : {', '.join(_missing)}")
    print(f"  Python in use    : {_sys.executable}")
    print(f"  Active venv      : {_os.environ.get('VIRTUAL_ENV', 'None')}")
    print(f"  Expected venv    : {_expected_venv}")
    print()

    # Detect if running with the wrong Python entirely
    if str(_expected_venv) not in _sys.executable:
        print("  !! You are NOT using the project .venv Python !!")
        print("  !! This is the most likely cause of the error.  !!")
        print()

    print("  Quick fix — run one of these from the project root:")
    print()
    print("    .\\run.ps1              # start F.R.I.D.A.Y.")
    print("    .\\setup.ps1            # rebuild the environment")
    print()
    print("  Or install manually into the correct venv:")
    print(f"    & '{_expected_python}' -m pip install SpeechRecognition>=3.10.0 PyAudio>=0.2.14")
    print()
    print("=" * 62)
    _sys.exit(1)

# Clean up gate-only names from module scope
del _os, _sys, _Path, _missing

import os
import warnings

# Force HuggingFace offline mode unless online models are explicitly requested
if os.environ.get("FRIDAY_ONLINE_MODELS") not in ("1", "true", "True", "yes"):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from friday.engine import FridayEngine
    from friday.tray import FridayTray

# Silence noisy third-party warnings
_original_showwarning = warnings.showwarning

def _silenced_showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,
    line: str | None = None,
) -> None:
    if issubclass(category, (UserWarning, FutureWarning, DeprecationWarning)):
        return
    _original_showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = _silenced_showwarning

import logging
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path regardless of working directory.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Logging configuration ─────────────────────────────────────────────────────

def _configure_logging() -> None:
    """
    Set up the root logger with both console and rotating file handlers.
    Log level and file path are sourced from config after it is imported.
    """
    from logging.handlers import RotatingFileHandler

    from friday import config

    level = getattr(logging, config.system.log_level, logging.INFO)

    # ── Formatters ────────────────────────────────────────────────────────────
    verbose_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    concise_fmt = logging.Formatter(
        fmt="%(levelname)-8s  %(name)s  %(message)s",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Prevent duplicate handlers if called more than once
    if root.handlers:
        return

    # ── Console handler ───────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(concise_fmt)
    root.addHandler(console)

    # ── Rotating file handler ─────────────────────────────────────────────────
    try:
        log_path = Path(config.system.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)   # Always capture full detail to file
        file_handler.setFormatter(verbose_fmt)
        root.addHandler(file_handler)
    except OSError as exc:
        logging.warning("Could not create log file at '%s': %s", config.system.log_file, exc)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "PIL", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Environment validation ────────────────────────────────────────────────────

def _validate_environment() -> list[str]:
    """
    Check for required dependencies and configuration before booting.

    Returns:
        List of human-readable warning strings (empty = all clear).
    """
    warnings: list[str] = []

    # Python version
    if sys.version_info < (3, 10):  # noqa: UP036
        warnings.append(
            f"Python 3.10+ recommended. Running {sys.version.split()[0]}."
        )

    # Critical environment variables
    from friday import config
    if not config.llm.api_key:
        warnings.append("NVIDIA_API_KEY is not set — LLM features disabled.")

    # Critical packages
    critical_packages = {
        "speech_recognition": "SpeechRecognition",
        "pyaudio":            "PyAudio",
        "openai":             "openai",
    }
    for module, package in critical_packages.items():
        try:
            __import__(module)
        except ImportError:
            warnings.append(f"Required package '{package}' is not installed.")

    # Optional but important packages
    optional_packages = {
        "PIL":           "Pillow (icon rendering)",
        "vosk":          "vosk (offline wake word / STT)",
        "psutil":        "psutil (system monitoring)",
        "chromadb":      "chromadb (semantic memory)",
        "noisereduce":   "noisereduce (noise cancellation)",
        "transformers":  "transformers (emotion detection)",
    }
    for module, label in optional_packages.items():
        try:
            __import__(module)
        except ImportError:
            warnings.append(f"Optional package '{label}' not installed — feature degraded.")

    return warnings


# ── Signal handling ───────────────────────────────────────────────────────────

_shutdown_event = threading.Event()
_engine_ref: "FridayEngine | None" = None
_tray_ref: "FridayTray | None" = None


def _handle_signal(signum: int, frame: object) -> None:
    """
    Handle OS signals (SIGINT, SIGTERM) for graceful shutdown.
    Sets the shutdown event so the main loop exits cleanly.
    """
    sig_name = signal.Signals(signum).name
    logger   = logging.getLogger(__name__)
    logger.info("[Main] Signal %s received — initiating shutdown.", sig_name)
    _shutdown_event.set()


def _register_signals() -> None:
    """Register SIGINT and SIGTERM handlers for clean shutdown."""
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    # Windows does not support SIGHUP — guard before registering
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_signal)


# ── Exception hook ────────────────────────────────────────────────────────────

def _install_exception_hook() -> None:
    """
    Install a global exception handler that logs unhandled exceptions
    to the log file before the process exits.
    This ensures crashes are captured even when running as pythonw.
    """
    logger = logging.getLogger("friday.crash")

    def _hook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical(
            "Unhandled exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _hook

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is None or issubclass(args.exc_type, SystemExit):
            return
        logger.critical(
            "Unhandled exception in thread '%s':\n%s",
            args.thread.name if args.thread else "unknown",
            "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback
            )),
        )

    threading.excepthook = _thread_hook


# ── Startup banner ────────────────────────────────────────────────────────────

def _print_banner(env_warnings: list[str]) -> None:
    """
    Print a startup banner to the console.
    Suppressed when running as pythonw (no stdout).
    """
    if not sys.stdout or not sys.stdout.isatty():
        return

    banner = """
  ███████╗██████╗ ██╗██████╗  █████╗ ██╗   ██╗
  ██╔════╝██╔══██╗██║██╔══██╗██╔══██╗╚██╗ ██╔╝
  █████╗  ██████╔╝██║██║  ██║███████║ ╚████╔╝
  ██╔══╝  ██╔══██╗██║██║  ██║██╔══██║  ╚██╔╝
  ██║     ██║  ██║██║██████╔╝██║  ██║   ██║
  ╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝

  Female Replacement Intelligent Digital Assistant Youth
  ──────────────────────────────────────────────────────
    """
    print(banner)

    if env_warnings:
        print("  ⚠  Environment warnings:")
        for w in env_warnings:
            print(f"     • {w}")
        print()


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _preflight() -> bool:
    """
    Run all startup checks. Abort if any critical package is missing.

    Returns:
        True if safe to proceed, False if critical requirements are missing.
    """
    logger = logging.getLogger(__name__)
    warnings = _validate_environment()

    critical_failures = [
        w for w in warnings
        if any(pkg in w for pkg in ("SpeechRecognition", "PyAudio", "openai"))
        and "Optional" not in w
    ]

    if critical_failures:
        for failure in critical_failures:
            logger.critical("[Main] Critical requirement missing: %s", failure)
        logger.critical(
            "[Main] Cannot start. Run: pip install -r requirements.txt"
        )
        return False

    for warning in warnings:
        logger.warning("[Main] %s", warning)

    return True


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> int:
    """
    Primary entry point for F.R.I.D.A.Y.
    Headless console runner — no tray, no browser window.

    Returns:
        Exit code (0 = clean exit, 1 = startup failure, 2 = crash).
    """
    # ── Step 1: Logging ───────────────────────────────────────────────────────
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("F.R.I.D.A.Y. starting up — PID %d", os.getpid())
    logger.info("=" * 60)

    # ── Step 2: Hooks and signals ─────────────────────────────────────────────
    _install_exception_hook()
    _register_signals()

    # ── Step 3: Environment validation ────────────────────────────────────────
    env_warnings = _validate_environment()
    _print_banner(env_warnings)

    if not _preflight():
        return 1

    # ── Step 4: Boot tray or headless engine ──────────────────────────────────
    global _engine_ref, _tray_ref
    try:
        if os.environ.get("FRIDAY_HEADLESS") in ("1", "true", "True", "yes"):
            raise ImportError("Headless mode requested by environment.")

        from friday.tray import FridayTray

        logger.info("[Main] Launching system tray...")
        tray = FridayTray()
        _tray_ref = tray
        tray.run()   # Blocks main thread (tray event loop)

    except (ImportError, Exception) as exc:
        logger.warning("[Main] System tray launch failed or disabled (%s) — falling back to console-only mode.", exc)
        try:
            from friday.engine import FridayEngine

            engine = FridayEngine()
            _engine_ref = engine

            logger.info("[Main] Setting up engine (console-only)...")
            engine.setup()

            logger.info("[Main] Starting engine (console-only)...")
            engine.start()

            logger.info("[Main] Engine running. Press Ctrl+C to stop.")

            # Sleep loop — wait for shutdown signal
            while not _shutdown_event.wait(timeout=1.0):
                pass
        except KeyboardInterrupt:
            logger.info("[Main] KeyboardInterrupt — shutting down.")
        except Exception as inner_exc:
            logger.critical("[Main] Fatal exception in console runner: %s", inner_exc, exc_info=True)
            return 2

    finally:
        # Graceful shutdown
        if _tray_ref is not None:
            try:
                _tray_ref.stop()
            except Exception as exc:
                logger.error("[Main] Tray stop error: %s", exc)
        elif _engine_ref is not None:
            try:
                _engine_ref.stop()
            except Exception as exc:
                logger.error("[Main] Engine stop error: %s", exc)
        logger.info("[Main] F.R.I.D.A.Y. process exiting.")

    return 0


# ── Process entry ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())

