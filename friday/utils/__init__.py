"""
F.R.I.D.A.Y. — Female Replacement Intelligent Digital Assistant Youth
Package initialisation module.

Exposes the top-level public API so external scripts and tests can import
Friday's core components without navigating the internal module structure.

Example:
    from friday import FridayEngine, FridayTray
    from friday import __version__
"""

from __future__ import annotations

# ── Version ───────────────────────────────────────────────────────────────────

__version__ = "1.0.0"
__author__ = "F.R.I.D.A.Y. Project"
__license__ = "MIT"


# ── Lazy public API ───────────────────────────────────────────────────────────
# Imports are deferred to avoid circular dependencies and to keep startup
# time low when only a subset of the package is needed.


def _lazy_import(module_path: str, name: str):
    """
    Return a lazily-imported attribute from a submodule.
    Raises ImportError with a clear message on failure.
    """
    import importlib

    try:
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Could not import '{name}' from '{module_path}': {exc}") from exc


# ── Core exports (imported on access via __getattr__) ─────────────────────────

_PUBLIC_API: dict[str, tuple[str, str]] = {
    # Engine
    "FridayEngine": (".engine", "FridayEngine"),
    "EngineState": (".engine", "EngineState"),
    "StateChangeEvent": (".engine", "StateChangeEvent"),
    # Tray
    "FridayTray": (".tray", "FridayTray"),
    # LLM
    "LLMBrain": (".llm", "LLMBrain"),
    "LLMResponse": (".llm", "LLMResponse"),
    # TTS / STT
    "TextToSpeech": (".tts", "TextToSpeech"),
    "SpeechToText": (".stt", "SpeechToText"),
    # Memory
    "MemoryManager": (".memory", "MemoryManager"),
    "MemoryCategory": (".memory", "MemoryCategory"),
    # Actions
    "ActionRouter": (".actions", "ActionRouter"),
    "ActionResult": (".actions", "ActionResult"),
    # Goals
    "GoalEngine": (".goal_engine", "GoalEngine"),
    "GoalPlanner": (".goal_planner", "GoalPlanner"),
    # Learning
    "LearningEngine": (".learning", "LearningEngine"),
    # Emotion
    "EmotionDetector": (".emotion", "EmotionDetector"),
    # Personality
    "EmotionalTone": (".personality", "EmotionalTone"),
    "ResponseCategory": (".personality", "ResponseCategory"),
    # Config
    "config": (".config", None),
}


def __getattr__(name: str):
    """
    Module-level __getattr__ enables lazy importing of public API members.
    Only triggered when the name is not found in the module's namespace.
    """
    if name in _PUBLIC_API:
        module_path, attr = _PUBLIC_API[name]

        # Special case: expose entire submodule
        if attr is None:
            import importlib

            return importlib.import_module(module_path, package=__name__)

        return _lazy_import(module_path, attr)

    raise AttributeError(f"module 'friday' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose public API members in dir() and IDE autocompletion."""
    return [*list(_PUBLIC_API.keys()), "__version__", "__author__", "__license__"]


# ── Package health check ──────────────────────────────────────────────────────


def check_dependencies() -> dict[str, bool]:
    """
    Probe optional dependencies and return their availability.
    Useful for diagnostics and conditional feature enabling.

    Returns:
        Dict mapping package name to availability (True/False).
    """
    packages = {
        "openai": "openai",
        "speech_recognition": "speech_recognition",
        "pyaudio": "pyaudio",
        "vosk": "vosk",
        "pyttsx3": "pyttsx3",
        "pygame": "pygame",
        "sounddevice": "sounddevice",
        "pystray": "pystray",
        "PIL": "PIL",
        "mss": "mss",
        "pytesseract": "pytesseract",
        "psutil": "psutil",
        "chromadb": "chromadb",
        "noisereduce": "noisereduce",
        "transformers": "transformers",
        "speechbrain": "speechbrain",
        "playwright": "playwright",
        "pyperclip": "pyperclip",
        "win32gui": "win32gui",
        "pycaw": "pycaw",
        "pyautogui": "pyautogui",
    }

    results: dict[str, bool] = {}
    for name, module in packages.items():
        try:
            __import__(module)
            results[name] = True
        except ImportError:
            results[name] = False

    return results


def get_version() -> str:
    """Return the current package version string."""
    return __version__


def get_status_summary() -> str:
    """
    Return a human-readable dependency status summary.
    Useful for diagnostics printed at startup or via CLI.

    Returns:
        Multi-line string listing available and missing packages.
    """
    deps = check_dependencies()
    present = [k for k, v in deps.items() if v]
    missing = [k for k, v in deps.items() if not v]

    lines = [
        f"F.R.I.D.A.Y. v{__version__}",
        f"Available ({len(present)}): {', '.join(present)}",
    ]
    if missing:
        lines.append(f"Missing  ({len(missing)}): {', '.join(missing)}")

    return "\n".join(lines)
