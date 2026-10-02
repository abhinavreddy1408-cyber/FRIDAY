"""
F.R.I.D.A.Y. Action Router Module
Handles all local command execution without requiring LLM involvement.
Parses incoming transcripts, matches them to registered actions, and
executes the appropriate handler instantly.

Action categories:
  - Application control  (open, close, switch)
  - System control       (volume, media, power)
  - Web & search         (browser, YouTube, Google)
  - File operations      (open files, directories)
  - System information   (time, date, battery, CPU)
  - Utility              (clipboard, screenshots)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from friday import config

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import pyautogui

    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False

try:
    import ctypes

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    _PYCAW_AVAILABLE = True
except ImportError:
    _PYCAW_AVAILABLE = False

try:
    import win32con  # noqa: F401
    import win32gui  # noqa: F401

    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False

try:
    import pyperclip

    _PYPERCLIP_AVAILABLE = True
except ImportError:
    _PYPERCLIP_AVAILABLE = False


# ── Result Types ──────────────────────────────────────────────────────────────


class ActionStatus(Enum):
    SUCCESS = auto()
    NOT_FOUND = auto()
    PERMISSION_ERROR = auto()
    EXECUTION_ERROR = auto()
    REQUIRES_CONFIRM = auto()
    UNSUPPORTED = auto()


@dataclass(frozen=True)
class ActionResult:
    """
    Immutable result of an action execution attempt.

    Attributes:
        status:   Outcome classification.
        message:  Human-readable response suitable for TTS.
        action:   The matched action name (empty if no match).
        duration: Execution time in seconds.
    """

    status: ActionStatus
    message: str
    action: str = ""
    duration: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == ActionStatus.SUCCESS

    def __str__(self) -> str:
        return self.message


# ── Action Definition ─────────────────────────────────────────────────────────


@dataclass
class Action:
    """
    Descriptor for a single registered action.

    Attributes:
        name:        Unique identifier.
        patterns:    List of regex patterns that trigger this action.
        handler:     Callable that executes the action.
        description: Brief human-readable description.
        destructive: Whether this action requires confirmation.
    """

    name: str
    patterns: list[str]
    handler: Callable[[re.Match], ActionResult]
    description: str = ""
    destructive: bool = False

    def __post_init__(self) -> None:
        self._compiled: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def match(self, text: str) -> re.Match | None:
        """Return the first pattern match against text, or None."""
        for pattern in self._compiled:
            m = pattern.search(text)
            if m:
                return m
        return None


# ── Application Registry ──────────────────────────────────────────────────────

_APP_REGISTRY: dict[str, str] = {
    # Browsers
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "brave": "brave",
    # Productivity
    "notepad": "notepad",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "outlook": "outlook",
    "teams": "teams",
    "onenote": "onenote",
    # Development
    "vscode": "code",
    "visual studio code": "code",
    "visual studio": "devenv",
    "terminal": "wt",
    "powershell": "powershell",
    "cmd": "cmd",
    "git bash": "git-bash",
    # Media & Communication
    "spotify": "spotify",
    "discord": "discord",
    "slack": "slack",
    "zoom": "zoom",
    "vlc": "vlc",
    # System
    "task manager": "taskmgr",
    "file explorer": "explorer",
    "explorer": "explorer",
    "control panel": "control",
    "settings": "ms-settings:",
    "calculator": "calc",
    "paint": "mspaint",
    "snipping tool": "snippingtool",
}


# ── System Utilities ──────────────────────────────────────────────────────────


def _launch(executable: str) -> bool:
    """
    Launch an application by executable name or URI scheme.
    Returns True on successful dispatch.
    """
    try:
        if executable.startswith("ms-"):
            os.startfile(executable)
        else:
            subprocess.Popen(
                executable,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        return True
    except Exception as exc:
        logger.warning("[Actions] Launch failed for '%s': %s", executable, exc)
        return False


def _open_url(url: str) -> bool:
    """Open a URL in the default browser."""
    import webbrowser

    try:
        webbrowser.open(url)
        return True
    except Exception as exc:
        logger.warning("[Actions] Failed to open URL '%s': %s", url, exc)
        return False


def _get_volume_interface() -> object | None:
    """Return the Windows audio endpoint volume interface, or None."""
    if not _PYCAW_AVAILABLE:
        return None
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
    except Exception:
        return None


# ── Handler Implementations ───────────────────────────────────────────────────


def _handle_open_app(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    app_name = (match.group("app") or "").strip().lower()

    executable = _APP_REGISTRY.get(app_name)
    if not executable:
        # Attempt direct launch for unregistered names
        executable = app_name

    if _launch(executable):
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Opening {app_name}.",
            action="open_app",
            duration=time.perf_counter() - start,
        )
    return ActionResult(
        status=ActionStatus.EXECUTION_ERROR,
        message=f"I couldn't open {app_name}. It may not be installed.",
        action="open_app",
        duration=time.perf_counter() - start,
    )


def _handle_close_app(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    app_name = (match.group("app") or "").strip().lower()

    if not _PSUTIL_AVAILABLE:
        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="Process management is unavailable.",
            action="close_app",
        )

    {
        name: proc
        for proc in psutil.process_iter(["name"])
        for name in [proc.info["name"].lower().replace(".exe", "")]
    }

    targets = [
        proc for proc in psutil.process_iter(["name"]) if app_name in proc.info["name"].lower()
    ]

    if not targets:
        return ActionResult(
            status=ActionStatus.NOT_FOUND,
            message=f"I couldn't find a running process matching '{app_name}'.",
            action="close_app",
            duration=time.perf_counter() - start,
        )

    closed = 0
    for proc in targets:
        try:
            proc.terminate()
            closed += 1
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            return ActionResult(
                status=ActionStatus.PERMISSION_ERROR,
                message=f"Access denied closing {app_name}.",
                action="close_app",
                duration=time.perf_counter() - start,
            )

    return ActionResult(
        status=ActionStatus.SUCCESS,
        message=f"Closed {closed} instance{'s' if closed != 1 else ''} of {app_name}.",
        action="close_app",
        duration=time.perf_counter() - start,
    )


def _handle_volume(match: re.Match) -> ActionResult:
    start = time.perf_counter()

    direction = (match.group("direction") or "").lower()
    amount_str = match.group("amount") if "amount" in match.groupdict() else None
    mute_word = (match.group("mute") or "").lower() if "mute" in match.groupdict() else ""

    # pycaw path (Windows precise control)
    volume_interface = _get_volume_interface()

    if volume_interface:
        try:
            if mute_word in ("mute", "unmute"):
                is_muted = volume_interface.GetMute()
                volume_interface.SetMute(not is_muted, None)
                state = "muted" if not is_muted else "unmuted"
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=f"Volume {state}.",
                    action="volume",
                    duration=time.perf_counter() - start,
                )

            current = volume_interface.GetMasterVolumeLevelScalar()
            step = float(amount_str) / 100 if amount_str else 0.1

            if direction == "up":
                new_vol = min(1.0, current + step)
            elif direction == "down":
                new_vol = max(0.0, current - step)
            else:
                new_vol = step  # absolute set

            volume_interface.SetMasterVolumeLevelScalar(new_vol, None)
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Volume set to {int(new_vol * 100)} percent.",
                action="volume",
                duration=time.perf_counter() - start,
            )
        except Exception as exc:
            logger.warning("[Actions] pycaw volume control failed: %s", exc)

    # pyautogui fallback
    if _PYAUTOGUI_AVAILABLE:
        try:
            if mute_word:
                pyautogui.press("volumemute")
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message="Volume toggled.",
                    action="volume",
                )
            key = "volumeup" if direction == "up" else "volumedown"
            presses = int(amount_str) if amount_str else 3
            pyautogui.press(key, presses=presses)
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Volume {'increased' if direction == 'up' else 'decreased'}.",
                action="volume",
                duration=time.perf_counter() - start,
            )
        except Exception as exc:
            logger.warning("[Actions] pyautogui volume fallback failed: %s", exc)

    return ActionResult(
        status=ActionStatus.UNSUPPORTED,
        message="Volume control is not available on this system.",
        action="volume",
    )


def _handle_media(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    command = (match.group("command") or "").lower()

    key_map = {
        "play": "playpause",
        "pause": "playpause",
        "stop": "stop",
        "next": "nexttrack",
        "previous": "prevtrack",
        "back": "prevtrack",
        "skip": "nexttrack",
    }

    key = key_map.get(command)
    if not key:
        return ActionResult(
            status=ActionStatus.NOT_FOUND,
            message=f"Unknown media command '{command}'.",
            action="media",
        )

    if _PYAUTOGUI_AVAILABLE:
        try:
            pyautogui.press(key)
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Media {command}.",
                action="media",
                duration=time.perf_counter() - start,
            )
        except Exception as exc:
            logger.warning("[Actions] Media control failed: %s", exc)

    return ActionResult(
        status=ActionStatus.UNSUPPORTED,
        message="Media control is unavailable.",
        action="media",
    )


def _handle_web_search(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    query = (match.group("query") or "").strip()

    if not query:
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="What would you like me to search for?",
            action="web_search",
        )

    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    if _open_url(url):
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Searching for {query}.",
            action="web_search",
            duration=time.perf_counter() - start,
        )
    return ActionResult(
        status=ActionStatus.EXECUTION_ERROR,
        message="I couldn't open the browser.",
        action="web_search",
    )


def _handle_youtube(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    query = (match.group("query") or "").strip()

    if not query:
        url = "https://www.youtube.com"
        msg = "Opening YouTube."
    else:
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        msg = f"Searching YouTube for {query}."

    if _open_url(url):
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="youtube",
            duration=time.perf_counter() - start,
        )
    return ActionResult(
        status=ActionStatus.EXECUTION_ERROR,
        message="I couldn't open YouTube.",
        action="youtube",
    )


def _handle_time(match: re.Match) -> ActionResult:
    import datetime

    now = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p").lstrip("0")
    return ActionResult(
        status=ActionStatus.SUCCESS,
        message=f"It's {time_str}.",
        action="time",
    )


def _handle_date(match: re.Match) -> ActionResult:
    import datetime

    now = datetime.datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    return ActionResult(
        status=ActionStatus.SUCCESS,
        message=f"Today is {date_str}.",
        action="date",
    )


def _handle_battery(match: re.Match) -> ActionResult:
    if not _PSUTIL_AVAILABLE:
        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="Battery information is unavailable.",
            action="battery",
        )
    battery = psutil.sensors_battery()
    if battery is None:
        return ActionResult(
            status=ActionStatus.NOT_FOUND,
            message="No battery detected. This appears to be a desktop system.",
            action="battery",
        )
    status = "charging" if battery.power_plugged else "discharging"
    return ActionResult(
        status=ActionStatus.SUCCESS,
        message=f"Battery is at {battery.percent:.0f} percent and {status}.",
        action="battery",
    )


def _handle_cpu(match: re.Match) -> ActionResult:
    if not _PSUTIL_AVAILABLE:
        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="System information is unavailable.",
            action="cpu",
        )
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    msg = (
        f"CPU is at {cpu:.0f} percent. "
        f"RAM usage is {ram.percent:.0f} percent, "
        f"with {ram.available / (1024**3):.1f} gigabytes free."
    )
    return ActionResult(
        status=ActionStatus.SUCCESS,
        message=msg,
        action="cpu",
    )


def _handle_screenshot(match: re.Match) -> ActionResult:
    if not _PYAUTOGUI_AVAILABLE:
        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="Screenshot capability is unavailable.",
            action="screenshot",
        )
    try:
        import datetime

        from friday import config

        screenshots_dir = config.storage.base_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        filename = screenshots_dir / f"screenshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
        pyautogui.screenshot(str(filename))
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message="Screenshot saved.",
            action="screenshot",
        )
    except Exception as exc:
        logger.warning("[Actions] Screenshot failed: %s", exc)
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="Screenshot failed.",
            action="screenshot",
        )


def _handle_clipboard(match: re.Match) -> ActionResult:
    if not _PYPERCLIP_AVAILABLE:
        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="Clipboard access is unavailable.",
            action="clipboard",
        )
    try:
        content = pyperclip.paste()
        if not content:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="The clipboard is empty.",
                action="clipboard",
            )
        preview = content[:100].replace("\n", " ")
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Clipboard contains: {preview}{'...' if len(content) > 100 else ''}",
            action="clipboard",
        )
    except Exception as exc:
        logger.warning("[Actions] Clipboard read failed: %s", exc)
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="I couldn't read the clipboard.",
            action="clipboard",
        )


def _handle_shutdown(match: re.Match) -> ActionResult:
    command = (match.group("command") or "shutdown").lower()
    action_map = {
        "shutdown": ("shutdown /s /t 30", "Shutting down in 30 seconds."),
        "restart": ("shutdown /r /t 30", "Restarting in 30 seconds."),
        "sleep": ("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", "Going to sleep."),
        "lock": ("rundll32.exe user32.dll,LockWorkStation", "Locking the workstation."),
        "hibernate": ("shutdown /h", "Hibernating."),
    }
    cmd, msg = action_map.get(command, (None, None))
    if cmd is None:
        return ActionResult(
            status=ActionStatus.NOT_FOUND,
            message=f"Unknown power command '{command}'.",
            action="power",
        )
    try:
        subprocess.run(
            cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="power",
        )
    except subprocess.CalledProcessError as exc:
        logger.error("[Actions] Power command failed: %s", exc)
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="Power command failed.",
            action="power",
        )


def _handle_open_url(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    url = (match.group("url") or "").strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if _open_url(url):
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Opening {url}.",
            action="open_url",
            duration=time.perf_counter() - start,
        )
    return ActionResult(
        status=ActionStatus.EXECUTION_ERROR,
        message="I couldn't open that URL.",
        action="open_url",
    )


# ── Action Registry ───────────────────────────────────────────────────────────

_ACTIONS: list[Action] = [
    Action(
        name="open_app",
        patterns=[
            r"(?:open|launch|start|run)\s+(?P<app>.+)",
        ],
        handler=_handle_open_app,
        description="Open an application by name.",
    ),
    Action(
        name="close_app",
        patterns=[
            r"(?:close|quit|exit|kill)\s+(?P<app>.+)",
        ],
        handler=_handle_close_app,
        description="Close a running application.",
        destructive=True,
    ),
    Action(
        name="volume",
        patterns=[
            r"volume\s+(?P<direction>up|down)(?:\s+(?P<amount>\d+))?",
            r"(?:turn\s+)?(?P<direction>up|down)\s+the\s+volume(?:\s+(?P<amount>\d+))?",
            r"(?:set\s+)?volume\s+to\s+(?P<amount>\d+)(?:\s+percent)?",
            r"(?P<mute>mute|unmute)(?:\s+the)?\s+(?:volume|audio|sound)?",
        ],
        handler=_handle_volume,
        description="Adjust system volume.",
    ),
    Action(
        name="media",
        patterns=[
            r"(?P<command>play|pause|stop|next|previous|back|skip)(?:\s+(?:the\s+)?(?:music|song|track|video))?",
            r"(?:media\s+)?(?P<command>next|previous|back|skip)\s+(?:track|song)",
        ],
        handler=_handle_media,
        description="Control media playback.",
    ),
    Action(
        name="web_search",
        patterns=[
            r"(?:search(?:\s+for)?|google|look\s+up)\s+(?P<query>.+)",
            r"(?:find|search)\s+(?P<query>.+)\s+(?:on(?:line)?|on\s+the\s+web)",
        ],
        handler=_handle_web_search,
        description="Search the web via Google.",
    ),
    Action(
        name="youtube",
        patterns=[
            r"(?:play|search|find|open)\s+(?P<query>.+?)\s+on\s+youtube",
            r"youtube\s+(?:search\s+)?(?P<query>.+)?",
            r"(?:open|go\s+to)\s+youtube",
        ],
        handler=_handle_youtube,
        description="Open or search YouTube.",
    ),
    Action(
        name="time",
        patterns=[
            r"what(?:'s|\s+is)?\s+the\s+time",
            r"(?:current\s+)?time\s+(?:please|now)?",
            r"what\s+time\s+is\s+it",
        ],
        handler=_handle_time,
        description="Report the current time.",
    ),
    Action(
        name="date",
        patterns=[
            r"what(?:'s|\s+is)?\s+(?:the\s+)?(?:date|day)(?:\s+today)?",
            r"(?:today(?:'s)?\s+date|what\s+day\s+is\s+it)",
        ],
        handler=_handle_date,
        description="Report today's date.",
    ),
    Action(
        name="battery",
        patterns=[
            r"(?:battery|power)\s+(?:level|status|percentage|percent|life)?",
            r"how\s+(?:much\s+)?(?:battery|charge|power)\s+(?:do\s+(?:i|we)\s+have|is\s+left|remains?)",
        ],
        handler=_handle_battery,
        description="Report battery status.",
    ),
    Action(
        name="cpu",
        patterns=[
            r"(?:system|cpu|processor|ram|memory)\s+(?:usage|status|stats?|load|info)",
            r"how(?:'s|\s+is)?\s+(?:the\s+)?(?:system|cpu|ram|memory)\s+(?:doing|performing|running)?",
        ],
        handler=_handle_cpu,
        description="Report CPU and RAM usage.",
    ),
    Action(
        name="screenshot",
        patterns=[
            r"(?:take\s+a?\s+)?screenshot",
            r"capture\s+(?:the\s+)?screen",
        ],
        handler=_handle_screenshot,
        description="Capture a screenshot.",
    ),
    Action(
        name="clipboard",
        patterns=[
            r"(?:what(?:'s|\s+is)?\s+(?:in|on)\s+)?(?:the\s+)?clipboard",
            r"(?:read|show|paste)\s+(?:the\s+)?clipboard",
        ],
        handler=_handle_clipboard,
        description="Read clipboard contents.",
    ),
    Action(
        name="power",
        patterns=[
            r"(?P<command>shutdown|restart|reboot|sleep|hibernate|lock)(?:\s+(?:the\s+)?(?:computer|pc|system|laptop))?",
            r"(?:turn\s+off|power\s+off)(?:\s+(?:the\s+)?(?:computer|pc|system))?",
        ],
        handler=_handle_shutdown,
        description="Power management commands.",
        destructive=True,
    ),
    Action(
        name="open_url",
        patterns=[
            r"(?:open|go\s+to|navigate\s+to|visit)\s+(?P<url>(?:https?://)?[\w.-]+\.\w+(?:/\S*)?)",
        ],
        handler=_handle_open_url,
        description="Open a specific URL.",
    ),
]


# ── Action Router ─────────────────────────────────────────────────────────────


class ActionRouter:
    """
    Matches transcribed text against registered actions and dispatches execution.

    The router iterates through all registered actions in order, returning
    the result of the first match. Destructive actions pass through a
    confirmation gate when CONFIRM_DESTRUCTIVE_ACTIONS is enabled.

    Usage:
        router = ActionRouter()
        result = router.route("open chrome")
        if result:
            print(result.message)
    """

    def __init__(self, actions: list[Action] | None = None) -> None:
        self._actions = actions or _ACTIONS
        self._pending_confirmation: tuple[Action, re.Match] | None = None
        logger.debug("[ActionRouter] Initialised with %d actions.", len(self._actions))

    def route(self, text: str) -> ActionResult | None:
        """
        Attempt to match and execute a local action for the given text.

        Args:
            text: Normalised transcript from STT.

        Returns:
            ActionResult if a match was found, None if no action matched
            (caller should escalate to LLM).
        """
        if not text:
            return None

        text = text.strip().lower()

        # Handle pending confirmation
        if self._pending_confirmation:
            return self._resolve_confirmation(text)

        for action in self._actions:
            match = action.match(text)
            if match:
                return self._dispatch(action, match)

        return None  # No match — let the engine route to LLM

    def register(self, action: Action) -> None:
        """
        Register a new action at runtime.
        New actions are checked before built-in ones (prepended).

        Args:
            action: The Action to register.
        """
        self._actions.insert(0, action)
        logger.debug("[ActionRouter] Registered action '%s'.", action.name)

    def unregister(self, name: str) -> bool:
        """
        Remove an action by name.

        Args:
            name: The action name to remove.

        Returns:
            True if found and removed, False otherwise.
        """
        for i, action in enumerate(self._actions):
            if action.name == name:
                self._actions.pop(i)
                logger.debug("[ActionRouter] Unregistered action '%s'.", name)
                return True
        return False

    def list_actions(self) -> list[dict]:
        """Return a summary of all registered actions."""
        return [
            {
                "name": a.name,
                "description": a.description,
                "destructive": a.destructive,
                "patterns": a.patterns,
            }
            for a in self._actions
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _dispatch(self, action: Action, match: re.Match) -> ActionResult:
        """
        Execute an action, gating destructive ones behind confirmation.

        Args:
            action: The matched Action.
            match:  The regex match object.

        Returns:
            ActionResult from the handler or confirmation gate.
        """
        if action.destructive and config.security.confirm_destructive:
            self._pending_confirmation = (action, match)
            return ActionResult(
                status=ActionStatus.REQUIRES_CONFIRM,
                message=f"Are you sure you want to {action.name.replace('_', ' ')}? Say yes to confirm.",
                action=action.name,
            )

        start = time.perf_counter()
        try:
            result = action.handler(match)
            logger.info(
                "[ActionRouter] '%s' → %s in %.3fs",
                action.name,
                result.status.name,
                time.perf_counter() - start,
            )
            return result
        except Exception as exc:
            logger.exception("[ActionRouter] Handler '%s' raised: %s", action.name, exc)
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Something went wrong executing that action.",
                action=action.name,
                duration=time.perf_counter() - start,
            )

    def _resolve_confirmation(self, text: str) -> ActionResult | None:
        """
        Resolve a pending destructive action based on user confirmation.

        Args:
            text: User's confirmation or denial response.

        Returns:
            ActionResult from execution or cancellation.
        """
        action, match = self._pending_confirmation
        self._pending_confirmation = None

        affirmative = any(
            word in text for word in ("yes", "yeah", "yep", "confirm", "do it", "go ahead")
        )
        if affirmative:
            return self._dispatch(action, match)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message="Action cancelled.",
            action=action.name,
        )
