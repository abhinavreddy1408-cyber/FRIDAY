"""
F.R.I.D.A.Y. Local Action Router
================================
Handles all local command execution (e.g., "volume up", "open chrome")
without requiring LLM involvement.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from friday.config import config

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
    import pyperclip  # noqa: F401

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
    FORBIDDEN = auto()


@dataclass(frozen=True)
class ActionResult:
    """
    Immutable result of an action execution attempt.
    """

    status: ActionStatus
    message: str
    action: str = ""
    duration: float = 0.0
    data: Any = None

    @property
    def success(self) -> bool:
        return self.status == ActionStatus.SUCCESS

    def __str__(self) -> str:
        return self.message


@dataclass
class Action:
    """
    Descriptor for a single registered action.
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
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "notepad": "notepad",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "vscode": "code",
    "spotify": "spotify",
    "discord": "discord",
    "task manager": "taskmgr",
    "file explorer": "explorer",
    "explorer": "explorer",
    "settings": "ms-settings:",
    "calculator": "calc",
    "paint": "mspaint",
}


# ── System Utilities ──────────────────────────────────────────────────────────


def _launch(executable: str) -> bool:
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
    import webbrowser

    try:
        webbrowser.open(url)
        return True
    except Exception as exc:
        logger.warning("[Actions] Failed to open URL '%s': %s", url, exc)
        return False


def _get_volume_interface() -> object | None:
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
    executable = _APP_REGISTRY.get(app_name, app_name)
    if _launch(executable):
        from friday.personality import ResponseCategory, get_voice_line

        prefix = get_voice_line(ResponseCategory.ACKNOWLEDGEMENT)
        return ActionResult(
            ActionStatus.SUCCESS,
            f"{prefix} Opening {app_name}.",
            "open_app",
            time.perf_counter() - start,
        )
    return ActionResult(
        ActionStatus.EXECUTION_ERROR,
        f"I couldn't open {app_name}.",
        "open_app",
        time.perf_counter() - start,
    )


def _handle_close_app(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    app_name = (match.group("app") or "").strip().lower()
    if not _PSUTIL_AVAILABLE:
        return ActionResult(
            ActionStatus.UNSUPPORTED, "Process management unavailable.", "close_app"
        )

    targets = [p for p in psutil.process_iter(["name"]) if app_name in p.info["name"].lower()]
    if not targets:
        return ActionResult(
            ActionStatus.NOT_FOUND,
            f"No process matching '{app_name}' found.",
            "close_app",
            time.perf_counter() - start,
        )

    closed = 0
    for proc in targets:
        try:
            proc.terminate()
            closed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return ActionResult(
        ActionStatus.SUCCESS,
        f"Closed {closed} instance(s) of {app_name}.",
        "close_app",
        time.perf_counter() - start,
    )


def _handle_volume(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    direction = (match.group("direction") or "").lower()
    amount_str = match.group("amount") if "amount" in match.groupdict() else None
    mute_word = (match.group("mute") or "").lower() if "mute" in match.groupdict() else ""

    volume_interface = _get_volume_interface()
    if volume_interface:
        try:
            if mute_word in ("mute", "unmute"):
                is_muted = volume_interface.GetMute()
                volume_interface.SetMute(not is_muted, None)
                return ActionResult(
                    ActionStatus.SUCCESS,
                    f"Volume {'muted' if not is_muted else 'unmuted'}.",
                    "volume",
                    time.perf_counter() - start,
                )

            current = volume_interface.GetMasterVolumeLevelScalar()
            step = float(amount_str) / 100 if amount_str else 0.1
            new_vol = (
                min(1.0, current + step)
                if direction == "up"
                else max(0.0, current - step)
                if direction == "down"
                else step
            )
            volume_interface.SetMasterVolumeLevelScalar(new_vol, None)
            return ActionResult(
                ActionStatus.SUCCESS,
                f"Volume set to {int(new_vol * 100)}%.",
                "volume",
                time.perf_counter() - start,
            )
        except Exception:
            pass

    if _PYAUTOGUI_AVAILABLE:
        try:
            if mute_word:
                pyautogui.press("volumemute")
            else:
                pyautogui.press(
                    "volumeup" if direction == "up" else "volumedown",
                    presses=int(amount_str) if amount_str else 3,
                )
            return ActionResult(
                ActionStatus.SUCCESS, "Volume adjusted.", "volume", time.perf_counter() - start
            )
        except Exception:
            pass

    return ActionResult(ActionStatus.UNSUPPORTED, "Volume control unavailable.", "volume")


def _handle_time(match: re.Match) -> ActionResult:
    from datetime import datetime

    return ActionResult(
        ActionStatus.SUCCESS, f"It's {datetime.now().strftime('%I:%M %p')}.", "time"
    )


def _handle_date(match: re.Match) -> ActionResult:
    from datetime import datetime

    return ActionResult(
        ActionStatus.SUCCESS, f"Today is {datetime.now().strftime('%A, %B %d, %Y')}.", "date"
    )


# ── Iron Man Skills Handlers ──────────────────────────────────────────────────


def _handle_vision(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    try:
        from friday.skills import vision_skill
    except ImportError:
        return ActionResult(
            ActionStatus.UNSUPPORTED,
            "Vision skill is unavailable — missing dependencies.",
            "vision",
        )
    mode = (match.group("mode") or "screen").lower()
    if "webcam" in mode or "camera" in mode:
        res = vision_skill.capture_webcam()
    else:
        res = vision_skill.capture_screen()
    status = ActionStatus.SUCCESS if res["success"] else ActionStatus.EXECUTION_ERROR
    return ActionResult(status, res["message"], "vision", time.perf_counter() - start)


def _handle_iot(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    try:
        from friday.skills import iot_skill
    except ImportError:
        return ActionResult(
            ActionStatus.UNSUPPORTED,
            "IoT skill is unavailable — missing dependencies.",
            "iot",
        )
    device = match.group("device")
    state = match.group("state")
    res = iot_skill.toggle_light(device, state)
    status = ActionStatus.SUCCESS if res["success"] else ActionStatus.EXECUTION_ERROR
    return ActionResult(status, res["message"], "iot", time.perf_counter() - start)


def _handle_security(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    try:
        from friday.skills import security_skill
    except ImportError:
        return ActionResult(
            ActionStatus.UNSUPPORTED,
            "Security skill is unavailable — missing dependencies.",
            "security",
        )
    res = security_skill.monitor_network_connections()
    return ActionResult(
        ActionStatus.SUCCESS, res["message"], "security", time.perf_counter() - start
    )


def _handle_research(match: re.Match) -> ActionResult:
    start = time.perf_counter()
    try:
        from friday.skills import research_skill
    except ImportError:
        return ActionResult(
            ActionStatus.UNSUPPORTED,
            "Research skill is unavailable — missing dependencies.",
            "research",
        )
    query = match.group("query")
    # For local command, we'll start a background research task
    res = research_skill.start_background_research(query)
    return ActionResult(
        ActionStatus.SUCCESS, res["message"], "research", time.perf_counter() - start
    )


# ── Action Registry ───────────────────────────────────────────────────────────

_ACTIONS: list[Action] = [
    Action(
        "open_app", [r"(?:open|launch|start|run)\s+(?P<app>.+)"], _handle_open_app, "Open an app."
    ),
    Action(
        "close_app",
        [r"(?:close|quit|exit|kill)\s+(?P<app>.+)"],
        _handle_close_app,
        "Close an app.",
        destructive=True,
    ),
    Action(
        "volume",
        [
            r"volume\s+(?P<direction>up|down)(?:\s+(?P<amount>\d+))?",
            r"(?P<mute>mute|unmute)(?:\s+the)?\s+(?:volume|audio|sound)?",
        ],
        _handle_volume,
        "Adjust volume.",
    ),
    Action(
        "time",
        [r"what(?:'s|\s+is)?\s+the\s+time", r"what\s+time\s+is\s+it"],
        _handle_time,
        "Report time.",
    ),
    Action(
        "date",
        [r"what(?:'s|\s+is)?\s+(?:the\s+)?(?:date|day)", r"what\s+day\s+is\s+it"],
        _handle_date,
        "Report date.",
    ),
    # New Skills
    Action(
        "vision",
        [r"(?:take a? |capture (the )?)(?P<mode>screen|screenshot|webcam|camera)"],
        _handle_vision,
        "Capture visual data.",
    ),
    Action(
        "iot",
        [r"turn (?P<state>on|off) (the )?(?P<device>[\w\s]+) (light|lamp)"],
        _handle_iot,
        "Control IoT devices.",
    ),
    Action(
        "security",
        [r"(?:run a? |perform a? |start (the )?)(?:security |network )?(?:scan|monitor|audit)"],
        _handle_security,
        "Monitor network security.",
    ),
    Action(
        "research",
        [r"research\s+(?P<query>.+)", r"look\s+up\s+(?P<query>.+)"],
        _handle_research,
        "Run background web research on a topic.",
    ),
]


# ── Action Router ─────────────────────────────────────────────────────────────


class ActionRouter:
    """
    Matches transcribed text against registered actions and dispatches execution.
    """

    def __init__(self, actions: list[Action] | None = None) -> None:
        self._actions = actions if actions is not None else list(_ACTIONS)
        self._pending_confirmation: tuple[Action, re.Match] | None = None

    def register(self, action: Action, priority: bool = False) -> None:
        """Dynamically add an action to the router."""
        if priority:
            self._actions.insert(0, action)
        else:
            self._actions.append(action)

    def route(self, text: str) -> ActionResult | None:
        if not text:
            return None
        text = text.strip().lower()

        if self._pending_confirmation:
            return self._resolve_confirmation(text)

        for action in self._actions:
            match = action.match(text)
            if match:
                return self._dispatch(action, match)
        return None

    def _dispatch(self, action: Action, match: re.Match) -> ActionResult:
        if action.destructive and config.security.confirm_destructive:
            self._pending_confirmation = (action, match)
            return ActionResult(
                ActionStatus.REQUIRES_CONFIRM,
                f"Are you sure you want to {action.name}? Say yes to confirm.",
                action.name,
            )

        try:
            return action.handler(match)
        except Exception as exc:
            logger.error(f"[ActionRouter] Error: {exc}")
            return ActionResult(ActionStatus.EXECUTION_ERROR, "Something went wrong.", action.name)

    def _resolve_confirmation(self, text: str) -> ActionResult | None:
        action, match = self._pending_confirmation
        self._pending_confirmation = None
        if any(w in text for w in ("yes", "yeah", "yep", "confirm", "do it")):
            return self._dispatch(action, match)
        return ActionResult(ActionStatus.SUCCESS, "Action cancelled.", action.name)
