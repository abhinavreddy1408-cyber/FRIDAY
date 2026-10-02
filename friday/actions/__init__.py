"""
F.R.I.D.A.Y. Action Dispatcher
==============================
Routes tool calls and local commands to their respective modules.
"""

import ctypes
import logging
import os
import re
import subprocess
import webbrowser
from collections.abc import Callable
from datetime import datetime
from typing import Any, Dict, Optional, Tuple  # noqa: UP035, F401

from friday import desktop_tools
from . import browser_tools, file_tools
from .local_router import (
    Action as Action,
    ActionResult as ActionResult,
    ActionRouter as ActionRouter,
    ActionStatus as ActionStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SAFE APP OPENER
# Finds real executable paths instead of running app name as terminal command
# =============================================================================


def _open_app_safe(name: str) -> dict:
    """
    Safely open common Windows applications by finding
    their actual executable path on disk.
    """
    name_lower = name.lower().strip()

    APP_PATHS = {
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "google chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "firefox": [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ],
        "edge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
        "microsoft edge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ],
        "notepad": [
            r"C:\Windows\System32\notepad.exe",
        ],
        "calculator": [
            r"C:\Windows\System32\calc.exe",
        ],
        "paint": [
            r"C:\Windows\System32\mspaint.exe",
        ],
        "word": [
            r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
            r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
        ],
        "excel": [
            r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
            r"C:\Program Files (x86)\Microsoft Office\root\Office16\EXCEL.EXE",
        ],
        "powerpoint": [
            r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
            r"C:\Program Files (x86)\Microsoft Office\root\Office16\POWERPNT.EXE",
        ],
        "spotify": [
            os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe"),
        ],
        "discord": [
            os.path.expandvars(r"%LOCALAPPDATA%\Discord\Update.exe"),
        ],
        "vscode": [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\Program Files\Microsoft VS Code\Code.exe",
        ],
        "visual studio code": [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
        ],
        "task manager": [r"C:\Windows\System32\Taskmgr.exe"],
        "file explorer": [r"C:\Windows\explorer.exe"],
        "explorer": [r"C:\Windows\explorer.exe"],
        "cmd": [r"C:\Windows\System32\cmd.exe"],
        "powershell": [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        ],
        "whatsapp": [
            os.path.expandvars(r"%LOCALAPPDATA%\WhatsApp\WhatsApp.exe"),
        ],
    }

    paths = APP_PATHS.get(name_lower, [])
    for path in paths:
        expanded = os.path.expandvars(path)
        if os.path.exists(expanded):
            try:
                subprocess.Popen(
                    [expanded],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"[APP] Opened '{name}' from: {expanded}")
                return {"success": True, "message": f"Opened {name}"}
            except Exception as e:
                logger.error(f"[APP] Failed: {expanded}: {e}")

    # Fallback: Windows start command
    try:
        subprocess.Popen(
            f'start "" "{name_lower}"',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"[APP] Opened '{name}' via start command")
        return {"success": True, "message": f"Opened {name}"}
    except Exception as e:
        logger.error(f"[APP] All methods failed for '{name}': {e}")
        return {
            "success": False,
            "error": f"Could not open '{name}'. Is it installed?",
        }


# =============================================================================
# MAIN TOOL DISPATCHER
# =============================================================================


def execute_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Routes a tool call from the Planner or LLM to the correct
    implementation. Always returns a dict with at minimum:
    {"success": bool, ...}
    """
    logger.info(f"[DISPATCH] Tool='{tool_name}' Args={args}")

    try:
        # ── VISION ────────────────────────────────────────────────────
        if tool_name == "capture_screen":
            from friday.skills import vision_skill
            return vision_skill.capture_screen()

        elif tool_name == "capture_webcam":
            from friday.skills import vision_skill
            return vision_skill.capture_webcam()

        # ── IOT ───────────────────────────────────────────────────────
        elif tool_name == "iot_control":
            device = args.get("device_id") or args.get("device")
            state = args.get("state")
            if not device or not state:
                return {"success": False, "error": "Missing device_id or state"}
            from friday.skills import iot_skill
            return iot_skill.toggle_light(device, state)

        # ── SECURITY ──────────────────────────────────────────────────
        elif tool_name == "security_scan":
            from friday.skills import security_skill
            return security_skill.monitor_network_connections()

        elif tool_name == "check_integrity":
            from friday.skills import security_skill
            return security_skill.check_pqc_integrity()

        # ── RESEARCH ──────────────────────────────────────────────────
        elif tool_name == "start_research":
            query = args.get("query")
            urls = args.get("urls", [])
            pdfs = args.get("pdf_paths", [])
            if not query:
                return {"success": False, "error": "No query provided"}
            from friday.skills import research_skill
            return research_skill.start_background_research(query, urls, pdfs)

        # ── TIME ──────────────────────────────────────────────────────
        if tool_name in ("get_time", "current_time", "what_time"):
            t = datetime.now().strftime("%I:%M %p")
            return {"success": True, "time": t, "message": f"It is {t}, Sir."}

        # ── DATE ──────────────────────────────────────────────────────
        elif tool_name in ("get_date", "current_date", "what_date"):
            d = datetime.now().strftime("%A, %B %d, %Y")
            return {"success": True, "date": d, "message": f"Today is {d}, Sir."}

        # ── OPEN URL ──────────────────────────────────────────────────
        elif tool_name in ("open_url", "browser_open", "browser_navigate", "navigate_to"):
            url = args.get("url", args.get("link", ""))
            if not url:
                return {"success": False, "error": "No URL provided"}
            if not url.startswith("http"):
                url = "https://" + url
            webbrowser.open(url)
            return {"success": True, "message": f"Opened {url}", "url": url}

        # ── WEB SEARCH ────────────────────────────────────────────────
        elif tool_name in ("browser_search", "google_search", "search_web", "web_search"):
            query = args.get("query") or args.get("q") or args.get("search") or ""
            if not query:
                return {"success": False, "error": "No search query provided"}
            try:
                result = browser_tools.google_search(query=query)
                return result
            except Exception:
                webbrowser.open(f"https://www.google.com/search?q={query}")
                return {"success": True, "message": f"Searched for {query}"}

        # ── YOUTUBE ───────────────────────────────────────────────────
        elif tool_name in ("youtube_play", "media_play_youtube", "play_youtube", "youtube_search"):
            query = (
                args.get("query")
                or args.get("video_name")
                or args.get("video")
                or args.get("search")
                or "music"
            )
            try:
                result = browser_tools.youtube_play(query=query)
                return result
            except Exception:
                webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
                return {"success": True, "message": f"Searching YouTube for {query}"}

        # ── BROWSER ACTIONS ───────────────────────────────────────────
        elif tool_name in ("browser_click", "browser_smart_click"):
            return browser_tools.browser_click(**args)

        elif tool_name in ("browser_type", "browser_smart_type"):
            return browser_tools.browser_type(**args)

        elif tool_name == "browser_read_page":
            return browser_tools.browser_read_current_page()

        elif tool_name == "browser_go_back":
            return browser_tools.browser_go_back()

        elif tool_name == "browser_screenshot":
            return browser_tools.browser_take_screenshot()

        elif tool_name == "browser_scroll":
            return browser_tools.browser_scroll(args.get("direction", "down"))

        elif tool_name == "browser_close":
            return browser_tools.browser_close()

        elif tool_name == "browser_get_current_url":
            return browser_tools.browser_get_current_url()

        # ── APP LAUNCHING ─────────────────────────────────────────────
        elif tool_name in ("desktop_open_app", "open_app", "launch_app", "start_app"):
            name = args.get("name") or args.get("app") or args.get("application") or ""
            return _open_app_safe(name)

        # ── DESKTOP CONTROL ───────────────────────────────────────────
        elif tool_name == "desktop_focus_window":
            title = args.get("title", args.get("window", ""))
            result = desktop_tools.desktop_focus_window(title=title)
            return {"success": True, "message": str(result)}

        elif tool_name == "desktop_list_windows":
            return desktop_tools.desktop_list_windows()

        elif tool_name == "desktop_type_text":
            text = args.get("text", args.get("content", ""))
            return desktop_tools.desktop_type_text(text=text)

        elif tool_name == "desktop_press_key":
            key = args.get("key", args.get("keys", ""))
            return desktop_tools.desktop_press_key(key=key)

        elif tool_name == "desktop_read_window":
            title = args.get("title", "")
            return desktop_tools.desktop_read_window(title=title)

        elif tool_name == "desktop_get_clipboard":
            return desktop_tools.desktop_get_clipboard()

        elif tool_name == "desktop_set_clipboard":
            text = args.get("text", "")
            return desktop_tools.desktop_set_clipboard(text=text)

        elif tool_name in ("desktop_close_window", "close_window"):
            title = args.get("title", args.get("window", ""))
            return desktop_tools.desktop_close_window(title=title)

        elif tool_name in ("desktop_mouse_click", "mouse_click"):
            x = args.get("x", 0)
            y = args.get("y", 0)
            return desktop_tools.desktop_mouse_click(x=x, y=y)

        # ── KILL APP ──────────────────────────────────────────────────
        elif tool_name in ("kill_app", "terminate_app", "close_app", "force_close"):
            app_name = args.get("name") or args.get("app") or ""
            app_name = app_name.replace(".exe", "")
            try:
                result = subprocess.run(
                    f"taskkill /F /IM {app_name}.exe",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return {"success": True, "message": f"Terminated {app_name}"}
                return {"success": False, "error": f"Could not find {app_name}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── BATTERY ───────────────────────────────────────────────────
        elif tool_name in ("battery_status", "get_battery", "get_battery_level", "battery"):
            return desktop_tools.get_battery_level()

        # ── WIFI ──────────────────────────────────────────────────────
        elif tool_name in ("wifi_status", "get_wifi", "network_status"):
            try:
                return desktop_tools.get_wifi_status()
            except Exception:
                import socket

                try:
                    socket.create_connection(("8.8.8.8", 53), timeout=3)
                    return {
                        "success": True,
                        "status": "Connected",
                        "message": "Internet connection active",
                    }
                except OSError:
                    return {
                        "success": True,
                        "status": "Disconnected",
                        "message": "No internet connection",
                    }

        # ── CPU / RAM ─────────────────────────────────────────────────
        elif tool_name in ("cpu_ram_status", "system_status", "get_cpu_ram", "performance"):
            try:
                import psutil

                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.virtual_memory()
                return {
                    "success": True,
                    "cpu_percent": cpu,
                    "ram_percent": ram.percent,
                    "ram_used_gb": round(ram.used / (1024**3), 1),
                    "ram_total_gb": round(ram.total / (1024**3), 1),
                    "message": (
                        f"CPU at {cpu}%, RAM at {ram.percent}% "
                        f"({round(ram.used / (1024**3), 1)}GB used)"
                    ),
                }
            except ImportError:
                return {"success": False, "error": "psutil not installed. Run: pip install psutil"}

        # ── RUNNING APPS ──────────────────────────────────────────────
        elif tool_name in ("running_apps", "list_processes"):
            try:
                return desktop_tools.running_apps()
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── DISK USAGE ────────────────────────────────────────────────
        elif tool_name in ("disk_usage", "storage_status"):
            try:
                import psutil

                drive = args.get("drive", "C:")
                usage = psutil.disk_usage(drive)
                return {
                    "success": True,
                    "drive": drive,
                    "total_gb": round(usage.total / (1024**3), 1),
                    "used_gb": round(usage.used / (1024**3), 1),
                    "free_gb": round(usage.free / (1024**3), 1),
                    "percent_used": usage.percent,
                    "message": (
                        f"{drive} drive: "
                        f"{round(usage.free / (1024**3), 1)}GB free "
                        f"of {round(usage.total / (1024**3), 1)}GB"
                    ),
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── SCREENSHOT ────────────────────────────────────────────────
        elif tool_name in ("take_screenshot", "screenshot", "capture_screen"):
            try:
                import tempfile
                from pathlib import Path

                import mss

                with mss.mss() as sct:
                    path = Path(tempfile.gettempdir()) / "friday_screenshot.png"
                    sct.shot(output=str(path))
                    return {"success": True, "path": str(path), "message": "Screenshot taken"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── LOCK SCREEN ───────────────────────────────────────────────
        elif tool_name in ("system_lock", "lock_screen", "lock_computer"):
            try:
                ctypes.windll.user32.LockWorkStation()
                return {"success": True, "message": "Workstation locked"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── SHUTDOWN ──────────────────────────────────────────────────
        elif tool_name in ("system_shutdown", "shutdown"):
            delay = args.get("delay", 60)
            subprocess.run(f"shutdown /s /t {delay}", shell=True)
            return {"success": True, "message": f"Shutting down in {delay} seconds"}

        # ── RESTART ───────────────────────────────────────────────────
        elif tool_name in ("system_restart", "restart"):
            delay = args.get("delay", 60)
            subprocess.run(f"shutdown /r /t {delay}", shell=True)
            return {"success": True, "message": f"Restarting in {delay} seconds"}

        # ── VOLUME ────────────────────────────────────────────────────
        elif tool_name in ("volume_set", "set_volume"):
            level = args.get("level", args.get("volume", 50))
            try:
                from ctypes import POINTER, cast

                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(
                    IAudioEndpointVolume._iid_,
                    CLSCTX_ALL,
                    None,
                )
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                vol = max(0.0, min(1.0, level / 100.0))
                volume.SetMasterVolumeLevelScalar(vol, None)
                return {"success": True, "message": f"Volume set to {level}%"}
            except Exception as e:
                return {"success": False, "error": f"Volume control failed: {e}"}

        elif tool_name in ("volume_mute", "mute_volume"):
            try:
                import pyautogui

                pyautogui.press("volumemute")
                return {"success": True, "message": "Muted"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name in ("volume_unmute", "unmute_volume"):
            try:
                import pyautogui

                pyautogui.press("volumemute")
                return {"success": True, "message": "Unmuted"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── FILE TOOLS ────────────────────────────────────────────────
        elif tool_name == "file_read":
            return {"success": True, "output": file_tools.file_read(**args)}

        elif tool_name == "file_write":
            return {"success": True, "output": file_tools.file_write(**args)}

        elif tool_name == "file_list":
            directory = args.get("directory", args.get("path", "."))
            return {"success": True, "output": file_tools.file_list(directory=directory)}

        elif tool_name == "file_exists":
            path = args.get("path", "")
            return {"success": True, "output": file_tools.file_exists(path=path)}

        elif tool_name == "file_delete":
            return {"success": True, "output": file_tools.file_delete(**args)}

        elif tool_name == "file_move":
            return {"success": True, "output": file_tools.file_move(**args)}

        elif tool_name == "file_copy":
            return {"success": True, "output": file_tools.file_copy(**args)}

        elif tool_name == "file_search":
            return {"success": True, "output": file_tools.file_search(**args)}

        elif tool_name == "file_create_folder":
            return {"success": True, "output": file_tools.file_create_folder(**args)}

        elif tool_name == "file_get_info":
            return {"success": True, "output": file_tools.file_get_info(**args)}

        # ── WEATHER ───────────────────────────────────────────────────
        elif tool_name in ("get_weather", "weather", "weather_status"):
            city = args.get("city", args.get("location", ""))
            try:
                import requests  # type: ignore[import-untyped]

                url = f"https://wttr.in/{city}?format=3" if city else "https://wttr.in/?format=3"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return {
                        "success": True,
                        "weather": resp.text.strip(),
                        "message": resp.text.strip(),
                    }
                return {"success": False, "error": "Could not fetch weather"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── MEDIA CONTROLS ────────────────────────────────────────────
        elif tool_name in ("media_pause", "pause_media", "pause"):
            try:
                import pyautogui

                pyautogui.press("playpause")
                return {"success": True, "message": "Media paused/resumed"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name in ("media_next", "next_track"):
            try:
                import pyautogui

                pyautogui.press("nexttrack")
                return {"success": True, "message": "Skipped to next track"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name in ("media_previous", "prev_track", "previous_track"):
            try:
                import pyautogui

                pyautogui.press("prevtrack")
                return {"success": True, "message": "Went to previous track"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool_name in ("media_play_spotify", "spotify_play", "open_spotify"):
            query = args.get("query") or args.get("song") or args.get("artist") or ""
            if query:
                webbrowser.open(f"https://open.spotify.com/search/{query}")
            else:
                webbrowser.open("https://open.spotify.com")
            return {
                "success": True,
                "message": f"Opening Spotify{' for ' + query if query else ''}",
            }

        # ── UNKNOWN TOOL ──────────────────────────────────────────────
        else:
            logger.warning(f"[DISPATCH] Unknown tool: '{tool_name}'")
            return {
                "success": False,
                "error": (
                    f"Tool '{tool_name}' is not registered. Add it to friday/actions/__init__.py"
                ),
            }

    except Exception as e:
        logger.error(
            f"[DISPATCH] Tool '{tool_name}' crashed: {e}",
            exc_info=True,
        )
        return {"success": False, "error": f"Tool crashed: {e!s}"}


# =============================================================================
# LOCAL INSTANT ROUTER
# Handles common commands instantly without calling the LLM
# =============================================================================

SITE_MAP = {
    "youtube": "https://youtube.com",
    "google": "https://google.com",
    "github": "https://github.com",
    "chatgpt": "https://chat.openai.com",
    "gmail": "https://mail.google.com",
    "reddit": "https://reddit.com",
    "twitter": "https://twitter.com",
    "instagram": "https://instagram.com",
    "netflix": "https://netflix.com",
    "linkedin": "https://linkedin.com",
    "spotify": "https://open.spotify.com",
    "amazon": "https://amazon.in",
    "flipkart": "https://flipkart.com",
    "whatsapp": "https://web.whatsapp.com",
    "facebook": "https://facebook.com",
    "wikipedia": "https://wikipedia.org",
}


def try_execute_local(
    command: str,
) -> tuple[bool, str | None, Callable[[], Any] | None]:
    """
    Handles common commands INSTANTLY without calling the LLM.

    Returns:
        (handled, response_text, action_callable)

        handled=True, action=callable → speak text then call action()
        handled=True, action=None    → response_text has the answer
        handled=False                → send to LLM planner
    """
    cmd = command.lower().strip()

    # ── TIME ──────────────────────────────────────────────────────────
    if any(
        p in cmd
        for p in [
            "what time",
            "current time",
            "tell me the time",
            "what's the time",
            "whats the time",
        ]
    ):
        t = datetime.now().strftime("%I:%M %p")
        return True, f"It is currently {t}, Sir.", None

    # ── DATE ──────────────────────────────────────────────────────────
    if any(
        p in cmd
        for p in ["what date", "today's date", "what day is it", "current date", "todays date"]
    ):
        d = datetime.now().strftime("%A, %B %d, %Y")
        return True, f"Today is {d}, Sir.", None

    # ── BATTERY ───────────────────────────────────────────────────────
    if "battery" in cmd:
        res = desktop_tools.get_battery_level()
        pct = res.get("percent", "unknown")
        charging = res.get("charging", False)
        status = " and charging" if charging else ""
        return (True, f"Your battery is at {pct} percent{status}.", None)

    # ── EXIT FRIDAY ───────────────────────────────────────────────────
    if any(
        p in cmd
        for p in [
            "goodbye friday",
            "shut down friday",
            "exit friday",
            "stop friday",
            "bye friday",
            "shutdown friday",
        ]
    ):
        return (True, "Goodbye Sir. Shutting down now.", lambda: os._exit(0))

    # ── VOLUME CONTROLS ───────────────────────────────────────────────
    if "volume up" in cmd or "increase volume" in cmd:

        def _vol_up() -> None:
            import pyautogui

            for _ in range(5):
                pyautogui.press("volumeup")

        return True, "Increasing volume, Sir.", _vol_up

    if "volume down" in cmd or "decrease volume" in cmd:

        def _vol_down() -> None:
            import pyautogui

            for _ in range(5):
                pyautogui.press("volumedown")

        return True, "Decreasing volume, Sir.", _vol_down

    if "mute" in cmd and "unmute" not in cmd:

        def _mute() -> None:
            import pyautogui

            pyautogui.press("volumemute")

        return True, "Toggling mute, Sir.", _mute

    # ── SCREENSHOT ────────────────────────────────────────────────────
    if "screenshot" in cmd or "take a screenshot" in cmd:

        def _screenshot() -> None:
            execute_tool("take_screenshot", {})

        return True, "Taking a screenshot, Sir.", _screenshot

    # ── LOCK SCREEN ───────────────────────────────────────────────────
    if "lock" in cmd and any(w in cmd for w in ["screen", "computer", "workstation", "pc"]):

        def _lock() -> None:
            ctypes.windll.user32.LockWorkStation()

        return True, "Locking your screen, Sir.", _lock

    # ── OPEN WEBSITES ─────────────────────────────────────────────────
    open_match = re.match(
        r"(?:open|launch|go to|navigate to|show me|start)\s+"
        r"([a-zA-Z0-9\s\.\-]+)",
        cmd,
    )
    if open_match:
        target = open_match.group(1).strip().lower()

        # Check site map first
        for site_key, site_url in SITE_MAP.items():
            if site_key in target:
                url = site_url

                def _open_site(u: str = url, k: str = site_key) -> None:
                    webbrowser.open(u)

                return (True, f"Opening {site_key.capitalize()}, Sir.", _open_site)

        # Looks like a domain (has dot, no spaces)
        if "." in target and " " not in target:
            url = f"https://{target}"

            def _open_url(u: str = url) -> None:
                webbrowser.open(u)

            return True, f"Opening {target}, Sir.", _open_url

        # Treat as app name
        def _open_app(t: str = target) -> None:
            _open_app_safe(t)

        return (True, f"Launching {target.capitalize()}, Sir.", _open_app)

    # ── PLAY ON YOUTUBE ───────────────────────────────────────────────
    play_match = re.match(r"play\s+(.+?)(?:\s+on\s+youtube)?$", cmd)
    if play_match:
        query = play_match.group(1).strip()

        def _play(q: str = query) -> None:
            webbrowser.open(f"https://www.youtube.com/results?search_query={q}")

        return (True, f"Playing {query} on YouTube, Sir.", _play)

    # ── SEARCH ────────────────────────────────────────────────────────
    search_match = re.match(
        r"(?:search|search for|google|look up|find)\s+(.+)",
        cmd,
    )
    if search_match:
        query = search_match.group(1).strip()

        def _search(q: str = query) -> None:
            webbrowser.open(f"https://www.google.com/search?q={q}")

        return True, f"Searching for {query}, Sir.", _search

    # ── NOT HANDLED LOCALLY ───────────────────────────────────────────
    return False, None, None
