import os
import subprocess
import time
from typing import Any

try:
    import pywinauto  # type: ignore[import-untyped] # noqa: F401
    from pywinauto import Application, Desktop  # noqa: F401
    from pywinauto.findwindows import (  # type: ignore[import-untyped] # noqa: F401
        ElementNotFoundError,
        find_windows,
    )

    _PYWINAUTO_AVAILABLE = True
except ImportError:
    _PYWINAUTO_AVAILABLE = False

try:
    import pyautogui

    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.1
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False


# ── App launch paths ────────────────────────────────────────

_KNOWN_APPS: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "wordpad": "wordpad.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "spotify": "spotify",
    "discord": "discord",
    "vscode": "code",
    "chrome": "chrome",
    "edge": "msedge",
}


# ── Tool implementations ────────────────────────────────────


def desktop_open_app(name: str = "") -> str:
    """Launch a Windows application by name."""
    if not name:
        return "No app name provided."
    app_lower = name.lower().strip()
    exe = _KNOWN_APPS.get(app_lower, app_lower)

    try:
        if exe.startswith("ms-"):
            os.system(f"start {exe}")
        else:
            subprocess.Popen(exe, shell=True)
        time.sleep(1)  # Give the app a moment to start
        return f"Launched {name}."
    except Exception as e:
        return f"Failed to launch {name}: {e}"


def desktop_focus_window(title: str = "") -> str:
    """Bring a window to the foreground by partial title match."""
    if not title:
        return "No window title provided."

    if _PYWINAUTO_AVAILABLE:
        try:
            desktop = Desktop(backend="uia")
            windows = desktop.windows()
            for w in windows:
                if title.lower() in w.window_text().lower():
                    w.set_focus()
                    return f"Focused: {w.window_text()}"
            return f"No window matching '{title}' found."
        except Exception as e:
            return f"Focus error (pywinauto): {e}"

    if _PYAUTOGUI_AVAILABLE:
        try:
            import pygetwindow as gw  # type: ignore[import-untyped]

            wins = gw.getWindowsWithTitle(title)
            if wins:
                wins[0].activate()
                return f"Focused: {wins[0].title}"
            return f"No window matching '{title}' found."
        except Exception as e:
            return f"Focus error (pyautogui): {e}"

    return "No automation library available."


def desktop_type_text(text: str = "") -> str:
    """Type text into the currently focused window."""
    if not text:
        return "No text provided."

    if _PYAUTOGUI_AVAILABLE:
        try:
            # Use pyperclip + Ctrl+V for reliable Unicode support
            import pyperclip

            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            return f"Typed: {text[:80]}{'...' if len(text) > 80 else ''}"
        except ImportError:
            # Fallback to direct typing (ASCII only)
            pyautogui.typewrite(text, interval=0.02)
            return f"Typed: {text[:80]}"
        except Exception as e:
            return f"Type error: {e}"

    return "pyautogui not available for typing."


def desktop_press_key(key: str = "") -> str:
    """
    Press a keyboard key or shortcut.
    Examples: 'enter', 'tab', 'ctrl+c', 'alt+f4', 'escape'
    """
    if not key:
        return "No key specified."

    if not _PYAUTOGUI_AVAILABLE:
        return "pyautogui not available."

    try:
        key_lower = key.lower().strip()
        if "+" in key_lower:
            # Hotkey combo like 'ctrl+c'
            keys = [k.strip() for k in key_lower.split("+")]
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(key_lower)
        return f"Pressed: {key}"
    except Exception as e:
        return f"Key press error: {e}"


def desktop_list_windows() -> str:
    """List all visible windows with their titles."""
    if _PYWINAUTO_AVAILABLE:
        try:
            desktop = Desktop(backend="uia")
            windows = desktop.windows()
            visible = [w.window_text() for w in windows if w.window_text().strip()]
            if not visible:
                return "No visible windows found."
            result = "Open windows:\n" + "\n".join(f"  - {t}" for t in visible[:20])
            if len(visible) > 20:
                result += f"\n  ... and {len(visible) - 20} more"
            return result
        except Exception as e:
            return f"Window listing error: {e}"

    if _PYAUTOGUI_AVAILABLE:
        try:
            import pygetwindow as gw

            titles = [w.title for w in gw.getAllWindows() if w.title.strip()]
            result = "Open windows:\n" + "\n".join(f"  - {t}" for t in titles[:20])
            return result
        except Exception as e:
            return f"Window listing error: {e}"

    return "No automation library available."


def desktop_read_window(title: str = "") -> str:
    """Read UI element text from a window using Windows UI Automation."""
    if not _PYWINAUTO_AVAILABLE:
        return "pywinauto not available for UI reading."

    if not title:
        return "No window title provided."

    try:
        desktop = Desktop(backend="uia")
        windows = desktop.windows()
        target = None
        for w in windows:
            if title.lower() in w.window_text().lower():
                target = w
                break

        if target is None:
            return f"Window '{title}' not found."

        # Read all text-containing elements
        texts = []
        try:
            children = target.descendants()
            for child in children[:50]:  # Limit to prevent slowness
                try:
                    t = child.window_text()
                    if t and t.strip():
                        texts.append(t.strip())
                except Exception:
                    pass
        except Exception:
            pass

        if not texts:
            return f"No readable UI text found in '{title}'."

        result = "\n".join(texts[:30])
        if len(result) > 2000:
            result = result[:2000] + "\n... [truncated]"
        return f"[Window: {target.window_text()}]\n{result}"
    except Exception as e:
        return f"UI read error: {e}"


# ── System Status Tools ────────────────────────────────────────


def get_battery_level() -> dict[str, Any]:
    """Return current battery percentage and power state."""
    try:
        import psutil

        battery = psutil.sensors_battery()
        if battery is None:
            return {"success": False, "message": "Battery not detected (Desktop?)."}

        return {
            "success": True,
            "percent": battery.percent,
            "power_plugged": battery.power_plugged,
            "message": f"Battery is at {battery.percent}% and {'plugged in' if battery.power_plugged else 'discharging'}.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_wifi_status() -> dict[str, Any]:
    """Return current WiFi connection status (Windows specific)."""
    try:
        # Use netsh to get interface information
        output = subprocess.check_output("netsh wlan show interfaces", shell=True).decode()

        ssid = "Unknown"
        signal = "Unknown"
        status = "Disconnected"

        for line in output.split("\n"):
            line = line.strip()
            if "SSID" in line and ":" in line and "BSSID" not in line:
                ssid = line.split(":")[1].strip()
            if "Signal" in line and ":" in line:
                signal = line.split(":")[1].strip()
            if "State" in line and ":" in line:
                status = line.split(":")[1].strip()

        return {
            "success": True,
            "connected": status == "connected",
            "ssid": ssid,
            "signal": signal,
            "message": f"WiFi is {status} to {ssid} with {signal} signal.",
        }
    except Exception:
        return {"success": False, "message": "Could not retrieve WiFi status."}


def get_cpu_ram_status() -> dict[str, Any]:
    """Return current system load."""
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        return {
            "success": True,
            "cpu": cpu,
            "ram": ram,
            "message": f"CPU usage is at {cpu}% and RAM is at {ram}%.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
