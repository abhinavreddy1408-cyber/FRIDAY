"""
F.R.I.D.A.Y. â€” Desktop Tools
==============================
Windows application and UI control.

Primary:  pywinauto (UIA backend, most reliable)
Fallback: pyautogui (pixel-based, last resort)

Risk levels:
  LOW:    list_windows, read_window, get_clipboard,
          focus_window, get_battery
  MEDIUM: open_app, type_text, press_key, set_clipboard
  HIGH:   close_window, mouse_click

All functions return {"success": bool, "message": str, ...}
"""

import logging
import time

logger = logging.getLogger(__name__)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _result(success: bool, message: str = "", **kwargs) -> dict:
    return {"success": success, "message": message, **kwargs}


def _get_pyautogui():
    """Import pyautogui with helpful error on failure."""
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        return pyautogui
    except ImportError:
        raise ImportError("pyautogui not installed. Run: pip install pyautogui")  # noqa: B904


def _get_pywinauto():
    """Import pywinauto with helpful error on failure."""
    try:
        from pywinauto import Application, Desktop

        return Application, Desktop
    except ImportError:
        raise ImportError("pywinauto not installed. Run: pip install pywinauto")  # noqa: B904


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# WINDOW MANAGEMENT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def desktop_list_windows() -> dict:
    """
    List all visible windows with their titles.
    Risk: LOW
    """
    try:
        _, Desktop = _get_pywinauto()
        desktop = Desktop(backend="uia")
        windows = []

        for win in desktop.windows():
            try:
                title = win.window_text().strip()
                if title and len(title) > 1:
                    windows.append(title)
            except Exception:
                continue

        windows = sorted(set(windows))
        return _result(
            True,
            f"Found {len(windows)} windows.",
            windows=windows,
            count=len(windows),
        )

    except ImportError as e:
        return _result(False, str(e))
    except Exception as e:
        return _result(False, f"Could not list windows: {e}")


def desktop_focus_window(title: str) -> dict:
    """
    Bring a window to the foreground by title.
    Risk: LOW

    Args:
        title: Window title or partial title
    """
    try:
        _, Desktop = _get_pywinauto()
        desktop = Desktop(backend="uia")

        for win in desktop.windows():
            try:
                win_title = win.window_text()
                if title.lower() in win_title.lower():
                    win.set_focus()
                    time.sleep(0.3)
                    return _result(
                        True,
                        f"Focused '{win_title}', Sir.",
                        window=win_title,
                    )
            except Exception:
                continue

        return _result(False, f"No window found matching '{title}', Sir.")

    except ImportError as e:
        return _result(False, str(e))
    except Exception as e:
        return _result(False, f"Focus window failed: {e}")


def desktop_close_window(title: str) -> dict:
    """
    Close a window by title.
    Risk: HIGH â€” requires confirmation

    Args:
        title: Window title or partial title
    """
    try:
        _, Desktop = _get_pywinauto()
        desktop = Desktop(backend="uia")

        for win in desktop.windows():
            try:
                win_title = win.window_text()
                if title.lower() in win_title.lower():
                    win.close()
                    time.sleep(0.3)
                    return _result(
                        True,
                        f"Closed '{win_title}', Sir.",
                        window=win_title,
                    )
            except Exception:
                continue

        return _result(False, f"No window found matching '{title}', Sir.")

    except ImportError as e:
        return _result(False, str(e))
    except Exception as e:
        return _result(False, f"Close window failed: {e}")


def desktop_read_window(title: str) -> dict:
    """
    Read UI elements and text from a window.
    Risk: LOW

    Args:
        title: Window title or partial title
    """
    try:
        _Application, Desktop = _get_pywinauto()
        desktop = Desktop(backend="uia")

        for win in desktop.windows():
            try:
                win_title = win.window_text()
                if title.lower() in win_title.lower():
                    elements = []
                    for ctrl in win.descendants():
                        try:
                            text = ctrl.window_text().strip()
                            if text and len(text) > 1:
                                elements.append(text)
                        except Exception:
                            continue

                    elements = list(dict.fromkeys(elements))[:50]
                    return _result(
                        True,
                        f"Read {len(elements)} elements from '{win_title}'.",
                        window=win_title,
                        elements=elements,
                    )
            except Exception:
                continue

        return _result(False, f"No window found matching '{title}', Sir.")

    except ImportError as e:
        return _result(False, str(e))
    except Exception as e:
        return _result(False, f"Read window failed: {e}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# KEYBOARD INPUT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def desktop_type_text(text: str) -> dict:
    """
    Type text into the currently focused window.
    Risk: MEDIUM

    Args:
        text: Text to type
    """
    try:
        pag = _get_pyautogui()
        time.sleep(0.2)
        pag.write(text, interval=0.05)
        return _result(
            True,
            "Typed text successfully, Sir.",
            characters=len(text),
        )
    except ImportError as e:
        return _result(False, str(e))
    except Exception as e:
        return _result(False, f"Type text failed: {e}")


def desktop_press_key(key: str) -> dict:
    """
    Press a keyboard key or shortcut.
    Risk: MEDIUM

    Args:
        key: Key name e.g. "enter", "ctrl+c", "alt+f4"
            Supports combinations with + separator.
    """
    try:
        pag = _get_pyautogui()

        if "+" in key:
            # Handle key combinations e.g. "ctrl+c"
            keys = [k.strip() for k in key.split("+")]
            pag.hotkey(*keys)
        else:
            pag.press(key.strip())

        return _result(
            True,
            f"Pressed '{key}', Sir.",
            key=key,
        )
    except ImportError as e:
        return _result(False, str(e))
    except Exception as e:
        return _result(False, f"Key press failed: {e}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CLIPBOARD
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def desktop_get_clipboard() -> dict:
    """
    Get current clipboard text content.
    Risk: LOW
    """
    try:
        import pyperclip

        text = pyperclip.paste()
        return _result(
            True,
            f"Clipboard: {text[:100]}" if text else "Clipboard is empty.",
            text=text,
            length=len(text),
        )
    except ImportError:
        return _result(False, "pyperclip not installed. Run: pip install pyperclip")
    except Exception as e:
        return _result(False, f"Clipboard read failed: {e}")


def desktop_set_clipboard(text: str) -> dict:
    """
    Set clipboard text content.
    Risk: MEDIUM

    Args:
        text: Text to copy to clipboard
    """
    try:
        import pyperclip

        pyperclip.copy(text)
        return _result(
            True,
            "Copied to clipboard, Sir.",
            length=len(text),
        )
    except ImportError:
        return _result(False, "pyperclip not installed. Run: pip install pyperclip")
    except Exception as e:
        return _result(False, f"Clipboard write failed: {e}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MOUSE CONTROL
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def desktop_mouse_click(x: int, y: int) -> dict:
    """
    Click at specific screen coordinates.
    Risk: HIGH â€” requires confirmation

    Args:
        x: Horizontal pixel coordinate
        y: Vertical pixel coordinate
    """
    try:
        pag = _get_pyautogui()

        # Safety: get screen size
        screen_w, screen_h = pag.size()
        if not (0 <= x <= screen_w and 0 <= y <= screen_h):
            return _result(
                False, f"Coordinates ({x},{y}) are outside screen ({screen_w}x{screen_h}), Sir."
            )

        pag.click(x, y)
        return _result(
            True,
            f"Clicked at ({x}, {y}), Sir.",
            x=x,
            y=y,
        )
    except ImportError as e:
        return _result(False, str(e))
    except Exception as e:
        return _result(False, f"Mouse click failed: {e}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# APPLICATION CONTROL
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def desktop_open_app(name: str) -> dict:
    """
    Launch an application by name.
    Delegates to local_tools.open_app for path resolution.
    Risk: MEDIUM

    Args:
        name: Application name e.g. "chrome", "notepad"
    """
    import subprocess
    try:
        subprocess.Popen(['start', name], shell=True)
        return _result(True, f'Opened {name}')
    except Exception as e:
        return _result(False, f'Open app failed: {e}')


def get_battery_level() -> dict:
    """
    Get battery level and charging status.
    Delegates to local_tools.battery_status.
    Risk: LOW
    """
    from friday.skills.system_skill import battery_status

    return battery_status()


def get_wifi_status() -> dict:
    """
    Get WiFi connection status.
    Delegates to local_tools.wifi_status.
    Risk: LOW
    """
    from friday.skills.system_skill import wifi_status

    return wifi_status()


def get_cpu_ram_status() -> dict:
    """
    Get CPU and RAM usage.
    Delegates to local_tools.cpu_ram_status.
    Risk: LOW
    """
    from friday.skills.system_skill import cpu_ram_status

    return cpu_ram_status()


def running_apps() -> dict:
    """
    List running applications.
    Delegates to local_tools.running_apps.
    Risk: LOW
    """
    from friday.skills.system_skill import running_apps as _running_apps

    return _running_apps()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STANDALONE TEST
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    print("=" * 60)
    print("  F.R.I.D.A.Y. Desktop Tools Test")
    print("=" * 60)

    tests = [
        ("list_windows", desktop_list_windows, {}),
        ("get_clipboard", desktop_get_clipboard, {}),
        ("get_battery", get_battery_level, {}),
        ("get_wifi", get_wifi_status, {}),
        ("get_cpu_ram", get_cpu_ram_status, {}),
        ("running_apps", running_apps, {}),
    ]

    passed = failed = 0
    for name, func, kwargs in tests:
        try:
            result = func(**kwargs)
            ok = result.get("success", False)
            msg = result.get("message", "")
            icon = "âœ…" if ok else "âš ï¸ "
            print(f"\n{icon} {name}")
            print(f"   {msg}")
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nâŒ {name}: CRASHED â€” {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

