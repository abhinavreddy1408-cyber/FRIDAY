"""
F.R.I.D.A.Y. Screen Awareness Module
Provides non-invasive screen monitoring capabilities including:
  - Screenshot capture and region extraction
  - OCR-based text extraction from screen content
  - Error dialog and notification detection
  - Active window tracking
  - Privacy-mode enforcement (disables all capture when active)

All capture operations are opt-in and respect the PRIVACY_MODE config flag.
Screen reading is never performed without explicit engine invocation.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from friday import config

logger = logging.getLogger(__name__)


# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import mss
    import mss.tools

    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False
    logger.warning("[Screen] mss not installed — screenshot capture disabled.")

try:
    import pytesseract
    from PIL import Image

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    logger.warning("[Screen] PIL/pytesseract not installed — OCR disabled.")

try:
    import psutil
    import win32gui
    import win32process

    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False

try:
    import numpy as np  # noqa: F401

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


# ── Data Models ───────────────────────────────────────────────────────────────


class CaptureRegion(Enum):
    """Predefined screen capture regions."""

    FULL_SCREEN = auto()
    TOP_HALF = auto()
    BOTTOM_HALF = auto()
    CENTER = auto()
    TOP_BAR = auto()  # Taskbar / menu bar area
    CUSTOM = auto()


class AlertSeverity(Enum):
    """Severity classification for detected screen alerts."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass(frozen=True)
class ScreenText:
    """
    Result of an OCR extraction pass.

    Attributes:
        text:       Raw extracted text.
        region:     Screen region that was captured.
        timestamp:  Unix timestamp of capture.
        confidence: OCR confidence score (0-100), None if unavailable.
        duration:   Wall-clock time for capture + OCR in seconds.
    """

    text: str
    region: CaptureRegion
    timestamp: float
    confidence: float | None
    duration: float

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def contains(self, pattern: str, case_sensitive: bool = False) -> bool:
        """Check if extracted text contains a pattern (string or regex)."""
        flags = 0 if case_sensitive else re.IGNORECASE
        return bool(re.search(pattern, self.text, flags))


@dataclass(frozen=True)
class WindowInfo:
    """
    Information about the currently active window.

    Attributes:
        title:       Window title string.
        process:     Process name (e.g. 'chrome.exe').
        pid:         Process ID.
        hwnd:        Windows handle (0 on non-Windows).
        timestamp:   Unix timestamp of capture.
    """

    title: str
    process: str
    pid: int
    hwnd: int
    timestamp: float

    @property
    def app_name(self) -> str:
        """Return a clean application name without extension."""
        return self.process.lower().replace(".exe", "")


@dataclass(frozen=True)
class ScreenAlert:
    """
    A detected error or notification on screen.

    Attributes:
        severity:  Classification of the alert.
        title:     Alert dialog title or heading.
        body:      Full text of the alert.
        source:    Window or application where detected.
        timestamp: Unix timestamp of detection.
    """

    severity: AlertSeverity
    title: str
    body: str
    source: str
    timestamp: float

    def to_speech(self) -> str:
        """Format the alert as a natural TTS-ready string."""
        sev = self.severity.name.lower()
        src = f" in {self.source}" if self.source else ""
        return f"{sev} alert{src}: {self.title}. {self.body}".strip()


# ── Error Pattern Registry ────────────────────────────────────────────────────


@dataclass
class _ErrorPattern:
    pattern: re.Pattern
    severity: AlertSeverity
    title: str


_ERROR_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        pattern=re.compile(
            r"\b(critical error|fatal error|system failure|blue screen|bsod)\b",
            re.IGNORECASE,
        ),
        severity=AlertSeverity.CRITICAL,
        title="Critical System Error",
    ),
    _ErrorPattern(
        pattern=re.compile(
            r"\b(error|exception|crash|not responding|stopped working|failed)\b",
            re.IGNORECASE,
        ),
        severity=AlertSeverity.ERROR,
        title="Application Error",
    ),
    _ErrorPattern(
        pattern=re.compile(
            r"\b(warning|disk (full|almost full)|low (memory|storage|battery))\b",
            re.IGNORECASE,
        ),
        severity=AlertSeverity.WARNING,
        title="System Warning",
    ),
    _ErrorPattern(
        pattern=re.compile(
            r"\b(update available|restart required|action required)\b",
            re.IGNORECASE,
        ),
        severity=AlertSeverity.INFO,
        title="System Notification",
    ),
]


# ── Screen Capturer ───────────────────────────────────────────────────────────


class _ScreenCapturer:
    """
    Thin wrapper around mss for screenshot capture.
    Converts captures to PIL Images for OCR processing.
    """

    def capture_full(self) -> Any | None:
        """Capture the entire primary monitor. Returns PIL Image or None."""
        if not _MSS_AVAILABLE:
            return None
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                raw = sct.grab(monitor)
                if _OCR_AVAILABLE:
                    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                return raw
        except Exception as exc:
            logger.debug("[Screen] Full capture failed: %s", exc)
            return None

    def capture_region(
        self,
        region: CaptureRegion,
        custom_bounds: dict[str, int] | None = None,
    ) -> Any | None:
        """
        Capture a specific screen region.

        Args:
            region:        Predefined region enum.
            custom_bounds: Required when region is CUSTOM.
                           Dict with keys: top, left, width, height.

        Returns:
            PIL Image if successful, None otherwise.
        """
        if not _MSS_AVAILABLE:
            return None

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                w, h = monitor["width"], monitor["height"]

                bounds = self._get_bounds(region, w, h, custom_bounds)
                if not bounds:
                    return None

                raw = sct.grab(bounds)
                if _OCR_AVAILABLE:
                    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                return raw

        except Exception as exc:
            logger.debug("[Screen] Region capture failed: %s", exc)
            return None

    @staticmethod
    def _get_bounds(
        region: CaptureRegion,
        w: int,
        h: int,
        custom: dict[str, int] | None,
    ) -> dict[str, int] | None:
        """Map CaptureRegion enum to mss-compatible bounds dict."""
        if region == CaptureRegion.FULL_SCREEN:
            return {"top": 0, "left": 0, "width": w, "height": h}
        if region == CaptureRegion.TOP_HALF:
            return {"top": 0, "left": 0, "width": w, "height": h // 2}
        if region == CaptureRegion.BOTTOM_HALF:
            return {"top": h // 2, "left": 0, "width": w, "height": h // 2}
        if region == CaptureRegion.CENTER:
            margin = 4
            return {
                "top": h // margin,
                "left": w // margin,
                "width": w // 2,
                "height": h // 2,
            }
        if region == CaptureRegion.TOP_BAR:
            return {"top": 0, "left": 0, "width": w, "height": 80}
        if region == CaptureRegion.CUSTOM:
            return custom
        return None


# ── OCR Engine ────────────────────────────────────────────────────────────────


class _OCREngine:
    """
    Wrapper around pytesseract for text extraction from PIL Images.
    Applies preprocessing to improve accuracy on UI elements.
    """

    # Tesseract config: page segmentation mode 6 (uniform block of text)
    _TESS_CONFIG = "--psm 6 --oem 3"

    def extract(self, image: Any) -> tuple[str, float | None]:
        """
        Extract text and confidence from a PIL Image.

        Args:
            image: PIL Image object.

        Returns:
            Tuple of (extracted_text, mean_confidence).
            Confidence is None if detailed data unavailable.
        """
        if not _OCR_AVAILABLE or image is None:
            return "", None

        try:
            preprocessed = self._preprocess(image)
            data = pytesseract.image_to_data(
                preprocessed,
                config=self._TESS_CONFIG,
                output_type=pytesseract.Output.DICT,
            )

            words = []
            confidences = []

            for i, word in enumerate(data["text"]):
                if word.strip():
                    words.append(word)
                    conf = int(data["conf"][i])
                    if conf >= 0:
                        confidences.append(conf)

            text = " ".join(words)
            mean_conf = sum(confidences) / len(confidences) if confidences else None
            return text, mean_conf

        except Exception as exc:
            logger.debug("[Screen/OCR] Extraction failed: %s", exc)
            return "", None

    @staticmethod
    def _preprocess(image: Any) -> Any:
        """
        Apply light preprocessing to improve OCR accuracy.
        Converts to grayscale and scales up small images.
        """
        if not _OCR_AVAILABLE:
            return image

        try:
            img = image.convert("L")  # Grayscale

            # Upscale small images — Tesseract accuracy drops below 300 DPI
            w, h = img.size
            if w < 1000 or h < 600:
                scale = max(1000 / w, 600 / h)
                img = img.resize(
                    (int(w * scale), int(h * scale)),
                    Image.LANCZOS,
                )
            return img
        except Exception:
            return image


# ── Window Tracker ────────────────────────────────────────────────────────────


class _WindowTracker:
    """
    Tracks the currently active foreground window.
    Windows-only via win32gui; returns a stub on other platforms.
    """

    def get_active_window(self) -> WindowInfo | None:
        """
        Return information about the currently focused window.

        Returns:
            WindowInfo if available, None on failure or non-Windows.
        """
        if not _WIN32_AVAILABLE:
            return None

        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or ""
            _, pid = win32process.GetWindowThreadProcessId(hwnd)

            try:
                proc = psutil.Process(pid)
                process = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process = "unknown"

            return WindowInfo(
                title=title,
                process=process,
                pid=pid,
                hwnd=hwnd,
                timestamp=time.time(),
            )
        except Exception as exc:
            logger.debug("[Screen] Window tracking failed: %s", exc)
            return None


# ── Alert Detector ────────────────────────────────────────────────────────────


class _AlertDetector:
    """
    Scans extracted screen text for error dialogs and system notifications.
    """

    def scan(
        self,
        screen_text: ScreenText,
        source: str = "",
    ) -> list[ScreenAlert]:
        """
        Identify alerts in extracted screen text.

        Args:
            screen_text: OCR result to scan.
            source:      Application/window name for context.

        Returns:
            List of ScreenAlert objects ordered by severity descending.
        """
        if screen_text.is_empty:
            return []

        alerts: list[ScreenAlert] = []
        text = screen_text.text

        for pattern in _ERROR_PATTERNS:
            if pattern.pattern.search(text):
                body = self._extract_context(text, pattern.pattern)
                alerts.append(
                    ScreenAlert(
                        severity=pattern.severity,
                        title=pattern.title,
                        body=body,
                        source=source,
                        timestamp=time.time(),
                    )
                )

        # Sort by severity (CRITICAL first)
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
        }
        alerts.sort(key=lambda a: severity_order[a.severity])
        return alerts

    @staticmethod
    def _extract_context(text: str, pattern: re.Pattern) -> str:
        """
        Extract a sentence-level context window around the first pattern match.

        Args:
            text:    Full OCR text.
            pattern: The compiled pattern that matched.

        Returns:
            Extracted context string (max ~200 characters).
        """
        match = pattern.search(text)
        if not match:
            return text[:200]

        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + 140)
        return text[start:end].strip()


# ── Screen Monitor ────────────────────────────────────────────────────────────


class ScreenMonitor:
    """
    Public interface for all screen awareness operations.

    Combines capture, OCR, window tracking, and alert detection behind
    a single, privacy-respecting API. All operations short-circuit
    immediately when PRIVACY_MODE is enabled.

    Background polling is optional and controlled by start_monitoring()
    and stop_monitoring(). Poll results are surfaced via a callback.

    Usage:
        monitor = ScreenMonitor()

        text = monitor.read_screen()
        if text and text.contains("error"):
            friday.speak("There appears to be an error on screen.")

        monitor.start_monitoring(on_alert=handle_alert)
    """

    def __init__(self) -> None:
        self._capturer = _ScreenCapturer()
        self._ocr = _OCREngine()
        self._tracker = _WindowTracker()
        self._detector = _AlertDetector()

        self._poll_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._monitoring: bool = False

        # Cache last known window to detect transitions
        self._last_window: WindowInfo | None = None

    # ── One-Shot Operations ───────────────────────────────────────────────────

    def read_screen(
        self,
        region: CaptureRegion = CaptureRegion.FULL_SCREEN,
        custom_bounds: dict[str, int] | None = None,
    ) -> ScreenText | None:
        """
        Capture a screen region and extract text via OCR.

        Args:
            region:        Which region to capture.
            custom_bounds: Required when region is CUSTOM.

        Returns:
            ScreenText result, or None if privacy mode is active
            or capture/OCR is unavailable.
        """
        if config.security.privacy_mode:
            logger.info("[Screen] Capture blocked — privacy mode active.")
            return None

        if not _MSS_AVAILABLE or not _OCR_AVAILABLE:
            return None

        start = time.perf_counter()
        image = self._capturer.capture_region(region, custom_bounds)

        if image is None:
            return None

        text, confidence = self._ocr.extract(image)
        duration = time.perf_counter() - start

        result = ScreenText(
            text=text,
            region=region,
            timestamp=time.time(),
            confidence=confidence,
            duration=duration,
        )

        logger.debug(
            "[Screen] OCR complete: %d words, %.0f%% confidence, %.2fs",
            result.word_count,
            confidence or 0,
            duration,
        )
        return result

    def get_active_window(self) -> WindowInfo | None:
        """
        Return information about the currently focused window.

        Returns:
            WindowInfo or None if privacy mode is active / unavailable.
        """
        if config.security.privacy_mode:
            return None
        return self._tracker.get_active_window()

    def scan_for_alerts(
        self,
        region: CaptureRegion = CaptureRegion.FULL_SCREEN,
    ) -> list[ScreenAlert]:
        """
        Capture a screen region and return any detected alerts.

        Args:
            region: Which region to scan.

        Returns:
            List of ScreenAlert objects (may be empty).
        """
        if config.security.privacy_mode:
            return []

        screen_text = self.read_screen(region)
        if not screen_text:
            return []

        window = self.get_active_window()
        source = window.app_name if window else ""
        return self._detector.scan(screen_text, source)

    def save_screenshot(self, path: str) -> bool:
        """
        Capture and save the full screen to disk.

        Args:
            path: File path for the saved image (PNG).

        Returns:
            True if saved successfully.
        """
        if config.security.privacy_mode:
            logger.info("[Screen] Screenshot blocked — privacy mode active.")
            return False

        if not _MSS_AVAILABLE:
            return False

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                raw = sct.grab(monitor)
                mss.tools.to_png(raw.rgb, raw.size, output=path)
            logger.info("[Screen] Screenshot saved to '%s'.", path)
            return True
        except Exception as exc:
            logger.error("[Screen] Save failed: %s", exc)
            return False

    # ── Background Monitoring ─────────────────────────────────────────────────

    def start_monitoring(
        self,
        on_alert: callable | None = None,
        on_window_change: callable | None = None,
        interval: int = config.system.screen_poll_interval,
    ) -> bool:
        """
        Start background screen polling on a daemon thread.

        Args:
            on_alert:          Callback invoked with a ScreenAlert when detected.
                               Signature: (alert: ScreenAlert) -> None
            on_window_change:  Callback invoked when the active window changes.
                               Signature: (window: WindowInfo) -> None
            interval:          Seconds between poll cycles.

        Returns:
            True if monitoring started, False if already running or
            privacy mode is active.
        """
        if config.security.privacy_mode:
            logger.info("[Screen] Monitoring blocked — privacy mode active.")
            return False

        if not config.system.monitor_screen:
            logger.debug("[Screen] Monitoring disabled in config.")
            return False

        if self._monitoring:
            logger.debug("[Screen] Already monitoring.")
            return True

        if not _MSS_AVAILABLE or not _OCR_AVAILABLE:
            logger.warning("[Screen] Cannot monitor — capture/OCR unavailable.")
            return False

        self._stop_event.clear()
        self._monitoring = True

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(on_alert, on_window_change, interval),
            daemon=True,
            name="screen-monitor",
        )
        self._poll_thread.start()
        logger.info("[Screen] Background monitoring started (interval: %ds).", interval)
        return True

    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread gracefully."""
        if not self._monitoring:
            return

        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
            self._poll_thread = None

        self._monitoring = False
        logger.info("[Screen] Background monitoring stopped.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_monitoring(self) -> bool:
        return self._monitoring

    @property
    def is_available(self) -> bool:
        """True if both capture and OCR are operational."""
        return _MSS_AVAILABLE and _OCR_AVAILABLE and not config.security.privacy_mode

    @property
    def capabilities(self) -> dict[str, bool]:
        """Report available capabilities for diagnostics."""
        return {
            "capture": _MSS_AVAILABLE,
            "ocr": _OCR_AVAILABLE,
            "window_track": _WIN32_AVAILABLE,
            "privacy_mode": config.security.privacy_mode,
            "monitoring_on": self._monitoring,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _poll_loop(
        self,
        on_alert: callable | None,
        on_window_change: callable | None,
        interval: int,
    ) -> None:
        """
        Background polling loop. Runs until stop_monitoring() is called.
        Catches all exceptions to ensure the thread never silently dies.
        """
        logger.debug("[Screen] Poll loop started.")

        while not self._stop_event.is_set():
            try:
                self._poll_cycle(on_alert, on_window_change)
            except Exception:
                logger.exception("[Screen] Unhandled error in poll loop.")

            self._stop_event.wait(timeout=interval)

        logger.debug("[Screen] Poll loop exited.")

    def _poll_cycle(
        self,
        on_alert: callable | None,
        on_window_change: callable | None,
    ) -> None:
        """
        Execute one monitoring cycle: check window + scan for alerts.
        """
        # Window change detection
        if on_window_change and _WIN32_AVAILABLE:
            current = self._tracker.get_active_window()
            if current and (self._last_window is None or current.title != self._last_window.title):
                self._last_window = current
                try:
                    on_window_change(current)
                except Exception:
                    logger.debug("[Screen] on_window_change callback raised.", exc_info=True)

        # Alert scanning
        if on_alert:
            alerts = self.scan_for_alerts(region=CaptureRegion.CENTER)
            for alert in alerts:
                logger.info(
                    "[Screen] Alert detected [%s]: %s",
                    alert.severity.name,
                    alert.title,
                )
                try:
                    on_alert(alert)
                except Exception:
                    logger.debug("[Screen] on_alert callback raised.", exc_info=True)


class ScreenManager:
    """Wraps ScreenMonitor and adds caching/throttling and privacy controls."""

    def __init__(self) -> None:
        self.monitor = ScreenMonitor()
        self._privacy_mode = False
        self._last_ocr_time = 0.0
        self._last_result = None

    @property
    def privacy_mode(self) -> bool:
        return self._privacy_mode or config.security.privacy_mode

    @privacy_mode.setter
    def privacy_mode(self, val: bool) -> None:
        self._privacy_mode = val
        object.__setattr__(config.security, "privacy_mode", val)

    def detect_errors_on_screen(self, force_ocr: bool = False) -> dict[str, Any]:
        if self.privacy_mode:
            return {"blocked": True, "errors_found": False}

        now = time.time()
        # Cache for 2 seconds to avoid OCR spam
        if not force_ocr and self._last_result and (now - self._last_ocr_time < 2.0):
            return {
                "errors_found": self._last_result.get("errors_found", False),
                "alerts": self._last_result.get("alerts", []),
                "from_cache": True,
                "ocr_ran": False,
            }

        alerts = self.monitor.scan_for_alerts()
        errors_found = any(
            a.severity in (AlertSeverity.ERROR, AlertSeverity.CRITICAL) for a in alerts
        )

        result = {
            "errors_found": errors_found,
            "alerts": [a.to_speech() for a in alerts],
            "from_cache": False,
            "ocr_ran": True,
        }
        self._last_result = result
        self._last_ocr_time = now
        return result


_global_manager = ScreenManager()


def detect_errors_on_screen(force_ocr: bool = False) -> dict[str, Any]:
    return _global_manager.detect_errors_on_screen(force_ocr=force_ocr)


def read_screen_for_context() -> dict[str, Any]:
    if config.security.privacy_mode or _global_manager.privacy_mode:
        return {"blocked": True, "summary": "", "active_window_title": ""}

    # Active window
    win = _global_manager.monitor.get_active_window()
    win_title = win.title if win else "Unknown Window"

    # Screen text
    st = _global_manager.monitor.read_screen()
    summary = st.text[:200] if st else ""

    return {"summary": summary, "active_window_title": win_title}


def verify_url_contains(substring: str) -> dict:
    """Check if the active window URL/title contains the given substring."""
    win = _global_manager.monitor.get_active_window() if hasattr(_global_manager, "monitor") else None
    title = win.title if win else ""
    return {"found": substring.lower() in title.lower(), "current_url": title}


def verify_screen_contains(text: str) -> dict:
    """Check if the screen OCR text contains the given substring."""
    try:
        result = _global_manager.detect_errors_on_screen(force_ocr=True)
        ocr_text = result.get("ocr_text", "") or ""
        if not ocr_text:
            st = _global_manager.monitor.read_screen() if hasattr(_global_manager, "monitor") else None
            ocr_text = st.text if st else ""
        return {"found": text.lower() in ocr_text.lower()}
    except Exception:
        return {"found": False}

