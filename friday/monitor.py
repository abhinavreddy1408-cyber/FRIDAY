"""
F.R.I.D.A.Y. Background Monitor Service
========================================
Monitors system state and environment conditions to provide proactive alerts.
Supports time-based reminders and system-level monitoring (CPU, Memory, etc.).
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class BackgroundMonitor:
    """
    Background service that periodically checks system and environment conditions.
    Triggers proactive alerts via the engine's trigger queue.
    """

    def __init__(
        self, push_trigger_callback: Callable[[str], None], get_state_callback: Callable[[], Any]
    ):
        self._push_trigger = push_trigger_callback
        self._get_state = get_state_callback
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        # State for threshold monitoring
        self._cpu_high_start: float | None = None
        self._alerts_sent: dict[str, float] = {}  # condition -> last_alert_time

    def start(self) -> None:
        """Start the background monitor thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._run, daemon=True, name="BackgroundMonitor"
        )
        self._monitor_thread.start()
        logger.info("[Monitor] Background monitor service started.")

    def stop(self) -> None:
        """Stop the background monitor thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("[Monitor] Background monitor service stopped.")

    def _run(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_conditions()
            except Exception as e:
                logger.error(f"[Monitor] Error in check loop: {e}", exc_info=True)

            # Sleep in small increments to remain responsive to stop_event
            for _ in range(30):  # 30 seconds total
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _check_conditions(self) -> None:
        """Evaluate all monitoring conditions."""

        # 1. CPU Usage Monitor
        self._check_cpu_usage()

        # 2. Time/Calendar Monitor (Placeholder for future calendar skill integration)
        self._check_scheduled_tasks()

    def _check_cpu_usage(self) -> None:
        """Monitor CPU usage and alert if it stays high for too long."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking

            # We want to alert if CPU is > 90% for more than 5 minutes
            threshold = 90.0
            duration_needed = 300  # 5 minutes

            if cpu_percent > threshold:
                if self._cpu_high_start is None:
                    self._cpu_high_start = time.time()
                elif time.time() - self._cpu_high_start > duration_needed:  # noqa: SIM102
                    # High CPU alert
                    if self._should_alert("cpu_usage_high"):
                        self._trigger_alert(
                            "Sir, the system CPU usage has been above 90% for over 5 minutes. "
                            "Would you like me to check which processes are responsible?"
                        )
                        self._cpu_high_start = time.time()  # Reset to avoid spam
            else:
                self._cpu_high_start = None

        except Exception as e:
            logger.debug(f"[Monitor] CPU check failed: {e}")

    def _check_scheduled_tasks(self) -> None:
        """Check for upcoming events or tasks (Placeholder)."""
        # This would interface with a calendar skill in a real implementation
        pass

    def _should_alert(self, condition: str, cooldown: int = 3600) -> bool:
        """Check if an alert for this condition should be sent (cooldown logic)."""
        last_time = self._alerts_sent.get(condition, 0)
        if time.time() - last_time > cooldown:
            self._alerts_sent[condition] = time.time()
            return True
        return False

    def _trigger_alert(self, message: str) -> None:
        """Send an alert to the engine."""
        logger.info(f"[Monitor] Triggering proactive alert: {message}")
        self._push_trigger(message)
