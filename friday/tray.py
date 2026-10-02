import logging
import threading
import time
from typing import Any

from PIL import Image, ImageDraw
import pystray
from pystray import MenuItem as item

from friday import config
from friday.engine import FridayEngine, EngineState, StateChangeEvent

logger = logging.getLogger(__name__)


def hex_to_rgba(hex_str: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert hex string (e.g. #10B981) to RGBA tuple."""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (r, g, b, alpha)


def create_orb_image(hex_color: str) -> Image.Image:
    """Create a 64x64 glowing glassy orb image in Pillow."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer glow (semi-transparent)
    draw.ellipse([4, 4, 60, 60], fill=hex_to_rgba(hex_color, 60))
    # Inner orb (solid)
    draw.ellipse([12, 12, 52, 52], fill=hex_to_rgba(hex_color, 255))
    # Glossy highlight
    draw.ellipse([18, 18, 34, 34], fill=(255, 255, 255, 120))

    # Draw 'F' in white in the center
    # Vertical line: x=24 to 28, y=20 to 44
    draw.rectangle([24, 20, 28, 44], fill=(255, 255, 255, 255))
    # Top horizontal bar: x=28 to 38, y=20 to 24
    draw.rectangle([28, 20, 38, 24], fill=(255, 255, 255, 255))
    # Middle horizontal bar: x=28 to 34, y=30 to 34
    draw.rectangle([28, 30, 34, 34], fill=(255, 255, 255, 255))

    return img


class FridayTray:
    """
    System Tray UI coordinator for F.R.I.D.A.Y.
    Displays a small glassy orb icon that changes color based on engine state.
    Provides menu items for showing status summary, pausing, and exiting.
    """

    def __init__(self) -> None:
        self._engine: FridayEngine | None = None
        self._paused = False

        # Start with a gray color (offline/booting)
        initial_img = create_orb_image("#6B7280")

        # Build the dynamic menu
        menu = pystray.Menu(
            item("Status", self._show_status),
            item(self._get_pause_label, self._toggle_pause),
            item("Stop", self._stop_action),
        )

        self._icon = pystray.Icon(
            name="friday_assistant",
            title="F.R.I.D.A.Y. Assistant",
            icon=initial_img,
            menu=menu,
        )

    def run(self) -> None:
        """Run the system tray event loop. Blocks the main thread."""
        logger.info("[Tray] Launching system tray.")
        self._icon.run(setup=self._on_tray_ready)

    def stop(self) -> None:
        """Gracefully stop the tray and the engine."""
        logger.info("[Tray] Stopping system tray.")
        if self._engine:
            try:
                self._engine.stop()
            except Exception as exc:
                logger.error("[Tray] Error stopping engine: %s", exc)
        self._icon.stop()

    def _on_tray_ready(self, icon: pystray.Icon) -> None:
        """Invoked by pystray inside a background thread once active."""
        icon.visible = True
        try:
            logger.info("[Tray] Initialising Friday Engine...")
            self._engine = FridayEngine(
                on_state_change=self._on_state_change,
                on_notify=self._on_notify,
            )

            if not self._engine.setup():
                logger.critical("[Tray] Engine setup failed.")
                self._icon.notify("Engine setup failed. Check logs.", "F.R.I.D.A.Y. Error")
                self._on_state_change(StateChangeEvent(EngineState.OFFLINE, EngineState.ERROR))
                return

            logger.info("[Tray] Starting Friday Engine...")
            self._engine.start()
            self._icon.notify("Ready and listening.", "F.R.I.D.A.Y. Active")

        except Exception as exc:
            logger.critical("[Tray] Exception during engine boot: %s", exc, exc_info=True)
            self._icon.notify(f"Boot exception: {exc}", "F.R.I.D.A.Y. Error")
            self._on_state_change(StateChangeEvent(EngineState.OFFLINE, EngineState.ERROR))

    def _on_state_change(self, event: StateChangeEvent) -> None:
        """Fires whenever the engine state transitions."""
        state = event.current
        if state in (EngineState.IDLE, EngineState.CONVERSATION):
            color = "#10B981"  # green
        elif state == EngineState.LISTENING:
            color = "#F59E0B"  # yellow
        elif state in (
            EngineState.PROCESSING,
            EngineState.SPEAKING,
            EngineState.BOOTING,
            EngineState.PROACTIVE_ALERT,
        ):
            color = "#06B6D4"  # blue
        elif state == EngineState.ERROR:
            color = "#EF4444"  # red
        else:
            color = "#6B7280"  # gray

        self._icon.icon = create_orb_image(color)

    def _on_notify(self, title: str, message: str, metadata: Any = None) -> None:
        """Triggered by the engine to show desktop notifications."""
        self._icon.notify(message, title)

    def _get_pause_label(self, item: Any) -> str:
        return "Resume" if self._paused else "Pause"

    def _toggle_pause(self, icon: pystray.Icon, item: Any) -> None:
        if not self._engine:
            return

        if self._paused:
            self._engine.resume()
            self._paused = False
            self._icon.notify("Listening for wake word.", "F.R.I.D.A.Y. Resumed")
        else:
            self._engine.pause()
            self._paused = True
            self._icon.notify("Microphone suspended.", "F.R.I.D.A.Y. Paused")

    def _show_status(self, icon: pystray.Icon, item: Any) -> None:
        if not self._engine:
            self._icon.notify("Engine is not booted.", "Status")
            return

        stats = self._engine.session_stats
        msg = (
            f"State: {stats.get('state')}\n"
            f"Uptime: {stats.get('uptime_seconds')}s\n"
            f"Commands Handled: {stats.get('commands_handled')}\n"
            f"Active Goals: {stats.get('active_goals')}\n"
            f"Errors: {stats.get('errors_total')}"
        )
        self._icon.notify(msg, "F.R.I.D.A.Y. Status")

    def _stop_action(self, icon: pystray.Icon, item: Any) -> None:
        self.stop()
