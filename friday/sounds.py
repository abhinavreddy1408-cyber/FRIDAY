"""
F.R.I.D.A.Y.  —  System Sounds Module
─────────────────────────────────────────────────────────────────────────────
Handles audio feedback chimes and status sounds via pygame.mixer.

Fixes over previous version
  • ensure_init() double-checked locking was broken — the inner check
    ``if self._initialized: return True`` was already inside the lock so
    the outer check was the only guard needed; removed the redundant inner.
  • _get_sound() was not thread-safe — two threads could both miss the
    cache and both try to load the same asset simultaneously.
  • f-strings used in logger calls — replaced with % formatting so the
    string is never built when the log level is suppressed.
  • play() spawned a new daemon thread per sound even for non-blocking
    calls — a rapid sequence of events would spawn dozens of threads.
    Now uses pygame channels directly; non-blocking means fire-and-forget
    on a channel, not a new thread.
  • No volume control — added per-event and global volume.
  • No preload — first playback always has a latency spike from disk I/O.
    Added preload() to warm the cache at startup.
  • No stop_all() — added to halt all active channels (e.g. on shutdown).
  • Asset path hardcoded as relative string — replaced with Path(__file__)
    so the module works regardless of the working directory.
  • pygame.mixer.quit() never called — added teardown() for clean shutdown.
"""

from __future__ import annotations

import logging
import os
import threading
from enum import Enum, auto
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False
    logging.getLogger(__name__).warning("[Sounds] pygame not installed — system sounds disabled.")

from friday.logger import get_logger

logger = get_logger(__name__)

# ── Asset resolution ──────────────────────────────────────────────────────────

# Absolute path to the assets directory — works regardless of cwd.
_ASSETS_DIR: Path = Path(__file__).parent / "assets"

# ── Sound events ──────────────────────────────────────────────────────────────


class SoundEvent(Enum):
    """Canonical system events that trigger audio feedback."""

    STARTUP = auto()
    SHUTDOWN = auto()
    WAKE = auto()  # wake word confirmed
    LISTEN = auto()  # listening for command
    PROCESSING = auto()  # command received, thinking
    SUCCESS = auto()  # task completed
    ERROR = auto()  # task failed or fatal error
    ALERT = auto()  # non-fatal warning / attention needed


# Map each event to its preferred asset filename.
_EVENT_ASSETS: dict[SoundEvent, str] = {
    SoundEvent.STARTUP: "startup.wav",
    SoundEvent.SHUTDOWN: "shutdown.wav",
    SoundEvent.WAKE: "wake.wav",
    SoundEvent.LISTEN: "listen.wav",
    SoundEvent.PROCESSING: "processing.wav",
    SoundEvent.SUCCESS: "success.wav",
    SoundEvent.ERROR: "error.wav",
    SoundEvent.ALERT: "alert.wav",
}

# Used when the event-specific file is missing.
_FALLBACK_ASSET: str = "chime.wav"

# Default playback volume (0.0 - 1.0).
_DEFAULT_VOLUME: float = 0.8

# ── Sound controller ──────────────────────────────────────────────────────────


class SoundController:
    """
    Manages mixer lifecycle, sample caching, volume, and playback.

    Thread safety
        All public methods acquire self._lock before touching shared state.
        Playback itself is non-blocking (pygame channels); no extra threads
        are spawned — pygame handles mixing internally.
    """

    def __init__(self, volume: float = _DEFAULT_VOLUME) -> None:
        self._volume: float = max(0.0, min(1.0, volume))
        self._initialised: bool = False
        self._lock = threading.Lock()

        # Cache: event → loaded Sound object (or None if unavailable)
        self._cache: dict[SoundEvent, pygame.mixer.Sound | None] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise the pygame mixer.

        Safe to call multiple times — subsequent calls are no-ops.

        Returns:
            True if the mixer is ready, False if pygame is unavailable or
            initialisation failed.
        """
        if not _PYGAME_AVAILABLE:
            return False

        with self._lock:
            if self._initialised:
                return True
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                self._initialised = True
                logger.debug("[Sounds] Mixer initialised.")
                return True
            except Exception as exc:
                logger.error("[Sounds] Mixer init failed: %s", exc)
                return False

    def teardown(self) -> None:
        """Release the mixer and clear the sample cache."""
        with self._lock:
            if not self._initialised:
                return
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
            except Exception as exc:
                logger.warning("[Sounds] Mixer teardown error: %s", exc)
            self._cache.clear()
            self._initialised = False
        logger.debug("[Sounds] Mixer torn down.")

    def preload(self, events: list[SoundEvent] | None = None) -> None:
        """
        Pre-load sound assets into the cache to avoid first-play latency.

        Args:
            events: Specific events to load. Defaults to all SoundEvent members.
        """
        if not self.setup():
            return
        targets = events or list(SoundEvent)
        for event in targets:
            self._load(event)
        logger.info("[Sounds] Preloaded %d sound(s).", len(targets))

    # ── Volume ────────────────────────────────────────────────────────────────

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """Set global volume (0.0 - 1.0). Updates all cached samples."""
        clamped = max(0.0, min(1.0, value))
        with self._lock:
            self._volume = clamped
            for sound in self._cache.values():
                if sound is not None:
                    sound.set_volume(clamped)
        logger.debug("[Sounds] Volume set to %.2f.", clamped)

    # ── Playback ──────────────────────────────────────────────────────────────

    def play(self, event: SoundEvent, *, blocking: bool = False) -> None:
        """
        Play the sound associated with *event*.

        Args:
            event:    The event to signify.
            blocking: If True, waits for the sound to finish before returning.
                      If False, returns immediately (pygame mixes in background).
        """
        if not self.setup():
            return

        sound = self._load(event)
        if sound is None:
            return

        try:
            channel = sound.play()
            if blocking and channel is not None:
                while channel.get_busy():
                    pygame.time.wait(10)
        except Exception as exc:
            logger.error("[Sounds] Playback error for %s: %s", event.name, exc)

    def stop_all(self) -> None:
        """Immediately stop all currently playing sounds."""
        if not _PYGAME_AVAILABLE or not self._initialised:
            return
        try:
            pygame.mixer.stop()
            logger.debug("[Sounds] All channels stopped.")
        except Exception as exc:
            logger.warning("[Sounds] stop_all() error: %s", exc)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self, event: SoundEvent) -> pygame.mixer.Sound | None:
        """
        Return the cached Sound for *event*, loading from disk if needed.
        Thread-safe — holds the lock for the full load operation so two
        threads cannot race to load the same asset simultaneously.
        """
        with self._lock:
            if event in self._cache:
                return self._cache[event]

            sound = self._load_asset(event)
            self._cache[event] = sound
            return sound

    def _load_asset(self, event: SoundEvent) -> pygame.mixer.Sound | None:
        """
        Resolve and load the WAV file for *event*.
        Falls back to the generic chime if the event-specific file is missing.
        Must be called while holding self._lock.
        """
        preferred_name = _EVENT_ASSETS.get(event, _FALLBACK_ASSET)
        preferred_path = _ASSETS_DIR / preferred_name
        fallback_path = _ASSETS_DIR / _FALLBACK_ASSET

        path = preferred_path if preferred_path.is_file() else fallback_path

        if not path.is_file():
            logger.warning(
                "[Sounds] No asset for %s (tried '%s' and '%s').",
                event.name,
                preferred_name,
                _FALLBACK_ASSET,
            )
            return None

        try:
            sound = pygame.mixer.Sound(str(path))
            sound.set_volume(self._volume)
            logger.debug("[Sounds] Loaded '%s' for %s.", path.name, event.name)
            return sound
        except Exception as exc:
            logger.error("[Sounds] Failed to load '%s': %s", path, exc)
            return None


# ── Module-level singleton and convenience API ────────────────────────────────

_controller = SoundController()


def setup(volume: float = _DEFAULT_VOLUME) -> bool:
    """
    Initialise the sound system.  Call once at startup.

    Args:
        volume: Initial global volume (0.0 - 1.0).

    Returns:
        True if the mixer is ready.
    """
    _controller.volume = volume
    return _controller.setup()


def preload(events: list[SoundEvent] | None = None) -> None:
    """Pre-load assets to eliminate first-play latency."""
    _controller.preload(events)


def play(event: SoundEvent, *, blocking: bool = False) -> None:
    """
    Play a system sound.

    Args:
        event:    The event to signify.
        blocking: Wait for playback to finish before returning.
    """
    _controller.play(event, blocking=blocking)


def stop_all() -> None:
    """Stop all currently playing sounds immediately."""
    _controller.stop_all()


def set_volume(value: float) -> None:
    """Set global playback volume (0.0 - 1.0)."""
    _controller.volume = value


def teardown() -> None:
    """Release all mixer resources.  Call at process shutdown."""
    _controller.teardown()
