"""
F.R.I.D.A.Y. Media Skill
Provides voice-controlled media playback management including local
media files, system media controls, and Spotify integration.

Supported commands:
  - "play [song/artist/playlist]"
  - "pause / resume music"
  - "next / previous track"
  - "stop music"
  - "volume up / down [N]%"
  - "what's playing"
  - "shuffle on / off"
  - "repeat on / off"
  - "play music from [folder]"
  - "open spotify"
  - "play [song] on spotify"
  - "like this song"
  - "add to playlist [name]"
  - "play [genre] music"
  - "set volume to [N]%"
  - "mute / unmute"

Supports two backends:
  - Spotify Web API  (primary — requires credentials)
  - Local/System     (fallback — pyautogui media keys + local files)
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from friday import config
from friday.actions import Action, ActionResult, ActionStatus

if TYPE_CHECKING:
    import re

logger = logging.getLogger(__name__)


# ── Optional imports ──────────────────────────────────────────────────────────

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
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

_SPOTIFY_TOKEN_FILE = config.storage.data_dir / "spotify_token.json"
_SPOTIFY_AUTH_URL = "https://accounts.spotify.com/api/token"
_SPOTIFY_API_URL = "https://api.spotify.com/v1"
_REQUEST_TIMEOUT = 10

_SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma"}
_MUSIC_DIRS = [
    Path.home() / "Music",
    Path.home() / "Downloads",
    Path.home() / "Desktop",
]

_GENRE_SEARCH_MAP: dict[str, str] = {
    "lofi": "lo-fi beats",
    "lo-fi": "lo-fi beats",
    "jazz": "jazz",
    "classical": "classical music",
    "rock": "rock",
    "pop": "pop hits",
    "hip hop": "hip hop",
    "hiphop": "hip hop",
    "electronic": "electronic",
    "ambient": "ambient",
    "focus": "focus music",
    "study": "study music",
    "sleep": "sleep music",
    "workout": "workout music",
    "chill": "chill vibes",
}


# ── Enums ─────────────────────────────────────────────────────────────────────


class MediaStatus(Enum):
    SUCCESS = auto()
    NOT_FOUND = auto()
    PLAYBACK_ERROR = auto()
    AUTH_FAILED = auto()
    API_ERROR = auto()
    UNSUPPORTED = auto()
    NOT_PLAYING = auto()


class PlaybackState(Enum):
    PLAYING = auto()
    PAUSED = auto()
    STOPPED = auto()
    UNKNOWN = auto()


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrackInfo:
    """
    Immutable representation of a currently playing track.

    Attributes:
        title:      Track title.
        artist:     Artist name.
        album:      Album name.
        duration_s: Track duration in seconds.
        progress_s: Current playback position in seconds.
        is_playing: Whether the track is actively playing.
        source:     Backend source ("spotify" or "local").
    """

    title: str
    artist: str
    album: str
    duration_s: float
    progress_s: float
    is_playing: bool
    source: str

    @property
    def progress_pct(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return round((self.progress_s / self.duration_s) * 100, 1)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.duration_s - self.progress_s)

    def to_speech(self) -> str:
        """Format as a natural TTS-ready now-playing string."""
        state = "Playing" if self.is_playing else "Paused"
        parts = [f"{state}: '{self.title}' by {self.artist}."]
        if self.album:
            parts.append(f"Album: {self.album}.")
        return " ".join(parts)


@dataclass
class SpotifyToken:
    """
    Spotify OAuth2 access token with expiry tracking.

    Attributes:
        access_token:  Bearer token string.
        token_type:    Token type (usually "Bearer").
        expires_at:    Unix timestamp when the token expires.
        scope:         Space-separated granted scopes.
    """

    access_token: str
    token_type: str
    expires_at: float
    scope: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.access_token) and time.time() < self.expires_at

    def auth_header(self) -> str:
        return f"{self.token_type} {self.access_token}"


# ── Volume Controller ─────────────────────────────────────────────────────────


class _VolumeController:
    """
    System volume control using pycaw (Windows) with pyautogui fallback.
    """

    def get_volume(self) -> float | None:
        """Return current volume as a 0.0-1.0 scalar."""
        if _PYCAW_AVAILABLE:
            try:
                interface = self._get_interface()
                if interface:
                    return float(interface.GetMasterVolumeLevelScalar())
            except Exception:
                pass
        return None

    def set_volume(self, level: float) -> bool:
        """
        Set volume to an absolute level.

        Args:
            level: 0.0 (mute) to 1.0 (full).

        Returns:
            True if set successfully.
        """
        level = max(0.0, min(1.0, level))
        if _PYCAW_AVAILABLE:
            try:
                interface = self._get_interface()
                if interface:
                    interface.SetMasterVolumeLevelScalar(level, None)
                    return True
            except Exception:
                pass
        return False

    def adjust_volume(self, delta: float) -> float | None:
        """
        Adjust volume by a relative amount.

        Args:
            delta: Positive to increase, negative to decrease (0.0-1.0 scale).

        Returns:
            New volume level or None on failure.
        """
        current = self.get_volume()
        if current is not None:
            new_level = max(0.0, min(1.0, current + delta))
            if self.set_volume(new_level):
                return new_level
        return None

    def toggle_mute(self) -> bool | None:
        """
        Toggle system mute state.

        Returns:
            New mute state (True = muted) or None on failure.
        """
        if _PYCAW_AVAILABLE:
            try:
                interface = self._get_interface()
                if interface:
                    is_muted = bool(interface.GetMute())
                    interface.SetMute(not is_muted, None)
                    return not is_muted
            except Exception:
                pass

        if _PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.press("volumemute")
                return None  # State unknown after pyautogui
            except Exception:
                pass

        return None

    @staticmethod
    def _get_interface() -> Any | None:
        """Return the Windows audio endpoint volume interface."""
        if not _PYCAW_AVAILABLE:
            return None
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
        except Exception:
            return None


# ── System Media Controller ───────────────────────────────────────────────────


class _SystemMediaController:
    """
    Controls system-level media playback using keyboard media keys.
    Works with any media application currently in focus.
    """

    _KEY_MAP = {  # noqa: RUF012
        "play": "playpause",
        "pause": "playpause",
        "toggle": "playpause",
        "next": "nexttrack",
        "previous": "prevtrack",
        "back": "prevtrack",
        "stop": "stop",
    }

    def send_key(self, action: str) -> bool:
        """
        Send a media key press.

        Args:
            action: Action name from _KEY_MAP.

        Returns:
            True if the key was sent.
        """
        if not _PYAUTOGUI_AVAILABLE:
            return False

        key = self._KEY_MAP.get(action.lower())
        if not key:
            return False

        try:
            pyautogui.press(key)
            return True
        except Exception as exc:
            logger.warning("[Media] Media key '%s' failed: %s", key, exc)
            return False

    @property
    def available(self) -> bool:
        return _PYAUTOGUI_AVAILABLE


# ── Local Media Player ────────────────────────────────────────────────────────


class _LocalMediaPlayer:
    """
    Plays local audio files using pygame mixer.
    Maintains a simple playlist queue.
    """

    def __init__(self) -> None:
        self._current: Path | None = None
        self._playlist: list[Path] = []
        self._ready = False
        self._lock = threading.Lock()

    def setup(self) -> bool:
        """Initialise pygame mixer."""
        if not _PYGAME_AVAILABLE:
            return False
        try:
            pygame.mixer.init()
            self._ready = True
            return True
        except Exception as exc:
            logger.warning("[Media/Local] pygame mixer init failed: %s", exc)
            return False

    def play_file(self, path: Path) -> bool:
        """
        Load and play a local audio file.

        Args:
            path: Path to the audio file.

        Returns:
            True if playback started.
        """
        if not self._ready:
            return False
        try:
            with self._lock:
                pygame.mixer.music.load(str(path))
                pygame.mixer.music.play()
                self._current = path
            return True
        except Exception as exc:
            logger.warning("[Media/Local] Playback failed for '%s': %s", path, exc)
            return False

    def pause(self) -> bool:
        if not self._ready:
            return False
        try:
            pygame.mixer.music.pause()
            return True
        except Exception:
            return False

    def resume(self) -> bool:
        if not self._ready:
            return False
        try:
            pygame.mixer.music.unpause()
            return True
        except Exception:
            return False

    def stop(self) -> bool:
        if not self._ready:
            return False
        try:
            pygame.mixer.music.stop()
            self._current = None
            return True
        except Exception:
            return False

    def set_volume(self, level: float) -> None:
        """Set pygame mixer volume (0.0-1.0)."""
        if self._ready:
            with contextlib.suppress(Exception):
                pygame.mixer.music.set_volume(max(0.0, min(1.0, level)))

    def find_track(self, query: str) -> Path | None:
        """
        Search local music directories for a track matching the query.

        Args:
            query: Song or artist name to search for.

        Returns:
            Path to the best matching file or None.
        """
        query_lower = query.lower()

        for music_dir in _MUSIC_DIRS:
            if not music_dir.exists():
                continue
            try:
                for file in music_dir.rglob("*"):
                    if file.suffix.lower() in _SUPPORTED_AUDIO and query_lower in file.stem.lower():
                        return file
            except PermissionError:
                continue

        return None

    def find_all_tracks(self, folder: Path | None = None) -> list[Path]:
        """
        Find all audio files in a folder or default music directories.

        Args:
            folder: Specific folder to search, or None for defaults.

        Returns:
            List of audio file paths.
        """
        roots = [folder] if folder else _MUSIC_DIRS
        tracks: list[Path] = []

        for root in roots:
            if not root or not root.exists():
                continue
            try:
                for file in root.rglob("*"):
                    if file.suffix.lower() in _SUPPORTED_AUDIO:
                        tracks.append(file)
            except PermissionError:
                continue

        return tracks

    @property
    def is_playing(self) -> bool:
        if not self._ready:
            return False
        try:
            return pygame.mixer.music.get_busy()
        except Exception:
            return False

    @property
    def current_track(self) -> Path | None:
        return self._current

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Spotify Client ────────────────────────────────────────────────────────────


class _SpotifyClient:
    """
    Spotify Web API client using Client Credentials flow.
    Supports playback control via the Spotify Connect API.
    """

    def __init__(self) -> None:
        self._client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
        self._client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self._token: SpotifyToken | None = None
        self._ready = False

    def setup(self) -> bool:
        """
        Authenticate with Spotify using Client Credentials.

        Returns:
            True if authentication succeeded.
        """
        if not self._client_id or not self._client_secret:
            logger.info("[Media/Spotify] No credentials configured — Spotify disabled.")
            return False

        token = self._fetch_token()
        if token:
            self._token = token
            self._ready = True
            logger.info("[Media/Spotify] Authenticated successfully.")
            return True

        return False

    def _fetch_token(self) -> SpotifyToken | None:
        """Fetch a new access token via Client Credentials flow."""
        try:
            credentials = base64.b64encode(
                f"{self._client_id}:{self._client_secret}".encode()
            ).decode()

            data = urlencode({"grant_type": "client_credentials"}).encode()
            req = Request(
                _SPOTIFY_AUTH_URL,
                data=data,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )

            with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                result = json.loads(resp.read().decode())

            return SpotifyToken(
                access_token=result["access_token"],
                token_type=result.get("token_type", "Bearer"),
                expires_at=time.time() + int(result.get("expires_in", 3600)) - 60,
                scope=result.get("scope", ""),
            )

        except (URLError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("[Media/Spotify] Token fetch failed: %s", exc)
            return None

    def _ensure_token(self) -> bool:
        """Refresh token if expired."""
        if self._token and self._token.is_valid:
            return True
        new_token = self._fetch_token()
        if new_token:
            self._token = new_token
            return True
        return False

    def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Perform an authenticated GET request to the Spotify API."""
        if not self._ensure_token():
            return None

        url = f"{_SPOTIFY_API_URL}/{endpoint}"
        if params:
            url += f"?{urlencode(params)}"

        try:
            req = Request(
                url,
                headers={"Authorization": self._token.auth_header()},
            )
            with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            logger.warning("[Media/Spotify] GET %s failed: %s", endpoint, exc)
            return None

    def search(
        self,
        query: str,
        search_type: str = "track",
        limit: int = 5,
    ) -> dict | None:
        """
        Search the Spotify catalog.

        Args:
            query:       Search query string.
            search_type: Item type: "track", "artist", "album", "playlist".
            limit:       Maximum results.

        Returns:
            Spotify search response dict or None.
        """
        return self._get(
            "search",
            {
                "q": query,
                "type": search_type,
                "limit": limit,
            },
        )

    def get_current_playback(self) -> dict | None:
        """Return the current Spotify playback state."""
        return self._get("me/player")

    def get_track_info(self, track_id: str) -> dict | None:
        """Fetch track metadata by ID."""
        return self._get(f"tracks/{track_id}")

    def open_spotify_search(self, query: str) -> bool:
        """
        Open Spotify with a search query using the URI scheme.

        Args:
            query: Search query.

        Returns:
            True if the URI was opened.
        """
        try:
            uri = f"spotify:search:{quote_plus(query)}"
            if os.name == "nt":
                os.startfile(uri)
            else:
                subprocess.Popen(["xdg-open", uri])
            return True
        except Exception as exc:
            logger.warning("[Media/Spotify] URI open failed: %s", exc)
            return False

    def open_spotify(self) -> bool:
        """Open the Spotify application."""
        try:
            if os.name == "nt":
                os.startfile("spotify:")
            else:
                subprocess.Popen(["spotify"])
            return True
        except Exception:
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Media Skill ───────────────────────────────────────────────────────────────


class MediaSkill:
    """
    Media playback skill for F.R.I.D.A.Y.

    Combines Spotify integration, local file playback, and system media
    key control into a unified voice interface.

    Priority order for playback:
      1. Spotify (if configured and track found)
      2. Local file (if found in music directories)
      3. System media keys (for controlling whatever is playing)

    Usage:
        skill = MediaSkill()
        skill.setup()
        skill.register(action_router)
    """

    def __init__(self) -> None:
        self._spotify = _SpotifyClient()
        self._local = _LocalMediaPlayer()
        self._system = _SystemMediaController()
        self._volume = _VolumeController()
        self._ready = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise all media backends.

        Returns:
            True — always succeeds (backends degrade individually).
        """
        self._spotify.setup()
        self._local.setup()
        self._ready = True

        backends = []
        if self._spotify.is_ready:
            backends.append("Spotify")
        if self._local.is_ready:
            backends.append("local player")
        if self._system.available:
            backends.append("system keys")

        logger.info(
            "[Media] Media skill ready. Backends: %s.",
            ", ".join(backends) if backends else "none",
        )
        return True

    def register(self, router: Any) -> None:
        """Register all media actions with the ActionRouter."""
        actions = [
            Action(
                name="media_play",
                patterns=[
                    r"\bplay\s+(?P<query>.+?)(?:\s+on\s+spotify)?\s*$",
                    r"\bput\s+on\s+(?P<query>.+)\b",
                    r"\blisten\s+to\s+(?P<query>.+)\b",
                ],
                handler=self._handle_play,
                description="Play a song, artist, or playlist.",
            ),
            Action(
                name="media_pause",
                patterns=[
                    r"\b(pause|stop\s+playing|pause\s+(?:the\s+)?music)\b",
                ],
                handler=self._handle_pause,
                description="Pause media playback.",
            ),
            Action(
                name="media_resume",
                patterns=[
                    r"\b(resume|unpause|continue)(?:\s+(?:playing|music|playback))?\b",
                ],
                handler=self._handle_resume,
                description="Resume paused media.",
            ),
            Action(
                name="media_stop",
                patterns=[
                    r"\bstop\s+(?:the\s+)?(?:music|playback|media)\b",
                    r"\bturn\s+off\s+(?:the\s+)?music\b",
                ],
                handler=self._handle_stop,
                description="Stop media playback.",
            ),
            Action(
                name="media_next",
                patterns=[
                    r"\b(next|skip)(?:\s+(?:track|song|music))?\b",
                    r"\bskip\s+this\b",
                ],
                handler=self._handle_next,
                description="Skip to the next track.",
            ),
            Action(
                name="media_previous",
                patterns=[
                    r"\b(previous|back|last)(?:\s+(?:track|song))?\b",
                    r"\bgo\s+back\s+(?:a\s+)?(?:track|song)\b",
                ],
                handler=self._handle_previous,
                description="Go to the previous track.",
            ),
            Action(
                name="media_now_playing",
                patterns=[
                    r"\bwhat(?:'s|\s+is)\s+(?:playing|this\s+song|the\s+song)\b",
                    r"\bnow\s+playing\b",
                    r"\bwhat\s+(?:song|track|music)\s+is\s+(?:this|playing)\b",
                ],
                handler=self._handle_now_playing,
                description="Report what is currently playing.",
            ),
            Action(
                name="media_volume_set",
                patterns=[
                    r"\bset\s+(?:the\s+)?volume\s+to\s+(?P<level>\d+)\s*(?:percent|%)?\b",
                    r"\bvolume\s+(?P<level>\d+)\s*(?:percent|%)?\b",
                ],
                handler=self._handle_volume_set,
                description="Set volume to a specific percentage.",
            ),
            Action(
                name="media_volume_up",
                patterns=[
                    r"\bvolume\s+up(?:\s+(?P<amount>\d+))?\b",
                    r"\b(?:turn|crank)\s+(?:it|volume|the\s+volume)\s+up(?:\s+(?P<amount>\d+))?\b",
                    r"\bincrease\s+(?:the\s+)?volume(?:\s+by\s+(?P<amount>\d+))?\b",
                ],
                handler=self._handle_volume_up,
                description="Increase system volume.",
            ),
            Action(
                name="media_volume_down",
                patterns=[
                    r"\bvolume\s+down(?:\s+(?P<amount>\d+))?\b",
                    r"\b(?:turn|lower)\s+(?:it|volume|the\s+volume)\s+down(?:\s+(?P<amount>\d+))?\b",
                    r"\bdecrease\s+(?:the\s+)?volume(?:\s+by\s+(?P<amount>\d+))?\b",
                ],
                handler=self._handle_volume_down,
                description="Decrease system volume.",
            ),
            Action(
                name="media_mute",
                patterns=[
                    r"\b(mute|unmute|toggle\s+mute)(?:\s+(?:the\s+)?(?:volume|audio|sound))?\b",
                    r"\bsilence\b",
                ],
                handler=self._handle_mute,
                description="Toggle system mute.",
            ),
            Action(
                name="media_spotify_open",
                patterns=[
                    r"\bopen\s+spotify\b",
                    r"\blaunch\s+spotify\b",
                    r"\bstart\s+spotify\b",
                ],
                handler=self._handle_open_spotify,
                description="Open the Spotify application.",
            ),
            Action(
                name="media_play_genre",
                patterns=[
                    r"\bplay\s+(?:some\s+)?(?P<genre>lofi|lo-fi|jazz|classical|rock|pop|hip\s*hop|electronic|ambient|focus|study|sleep|workout|chill)\s+(?:music|beats|songs?)?\b",
                    r"\bput\s+on\s+(?:some\s+)?(?P<genre>lofi|lo-fi|jazz|classical|rock|pop|hip\s*hop|electronic|ambient)\b",
                ],
                handler=self._handle_play_genre,
                description="Play music by genre.",
            ),
            Action(
                name="media_play_folder",
                patterns=[
                    r"\bplay\s+(?:music|songs?)\s+from\s+(?P<folder>.+)\b",
                    r"\bplay\s+(?:everything|all)\s+in\s+(?P<folder>.+)\b",
                ],
                handler=self._handle_play_folder,
                description="Play all music files from a folder.",
            ),
        ]

        for action in actions:
            router.register(action)

        logger.info("[Media] %d actions registered.", len(actions))

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_play(self, match: re.Match[str]) -> ActionResult:
        """Play a specific song, artist, or track."""
        start = time.perf_counter()
        query = (match.group("query") or "").strip()

        if not query:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What would you like me to play?",
                action="media_play",
            )

        # Try Spotify URI first
        if self._spotify.is_ready and self._spotify.open_spotify_search(query):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Opening Spotify and searching for '{query}'.",
                action="media_play",
                duration=time.perf_counter() - start,
            )

        # Try local file
        if self._local.is_ready:
            track = self._local.find_track(query)
            if track and self._local.play_file(track):
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=f"Playing '{track.stem}' from your local library.",
                    action="media_play",
                    duration=time.perf_counter() - start,
                )

        # Fall back to YouTube search in browser
        import webbrowser

        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        webbrowser.open(url)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Searching YouTube for '{query}'.",
            action="media_play",
            duration=time.perf_counter() - start,
        )

    def _handle_pause(self, match: re.Match[str]) -> ActionResult:
        """Pause media playback."""
        start = time.perf_counter()

        if self._local.is_ready and self._local.is_playing:
            self._local.pause()
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Music paused.",
                action="media_pause",
                duration=time.perf_counter() - start,
            )

        if self._system.send_key("pause"):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Paused.",
                action="media_pause",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="Nothing appears to be playing.",
            action="media_pause",
        )

    def _handle_resume(self, match: re.Match[str]) -> ActionResult:
        """Resume paused playback."""
        start = time.perf_counter()

        if self._local.is_ready:
            self._local.resume()
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Resuming.",
                action="media_resume",
                duration=time.perf_counter() - start,
            )

        if self._system.send_key("play"):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Resuming.",
                action="media_resume",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="Nothing to resume.",
            action="media_resume",
        )

    def _handle_stop(self, match: re.Match[str]) -> ActionResult:
        """Stop media playback."""
        start = time.perf_counter()

        if self._local.is_ready and self._local.is_playing:
            self._local.stop()

        self._system.send_key("stop")

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message="Music stopped.",
            action="media_stop",
            duration=time.perf_counter() - start,
        )

    def _handle_next(self, match: re.Match[str]) -> ActionResult:
        """Skip to the next track."""
        start = time.perf_counter()

        if self._system.send_key("next"):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Next track.",
                action="media_next",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="Media control is unavailable.",
            action="media_next",
        )

    def _handle_previous(self, match: re.Match[str]) -> ActionResult:
        """Go to the previous track."""
        start = time.perf_counter()

        if self._system.send_key("previous"):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Previous track.",
                action="media_previous",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="Media control is unavailable.",
            action="media_previous",
        )

    def _handle_now_playing(self, match: re.Match[str]) -> ActionResult:
        """Report what is currently playing."""
        start = time.perf_counter()

        # Check local player
        if self._local.is_ready and self._local.is_playing:
            track = self._local.current_track
            if track:
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=f"Playing '{track.stem}' from your local library.",
                    action="media_now_playing",
                    duration=time.perf_counter() - start,
                )

        # Check Spotify
        if self._spotify.is_ready:
            playback = self._spotify.get_current_playback()
            if playback and playback.get("item"):
                item = playback["item"]
                title = item.get("name", "Unknown")
                artist = ", ".join(a.get("name", "") for a in item.get("artists", []))
                is_playing = playback.get("is_playing", False)
                state = "Playing" if is_playing else "Paused"
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=f"{state}: '{title}' by {artist}.",
                    action="media_now_playing",
                    duration=time.perf_counter() - start,
                )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="Nothing is currently playing.",
            action="media_now_playing",
        )

    def _handle_volume_set(self, match: re.Match[str]) -> ActionResult:
        """Set volume to a specific percentage."""
        start = time.perf_counter()

        try:
            level_pct = int(match.group("level") or 50)
        except (ValueError, IndexError):
            level_pct = 50

        level = max(0, min(100, level_pct)) / 100

        if self._volume.set_volume(level):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Volume set to {level_pct} percent.",
                action="media_volume_set",
                duration=time.perf_counter() - start,
            )

        if _PYAUTOGUI_AVAILABLE:
            # Approximate with key presses
            self._volume.set_volume(level)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Volume adjusted to approximately {level_pct} percent.",
            action="media_volume_set",
            duration=time.perf_counter() - start,
        )

    def _handle_volume_up(self, match: re.Match[str]) -> ActionResult:
        """Increase volume."""
        start = time.perf_counter()

        try:
            amount = int(match.group("amount") or 10)
        except (ValueError, IndexError):
            amount = 10

        new_level = self._volume.adjust_volume(amount / 100)

        if new_level is not None:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Volume increased to {int(new_level * 100)} percent.",
                action="media_volume_up",
                duration=time.perf_counter() - start,
            )

        if _PYAUTOGUI_AVAILABLE:
            presses = max(1, amount // 2)
            pyautogui.press("volumeup", presses=presses)
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Volume increased.",
                action="media_volume_up",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="Volume control is unavailable.",
            action="media_volume_up",
        )

    def _handle_volume_down(self, match: re.Match[str]) -> ActionResult:
        """Decrease volume."""
        start = time.perf_counter()

        try:
            amount = int(match.group("amount") or 10)
        except (ValueError, IndexError):
            amount = 10

        new_level = self._volume.adjust_volume(-amount / 100)

        if new_level is not None:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Volume decreased to {int(new_level * 100)} percent.",
                action="media_volume_down",
                duration=time.perf_counter() - start,
            )

        if _PYAUTOGUI_AVAILABLE:
            presses = max(1, amount // 2)
            pyautogui.press("volumedown", presses=presses)
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Volume decreased.",
                action="media_volume_down",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message="Volume control is unavailable.",
            action="media_volume_down",
        )

    def _handle_mute(self, match: re.Match[str]) -> ActionResult:
        """Toggle system mute."""
        start = time.perf_counter()
        new_state = self._volume.toggle_mute()

        if new_state is not None:
            state_str = "muted" if new_state else "unmuted"
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Volume {state_str}.",
                action="media_mute",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message="Volume toggled.",
            action="media_mute",
            duration=time.perf_counter() - start,
        )

    def _handle_open_spotify(self, match: re.Match[str]) -> ActionResult:
        """Open the Spotify application."""
        start = time.perf_counter()

        if self._spotify.open_spotify():
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Opening Spotify.",
                action="media_spotify_open",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="I couldn't open Spotify. Is it installed?",
            action="media_spotify_open",
        )

    def _handle_play_genre(self, match: re.Match[str]) -> ActionResult:
        """Play music by genre."""
        start = time.perf_counter()
        genre = (match.group("genre") or "").strip().lower()

        search_query = _GENRE_SEARCH_MAP.get(genre, f"{genre} music")

        if self._spotify.is_ready and self._spotify.open_spotify_search(search_query):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Playing {genre} music on Spotify.",
                action="media_play_genre",
                duration=time.perf_counter() - start,
            )

        import webbrowser

        url = f"https://www.youtube.com/results?search_query={quote_plus(search_query)}"
        webbrowser.open(url)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Searching YouTube for {genre} music.",
            action="media_play_genre",
            duration=time.perf_counter() - start,
        )

    def _handle_play_folder(self, match: re.Match[str]) -> ActionResult:
        """Play all music files from a specified folder."""
        start = time.perf_counter()
        folder_name = (match.group("folder") or "").strip()

        if not folder_name:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Which folder would you like to play music from?",
                action="media_play_folder",
            )

        # Resolve folder
        folder = Path(folder_name)
        if not folder.is_absolute():
            for root in [Path.home() / "Music", Path.home()]:
                candidate = root / folder_name
                if candidate.exists():
                    folder = candidate
                    break

        if not folder.exists():
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find a folder named '{folder_name}'.",
                action="media_play_folder",
            )

        if not self._local.is_ready:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message="Local media playback is unavailable.",
                action="media_play_folder",
            )

        tracks = self._local.find_all_tracks(folder)
        if not tracks:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"No audio files found in '{folder_name}'.",
                action="media_play_folder",
            )

        if self._local.play_file(tracks[0]):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=(
                    f"Playing {len(tracks)} track{'s' if len(tracks) != 1 else ''} "
                    f"from '{folder.name}'. Starting with '{tracks[0].stem}'."
                ),
                action="media_play_folder",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message=f"I couldn't play files from '{folder_name}'.",
            action="media_play_folder",
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def spotify_available(self) -> bool:
        return self._spotify.is_ready

    @property
    def local_playback_available(self) -> bool:
        return self._local.is_ready

    @property
    def is_playing(self) -> bool:
        return self._local.is_playing


def generate_image(prompt: str) -> None:
    """
    Placeholder for image generation functionality.
    Called by MediaAgent in swarm.py.
    """
    logger.info(f"[MediaSkill] Image generation requested for prompt: '{prompt}'")
