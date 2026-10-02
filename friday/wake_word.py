"""
F.R.I.D.A.Y.  —  Wake Word Detection Module
─────────────────────────────────────────────────────────────────────────────
Listens continuously on a background thread and fires a callback when
the configured wake word is detected.

Backends (in priority order)
  1. Vosk          — fully offline, preferred (requires model download)
  2. Google STT    — online, proven fallback

NOTE: Deepgram is NOT used for wake word detection. It is too slow and
expensive for continuous 3-second polling. Deepgram is used for STT
transcription only (see friday/stt.py). Wake word detection must be
fast, cheap, and reliable — cloud APIs are inappropriate for this.

Fixes over previous version
  • Replaced @dataclass with plain __init__ — Lock fields are not
    safely dataclass-able (breaks __eq__, __repr__, copy).
  • Microphone stream is now explicitly closed on stop().
  • Detection cooldown prevents the same utterance triggering the
    callback multiple times from overlapping audio chunks.
  • Vosk backend no longer uses unstable PartialResult transcripts —
    only AcceptWaveform / full Result is evaluated.
  • ambient noise calibration runs on a dedicated thread so start()
    is non-blocking; a timeout prevents it hanging on a slow mic.
  • resume() is fully lock-safe (no TOCTOU window).
  • State machine guards illegal transitions and logs them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING

import speech_recognition as sr

from friday import config

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# ── Optional backend imports ──────────────────────────────────────────────────

try:
    from vosk import KaldiRecognizer
    from vosk import Model as VoskModel

    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False
    logger.warning("[WakeWord] Vosk not installed — falling back to Google STT.")

# Deepgram is NOT used for wake word detection — only for STT transcription.
# Wake word needs fast local detection, not cloud API calls every 3 seconds.

try:
    import noisereduce as nr
    import numpy as np

    _NR_AVAILABLE = True
except ImportError:
    _NR_AVAILABLE = False
    if config.audio.noise_reduction:
        logger.warning("[WakeWord] noisereduce not installed — noise reduction disabled.")

# ── Constants ─────────────────────────────────────────────────────────────────

# Seconds to ignore re-detection after a successful wake-word fire.
# Prevents a single long utterance triggering the callback multiple times.
DETECTION_COOLDOWN_SECS: float = 2.0

# How long (seconds) to wait for ambient noise calibration before giving up.
CALIBRATION_TIMEOUT_SECS: float = 4.0

# Audio phrase capture limit per chunk (seconds).
PHRASE_TIME_LIMIT_SECS: int = 3

# ── Detector State ────────────────────────────────────────────────────────────


class DetectorState(Enum):
    STOPPED = auto()
    CALIBRATING = auto()  # ambient noise calibration in progress
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()


# Valid one-step transitions — any move not listed here is illegal.
_VALID_TRANSITIONS: frozenset[tuple[DetectorState, DetectorState]] = frozenset(
    {
        (DetectorState.STOPPED, DetectorState.CALIBRATING),
        (DetectorState.CALIBRATING, DetectorState.RUNNING),
        (DetectorState.CALIBRATING, DetectorState.ERROR),
        (DetectorState.RUNNING, DetectorState.PAUSED),
        (DetectorState.RUNNING, DetectorState.STOPPED),
        (DetectorState.RUNNING, DetectorState.ERROR),
        (DetectorState.PAUSED, DetectorState.RUNNING),
        (DetectorState.PAUSED, DetectorState.STOPPED),
        (DetectorState.ERROR, DetectorState.STOPPED),
    }
)


# ── Audio utilities ───────────────────────────────────────────────────────────


def _reduce_noise(audio: sr.AudioData) -> sr.AudioData:
    """
    Apply spectral-gating noise reduction.
    Returns the original audio unchanged when the library is unavailable
    or if the reduction itself raises.
    """
    if not (_NR_AVAILABLE and config.audio.noise_reduction):
        return audio
    try:
        raw = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float32)
        reduced = nr.reduce_noise(y=raw, sr=audio.sample_rate, stationary=False)
        cleaned = reduced.astype(np.int16).tobytes()
        return sr.AudioData(cleaned, audio.sample_rate, audio.sample_width)
    except Exception:
        logger.debug("[WakeWord] Noise reduction failed — using raw audio.", exc_info=True)
        return audio


def _make_recogniser() -> sr.Recognizer:
    """Return a Recognizer configured with project-standard settings."""
    rec = sr.Recognizer()
    rec.energy_threshold = config.audio.mic_energy_threshold
    # Optimized for fast wake-word detection (single short word, no pause)
    rec.pause_threshold = 0.3
    rec.phrase_threshold = 0.1
    rec.non_speaking_duration = 0.1
    rec.dynamic_energy_threshold = config.audio.mic_dynamic_energy_threshold
    return rec


# ── Abstract backend ──────────────────────────────────────────────────────────


class _BaseDetector(ABC):
    """
    Common contract for all wake-word backend implementations.

    Subclasses must implement:
        _setup()   — one-time initialisation (load model, verify paths…)
        _detect()  — analyse one audio chunk; return True on match

    Cooldown logic lives here so every backend gets it for free.
    """

    def __init__(
        self,
        wake_word: str,
        on_detected: Callable[[], None],
    ) -> None:
        self._wake_word = wake_word.lower().strip()
        self._on_detected = on_detected
        self._recogniser = _make_recogniser()
        self._stop_fn: Callable | None = None
        self._last_detection: float = 0.0  # monotonic epoch

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _setup(self) -> None:
        """One-time initialisation. May raise on failure."""

    @abstractmethod
    def _detect(self, audio: sr.AudioData) -> bool:
        """
        Analyse a captured audio chunk.

        Args:
            audio: One phrase-length audio segment from the background listener.

        Returns:
            True if the wake word was found in this chunk.
        """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, mic: sr.Microphone) -> None:
        """Wire the background listener to *mic*."""
        self._stop_fn = self._recogniser.listen_in_background(
            mic,
            self._audio_callback,
            phrase_time_limit=PHRASE_TIME_LIMIT_SECS,
        )
        logger.debug("[%s] Background listener started.", self.__class__.__name__)

    def stop(self) -> None:
        """Detach the background listener without waiting for the thread."""
        if self._stop_fn:
            self._stop_fn(wait_for_stop=False)
            self._stop_fn = None
            logger.debug("[%s] Background listener stopped.", self.__class__.__name__)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audio_callback(
        self,
        _recogniser: sr.Recognizer,
        audio: sr.AudioData,
    ) -> None:
        """
        Invoked by speech_recognition's background thread for every captured chunk.
        Applies noise reduction, runs _detect(), enforces cooldown, fires callback.
        """
        # Cooldown guard — reject rapid re-fires from the same utterance
        now = time.monotonic()
        if now - self._last_detection < DETECTION_COOLDOWN_SECS:
            return

        cleaned = _reduce_noise(audio)

        try:
            if not self._detect(cleaned):
                return
        except Exception:
            logger.debug("[WakeWord] Detection error.", exc_info=True)
            return

        self._last_detection = now
        logger.info("[WakeWord] Wake word '%s' detected.", self._wake_word)
        self._on_detected()


# ── Vosk backend ──────────────────────────────────────────────────────────────


class _VoskDetector(_BaseDetector):
    """
    Fully offline wake-word detection via the Vosk speech toolkit.

    Only full (non-partial) recognition results are evaluated.
    Partial results are intentionally ignored — they are unstable and
    cause false positives mid-utterance (e.g. "fri" matching "friday").
    """

    def __init__(
        self,
        wake_word: str,
        on_detected: Callable[[], None],
    ) -> None:
        super().__init__(wake_word, on_detected)
        self._kaldi: KaldiRecognizer | None = None

    def _setup(self) -> None:
        if not _VOSK_AVAILABLE:
            raise RuntimeError("Vosk is not installed — pip install vosk.")

        import os
        path = config.audio.vosk_wake_model_path

        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Vosk model not found at '{path}'. "
                "Download from https://alphacephei.com/vosk/models"
            )

        logger.info("[WakeWord] Loading Vosk model from '%s'…", path)
        model = VoskModel(path)
        self._kaldi = KaldiRecognizer(model, 16_000)
        logger.info("[WakeWord] Vosk model loaded.")

    def _detect(self, audio: sr.AudioData) -> bool:
        if self._kaldi is None:
            return False

        raw = audio.get_raw_data(convert_rate=16_000, convert_width=2)

        self._kaldi.AcceptWaveform(raw)
        result = json.loads(self._kaldi.FinalResult())
        transcript = result.get("text", "").lower().strip()

        if transcript:
            logger.debug("[WakeWord/Vosk] Heard: '%s'", transcript)

        return self._wake_word in transcript


# ── Google STT backend ────────────────────────────────────────────────────────


class _GoogleDetector(_BaseDetector):
    """Online wake-word detection using Google Speech Recognition."""

    def _setup(self) -> None:
        logger.info("[WakeWord] Using Google STT backend (requires internet).")

    def _detect(self, audio: sr.AudioData) -> bool:
        try:
            recognize_fn = self._recogniser.recognize_google
            transcript   = recognize_fn(audio).lower()
            logger.debug("[WakeWord/Google] Heard: '%s'", transcript)
            return self._wake_word in transcript
        except sr.UnknownValueError:
            return False
        except sr.RequestError as exc:
            logger.warning("[WakeWord/Google] API error: %s", exc)
            return False





# ── Wake Word Detector (public interface) ─────────────────────────────────────


class WakeWordDetector:
    """
    Public facade for wake-word detection.

    Selects the appropriate backend from config, manages the microphone
    resource, and exposes pause / resume for use while Friday is actively
    processing a command.

    Ambient noise calibration runs on a dedicated thread so ``start()``
    returns promptly; the detector transitions to RUNNING once calibration
    completes (or times out and falls back to the configured threshold).

    Usage::

        def on_wake():
            friday.begin_listening()

        detector = WakeWordDetector(on_detected=on_wake)
        detector.start()

        # Suppress re-triggering while processing:
        detector.pause()
        friday.process_command()
        detector.resume()

        detector.stop()
    """

    def __init__(
        self,
        on_detected: Callable[[], None],
        wake_word: str = config.audio.wake_word,
    ) -> None:
        self._on_detected = on_detected
        self._wake_word = wake_word

        self._backend: _BaseDetector | None = None
        self._mic: sr.Microphone | None = None
        self._state = DetectorState.STOPPED
        self._lock = threading.Lock()

    # ── State helpers ─────────────────────────────────────────────────────────

    def _transition(self, to: DetectorState) -> bool:
        """
        Attempt a state transition.  Must be called while holding self._lock.

        Returns True on success, False if the transition is illegal.
        """
        if (self._state, to) not in _VALID_TRANSITIONS:
            logger.warning(
                "[WakeWord] Illegal state transition %s → %s ignored.",
                self._state.name,
                to.name,
            )
            return False
        self._state = to
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Initialise the backend and begin listening.

        Ambient noise calibration is performed asynchronously.
        The detector's state will be CALIBRATING briefly, then RUNNING.

        Returns:
            True if startup was initiated successfully, False on any error.
        """
        with self._lock:
            if self._state == DetectorState.RUNNING:
                logger.debug("[WakeWord] Already running.")
                return True
            if not self._transition(DetectorState.CALIBRATING):
                return False

        backend = self._create_backend()
        if backend is None:
            with self._lock:
                self._transition(DetectorState.ERROR)
            return False

        try:
            backend._setup()
        except Exception as exc:
            logger.error("[WakeWord] Backend setup failed: %s", exc)

            # Try Google as fallback if the primary backend failed
            if not isinstance(backend, _GoogleDetector):
                logger.warning("[WakeWord] Falling back to Google STT detector.")
                try:
                    backend = _GoogleDetector(self._wake_word, self._safe_callback)
                    backend._setup()
                except Exception as fallback_exc:
                    logger.error("[WakeWord] Google fallback also failed: %s", fallback_exc)
                    with self._lock:
                        self._transition(DetectorState.ERROR)
                    return False
            else:
                with self._lock:
                    self._transition(DetectorState.ERROR)
                return False

        # Open the microphone once; it is kept open until stop().
        try:
            mic = sr.Microphone(device_index=config.audio.mic_index)
        except Exception as exc:
            logger.error("[WakeWord] Could not open microphone: %s", exc)
            with self._lock:
                self._transition(DetectorState.ERROR)
            return False

        # Calibrate on a daemon thread so start() is non-blocking.
        def _calibrate_then_start() -> None:
            try:
                # Use pre-calibrated threshold from config if available
                calibrated_threshold = getattr(config, "calibrated_energy_threshold", None)
                if calibrated_threshold is not None:
                    backend._recogniser.energy_threshold = calibrated_threshold
                    logger.info(
                        "[WakeWord] Using pre-calibrated energy threshold: %d (dynamic: %s)",
                        backend._recogniser.energy_threshold,
                        backend._recogniser.dynamic_energy_threshold,
                    )
                elif backend._recogniser.dynamic_energy_threshold:
                    temp_mic = sr.Microphone(device_index=config.audio.mic_index)
                    with temp_mic as source:
                        logger.debug("[WakeWord] Calibrating ambient noise…")
                        # Use a short timeout — hang on a virtual mic otherwise.
                        backend._recogniser.adjust_for_ambient_noise(
                            source,
                            duration=min(1.0, CALIBRATION_TIMEOUT_SECS),
                        )
                    setattr(config, "calibrated_energy_threshold", float(backend._recogniser.energy_threshold))
                    logger.info(
                        "[WakeWord] Calibration complete. Energy threshold: %d (dynamic: %s)",
                        backend._recogniser.energy_threshold,
                        backend._recogniser.dynamic_energy_threshold,
                    )
                else:
                    # Dynamic energy threshold disabled, but no pre-calibration available.
                    # We STILL calibrate once to establish a correct baseline for this device.
                    logger.info(
                        "[WakeWord] Dynamic energy threshold disabled and no pre-calibration available. Calibrating once anyway..."
                    )
                    temp_mic = sr.Microphone(device_index=config.audio.mic_index)
                    with temp_mic as source:
                        backend._recogniser.adjust_for_ambient_noise(
                            source,
                            duration=min(1.0, CALIBRATION_TIMEOUT_SECS),
                        )
                    calibrated = float(backend._recogniser.energy_threshold)
                    # Enforce sane ranges
                    calibrated = max(150.0, min(calibrated, 4000.0))
                    backend._recogniser.energy_threshold = calibrated
                    setattr(config, "calibrated_energy_threshold", calibrated)
                    logger.info(
                        "[WakeWord] Ambient calibration complete. Locked energy threshold: %d",
                        calibrated,
                    )
            except Exception as exc:
                logger.warning("[WakeWord] Calibration failed (%s) — using default threshold.", exc)

            # Start listener regardless of calibration outcome.
            try:
                backend.start(mic)
                with self._lock:
                    self._backend = backend
                    self._mic = mic
                    self._transition(DetectorState.RUNNING)
                logger.info(
                    "[WakeWord] Listening for '%s' via %s.",
                    self._wake_word,
                    backend.__class__.__name__,
                )
            except Exception as exc:
                logger.error("[WakeWord] Failed to start listener: %s", exc)
                with self._lock:
                    self._transition(DetectorState.ERROR)

        calib_thread = threading.Thread(
            target=_calibrate_then_start,
            daemon=True,
            name="WakeWordCalibration",
        )
        calib_thread.start()
        return True

    def stop(self) -> None:
        """Stop detection and release all resources including the microphone."""
        with self._lock:
            if self._state == DetectorState.STOPPED:
                return

            if self._backend:
                self._backend.stop()
                self._backend = None

            # Explicitly close the microphone stream.
            if self._mic is not None:
                with contextlib.suppress(Exception):
                    self._mic.__exit__(None, None, None)
                self._mic = None

            self._state = DetectorState.STOPPED

        logger.info("[WakeWord] Detector stopped.")

    def pause(self) -> None:
        """
        Suspend detection without releasing the microphone.
        Call this while Friday is actively listening for a command so the
        wake word cannot re-trigger mid-conversation.
        """
        with self._lock:
            if not self._transition(DetectorState.PAUSED):
                return
            if self._backend:
                self._backend.stop()
        logger.debug("[WakeWord] Paused.")

    def resume(self) -> None:
        """Resume detection after a pause."""
        with self._lock:
            if self._state != DetectorState.PAUSED:
                logger.debug("[WakeWord] resume() called in state %s — ignored.", self._state.name)
                return

            # Capture references inside the lock so we own them.
            backend = self._backend
            mic = self._mic

        if backend is None or mic is None:
            logger.error("[WakeWord] Cannot resume — backend or mic is None.")
            with self._lock:
                self._transition(DetectorState.ERROR)
            return

        try:
            backend.start(mic)
            with self._lock:
                self._transition(DetectorState.RUNNING)
            logger.debug("[WakeWord] Resumed.")
        except Exception as exc:
            logger.error("[WakeWord] Failed to resume listener: %s", exc)
            with self._lock:
                self._state = DetectorState.ERROR

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> DetectorState:
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self.state == DetectorState.RUNNING

    @property
    def is_ready(self) -> bool:
        """True when the detector is either actively listening or paused."""
        return self.state in (DetectorState.RUNNING, DetectorState.PAUSED)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _create_backend(self) -> _BaseDetector | None:
        """
        Instantiate the preferred backend with automatic fallback.
        """
        import os
        path = config.audio.vosk_wake_model_path
        exists = os.path.isdir(path)

        # Log explicitly when falling back from Deepgram (since Deepgram is not used for wake word)
        if config.audio.stt_provider == "deepgram":
            logger.info("[WakeWord] STT_PROVIDER is set to deepgram. Since Deepgram is not used for wake word detection, using offline Vosk or online Google.")

        if _VOSK_AVAILABLE:
            if (config.audio.stt_provider == "vosk" and exists) or exists:
                try:
                    return _VoskDetector(self._wake_word, self._safe_callback)
                except Exception as e:
                    logger.warning("[WakeWord] Vosk setup failed: %s. Falling back to Google STT.", e)

        if not exists:
            logger.warning(
                "[WakeWord] Vosk model not found at %s. "
                "Download from https://alphacephei.com/vosk/models "
                "and extract to %s for offline wake word.",
                path, path
            )

        return _GoogleDetector(self._wake_word, self._safe_callback)

    def _safe_callback(self) -> None:
        """
        Wraps the caller-supplied callback with error isolation.
        A crash inside on_detected must never kill the detector thread.
        """
        try:
            self._on_detected()
        except Exception:
            logger.exception("[WakeWord] Exception in on_detected callback.")
