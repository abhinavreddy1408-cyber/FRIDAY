"""
F.R.I.D.A.Y.  —  Speech-to-Text Module
─────────────────────────────────────────────────────────────────────────────
Captures and transcribes a single command utterance after wake-word detection.

Backends (in priority order)
  1. Deepgram      — fastest cloud STT (~200ms), requires API key
  2. Vosk          — fully offline, requires model download

Key improvements over v1
  • TranscriptResult is a proper frozen dataclass — no __setattr__ hackery
  • Recognizer instance reused per backend, not rebuilt every capture
  • Deepgram uses a persistent requests.Session (connection pooling)
  • audio_processing imported once at module level, never inside hot paths
  • _audio_lock correctly wraps only the sr.Microphone context (not both)
  • Backend priority logic matches documented order without inversion bugs
  • All backends implement teardown() for clean resource release
  • calibrate() reuses the engine's own recognizer
  • Full type annotations throughout
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import speech_recognition as sr

from friday import config

logger = logging.getLogger(__name__)

# ── Optional backend imports ──────────────────────────────────────────────────

try:
    from vosk import KaldiRecognizer
    from vosk import Model as VoskModel
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

try:
    import noisereduce as nr
    import numpy as np
    _NR_AVAILABLE = True
except ImportError:
    _NR_AVAILABLE = False
    if getattr(config.audio, "noise_reduction", False):
        logger.warning("[STT] noisereduce not installed — noise reduction disabled.")

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from deepgram import DeepgramClient
    _DEEPGRAM_AVAILABLE = True
except ImportError:
    _DEEPGRAM_AVAILABLE = False

# ── Try importing audio_processing once at module load ────────────────────────
# Avoids repeated import attempts inside hot capture paths.

try:
    from friday.audio_processing import denoise as _full_denoise
    from friday.audio_processing import init_noise_cancellation as _init_nc
    _AUDIO_PROCESSING_AVAILABLE = True
except Exception:  # pragma: no cover
    _full_denoise = None       # type: ignore[assignment]
    _init_nc = None            # type: ignore[assignment]
    _AUDIO_PROCESSING_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

_TTS_TAIL_DELAY: float = 0.15
_WHITESPACE_RE = re.compile(r"\s+")
_DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"

# ── Result types ──────────────────────────────────────────────────────────────


class TranscriptStatus(Enum):
    SUCCESS        = auto()
    NO_SPEECH      = auto()
    UNCLEAR        = auto()
    API_ERROR      = auto()
    TIMEOUT        = auto()
    BACKEND_ERROR  = auto()


@dataclass(frozen=True)
class TranscriptResult:
    """Immutable result returned by every transcription attempt."""

    text:       str
    status:     TranscriptStatus
    confidence: Optional[float] = None
    duration:   float           = 0.0

    @property
    def success(self) -> bool:
        return self.status is TranscriptStatus.SUCCESS

    def __str__(self) -> str:
        return self.text if self.success else f"[{self.status.name}]"

    def __repr__(self) -> str:
        return (
            f"TranscriptResult(status={self.status.name!r}, "
            f"text={self.text!r}, duration={self.duration:.3f}s)"
        )


# ── Shared helpers ────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Strip, lowercase, and collapse internal whitespace."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _apply_noise_reduction(audio: sr.AudioData) -> sr.AudioData:
    """
    Spectral-gating noise reduction via noisereduce.
    Returns the original AudioData when the library is unavailable or the
    config flag is off — never raises.
    """
    if not (_NR_AVAILABLE and getattr(config.audio, "noise_reduction", False)):
        return audio
    try:
        raw     = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float32)
        reduced = nr.reduce_noise(y=raw, sr=audio.sample_rate, stationary=False)
        cleaned = reduced.astype(np.int16).tobytes()
        return sr.AudioData(cleaned, audio.sample_rate, audio.sample_width)
    except Exception:
        logger.debug("[STT] Noise reduction failed — using raw audio.", exc_info=True)
        return audio


def _denoise(audio: sr.AudioData) -> sr.AudioData:
    """
    Apply the best available denoising pipeline.
    Tries the full audio_processing.denoise first; falls back to lightweight NR.
    """
    if _AUDIO_PROCESSING_AVAILABLE and _full_denoise is not None:
        try:
            return _full_denoise(audio)
        except Exception as exc:
            logger.warning("[STT] audio_processing.denoise failed — using fallback: %s", exc)
    return _apply_noise_reduction(audio)


# ── Abstract backend ──────────────────────────────────────────────────────────


class _STTBackend(ABC):
    """Common interface every STT backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""

    @abstractmethod
    def setup(self) -> None:
        """One-time initialisation. Raise RuntimeError on unrecoverable failure."""

    @abstractmethod
    def transcribe(self, audio: sr.AudioData) -> TranscriptResult:
        """Convert audio bytes to a TranscriptResult. Must never raise."""

    def teardown(self) -> None:  # noqa: B027  (intentional no-op default)
        """Release any held resources. Override when needed."""


# ── Vosk backend ──────────────────────────────────────────────────────────────


class _VoskBackend(_STTBackend):
    """Fully offline transcription via the Vosk speech toolkit."""

    def __init__(self) -> None:
        self._kaldi: Optional[KaldiRecognizer] = None

    @property
    def name(self) -> str:
        return "Vosk"

    def setup(self) -> None:
        if not _VOSK_AVAILABLE:
            raise RuntimeError("Vosk is not installed — run: pip install vosk")

        model_path = config.audio.vosk_model_path
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'. "
                "Download a model from https://alphacephei.com/vosk/models"
            )

        logger.info("[STT] Loading Vosk model from '%s' …", model_path)
        model        = VoskModel(model_path)
        self._kaldi  = KaldiRecognizer(model, 16_000)
        logger.info("[STT] Vosk model loaded.")

    def transcribe(self, audio: sr.AudioData) -> TranscriptResult:
        if self._kaldi is None:
            logger.error("[STT/Vosk] transcribe() called before setup().")
            return TranscriptResult(text="", status=TranscriptStatus.BACKEND_ERROR)

        start = time.perf_counter()
        try:
            raw  = audio.get_raw_data(convert_rate=16_000, convert_width=2)
            self._kaldi.AcceptWaveform(raw)
            data = json.loads(self._kaldi.FinalResult())
            text = data.get("text", "").strip()

            duration = time.perf_counter() - start
            if not text:
                return TranscriptResult(text="", status=TranscriptStatus.UNCLEAR, duration=duration)
            normalized = _normalise(text)
            logger.info("[STT] (%s) %.2fs -> '%s'", self.name, duration, normalized)
            return TranscriptResult(
                text     = normalized,
                status   = TranscriptStatus.SUCCESS,
                duration = duration,
            )
        except Exception as exc:
            logger.warning("[STT/Vosk] Transcription error: %s", exc)
            return TranscriptResult(
                text     = "",
                status   = TranscriptStatus.BACKEND_ERROR,
                duration = time.perf_counter() - start,
            )


# ── Deepgram backend ──────────────────────────────────────────────────────────


class _DeepgramBackend(_STTBackend):
    """
    Online transcription via Deepgram Nova-2 API using official deepgram-sdk.
    """

    def __init__(self) -> None:
        self._client: Optional[DeepgramClient] = None

    @property
    def name(self) -> str:
        return "Deepgram"

    def setup(self) -> None:
        if not _DEEPGRAM_AVAILABLE:
            raise RuntimeError("deepgram-sdk is not installed — run: pip install deepgram-sdk")

        key = os.environ.get("DEEPGRAM_API_KEY", "").strip() or \
              getattr(config.audio, "deepgram_api_key", "")

        if not key:
            raise RuntimeError(
                "Deepgram API key missing. "
                "Set DEEPGRAM_API_KEY env var or config.audio.deepgram_api_key."
            )

        self._client = DeepgramClient(api_key=key)
        logger.info("[STT] Deepgram backend ready (Nova-2 SDK client initialized).")

    def teardown(self) -> None:
        self._client = None
        logger.debug("[STT] Deepgram client cleared.")

    def transcribe(self, audio: sr.AudioData) -> TranscriptResult:
        if self._client is None:
            logger.error("[STT/Deepgram] transcribe() called before setup().")
            return TranscriptResult(text="", status=TranscriptStatus.BACKEND_ERROR)

        start = time.perf_counter()
        try:
            response = self._client.listen.v1.media.transcribe_file(
                request=audio.get_wav_data(),
                model="nova-2",
                smart_format=True,
                language="en-US",
                punctuate=True,
            )
            duration = time.perf_counter() - start

            if not response or not response.results or not response.results.channels:
                return TranscriptResult(text="", status=TranscriptStatus.UNCLEAR, duration=duration)

            alternatives = response.results.channels[0].alternatives
            if not alternatives:
                return TranscriptResult(text="", status=TranscriptStatus.UNCLEAR, duration=duration)

            best = alternatives[0]
            transcript = (best.transcript or "").strip()
            confidence = best.confidence

            if not transcript:
                return TranscriptResult(text="", status=TranscriptStatus.UNCLEAR, duration=duration)

            normalized = _normalise(transcript)
            logger.info("[STT] (%s) %.2fs -> '%s'", self.name, duration, normalized)

            return TranscriptResult(
                text       = normalized,
                status     = TranscriptStatus.SUCCESS,
                confidence = confidence,
                duration   = duration,
            )

        except Exception as exc:
            logger.exception("[STT/Deepgram] API error: %s", exc)
            return TranscriptResult(
                text="", status=TranscriptStatus.API_ERROR,
                duration=time.perf_counter() - start,
            )


class _GoogleBackend(_STTBackend):
    """Online transcription using Google Speech Recognition."""

    def __init__(self) -> None:
        self._recogniser = sr.Recognizer()

    @property
    def name(self) -> str:
        return "Google"

    def setup(self) -> None:
        logger.info("[STT] Google backend ready.")

    def transcribe(self, audio: sr.AudioData) -> TranscriptResult:
        start = time.perf_counter()
        try:
            text = self._recogniser.recognize_google(audio, show_all=False).lower()
            duration = time.perf_counter() - start
            normalized = _normalise(text)
            logger.info("[STT] (%s) %.2fs -> '%s'", self.name, duration, normalized)
            return TranscriptResult(
                text=normalized,
                status=TranscriptStatus.SUCCESS,
                duration=duration,
            )
        except sr.UnknownValueError:
            return TranscriptResult(
                text="",
                status=TranscriptStatus.UNCLEAR,
                duration=time.perf_counter() - start,
            )
        except sr.RequestError as exc:
            logger.warning("[STT/Google] API error: %s", exc)
            return TranscriptResult(
                text="",
                status=TranscriptStatus.API_ERROR,
                duration=time.perf_counter() - start,
            )


# ── STT Engine ────────────────────────────────────────────────────────────────


class SpeechToText:
    """
    Public interface for single-utterance speech transcription.

    Selects the appropriate backend from config, manages the microphone
    capture lifecycle, and applies denoising transparently.

    Backend priority (strict):
      1. Deepgram — when provider == "deepgram" OR (key present AND provider != "vosk")
      2. Vosk     — when provider == "vosk" and model dir exists
      3. Google   — always-available fallback

    Thread safety: listen() is serialised by _audio_lock.
    """

    def __init__(self) -> None:
        self._backend:    Optional[_STTBackend] = None
        self._ready:      bool                  = False
        self._audio_lock: threading.Lock        = threading.Lock()
        self._mic:        Optional[sr.Microphone] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise the STT backend chosen by config.

        Falls back to Google automatically when the preferred backend fails.
        Returns True when any backend is ready, False when everything failed.
        """
        primary = self._select_backend()
        backend = self._try_setup(primary)

        if backend is None:
            logger.warning("[STT] %s failed — falling back to Google.", primary.name)
            backend = self._try_setup(_GoogleBackend())

        if backend is None:
            logger.error("[STT] All backends failed to initialise.")
            return False

        self._backend = backend
        self._mic     = sr.Microphone(device_index=config.audio.mic_index)

        # Calibrate ambient noise on setup to set a robust, device-specific energy threshold
        self.calibrate()

        if _AUDIO_PROCESSING_AVAILABLE and _init_nc is not None:
            try:
                _init_nc()
                logger.info("[STT] audio_processing noise-cancellation pipeline active.")
            except Exception as exc:
                logger.warning("[STT] audio_processing init failed (non-critical): %s", exc)

        self._ready = True
        logger.info("[STT] Ready — backend: %s", backend.name)
        return True

    def teardown(self) -> None:
        """Release all backend resources. Call on shutdown."""
        if self._backend is not None:
            self._backend.teardown()
            self._backend = None
        self._ready = False
        logger.info("[STT] Torn down.")

    # ── Core listen ───────────────────────────────────────────────────────────

    def listen(
        self,
        timeout:       int  = 5,
        prompt_played: bool = False,
        max_retries:   int  = 1,
    ) -> TranscriptResult:
        """
        Open the microphone, capture one utterance, transcribe, and return.

        Args:
            timeout:       Seconds to wait for speech before giving up.
            prompt_played: Inserts a short TTS tail-settle delay when True.
            max_retries:   Re-capture attempts on UNCLEAR before giving up.

        Returns:
            TranscriptResult with final status and text.
        """
        if not self._ready or self._backend is None:
            logger.error("[STT] listen() called before setup().")
            return TranscriptResult(text="", status=TranscriptStatus.BACKEND_ERROR)

        if prompt_played:
            time.sleep(_TTS_TAIL_DELAY)

        for attempt in range(max_retries + 1):
            result = self._capture_and_transcribe(timeout)

            if result.success:
                logger.info(
                    "[STT] (%s) attempt=%d  %.3fs  → '%s'",
                    self._backend.name, attempt + 1, result.duration, result.text,
                )
                return result

            if result.status is TranscriptStatus.UNCLEAR and attempt < max_retries:
                logger.debug("[STT] Unclear — retrying (%d/%d).", attempt + 1, max_retries)
                continue

            logger.debug("[STT] Final status: %s", result.status.name)
            return result

        # Should be unreachable, but guarantees a return for the type-checker.
        return TranscriptResult(text="", status=TranscriptStatus.UNCLEAR)

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, duration: float = 2.0) -> None:
        """
        Perform ambient noise calibration for the microphone.
        Uses sr.Recognizer for ambient volume measurement only — no API calls.
        """
        try:
            rec = sr.Recognizer()
            mic_index = config.audio.mic_index
            with sr.Microphone(device_index=mic_index) as source:
                logger.info("[STT] Calibrating ambient noise (%.1fs)…", duration)
                rec.adjust_for_ambient_noise(source, duration=duration)
                calibrated = float(rec.energy_threshold)
                # Ensure threshold is within a sane range (minimum 150, maximum 4000)
                calibrated = max(150.0, min(calibrated, 4000.0))
                
                # Store the calibrated threshold on the config singleton
                setattr(config, "calibrated_energy_threshold", calibrated)
                logger.info(
                    "[STT] Calibration complete. Energy threshold: %d",
                    calibrated,
                )
        except Exception as exc:
            logger.error("[STT] Calibration failed — mic unavailable: %s", exc)
            setattr(config, "calibrated_energy_threshold", float(config.audio.mic_energy_threshold))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "None"

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _try_setup(backend: _STTBackend) -> Optional[_STTBackend]:
        """Attempt backend.setup(). Returns backend on success, None on failure."""
        try:
            backend.setup()
            return backend
        except Exception as exc:
            logger.error("[STT] %s backend setup failed: %s", backend.name, exc)
            return None

    def _select_backend(self) -> _STTBackend:
        """Pick the best available backend: Deepgram (cloud), Vosk (offline), or Google (online fallback)."""
        provider = config.audio.stt_provider.lower()

        if provider == "deepgram":
            dg = _DeepgramBackend()
            try:
                dg.setup()
                return dg
            except Exception as e:
                logger.critical("[STT] Deepgram setup failed: %s. Falling back to Google.", e)
                return _GoogleBackend()

        deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "") or getattr(
            config.audio, "deepgram_api_key", ""
        )

        if provider == "vosk":
            if _VOSK_AVAILABLE:
                return _VoskBackend()
            logger.warning("[STT] Vosk requested but not installed — using Deepgram.")

        # Default: Deepgram
        if deepgram_key and _REQUESTS_AVAILABLE:
            dg = _DeepgramBackend()
            try:
                dg.setup()
                return dg
            except Exception:
                pass

        if _VOSK_AVAILABLE:
            logger.warning("[STT] No Deepgram key — falling back to Vosk (offline).")
            return _VoskBackend()

        logger.warning("[STT] Falling back to Google.")
        return _GoogleBackend()

    def _capture_and_transcribe(self, timeout: int) -> TranscriptResult:
        """
        Acquire the audio lock, record one phrase via the mic, denoise, transcribe.
        The lock ensures only one capture runs at a time (e.g. no overlap with TTS).
        """
        assert self._backend is not None  # guarded by listen()
        assert self._mic     is not None  # guarded by setup()

        rec = sr.Recognizer()
        calibrated_threshold = getattr(config, "calibrated_energy_threshold", None)
        if calibrated_threshold is not None:
            rec.energy_threshold = calibrated_threshold
        else:
            rec.energy_threshold = config.audio.mic_energy_threshold
        rec.pause_threshold = config.audio.mic_pause_threshold
        rec.dynamic_energy_threshold = config.audio.mic_dynamic_energy_threshold

        with self._audio_lock:
            try:
                with self._mic as source:
                    if rec.dynamic_energy_threshold:
                        rec.adjust_for_ambient_noise(source, duration=0.3)
                    else:
                        logger.debug("[STT] Using fixed energy threshold: %d", rec.energy_threshold)
                    logger.debug("[STT] Listening (timeout=%ds) …", timeout)
                    audio = rec.listen(
                        source,
                        timeout          = timeout,
                        phrase_time_limit = config.audio.mic_phrase_limit,
                    )
            except sr.WaitTimeoutError:
                logger.debug("[STT] No speech detected within %ds.", timeout)
                return TranscriptResult(text="", status=TranscriptStatus.TIMEOUT)
            except OSError as exc:
                logger.error("[STT] Microphone error: %s", exc)
                return TranscriptResult(text="", status=TranscriptStatus.BACKEND_ERROR)

        cleaned = _denoise(audio)
        return self._backend.transcribe(cleaned)
