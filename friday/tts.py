"""
F.R.I.D.A.Y. Text-to-Speech Module
Converts text responses into natural speech output.

Supports three backends (tried in priority order):
  1. Kokoro     — local neural TTS via the kokoro package
  2. Edge TTS   — free Microsoft neural TTS via edge-tts
  3. pyttsx3    — offline SAPI/eSpeak fallback, zero-latency

Playback is non-blocking by default; the engine may await completion
explicitly when synchronisation is required.
"""

from __future__ import annotations

import contextlib
import io
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from friday import config

logger = logging.getLogger(__name__)


# ── Optional backend imports ──────────────────────────────────────────────────

try:
    import pythoncom

    _PYTHONCOM_AVAILABLE = True
except ImportError:
    _PYTHONCOM_AVAILABLE = False

try:
    import pyttsx3

    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False
    logger.warning("[TTS] pyttsx3 not installed — offline TTS disabled.")

try:
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    import numpy as np
    import sounddevice as sd

    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

try:
    import edge_tts

    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False

try:
    from kokoro import KPipeline

    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False


# ── Enums & Constants ─────────────────────────────────────────────────────────


class TTSStatus(Enum):
    SUCCESS = auto()
    EMPTY_INPUT = auto()
    BACKEND_ERROR = auto()
    PLAYBACK_ERROR = auto()
    QUOTA_EXCEEDED = auto()


class SpeechPriority(Enum):
    """Controls queuing behaviour when a new utterance arrives mid-speech."""

    LOW = 0  # Queue behind current speech
    NORMAL = 1  # Default
    HIGH = 2  # Interrupt current speech


# ── Text Preprocessing ────────────────────────────────────────────────────────

_ABBREVIATIONS: dict[str, str] = {
    "cpu": "C P U",
    "gpu": "G P U",
    "ram": "R A M",
    "url": "U R L",
    "api": "A P I",
    "os": "O S",
    "ui": "U I",
    "id": "I D",
    "ok": "okay",
    "friday": "Friday",
}


def _preprocess(text: str) -> str:
    """
    Sanitise and normalise text before synthesis.

    - Strips markdown formatting (bold, italic, code, headers)
    - Expands known abbreviations for cleaner pronunciation
    - Collapses excessive whitespace
    - Ensures the string ends with terminal punctuation
    """
    import re

    if not text:
        return ""

    # Strip markdown
    text = re.sub(r"#{1,6}\s*", "", text)  # headers
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold / italic
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)  # inline code / blocks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links → label
    text = re.sub(r"[-•·]\s+", "", text)  # bullet points

    # Expand abbreviations (word-boundary aware)
    for abbr, expansion in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", expansion, text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Ensure terminal punctuation for natural cadence
    if text and text[-1] not in ".!?":
        text += "."

    return text


# ── Abstract Backend ──────────────────────────────────────────────────────────


class _TTSBackend(ABC):
    """
    Common interface for TTS backend implementations.
    Each backend is responsible only for synthesis — playback is handled
    by the backend itself or delegated to a shared player.
    """

    @abstractmethod
    def setup(self) -> None:
        """One-time initialisation. Raise on unrecoverable failure."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes | None:
        """
        Convert text to audio bytes (MP3/WAV) without playing them.
        Returns None if synthesis fails.
        """

    @abstractmethod
    def play_bytes(self, data: bytes) -> bool:
        """Play raw audio data synchronously."""

    @abstractmethod
    def speak(self, text: str) -> TTSStatus:
        """
        Synthesise and play the given text synchronously.

        Args:
            text: Pre-processed speech string.

        Returns:
            TTSStatus indicating outcome.
        """

    @abstractmethod
    def stop(self) -> None:
        """Immediately halt any in-progress speech."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""


# ── Audio Playback Helpers ────────────────────────────────────────────────────


def _play_mp3_bytes(data: bytes) -> bool:
    """
    Play MP3 audio from a bytes buffer.
    Tries pygame first, then sounddevice+pydub, returns False if all fail.
    """
    # pygame path
    if _PYGAME_AVAILABLE:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44_100, size=-16, channels=1, buffer=512)
            buf = io.BytesIO(data)
            pygame.mixer.music.load(buf, "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            return True
        except Exception:
            logger.debug("[TTS] pygame playback failed.", exc_info=True)

    # pydub + sounddevice path
    if _SD_AVAILABLE:
        try:
            from pydub import AudioSegment

            segment = AudioSegment.from_mp3(io.BytesIO(data))
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            samples /= 2**15  # normalise int16 → float32
            sd.play(samples, samplerate=segment.frame_rate, device=config.audio.speaker_index)
            sd.wait()
            return True
        except Exception:
            logger.debug("[TTS] sounddevice playback failed.", exc_info=True)

    logger.error("[TTS] No audio playback backend available for MP3.")
    return False


def _play_wav_bytes(data: bytes) -> bool:
    """
    Play WAV audio from a bytes buffer.
    Tries sounddevice first, then pygame, returns False if all fail.
    """
    if _SD_AVAILABLE:
        try:
            import soundfile as sf

            audio_data, samplerate = sf.read(io.BytesIO(data), dtype="float32")
            sd.play(audio_data, samplerate=samplerate, device=config.audio.speaker_index)
            sd.wait()
            return True
        except Exception:
            logger.debug("[TTS] sounddevice WAV playback failed.", exc_info=True)

    if _PYGAME_AVAILABLE:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=24_000, size=-16, channels=1, buffer=512)
            buf = io.BytesIO(data)
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            return True
        except Exception:
            logger.debug("[TTS] pygame WAV playback failed.", exc_info=True)

    logger.error("[TTS] No audio playback backend available for WAV.")
    return False


# ── Edge TTS Backend ──────────────────────────────────────────────────────


class _EdgeTTSBackend(_TTSBackend):
    """
    Free Microsoft neural TTS via the edge-tts package.
    Produces high-quality MP3 audio using Azure Edge voices.
    Requires network access but no API key.
    """

    _VOICE = "en-GB-SoniaNeural"
    _FALLBACK_VOICE = "en-US-AriaNeural"

    def __init__(self) -> None:
        self._voice: str = self._VOICE

    @property
    def name(self) -> str:
        return "EdgeTTS"

    def setup(self) -> None:
        if not _EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts package is not installed.")
        # edge-tts is async — verify we can create a communicate object
        try:
            _ = edge_tts.Communicate("test", self._voice)
        except Exception:
            self._voice = self._FALLBACK_VOICE
            try:
                _ = edge_tts.Communicate("test", self._voice)
            except Exception as exc:
                raise RuntimeError(f"Edge TTS voice initialisation failed: {exc}") from exc
        logger.info("[TTS] EdgeTTS backend ready (voice: %s).", self._voice)

    def synthesize(self, text: str) -> bytes | None:
        from friday.async_runner import run_async

        async def _synth() -> bytes:
            rate_pct = int((getattr(config.tts, "tts_speed", 1.25) - 1.0) * 100)
            rate_str = f"{rate_pct:+d}%"
            comm = edge_tts.Communicate(text, self._voice, rate=rate_str)
            chunks: list[bytes] = []
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        try:
            return run_async(_synth(), timeout=30)
        except Exception as exc:
            logger.error("[TTS/EdgeTTS] Synthesis failure: %s", exc)
            return None

    def play_bytes(self, data: bytes) -> bool:
        return _play_mp3_bytes(data)

    def speak(self, text: str) -> TTSStatus:
        audio_bytes = self.synthesize(text)
        if not audio_bytes:
            return TTSStatus.BACKEND_ERROR
        success = self.play_bytes(audio_bytes)
        return TTSStatus.SUCCESS if success else TTSStatus.PLAYBACK_ERROR

    def stop(self) -> None:
        if _PYGAME_AVAILABLE:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
            except Exception:
                pass
        if _SD_AVAILABLE:
            with contextlib.suppress(Exception):
                sd.stop()


# ── Kokoro Backend ────────────────────────────────────────────────────────


class _KokoroBackend(_TTSBackend):
    """
    Local neural TTS via the kokoro package.
    Produces high-quality WAV audio entirely offline after initial model download.
    """

    _VOICES = ("af_heart", "af_sky")

    def __init__(self) -> None:
        self._pipeline = None
        self._voice: str = self._VOICES[0]
        self._stop_requested = False

    @property
    def name(self) -> str:
        return "Kokoro"

    def setup(self) -> None:
        if not _KOKORO_AVAILABLE:
            raise RuntimeError("kokoro package is not installed.")
        
        import os
        
        # Save current HF_HUB_OFFLINE state to restore later
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        
        # Try loading from local cache first to avoid slow network checks/ConnectionReset warnings
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            self._pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
            # Pre-load voice while in offline mode to cache it in-memory
            self._pipeline.load_voice(self._voice)
            logger.info("[TTS] Kokoro backend ready (voice: %s, loaded offline).", self._voice)
        except Exception as offline_exc:
            logger.debug("[TTS] Kokoro offline load failed, trying online: %s", offline_exc)
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline
                
            try:
                self._pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
                # Load voice in online mode so huggingface_hub can download it
                self._pipeline.load_voice(self._voice)
                logger.info("[TTS] Kokoro backend ready (voice: %s, loaded online).", self._voice)
            except Exception as exc:
                raise RuntimeError(f"Kokoro pipeline init failed: {exc}") from exc
        finally:
            # Restore the original state
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline

    def synthesize(self, text: str) -> bytes | None:
        if self._pipeline is None:
            return None
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile

            # Kokoro returns generator of (graphemes, phonemes, audio_tensor)
            chunks = []
            speed = getattr(config.tts, "tts_speed", 1.25)
            for _gs, _ps, audio in self._pipeline(text, voice=self._voice, speed=speed):
                if isinstance(audio, np.ndarray):
                    chunks.append(audio)
                elif hasattr(audio, "numpy"):
                    chunks.append(audio.numpy())
                else:
                    chunks.append(np.array(audio))
            
            if not chunks:
                return None
                
            # Concatenate all generated audio chunks
            full_audio = np.concatenate(chunks)
            
            # Convert float32 in [-1.0, 1.0] to 16-bit PCM WAV
            int_audio = (full_audio * 32767).astype(np.int16)
            
            buf = io.BytesIO()
            wavfile.write(buf, 24000, int_audio)
            buf.seek(0)
            return buf.read()
        except Exception as exc:
            logger.error("[TTS/Kokoro] Synthesis failure: %s", exc)
            return None

    def play_bytes(self, data: bytes) -> bool:
        return _play_wav_bytes(data)

    def speak(self, text: str) -> TTSStatus:
        if self._pipeline is None:
            return TTSStatus.BACKEND_ERROR

        import queue
        import threading
        import time
        import re
        import numpy as np
        import scipy.io.wavfile as wavfile

        # Split text into sentences and join with newlines to trigger Kokoro's default split_pattern (r'\n+')
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        text_with_newlines = "\n".join(sentences)

        self._stop_requested = False
        audio_queue = queue.Queue()
        stop_event = threading.Event()
        error_event = threading.Event()

        start_synth = time.perf_counter()
        logger.debug("[TTS/Kokoro] Starting streaming synthesis...")

        def _synthesize_worker():
            try:
                chunk_idx = 0
                # Iterate over the pipeline generator
                speed = getattr(config.tts, "tts_speed", 1.25)
                for _gs, _ps, audio in self._pipeline(text_with_newlines, voice=self._voice, speed=speed):
                    if stop_event.is_set() or self._stop_requested:
                        break
                    
                    if isinstance(audio, np.ndarray):
                        chunk = audio
                    elif hasattr(audio, "numpy"):
                        chunk = audio.numpy()
                    else:
                        chunk = np.array(audio)

                    # Convert chunk to WAV bytes immediately
                    int_audio = (chunk * 32767).astype(np.int16)
                    buf = io.BytesIO()
                    wavfile.write(buf, 24000, int_audio)
                    buf.seek(0)
                    audio_queue.put((chunk_idx, buf.read()))
                    logger.debug(
                        "[TTS/Kokoro] Chunk %d synthesized in %.4fs",
                        chunk_idx,
                        time.perf_counter() - start_synth,
                    )
                    chunk_idx += 1
            except Exception as exc:
                logger.error("[TTS/Kokoro] Streaming synthesis failure: %s", exc)
                error_event.set()
            finally:
                audio_queue.put(None)  # Sentinel to signal end of stream

        # Start the generator/synthesis in a background thread
        synth_thread = threading.Thread(target=_synthesize_worker, daemon=True)
        synth_thread.start()

        # Play chunks sequentially on the caller's thread
        playback_success = True
        try:
            while not self._stop_requested:
                if error_event.is_set():
                    playback_success = False
                    break
                
                try:
                    # Bounded block wait for the next chunk
                    item = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    break

                chunk_idx, chunk_bytes = item
                logger.debug(
                    "[TTS/Kokoro] Starting playback of chunk %d (elapsed: %.4fs)",
                    chunk_idx,
                    time.perf_counter() - start_synth,
                )

                # Play the chunk (blocking playback call)
                if not self.play_bytes(chunk_bytes):
                    playback_success = False
                    # Stop synthesis thread if playback fails
                    stop_event.set()
                    break
        except Exception as exc:
            logger.error("[TTS/Kokoro] Streaming playback exception: %s", exc)
            playback_success = False
            stop_event.set()
        finally:
            stop_event.set()  # Ensure synthesis worker stops if we exit early
            synth_thread.join()

        if error_event.is_set():
            return TTSStatus.BACKEND_ERROR
        return TTSStatus.SUCCESS if playback_success else TTSStatus.PLAYBACK_ERROR

    def stop(self) -> None:
        self._stop_requested = True
        if _SD_AVAILABLE:
            with contextlib.suppress(Exception):
                sd.stop()
        if _PYGAME_AVAILABLE:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
            except Exception:
                pass


# ── pyttsx3 Backend ───────────────────────────────────────────────────────────


class _Pyttsx3Backend(_TTSBackend):
    """Offline TTS using the pyttsx3 engine."""

    def __init__(self) -> None:
        self._engine: pyttsx3.Engine | None = None
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._done_event = threading.Event()
        self._setup_event = threading.Event()
        self._setup_success = False

    @property
    def name(self) -> str:
        return "pyttsx3"

    def _configure_engine(self, engine: pyttsx3.Engine) -> None:
        engine.setProperty("rate", 200)  # words per minute (faster and more natural)
        engine.setProperty("volume", 0.92)

        # Prefer a female voice if available
        voices = engine.getProperty("voices")
        female = next(
            (v for v in voices if "female" in v.name.lower() or "zira" in v.name.lower()),
            None,
        )
        if female:
            engine.setProperty("voice", female.id)
        elif voices:
            engine.setProperty("voice", voices[0].id)

    def setup(self) -> None:
        if not _PYTTSX3_AVAILABLE:
            raise RuntimeError("pyttsx3 is not installed.")

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="pyttsx3-thread")
        self._thread.start()
        self._setup_event.wait()
        if not self._setup_success:
            raise RuntimeError("pyttsx3 backend setup failed on background thread.")
        logger.info("[TTS] pyttsx3 backend ready (on dedicated thread).")

    def _run_loop(self) -> None:
        logger.debug("[TTS/pyttsx3] Dedicated thread started.")
        if _PYTHONCOM_AVAILABLE:
            try:
                pythoncom.CoInitialize()
            except Exception as exc:
                logger.error("[TTS/pyttsx3] CoInitialize error: %s", exc)

        try:
            self._engine = pyttsx3.init()
            self._configure_engine(self._engine)

            def on_word(name, location, length):
                if self._stop_requested:
                    logger.debug("[TTS/pyttsx3] Stop requested, stopping engine in callback")
                    self._engine.stop()

            def on_end(name, completed):
                logger.debug("[TTS/pyttsx3] Finished utterance callback (completed: %s)", completed)
                self._done_event.set()

            self._engine.connect("started-word", on_word)
            self._engine.connect("finished-utterance", on_end)
            self._setup_success = True
        except Exception as exc:
            logger.error("[TTS/pyttsx3] Engine initialization failed: %s", exc, exc_info=True)
            self._setup_success = False
        finally:
            self._setup_event.set()

        if not self._setup_success:
            if _PYTHONCOM_AVAILABLE:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
            return

        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break
                
                cmd, text = item
                if cmd == "speak":
                    self._stop_requested = False
                    self._done_event.clear()
                    try:
                        self._engine.say(text)
                        self._engine.runAndWait()
                    except Exception as exc:
                        logger.error("[TTS/pyttsx3] runAndWait exception: %s", exc, exc_info=True)
                    finally:
                        self._done_event.set()
            except Exception as exc:
                logger.error("[TTS/pyttsx3] Exception in thread loop: %s", exc, exc_info=True)

        self._engine = None
        if _PYTHONCOM_AVAILABLE:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass
        logger.debug("[TTS/pyttsx3] Dedicated thread loop stopped.")

    def synthesize(self, text: str) -> bytes | None:
        return None

    def play_bytes(self, data: bytes) -> bool:
        return False

    def speak(self, text: str) -> TTSStatus:
        if self._engine is None:
            return TTSStatus.BACKEND_ERROR

        self._queue.put(("speak", text))
        self._done_event.wait()
        return TTSStatus.SUCCESS

    def stop(self) -> None:
        self._stop_requested = True
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        self._done_event.set()


# ── TTS Engine ────────────────────────────────────────────────────────────────


@dataclass
class TextToSpeech:
    """
    Public interface for speech synthesis.

    Manages backend selection, fallback chain, and a non-blocking speech queue.
    The engine calls `speak()` or `speak_async()` — never interacts with
    backends directly.

    Backend priority: Kokoro → EdgeTTS → pyttsx3

    Usage:
        tts = TextToSpeech()
        tts.setup()
        tts.speak("Online and ready, boss.")
    """

    _primary: _TTSBackend | None = field(default=None, init=False, repr=False)
    _fallback: _TTSBackend | None = field(default=None, init=False, repr=False)
    _active_backend: _TTSBackend | None = field(default=None, init=False, repr=False)
    # Ordered fallback chain: each entry tried if the previous fails.
    _fallback_chain: list[_TTSBackend] = field(default_factory=list, init=False, repr=False)

    _speech_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _current_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _speaking: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _ready: bool = field(default=False, init=False, repr=False)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise backends and determine priority order.
        Respects config.tts.tts_engine as primary, with fallbacks.

        Returns:
            True if at least one backend is ready.
        """
        backends_map: dict[str, _TTSBackend] = {}

        # 1. Kokoro (local neural)
        if _KOKORO_AVAILABLE:
            backend = _KokoroBackend()
            try:
                backend.setup()
                backends_map["kokoro"] = backend
                logger.info("[TTS] Kokoro backend initialised.")
            except Exception as exc:
                logger.warning("[TTS] Kokoro setup failed: %s", exc)

        # 2. Edge TTS (free cloud neural)
        if _EDGE_TTS_AVAILABLE:
            backend = _EdgeTTSBackend()
            try:
                backend.setup()
                backends_map["edgetts"] = backend
                logger.info("[TTS] EdgeTTS backend initialised.")
            except Exception as exc:
                logger.warning("[TTS] EdgeTTS setup failed: %s", exc)

        # 3. pyttsx3 (offline SAPI fallback)
        if _PYTTSX3_AVAILABLE:
            backend = _Pyttsx3Backend()
            try:
                backend.setup()
                backends_map["pyttsx3"] = backend
                logger.info("[TTS] pyttsx3 backend initialised.")
            except Exception as exc:
                logger.warning("[TTS] pyttsx3 setup failed: %s", exc)

        if not backends_map:
            logger.error("[TTS] No TTS backend available — speech disabled.")
            return False

        # Determine priority order based on tts_engine preference
        preferred = config.tts.tts_engine
        default_order = ["kokoro", "edgetts", "pyttsx3"]
        if preferred in default_order:
            order = [preferred] + [b for b in default_order if b != preferred]
        else:
            order = default_order

        ready_backends = [backends_map[name] for name in order if name in backends_map]

        self._fallback_chain = ready_backends
        self._active_backend = ready_backends[0]
        self._primary = ready_backends[0]
        self._fallback = ready_backends[-1] if len(ready_backends) > 1 else None

        logger.info(
            "[TTS] Active backend: %s  |  Chain: %s",
            self._active_backend.name,
            " -> ".join(b.name for b in ready_backends),
        )

        self._ready = True
        return True

    def teardown(self) -> None:
        """Stop speech and release resources."""
        self.stop()
        self._primary = None
        self._fallback = None
        self._active_backend = None
        self._fallback_chain.clear()
        self._ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> bytes | None:
        """Synthesise text to bytes using the active backend."""
        if not self._ready or self._active_backend is None:
            return None
        processed = _preprocess(text)
        return self._active_backend.synthesize(processed)

    def play_bytes(self, data: bytes) -> bool:
        """Play pre-synthesised audio bytes."""
        if not self._ready or self._active_backend is None:
            return False
        return self._active_backend.play_bytes(data)

    def speak(self, text: str, priority: SpeechPriority = SpeechPriority.NORMAL) -> TTSStatus:
        """
        Synthesise and play text synchronously.

        High-priority speech interrupts any in-progress utterance.
        Normal/low priority waits for current speech to finish.

        Args:
            text:     Text to speak (markdown will be stripped).
            priority: Queuing behaviour.

        Returns:
            TTSStatus indicating outcome.
        """
        if not self._ready:
            logger.warning("[TTS] speak() called before setup().")
            return TTSStatus.BACKEND_ERROR

        processed = _preprocess(text)
        if not processed:
            return TTSStatus.EMPTY_INPUT

        if priority == SpeechPriority.HIGH:
            self.stop()

        with self._speech_lock:
            return self._synthesise(processed)

    def speak_async(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.NORMAL,
        on_complete: callable | None = None,
    ) -> threading.Thread:
        """
        Synthesise and play text on a background thread.

        Args:
            text:        Text to speak.
            priority:    Queuing behaviour.
            on_complete: Optional callback invoked after playback finishes.

        Returns:
            The daemon thread handling playback.
        """

        def _run() -> None:
            self._speaking.set()
            try:
                self.speak(text, priority)
            finally:
                self._speaking.clear()
                if on_complete:
                    try:
                        on_complete()
                    except Exception:
                        logger.debug("[TTS] on_complete callback raised.", exc_info=True)

        thread = threading.Thread(target=_run, daemon=True, name="tts-speak")
        self._current_thread = thread
        thread.start()
        return thread

    def stop(self) -> None:
        """Immediately halt any in-progress speech."""
        for backend in self._fallback_chain:
            try:
                backend.stop()
            except Exception:
                pass
        self._speaking.clear()

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """
        Block until current speech finishes.

        Args:
            timeout: Maximum seconds to wait (None = wait indefinitely).

        Returns:
            True if speech finished, False if timed out.
        """
        if self._current_thread and self._current_thread.is_alive():
            self._current_thread.join(timeout=timeout)
            return not self._current_thread.is_alive()
        return True

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def active_backend_name(self) -> str:
        return self._active_backend.name if self._active_backend else "None"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _synthesise(self, text: str) -> TTSStatus:
        """
        Attempt synthesis on the active backend, cascading through the
        fallback chain on failure.

        Args:
            text: Pre-processed text ready for synthesis.

        Returns:
            TTSStatus from the backend that succeeded (or final failure status).
        """
        backend = self._active_backend
        if backend is None:
            return TTSStatus.BACKEND_ERROR

        start = time.perf_counter()
        status = backend.speak(text)

        if status == TTSStatus.SUCCESS:
            elapsed = time.perf_counter() - start
            logger.debug("[TTS] '%s…' via %s in %.2fs", text[:40], backend.name, elapsed)
            return status

        # Active backend failed — cascade through remaining chain
        logger.warning(
            "[TTS] %s failed (%s) — cascading through fallback chain.",
            backend.name,
            status.name,
        )

        # Find current backend's position in the chain
        try:
            current_idx = next(
                i for i, b in enumerate(self._fallback_chain) if b is backend
            )
        except StopIteration:
            current_idx = -1

        backend_status = status

        # Try every backend after the current one
        for candidate in self._fallback_chain[current_idx + 1:]:
            logger.info("[TTS] Trying fallback: %s…", candidate.name)
            status = candidate.speak(text)
            if status == TTSStatus.SUCCESS:
                # Permanently switch to this backend for the session
                logger.warning(
                    "[TTS] Switching active backend from %s to %s for this session.",
                    backend.name,
                    candidate.name,
                )
                self._active_backend = candidate
                return status

        logger.error("[TTS] All backends in fallback chain failed.")
        return TTSStatus.BACKEND_ERROR
