"""
F.R.I.D.A.Y. — Audio Processing Module
=======================================
Real-time audio pre-processing pipeline that cleans captured speech before it
reaches the STT / speaker-verification stages.

Pipeline (each stage is optional and independently toggleable):
    raw audio
      └─► DC-offset removal
            └─► pre-emphasis filter (optional)
                  └─► spectral-gate noise reduction (noisereduce)
                        └─► RMS normalisation (loudness target)
                              └─► soft-clip peak limiter
                                    └─► clean AudioData → STT

Key features
------------
• Thread-safe stationary-noise profile (captured once during ambient calibration,
  then re-used so the first 0.3 s of every utterance isn't wasted).
• Fail-open design: any pipeline stage may throw, and the original audio is
  returned so the assistant never silently breaks.
• Detailed metrics (SNR estimate, peak, RMS, duration) per invocation, accessible
  via `get_audio_stats()` for debug UI / tray tooltips.
• Structured logging through `friday.audio` logger (mirrors `security.py` style).
• Tunable via a single `AudioConfig` dataclass — no scattered magic numbers.

Public API
----------
    init_noise_cancellation()       → load libs + reset metrics
    calibrate_noise(audio)          → learn ambient profile from a silent clip
    denoise(audio)                  → main entry point (full pipeline)
    is_speech(audio)                → lightweight energy-based VAD
    get_audio_stats()               → runtime metrics dict
"""

from __future__ import annotations

import os
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import speech_recognition as sr

os.environ.setdefault("NUMPY_EXPERIMENTAL_ARRAY_FUNCTION", "0")

try:
    import noisereduce as nr

    _NR_AVAILABLE = True
except ImportError:
    _NR_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AudioConfig:
    # Noise reduction
    prop_decrease: float = 0.85  # 0.0 → 1.0 — aggressiveness of NR
    n_std_thresh_stationary: float = 1.5
    use_stationary: bool = True  # stationary vs non-stationary algorithm

    # Pre-emphasis (boosts high frequencies → better consonant clarity for STT)
    pre_emphasis: float = 0.97  # set to 0.0 to disable
    apply_pre_emphasis: bool = True

    # Normalisation
    target_rms: float = 0.12  # desired RMS amplitude (0.0-1.0)
    max_gain_db: float = 20.0  # safety cap so whispers don't blow up
    apply_normalization: bool = True

    # Soft-clip limiter — prevents harsh clipping after gain
    apply_soft_clip: bool = True
    soft_clip_threshold: float = 0.95

    # VAD (simple energy-based)
    vad_rms_threshold: float = 0.008

    # Metrics history
    history_size: int = 50


CONFIG = AudioConfig()


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger("friday.audio")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s [AUDIO] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)


# ─────────────────────────────────────────────────────────────
# Runtime state
# ─────────────────────────────────────────────────────────────
@dataclass
class AudioState:
    noise_profile: np.ndarray | None = None  # learned ambient floor
    noise_profile_sr: int = 0
    total_processed: int = 0
    total_failed: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=CONFIG.history_size))


_state = AudioState()
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# Low-level DSP helpers
# ─────────────────────────────────────────────────────────────
def _pcm16_to_float(data: np.ndarray) -> np.ndarray:
    """int16 PCM → float32 in [-1.0, 1.0]"""
    return data.astype(np.float32) / 32768.0


def _float_to_pcm16(data: np.ndarray) -> np.ndarray:
    """float32 in [-1.0, 1.0] → int16 PCM (clipped)"""
    return np.clip(data * 32768.0, -32768, 32767).astype(np.int16)


def _remove_dc_offset(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def _pre_emphasis(x: np.ndarray, alpha: float) -> np.ndarray:
    """y[n] = x[n] - alpha * x[n-1]   (boosts highs, helps STT accuracy)"""
    if alpha <= 0.0:
        return x
    return np.append(x[0], x[1:] - alpha * x[:-1])


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _normalize_rms(x: np.ndarray, target: float, max_gain_db: float) -> np.ndarray:
    cur = _rms(x)
    if cur < 1e-6:
        return x  # silence — don't amplify hiss
    gain = target / cur
    # Cap the gain so whispers / near-silence don't blow up
    max_gain = 10 ** (max_gain_db / 20.0)
    gain = min(gain, max_gain)
    return x * gain


def _soft_clip(x: np.ndarray, threshold: float) -> np.ndarray:
    """tanh-style soft limiter — smoother than hard clipping."""
    return np.tanh(x / threshold) * threshold


def _estimate_snr_db(clean: np.ndarray, original: np.ndarray) -> float:
    """Rough SNR estimate comparing clean signal vs (original - clean) as noise."""
    noise = original.astype(np.float32) - clean.astype(np.float32)
    sig_power = float(np.mean(clean.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if noise_power < 1e-12 or sig_power < 1e-12:
        return 99.0
    return 10.0 * float(np.log10(sig_power / noise_power))


# ─────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────
def init_noise_cancellation() -> None:
    """Prepare the audio pipeline. Safe to call multiple times."""
    with _lock:
        _state.total_processed = 0
        _state.total_failed = 0
        _state.history.clear()

    if not _NR_AVAILABLE:
        logger.warning(
            "'noisereduce' not installed — spectral gating disabled "
            "(other DSP stages still active)."
        )
    else:
        logger.info(
            "Noise cancellation module ready "
            f"(prop_decrease={CONFIG.prop_decrease}, "
            f"stationary={CONFIG.use_stationary})."
        )


# ─────────────────────────────────────────────────────────────
# Ambient-noise calibration
# ─────────────────────────────────────────────────────────────
def calibrate_noise(audio: sr.AudioData) -> bool:
    """
    Learn the ambient noise profile from a silent clip (recommended ~1 s of
    background silence at startup). Subsequent denoise() calls will use this
    profile via noisereduce's `y_noise` parameter for better results than
    letting it auto-estimate from the first 0.3 s of each utterance.
    """
    if not _NR_AVAILABLE:
        return False
    try:
        raw = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
        if raw.size == 0:
            logger.warning("Calibration skipped — empty audio.")
            return False
        with _lock:
            _state.noise_profile = raw.copy()
            _state.noise_profile_sr = audio.sample_rate
        logger.info(
            f"Ambient noise profile captured "
            f"({raw.size / audio.sample_rate:.2f}s @ {audio.sample_rate}Hz)."
        )
        return True
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Voice Activity Detection (lightweight)
# ─────────────────────────────────────────────────────────────
def is_speech(audio: sr.AudioData) -> bool:
    """Quick energy-based VAD. Cheap; not a replacement for WebRTC VAD."""
    try:
        raw = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
        if raw.size == 0:
            return False
        rms = _rms(_pcm16_to_float(raw))
        return rms >= CONFIG.vad_rms_threshold
    except Exception:
        return True  # fail-open — let STT decide


# ─────────────────────────────────────────────────────────────
# Main denoising pipeline
# ─────────────────────────────────────────────────────────────
def denoise(audio: sr.AudioData) -> sr.AudioData:
    """
    Run the full pre-processing pipeline on captured audio and return a new
    AudioData object. On any failure, returns the original audio unchanged.
    """
    start = time.perf_counter()

    try:
        raw_bytes = audio.get_raw_data()
        if not raw_bytes:
            logger.debug("Empty audio — passing through.")
            return audio

        pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
        if pcm16.size == 0:
            return audio

        # Convert to float for DSP
        x = _pcm16_to_float(pcm16)
        original_float = x.copy()

        # 1) DC-offset removal
        x = _remove_dc_offset(x)

        # 2) Pre-emphasis (optional)
        if CONFIG.apply_pre_emphasis:
            x = _pre_emphasis(x, CONFIG.pre_emphasis)

        # 3) Spectral gating noise reduction
        if _NR_AVAILABLE:
            # Use calibrated profile if available & compatible
            y_noise = None
            if _state.noise_profile is not None and _state.noise_profile_sr == audio.sample_rate:
                y_noise = _pcm16_to_float(_state.noise_profile)

            x = nr.reduce_noise(
                y=x,
                sr=audio.sample_rate,
                y_noise=y_noise,
                prop_decrease=CONFIG.prop_decrease,
                n_std_thresh_stationary=CONFIG.n_std_thresh_stationary,
                stationary=CONFIG.use_stationary,
            ).astype(np.float32)

        # 4) Normalisation to target RMS (post-NR so we normalise the clean signal)
        if CONFIG.apply_normalization:
            x = _normalize_rms(x, CONFIG.target_rms, CONFIG.max_gain_db)

        # 5) Soft-clip limiter (prevents harsh clipping after gain)
        if CONFIG.apply_soft_clip:
            x = _soft_clip(x, CONFIG.soft_clip_threshold)

        # Back to int16 bytes
        clean_pcm16 = _float_to_pcm16(x)
        clean_audio = sr.AudioData(
            clean_pcm16.tobytes(),
            audio.sample_rate,
            audio.sample_width,
        )

        # Metrics
        elapsed = time.perf_counter() - start
        snr = _estimate_snr_db(x, original_float)
        peak = float(np.max(np.abs(x)))
        rms = _rms(x)

        with _lock:
            _state.total_processed += 1
            _state.history.append(
                {
                    "duration_s": round(pcm16.size / audio.sample_rate, 3),
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "snr_db": round(snr, 1),
                    "peak": round(peak, 3),
                    "rms": round(rms, 4),
                }
            )

        logger.info(
            f"Cleaned  ({pcm16.size / audio.sample_rate:.2f}s, "
            f"SNR≈{snr:+.1f}dB, peak={peak:.2f}, rms={rms:.3f}, "
            f"{elapsed * 1000:.1f}ms)"
        )
        return clean_audio

    except Exception as e:
        with _lock:
            _state.total_failed += 1
        logger.error(f"Denoising failed — passing original audio: {e}")
        return audio


# ─────────────────────────────────────────────────────────────
# Stats / diagnostics
# ─────────────────────────────────────────────────────────────
def get_audio_stats() -> dict:
    """Runtime metrics for debug UI / tray tooltip."""
    with _lock:
        hist = list(_state.history)

    avg_ms = (sum(h["elapsed_ms"] for h in hist) / len(hist)) if hist else 0.0
    avg_snr = (sum(h["snr_db"] for h in hist) / len(hist)) if hist else 0.0

    return {
        "noisereduce_available": _NR_AVAILABLE,
        "calibrated": _state.noise_profile is not None,
        "total_processed": _state.total_processed,
        "total_failed": _state.total_failed,
        "avg_latency_ms": round(avg_ms, 1),
        "avg_snr_db": round(avg_snr, 1),
        "config": {
            "prop_decrease": CONFIG.prop_decrease,
            "stationary": CONFIG.use_stationary,
            "pre_emphasis": CONFIG.apply_pre_emphasis,
            "normalization": CONFIG.apply_normalization,
            "soft_clip": CONFIG.apply_soft_clip,
        },
        "recent": hist[-5:],
    }


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running audio_processing self-test...\n")
    init_noise_cancellation()

    # Generate a synthetic noisy sine wave to prove the pipeline runs
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * np.random.randn(t.size)
    noisy = _float_to_pcm16(tone + noise)

    test_audio = sr.AudioData(noisy.tobytes(), sample_rate, 2)
    print(f"is_speech? → {is_speech(test_audio)}")
    cleaned = denoise(test_audio)
    print(f"Cleaned sample bytes: {len(cleaned.get_raw_data())}\n")

    stats = get_audio_stats()
    for k, v in stats.items():
        if k not in ("recent", "config"):
            print(f"  {k:.<25} {v}")
    print("\nDone.")
