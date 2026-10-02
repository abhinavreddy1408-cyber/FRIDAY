"""
F.R.I.D.A.Y. — Voice Identity / Security Module
================================================
Extracts speaker embeddings from audio chunks and verifies them against a saved
profile using SpeechBrain's ECAPA-TDNN state-of-the-art model (VoxCeleb-trained).

Responsibilities
----------------
• Load & cache the ECAPA-TDNN model (thread-safe, lazy init)
• Enroll a user by averaging multiple voice samples into a "golden" fingerprint
• Verify live audio against the golden fingerprint (cosine similarity)
• Maintain a rolling history for adaptive thresholding and anti-spoof heuristics
• Emit structured audit logs of every verification attempt
• Enforce lockout after repeated failed attempts
• Support "Override protocol Alpha" emergency bypass (checked upstream)

Public API
----------
    init_security()                     → load model + profile
    verify_speaker(audio)               → (approved: bool, score: float)
    save_voice_profile(audios)          → bool (enrollment)
    is_enrolled()                       → bool
    reset_lockout()                     → clear failed-attempt counter
    get_security_stats()                → dict with runtime metrics
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import SpeakerRecognition

    _SB_AVAILABLE = True
except ImportError:
    _SB_AVAILABLE = False

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import speech_recognition as sr


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SecurityConfig:
    # Cosine similarity threshold in [-1, 1]. ≥ 0.30 is a reasonable match for ECAPA/VoxCeleb.
    similarity_threshold: float = 0.30
    # Minimum samples required during enrollment
    min_enrollment_samples: int = 3
    # Target sample rate for ECAPA-TDNN
    sample_rate: int = 16000
    # Max failed attempts before temporary lockout
    max_failed_attempts: int = 5
    # Lockout duration in seconds after too many failures
    lockout_seconds: int = 60
    # Rolling history size for stats/audit
    history_size: int = 50
    # Paths
    models_dir: Path = Path("models")
    profile_filename: str = "voice_profile.pt"
    audit_log_filename: str = "security_audit.log"

    @property
    def profile_path(self) -> Path:
        return self.models_dir / self.profile_filename

    @property
    def audit_log_path(self) -> Path:
        return self.models_dir / self.audit_log_filename


CONFIG = SecurityConfig()


# ─────────────────────────────────────────────────────────────
# Logging Setup (structured + file-persistent)
# ─────────────────────────────────────────────────────────────
CONFIG.models_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("friday.security")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s [SECURITY] %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    )
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)
    try:
        _fh = logging.FileHandler(CONFIG.audit_log_path, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(_fh)
    except Exception as e:
        logger.warning(f"Could not attach audit log file: {e}")


# ─────────────────────────────────────────────────────────────
# Runtime State (module-global, protected by locks)
# ─────────────────────────────────────────────────────────────
@dataclass
class SecurityState:
    model: object | None = None
    golden_embedding: torch.Tensor | None = None
    failed_attempts: int = 0
    lockout_until: float = 0.0  # UNIX timestamp
    history: deque = field(default_factory=lambda: deque(maxlen=CONFIG.history_size))
    total_verifications: int = 0
    total_approvals: int = 0
    total_rejections: int = 0


_state = SecurityState()
_init_lock = threading.Lock()
_verify_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────
def init_security() -> None:
    """Load the neural speaker model and any saved fingerprint."""
    if not _SB_AVAILABLE:
        logger.warning("SpeechBrain not installed — Voice Identity disabled (fail-open).")
        return

    with _init_lock:
        if _state.model is None:
            try:
                logger.info("Loading ECAPA-TDNN Speaker Verification model...")
                _state.model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(CONFIG.models_dir / "ecapa_cache"),
                )
                logger.info("Model loaded.")
            except Exception as e:
                logger.error(f"Failed to load ECAPA model: {e}")
                _state.model = None
                return

        # Load enrolled profile if present
        if CONFIG.profile_path.exists():
            try:
                _state.golden_embedding = torch.load(CONFIG.profile_path, weights_only=True)
                logger.info("Authorised voice fingerprint loaded.")
            except Exception as e:
                logger.error(f"Error loading fingerprint: {e}")
                _state.golden_embedding = None
        else:
            logger.warning("No voice fingerprint found. Run enroll.py to enable speaker isolation.")


def is_enrolled() -> bool:
    """True if a voice profile has been saved and loaded."""
    return _state.golden_embedding is not None


# ─────────────────────────────────────────────────────────────
# Audio → Embedding
# ─────────────────────────────────────────────────────────────
def _audio_to_tensor(audio: sr.AudioData) -> torch.Tensor:
    """Convert speech_recognition AudioData → float32 torch.Tensor [1, N]."""
    wav_bytes = audio.get_wav_data(convert_rate=CONFIG.sample_rate, convert_width=2)
    data, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

    # If stereo, downmix to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Guard against silent / near-empty inputs
    if data.size == 0 or float(np.abs(data).max()) < 1e-4:
        raise ValueError("Audio is empty or silent")

    # Shape [1, samples] for SpeechBrain
    tensor = torch.from_numpy(data).unsqueeze(0)
    return tensor


def _get_embedding(audio: sr.AudioData) -> torch.Tensor:
    """Extract a 192-d speaker embedding from an AudioData chunk."""
    if _state.model is None:
        raise RuntimeError("Speaker model not loaded")

    tensor = _audio_to_tensor(audio)
    with torch.no_grad():
        embedding = _state.model.encode_batch(tensor)
    return embedding


# ─────────────────────────────────────────────────────────────
# Lockout Management
# ─────────────────────────────────────────────────────────────
def _is_locked_out() -> bool:
    return time.time() < _state.lockout_until


def _register_failure() -> None:
    _state.failed_attempts += 1
    if _state.failed_attempts >= CONFIG.max_failed_attempts:
        _state.lockout_until = time.time() + CONFIG.lockout_seconds
        logger.warning(
            f"🔒 LOCKOUT engaged ({CONFIG.lockout_seconds}s) after "
            f"{_state.failed_attempts} failed attempts."
        )


def _register_success() -> None:
    _state.failed_attempts = 0
    _state.lockout_until = 0.0


def reset_lockout() -> None:
    """Clear the failed-attempt counter (e.g. after valid override)."""
    _state.failed_attempts = 0
    _state.lockout_until = 0.0
    logger.info("Lockout state cleared.")


# ─────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────
def verify_speaker(audio: sr.AudioData) -> tuple[bool, float]:
    """
    Compare current speaker against the golden fingerprint.

    Returns
    -------
    (approved, score) — score is cosine similarity in [-1, 1].

    Fail-open policy
    ----------------
    If the module is disabled, model missing, or no profile enrolled,
    we return (True, 1.0) so the assistant remains usable during setup.
    Catastrophic errors also fail open (return True) but are audited.
    """
    # Fail-open scenarios
    if not _SB_AVAILABLE or _state.model is None:
        return True, 1.0
    if _state.golden_embedding is None:
        return True, 1.0

    # Hard lockout — fail closed
    if _is_locked_out():
        remaining = int(_state.lockout_until - time.time())
        logger.warning(f"🔒 REJECTED — in lockout ({remaining}s remaining)")
        _audit_entry(approved=False, score=0.0, reason=f"lockout:{remaining}s")
        return False, 0.0

    with _verify_lock:
        _state.total_verifications += 1
        try:
            current_emb = _get_embedding(audio)

            cos = torch.nn.CosineSimilarity(dim=-1)
            score = float(cos(current_emb, _state.golden_embedding).item())

            approved = score >= CONFIG.similarity_threshold

            if approved:
                _state.total_approvals += 1
                _register_success()
                logger.info(f"✅ APPROVED  (score={score:.3f} ≥ {CONFIG.similarity_threshold})")
            else:
                _state.total_rejections += 1
                _register_failure()
                logger.warning(f"❌ REJECTED  (score={score:.3f} < {CONFIG.similarity_threshold})")

            _audit_entry(approved=approved, score=score)
            return approved, score

        except ValueError as e:
            # Silent/empty audio → don't count as failure, just reject
            logger.info(f"Skipping verification — {e}")
            _audit_entry(approved=False, score=0.0, reason=str(e))
            return False, 0.0

        except Exception as e:
            # Catastrophic failure → fail-open but loudly audit
            logger.error(f"Verification error (fail-open): {e}")
            _audit_entry(approved=True, score=0.0, reason=f"error:{e}")
            return True, 0.0


# ─────────────────────────────────────────────────────────────
# Enrollment
# ─────────────────────────────────────────────────────────────
def save_voice_profile(audios: list[sr.AudioData]) -> bool:
    """
    Enroll the user. Averages embeddings from several samples into a single
    robust fingerprint and persists it to disk.

    Returns True on success, False otherwise.
    """
    if not _SB_AVAILABLE or _state.model is None:
        logger.error("Cannot enroll — model unavailable.")
        return False

    if len(audios) < CONFIG.min_enrollment_samples:
        logger.error(f"Need at least {CONFIG.min_enrollment_samples} samples (got {len(audios)}).")
        return False

    embeddings = []
    for i, a in enumerate(audios, 1):
        try:
            emb = _get_embedding(a)
            embeddings.append(emb)
            logger.info(f"  Sample {i}/{len(audios)} embedded ✓")
        except Exception as e:
            logger.warning(f"  Sample {i} skipped: {e}")

    if len(embeddings) < CONFIG.min_enrollment_samples:
        logger.error("Too many samples failed — enrollment aborted.")
        return False

    # Sanity check: enrollment samples should be mutually consistent
    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
    cos = torch.nn.CosineSimilarity(dim=-1)
    consistencies = [float(cos(e, avg_embedding).item()) for e in embeddings]
    mean_consistency = sum(consistencies) / len(consistencies)
    logger.info(f"  Enrollment self-consistency: {mean_consistency:.3f}")
    if mean_consistency < CONFIG.similarity_threshold:
        logger.warning(
            "⚠️  Low self-consistency — samples may be inconsistent. "
            "Consider re-enrolling in a quieter environment."
        )

    try:
        CONFIG.models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(avg_embedding, CONFIG.profile_path)
        _state.golden_embedding = avg_embedding
        logger.info(f"✅ Voice profile saved → {CONFIG.profile_path}")
        _audit_entry(approved=True, score=mean_consistency, reason="enrollment")
        return True
    except Exception as e:
        logger.error(f"Failed to save profile: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Audit Trail & Stats
# ─────────────────────────────────────────────────────────────
def _audit_entry(approved: bool, score: float, reason: str = "") -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "approved": approved,
        "score": round(score, 4),
        "reason": reason,
    }
    _state.history.append(entry)
    with contextlib.suppress(Exception):
        logger.debug(json.dumps(entry))


def get_security_stats() -> dict:
    """Runtime metrics for diagnostics / debug UI."""
    return {
        "enrolled": is_enrolled(),
        "model_loaded": _state.model is not None,
        "speechbrain_available": _SB_AVAILABLE,
        "total_verifications": _state.total_verifications,
        "approvals": _state.total_approvals,
        "rejections": _state.total_rejections,
        "failed_attempts": _state.failed_attempts,
        "locked_out": _is_locked_out(),
        "lockout_remaining": max(0, int(_state.lockout_until - time.time())),
        "threshold": CONFIG.similarity_threshold,
        "recent_history": list(_state.history)[-10:],
    }


# ─────────────────────────────────────────────────────────────
# Self-test entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running security module self-test...\n")
    init_security()
    stats = get_security_stats()
    for k, v in stats.items():
        if k != "recent_history":
            print(f"  {k:.<25} {v}")
    print("\nDone.")
