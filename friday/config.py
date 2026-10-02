"""
F.R.I.D.A.Y. Configuration Module
Loads, validates, and exposes all environment-based settings.
"""

from __future__ import annotations

import os
import sys
import warnings

# Force HuggingFace offline mode unless online models are explicitly requested
if os.environ.get("FRIDAY_ONLINE_MODELS") not in ("1", "true", "True", "yes"):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# Silence noisy third-party warnings
_original_showwarning = warnings.showwarning

def _silenced_showwarning(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, (UserWarning, FutureWarning, DeprecationWarning)):
        return
    _original_showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = _silenced_showwarning

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── Path Setup ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class ConfigError(Exception):
    """Custom exception for missing required configuration."""

    pass


def _get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """Internal helper to fetch and cast environment variables."""
    value = os.getenv(key)
    if required and value is None:
        raise ConfigError(f"FATAL: Required environment variable '{key}' is not set.")
    return value if value is not None else default


def _cast_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = _get_env("NVIDIA_API_KEY", "ollama")
    base_url: str = _get_env("NVIDIA_BASE_URL", "http://localhost:11434/v1")
    model: str = _get_env("NVIDIA_MODEL", "qwen2.5:3b-instruct-q4_K_M")
    max_tokens: int = int(_get_env("LLM_MAX_TOKENS", 512))
    temperature: float = float(_get_env("LLM_TEMPERATURE", 0.7))
    timeout: float = float(_get_env("LLM_TIMEOUT", 60.0))


@dataclass(frozen=True)
class TTSConfig:
    tts_engine: str = _get_env("TTS_ENGINE", "pyttsx3").lower()
    tts_speed: float = float(_get_env("TTS_SPEED", 1.25))



@dataclass(frozen=True)
class AudioConfig:
    stt_provider: str = _get_env("STT_PROVIDER", "deepgram").lower()
    deepgram_api_key: str = _get_env("DEEPGRAM_API_KEY", "")
    vosk_model_path: str = _get_env("VOSK_MODEL_PATH", "models/vosk-model")
    vosk_wake_model_path: str = _get_env("VOSK_WAKE_MODEL_PATH", "models/vosk-model-small-en-us")
    wake_word: str = _get_env("WAKE_WORD", "friday").lower()
    wake_word_sensitivity: float = float(_get_env("WAKE_WORD_SENSITIVITY", 0.5))
    mic_energy_threshold: int = int(_get_env("MIC_ENERGY_THRESHOLD") or _get_env("STT_ENERGY_THRESHOLD") or 300)
    mic_pause_threshold: float = float(_get_env("MIC_PAUSE_THRESHOLD") or _get_env("STT_PAUSE_THRESHOLD") or 0.8)
    mic_phrase_limit: int = int(_get_env("MIC_PHRASE_LIMIT") or _get_env("STT_PHRASE_TIME_LIMIT") or 15)
    noise_reduction: bool = _cast_bool(_get_env("NOISE_REDUCTION_ENABLED", True))
    mic_dynamic_energy_threshold: bool = _cast_bool(
        _get_env("MIC_DYNAMIC_ENERGY_THRESHOLD") or _get_env("STT_DYNAMIC_ENERGY_THRESHOLD") or True
    )
    mic_index: int | None = (
        int(_get_env("MIC_DEVICE_INDEX")) if _get_env("MIC_DEVICE_INDEX") else None
    )
    speaker_index: int | None = (
        int(_get_env("SPEAKER_DEVICE_INDEX")) if _get_env("SPEAKER_DEVICE_INDEX") else None
    )


@dataclass(frozen=True)
class SecurityConfig:
    verification_enabled: bool = _cast_bool(_get_env("SPEAKER_VERIFICATION_ENABLED", True))
    voice_profile_path: str = _get_env("VOICE_PROFILE_PATH", "data/voice_profile.npy")
    speaker_threshold: float = float(_get_env("SPEAKER_THRESHOLD", 0.75))
    confirm_destructive: bool = _cast_bool(_get_env("CONFIRM_DESTRUCTIVE_ACTIONS", True))
    privacy_mode: bool = _cast_bool(_get_env("PRIVACY_MODE", False))


@dataclass(frozen=True)
class StorageConfig:
    memory_db: str = _get_env("MEMORY_DB_PATH", "data/memory.db")
    chroma_db: str = _get_env("CHROMA_DB_PATH", "data/chroma")
    assistant_db: str = _get_env("ASSISTANT_DB_PATH", "data/assistant.db")
    goals_db: str = _get_env("GOALS_DB_PATH", "data/goals.db")
    learning_db: str = _get_env("LEARNING_DB_PATH", "data/learning.db")
    max_history: int = int(_get_env("MAX_CONVERSATION_HISTORY", 20))

    # ChromaDB configuration (Priority 5: migration support)
    chroma_mode: str = _get_env("CHROMA_MODE", "persistent")        # "persistent" or "http"
    chroma_host: str = _get_env("CHROMA_HOST", "localhost")
    chroma_port: int = int(_get_env("CHROMA_PORT", 8000))
    chroma_hnsw_m: int = int(_get_env("CHROMA_HNSW_M", 16))
    chroma_hnsw_ef_search: int = int(_get_env("CHROMA_HNSW_EF_SEARCH", 100))
    chroma_batch_size: int = int(_get_env("CHROMA_BATCH_SIZE", 500))

    # Path objects
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    credentials_dir: Path = BASE_DIR / "credentials"
    models_dir: Path = BASE_DIR / "models"
    logs_dir: Path = BASE_DIR / "logs"


@dataclass(frozen=True)
class SandboxConfig:
    dir: Path = BASE_DIR / "data" / "sandbox"
    max_memory_mb: int = int(_get_env("SANDBOX_MAX_MEMORY_MB", 256))
    timeout_seconds: int = int(_get_env("SANDBOX_TIMEOUT_SECONDS", 30))
    max_cpu_percent: float = float(_get_env("SANDBOX_MAX_CPU_PERCENT", 80.0))
    cpu_sustained_seconds: int = int(_get_env("SANDBOX_CPU_SUSTAINED_SECONDS", 5))


@dataclass(frozen=True)
class SystemConfig:
    log_level: str = _get_env("LOG_LEVEL", "INFO").upper()
    log_file: str = str(BASE_DIR / "logs" / "friday.log")
    conv_timeout: int = int(_get_env("CONVERSATION_TIMEOUT", 30))
    monitor_cpu: bool = _cast_bool(_get_env("MONITOR_CPU", True))
    cpu_threshold: float = float(_get_env("CPU_ALERT_THRESHOLD", 85.0))
    monitor_screen: bool = _cast_bool(_get_env("MONITOR_SCREEN", False))
    screen_poll_interval: int = int(_get_env("SCREEN_POLL_INTERVAL", 60))
    auto_execute_goals: bool = _cast_bool(_get_env("AUTO_EXECUTE_GOALS", False))
    gmail_creds: str = _get_env("GMAIL_CREDENTIALS_PATH", "credentials/gmail_credentials.json")
    gmail_token: str = _get_env("GMAIL_TOKEN_PATH", "credentials/gmail_token.json")


class FridayConfig:
    """Central Configuration Singleton"""

    def __init__(self):
        try:
            self.llm = LLMConfig()
            self.tts = TTSConfig()
            self.audio = AudioConfig()
            self.security = SecurityConfig()
            self.storage = StorageConfig()
            self.sandbox = SandboxConfig()
            self.system = SystemConfig()

            if not self.audio.deepgram_api_key:
                logger.warning("[Config] DEEPGRAM_API_KEY is missing. Deepgram STT will not be available.")

            self._ensure_dirs()
        except ConfigError as e:
            print(f"[CONFIG] {e}")
            sys.exit(1)

    def _ensure_dirs(self):
        """Ensure all critical directories exist."""
        dirs = [
            self.storage.data_dir,
            self.storage.credentials_dir,
            self.storage.models_dir,
            self.storage.logs_dir,
            self.sandbox.dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Initialize the singleton instance
config = FridayConfig()
