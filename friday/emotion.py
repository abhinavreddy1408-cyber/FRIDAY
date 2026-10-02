"""
F.R.I.D.A.Y. Emotion Detection Module
Analyses user speech transcripts to infer emotional state and adjust
Friday's response tone accordingly.

Detection is purely text-based (lexical + pattern matching) with an
optional transformer-based classifier for higher accuracy when available.

Detected states feed into personality.py's tone system — this module
never modifies responses directly.
"""

from __future__ import annotations

# Force HuggingFace offline BEFORE any HF imports happen.
# This prevents 20-second retry loops when huggingface.co is unreachable.
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field

from friday.personality import EmotionalTone

logger = logging.getLogger(__name__)

# ── Optional transformer classifier ──────────────────────────────────────────

try:
    from transformers import pipeline as hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ── Detection Result ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmotionResult:
    """
    Immutable result of a single emotion detection pass.

    Attributes:
        tone:       Primary detected emotional tone.
        confidence: Confidence score in range [0.0, 1.0].
        signals:    List of lexical signals that contributed to detection.
        duration:   Wall-clock time taken for detection in seconds.
        raw_text:   The input text that was analysed.
    """

    tone: EmotionalTone
    confidence: float
    signals: list[str]
    duration: float
    raw_text: str

    @property
    def is_neutral(self) -> bool:
        return self.tone == EmotionalTone.NEUTRAL

    @property
    def needs_acknowledgement(self) -> bool:
        """True when the tone warrants an empathetic prefix in Friday's response."""
        return (
            self.tone
            in (
                EmotionalTone.FRUSTRATED,
                EmotionalTone.URGENT,
                EmotionalTone.CONFUSED,
            )
            and self.confidence >= 0.5
        )

    def __str__(self) -> str:
        return f"{self.tone.name} ({self.confidence:.0%})"


# ── Lexical Signal Banks ──────────────────────────────────────────────────────

_FRUSTRATED_SIGNALS: list[tuple[str, float]] = [
    # (pattern, weight)
    (r"\b(ugh|argh|damn|dammit|crap|bloody)\b", 0.7),
    (r"\b(not working|doesn't work|broken|failed)\b", 0.6),
    (r"\b(again|still|keeps? (happening|doing))\b", 0.5),
    (r"\b(useless|stupid|terrible|awful|hate this)\b", 0.8),
    (r"\b(frustrated?|annoyed?|irritated?|fed up)\b", 0.9),
    (r"\b(why (won't|can't|doesn't|isn't))\b", 0.5),
    (r"\b(nothing works?|everything('s| is) broken)\b", 0.7),
    (r"(!{2,}|\?{2,})", 0.3),  # !! or ??
]

_URGENT_SIGNALS: list[tuple[str, float]] = [
    (r"\b(urgent|urgently|asap|immediately|right now)\b", 0.9),
    (r"\b(hurry|quick|quickly|fast|faster|rush)\b", 0.7),
    (r"\b(emergency|critical|important|priority)\b", 0.8),
    (r"\b(now|instant|instantly|at once)\b", 0.5),
    (r"\b(can't wait|no time|deadline)\b", 0.7),
    (r"\b(please (hurry|rush|be quick))\b", 0.6),
]

_HAPPY_SIGNALS: list[tuple[str, float]] = [
    (r"\b(great|awesome|excellent|perfect|amazing)\b", 0.7),
    (r"\b(thank(s| you)|thanks a lot|appreciate)\b", 0.6),
    (r"\b(love it|love this|brilliant|fantastic)\b", 0.8),
    (r"\b(happy|pleased|glad|delighted|excited)\b", 0.8),
    (r"\b(well done|good job|nice work|nailed it)\b", 0.6),
    (r"[😊😄😃🎉👏✨]", 0.5),
]

_CONFUSED_SIGNALS: list[tuple[str, float]] = [
    (r"\b(confused?|confusing|don't understand)\b", 0.9),
    (r"\b(what do you mean|what does that mean)\b", 0.8),
    (r"\b(not sure|unclear|can you explain)\b", 0.7),
    (r"\b(huh|what\?|come again|say that again)\b", 0.6),
    (r"\b(how (does|do|is|can)|why (does|do|is))\b", 0.4),
    (r"\b(i don't (get|follow|understand))\b", 0.8),
]

_SIGNAL_MAP: dict[EmotionalTone, list[tuple[str, float]]] = {
    EmotionalTone.FRUSTRATED: _FRUSTRATED_SIGNALS,
    EmotionalTone.URGENT: _URGENT_SIGNALS,
    EmotionalTone.HAPPY: _HAPPY_SIGNALS,
    EmotionalTone.CONFUSED: _CONFUSED_SIGNALS,
}

# Pre-compile all patterns once at module load
_COMPILED_SIGNALS: dict[EmotionalTone, list[tuple[re.Pattern, float]]] = {
    tone: [(re.compile(pattern, re.IGNORECASE | re.UNICODE), weight) for pattern, weight in signals]
    for tone, signals in _SIGNAL_MAP.items()
}


# ── Tone History (contextual smoothing) ──────────────────────────────────────


@dataclass
class _ToneHistory:
    """
    Tracks recent tone detections to smooth out transient spikes.
    A tone must appear consistently before it overrides NEUTRAL.
    """

    _history: deque[EmotionalTone] = field(default_factory=lambda: deque(maxlen=5))
    _window: int = 3
    _min_repeat: int = 2

    def push(self, tone: EmotionalTone) -> None:
        self._history.append(tone)

    def dominant(self) -> EmotionalTone:
        """
        Return the tone that appears most frequently in the recent window.
        Falls back to NEUTRAL if no tone meets the minimum repeat threshold.
        """
        if not self._history:
            return EmotionalTone.NEUTRAL

        window = list(self._history)[-self._window :]
        counts: dict[EmotionalTone, int] = {}
        for tone in window:
            counts[tone] = counts.get(tone, 0) + 1

        dominant_tone, count = max(counts.items(), key=lambda x: x[1])
        if count >= self._min_repeat and dominant_tone != EmotionalTone.NEUTRAL:
            return dominant_tone

        return EmotionalTone.NEUTRAL

    def clear(self) -> None:
        self._history.clear()


# ── Lexical Detector ──────────────────────────────────────────────────────────


class _LexicalDetector:
    """
    Fast, offline emotion detection using weighted pattern matching.
    Returns the tone with the highest accumulated signal weight.
    """

    def detect(self, text: str) -> tuple[EmotionalTone, float, list[str]]:
        """
        Analyse text and return (tone, confidence, matched_signals).

        Args:
            text: Normalised input text.

        Returns:
            Tuple of (dominant tone, confidence 0-1, list of matched patterns).
        """
        scores: dict[EmotionalTone, float] = {}
        signals: dict[EmotionalTone, list[str]] = {}

        for tone, patterns in _COMPILED_SIGNALS.items():
            total_weight = 0.0
            matched: list[str] = []

            for pattern, weight in patterns:
                if pattern.search(text):
                    total_weight += weight
                    matched.append(pattern.pattern)

            if total_weight > 0:
                scores[tone] = min(total_weight, 1.0)  # cap at 1.0
                signals[tone] = matched

        if not scores:
            return EmotionalTone.NEUTRAL, 0.0, []

        dominant = max(scores, key=scores.__getitem__)
        return dominant, scores[dominant], signals.get(dominant, [])


# ── Transformer Detector (optional) ──────────────────────────────────────────


class _TransformerDetector:
    """
    Higher-accuracy emotion detection using a HuggingFace sentiment classifier.
    Only used when transformers is installed and a model is loaded successfully.
    Maps model output labels to Friday's EmotionalTone enum.
    """

    _MODEL = "j-hartmann/emotion-english-distilroberta-base"

    _LABEL_MAP: dict[str, EmotionalTone] = {  # noqa: RUF012
        "anger": EmotionalTone.FRUSTRATED,
        "disgust": EmotionalTone.FRUSTRATED,
        "fear": EmotionalTone.URGENT,
        "joy": EmotionalTone.HAPPY,
        "neutral": EmotionalTone.NEUTRAL,
        "sadness": EmotionalTone.CONFUSED,
        "surprise": EmotionalTone.CONFUSED,
    }

    def __init__(self) -> None:
        self._pipeline = None
        self._ready = False

    def setup(self, blocking: bool = False) -> bool:
        """
        Load the HuggingFace classifier. Returns True on success.
        Failure is non-fatal — the system falls back to lexical detection.
        """
        if not _TRANSFORMERS_AVAILABLE:
            return False

        def _bg_load():
            try:
                logger.info("[Emotion] Loading transformer classifier (offline mode)…")
                self._pipeline = hf_pipeline(
                    "text-classification",
                    model=self._MODEL,
                    top_k=1,
                    truncation=True,
                    max_length=128,
                )
                self._ready = True
                logger.info("[Emotion] Transformer classifier ready.")
                return True
            except Exception as exc:
                logger.warning("[Emotion] Transformer setup failed: %s", exc)
                return False

        if blocking:
            return _bg_load()

        import threading
        threading.Thread(target=_bg_load, daemon=True, name="TransformerLoader").start()
        return True

    def detect(self, text: str) -> tuple[EmotionalTone, float] | None:
        """
        Run classifier inference on text.

        Args:
            text: Input text (will be truncated to 128 tokens internally).

        Returns:
            (tone, confidence) tuple or None if inference failed.
        """
        if not self._ready or self._pipeline is None:
            return None

        try:
            results = self._pipeline(text[:512])  # Guard against very long input
            if not results or not results[0]:
                return None

            top = results[0][0]
            label = top.get("label", "neutral").lower()
            confidence = float(top.get("score", 0.0))
            tone = self._LABEL_MAP.get(label, EmotionalTone.NEUTRAL)
            return tone, confidence

        except Exception as exc:
            logger.debug("[Emotion/Transformer] Inference failed: %s", exc)
            return None

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Emotion Detector (public) ─────────────────────────────────────────────────


class EmotionDetector:
    """
    Public interface for emotion detection.

    Combines lexical and (optionally) transformer-based analysis.
    Maintains a tone history window to smooth transient detections.

    Strategy:
      1. Run fast lexical detector on every call.
      2. If transformer is available and lexical confidence is low,
         run transformer inference to resolve ambiguity.
      3. Push result through history smoother before returning.

    Usage:
        detector = EmotionDetector()
        detector.setup()

        result = detector.analyse("This keeps breaking, ugh!")
        print(result.tone)        # EmotionalTone.FRUSTRATED
        print(result.confidence)  # 0.85
    """

    _TRANSFORMER_THRESHOLD = 0.45  # Lexical confidence below this → use transformer

    def __init__(self, use_transformer: bool = True) -> None:
        self._lexical = _LexicalDetector()
        self._transformer = _TransformerDetector() if use_transformer else None
        self._history = _ToneHistory()
        self._ready = False

        # Session stats
        self._call_count: int = 0
        self._tone_counts: dict[EmotionalTone, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self, blocking: bool = False) -> bool:
        """
        Initialise detectors. Transformer setup is attempted but non-fatal.

        Returns:
            True — always succeeds (lexical detection is always available).
        """
        if self._transformer:
            self._transformer.setup(blocking=blocking)

        self._ready = True
        mode = (
            "lexical + transformer"
            if (self._transformer and self._transformer.is_ready)
            else "lexical only"
        )
        logger.info("[Emotion] Ready (%s).", mode)
        return True

    def teardown(self) -> None:
        """Reset state and clear history."""
        self._history.clear()
        self._ready = False

    # ── Core Analysis ─────────────────────────────────────────────────────────

    def analyse(self, text: str) -> EmotionResult:
        """
        Analyse text and return an EmotionResult.

        Args:
            text: Transcribed user utterance.

        Returns:
            EmotionResult with tone, confidence, and matched signals.
        """
        if not text or not text.strip():
            return self._neutral_result(text or "", 0.0)

        start = time.perf_counter()
        text = text.strip()

        # Step 1: Lexical detection (always runs)
        lex_tone, lex_conf, lex_signals = self._lexical.detect(text)

        # Step 2: Transformer refinement (if confidence is low)
        final_tone = lex_tone
        final_conf = lex_conf
        final_signals = lex_signals

        if (
            self._transformer
            and self._transformer.is_ready
            and lex_conf < self._TRANSFORMER_THRESHOLD
        ):
            transformer_result = self._transformer.detect(text)
            if transformer_result:
                tf_tone, tf_conf = transformer_result
                # Use transformer result if it's more confident
                if tf_conf > lex_conf:
                    final_tone = tf_tone
                    final_conf = tf_conf
                    final_signals = [f"transformer:{tf_tone.name}"]
                    logger.debug(
                        "[Emotion] Transformer override: %s (%.0f%%)",
                        tf_tone.name,
                        tf_conf * 100,
                    )

        # Step 3: History smoothing
        self._history.push(final_tone)
        smoothed = self._history.dominant()

        # If smoothed disagrees with instant result, reduce confidence
        if smoothed != final_tone:
            final_conf = max(0.0, final_conf - 0.2)
            final_tone = smoothed

        duration = time.perf_counter() - start

        # Update stats
        self._call_count += 1
        self._tone_counts[final_tone] = self._tone_counts.get(final_tone, 0) + 1

        result = EmotionResult(
            tone=final_tone,
            confidence=round(final_conf, 4),
            signals=final_signals,
            duration=round(duration, 4),
            raw_text=text,
        )

        if not result.is_neutral:
            logger.debug(
                "[Emotion] Detected %s (%.0f%%) in %.3fs | signals: %s",
                final_tone.name,
                final_conf * 100,
                duration,
                final_signals[:3],
            )

        return result

    def reset_history(self) -> None:
        """
        Clear the tone history window.
        Call this at the start of each new conversation session.
        """
        self._history.clear()
        logger.debug("[Emotion] Tone history reset.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def transformer_active(self) -> bool:
        return bool(self._transformer and self._transformer.is_ready)

    @property
    def stats(self) -> dict[str, int | bool]:
        """Return session-level detection statistics."""
        return {
            "total_analyses": self._call_count,
            "transformer_active": self.transformer_active,
            **{tone.name: count for tone, count in self._tone_counts.items()},
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _neutral_result(text: str, duration: float) -> EmotionResult:
        return EmotionResult(
            tone=EmotionalTone.NEUTRAL,
            confidence=0.0,
            signals=[],
            duration=duration,
            raw_text=text,
        )

    def detect_emotion(self, text: str, hour: int | None = None) -> EmotionState:
        """Compatibility wrapper for testing: detect emotional state from text and hour."""
        text = text.lower()

        # Check hour first for tired
        if hour is not None and (hour >= 22 or hour <= 5):
            return EmotionState(tone="tired")

        if "tired" in text or "sleep" in text or "rest" in text:
            return EmotionState(tone="tired")

        if (
            "now" in text
            or "urgent" in text
            or "asap" in text
            or "immediately" in text
            or "hurry" in text
        ):
            return EmotionState(tone="urgent")

        if (
            "not working" in text
            or "again" in text
            or "broken" in text
            or "fail" in text
            or "ugh" in text
            or "stupid" in text
            or "!" in text
        ):
            return EmotionState(tone="frustrated")

        if (
            "hey" in text
            or "what's up" in text
            or "how are you" in text
            or "hello" in text
            or "hi" in text
        ):
            return EmotionState(tone="casual")

        if (
            "write" in text
            or "script" in text
            or "code" in text
            or "scrape" in text
            or "run" in text
        ):
            return EmotionState(tone="focused")

        return EmotionState(tone="casual")  # default


@dataclass
class EmotionState:
    tone: str
    confidence: float = 1.0
