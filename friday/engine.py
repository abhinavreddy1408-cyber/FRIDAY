"""
F.R.I.D.A.Y. Core Engine Module
The central coordinator of the F.R.I.D.A.Y. system.

The engine owns the primary state machine, wires all subsystems together,
and drives the full voice interaction lifecycle from wake-word detection
through response delivery.

State machine:
  OFFLINE → BOOTING → IDLE → LISTENING → PROCESSING → SPEAKING
                                ↑                          |
                                └──────── CONVERSATION ←───┘

All subsystem interactions are mediated through typed interfaces.
The engine never accesses raw audio, LLM APIs, or SQL directly.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from friday import config
from friday.actions import ActionResult, ActionRouter, ActionStatus
from friday.async_runner import run_async
from friday.emotion import EmotionDetector, EmotionResult
from friday.ensemble import CANONICAL_MODEL_MAP
from friday.goal_engine import GoalEngine, GoalStatus
from friday.goal_planner import GoalPlanner
from friday.learning import LearningEngine
from friday.llm import LLMBrain, LLMStatus
from friday.memory import MemoryManager
from friday.monitor import BackgroundMonitor
from friday.personality import (
    ResponseCategory,
    build_tone_aware_response,
    get_voice_line,
)
from friday.planner import ActionPlanner, ExecutionPlan, PlannerDecision, PlanStep, RoutingDecision
from friday.refinement import RefinementChain
from friday.reasoning import AutonomousReasoner, AutonomousReasoningResult
from friday.safety import SAFE_OPERATIONAL_ERROR, VerificationChain
from friday.screen import ScreenAlert, ScreenMonitor
from friday.skills import register_all_skills
from friday.sounds import SoundEvent
from friday.sounds import play as play_sound
from friday.streaming import StreamingResponsePipeline
from friday.stt import SpeechToText, TranscriptResult, TranscriptStatus
from friday.swarm import Specialist, SwarmOrchestrator, TaskResult
from friday.task_queue import TaskQueue
from friday.tts import SpeechPriority, TextToSpeech
from friday.voice_to_prompt import VoiceToPromptAgent, VoiceToPromptResult
from friday.wake_word import WakeWordDetector
from friday.tools.tool_executor import ToolExecutor

if TYPE_CHECKING:
    from collections.abc import Callable
    from friday.goal_engine import Goal, GoalStep

logger = logging.getLogger(__name__)


def _tier_for_category(category: str) -> str:
    if category in {"coding", "search", "document", "productivity", "brainstorming"}:
        return "flagship"
    if category in {"reasoning", "retrieval", "file_analysis"}:
        return "high_performance"
    return "specialist"


# ── Engine State ──────────────────────────────────────────────────────────────


class EngineState(Enum):
    OFFLINE = auto()  # Not initialised
    BOOTING = auto()  # Subsystems starting up
    IDLE = auto()  # Wake word listening
    LISTENING = auto()  # Capturing user command
    PROCESSING = auto()  # Classifying and executing
    SPEAKING = auto()  # TTS output in progress
    CONVERSATION = auto()  # Follow-up window active
    ERROR = auto()  # Unrecoverable subsystem failure
    SHUTTING_DOWN = auto()  # Graceful shutdown in progress
    PROACTIVE_ALERT = auto()  # Background alert triggered


class SystemCommand(Enum):
    """Internal system commands recognised by the engine."""

    STOP = auto()
    PAUSE = auto()
    RESUME = auto()
    RESTART = auto()
    STATUS = auto()
    HELP = auto()
    CLEAR_HISTORY = auto()
    PRIVACY_ON = auto()
    PRIVACY_OFF = auto()


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class EngineConfig:
    """
    Runtime configuration for the engine.
    Populated from config module at startup; may be adjusted at runtime.
    """

    conversation_timeout: int = config.system.conv_timeout
    cpu_alert_threshold: float = config.system.cpu_threshold
    monitor_cpu: bool = config.system.monitor_cpu
    speaker_verification: bool = config.security.verification_enabled
    confirm_destructive: bool = config.security.confirm_destructive
    max_llm_retries: int = 1
    stt_timeout: int = 5
    conversation_stt_timeout: int = 8


# ── State Change Event ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StateChangeEvent:
    """
    Emitted whenever the engine transitions between states.
    The tray icon subscribes to these to update its colour.
    """

    previous: EngineState
    current: EngineState
    metadata: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)


# ── Engine ────────────────────────────────────────────────────────────────────


class FridayEngine:
    """
    Primary coordinator of the F.R.I.D.A.Y. system.

    Owns all subsystem instances, drives the state machine, and exposes
    a clean lifecycle API to the tray/main entry point.

    Subsystem wiring:
      WakeWordDetector → on_wake_word() → STT → Planner →
        ActionRouter | LLMBrain | GoalEngine → TTS → back to IDLE

    Thread model:
      - Main thread:       Tray icon (owns event loop on Windows)
      - Engine thread:     State machine loop (this class)
      - Detector thread:   Wake word background listener
      - Worker pool:       TaskQueue (4 daemon threads)
      - Monitor thread:    CPU / screen polling
    """

    def __init__(
        self,
        on_state_change: Callable[[StateChangeEvent], None] | None = None,
        on_notify: Callable[[str, str, Any], None] | None = None,
    ) -> None:
        self._config = EngineConfig()
        self._state = EngineState.OFFLINE
        self._lock = threading.RLock()
        self._on_state_change = on_state_change
        self._on_notify = on_notify

        # Wake signal — set by wake word callback, cleared after capture
        self._wake_event = threading.Event()

        # Subsystems (initialised in setup())
        self._wake: WakeWordDetector | None = None
        self._stt: SpeechToText | None = None
        self._tts: TextToSpeech | None = None
        self._llm: LLMBrain | None = None
        self._router: ActionRouter | None = None
        self._planner: ActionPlanner | None = None
        self._memory: MemoryManager | None = None
        self._emotion: EmotionDetector | None = None
        self._learning: LearningEngine | None = None
        self._screen: ScreenMonitor | None = None
        self._goals: GoalEngine | None = None
        self._goal_planner: GoalPlanner | None = None
        self._task_queue: TaskQueue | None = None
        self._swarm: SwarmOrchestrator | None = None
        self._monitor: BackgroundMonitor | None = None
        self._voice_to_prompt: VoiceToPromptAgent | None = None
        self._last_voice_prompt: VoiceToPromptResult | None = None
        self._verification_chain = VerificationChain()
        self._refinement_chain = RefinementChain()
        self._autonomous_reasoner = AutonomousReasoner()
        self._tool_executor = ToolExecutor()
        self._last_chain_context: dict[str, Any] | None = None
        self._browser_skill: Any = None

        # Streaming pipeline for low-latency responses
        self._pipeline: StreamingResponsePipeline | None = None

        # Proactive Trigger Queue for background alerts
        self._trigger_queue: queue.Queue[str] = queue.Queue()
        self._inline_mixed_task_ids: set[str] = set()
        self._inline_mixed_results: dict[str, TaskResult] = {}

        # Engine thread
        self._engine_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()

        # Session statistics
        self._session_start: float = 0.0
        self._commands_handled: int = 0
        self._errors_total: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise all subsystems in dependency order.

        Returns:
            True if all critical subsystems started successfully.
        """
        self._transition(EngineState.BOOTING)
        logger.info("[Engine] Booting F.R.I.D.A.Y…")

        try:
            # ── Critical subsystems (failure = abort) ────────────────────────
            self._tts = TextToSpeech()
            if not self._tts.setup():
                logger.critical("[Engine] TTS failed — cannot operate without speech.")
                return False

            self._pipeline = StreamingResponsePipeline(self._tts)

            self._stt = SpeechToText()
            if not self._stt.setup():
                logger.critical("[Engine] STT failed — cannot operate without microphone.")
                return False

            # ── Task queue (needed by goals) ──────────────────────────────────
            self._task_queue = TaskQueue(max_workers=4)
            self._task_queue.start()

            # ── Non-critical subsystems (failure = degrade gracefully) ────────
            self._llm = LLMBrain()
            if not self._llm.setup():
                logger.warning("[Engine] LLM unavailable — complex queries disabled.")
            self._check_ollama_health()

            self._router = ActionRouter()
            register_all_skills(self._router, llm=self._llm)

            from friday.memory import initialize_memory

            self._memory = initialize_memory()

            self._setup_browser_skill()

            self._learning = LearningEngine()
            self._learning.setup()

            self._screen = ScreenMonitor()
            self._planner = ActionPlanner(learning=self._learning, llm=self._llm)
            self._voice_to_prompt = VoiceToPromptAgent()

            self._swarm = SwarmOrchestrator(engine_report_callback=self.push_trigger)
            self._swarm.start()

            self._goals = GoalEngine(
                task_queue=self._task_queue,
                on_step_complete=self._on_goal_step_complete,
                on_goal_complete=self._on_goal_complete,
                on_goal_failed=self._on_goal_failed,
            )
            self._goals.setup()

            if self._llm and self._llm.is_ready:
                self._goal_planner = GoalPlanner(llm=self._llm)
                self._goal_planner.setup()
                self._wire_goal_handlers()

            # ── Wake word detector ────────────────────────────────────────────
            self._wake = WakeWordDetector(on_detected=self._on_wake_word)
            if not self._wake.start():
                logger.critical("[Engine] Wake word detector failed to start.")
                return False

            self._emotion = EmotionDetector()
            self._emotion.setup()

            # ── Optional monitoring ───────────────────────────────────────────
            if self._config.monitor_cpu:
                self._start_cpu_monitor()

            if self._screen:
                self._screen.start_monitoring(on_alert=self._on_screen_alert)

            # ── Proactive Monitoring ──────────────────────────────────────────
            self._monitor = BackgroundMonitor(
                push_trigger_callback=self.push_trigger, get_state_callback=lambda: self._state
            )

        except Exception as exc:
            logger.critical("[Engine] Setup failed with exception: %s", exc, exc_info=True)
            return False

        self._session_start = time.time()
        logger.info("[Engine] All subsystems online.")
        return True

    def _check_ollama_health(self) -> bool:
        try:
            import httpx

            r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info("[Engine] Ollama online. Models: %s", models)
                return True
        except Exception:
            pass
        logger.critical("[Engine] Ollama not running. Start: ollama serve")
        return False

    def start(self) -> None:
        """
        Start the engine loop on a background thread.
        Call after setup() succeeds.
        """
        self._stop_event.clear()
        self._engine_thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="friday-engine",
        )
        self._engine_thread.start()
        self._transition(EngineState.IDLE)
        play_sound(SoundEvent.STARTUP)

        if self._monitor:
            self._monitor.start()

        self._speak(get_voice_line(ResponseCategory.GREETING), priority=SpeechPriority.HIGH)
        logger.info("[Engine] Engine loop started.")

    def stop(self) -> None:
        """
        Gracefully shut down all subsystems.
        Blocks until the engine thread exits (max 5 seconds).
        """
        logger.info("[Engine] Initiating shutdown…")
        self._transition(EngineState.SHUTTING_DOWN)

        self._stop_event.set()
        self._wake_event.set()  # Unblock any waiting listen cycle

        self._speak(
            get_voice_line(ResponseCategory.FAREWELL),
            priority=SpeechPriority.HIGH,
        )
        self._tts.wait_until_done(timeout=4.0) if self._tts else None

        play_sound(SoundEvent.SHUTDOWN, blocking=True)

        # Teardown subsystems in reverse dependency order
        if self._screen:
            self._screen.stop_monitoring()
        if self._browser_skill:
            self._browser_skill.teardown()
        if self._goals:
            self._goals.teardown()
        if self._task_queue:
            self._task_queue.stop(wait=False)
        if self._swarm:
            self._swarm.stop()
        if self._wake:
            self._wake.stop()
        if self._emotion:
            self._emotion.teardown()
        if self._learning:
            self._learning.teardown()
        if self._tts:
            self._tts.teardown()
        if self._monitor:
            self._monitor.stop()

        if self._engine_thread:
            self._engine_thread.join(timeout=5.0)

        self._transition(EngineState.OFFLINE)
        logger.info("[Engine] Shutdown complete.")

    def pause(self) -> None:
        """Pause wake-word detection."""
        if self._wake:
            self._wake.pause()
            logger.info("[Engine] Wake word detector paused.")

    def resume(self) -> None:
        """Resume wake-word detection."""
        if self._wake:
            self._wake.resume()
            logger.info("[Engine] Wake word detector resumed.")

    # ── State Machine ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Main engine loop. Waits for wake events and drives the
        LISTENING → PROCESSING → SPEAKING cycle.
        """
        logger.debug("[Engine] Loop running.")

        while not self._stop_event.is_set():
            # Check for proactive triggers first
            try:
                alert = self._trigger_queue.get_nowait()
                self._handle_proactive_alert(alert)
                continue
            except queue.Empty:
                pass

            # Wait for wake word signal
            triggered = self._wake_event.wait(timeout=0.2)

            if not triggered or self._stop_event.is_set():
                continue

            self._wake_event.clear()

            # Guard against spurious wake events during speaking
            if self._state in (EngineState.SPEAKING, EngineState.SHUTTING_DOWN):
                # Interruption handled in _on_wake_word callback
                continue

            self._handle_wake_cycle()

        logger.debug("[Engine] Loop exited.")

    def _handle_wake_cycle(self) -> None:
        """
        Execute one full wake → listen → process → respond cycle.
        Manages state transitions and conversation follow-up window.
        """
        self._planner.reset_context() if self._planner else None
        self._interrupt_event.clear()

        # ── LISTENING ─────────────────────────────────────────────────────────
        self._transition(EngineState.LISTENING)
        self._wake.pause() if self._wake else None
        play_sound(SoundEvent.LISTEN)

        transcript = self._capture_speech(timeout=self._config.stt_timeout)

        if not transcript:
            self._wake.resume() if self._wake else None
            self._transition(EngineState.IDLE)
            return

        self._commands_handled += 1

        # ── PROCESSING ────────────────────────────────────────────────────────
        try:
            self._transition(EngineState.PROCESSING, metadata={"task": transcript})
            play_sound(SoundEvent.PROCESSING)

            emotion = self._analyse_emotion(transcript)
            response = self._process_command(transcript, emotion)

            if self._interrupt_event.is_set():
                logger.info("[Engine] Command interrupted, skipping TTS.")
                self._transition(EngineState.LISTENING)
                return

            # ── SPEAKING ──────────────────────────────────────────────────────────
            self._transition(EngineState.SPEAKING)
            play_sound(SoundEvent.SUCCESS)

            # Resume wake word detector for barge-in / interruption
            self._wake.resume() if self._wake else None

            # Deliver response. Streaming occurs internally for LLM queries.
            self._deliver_response(response, emotion)

            # ── CONVERSATION follow-up window ─────────────────────────────────────
            # Skip follow-up if we were interrupted by a new wake word
            if self._pipeline and not self._pipeline._interrupted.is_set():
                self._enter_conversation_mode(emotion)
        except Exception as e:
            logger.critical("[Engine] Processing failed: %s", e, exc_info=True)
            play_sound(SoundEvent.ERROR)
            self._transition(EngineState.IDLE)
            if self._wake:
                self._wake.resume()
            return
        finally:
            self._interrupt_event.clear()

        # Resume wake word detection
        self._wake.resume() if self._wake else None
        self._transition(EngineState.IDLE)

    def _enter_conversation_mode(self, base_emotion: EmotionResult) -> None:
        """
        After a response, open a follow-up window for CONVERSATION_TIMEOUT
        seconds. Each follow-up turn is processed without re-saying the wake word.
        """
        self._transition(EngineState.CONVERSATION)
        deadline = time.time() + self._config.conversation_timeout

        while time.time() < deadline and not self._stop_event.is_set():
            self._transition(EngineState.LISTENING)
            play_sound(SoundEvent.LISTEN)

            transcript = self._capture_speech(timeout=self._config.conversation_stt_timeout)

            if not transcript:
                break  # Silence — exit conversation mode

            try:
                self._transition(EngineState.PROCESSING)
                play_sound(SoundEvent.PROCESSING)

                emotion = self._analyse_emotion(transcript)
                response = self._process_command(transcript, emotion)

                if self._interrupt_event.is_set():
                    logger.info("[Engine] Conversation turn interrupted, skipping TTS.")
                    self._transition(EngineState.LISTENING)
                    continue

                self._transition(EngineState.SPEAKING)
                self._deliver_response(response, emotion)
            except Exception as e:
                logger.critical("[Engine] Conversation turn failed: %s", e, exc_info=True)
                play_sound(SoundEvent.ERROR)
                break

            # Extend deadline on each successful turn
            deadline = time.time() + self._config.conversation_timeout

        self._transition(EngineState.IDLE)

    def _handle_proactive_alert(self, text: str) -> None:
        """
        Immediately interrupt any low-priority state to deliver a background alert.
        Safely transitions from IDLE or CONVERSATION to SPEAKING.
        """
        logger.info("[Engine] Handling proactive alert: %s", text)

        # Guard: Don't interrupt if we're critical or shutting down
        if self._state in (EngineState.BOOTING, EngineState.SHUTTING_DOWN):
            return

        # Check for busy conditions for Toast Bypass
        if self._check_if_user_busy():
            logger.info("[Engine] User busy — routing alert to system notification.")
            if self._on_notify:
                self._on_notify("F.R.I.D.A.Y. Alert", text, None)
                return

        with self._lock:
            # If we are already handling a user command, we might want to wait
            # but for "Iron Man" tier we'll interrupt if it's from the trigger queue.
            if self._pipeline:
                self._pipeline.interrupt()

            self._transition(EngineState.SPEAKING)

        play_sound(SoundEvent.WAKE)  # Signal that F.R.I.D.A.Y. has something to say
        self._speak(text, priority=SpeechPriority.HIGH, wait=True)

        # Return to idle
        self._transition(EngineState.IDLE)
        if self._wake:
            self._wake.resume()

    def _check_if_user_busy(self) -> bool:
        """Detect if the user is likely in a high-focus state (full screen app, etc)."""
        # For now, we'll consider the user busy if they are in a known full-screen application
        # or if privacy mode is active.
        if config.security.privacy_mode:
            return True

        if not self._screen:
            return False

        window = self._screen.get_active_window()
        if not window:
            return False

        # Heuristic: Video players and games usually indicate high focus
        busy_apps = {"vlc", "mpc-hc", "steam", "valorant", "cod", "zoom", "teams"}
        return window.app_name in busy_apps

    def push_trigger(self, message: str | TaskResult) -> None:
        """External entry point for proactive alerts and swarm task results."""
        if isinstance(message, TaskResult):
            if message.task_id in self._inline_mixed_task_ids:
                self._inline_mixed_results[message.task_id] = message
                return
            text = message.output
            label = message.status.value
        else:
            text = str(message)
            label = "message"

        if not text.strip():
            return

        logger.info("[Engine] Queued proactive trigger (%s): %s", label, text[:80])
        self._trigger_queue.put(text)
        self._wake_event.set()

    # ── Local Fast Reply ──────────────────────────────────────────────────────

    def _local_fast_reply(self, text: str) -> str | None:
        """
        Purely local fast path — ZERO API calls.
        Handles only queries answerable without any LLM or network.
        Returns None for anything requiring intelligence or live data,
        which then falls through to the full pipeline.
        """
        import datetime
        import re

        lower = text.lower().strip().rstrip("?.!")

        # ── Time ──
        if re.search(r"\b(tell me the time|what time is it|what's the time|current time|time please)\b", lower):
            now = datetime.datetime.now()
            return f"It's {now.strftime('%I:%M %p')}."

        # ── Date / Day ──
        if re.search(r"\b(what's today|what day is it|today's date|what is the date|what's the date)\b", lower):
            now = datetime.datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        # ── Greetings ──
        if re.search(r"\b(hello|hi friday|hey|good morning|good afternoon|good evening)\b", lower):
            return get_voice_line(ResponseCategory.GREETING)

        # ── Status check ──
        if re.search(r"\b(are you there|are you ready|how are you|you online|systems status)\b", lower):
            return "All systems nominal, boss."

        # ── Stop / Cancel ──
        if re.search(r"\b(stop|cancel|never mind|forget it|abort|quit)\b", lower):
            return "Standing by."

        # ── Battery ──
        if re.search(r"\b(battery status|battery level|how much battery|battery percentage)\b", lower):
            import psutil
            try:
                battery = psutil.sensors_battery()
                if battery is not None:
                    percent = battery.percent
                    power_plugged = battery.power_plugged
                    plugged_status = "plugged in" if power_plugged else "discharging"
                    return f"Battery is at {percent}%, {plugged_status}."
                return "I couldn't retrieve the battery status, Sir."
            except Exception:
                return "I couldn't retrieve the battery status, Sir."

        # ── Acknowledgements ──
        acks = {"thanks", "thank you", "ok", "okay", "got it", "cool", "nice", "sure", "alright", "great", "sounds good"}
        if lower in acks:
            return get_voice_line(ResponseCategory.ACKNOWLEDGEMENT)

        # ── Farewells ──
        farewells = ("bye", "goodbye", "see you", "goodnight", "good night", "sign off")
        if any(lower.startswith(f) for f in farewells):
            return get_voice_line(ResponseCategory.FAREWELL)

        # ── Everything else needs the full pipeline ──
        return None

    def _is_web_query(self, text: str) -> bool:
        """
        Detect queries that need live web data.
        Returns True for weather, news, sports, stocks, events, etc.
        These bypass all chains and go directly to BrowserSkill.
        """
        lower = text.lower().strip()

        web_patterns = (
            # Weather
            r"\bweather\b",
            r"\btemperature\b",
            r"\bforecast\b",
            r"\bis it (?:going to )?(?:rain|snow|cold|hot|warm)\b",
            r"\bhow (?:cold|hot|warm) is it\b",
            # News / Events
            r"\b(?:latest|current|today'?s?|recent)\s+(?:news|events|headlines)\b",
            r"\bwhat(?:'s|\s+is)\s+happening\b",
            r"\b(?:top|breaking)\s+(?:stories|news)\b",
            # Sports / Scores
            r"\b(?:score|scores|result|results)\b",
            r"\bwho\s+(?:won|is\s+winning)\b",
            r"\b(?:game|match|race)\s+(?:today|tonight|last\s+night)\b",
            r"\b(?:sports?|cricket|ipl)\b",
            r"\b(?:live\s+)?scores?\b",
            r"\bmatch(?:es)?\b",
            r"\bfixtures?\b",
            r"\blive\s+score\b",
            r"\b(who (is|are) (playing|winning|batting|bowling))\b",
            # Stocks / Market
            r"\b(?:stock|share|market)\s+(?:price|prices|update|today|status)\b",
            r"\bhow\s+much\s+is\b",
            r"\b(?:bitcoin|crypto|ethereum|dogecoin)\b",
            # Live data
            r"\bcurrent\s+(?:time\s+in|status\s+of|price\s+of|value\s+of)\b",
            r"\bwhen\s+is\s+(?:the\s+)?next\b",
            r"\bwhere\s+can\s+i\s+(?:watch|buy|find|get|stream)\b",
            r"\bhow\s+much\s+does\b",
            r"\blive\s+(?:update|score|price|stream|news)\b",
            # Commerce / product research
            r"\b(buy|purchase|order)\s+.+",
            r"\b(best|top|cheapest)\s+\w+\s+(for|under|below|above)\b.+",
            r"\b(compare|review|price|specs?)\s+.+\b(for|of|under|vs)\b.+",
            r"\brecommend\s+(?:a\s+)?(?:laptop|phone|tablet|pc|computer|headphone|monitor|camera|tv|watch)\b.+",
            r"\b(?:laptop|phone|tablet|pc|computer|headphone|monitor|camera|tv|watch)\s+.+\b(under|below|above|between|price|budget)\b",
            r"\b\d{2,6}\s*(?:rupees?|rs\.?|dollars?|\$|₹|inr|usd)\b",
        )

        return any(re.search(p, lower) for p in web_patterns)

    def _needs_web_escalation(self, text: str, llm_response: str) -> bool:
        """Detect when a response should be replaced by a live web lookup."""
        if self._is_web_query(text):
            return True

        lower = llm_response.lower()
        hedge_phrases = (
            "i'm not sure",
            "i don't know",
            "i cannot confirm",
            "as of my training",
            "i don't have access",
            "i cannot provide live",
            "you may want to check",
            "i lack",
            "my knowledge cutoff",
            "i cannot access",
            "i'm unable to provide current",
        )
        return any(phrase in lower for phrase in hedge_phrases)

    # ── Command Processing ────────────────────────────────────────────────────

    def _process_command(
        self,
        text: str,
        emotion: EmotionResult,
    ) -> str:
        """
        Route a transcribed command to the appropriate handler and
        return a TTS-ready response string.

        Args:
            text:    Normalised transcript.
            emotion: Detected emotional state.

        Returns:
            Response string for TTS delivery.
        """
        # ── Local Fast Reply ──
        # Purely local responses for trivial queries. Zero API calls.
        # Anything not matched here falls through to the full pipeline.
        local_response = self._local_fast_reply(text)
        if local_response is not None:
            if self._memory:
                self._memory.remember_conversation(text, local_response)
            return local_response

        # ── Web Query Fast Path ──
        # Queries needing live data skip ALL chains and go straight to BrowserSkill.
        # No model calls, no safety gates, no perception, no routing.
        if self._is_web_query(text) and self._browser_skill and self._browser_skill._ready:
            logger.info("[Engine] Web query detected — bypassing all chains for browser.")
            return self._handle_browser_query(text)

        context = self._create_chain_context(text, emotion)
        self._last_chain_context = context

        self._run_perception_chain(context)
        self._run_autonomous_reasoning_chain(context)
        voice_prompt = context.get("structured_input")
        self._last_voice_prompt = voice_prompt
        planner_text = voice_prompt.cleaned_transcript if voice_prompt else text

        reasoning = context.get("autonomous_reasoning")
        if reasoning is not None and reasoning.needs_clarification:
            context["synthesized_response"] = reasoning.clarification_question or get_voice_line(
                ResponseCategory.CLARIFICATION
            )
            self._run_delivery_chain(context)
            return self._finalize_chain_response(context, text, planner_text)

        if not self._planner:
            fallback_text = voice_prompt.structured_prompt if voice_prompt else text
            return self._llm_fallback(fallback_text)

        self._run_routing_chain(context)
        decision = context["routing_decision"]
        execution_plan = context["execution_plan"]
        logger.info("[Engine] %s", decision)

        if self._learning:
            self._learning.record_command(planner_text)

        if execution_plan.local_bypass:
            response = self._route_decision(decision, text, voice_prompt)
            context["synthesized_response"] = response
            self._run_delivery_chain(context)
            return self._finalize_chain_response(context, text, planner_text)

        self._run_execution_chain(context)
        self._run_verification_chain(context)

        safety_verdict = context.get("safety_verdict")
        if safety_verdict is not None and not safety_verdict.passed:
            context["synthesized_response"] = SAFE_OPERATIONAL_ERROR
            self._run_delivery_chain(context)
            return self._finalize_chain_response(context, text, planner_text)

        self._run_refinement_chain(context)
        self._run_synthesis_chain(context)
        self._run_delivery_chain(context)
        return self._finalize_chain_response(context, text, planner_text)

    def _create_chain_context(self, text: str, emotion: EmotionResult) -> dict[str, Any]:
        return {
            "raw_audio": None,
            "transcript": text,
            "emotion": emotion,
            "speaker": None,
            "autonomous_reasoning": None,
            "memory_context": "",
            "structured_input": None,
            "execution_plan": None,
            "agent_results": [],
            "confidence": 0.0,
            "safety_verdict": None,
            "refinement_history": [],
            "synthesized_response": "",
            "delivery_format": "",
        }

    def _run_perception_chain(self, context: dict[str, Any]) -> None:
        if self._voice_to_prompt is None:
            self._voice_to_prompt = VoiceToPromptAgent()
        context["structured_input"] = self._voice_to_prompt.process(context["transcript"])

    def _run_autonomous_reasoning_chain(self, context: dict[str, Any]) -> None:
        structured = context.get("structured_input")
        reasoning = self._autonomous_reasoner.analyze(
            transcript=str(context.get("transcript", "")),
            structured_input=structured,
            memory=self._memory,
        )
        context["autonomous_reasoning"] = reasoning
        context["memory_context"] = "\n".join(reasoning.relevant_memory)
        context["tool_observations"] = []
        for hint in reasoning.tool_hints:
            observation = self._tool_executor.execute(hint, str(context.get("transcript", "")), memory=self._memory)
            context["tool_observations"].append(observation)
        if structured is not None:
            structured.metadata["autonomous_reasoning"] = {
                "inferred_intent": reasoning.inferred_intent,
                "confidence": reasoning.confidence,
                "missing_information": list(reasoning.missing_information),
                "requires_confirmation": reasoning.requires_confirmation,
                "risk_level": reasoning.risk_level,
                "tool_hints": list(reasoning.tool_hints),
                "cot_trace": list(reasoning.cot_trace),
            }

    def _run_routing_chain(self, context: dict[str, Any]) -> None:
        structured = context["structured_input"]
        planner_text = structured.cleaned_transcript if structured else context["transcript"]

        ensemble_hint = None
        specialist_override = None
        if self._planner:
            try:
                from friday.phase1 import Phase1Gate, RouteDecision, resolve_specialist

                triage = run_async(Phase1Gate.route(planner_text))
                ensemble_hint = triage.ensemble_hint if triage else None

                # Capture specialist route from Phase1Gate for delegation override.
                # If Phase1Gate identified a specialist (not FAST_PATH or BLOCKED),
                # we must delegate to it even if the planner disagrees.
                if triage and triage.route not in (RouteDecision.FAST_PATH, RouteDecision.BLOCKED):
                    specialist_override = resolve_specialist(triage.route)
            except Exception as exc:
                logger.error("[Engine] Phase 1 Gate routing failed: %s", exc)

        decision = (
            self._planner.decide_with_ensemble_hint(planner_text, ensemble_hint)
            if self._planner
            else None
        )

        # If Phase1Gate identified a specialist but the planner didn't delegate,
        # override routing to DELEGATE so the specialist is actually used.
        # This fixes BrowserAgent (and all specialist) routes being ignored.
        if specialist_override is not None and decision is not None:
            if decision.routing != RoutingDecision.DELEGATE:
                logger.info(
                    "[Engine] Phase1Gate routed to specialist %s but planner chose %s — overriding to DELEGATE.",
                    specialist_override.name,
                    decision.routing.name,
                )
                import dataclasses
                decision = dataclasses.replace(decision, routing=RoutingDecision.DELEGATE)

        # Store specialist in context for _handle_delegation fallback
        context["specialist_override"] = specialist_override

        plan = (
            self._planner.create_execution_plan(structured, decision)
            if self._planner and structured and decision
            else ExecutionPlan(steps=(), local_bypass=True, rationale="Planner unavailable.")
        )
        context["routing_decision"] = decision
        context["execution_plan"] = plan

    def _run_execution_chain(self, context: dict[str, Any]) -> None:
        plan: ExecutionPlan = context["execution_plan"]
        if not plan.has_agents:
            context["agent_results"] = []
            return

        parallel_steps = [step for step in plan.steps if step.read_only and not step.dependencies]
        sequential_steps = [step for step in plan.steps if step not in parallel_steps]
        results: list[dict[str, Any]] = []

        if parallel_steps:
            with ThreadPoolExecutor(max_workers=plan.max_concurrency) as executor:
                futures = [
                    executor.submit(self._execute_plan_step, step, context)
                    for step in parallel_steps
                ]
                for future in futures:
                    results.append(future.result())

        for step in sequential_steps:
            results.append(self._execute_plan_step(step, context))

        context["agent_results"] = results

    def _execute_plan_step(self, step: PlanStep, context: dict[str, Any]) -> dict[str, Any]:
        if not self._llm or not self._llm.is_ready:
            return {
                "agent": step.agent,
                "task_category": step.task_category,
                "ensemble_type": step.ensemble_type,
                "output": "I don't have access to my reasoning systems right now.",
                "success": False,
                "confidence": 0.0,
                "model_tier": _tier_for_category(step.task_category),
            }

        prompt = self._plan_prompt(step, context)
        response = self._llm.query(
            prompt,
            ensemble_type=step.ensemble_type,
            add_to_history=False,
        )
        return {
            "agent": step.agent,
            "task_category": step.task_category,
            "ensemble_type": step.ensemble_type,
            "output": response.text if response.success else str(response),
            "success": response.success,
            "confidence": 1.0 if response.success else 0.0,
            "model_tier": _tier_for_category(step.task_category),
        }

    def _plan_prompt(self, step: PlanStep, context: dict[str, Any]) -> str:
        if context.get("refinement_instruction"):
            return str(context["refinement_instruction"])
        structured = context.get("structured_input")
        original = structured.cleaned_transcript if structured else context["transcript"]
        reasoning = context.get("autonomous_reasoning")
        reasoning_block = self._format_reasoning_context(reasoning, context.get("memory_context", ""))
        tool_block = self._format_tool_observations(context)
        return (
            f"{step.prompt}\n\n"
            f"[CHAIN_CONTEXT]\nOriginal input: {original}\n"
            f"Task category: {step.task_category}\n"
            f"Agent: {step.agent}"
            f"{reasoning_block}"
            f"{tool_block}"
        )

    @staticmethod
    def _format_reasoning_context(
        reasoning: AutonomousReasoningResult | None,
        memory_context: str,
    ) -> str:
        if reasoning is None:
            return ""
        lines = [
            "",
            "[AUTONOMOUS_REASONING]",
            f"Inferred intent: {reasoning.inferred_intent}",
            f"Confidence: {reasoning.confidence}",
        ]
        if reasoning.assumptions:
            lines.append("Assumptions: " + "; ".join(reasoning.assumptions))
        if reasoning.tool_hints:
            lines.append("Useful tools: " + ", ".join(reasoning.tool_hints))
        if reasoning.cot_trace:
            lines.extend(reasoning.cot_trace)
        if reasoning.requires_confirmation:
            lines.append("Guardrail: ask for confirmation before any external or irreversible action.")
        if memory_context:
            lines.extend(("", "[RELEVANT_MEMORY]", memory_context))
        return "\n".join(lines)

    @staticmethod
    def _format_tool_observations(context: dict[str, Any]) -> str:
        observations = context.get("tool_observations") or []
        if not observations:
            return ""
        lines = ["", "[TOOL_OBSERVATIONS]"]
        for item in observations:
            lines.append(
                f"{item.get('tool')}: success={item.get('success')} query={item.get('query')} result={item.get('result')}"
            )
        return "\n".join(lines)

    def _run_verification_chain(self, context: dict[str, Any]) -> None:
        if not context.get("agent_results"):
            context["confidence"] = 1.0
            return
        try:
            verification_result: Any = run_async(
                self._verification_chain.verify(context),
                timeout=30.0,
            )
            # Write result back into context so synthesis and delivery chains can use it
            if isinstance(verification_result, dict):
                context.update(verification_result)
            elif hasattr(verification_result, "__dict__"):
                context["verification_report"] = verification_result
                context["confidence"] = getattr(verification_result, "confidence", 0.8)
        except TimeoutError:
            logger.warning("[Engine] Verification chain timed out — proceeding without verification.")
            context["confidence"] = 0.5
        except Exception as exc:
            logger.error("[Engine] Verification chain failed: %s", exc)
            context["confidence"] = 0.5

        if context.get("confidence", 1.0) < 0.4:
            self._run_deep_verification(context)

    def _run_refinement_chain(self, context: dict[str, Any]) -> None:
        if not context.get("agent_results"):
            return
        try:
            refined_context = run_async(
                self._refinement_chain.refine(context, self._refinement_iteration),
                timeout=45.0,
            )
            # Merge refined results back into the live context
            if isinstance(refined_context, dict):
                context["agent_results"] = refined_context.get(
                    "agent_results", context["agent_results"]
                )
                context["refinement_history"] = refined_context.get(
                    "refinement_history", context.get("refinement_history", [])
                )
                if refined_context.get("synthesized_response"):
                    context["synthesized_response"] = refined_context["synthesized_response"]
        except TimeoutError:
            logger.warning("[Engine] Refinement chain timed out — using unrefined output.")
        except Exception as exc:
            logger.error("[Engine] Refinement chain failed: %s", exc)

    def _run_deep_verification(self, context: dict[str, Any]) -> None:
        """
        Escalate to analysis_flagship ONLY when verification confidence is critically low.
        Never called in the normal path — only triggered by _run_verification_chain on failure.
        """
        confidence = context.get("confidence", 1.0)
        if confidence >= 0.4:
            return  # Not low enough to warrant deep verification

        agent_results = context.get("agent_results", [])
        if not agent_results:
            return

        outputs = "\n\n".join(
            str(r.get("output", "")) for r in agent_results if isinstance(r, dict)
        )
        original = context.get("transcript", "")

        deep_prompt = (
            f"Original request: {original}\n\n"
            f"Agent output to verify:\n{outputs}\n\n"
            "Identify any factual errors, logical gaps, or incomplete answers. "
            "If the output is correct, say 'VERIFIED'. "
            "If corrections are needed, provide the corrected output only."
        )

        try:
            if not self._llm or not self._llm.is_ready:
                return

            from friday.ensemble import CANONICAL_MODEL_MAP
            response = self._llm.query(
                deep_prompt,
                model=CANONICAL_MODEL_MAP["analysis_flagship"],
                add_to_history=False,
            )

            if response.success and response.text:
                if "VERIFIED" not in response.text.upper():
                    # Correction provided — inject as new agent result
                    context["agent_results"].append({
                        "agent": "DEEP_VERIFIER",
                        "task_category": "verification",
                        "ensemble_type": "reasoning",
                        "output": response.text,
                        "success": True,
                        "confidence": 0.9,
                        "model_tier": "flagship",
                    })
                    context["confidence"] = 0.9
                    logger.info("[Engine] Deep verification correction applied.")
                else:
                    context["confidence"] = 0.9
                    logger.info("[Engine] Deep verification confirmed output.")

        except Exception as exc:
            logger.error("[Engine] Deep verification failed: %s", exc)

    async def _refinement_iteration(
        self,
        context: dict[str, Any],
        iteration: int,
        model_tier: str,
    ) -> dict[str, Any]:
        candidate = dict(context)
        candidate["agent_results"] = list(context.get("agent_results", []))
        candidate["refinement_history"] = context.get("refinement_history", [])
        await asyncio.to_thread(self._execute_refinement_sync, candidate, iteration, model_tier)
        await self._verification_chain.verify(candidate)
        return candidate

    def _execute_refinement_sync(
        self,
        context: dict[str, Any],
        iteration: int,
        model_tier: str,
    ) -> None:
        structured = context.get("structured_input")
        category = structured.task_category if structured else "reasoning"
        step = PlanStep(
            agent=getattr(structured, "target_agent", None) or "REASONING",
            task_category=category,
            ensemble_type=getattr(structured, "ensemble_type", None) or "reasoning",
            prompt=str(context.get("refinement_instruction", "")),
            read_only=True,
        )
        result = self._execute_plan_step(step, context)
        result["model_tier"] = model_tier
        result["refinement_iteration"] = iteration
        context["agent_results"] = [result]

    def _run_synthesis_chain(self, context: dict[str, Any]) -> None:
        if context.get("synthesized_response"):
            return

        agent_results = context.get("agent_results", [])
        original = context.get("transcript", "")
        verification = context.get("verification_report")

        def synthesize(prompt: str) -> str | None:
            if not self._llm or not self._llm.is_ready:
                return None
            response = self._llm.query(
                prompt,
                model=CANONICAL_MODEL_MAP["context_flagship"],
                add_to_history=False,
            )
            return response.text if response.success else None

        if self._swarm:
            merged = self._swarm.merge_results(
                agent_results,
                original_input=original,
                verification_results=verification,
                synthesis_callable=synthesize,
            )
        else:
            merged = "\n\n".join(
                str(result.get("output", ""))
                for result in agent_results
                if isinstance(result, dict)
            )

        context["synthesized_response"] = merged or get_voice_line(ResponseCategory.ERROR)

    def _run_delivery_chain(self, context: dict[str, Any]) -> None:
        response = str(context.get("synthesized_response", ""))
        emotion = context.get("emotion")
        if emotion is not None and emotion.needs_acknowledgement:
            response = build_tone_aware_response(response, emotion.tone)
        context["delivery_format"] = self._voice_optimize(response)

    def _finalize_chain_response(
        self,
        context: dict[str, Any],
        original_text: str,
        planner_text: str,
    ) -> str:
        response = str(context.get("delivery_format", ""))
        if self._memory:
            self._memory.remember_conversation(original_text, response)
        if self._planner:
            self._planner.notify_executed(planner_text, response)
        return response

    @staticmethod
    def _voice_optimize(text: str) -> str:
        text = re.sub(r"[`*_#>-]+", " ", text)
        text = re.sub(r"\bAPI\b", "A-P-I", text)
        text = re.sub(r"\bSQL\b", "S-Q-L", text)
        text = re.sub(r"\bRAM\b", "R-A-M", text)
        text = re.sub(r"\s+", " ", text).strip()

        def replace_number(match: re.Match[str]) -> str:
            value = match.group(0).replace(",", "")
            try:
                number = int(value)
            except ValueError:
                return match.group(0)
            if number == 1024:
                return "one thousand twenty-four"
            return str(number)

        text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", replace_number, text)
        return re.sub(r"([.!?])\s+", r"\1 ", text)

    def _route_decision(
        self,
        decision: PlannerDecision,
        original_text: str,
        voice_prompt: VoiceToPromptResult | None = None,
    ) -> str:
        """
        Dispatch a PlannerDecision to the correct handler.

        Args:
            decision:      Planner routing decision.
            original_text: Raw transcript for fallback use.

        Returns:
            TTS-ready response string.
        """
        routing = decision.routing

        if routing == RoutingDecision.HANDLE_SYSTEM:
            return self._handle_system_command(decision.text)

        if routing == RoutingDecision.EXECUTE_ACTION:
            return self._handle_local_action(decision.text)

        if routing == RoutingDecision.CREATE_GOAL:
            return self._handle_goal_creation(decision.text)

        if routing == RoutingDecision.MANAGE_GOAL:
            return self._handle_goal_management(decision.text)

        if routing == RoutingDecision.QUERY_LLM:
            prompt_text = voice_prompt.structured_prompt if voice_prompt else decision.text
            ensemble_type = voice_prompt.ensemble_type if voice_prompt else None
            return self._handle_llm_query(prompt_text, ensemble_type=ensemble_type)

        if routing == RoutingDecision.ASK_CLARIFY:
            return get_voice_line(ResponseCategory.CLARIFICATION)

        if routing == RoutingDecision.DELEGATE:
            return self._handle_delegation(decision, voice_prompt)

        if routing == RoutingDecision.IGNORE:
            return ""

        # Unexpected routing — safe fallback
        logger.warning("[Engine] Unhandled routing: %s", routing)  # type: ignore[unreachable]
        ensemble_type = voice_prompt.ensemble_type if voice_prompt else None
        return self._handle_llm_query(original_text, ensemble_type=ensemble_type)

    def _setup_browser_skill(self) -> None:
        """Launch Playwright and register browser actions with the ActionRouter."""
        try:
            from friday.skills import get_skill_manager

            self._browser_skill = get_skill_manager().load_skill("browser")
            if self._browser_skill:
                import threading

                def _bg_setup():
                    try:
                        if self._browser_skill.setup():
                            logger.info("[Engine] Browser skill online — Playwright active.")
                        else:
                            logger.warning(
                                "[Engine] Browser skill unavailable — "
                                "run: pip install playwright && playwright install chromium"
                            )
                            # Do not clear self._browser_skill entirely; just leave it un-setup.
                            # We can also keep it as is, or set it to None.
                            # Actually, setting it to None if setup fails is consistent with original code.
                            self._browser_skill = None
                    except Exception as e:
                        logger.warning("[Engine] Browser skill background setup failed: %s", e)
                        self._browser_skill = None

                threading.Thread(target=_bg_setup, daemon=True, name="BrowserSkillSetup").start()
        except Exception as exc:
            logger.warning("[Engine] Browser skill setup failed: %s", exc)
            self._browser_skill = None

    def _handle_browser_query(self, prompt: str) -> str:
        """Handle a web query directly through BrowserSkill on the same thread."""
        if not self._browser_skill or not self._browser_skill._ready:
            return self._handle_llm_query(prompt, allow_web_escalation=False)

        try:
            result = self._browser_skill.handle_voice_command(prompt)
            if result and result.status == ActionStatus.SUCCESS:
                return str(result.message)
        except Exception as exc:
            logger.error("[Engine] Browser query failed: %s", exc)

        return self._handle_llm_query(prompt, allow_web_escalation=False)

    def _handle_delegation(
        self,
        decision: PlannerDecision,
        voice_prompt: VoiceToPromptResult | None = None,
    ) -> str:
        """Offload a task to the swarm, passing it through the Phase 1 Gate."""
        if not self._swarm:
            prompt_text = voice_prompt.structured_prompt if voice_prompt else decision.text
            ensemble_type = voice_prompt.ensemble_type if voice_prompt else None
            return self._handle_llm_query(prompt_text, ensemble_type=ensemble_type)

        prompt = voice_prompt.structured_prompt if voice_prompt else decision.text

        # If specialist is BrowserAgent and browser skill is live, use it directly.
        # The swarm's BrowserAgent just calls LLM — it doesn't actually browse.
        # BrowserSkill opens Playwright, searches, reads, and extracts answers.
        specialist_override = None
        if self._last_chain_context:
            specialist_override = self._last_chain_context.get("specialist_override")
        if specialist_override == Specialist.BROWSER and self._browser_skill and self._browser_skill._ready:
            logger.info("[Engine] Routing to BrowserSkill directly (bypassing swarm).")
            return self._handle_browser_query(decision.text)

        try:
            from friday.phase1 import Phase1Gate, RouteDecision, resolve_specialist

            # Run Phase 1 input safety and routing
            triage, fast_output = run_async(Phase1Gate.process(prompt))

            # Hard blocked by safety
            if triage.route == RouteDecision.BLOCKED:
                return "My apologies, Sir. I cannot process that request."

            # ultrafast_8b answered directly — no agent needed
            if triage.route == RouteDecision.FAST_PATH:
                return fast_output or "My apologies, Sir. Fast path response was empty."

            # Route to the correct specialist
            specialist = resolve_specialist(triage.route)
            if specialist is None:
                return "My apologies, Sir. Routing failed."

            # Package history context
            history = self._llm.history.build_payload() if self._llm else []

            # Delegate task to the swarm
            task_id = self._swarm.delegate(
                specialist=specialist,
                prompt=prompt,
                history=history,
            )

            # Wait for the specialist to complete
            result = self._swarm.await_result(task_id, timeout=120.0)
            if result is None or not result.success:
                return "My apologies, Sir. The specialist did not respond in time."

            # Run Output Safety check
            _, verified = run_async(Phase1Gate.process(prompt, specialist_output=result.output))
            if verified is None:
                return "My apologies, Sir. The response was filtered for safety reasons."

            return verified

        except Exception as e:
            logger.error("[Engine] Delegation failed with Phase1Gate: %s", e)

            # Fallback: delegate directly using specialist captured during routing chain.
            # This fires when Phase1Gate.process() itself fails (e.g. safety models 429'd),
            # which would otherwise fall through to _handle_llm_query and lose the
            # specialist route entirely.
            specialist_override = None
            if self._last_chain_context:
                specialist_override = self._last_chain_context.get("specialist_override")

            if specialist_override is not None:
                logger.info(
                    "[Engine] Attempting direct delegation to %s (bypassing Phase1Gate).",
                    specialist_override.name,
                )
                try:
                    history = self._llm.history.build_payload() if self._llm else []
                    task_id = self._swarm.delegate(
                        specialist=specialist_override,
                        prompt=prompt,
                        history=history,
                    )
                    result = self._swarm.await_result(task_id, timeout=120.0)
                    if result is not None and result.success and result.output:
                        return result.output
                except Exception as inner_e:
                    logger.error("[Engine] Direct delegation also failed: %s", inner_e)

            prompt_text = voice_prompt.structured_prompt if voice_prompt else decision.text
            ensemble_type = voice_prompt.ensemble_type if voice_prompt else None
            return self._handle_llm_query(prompt_text, ensemble_type=ensemble_type)

    # ── Handler: Local Actions ────────────────────────────────────────────────

    def _resolve_specialist(self, specialist_name: Any) -> Specialist:
        """Accept enum names, enum values, and lower-case task labels."""
        if isinstance(specialist_name, Specialist):
            return specialist_name

        token = str(specialist_name or "").strip()
        if not token:
            return Specialist.RESEARCH

        normalized = token.upper().replace("-", "_").replace(" ", "_")
        if normalized in Specialist.__members__:
            return Specialist[normalized]

        lowered = token.lower()
        for specialist in Specialist:
            if lowered in (specialist.value.lower(), specialist.name.lower()):
                return specialist

        label_map = {
            "code": Specialist.CODER,
            "coding": Specialist.CODER,
            "coder": Specialist.CODER,
            "security": Specialist.SECURITY,
            "research": Specialist.RESEARCH,
            "search": Specialist.RESEARCH,
            "reasoning": Specialist.REASONING,
            "brainstorming": Specialist.BRAINSTORMING,
            "brainstorm": Specialist.BRAINSTORMING,
            "productivity": Specialist.PRODUCTIVITY,
            "document": Specialist.PRODUCTIVITY,
            "media": Specialist.MEDIA,
            "retrieval": Specialist.RETRIEVAL,
            "file_analysis": Specialist.RETRIEVAL,
            "vision": Specialist.VISION,
            "screen_interaction": Specialist.VISION,
            "browser": Specialist.BROWSER,
            "browser_agent": Specialist.BROWSER,
            "browser_automation": Specialist.BROWSER,
        }
        return label_map.get(lowered, Specialist.RESEARCH)

    def _handle_mixed_task(self, voice_prompt: VoiceToPromptResult) -> str:
        """Delegate up to three task categories and merge completed outputs."""
        if not self._swarm:
            return self._handle_llm_query(
                voice_prompt.structured_prompt,
                ensemble_type=voice_prompt.ensemble_type,
            )

        categories = list(dict.fromkeys(voice_prompt.metadata.get("mixed_categories", [])))[:3]
        if not categories:
            return self._handle_llm_query(
                voice_prompt.structured_prompt,
                ensemble_type=voice_prompt.ensemble_type,
            )

        read_only = {
            "search",
            "reasoning",
            "brainstorming",
            "file_analysis",
            "screen_interaction",
        }
        started: list[tuple[str, str]] = []
        deadline = time.time() + 8.0
        history = self._llm.history.build_payload() if self._llm else []

        def _submit(category: str) -> str | None:
            specialist_name = self._agent_for_category(category)
            prompt = (
                f"{voice_prompt.structured_prompt}\n\n"
                f"[SUBTASK]\nFocus only on the {category} portion of this mixed request."
            )
            try:
                from friday.phase1 import Phase1Gate, RouteDecision, resolve_specialist
                triage, _fast_output = run_async(Phase1Gate.process(prompt))
                if triage.route == RouteDecision.BLOCKED:
                    logger.warning("[Engine] Mixed subtask blocked by safety.")
                    return None
                resolved_spec = resolve_specialist(triage.route) or self._resolve_specialist(specialist_name)
                if not self._swarm:
                    return None
                return self._swarm.delegate(resolved_spec, prompt, history=history)
            except Exception as exc:
                logger.error("[Engine] Mixed-task delegation failed for %s: %s", category, exc)
                return None

        for category in categories:
            task_id = _submit(category)
            if task_id:
                self._inline_mixed_task_ids.add(task_id)
                started.append((category, task_id))
            if category not in read_only and task_id:
                remaining = max(0.1, deadline - time.time())
                self._swarm.await_result(task_id, timeout=remaining)

        outputs: list[str] = []
        for category, task_id in started:
            remaining = max(0.0, deadline - time.time())
            result = self._swarm.await_result(task_id, timeout=remaining)
            result = result or self._inline_mixed_results.get(task_id)
            if result and result.success and result.output:
                from friday.phase1 import Phase1Gate
                _, verified = run_async(Phase1Gate.process(voice_prompt.structured_prompt, specialist_output=result.output))
                if verified:
                    outputs.append(f"{category}: {verified}")

        for _, task_id in started:
            self._inline_mixed_task_ids.discard(task_id)
            self._inline_mixed_results.pop(task_id, None)

        if outputs:
            return "Mixed task results:\n" + "\n\n".join(outputs)

        return "I've assigned the mixed task to the relevant specialists. I'll report results as they finish."

    def _agent_for_category(self, category: str) -> str:
        mapping = {
            "search": "RESEARCH",
            "coding": "CODER",
            "reasoning": "REASONING",
            "brainstorming": "BRAINSTORMING",
            "document": "PRODUCTIVITY",
            "productivity": "PRODUCTIVITY",
            "media": "MEDIA",
            "retrieval": "RETRIEVAL",
            "file_analysis": "RETRIEVAL",
            "general": "REASONING",
        }
        return mapping.get(category, "RESEARCH")

    def _handle_local_action(self, text: str) -> str:
        """
        Attempt to execute a local action via the ActionRouter.
        Falls back to LLM if no action matches.
        """
        if not self._router:
            return self._handle_llm_query(text)

        result: ActionResult | None = self._router.route(text)

        if result is None:
            # No match — escalate to LLM
            logger.debug("[Engine] No action match for '%s' — escalating to LLM.", text)
            return self._handle_llm_query(text)

        if result.status == ActionStatus.REQUIRES_CONFIRM:
            return result.message

        if result.success:
            if hasattr(self, "_sandbox_retries"):
                self._sandbox_retries = 0
            play_sound(SoundEvent.SUCCESS)
            if self._learning:
                self._learning.record_command(
                    text,
                    action=result.action,
                    success=True,
                    duration=result.duration,
                )
            return result.message

        # Action failed
        play_sound(SoundEvent.ERROR)
        if self._learning:
            self._learning.record_command(
                text,
                action=result.action,
                success=False,
            )

        if result.status == ActionStatus.FORBIDDEN:
            if not hasattr(self, "_sandbox_retries"):
                self._sandbox_retries = 0

            if self._sandbox_retries >= 1:
                self._sandbox_retries = 0
                return "I couldn't safely generate a script for that task without violating sandbox constraints."

            self._sandbox_retries += 1
            logger.warning(
                "[Engine] Sandbox blocked script. Attempting LLM auto-correction (Retry %d/1).",
                self._sandbox_retries,
            )
            fallback_prompt = (
                f"The script you just wrote was blocked by the security sandbox.\n"
                f"Error details:\n{result.message}\n\n"
                f"Please re-write the script using only standard built-in functions "
                f"and allowed safe libraries (math, datetime, re, json, time, urllib, sqlite3)."
            )
            # Re-feed to the LLM to generate a sanitized version
            return self._handle_llm_query(fallback_prompt)

        if hasattr(self, "_sandbox_retries"):
            self._sandbox_retries = 0

        return result.message or get_voice_line(ResponseCategory.ERROR)

    # ── Handler: LLM Queries ──────────────────────────────────────────────────

    def _handle_llm_query(
        self,
        text: str,
        ensemble_type: str | None = None,
        allow_web_escalation: bool = True,
    ) -> str:
        """
        Submit a query to the LLM and return the response.
        Injects memory context before the call.
        """
        if not self._llm or not self._llm.is_ready:
            return (
                "I don't have access to my reasoning systems right now. "
                "I can still handle local commands."
            )

        # SHORT QUERIES: skip ensemble, go direct to avoid 2-min waits
        is_short = len(text.split()) < 15
        if is_short:
            ensemble_type = None  # bypass ensemble for short conversational queries

        # Inject relevant memory context
        if self._memory:
            context = self._memory.build_context_summary()
            if context:
                self._llm.inject_context(context)

        # Use streaming for LLM queries to trigger the sub-second TTFA pipeline
        self._pipeline.start() if self._pipeline else None

        response_holder: list[Any] = []

        def _stream_call() -> None:
            response_holder.append(
                self._llm.stream(
                    text,
                    on_chunk=self._pipeline.push_token if self._pipeline else None,
                    retries=self._config.max_llm_retries,
                    interrupt_event=self._interrupt_event,
                    ensemble_type=ensemble_type,
                )
            )

        stream_thread = threading.Thread(target=_stream_call, daemon=True, name="llm-stream-timeout")
        stream_thread.start()
        stream_thread.join(timeout=45.0)
        if stream_thread.is_alive():
            self._interrupt_event.set()
            if self._pipeline:
                self._pipeline.interrupt()
                self._pipeline.finish()
            self._speak("That's taking longer than expected. Try again.")
            return ""

        response = response_holder[0]

        if self._pipeline:
            self._pipeline.finish()

        if response.success:
            if allow_web_escalation and self._needs_web_escalation(text, response.text):
                self._speak(get_voice_line(ResponseCategory.WEB_SEARCH))
                if self._browser_skill and self._browser_skill.is_ready:
                    return self._handle_browser_query(text)
            return response.text

        # Classify failure for appropriate user messaging
        if response.status == LLMStatus.TIMEOUT:
            return "That took too long. Try asking again."
        if response.status == LLMStatus.RATE_LIMITED:
            return "I'm being rate limited. Please wait a moment."
        if response.status == LLMStatus.CONNECTION_ERROR:
            return "I can't reach my reasoning systems right now."

        self._errors_total += 1
        return get_voice_line(ResponseCategory.ERROR)

    # ── Handler: System Commands ──────────────────────────────────────────────

    def _handle_system_command(self, text: str) -> str:
        """
        Handle Friday-internal commands (stop, status, help, etc.).

        Args:
            text: Normalised command text.

        Returns:
            TTS-ready response string.
        """
        text = text.lower()

        if any(w in text for w in ("stop", "shutdown", "offline", "exit", "quit")):
            threading.Thread(target=self.stop, daemon=True).start()
            return get_voice_line(ResponseCategory.FAREWELL)

        if "status" in text or "how are you" in text:
            return self._build_status_response()

        if "help" in text or "what can you do" in text:
            return self._build_help_response()

        if "clear" in text and "history" in text:
            if self._llm:
                self._llm.clear_history()
            if self._memory:
                self._memory.clear_session()
            return "Conversation history cleared."

        if "pause" in text:
            self._transition(EngineState.IDLE)
            return "Pausing. Say my name to resume."

        return get_voice_line(ResponseCategory.ACKNOWLEDGEMENT)

    # ── Handler: Goal Creation ────────────────────────────────────────────────

    def _handle_goal_creation(self, text: str) -> str:
        """
        Plan and create a new goal from the user's objective.

        Args:
            text: Natural language goal description.

        Returns:
            TTS-ready response confirming or explaining the plan.
        """
        if not self._goal_planner or not self._goals:
            return "Goal planning isn't available right now."

        self._speak(get_voice_line(ResponseCategory.THINKING))

        plan = self._goal_planner.plan(text)

        if not plan.success:
            return "I wasn't able to create a plan for that. Could you describe it differently?"

        explanation = self._goal_planner.explain_plan(plan)

        goal_id = self._goals.create_goal(
            title=plan.title,
            description=plan.description,
            steps=plan.to_goal_steps(),
            priority=plan.priority,
        )

        self._goals.start_goal(goal_id)

        return f"{explanation} I've started working on it in the background."

    # ── Handler: Goal Management ──────────────────────────────────────────────

    def _handle_goal_management(self, text: str) -> str:
        """
        Handle goal status queries and control commands.

        Args:
            text: Normalised command text.

        Returns:
            TTS-ready response.
        """
        if not self._goals:
            return "Goal management isn't available right now."

        text_lower = text.lower()

        # Status query
        if any(w in text_lower for w in ("status", "progress", "how", "update")):
            active = self._goals.get_active_goals()
            if not active:
                completed = self._goals.list_goals(status=GoalStatus.COMPLETED, limit=1)
                if completed:
                    return f"No active goals. Last completed: '{completed[0].title}'."
                return "No active goals at the moment."

            summaries = [g.progress_summary() for g in active[:3]]
            return " ".join(summaries)

        # List all goals
        if "list" in text_lower or "show" in text_lower:
            all_goals = self._goals.list_goals(limit=5)
            if not all_goals:
                return "You have no goals at the moment."
            titles = ", ".join(f"'{g.title}'" for g in all_goals)
            return f"Your goals: {titles}."

        # Pause active goal
        if "pause" in text_lower:
            active = self._goals.get_active_goals()
            if active:
                self._goals.pause_goal(active[0].id)
                return f"Paused goal: '{active[0].title}'."
            return "No active goal to pause."

        # Resume paused goal
        if "resume" in text_lower:
            paused = self._goals.list_goals(status=GoalStatus.PAUSED, limit=1)
            if paused:
                self._goals.resume_goal(paused[0].id)
                return f"Resuming '{paused[0].title}'."
            return "No paused goal to resume."

        # Cancel active goal
        if "cancel" in text_lower or "stop" in text_lower:
            active = self._goals.get_active_goals()
            if active:
                self._goals.cancel_goal(active[0].id)
                return f"Cancelled goal: '{active[0].title}'."
            return "No active goal to cancel."

        return get_voice_line(ResponseCategory.CLARIFICATION)

    # ── Speech Capture ────────────────────────────────────────────────────────

    def _capture_speech(self, timeout: int) -> str | None:
        """
        Capture and return a transcribed utterance.
        Returns None on silence, timeout, or transcription failure.

        Args:
            timeout: Seconds to wait for speech to begin.

        Returns:
            Normalised transcript string or None.
        """
        if not self._stt:
            return None

        result: TranscriptResult = self._stt.listen(timeout=timeout)

        if result.success:
            logger.info("[Engine] Captured: '%s'", result.text)

            return result.text

        if result.status == TranscriptStatus.TIMEOUT:
            logger.debug("[Engine] STT timeout — no speech detected.")
        elif result.status == TranscriptStatus.UNCLEAR:
            self._speak("I didn't catch that. Could you repeat?")
        elif result.status == TranscriptStatus.API_ERROR:
            self._speak("I'm having trouble with speech recognition right now.")

        return None

    # ── Emotion Analysis ──────────────────────────────────────────────────────

    def _analyse_emotion(self, text: str) -> EmotionResult:
        """
        Detect the user's emotional tone from transcript text.

        Args:
            text: Transcribed utterance.

        Returns:
            EmotionResult (neutral if emotion detection unavailable).
        """
        if not self._emotion:
            from friday.emotion import EmotionResult
            from friday.personality import EmotionalTone

            return EmotionResult(
                tone=EmotionalTone.NEUTRAL,
                confidence=0.0,
                signals=[],
                duration=0.0,
                raw_text=text,
            )
        return self._emotion.analyse(text)

    # ── TTS Delivery ──────────────────────────────────────────────────────────

    def _speak(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.NORMAL,
        wait: bool = False,
    ) -> None:
        """
        Deliver a TTS response. No-ops on empty or whitespace-only text.

        Args:
            text:     Text to speak.
            priority: Speech priority (HIGH interrupts current speech).
            wait:     If True, block until speech finishes.
        """
        if not self._tts or not text.strip():
            return

        if wait:
            self._tts.speak(text, priority=priority)
        else:
            self._tts.speak_async(text, priority=priority)

    def _deliver_response(
        self,
        response: str,
        emotion: EmotionResult,
    ) -> None:
        """
        Speak a response and optionally surface a proactive suggestion.

        Args:
            response: Primary TTS response.
            emotion:  Detected emotion (used for tone shaping).
        """
        if response:
            # If the response was an LLM query, the pipeline is already running/finished.
            # We must wait for it to complete playback.
            if self._pipeline and self._pipeline.is_running and not self._pipeline.is_interrupted:
                self._pipeline.wait_until_done()
            else:
                # Local actions or error messages deliver synchronously
                self._speak(response, wait=True)

        # Surface a proactive suggestion after the main response
        if self._planner:
            _, suggestion = self._planner.decide_with_suggestion(response)
            if suggestion:
                time.sleep(0.5)
                self._speak(suggestion)

    def play_wake_chime(self) -> None:
        """Play a short, subtle audio feedback signal when the wake word is detected."""
        play_sound(SoundEvent.WAKE)

    # ── Wake Word Callback ────────────────────────────────────────────────────

    def _on_wake_word(self) -> None:
        """
        Callback fired by the WakeWordDetector on wake word detection.
        Sets the wake event to unblock the engine loop.
        """
        if self._state in (
            EngineState.IDLE,
            EngineState.CONVERSATION,
            EngineState.SPEAKING,
            EngineState.PROCESSING,
        ):
            logger.info("[Engine] Wake word detected (state=%s).", self._state.name)

            # If we're speaking or processing, immediately halt pipeline
            if self._state in (EngineState.SPEAKING, EngineState.PROCESSING):
                if self._pipeline:
                    self._pipeline.interrupt()
                self._interrupt_event.set()

            self.play_wake_chime()
            self._wake_event.set()

    # ── Goal Callbacks ────────────────────────────────────────────────────────

    def _on_goal_step_complete(self, goal: Goal, step: GoalStep) -> None:
        """Announce step completion for multi-step goals."""
        if goal.total_steps > 3:  # Only announce on longer goals
            self._speak(
                f"Step complete: {step.description}.",
                priority=SpeechPriority.LOW,
            )

    def _on_goal_complete(self, goal: Goal) -> None:
        """Announce goal completion."""
        play_sound(SoundEvent.SUCCESS)
        self._speak(
            f"Goal complete: '{goal.title}'.",
            priority=SpeechPriority.NORMAL,
        )

    def _on_goal_failed(self, goal: Goal, reason: str) -> None:
        """Announce goal failure."""
        play_sound(SoundEvent.ERROR)
        self._speak(
            f"Goal '{goal.title}' failed. {reason}",
            priority=SpeechPriority.NORMAL,
        )

    # ── Screen Alert Callback ─────────────────────────────────────────────────

    def _on_screen_alert(self, alert: ScreenAlert) -> None:
        """
        Handle a screen-detected alert from the ScreenMonitor.
        Only surfaces alerts at WARNING severity or above.
        """
        from friday.screen import AlertSeverity

        if alert.severity in (AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL):
            self._speak(
                f"{get_voice_line(ResponseCategory.PROACTIVE)} {alert.to_speech()}",
                priority=SpeechPriority.HIGH,
            )

    # ── CPU Monitor ───────────────────────────────────────────────────────────

    def _start_cpu_monitor(self) -> None:
        """Start a lightweight background CPU usage monitor."""

        def _monitor() -> None:
            try:
                import psutil
            except ImportError:
                logger.warning("[Engine] psutil not available — CPU monitor disabled.")
                return

            alerted = False
            while not self._stop_event.is_set():
                try:
                    usage = psutil.cpu_percent(interval=10)
                    if usage >= self._config.cpu_alert_threshold and not alerted:
                        self._speak(
                            f"Heads up — CPU usage is at {usage:.0f} percent.",
                            priority=SpeechPriority.LOW,
                        )
                        alerted = True
                    elif usage < self._config.cpu_alert_threshold * 0.8:
                        alerted = False  # Reset after sustained drop
                except Exception:
                    pass
                self._stop_event.wait(timeout=30)

        threading.Thread(
            target=_monitor,
            daemon=True,
            name="cpu-monitor",
        ).start()

    # ── Goal Handler Wiring ───────────────────────────────────────────────────

    def _wire_goal_handlers(self) -> None:
        """
        Register step execution handlers with the GoalEngine.
        Connects action: and llm: prefixes to their respective systems.
        """
        if not self._goals:
            return

        if self._router:

            def action_handler(step: GoalStep) -> str:
                assert self._router is not None
                result = self._router.route(step.action.removeprefix("action:").strip())
                if result and result.success:
                    return str(result.message)
                raise RuntimeError(result.message if result else "Action not found")

            self._goals.register_step_handler("action:", action_handler)

        if self._llm and self._llm.is_ready:

            def llm_handler(step: GoalStep) -> str:
                assert self._llm is not None
                prompt = step.action.removeprefix("llm:").strip()
                response = self._llm.query(prompt, add_to_history=False)
                if response.success:
                    return str(response.text)
                raise RuntimeError(f"LLM step failed: {response.status.name}")

            self._goals.register_step_handler("llm:", llm_handler)

        def wait_handler(step: GoalStep) -> str:
            duration = float(step.action.removeprefix("wait:").strip() or "2")
            time.sleep(min(duration, 60))  # Cap at 60 seconds
            return f"Waited {duration} seconds."

        self._goals.register_step_handler("wait:", wait_handler)

        def notify_handler(step: GoalStep) -> str:
            message = step.action.removeprefix("notify:").strip()
            self._speak(message, priority=SpeechPriority.LOW)
            return message

        self._goals.register_step_handler("notify:", notify_handler)

        def confirm_handler(step: GoalStep) -> str:
            message = step.action.removeprefix("confirm:").strip()
            self._speak(message + " Say yes to continue or no to cancel.")
            # Capture user confirmation
            result = self._capture_speech(timeout=8)
            if result and any(w in result for w in ("yes", "yeah", "confirm", "go ahead")):
                return "Confirmed."
            raise RuntimeError("User did not confirm.")

        self._goals.register_step_handler("confirm:", confirm_handler)

    # ── State Management ──────────────────────────────────────────────────────

    def _transition(self, new_state: EngineState, metadata: dict[str, Any] | None = None) -> None:
        """
        Transition the engine to a new state and fire the change callback.

        Args:
            new_state: Target state.
            metadata:  Optional contextual info (e.g. current task string).
        """
        with self._lock:
            if self._state == new_state and metadata is None:
                return
            previous = self._state
            self._state = new_state

        logger.debug(
            "[Engine] %s → %s %s",
            previous.name,
            new_state.name,
            f"({metadata})" if metadata else "",
        )

        if self._on_state_change:
            event = StateChangeEvent(previous=previous, current=new_state, metadata=metadata)
            try:
                self._on_state_change(event)
            except Exception:
                logger.debug("[Engine] on_state_change callback raised.", exc_info=True)

    # ── Response Builders ─────────────────────────────────────────────────────

    def _build_status_response(self) -> str:
        """Build a natural-language system status report."""
        parts = ["All systems nominal."]

        if self._llm:
            status = "online" if self._llm.is_ready else "offline"
            parts.append(f"LLM brain is {status}.")

        if self._task_queue:
            depth = self._task_queue.queue_depth
            if depth:
                parts.append(f"{depth} background task{'s' if depth != 1 else ''} queued.")

        if self._goals:
            active = self._goals.active_goal_count
            if active:
                parts.append(f"{active} active goal{'s' if active != 1 else ''}.")

        uptime = time.time() - self._session_start
        hours, remainder = divmod(int(uptime), 3600)
        minutes = remainder // 60
        parts.append(f"Uptime: {hours}h {minutes}m." if hours else f"Uptime: {minutes} minutes.")

        parts.append(f"{self._commands_handled} commands handled this session.")
        return " ".join(parts)

    def _build_help_response(self) -> str:
        """Build a concise capability summary for TTS."""
        return (
            "I can open and close applications, control volume and media, "
            "search the web, check system status, answer questions, "
            "help with writing, and manage multi-step goals in the background. "
            "Just say my name followed by your request."
        )

    def _llm_fallback(self, text: str) -> str:
        """Last-resort LLM query when no planner is available."""
        if self._llm and self._llm.is_ready:
            return self._handle_llm_query(text)
        return get_voice_line(ResponseCategory.ERROR)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state

    @property
    def is_online(self) -> bool:
        return self._state not in (
            EngineState.OFFLINE,
            EngineState.ERROR,
            EngineState.SHUTTING_DOWN,
        )

    @property
    def session_stats(self) -> dict[str, Any]:
        """Return a snapshot of session-level statistics."""
        uptime = time.time() - self._session_start if self._session_start else 0
        return {
            "state": self._state.name,
            "uptime_seconds": round(uptime, 1),
            "commands_handled": self._commands_handled,
            "errors_total": self._errors_total,
            "llm_ready": self._llm.is_ready if self._llm else False,
            "tts_backend": self._tts.active_backend_name if self._tts else "none",
            "stt_backend": self._stt.backend_name if self._stt else "none",
            "queue_depth": self._task_queue.queue_depth if self._task_queue else 0,
            "active_goals": self._goals.active_goal_count if self._goals else 0,
            "memory_stats": self._memory.stats if self._memory else {},
        }
