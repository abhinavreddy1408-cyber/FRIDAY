"""
F.R.I.D.A.Y  —  Swarm Orchestration Module
─────────────────────────────────────────────────────────────────────────────
Architecture
  SwarmOrchestrator  ──owns──▶  SwarmBus (internal queue)
                     ──owns──▶  ThreadPoolExecutor
                     ──owns──▶  Dict[Specialist → BaseAgent]

  BaseAgent          ──owns──▶  LLMBrain  (one client per specialist, reused)
                     ──borrows─▶ shared LongTermMemory handle (injected)

Task lifecycle
  PENDING ──▶ IN_PROGRESS ──▶ COMPLETED
                           └─▶ FAILED
                           └─▶ TIMED_OUT

Thread safety
  - SwarmBus is private to the orchestrator (no module-level global).
  - LongTermMemory is injected once and shared; its internal lock handles
    concurrent writes (WAL mode assumed in manager.py).
  - Result store uses a plain dict protected by a threading.Lock so
    callers can poll without touching the DB.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from friday.config import config
from friday.llm import LLMBrain
from friday.memory.manager import LongTermMemory

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# For backward compatibility and testing
SwarmBus = queue.Queue


# ── Constants ─────────────────────────────────────────────────────────────────

# How long (seconds) an agent is allowed to run before it is marked TIMED_OUT.
DEFAULT_TASK_TIMEOUT: float = 120.0

# Maximum number of concurrent specialist threads.
SWARM_POOL_SIZE: int = 5


# ── Task Status ───────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


# ── Specialist Registry ────────────────────────────────────────────────────────


class Specialist(str, Enum):
    CODER         = "CoderAgent"
    SECURITY      = "SecurityAgent"
    RESEARCH      = "ResearchAgent"
    REASONING     = "ReasoningAgent"
    BRAINSTORMING = "BrainstormingAgent"
    PRODUCTIVITY  = "ProductivityAgent"
    MEDIA         = "MediaAgent"
    RETRIEVAL     = "RetrievalAgent"
    VISION        = "VisionAgent"
    BROWSER       = "BrowserAgent"
    MULTILINGUAL  = "MultilingualAgent"  # NEW


# ── Task Payload ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskPayload:
    """
    Immutable description of a delegated task.

    Attributes:
        task_id:    UUID string; used as the primary key in the DB.
        specialist: Which specialist should handle this task.
        prompt:     The full instruction for the specialist.
        history:    Optional conversation turns for context injection.
        created_at: Unix timestamp of submission.
        timeout:    Maximum execution time in seconds.
    """

    task_id: str
    specialist: Specialist
    prompt: str
    history: tuple[dict[str, str], ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)
    timeout: float = DEFAULT_TASK_TIMEOUT


# ── Task Result ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskResult:
    """
    Immutable result produced when a task reaches a terminal state.

    Attributes:
        task_id:    Matches the originating TaskPayload.task_id.
        status:     Terminal status of the task.
        output:     The specialist's response text, or an error message.
        duration:   Wall-clock seconds from pickup to completion.
    """

    task_id: str
    status: TaskStatus
    output: str
    duration: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED


@dataclass(frozen=True)
class MergeResult:
    """Result of merge operation with source tracking."""
    output: str
    source: str  # "synthesis" | "ranked_fallback" | "raw_concat" | "empty"


def _merge_results(
    agent_results: list[Any],
    original_input: str = "",
    verification_results: Any = None,
    synthesis_callable: Callable[[str], str | None] | None = None,
) -> str:
    """Merge Execution Chain outputs using the PDF conflict-resolution order."""
    result = _merge_results_tracked(
        agent_results,
        original_input,
        verification_results,
        synthesis_callable,
    )
    return result.output


def _merge_results_tracked(
    agent_results: list[Any],
    original_input: str = "",
    verification_results: Any = None,
    synthesis_callable: Callable[[str], str | None] | None = None,
) -> MergeResult:
    """
    Merge Execution Chain outputs with full fallback tracking.
    Fallback chain: synthesis -> ranked -> raw_concat -> empty
    """
    normalized = [_normalise_agent_result(result) for result in agent_results]
    normalized = [result for result in normalized if result.get("output", "").strip()]

    if not normalized:
        return MergeResult(output="", source="empty")

    if len(normalized) == 1:
        return MergeResult(output=normalized[0]["output"], source="synthesis")

    if synthesis_callable:
        try:
            prompt = _build_synthesis_prompt(normalized, original_input, verification_results)
            synthesized = synthesis_callable(prompt)
            if synthesized and synthesized.strip():
                return MergeResult(output=synthesized.strip(), source="synthesis")
        except Exception as exc:
            logger.warning("[Swarm][Merge] Synthesis callable failed: %s", exc)

    try:
        sorted_results = sorted(
            normalized,
            key=lambda item: (
                float(item.get("confidence") or 0.0),
                _tier_rank(str(item.get("model_tier") or "")),
            ),
            reverse=True,
        )
        ranked_output = "\n\n".join(
            f"{item['agent']}: {item['output']}" for item in sorted_results
        )
        if ranked_output.strip():
            return MergeResult(output=ranked_output, source="ranked_fallback")
    except Exception as exc:
        logger.warning("[Swarm][Merge] Ranked merge failed: %s", exc)

    try:
        raw_parts = []
        for item in normalized:
            raw_parts.append(f"[{item['agent']} - unverified]\n{item['output']}")
        raw_output = "\n\n---\n\n".join(raw_parts)
        if raw_output.strip():
            return MergeResult(output=raw_output, source="raw_concat")
    except Exception as exc:
        logger.error("[Swarm][Merge] Raw concat failed: %s", exc)

    return MergeResult(output="", source="empty")


def _normalise_agent_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {
            "agent": str(result.get("agent", "agent")),
            "output": str(result.get("output", "")),
            "confidence": result.get("confidence"),
            "model_tier": result.get("model_tier", ""),
        }
    return {
        "agent": str(getattr(result, "agent", getattr(result, "task_id", "agent"))),
        "output": str(getattr(result, "output", "")),
        "confidence": getattr(result, "confidence", None),
        "model_tier": getattr(result, "model_tier", ""),
    }


def _build_synthesis_prompt(
    normalized_results: list[dict[str, Any]],
    original_input: str,
    verification_results: Any,
) -> str:
    body = [
        "Merge these agent outputs into one coherent response.",
        "Resolve conflicts by preferring higher confidence, then higher model tier.",
        "Eliminate redundancy and preserve the answer most consistent with the original input.",
        "",
        f"Original input:\n{original_input}",
        "",
        f"Verification:\n{verification_results}",
        "",
    ]
    for result in normalized_results:
        body.append(
            f"Agent: {result['agent']}\n"
            f"Confidence: {result.get('confidence')}\n"
            f"Tier: {result.get('model_tier')}\n"
            f"Output:\n{result['output']}\n"
        )
    return "\n".join(body)


def _tier_rank(tier: str) -> int:
    token = tier.lower()
    if "flagship" in token:
        return 3
    if "high" in token or "medium" in token:
        return 2
    if "specialist" in token or "fast" in token:
        return 1
    return 0


# ── Base Agent ────────────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """
    Abstract base for all swarm specialists.
    Each specialist creates its own LLMBrain wrapper (for isolated history)
    but shares the injected OpenAI client (for shared socket pool).
    """

    def __init__(
        self,
        report_callback: Callable[[TaskResult], None],
        memory: LongTermMemory,
        shared_client: OpenAI | None = None,
        nim_model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.name = self.__class__.__name__
        self._report_cb = report_callback
        self._memory = memory

        # Create own LLMBrain, but inject the shared socket pool
        self._llm = LLMBrain(shared_client=shared_client)
        if not self._llm.setup():
            logger.error("[%s] LLMBrain.setup() failed — tasks will error.", self.name)

        if system_prompt:
            self._llm.set_system_prompt(system_prompt)

        self._nim_model = nim_model or "qwen2.5:3b-instruct-q4_K_M"
        self._model = self._nim_model
        logger.info("[%s] Initialised. Model source: %s", self.name, self._nim_model)

    def run_task(self, payload: TaskPayload) -> TaskResult:
        logger.info("[%s] Starting task %s", self.name, payload.task_id)
        t_start = time.perf_counter()

        self._persist(payload.task_id, TaskStatus.IN_PROGRESS)

        result_holder: list[str] = []
        error_holder: list[str] = []

        def _target() -> None:
            try:
                result_holder.append(self.execute(payload))
            except Exception as exc:
                logger.exception("[%s] execute() raised: %s", self.name, exc)
                error_holder.append(str(exc))

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(timeout=payload.timeout)

        duration = time.perf_counter() - t_start

        if worker.is_alive():
            logger.error(
                "[%s] Task %s exceeded timeout (%.0fs). Marking TIMED_OUT.",
                self.name, payload.task_id, payload.timeout,
            )
            task_result = TaskResult(
                task_id=payload.task_id,
                status=TaskStatus.TIMED_OUT,
                output=f"Sir, the {self.name} timed out after {payload.timeout:.0f}s.",
                duration=duration,
            )
        elif error_holder:
            task_result = TaskResult(
                task_id=payload.task_id,
                status=TaskStatus.FAILED,
                output=f"Sir, the {self.name} encountered an error: {error_holder[0]}",
                duration=duration,
            )
        else:
            task_result = TaskResult(
                task_id=payload.task_id,
                status=TaskStatus.COMPLETED,
                output=result_holder[0] if result_holder else "",
                duration=duration,
            )

        self._persist(payload.task_id, task_result.status, task_result.output)

        logger.info("[%s] Task %s → %s in %.2fs", self.name, payload.task_id, task_result.status.value, duration)

        try:
            self._report_cb(task_result)
        except Exception as cb_exc:
            logger.error("[%s] report_callback raised: %s", self.name, cb_exc)

        return task_result

    @abstractmethod
    def execute(self, payload: TaskPayload) -> str:
        pass

    def _query(self, prompt: str, *, temperature: float | None = None, ensemble_type: str | None = None) -> str:
        response = self._llm.query(prompt, temperature=temperature, ensemble_type=ensemble_type)
        if not response.success:
            raise RuntimeError(f"LLM returned {response.status.name}")
        return response.text

    def _persist(self, task_id: str, status: TaskStatus, output: str | None = None) -> None:
        try:
            if output is not None:
                self._memory.update_task(task_id, status.value, output)
            else:
                self._memory.update_task(task_id, status.value)
        except Exception as exc:
            logger.warning("[%s] DB persist failed for %s: %s", self.name, task_id, exc)


# ── Specialist Implementations ────────────────────────────────────────────────


class CoderAgent(BaseAgent):
    _SYSTEM_PROMPT = (
        "You are the F.R.I.D.A.Y. Coder Agent — a world-class software engineer. "
        "Provide production-ready, well-commented code solutions. "
        "When debugging, reason step-by-step before proposing a fix. "
        "Respond concisely; omit filler. Default language: Python unless specified."
    )

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(
            report_callback,
            memory,
            shared_client,
            nim_model="qwen2.5-coder:3b-instruct-q4_K_M",
            system_prompt=self._SYSTEM_PROMPT,
        )

    def execute(self, payload: TaskPayload) -> str:
        return self._query(payload.prompt, ensemble_type="coding")


class SecurityAgent(BaseAgent):
    _SYSTEM_PROMPT = (
        "You are the F.R.I.D.A.Y. Security Agent. "
        "You specialise in network overwatch, vulnerability assessment, "
        "and post-quantum cryptography key monitoring. "
        "Your tone is clinical, precise, and risk-aware. "
        "Always quantify severity (CVSS where applicable) and recommend mitigations."
    )

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(
            report_callback,
            memory,
            shared_client,
            nim_model="qwen2.5:3b-instruct-q4_K_M",
            system_prompt=self._SYSTEM_PROMPT,
        )

    def execute(self, payload: TaskPayload) -> str:
        enriched = f"Perform intensive security analysis:\n\n{payload.prompt}"
        return self._query(enriched, temperature=0.2, ensemble_type="reasoning")


class ResearchAgent(BaseAgent):
    _SYSTEM_PROMPT = (
        "You are the F.R.I.D.A.Y. Research Agent. "
        "You specialise in synthesising large volumes of data into concise, "
        "actionable reports. Cite sources where possible. "
        "Structure output with a brief summary, key findings, and recommended actions."
    )

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(
            report_callback,
            memory,
            shared_client,
            nim_model="qwen2.5:3b-instruct-q4_K_M",
            system_prompt=self._SYSTEM_PROMPT,
        )

    def execute(self, payload: TaskPayload) -> str:
        try:
            from friday.skills import research_skill
            research_skill.start_background_research(payload.prompt)
        except Exception as exc:
            logger.warning("[ResearchAgent] Background research indexing failed: %s", exc)

        summary = self._query(
            f"Synthesise what is known about the following and outline a research plan:\n\n{payload.prompt}",
            ensemble_type="search",
        )
        return f"{summary}\n\n[Background indexing initiated — findings will be available in semantic memory.]"


class ReasoningAgent(BaseAgent):
    _SYSTEM_PROMPT = "You are the F.R.I.D.A.Y. Reasoning Agent. Solve complex logical problems step-by-step."

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(report_callback, memory, shared_client, system_prompt=self._SYSTEM_PROMPT)

    def execute(self, payload: TaskPayload) -> str:
        return self._query(payload.prompt, ensemble_type="reasoning")


class BrainstormingAgent(BaseAgent):
    _SYSTEM_PROMPT = "You are the F.R.I.D.A.Y. Brainstorming Agent. Generate diverse creative ideas."

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(report_callback, memory, shared_client, system_prompt=self._SYSTEM_PROMPT)

    def execute(self, payload: TaskPayload) -> str:
        return self._query(payload.prompt, ensemble_type="brainstorming")


class ProductivityAgent(BaseAgent):
    _SYSTEM_PROMPT = "You are the F.R.I.D.A.Y. Productivity Agent. Help draft documents and summaries."

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(report_callback, memory, shared_client, system_prompt=self._SYSTEM_PROMPT)

    def execute(self, payload: TaskPayload) -> str:
        return self._query(payload.prompt, ensemble_type="document")


class MediaAgent(BaseAgent):
    _SYSTEM_PROMPT = "You are the F.R.I.D.A.Y. Media Agent."

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(report_callback, memory, shared_client, system_prompt=self._SYSTEM_PROMPT)

    def execute(self, payload: TaskPayload) -> str:
        try:
            from friday.skills import media_skill
            media_skill.generate_image(payload.prompt)
        except Exception as e:
            logger.warning(f"[MediaAgent] Failed to launch media generation: {e}")
        return f"[Image generation initiated for prompt: '{payload.prompt}']"


class RetrievalAgent(BaseAgent):
    _SYSTEM_PROMPT = "You are the F.R.I.D.A.Y. Retrieval Agent. Analyze files and extract information."

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(report_callback, memory, shared_client, system_prompt=self._SYSTEM_PROMPT)

    def execute(self, payload: TaskPayload) -> str:
        return self._query(payload.prompt, ensemble_type="file_analysis")


class BrowserAgent(BaseAgent):
    _SYSTEM_PROMPT = """FRIDAY BROWSER AGENT — SYSTEM PROMPT v2.0
==========================================
IDENTITY & ROLE
---------------
You are FRIDAY's Browser Agent — a specialized autonomous sub-agent responsible for
all browser-based task execution within the FRIDAY multi-agent ecosystem. You operate
as a precise, methodical, human-like web operator. You do not answer general questions,
generate content, or perform tasks outside browser interaction. You receive tasks from
FRIDAY's Planner Agent, execute them on the live web, and return structured results.

You are not a search wrapper. You are not an API caller. You are a full browser
operator capable of navigating, interacting with, extracting from, and verifying
real websites exactly as a skilled human user would — carefully, deliberately, and
without assumptions.

CORE OPERATING PRINCIPLES
--------------------------
1. OBSERVE BEFORE ACTING
2. ACT LIKE A CAREFUL HUMAN
3. VERIFY EVERY ACTION
4. NEVER INVENT OR HALLUCINATE
5. OPERATE WITHIN BOUNDARIES

OUTPUT FORMAT
-------------
STATUS: [SUCCESS | PARTIAL | BLOCKED | FAILED]
TASK SUMMARY: [One sentence describing what was accomplished]
RESULT: [Structured data or extracted content]
ACTIONS TAKEN: [Ordered list of actions performed]
BLOCKERS (if any): [Exact description of blockers]
VERIFICATION: [How the final result was verified]
"""

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(
            report_callback,
            memory,
            shared_client,
            nim_model="qwen2.5:3b-instruct-q4_K_M",
            system_prompt=self._SYSTEM_PROMPT,
        )

    def execute(self, payload: TaskPayload) -> str:
        return self._query(payload.prompt, ensemble_type="screen_interaction")


class VisionAgent(BaseAgent):
    _SYSTEM_PROMPT = "You are the F.R.I.D.A.Y. Vision Agent. Analyze screen context and layout."

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(report_callback, memory, shared_client, system_prompt=self._SYSTEM_PROMPT)

    def execute(self, payload: TaskPayload) -> str:
        return self._query(payload.prompt, ensemble_type="screen_interaction")


class MultilingualAgent(BaseAgent):
    """
    Handles non-English queries by processing and responding in the user's language.
    """
    _SYSTEM_PROMPT = """You are the F.R.I.D.A.Y. Multilingual Agent.
You are a polyglot specialist who can understand and respond in any language.
OPERATING RULES:
1. Detect the language of the user's input.
2. Process the query in its original language — do NOT translate to English first.
3. Respond in the SAME language as the input.
4. If the query requires specialist knowledge (coding, research, etc.), handle it directly.
5. Maintain natural, fluent grammar and vocabulary in the response language.
You are NOT a translation service. You are an intelligent assistant that happens
to be fluent in every language. Think and respond natively in whatever language
the user speaks to you."""

    def __init__(self, report_callback: Callable[[TaskResult], None], memory: LongTermMemory, shared_client: OpenAI | None = None) -> None:
        super().__init__(report_callback, memory, shared_client, system_prompt=self._SYSTEM_PROMPT)

    def execute(self, payload: TaskPayload) -> str:
        return self._query(payload.prompt, ensemble_type="multilingual")


# ── Agent Factory ─────────────────────────────────────────────────────────────

_AGENT_CLASSES: dict[Specialist, type[BaseAgent]] = {
    Specialist.CODER: CoderAgent,
    Specialist.SECURITY: SecurityAgent,
    Specialist.RESEARCH: ResearchAgent,
    Specialist.REASONING: ReasoningAgent,
    Specialist.BRAINSTORMING: BrainstormingAgent,
    Specialist.PRODUCTIVITY: ProductivityAgent,
    Specialist.MEDIA: MediaAgent,
    Specialist.RETRIEVAL: RetrievalAgent,
    Specialist.VISION: VisionAgent,
    Specialist.BROWSER: BrowserAgent,
    Specialist.MULTILINGUAL: MultilingualAgent,
}


# ── Swarm Orchestrator ────────────────────────────────────────────────────────


class SwarmOrchestrator:
    """
    Manages the swarm bus, thread pool, and specialist agent lifecycle.
    Creates a single OpenAI client and injects it into all specialist agents
    so they share one socket pool but maintain isolated conversation histories.
    """

    def __init__(
        self,
        report_callback: Callable[[TaskResult], None] | None = None,
        pool_size: int = SWARM_POOL_SIZE,
        task_timeout: float = DEFAULT_TASK_TIMEOUT,
        shared_client: OpenAI | None = None,
        **kwargs: Any,
    ) -> None:
        cb = report_callback or kwargs.get("engine_report_callback")
        if cb is None:
            raise TypeError("Missing required argument: report_callback or engine_report_callback")
        self._report_cb = cb
        self._task_timeout = task_timeout

        # Create the shared OpenAI client (1 socket pool for all agents)
        self._shared_client: OpenAI | None = None
        if shared_client is not None:
            self._shared_client = shared_client
            logger.info("[Swarm] Using injected shared OpenAI client.")
        else:
            try:
                self._shared_client = OpenAI(
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url,
                    timeout=config.llm.timeout,
                    max_retries=1,
                )
                logger.info("[Swarm] Created new shared OpenAI client for swarm.")
            except Exception as exc:
                logger.error("[Swarm] Shared client creation failed: %s", exc)
                self._shared_client = None

        # Shared DB handle
        self._memory = LongTermMemory()

        # Pass shared CLIENT to all agents (each gets its own LLMBrain + History)
        self._agents: dict[Specialist, BaseAgent] = {
            spec: cls(
                report_callback=self._on_task_complete,
                memory=self._memory,
                shared_client=self._shared_client,
            )
            for spec, cls in _AGENT_CLASSES.items()
        }

        self._bus: SwarmBus[TaskPayload] = SwarmBus()
        self._results: dict[str, TaskResult] = {}
        self._results_lock: threading.Lock = threading.Lock()
        self._futures: dict[str, Future[TaskResult]] = {}
        self._futures_lock: threading.Lock = threading.Lock()

        self._executor = ThreadPoolExecutor(
            max_workers=pool_size,
            thread_name_prefix="SwarmAgent",
        )
        self._running = False
        self._monitor_thread: threading.Thread | None = None

    @property
    def shared_client(self) -> OpenAI | None:
        """Access the shared OpenAI client instance."""
        return self._shared_client

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            logger.warning("[Swarm] start() called on already-running orchestrator.")
            return
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="SwarmMonitor",
        )
        self._monitor_thread.start()
        logger.info(
            "[Swarm] Orchestrator online. Specialists: %s",
            ", ".join(s.value for s in self._agents),
        )

    def stop(self, graceful: bool = True) -> None:
        logger.info("[Swarm] Shutting down (graceful=%s)…", graceful)
        self._running = False
        self._executor.shutdown(wait=graceful, cancel_futures=not graceful)
        logger.info("[Swarm] Orchestrator offline.")

    # ── Public API ────────────────────────────────────────────────────────────

    def delegate(
        self,
        specialist: Specialist,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        timeout: float | None = None,
    ) -> str:
        if not self._running:
            logger.warning("[Swarm] delegate() called before start(). Task queued.")

        task_id = str(uuid.uuid4())
        payload = TaskPayload(
            task_id=task_id,
            specialist=specialist,
            prompt=prompt,
            history=tuple(history or []),
            timeout=timeout if timeout is not None else self._task_timeout,
        )

        try:
            self._memory.create_task(task_id, specialist.value, prompt)
        except Exception as exc:
            logger.warning("[Swarm] Failed to persist PENDING record for %s: %s", task_id, exc)

        self._bus.put(payload)
        logger.info("[Swarm] Task %s queued for %s.", task_id, specialist.value)
        return task_id

    def get_result(self, task_id: str) -> TaskResult | None:
        with self._results_lock:
            return self._results.get(task_id)

    def await_result(self, task_id: str, timeout: float | None = None) -> TaskResult | None:
        with self._futures_lock:
            future = self._futures.get(task_id)

        if future is None:
            return self.get_result(task_id)

        try:
            return future.result(timeout=timeout)
        except Exception:
            return self.get_result(task_id)

    def merge_results(
        self,
        agent_results: list[Any],
        original_input: str = "",
        verification_results: Any = None,
        synthesis_callable: Callable[[str], str | None] | None = None,
    ) -> str:
        """Synthesis Chain merge helper. Never returns None."""
        result = _merge_results_tracked(
            agent_results,
            original_input=original_input,
            verification_results=verification_results,
            synthesis_callable=synthesis_callable,
        )

        if result.source != "synthesis":
            logger.info("[Swarm][Merge] Used fallback: %s (results: %d)", result.source, len(agent_results))

        return result.output

    @property
    def pending_count(self) -> int:
        return self._bus.qsize()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Drain the task bus and submit each payload to the thread pool."""
        while self._running:
            try:
                payload = self._bus.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                agent = self._agents.get(payload.specialist)
                if agent is None:
                    logger.error(
                        "[Swarm] No agent registered for specialist '%s'. Skipping task %s.",
                        payload.specialist, payload.task_id,
                    )
                    with contextlib.suppress(Exception):
                        self._memory.update_task(
                            payload.task_id, TaskStatus.FAILED.value,
                            f"No agent for specialist {payload.specialist}",
                        )
                    self._bus.task_done()
                    continue

                try:
                    future = self._executor.submit(agent.run_task, payload)
                    with self._futures_lock:
                        self._futures[payload.task_id] = future
                except Exception as submit_exc:
                    logger.error(
                        "[Swarm] executor.submit failed for task %s: %s",
                        payload.task_id, submit_exc,
                    )
                    with contextlib.suppress(Exception):
                        self._memory.update_task(
                            payload.task_id, TaskStatus.FAILED.value,
                            f"Thread pool submit failed: {submit_exc}",
                        )

            except Exception as loop_exc:
                logger.error(
                    "[Swarm] Unexpected error in dispatch loop for task %s: %s",
                    getattr(payload, 'task_id', 'unknown'), loop_exc,
                )
            finally:
                self._bus.task_done()

    def _on_task_complete(self, result: TaskResult) -> None:
        """Internal callback wired into every agent."""
        with self._results_lock:
            self._results[result.task_id] = result

        try:
            self._report_cb(result)
        except Exception as exc:
            logger.error("[Swarm] report_callback raised: %s", exc)
