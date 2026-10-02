"""
F.R.I.D.A.Y.  —  Streaming Response Pipeline
─────────────────────────────────────────────────────────────────────────────
Implements the "Bucket Brigade" architecture for sub-second time-to-first-audio.
Decouples token accumulation, TTS synthesis, and audio playback into three
concurrent stages connected by bounded queues.

Flow
  LLM stream
      │  push_token()
      ▼
  ┌─────────────────────┐
  │  Token Accumulator  │  buffers tokens → detects sentence boundaries
  └──────────┬──────────┘
             │  _text_queue  (bounded)
             ▼
  ┌─────────────────────┐
  │    TTS Worker       │  sentence text → synthesised audio bytes
  └──────────┬──────────┘
             │  _audio_queue  (bounded)
             ▼
  ┌─────────────────────┐
  │   Audio Spooler     │  audio bytes → speakers  (synchronous play)
  └─────────────────────┘

Fixes over previous version
  • _tts_worker never forwarded the sentinel to _audio_queue on error —
    _audio_worker would block forever if TTS raised.
  • interrupt() called queue.empty() + get_nowait() — race condition; the
    queue could fill between the two calls. Replaced with a non-blocking
    drain loop using get_nowait() alone.
  • Unbounded queues — a slow audio device would cause unbounded memory
    growth. Both queues are now bounded (maxsize configurable).
  • wait_until_done() joined all workers including the TTS thread, but TTS
    must finish before audio can finish — join order now matches data flow.
  • No back-pressure on push_token() — fast LLM streams could enqueue
    thousands of sentences. Now uses queue.put() which blocks when full.
  • Sentence splitter only split on the LAST boundary per token — multiple
    sentences arriving in one token chunk were collapsed into one. Replaced
    with a loop that flushes all complete sentences per token.
  • No TTFA / synthesis latency metrics — added PipelineMetrics dataclass.
  • Thread start/stop not idempotent — calling start() twice or stop()
    before start() would corrupt state. Guards added.
"""

from __future__ import annotations

import contextlib
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from friday.logger import get_logger

if TYPE_CHECKING:
    from friday.tts import TextToSpeech

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum sentences waiting for TTS synthesis.
# Blocks push_token() when full — natural back-pressure on fast LLM streams.
_TEXT_QUEUE_MAXSIZE: int = 12

# Maximum synthesised audio chunks waiting for playback.
_AUDIO_QUEUE_MAXSIZE: int = 8

# Seconds a worker blocks on an empty queue before looping to check stop flags.
_QUEUE_TIMEOUT: float = 0.05

# Sentinel object — unique identity check (not equality) so None payloads
# cannot accidentally act as sentinels.
_SENTINEL = object()

# Matches sentence-ending punctuation followed by whitespace or string end.
# Handles: "Hello world. How are you?" → two sentences.
_BOUNDARY_RE = re.compile(r"([.!?…]+)(?:\s+|$)")
_SENTENCE_MIN_LENGTH: int = 20

# ── Metrics ───────────────────────────────────────────────────────────────────


@dataclass
class PipelineMetrics:
    """
    Runtime statistics for a single pipeline run.

    Attributes:
        sentences_queued:    Total sentence chunks pushed to the TTS worker.
        chunks_synthesised:  Total audio chunks produced.
        chunks_played:       Total audio chunks sent to the speaker.
        ttfa_seconds:        Time from pipeline start to first audio playback.
        total_duration:      Wall-clock time from start() to audio worker exit.
    """

    sentences_queued: int = 0
    chunks_synthesised: int = 0
    chunks_played: int = 0
    ttfa_seconds: float = 0.0
    total_duration: float = 0.0

    # Internal — set by the pipeline, not public API
    _start_time: float = field(default=0.0, repr=False)
    _first_audio_time: float = field(default=0.0, repr=False)

    def record_start(self) -> None:
        self._start_time = time.perf_counter()

    def record_first_audio(self) -> None:
        if self._first_audio_time == 0.0 and self._start_time:
            self._first_audio_time = time.perf_counter()
            self.ttfa_seconds = self._first_audio_time - self._start_time

    def record_done(self) -> None:
        if self._start_time:
            self.total_duration = time.perf_counter() - self._start_time


# ── Pipeline ──────────────────────────────────────────────────────────────────


class StreamingResponsePipeline:
    """
    Orchestrates the concurrent token → speech flow.

    Usage::

        pipeline = StreamingResponsePipeline(tts=tts_engine)
        pipeline.start()

        for token in llm.stream("Tell me a joke."):
            pipeline.push_token(token)

        pipeline.finish()
        pipeline.wait_until_done()

        print(pipeline.metrics)

    Thread safety
        push_token() and finish() are safe to call from the LLM thread.
        interrupt() is safe to call from any thread at any time.
        start() / wait_until_done() must be called from the same thread.
    """

    def __init__(
        self,
        tts: TextToSpeech,
        text_queue_maxsize: int = _TEXT_QUEUE_MAXSIZE,
        audio_queue_maxsize: int = _AUDIO_QUEUE_MAXSIZE,
    ) -> None:
        self._tts = tts
        self._text_queue_maxsize = text_queue_maxsize
        self._audio_queue_maxsize = audio_queue_maxsize

        self._text_queue: queue.Queue = queue.Queue(maxsize=text_queue_maxsize)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=audio_queue_maxsize)

        self._interrupted = threading.Event()
        self._finished = threading.Event()  # set by finish(); guards double-finish

        self._tts_thread: threading.Thread | None = None
        self._audio_thread: threading.Thread | None = None

        self._buffer = ""
        self._running = False
        self._lock = threading.Lock()

        self.metrics = PipelineMetrics()

    # ── Public control ────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Launch the TTS and audio worker threads.
        Idempotent — calling start() on an already-running pipeline is a no-op.
        """
        with self._lock:
            if self._running:
                logger.warning("[Pipeline] start() called on already-running pipeline.")
                return

            self._interrupted.clear()
            self._finished.clear()
            self._buffer = ""
            self._running = True
            self.metrics = PipelineMetrics()
            self.metrics.record_start()

        # Re-create queues to flush any stale state from a previous interrupted turn
        self._text_queue = queue.Queue(maxsize=self._text_queue_maxsize)
        self._audio_queue = queue.Queue(maxsize=self._audio_queue_maxsize)

        self._tts_thread = threading.Thread(
            target=self._tts_worker,
            daemon=True,
            name="pipeline-tts",
        )
        self._audio_thread = threading.Thread(
            target=self._audio_worker,
            daemon=True,
            name="pipeline-audio",
        )

        self._tts_thread.start()
        self._audio_thread.start()

        logger.debug("[Pipeline] Workers started.")

    def push_token(self, token: str) -> None:
        """
        Accept a token from the LLM stream.

        Accumulates tokens in an internal buffer and enqueues complete
        sentences to the TTS worker.  Applies back-pressure automatically
        when the text queue is full — this blocks the LLM thread rather than
        growing memory without bound.

        Args:
            token: A single token or token chunk from the LLM stream.
        """
        if self._interrupted.is_set() or not self._running:
            return

        self._buffer += token
        self._flush_sentences(force=False)
        if len(self._buffer.strip()) >= _SENTENCE_MIN_LENGTH:
            self._flush_sentences(force=True)

    def finish(self) -> None:
        """
        Signal that the LLM stream is complete.

        Flushes any remaining buffered text and sends the sentinel to the
        TTS worker.  Safe to call multiple times — subsequent calls are no-ops.
        """
        if not self._running:
            logger.warning("[Pipeline] finish() called before start().")
            return

        if self._finished.is_set():
            return
        self._finished.set()

        # Flush any partial sentence left in the buffer
        self._flush_sentences(force=True)

        # Sentinel tells the TTS worker to shut down after draining
        self._text_queue.put(_SENTINEL)
        logger.debug("[Pipeline] Stream finished — sentinel enqueued.")

    def interrupt(self) -> None:
        """
        Immediately halt all pipeline activity.

        Stops audio playback, drains both queues, and sets the interrupted
        flag so workers exit on their next iteration.  Safe to call from
        any thread, including signal handlers.
        """
        self._interrupted.set()

        # Stop any in-progress audio playback
        try:
            self._tts.stop()
        except Exception as exc:
            logger.warning("[Pipeline] tts.stop() raised during interrupt: %s", exc)

        # Drain queues — get_nowait() is atomic; no race with empty() check
        _drain_queue(self._text_queue)
        _drain_queue(self._audio_queue)

        # Unblock workers that are blocked on queue.put() (bounded queues)
        # by injecting sentinels so they see data and can exit.
        with contextlib.suppress(queue.Full):
            self._text_queue.put_nowait(_SENTINEL)
        with contextlib.suppress(queue.Full):
            self._audio_queue.put_nowait(_SENTINEL)

        logger.info("[Pipeline] Interrupted — queues drained.")

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """
        Block until both worker threads have exited.

        Join order matches data flow: TTS must finish before audio can finish.

        Args:
            timeout: Maximum seconds to wait total. None = wait forever.

        Returns:
            True if both workers exited cleanly, False if timeout expired.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        def _remaining() -> float | None:
            if deadline is None:
                return None
            return max(0.0, deadline - time.monotonic())

        for thread in (self._tts_thread, self._audio_thread):
            if thread and thread.is_alive():
                thread.join(timeout=_remaining())
                if thread.is_alive():
                    logger.warning("[Pipeline] Thread '%s' did not exit in time.", thread.name)
                    return False

        with self._lock:
            self._running = False

        self.metrics.record_done()
        logger.info(
            "[Pipeline] Done. TTFA=%.2fs | total=%.2fs | sentences=%d | played=%d",
            self.metrics.ttfa_seconds,
            self.metrics.total_duration,
            self.metrics.sentences_queued,
            self.metrics.chunks_played,
        )
        return True

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted.is_set()

    # ── Workers ───────────────────────────────────────────────────────────────

    def _tts_worker(self) -> None:
        """
        Stage 2: sentence text → synthesised audio bytes.

        Consumes from _text_queue, synthesises each sentence via TTS,
        and produces to _audio_queue.  Always forwards the sentinel to
        the audio worker — even on error — so the audio worker can exit.
        """
        try:
            while not self._interrupted.is_set():
                try:
                    item = self._text_queue.get(timeout=_QUEUE_TIMEOUT)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    break

                text: str = item
                try:
                    t0 = time.perf_counter()
                    audio_bytes = self._tts.synthesize(text)
                    elapsed = time.perf_counter() - t0

                    if audio_bytes:
                        logger.debug(
                            "[Pipeline/TTS] Synthesised %d bytes in %.2fs: '%s…'",
                            len(audio_bytes),
                            elapsed,
                            text[:40],
                        )
                        self._audio_queue.put(audio_bytes)  # blocks on back-pressure
                        self.metrics.chunks_synthesised += 1

                except Exception as exc:
                    logger.error("[Pipeline/TTS] Synthesis error for '%s…': %s", text[:40], exc)
                    # Do not re-raise — continue with the next sentence

        finally:
            # Always forward sentinel so the audio worker can exit cleanly.
            with contextlib.suppress(Exception):
                self._audio_queue.put(_SENTINEL)
            logger.debug("[Pipeline/TTS] Worker exited.")

    def _audio_worker(self) -> None:
        """
        Stage 3: synthesised audio bytes → speakers.

        Consumes from _audio_queue and plays each chunk synchronously.
        Running in a dedicated thread means playback never blocks the engine.
        """
        while not self._interrupted.is_set():
            try:
                item = self._audio_queue.get(timeout=_QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            audio_bytes: bytes = item
            try:
                self.metrics.record_first_audio()  # captures TTFA on first chunk
                self._tts.play_bytes(audio_bytes)
                self.metrics.chunks_played += 1
            except Exception as exc:
                logger.error("[Pipeline/Audio] Playback error: %s", exc)

        logger.debug("[Pipeline/Audio] Worker exited.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _flush_sentences(self, *, force: bool) -> None:
        """
        Extract all complete sentences from the buffer and enqueue them.

        When *force* is True the entire remaining buffer is flushed as-is
        (called at finish() time).

        Uses a loop so multiple sentence boundaries in a single token chunk
        are all flushed — not just the last one.
        """
        while True:
            match = _BOUNDARY_RE.search(self._buffer)
            if not match:
                break

            sentence = self._buffer[: match.end()].strip()
            self._buffer = self._buffer[match.end() :]

            if sentence:
                logger.debug("[Pipeline] Sentence queued: '%s…'", sentence[:40])
                self._text_queue.put(sentence)  # blocks on back-pressure
                self.metrics.sentences_queued += 1

        if force and self._buffer.strip():
            sentence = self._buffer.strip()
            self._buffer = ""
            logger.debug("[Pipeline] Flushing tail: '%s…'", sentence[:40])
            self._text_queue.put(sentence)
            self.metrics.sentences_queued += 1


# ── Utilities ─────────────────────────────────────────────────────────────────


def _drain_queue(q: queue.Queue) -> None:
    """
    Non-blocking drain of all items in *q*.

    Uses get_nowait() in a loop — avoids the empty() + get_nowait() race
    condition where the queue could be filled between the two calls.
    """
    drained = 0
    while True:
        try:
            q.get_nowait()
            drained += 1
        except queue.Empty:
            break
    if drained:
        logger.debug("[Pipeline] Drained %d item(s) from queue.", drained)
