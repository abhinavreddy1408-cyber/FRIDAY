import time

from friday.streaming import PipelineMetrics, StreamingResponsePipeline, _drain_queue
from friday.task_queue import TaskPriority, TaskQueue, TaskResult, TaskStatus


class FakeTTS:
    def __init__(self, fail_text=""):
        self.fail_text = fail_text
        self.synthesized = []
        self.played = []
        self.stopped = False

    def synthesize(self, text):
        self.synthesized.append(text)
        if self.fail_text and self.fail_text in text:
            raise RuntimeError("tts failed")
        return f"audio:{text}".encode()

    def play_bytes(self, audio):
        self.played.append(audio)

    def stop(self):
        self.stopped = True


def test_pipeline_metrics_records_start_audio_and_done():
    metrics = PipelineMetrics()
    metrics.record_start()
    metrics.record_first_audio()
    metrics.record_done()

    assert metrics.ttfa_seconds >= 0
    assert metrics.total_duration >= metrics.ttfa_seconds


def test_streaming_pipeline_splits_multiple_sentences_and_plays_audio():
    tts = FakeTTS()
    pipeline = StreamingResponsePipeline(tts, text_queue_maxsize=4, audio_queue_maxsize=4)

    pipeline.start()
    pipeline.push_token("Hello there. Second sentence? Tail")
    pipeline.finish()

    assert pipeline.wait_until_done(timeout=2.0) is True
    assert tts.synthesized == ["Hello there.", "Second sentence?", "Tail"]
    assert len(tts.played) == 3
    assert pipeline.metrics.sentences_queued == 3
    assert pipeline.metrics.chunks_synthesised == 3
    assert pipeline.metrics.chunks_played == 3
    assert pipeline.is_running is False


def test_streaming_pipeline_ignores_double_start_finish_and_tts_errors():
    tts = FakeTTS(fail_text="bad")
    pipeline = StreamingResponsePipeline(tts)

    pipeline.finish()
    pipeline.start()
    pipeline.start()
    pipeline.push_token("bad sentence. good sentence.")
    pipeline.finish()
    pipeline.finish()

    assert pipeline.wait_until_done(timeout=2.0) is True
    assert tts.synthesized == ["bad sentence.", "good sentence."]
    assert len(tts.played) == 1


def test_streaming_pipeline_interrupt_drains_queues_and_stops_tts():
    tts = FakeTTS()
    pipeline = StreamingResponsePipeline(tts)
    pipeline.start()
    pipeline.push_token("Stop now.")

    pipeline.interrupt()
    assert pipeline.is_interrupted is True
    assert tts.stopped is True
    assert pipeline.wait_until_done(timeout=2.0) is True


def test_drain_queue_removes_all_items():
    import queue

    q = queue.Queue()
    q.put("one")
    q.put("two")

    _drain_queue(q)

    assert q.empty()


def test_task_result_string_success_and_failure():
    success = TaskResult("1", "ok", TaskStatus.COMPLETED, 7, None, 1.0, 2.5)
    failure = TaskResult("2", "bad", TaskStatus.FAILED, None, ValueError("boom"), 1.0, 2.0)

    assert success.success is True
    assert success.duration == 1.5
    assert "completed" in str(success)
    assert "boom" in str(failure)


def test_task_queue_executes_success_failure_callbacks_and_history():
    completed = []
    errors = []
    queue = TaskQueue(max_workers=2, rate_per_sec=100, burst_size=10, history_limit=10)
    queue.start()
    try:
        ok_id = queue.submit(
            fn=lambda x: x + 1,
            args=(4,),
            priority=TaskPriority.HIGH,
            name="increment",
            on_complete=completed.append,
        )

        def fail():
            raise RuntimeError("failed task")

        bad_id = queue.submit(
            fn=fail,
            name="fail",
            on_error=lambda exc: errors.append(str(exc)),
        )

        ok_result = queue.wait_for(ok_id, timeout=3.0, poll_interval=0.01)
        bad_result = queue.wait_for(bad_id, timeout=3.0, poll_interval=0.01)

        assert ok_result.success is True
        assert ok_result.value == 5
        assert bad_result.status == TaskStatus.FAILED
        assert completed == [5]
        assert errors == ["failed task"]
        assert queue.get_result(ok_id).value == 5
        assert queue.get_status(bad_id) == TaskStatus.FAILED
        assert len(queue.get_history(limit=5)) >= 2
        assert queue.stats["history_count"] >= 2
    finally:
        queue.stop()


def test_task_queue_cancel_all_and_compatibility_lists():
    queue = TaskQueue(max_workers=1, rate_per_sec=0.01, burst_size=1, history_limit=5)
    queue.start()
    try:
        first = queue.submit(lambda: "first", name="first")
        second = queue.submit(lambda: "second", priority=TaskPriority.LOW, name="second")
        third = queue.submit(lambda: "third", priority=TaskPriority.BACKGROUND, name="third")
        time.sleep(0.05)

        cancelled_low = queue.cancel_all(priority=TaskPriority.LOW)

        assert cancelled_low in {0, 1}
        assert first in queue.list_active_tasks() or queue.get_status(first) in {
            TaskStatus.RUNNING,
            TaskStatus.COMPLETED,
        }
        assert queue.get_status(second) in {TaskStatus.CANCELLED, TaskStatus.PENDING, TaskStatus.RUNNING}
        assert queue.get_status(third) in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED}
    finally:
        queue.stop(wait=False)


def test_task_queue_submit_requires_start_and_unknown_wait_times_out():
    queue = TaskQueue()

    try:
        queue.submit(lambda: None)
    except RuntimeError as exc:
        assert "must be started" in str(exc)
    else:
        raise AssertionError("submit should require start")

    assert queue.wait_for("missing", timeout=0.01, poll_interval=0.001) is None
    assert queue.cancel("missing") is False
