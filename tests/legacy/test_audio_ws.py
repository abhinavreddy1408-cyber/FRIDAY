import json

from fastapi.testclient import TestClient


def _build_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("ENABLE_REAL_AUDIO", "0")
    monkeypatch.setenv("VAD_PROVIDER", "energy")
    monkeypatch.setenv("VAD_SILENCE_MS", "300")
    monkeypatch.setenv("VAD_CHUNK_MS", "100")
    import app.config
    import app.server.api as api

    app.config.get_settings.cache_clear()  # type: ignore
    api.app = None
    return TestClient(api.build_app())


def _pcm_frame(amplitude: int, samples: int = 1600) -> bytes:
    # 16-bit PCM mono frame (~100ms at 16kHz).
    import struct

    return struct.pack("<" + "h" * samples, *([amplitude] * samples))


def test_audio_ws_end_to_end(monkeypatch):
    client = _build_client(monkeypatch)
    with client.websocket_connect("/audio/stream") as ws:
        # speech frames
        ws.send_bytes(_pcm_frame(2000))
        ws.send_bytes(_pcm_frame(2000))
        # silence frames should auto-trigger processing
        ws.send_bytes(_pcm_frame(0))
        ws.send_bytes(_pcm_frame(0))
        ws.send_bytes(_pcm_frame(0))
        start = json.loads(ws.receive_text())
        assert start["event"] == "processing_start"
        # At least one synthesized audio frame should be returned
        first_audio = ws.receive_bytes()
        assert isinstance(first_audio, (bytes, bytearray))
        # Keep receiving until processing_end text appears
        while True:
            message = ws.receive()
            if message.get("text"):
                payload = json.loads(message["text"])
                if payload.get("event") == "processing_end":
                    break


def test_audio_ws_invalid_control(monkeypatch):
    client = _build_client(monkeypatch)
    with client.websocket_connect("/audio/stream") as ws:
        ws.send_text("not-json")
        err = json.loads(ws.receive_text())
        assert err["event"] == "error"
        assert err["message"] == "invalid_json_control"


def test_audio_ws_barge_in_interrupts(monkeypatch):
    import asyncio

    import app.config
    import app.server.api as api

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("ENABLE_REAL_AUDIO", "0")
    monkeypatch.setenv("VAD_SILENCE_MS", "300")
    monkeypatch.setenv("VAD_CHUNK_MS", "100")
    app.config.get_settings.cache_clear()  # type: ignore
    api.app = None

    class _VAD:
        def is_speech(self, frame_bytes: bytes) -> bool:
            return any(frame_bytes)

    class SlowAudioService:
        vad_detector = _VAD()

        async def handle_audio_turn(self, audio_bytes, assistant, conversation_id=None, cancel_event=None):
            for _ in range(20):
                if cancel_event and cancel_event.is_set():
                    raise asyncio.CancelledError
                await asyncio.sleep(0.01)
                yield b"a"

    monkeypatch.setattr(api, "get_audio_service", lambda: SlowAudioService())
    client = TestClient(api.build_app())
    with client.websocket_connect("/audio/stream") as ws:
        # first utterance
        ws.send_bytes(b"\x01" * 3200)
        ws.send_bytes(_pcm_frame(0))
        ws.send_bytes(_pcm_frame(0))
        ws.send_bytes(_pcm_frame(0))
        start = json.loads(ws.receive_text())
        assert start["event"] == "processing_start"
        _ = ws.receive_bytes()
        # new speech while potentially still streaming should interrupt
        ws.send_bytes(b"\x01" * 3200)
        interrupted = None
        for _ in range(20):
            message = ws.receive()
            if message.get("text"):
                payload = json.loads(message["text"])
                if payload.get("type") == "interrupted":
                    interrupted = payload
                    break
        assert interrupted is not None
