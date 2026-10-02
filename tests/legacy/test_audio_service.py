import pytest
from app.audio.providers import MockSTT, MockTTS
from app.audio.service import AudioService


class FakeAssistant:
    async def stream_user_message(self, text: str, conversation_id=None):
        yield {"type": "token", "data": "hello "}
        yield {"type": "tool_start", "tool_name": "recall"}
        yield {"type": "token", "data": "world"}
        yield {"type": "tool_end", "tool_name": "recall", "duration": 0.1, "status": "ok"}
        yield {"type": "done"}


@pytest.mark.asyncio
async def test_audio_service_roundtrip_filters_non_token():
    service = AudioService(stt_provider=MockSTT("hi"), tts_provider=MockTTS())
    assistant = FakeAssistant()
    chunks = []
    async for chunk in service.handle_audio_turn(b"fake-bytes", assistant):
        chunks.append(chunk)
    assert b"hello " in chunks
    assert b"world" in chunks
    combined = b"".join(chunks)
    assert b"tool_start" not in combined
    assert b"tool_end" not in combined


@pytest.mark.asyncio
async def test_audio_service_stt_failure():
    class BrokenSTT(MockSTT):
        async def transcribe(self, audio_bytes: bytes) -> str:
            raise RuntimeError("stt failed")

    service = AudioService(stt_provider=BrokenSTT("ignored"), tts_provider=MockTTS())
    assistant = FakeAssistant()
    with pytest.raises(RuntimeError):
        async for _ in service.handle_audio_turn(b"fake-bytes", assistant):
            pass


@pytest.mark.asyncio
async def test_audio_service_empty_transcript_guard():
    service = AudioService(stt_provider=MockSTT("   "), tts_provider=MockTTS())
    assistant = FakeAssistant()
    with pytest.raises(ValueError):
        async for _ in service.handle_audio_turn(b"fake-bytes", assistant):
            pass


@pytest.mark.asyncio
async def test_audio_service_discards_without_wake_word(monkeypatch):
    monkeypatch.setenv("ENABLE_WAKE_WORD", "1")
    monkeypatch.setenv("WAKE_WORD", "friday")
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore
    service = AudioService(stt_provider=MockSTT("hello there"), tts_provider=MockTTS())
    assistant = FakeAssistant()
    chunks = []
    async for chunk in service.handle_audio_turn(b"fake-bytes", assistant):
        chunks.append(chunk)
    assert chunks == []
