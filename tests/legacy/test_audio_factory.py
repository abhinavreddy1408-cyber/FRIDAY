from app.audio.factory import get_audio_service
from app.audio.providers import MockSTT, MockTTS


def test_audio_factory_mock_defaults(monkeypatch):
    monkeypatch.setenv("ENABLE_REAL_AUDIO", "0")
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore
    service = get_audio_service()
    assert isinstance(service.stt_provider, MockSTT)
    assert isinstance(service.tts_provider, MockTTS)


def test_audio_factory_real_provider_selection(monkeypatch):
    monkeypatch.setenv("ENABLE_REAL_AUDIO", "1")
    monkeypatch.setenv("AUDIO_STT_PROVIDER", "openai_whisper")
    monkeypatch.setenv("AUDIO_TTS_PROVIDER", "openai_tts")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import app.audio.factory as factory
    import app.config

    class DummySTT:
        async def transcribe(self, audio_bytes: bytes) -> str:
            return "x"

    class DummyTTS:
        async def synthesize_stream(self, text_chunks):
            if False:
                yield b""

    app.config.get_settings.cache_clear()  # type: ignore
    monkeypatch.setattr(factory, "OpenAIWhisperSTT", DummySTT)
    monkeypatch.setattr(factory, "OpenAITTS", lambda voice=None: DummyTTS())
    service = factory.get_audio_service()
    assert isinstance(service.stt_provider, DummySTT)
    assert isinstance(service.tts_provider, DummyTTS)
