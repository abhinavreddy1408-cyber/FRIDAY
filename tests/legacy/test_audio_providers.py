import pytest
from app.audio.providers import OpenAITTS, OpenAIWhisperSTT


@pytest.mark.asyncio
async def test_openai_whisper_stt_mocked_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import app.audio.providers as providers
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore

    class FakeTranscriptions:
        async def create(self, model, file):
            class Resp:
                text = "transcribed text"

            return Resp()

    class FakeAudio:
        def __init__(self):
            self.transcriptions = FakeTranscriptions()

    class FakeClient:
        def __init__(self, api_key=None):
            self.audio = FakeAudio()

    monkeypatch.setattr(providers, "AsyncOpenAI", FakeClient)
    stt = OpenAIWhisperSTT()
    text = await stt.transcribe(b"fake-audio")
    assert text == "transcribed text"


@pytest.mark.asyncio
async def test_openai_tts_mocked_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUDIO_TTS_VOICE", "alloy")
    import app.audio.providers as providers
    import app.config

    app.config.get_settings.cache_clear()  # type: ignore

    class FakeSpeechResp:
        content = b"audio-bytes"

    class FakeSpeech:
        async def create(self, model, voice, input):
            return FakeSpeechResp()

    class FakeAudio:
        def __init__(self):
            self.speech = FakeSpeech()

    class FakeClient:
        def __init__(self, api_key=None):
            self.audio = FakeAudio()

    monkeypatch.setattr(providers, "AsyncOpenAI", FakeClient)
    tts = OpenAITTS()

    async def text_chunks():
        yield "hello"

    out = []
    async for chunk in tts.synthesize_stream(text_chunks()):
        out.append(chunk)
    assert out == [b"audio-bytes"]
