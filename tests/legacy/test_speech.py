import pytest
from app.config import get_settings
from app.speech.stt import SpeechNotEnabled, SpeechToText
from app.speech.tts import TextToSpeech


def test_speech_stubs_skip_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_STT", "0")
    monkeypatch.setenv("ENABLE_TTS", "0")
    get_settings.cache_clear()  # type: ignore
    with pytest.raises(SpeechNotEnabled):
        SpeechToText()
    with pytest.raises(SpeechNotEnabled):
        TextToSpeech()
