import pytest
from app.config import get_settings
from app.llm.provider import OpenAIProvider


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("ENABLE_STT", "0")
    monkeypatch.setenv("ENABLE_TTS", "0")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    get_settings.cache_clear()  # type: ignore
    settings = get_settings()
    assert settings.enable_stt is False
    assert settings.enable_tts is False
    assert settings.db_url.startswith("sqlite")


def test_openai_provider_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    get_settings.cache_clear()  # type: ignore
    with pytest.raises(Exception):  # noqa: B017
        OpenAIProvider()


def test_llm_provider_validation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()  # type: ignore
    settings = get_settings()
    assert settings.llm_provider == "mock"
    monkeypatch.setenv("LLM_PROVIDER", "badprovider")
    get_settings.cache_clear()  # type: ignore
    with pytest.raises(Exception):  # noqa: B017
        get_settings()
