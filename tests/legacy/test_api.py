import pytest
from app.server.api import build_app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import os
    os.environ["LLM_PROVIDER"] = "mock"
    import app.config
    app.config.get_settings.cache_clear()  # type: ignore
    app = build_app()
    return TestClient(app)


def test_chat_validation(client):
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "conversation_id" in data
    assert "reply" in data


def test_chat_rejects_empty_message(client):
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_streaming(client):
    with client.stream("POST", "/chat/stream", json={"message": "hello"}) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes())
        assert b"event:" in body or b"data:" in body


def test_chat_accepts_multimodal_message_blocks(client):
    payload = {
        "message": [
            {"type": "text", "text": "What is on my screen?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
        ]
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
