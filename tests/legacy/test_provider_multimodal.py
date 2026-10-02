from app.llm.provider import OpenAIProvider
from app.models import ImageContentBlock, ImageUrlPayload, Message, MessageRole, TextContentBlock


def test_openai_payload_formats_base64_image_blocks():
    messages = [
        Message(
            role=MessageRole.user,
            content=[
                TextContentBlock(text="Look at my screen"),
                ImageContentBlock(image_url=ImageUrlPayload(url="ZmFrZS1pbWFnZS1ieXRlcw==")),
            ],
        )
    ]
    payload = OpenAIProvider.to_openai_messages(messages, supports_images=True)
    content = payload[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_payload_strips_images_for_text_only_models():
    messages = [
        Message(
            role=MessageRole.user,
            content=[
                TextContentBlock(text="Please explain this"),
                ImageContentBlock(image_url=ImageUrlPayload(url="ZmFrZS1pbWFnZS1ieXRlcw==")),
            ],
        )
    ]
    payload = OpenAIProvider.to_openai_messages(messages, supports_images=False)
    assert payload[0]["content"] == "Please explain this"
