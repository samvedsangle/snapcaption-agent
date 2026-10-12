from types import SimpleNamespace

import pytest

from snapcaption.config import Settings
from snapcaption.llm import AnthropicLLM, GeminiLLM, LLMError, get_llm, parse_json


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("snapcaption.llm.time.sleep", lambda s: None)


# --------------------------------------------------------------- parse_json
def test_parse_plain_and_fenced_json():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_with_chatter():
    assert parse_json('Sure! Here you go:\n{"a": {"b": 2}}\nHope that helps.') == {"a": {"b": 2}}


@pytest.mark.parametrize("bad", ["", "   ", None, "no json here", "[1, 2]", '{"a": ', '{"a": }'])
def test_parse_json_rejects_garbage(bad):
    with pytest.raises(LLMError):
        parse_json(bad)


# ------------------------------------------------------------------- factory
def test_get_llm_needs_a_key():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        get_llm(Settings(provider="gemini"))
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_llm(Settings(provider="anthropic"))
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        get_llm(Settings(provider="mystery"))


def test_get_llm_builds_the_right_class():
    assert isinstance(get_llm(Settings(provider="gemini", gemini_api_key="k")), GeminiLLM)
    assert isinstance(get_llm(Settings(provider="anthropic", anthropic_api_key="k")), AnthropicLLM)


# -------------------------------------------------------------------- Gemini
class FakeGeminiModels:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


def gemini(replies):
    models = FakeGeminiModels(replies)
    return GeminiLLM("key", "model-x", client=SimpleNamespace(models=models)), models


def test_gemini_text_request_shape():
    llm, models = gemini(['{"ok": true}'])
    assert llm.text_json("hello", temperature=0.3) == {"ok": True}
    call = models.calls[0]
    assert call["model"] == "model-x"
    assert call["contents"] == ["hello"]
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].temperature == 0.3


def test_gemini_vision_sends_an_image_part():
    llm, models = gemini(['{"scene": "x"}'])
    assert llm.vision_json(b"\xff\xd8jpegbytes", "image/jpeg", "describe") == {"scene": "x"}
    part, prompt = models.calls[0]["contents"]
    assert prompt == "describe"
    assert part.inline_data.mime_type == "image/jpeg"
    assert part.inline_data.data == b"\xff\xd8jpegbytes"


def test_gemini_retries_after_an_api_error_and_bad_json():
    llm, models = gemini([RuntimeError("503"), "not json", '{"ok": 1}'])
    with pytest.raises(LLMError):  # 2 attempts: error, then bad JSON
        llm.text_json("p")
    llm2, models2 = gemini([RuntimeError("503"), '{"ok": 1}'])
    assert llm2.text_json("p") == {"ok": 1}
    assert len(models2.calls) == 2


def test_gemini_gives_up_with_a_clear_error():
    llm, _ = gemini([RuntimeError("quota"), RuntimeError("quota")])
    with pytest.raises(LLMError, match="quota"):
        llm.text_json("p")


# ----------------------------------------------------------------- Anthropic
class FakeAnthropicMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def anthropic_llm(replies):
    messages = FakeAnthropicMessages(replies)
    return AnthropicLLM("key", "claude-x", client=SimpleNamespace(messages=messages)), messages


def test_anthropic_text_request_shape():
    llm, messages = anthropic_llm(['```json\n{"ok": true}\n```'])
    assert llm.text_json("hello") == {"ok": True}
    call = messages.calls[0]
    assert call["model"] == "claude-x" and call["max_tokens"] > 0
    assert call["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_vision_sends_a_base64_image_block():
    llm, messages = anthropic_llm(['{"scene": "x"}'])
    llm.vision_json(b"abc", "image/jpeg", "describe")
    image, text = messages.calls[0]["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    assert image["source"]["data"] == "YWJj"  # base64 of b"abc"
    assert text == {"type": "text", "text": "describe"}
