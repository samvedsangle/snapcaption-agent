from types import SimpleNamespace

import pytest

from snapcaption.config import Settings
from snapcaption.llm import AnthropicLLM, GeminiLLM, LLMError, OllamaLLM, get_llm, parse_json


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
    assert isinstance(get_llm(Settings(provider="ollama")), OllamaLLM)  # local: no key needed


# -------------------------------------------------------------------- Ollama
class FakeOllamaSession:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        reply = self.replies.pop(0)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"message": {"content": reply}})


def test_ollama_request_shape_and_image():
    session = FakeOllamaSession(['{"ok": true}', '{"scene": "x"}'])
    llm = OllamaLLM("http://box:11434/", "gemma3:4b", client=session)
    assert llm.text_json("hello", temperature=0.4) == {"ok": True}
    llm.vision_json(b"abc", "image/jpeg", "describe")
    text_call, vision_call = session.calls
    assert text_call["url"] == "http://box:11434/api/chat"
    body = text_call["json"]
    assert body["model"] == "gemma3:4b" and body["format"] == "json" and body["stream"] is False
    assert body["options"]["temperature"] == 0.4
    assert vision_call["json"]["messages"][0]["images"] == ["YWJj"]


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


def test_gemini_thinking_is_kept_low_per_model_family():
    models = FakeGeminiModels(['{"ok": 1}'] * 3)
    for name in ("gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-pro"):
        GeminiLLM("key", name, client=SimpleNamespace(models=models)).text_json("p")
    assert models.calls[0]["config"].thinking_config.thinking_budget == 0
    assert models.calls[1]["config"].thinking_config.thinking_level.value.lower() == "low"
    assert models.calls[2]["config"].thinking_config is None


def test_gemini_drops_a_rejected_thinking_setting():
    models = FakeGeminiModels([RuntimeError("400 Thinking level LOW is not supported"), '{"ok": 1}'])
    llm = GeminiLLM("key", "gemini-9-flash", client=SimpleNamespace(models=models))
    assert llm.text_json("p") == {"ok": 1}
    assert models.calls[1]["config"].thinking_config is None


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


def test_anthropic_refusal_fallback_only_where_supported():
    messages = FakeAnthropicMessages(['{"ok": 1}', '{"ok": 1}'])
    AnthropicLLM("key", "claude-opus-5-5", client=SimpleNamespace(messages=messages)).text_json("p")
    AnthropicLLM("key", "claude-haiku-5-5", client=SimpleNamespace(messages=messages)).text_json("p")
    assert messages.calls[0]["extra_body"] == {"fallbacks": "default"}
    assert "extra_body" not in messages.calls[1]


def test_anthropic_vision_sends_a_base64_image_block():
    llm, messages = anthropic_llm(['{"scene": "x"}'])
    llm.vision_json(b"abc", "image/jpeg", "describe")
    image, text = messages.calls[0]["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    assert image["source"]["data"] == "YWJj"  # base64 of b"abc"
    assert text == {"type": "text", "text": "describe"}
