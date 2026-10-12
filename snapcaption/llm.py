"""The only module that talks to a model SDK. Every call returns parsed JSON (a dict)."""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Callable, Optional

from .config import Settings


class LLMError(RuntimeError):
    """The model could not be reached or did not return usable JSON."""


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def parse_json(text: Optional[str]) -> dict:
    """Parse a JSON object out of model output, tolerating code fences and chatter."""
    if not text or not text.strip():
        raise LLMError("The model returned an empty response.")
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        start = body.find("{")
        if start < 0:
            raise LLMError("The model did not return JSON.") from None
        try:
            value, _ = json.JSONDecoder().raw_decode(body[start:])
        except json.JSONDecodeError as e:
            raise LLMError(f"The model returned broken JSON ({e.msg}).") from None
    if not isinstance(value, dict):
        raise LLMError("The model returned JSON, but not an object.")
    return value


class _BaseLLM:
    name = "model"
    attempts = 2
    backoff = 1.5  # seconds, times the attempt number

    def _json_with_retries(self, call: Callable[[], Optional[str]]) -> dict:
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                return parse_json(call())
            except Exception as e:  # noqa: BLE001 - SDK errors vary by provider
                last = e
                if attempt < self.attempts - 1:
                    time.sleep(self.backoff * (attempt + 1))
        raise LLMError(f"{self.name} request failed: {last}") from last


class GeminiLLM(_BaseLLM):
    name = "Gemini"

    def __init__(self, api_key: str, model: str, client: Any = None):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client, self.model = client, model

    def _call(self, contents: list, temperature: float) -> Optional[str]:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=temperature
            ),
        )
        return resp.text

    def text_json(self, prompt: str, temperature: float = 0.7) -> dict:
        return self._json_with_retries(lambda: self._call([prompt], temperature))

    def vision_json(self, image_bytes: bytes, mime: str, prompt: str, temperature: float = 0.2) -> dict:
        from google.genai import types

        part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
        return self._json_with_retries(lambda: self._call([part, prompt], temperature))


class AnthropicLLM(_BaseLLM):
    """Claude via the Anthropic SDK. Current Claude models don't take `temperature`,
    so the argument is accepted for interface parity and ignored; `effort` controls depth."""

    name = "Claude"
    max_tokens = 16000
    effort = "medium"

    def __init__(self, api_key: str, model: str, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self.client, self.model = client, model

    def _call(self, content) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort},
            # If a safety classifier declines, let the API retry on a fallback model.
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMError("Claude declined this request.")
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

    def text_json(self, prompt: str, temperature: float = 0.7) -> dict:
        return self._json_with_retries(lambda: self._call(prompt))

    def vision_json(self, image_bytes: bytes, mime: str, prompt: str, temperature: float = 0.2) -> dict:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.standard_b64encode(image_bytes).decode("ascii"),
                },
            },
            {"type": "text", "text": prompt},
        ]
        return self._json_with_retries(lambda: self._call(content))


def get_llm(settings: Settings):
    provider = (settings.provider or "").strip().lower()
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                "and put it in .env (or your host's secrets)."
            )
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to .env (or your host's secrets).")
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    raise LLMError(f"Unknown LLM_PROVIDER {settings.provider!r}. Use 'gemini' or 'anthropic'.")
