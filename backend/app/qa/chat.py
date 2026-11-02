"""Chat models behind one small interface.

- GeminiChat calls the Google AI Studio REST API.
- OpenRouterChat calls OpenRouter. It is only the fallback.
- FallbackChat tries Gemini first and switches to OpenRouter when Gemini is
  busy or down (429, 5xx, no network). It does NOT switch on a 400, because a
  wrong request would fail on the other service too, and hiding it would make
  the bug harder to find.
- FakeChat returns a fixed reply. Tests use it, so no test needs a key.

Every model gets a system text and a user text and must return JSON text.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx

from app.http_retry import RETRY_STATUS, NetworkFailure, post_with_retries

GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ChatError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        # True means "the service was busy or down", so another service may work
        self.retryable = retryable


@dataclass(frozen=True)
class ChatReply:
    text: str
    model: str  # which model really answered
    fell_back: bool = False


class ChatModel(Protocol):
    name: str

    def generate_json(self, system: str, user: str) -> ChatReply: ...


class GeminiChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_tries: int = 5,
    ):
        if not api_key:
            raise ChatError("GEMINI_API_KEY is empty. Put it in your .env file.")
        if not model:
            raise ChatError("GEMINI_CHAT_MODEL is empty. Run `python -m app.tools.list_models`.")
        self.api_key = api_key
        self.model = model.removeprefix("models/")
        self.name = f"gemini:{self.model}"
        self.temperature = temperature
        self.client = client or httpx.Client(timeout=90)
        self.sleep = sleep
        self.max_tries = max_tries

    def generate_json(self, system: str, user: str) -> ChatReply:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.temperature,
            },
        }
        headers = {"x-goog-api-key": self.api_key}  # header, never the URL
        try:
            response = post_with_retries(
                self.client,
                f"{GEMINI_ROOT}/models/{self.model}:generateContent",
                body,
                headers,
                sleep=self.sleep,
                max_tries=self.max_tries,
            )
        except NetworkFailure as error:
            raise ChatError(f"Gemini: {error}", retryable=True) from error
        if response.status_code != 200:
            raise ChatError(
                f"Gemini returned {response.status_code}: {response.text[:300]}",
                retryable=response.status_code in RETRY_STATUS,
            )
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ChatError(f"Gemini gave no answer ({reason})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise ChatError(f"Gemini gave an empty answer (finish reason: {candidates[0].get('finishReason')})")
        return ChatReply(text=text, model=self.name)


class OpenRouterChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_tries: int = 3,
    ):
        if not api_key:
            raise ChatError("OPENROUTER_API_KEY is empty.")
        if not model:
            raise ChatError("OPENROUTER_MODEL is empty. Run `python -m app.tools.list_openrouter_models`.")
        self.api_key = api_key
        self.model = model
        self.name = f"openrouter:{self.model}"
        self.temperature = temperature
        self.client = client or httpx.Client(timeout=90)
        self.sleep = sleep
        self.max_tries = max_tries

    def generate_json(self, system: str, user: str) -> ChatReply:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = post_with_retries(
                self.client, OPENROUTER_URL, body, headers, sleep=self.sleep, max_tries=self.max_tries
            )
        except NetworkFailure as error:
            raise ChatError(f"OpenRouter: {error}", retryable=True) from error
        if response.status_code != 200:
            raise ChatError(
                f"OpenRouter returned {response.status_code}: {response.text[:300]}",
                retryable=response.status_code in RETRY_STATUS,
            )
        data = response.json()
        # OpenRouter can answer 200 and still carry an error in the body
        if data.get("error"):
            error = data["error"]
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ChatError(f"OpenRouter error: {str(message)[:300]}", retryable=code in RETRY_STATUS)
        choices = data.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        if not text.strip():
            raise ChatError("OpenRouter gave an empty answer")
        return ChatReply(text=text, model=self.name)


class FallbackChat:
    def __init__(self, primary: ChatModel, secondary: ChatModel | None):
        self.primary = primary
        self.secondary = secondary
        self.name = primary.name

    def generate_json(self, system: str, user: str) -> ChatReply:
        try:
            return self.primary.generate_json(system, user)
        except ChatError as first:
            if not first.retryable or self.secondary is None:
                raise
            try:
                reply = self.secondary.generate_json(system, user)
            except ChatError as second:
                raise ChatError(
                    f"{first} | fallback also failed: {second}", retryable=second.retryable
                ) from second
            return ChatReply(text=reply.text, model=reply.model, fell_back=True)


class FakeChat:
    """Returns a fixed reply and remembers what it was asked."""

    def __init__(self, reply: str | Callable[[str, str], str], name: str = "fake-chat"):
        self.reply = reply
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, system: str, user: str) -> ChatReply:
        self.calls.append((system, user))
        text = self.reply(system, user) if callable(self.reply) else self.reply
        return ChatReply(text=text, model=self.name)
