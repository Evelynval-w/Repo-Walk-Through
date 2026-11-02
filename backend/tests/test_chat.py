import dataclasses
import json

import httpx
import pytest

from app.qa.chat import ChatError, ChatReply, FallbackChat, GeminiChat, OpenRouterChat
from app.qa.services import build_chat

KEY = "test-key-not-real"


def gemini(handler, sleeps=None, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleeps = [] if sleeps is None else sleeps
    return GeminiChat(KEY, "models/chat-test", client=client, sleep=sleeps.append, **kwargs)


def openrouter(handler, sleeps=None, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleeps = [] if sleeps is None else sleeps
    return OpenRouterChat(KEY, "some/model:free", client=client, sleep=sleeps.append, **kwargs)


def gemini_ok(text):
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


# ---------- Gemini ----------


def test_gemini_request_has_the_key_in_a_header_and_asks_for_json():
    seen = []

    def handler(request):
        seen.append(request)
        return gemini_ok('{"answer": "hi", "used": []}')

    reply = gemini(handler).generate_json("be short", "what is x?")

    request = seen[0]
    assert reply.text == '{"answer": "hi", "used": []}'
    assert reply.model == "gemini:chat-test"
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url)
    assert str(request.url).endswith("/models/chat-test:generateContent")
    body = json.loads(request.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "be short"
    assert body["contents"][0]["parts"][0]["text"] == "what is x?"
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_ignores_thought_parts():
    def handler(request):
        parts = [{"text": "thinking...", "thought": True}, {"text": '{"answer": "ok"}'}]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": parts}}]})

    assert gemini(handler).generate_json("s", "u").text == '{"answer": "ok"}'


def test_gemini_400_is_not_retried_and_is_not_retryable():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad model name")

    with pytest.raises(ChatError) as error:
        gemini(handler, sleeps).generate_json("s", "u")
    assert len(calls) == 1 and sleeps == []
    assert error.value.retryable is False
    assert "400" in str(error.value) and KEY not in str(error.value)


def test_gemini_429_is_retried_then_succeeds():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text="slow down") if len(calls) == 1 else gemini_ok("{}")

    assert gemini(handler, sleeps).generate_json("s", "u").text == "{}"
    assert len(calls) == 2 and len(sleeps) == 1


def test_gemini_503_after_all_tries_is_retryable():
    sleeps = []
    with pytest.raises(ChatError) as error:
        gemini(lambda r: httpx.Response(503, text="down"), sleeps, max_tries=3).generate_json("s", "u")
    assert error.value.retryable is True
    assert len(sleeps) == 2


def test_gemini_network_failure_is_retryable_and_hides_the_url():
    def handler(request):
        raise httpx.ConnectError("could not connect to https://x?key=SECRET")

    with pytest.raises(ChatError) as error:
        gemini(handler, max_tries=2).generate_json("s", "u")
    assert error.value.retryable is True
    assert "SECRET" not in str(error.value)


def test_gemini_blocked_prompt_gives_a_clear_error():
    def handler(request):
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(ChatError) as error:
        gemini(handler).generate_json("s", "u")
    assert "SAFETY" in str(error.value)
    assert error.value.retryable is False


def test_gemini_needs_a_key_and_a_model():
    with pytest.raises(ChatError):
        GeminiChat("", "m")
    with pytest.raises(ChatError):
        GeminiChat(KEY, "")


# ---------- OpenRouter ----------


def test_openrouter_request_and_reply():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "yes"}'}}]})

    reply = openrouter(handler).generate_json("sys", "usr")

    request = seen[0]
    assert reply.text == '{"answer": "yes"}'
    assert reply.model == "openrouter:some/model:free"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["model"] == "some/model:free"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "usr"}
    assert body["response_format"] == {"type": "json_object"}


def test_openrouter_error_inside_a_200_reply_is_understood():
    def handler(request):
        return httpx.Response(200, json={"error": {"code": 429, "message": "rate limited upstream"}})

    with pytest.raises(ChatError) as error:
        openrouter(handler).generate_json("s", "u")
    assert error.value.retryable is True
    assert "rate limited" in str(error.value)


def test_openrouter_401_is_not_retryable():
    with pytest.raises(ChatError) as error:
        openrouter(lambda r: httpx.Response(401, text="no auth")).generate_json("s", "u")
    assert error.value.retryable is False


# ---------- fallback ----------


class Scripted:
    def __init__(self, name, outcome):
        self.name = name
        self.outcome = outcome
        self.calls = 0

    def generate_json(self, system, user):
        self.calls += 1
        if isinstance(self.outcome, ChatError):
            raise self.outcome
        return ChatReply(text=self.outcome, model=self.name)


def test_fallback_not_used_when_the_first_model_works():
    first, second = Scripted("a", "{}"), Scripted("b", "{}")
    reply = FallbackChat(first, second).generate_json("s", "u")
    assert reply.model == "a" and reply.fell_back is False
    assert second.calls == 0


def test_fallback_is_used_when_the_first_model_is_busy():
    first = Scripted("a", ChatError("busy", retryable=True))
    second = Scripted("b", '{"answer": "from b"}')
    reply = FallbackChat(first, second).generate_json("s", "u")
    assert reply.model == "b" and reply.fell_back is True
    assert reply.text == '{"answer": "from b"}'


def test_fallback_is_not_used_for_a_wrong_request():
    first = Scripted("a", ChatError("bad request", retryable=False))
    second = Scripted("b", "{}")
    with pytest.raises(ChatError, match="bad request"):
        FallbackChat(first, second).generate_json("s", "u")
    assert second.calls == 0


def test_no_fallback_configured_means_the_original_error_comes_through():
    first = Scripted("a", ChatError("busy", retryable=True))
    with pytest.raises(ChatError, match="busy"):
        FallbackChat(first, None).generate_json("s", "u")


def test_both_failing_reports_both_reasons():
    first = Scripted("a", ChatError("gemini busy", retryable=True))
    second = Scripted("b", ChatError("openrouter down", retryable=True))
    with pytest.raises(ChatError) as error:
        FallbackChat(first, second).generate_json("s", "u")
    assert "gemini busy" in str(error.value) and "openrouter down" in str(error.value)


# ---------- building from settings ----------


def test_build_chat_adds_the_fallback_only_when_key_and_model_are_set(settings):
    with_gemini = dataclasses.replace(settings, gemini_api_key=KEY)
    assert build_chat(with_gemini).secondary is None

    only_key = dataclasses.replace(with_gemini, openrouter_api_key=KEY)
    assert build_chat(only_key).secondary is None

    both = dataclasses.replace(only_key, openrouter_model="some/model:free")
    assert isinstance(build_chat(both).secondary, OpenRouterChat)


def test_build_chat_without_a_gemini_key_explains_what_to_do(settings):
    with pytest.raises(ChatError, match="GEMINI_API_KEY"):
        build_chat(settings)
