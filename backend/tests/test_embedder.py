import json
import math

import httpx
import pytest

from app.ingest.embedder import EmbeddingError, FakeEmbedder, GeminiEmbedder

KEY = "test-key-not-real"


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_fake_embedder_is_deterministic_and_puts_similar_text_close():
    fake = FakeEmbedder()
    assert fake.embed_query("flight booking") == fake.embed_query("flight booking")
    close = cosine(fake.embed_query("flight booking seats"), fake.embed_query("book a flight"))
    far = cosine(fake.embed_query("flight booking seats"), fake.embed_query("pandas dataframe loan default"))
    assert close > far


def make_embedder(handler, sleeps, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiEmbedder(KEY, "models/embed-test", client=client, sleep=sleeps.append, **kwargs)


def ok_batch(request):
    body = json.loads(request.content)
    return httpx.Response(200, json={"embeddings": [{"values": [float(len(r["content"]["parts"][0]["text"]))]} for r in body["requests"]]})


def test_documents_are_sent_in_batches_in_order():
    calls, sleeps = [], []

    def handler(request):
        calls.append(request)
        return ok_batch(request)

    embedder = make_embedder(handler, sleeps, batch_size=2, pause_seconds=0.5)
    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert len(calls) == 3  # 2 + 2 + 1
    assert sleeps == [0.5, 0.5]  # a pause between batches, not after the last one
    first = calls[0]
    assert first.headers["x-goog-api-key"] == KEY  # key goes in a header
    assert KEY not in str(first.url)  # and never in the URL
    assert str(first.url).endswith("/models/embed-test:batchEmbedContents")
    assert json.loads(first.content)["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


def test_rate_limit_is_retried_with_a_pause():
    sleeps, attempts = [], []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, text="slow down")
        return ok_batch(request)

    embedder = make_embedder(handler, sleeps)
    assert embedder.embed_documents(["abc"]) == [[3.0]]
    assert len(attempts) == 2
    assert len(sleeps) == 1


def test_a_client_error_stops_with_a_clear_message_and_no_key():
    embedder = make_embedder(lambda request: httpx.Response(400, text="bad model"), [])
    with pytest.raises(EmbeddingError) as error:
        embedder.embed_documents(["abc"])
    assert "400" in str(error.value)
    assert KEY not in str(error.value)


def test_gives_up_after_the_last_try():
    sleeps = []
    embedder = make_embedder(lambda request: httpx.Response(503, text="down"), sleeps, max_tries=3)
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["abc"])
    assert len(sleeps) == 2


def test_query_uses_the_query_task_type():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"embedding": {"values": [0.1, 0.2]}})

    embedder = make_embedder(handler, [])
    assert embedder.embed_query("where is booking?") == [0.1, 0.2]
    assert seen[0]["taskType"] == "RETRIEVAL_QUERY"


def test_missing_key_or_model_gives_a_helpful_error():
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        GeminiEmbedder("", "some-model")
    with pytest.raises(EmbeddingError, match="list_models"):
        GeminiEmbedder("key", "")
