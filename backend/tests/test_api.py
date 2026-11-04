import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.ingest.embedder import EmbeddingError, FakeEmbedder
from app.ingest.models import Chunk
from app.ingest.store import ChunkStore
from app.qa.chat import ChatError, FakeChat

QUESTION = "which service books a flight and checks seats"


class BrokenEmbedder(FakeEmbedder):
    def embed_query(self, text):
        raise EmbeddingError("Gemini returned 503: down")


class BrokenChat:
    name = "broken"

    def generate_json(self, system, user):
        raise ChatError("Gemini returned 429: slow down", retryable=True)


@pytest.fixture
def store(tmp_path):
    fake = FakeEmbedder()
    store = ChunkStore(tmp_path / "db", fake.name)
    chunks = [
        Chunk("airline", "BookingService.java", "BookingService.book", "method", "java",
              "book a flight, check available seats, create the booking", 10, 30),
        Chunk("airline", "MilesRewardService.java", "MilesRewardService.reward", "method", "java",
              "after three flights in a year create a discount code", 5, 20),
    ]
    store.upsert(chunks, fake.embed_documents([c.embed_text() for c in chunks]))
    return store


def client(store, chat=None, embedder=None):
    chat = chat or FakeChat(json.dumps({"answer": "BookingService.book does it.", "used": [1]}))
    return TestClient(create_app(embedder=embedder or FakeEmbedder(), store=store, chat=chat))


def test_health_says_what_the_server_runs_with(store):
    body = client(store).get("/health").json()
    assert body == {"status": "ok", "chunks": 2, "embedder": "fake-hash-64", "chat": "fake-chat"}


def test_ask_returns_an_answer_with_references(store):
    response = client(store).post("/ask", json={"question": QUESTION, "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "BookingService.book does it."
    assert body["references"][0]["path"] == "BookingService.java"
    assert body["references"][0]["start_line"] == 10
    assert body["references"][0]["owner"] == "me"
    assert len(body["chunks_used"]) == 2
    assert body["model_used"] == "fake-chat" and body["fell_back"] is False


def test_top_k_has_a_default(store):
    assert client(store).post("/ask", json={"question": QUESTION}).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},
        {"question": QUESTION, "top_k": 0},
        {"question": QUESTION, "top_k": 99},
        {"top_k": 3},
        {},
    ],
)
def test_bad_requests_are_refused_with_422(store, payload):
    assert client(store).post("/ask", json=payload).status_code == 422


def test_a_failing_chat_service_gives_502(store):
    response = client(store, chat=BrokenChat()).post("/ask", json={"question": QUESTION})
    assert response.status_code == 502
    assert "429" in response.json()["detail"]


def test_a_failing_embedding_service_gives_502(store):
    response = client(store, embedder=BrokenEmbedder()).post("/ask", json={"question": QUESTION})
    assert response.status_code == 502


def test_a_broken_model_answer_gives_502_not_a_fake_answer(store):
    response = client(store, chat=FakeChat("not json at all")).post("/ask", json={"question": QUESTION})
    assert response.status_code == 502
    assert "Bad answer" in response.json()["detail"]


def test_an_empty_index_gives_503(tmp_path):
    empty = ChunkStore(tmp_path / "empty", FakeEmbedder().name)
    response = client(empty).post("/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert "index is empty" in response.json()["detail"]
