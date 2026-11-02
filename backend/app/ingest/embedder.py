"""Turn text into vectors.

Two embedders share one small interface:
- GeminiEmbedder calls the Google AI Studio REST API.
- FakeEmbedder makes fake vectors from the words in the text. It is used in
  tests, so no test needs a key or a network.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Callable, Protocol

import httpx

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_BATCH = 100  # the batch endpoint accepts at most 100 texts per call


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str  # stored with the index, so we notice if the model changes

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FakeEmbedder:
    """Bag-of-words vectors. Texts that share words get similar vectors."""

    name = "fake-hash-64"
    dim = 64

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[digest[0] % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str,
        batch_size: int = 50,
        pause_seconds: float = 0.5,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_tries: int = 5,
    ):
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is empty. Put it in your .env file.")
        if not model:
            raise EmbeddingError(
                "GEMINI_EMBED_MODEL is empty. Run `python -m app.tools.list_models` "
                "and put a model that supports embedContent in your .env file."
            )
        self.api_key = api_key
        self.model = model.removeprefix("models/")
        self.name = f"gemini:{self.model}"
        self.batch_size = min(batch_size, MAX_BATCH)
        self.pause_seconds = pause_seconds
        self.client = client or httpx.Client(timeout=60)
        self.sleep = sleep
        self.max_tries = max_tries

    def _post(self, url: str, body: dict) -> dict:
        """POST with retries for rate limits and server errors."""
        headers = {"x-goog-api-key": self.api_key}  # the key goes in a header, not the URL
        for attempt in range(1, self.max_tries + 1):
            try:
                response = self.client.post(url, json=body, headers=headers)
            except httpx.TransportError as error:
                if attempt == self.max_tries:
                    raise EmbeddingError(f"network error: {type(error).__name__}") from error
                self.sleep(2 ** attempt)
                continue
            if response.status_code in RETRY_STATUS and attempt < self.max_tries:
                self.sleep(2 ** attempt)
                continue
            if response.status_code != 200:
                raise EmbeddingError(f"Gemini returned {response.status_code}: {response.text[:300]}")
            return response.json()
        raise EmbeddingError("too many retries")  # not reached, kept for safety

    def _request(self, text: str, task_type: str) -> dict:
        return {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            body = {"requests": [self._request(t, "RETRIEVAL_DOCUMENT") for t in batch]}
            data = self._post(f"{API_ROOT}/models/{self.model}:batchEmbedContents", body)
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(batch):
                raise EmbeddingError(f"asked for {len(batch)} vectors, got {len(embeddings)}")
            vectors.extend(item["values"] for item in embeddings)
            if start + self.batch_size < len(texts):
                self.sleep(self.pause_seconds)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        data = self._post(
            f"{API_ROOT}/models/{self.model}:embedContent",
            self._request(text, "RETRIEVAL_QUERY"),
        )
        return data["embedding"]["values"]
