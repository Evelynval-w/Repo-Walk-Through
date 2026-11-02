"""Find the chunks that are closest to a question."""
from __future__ import annotations

from dataclasses import dataclass

from app.ingest.embedder import Embedder
from app.ingest.store import ChunkStore


@dataclass(frozen=True)
class Retrieved:
    id: str
    text: str
    metadata: dict
    distance: float  # cosine distance: 0 is identical, bigger is further away


def retrieve(
    question: str,
    embedder: Embedder,
    store: ChunkStore,
    k: int,
    repo: str | None = None,
) -> list[Retrieved]:
    vector = embedder.embed_query(question)
    hits = store.query(vector, k=k, repo=repo)
    return [Retrieved(h["id"], h["text"], h["metadata"], h["distance"]) for h in hits]
