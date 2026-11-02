"""A small wrapper around Chroma, the vector database.

Chroma keeps the vectors, the chunk text and the metadata on disk.
We compute the vectors ourselves and hand them in, so Chroma never calls a model.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import chromadb

from app.ingest.models import Chunk

COLLECTION = "chunks"
WRITE_BATCH = 500


class StoreMismatch(RuntimeError):
    pass


class ChunkStore:
    def __init__(self, path: Path, embedder_name: str):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.embedder_name = embedder_name
        self.collection = self.client.get_or_create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine", "embedder": embedder_name},
            embedding_function=None,
        )
        built_with = (self.collection.metadata or {}).get("embedder")
        if built_with and built_with != embedder_name:
            # Vectors from two different models cannot be compared. Fail loudly.
            raise StoreMismatch(
                f"This index was built with '{built_with}' but you are using '{embedder_name}'. "
                "Run again with --reset to rebuild it."
            )

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine", "embedder": self.embedder_name},
            embedding_function=None,
        )

    def delete_repo(self, repo: str) -> None:
        """Remove old chunks of a repo before we add the new ones."""
        self.collection.delete(where={"repo": repo})

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for start in range(0, len(chunks), WRITE_BATCH):
            part = slice(start, start + WRITE_BATCH)
            self.collection.upsert(
                ids=[c.id for c in chunks[part]],
                embeddings=vectors[part],
                documents=[c.text for c in chunks[part]],
                metadatas=[c.metadata() for c in chunks[part]],
            )

    def count(self) -> int:
        return self.collection.count()

    def repo_counts(self) -> Counter:
        found = self.collection.get(include=["metadatas"])
        return Counter(m["repo"] for m in found["metadatas"])

    def query(self, vector: list[float], k: int = 5, repo: str | None = None) -> list[dict]:
        """Nearest chunks to a vector. Phase 2 builds on this."""
        found = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            where={"repo": repo} if repo else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": id_, "text": doc, "metadata": meta, "distance": dist}
            for id_, doc, meta, dist in zip(
                found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]
            )
        ]
