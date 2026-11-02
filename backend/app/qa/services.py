"""Build the real embedder, store and chat model from the settings.

The CLI and the API both start here, so they behave the same.
"""
from __future__ import annotations

from app.config import Settings
from app.ingest.embedder import GeminiEmbedder
from app.ingest.store import ChunkStore
from app.qa.chat import FallbackChat, GeminiChat, OpenRouterChat


def build_chat(settings: Settings) -> FallbackChat:
    primary = GeminiChat(
        settings.gemini_api_key,
        settings.gemini_chat_model,
        temperature=settings.chat_temperature,
    )
    secondary = None
    # The fallback is optional. It is only built when both the key and the model are set.
    if settings.openrouter_api_key and settings.openrouter_model:
        secondary = OpenRouterChat(
            settings.openrouter_api_key,
            settings.openrouter_model,
            temperature=settings.chat_temperature,
        )
    return FallbackChat(primary, secondary)


def build_services(settings: Settings):
    """Return (embedder, store, chat). Raises EmbeddingError, ChatError or StoreMismatch."""
    embedder = GeminiEmbedder(
        settings.gemini_api_key,
        settings.gemini_embed_model,
        batch_size=settings.embed_batch_size,
        pause_seconds=settings.embed_pause_seconds,
    )
    store = ChunkStore(settings.chroma_dir, embedder.name)
    chat = build_chat(settings)
    return embedder, store, chat
