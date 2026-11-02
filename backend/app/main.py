"""Start the real server. Run from the backend folder:

    uvicorn app.main:app --port 8000
"""
from __future__ import annotations

from app.api import create_app
from app.config import load_settings
from app.ingest.embedder import EmbeddingError
from app.ingest.store import StoreMismatch
from app.qa.chat import ChatError
from app.qa.services import build_services


def build_app():
    try:
        embedder, store, chat = build_services(load_settings())
    except (EmbeddingError, ChatError, StoreMismatch) as error:
        # a short message is friendlier than a long traceback
        raise SystemExit(f"Cannot start: {error}")
    return create_app(embedder=embedder, store=store, chat=chat)


app = build_app()
