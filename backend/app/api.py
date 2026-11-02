"""The web API.

    POST /ask     {"question": "...", "top_k": 6}  ->  answer with references
    GET  /health  what the server is running with

create_app takes the embedder, store and chat model as arguments. That way the
tests can pass fakes, and the real server passes the real ones (see main.py).
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.ingest.embedder import Embedder, EmbeddingError
from app.ingest.store import ChunkStore
from app.qa.answer import BadModelAnswer, IndexEmpty, answer_question
from app.qa.chat import ChatError, ChatModel
from app.qa.models import AskRequest, AskResponse


def create_app(*, embedder: Embedder, store: ChunkStore, chat: ChatModel) -> FastAPI:
    app = FastAPI(title="Repo-Walk-Through")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "chunks": store.count(),
            "embedder": embedder.name,
            "chat": chat.name,
        }

    # A plain `def` (not `async def`) on purpose: the clients are blocking, and
    # FastAPI runs plain functions in a thread pool so they do not freeze the server.
    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            return answer_question(request, embedder=embedder, store=store, chat=chat)
        except IndexEmpty as error:
            raise HTTPException(status_code=503, detail=str(error))
        except (EmbeddingError, ChatError) as error:
            raise HTTPException(status_code=502, detail=f"A model service failed: {error}")
        except BadModelAnswer as error:
            raise HTTPException(status_code=502, detail=f"Bad answer from the model: {error}")

    return app
