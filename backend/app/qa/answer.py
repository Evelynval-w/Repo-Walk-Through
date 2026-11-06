"""Question in, validated answer with references out.

Steps:
1. Embed the question and fetch the top-k chunks.
2. Show the chunks to the chat model, numbered, and ask for JSON.
3. Check the JSON with Pydantic.
4. Build the references from the metadata of the chunks the model pointed at.
   The model only says "I used excerpts 1 and 3". It never writes a file name,
   so it cannot invent one.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.ingest.embedder import Embedder
from app.ingest.store import ChunkStore
from app.qa.chat import ChatModel
from app.qa.models import AskRequest, AskResponse, ModelAnswer, Reference
from app.qa.prompt import SYSTEM, build_user_prompt
from app.qa.retrieve import retrieve


class IndexEmpty(RuntimeError):
    pass


class BadModelAnswer(RuntimeError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_model_answer(text: str, chunk_count: int, finish_reason: str = "") -> ModelAnswer:
    """Turn the model's text into a ModelAnswer, or raise BadModelAnswer."""
    cleaned = _FENCE.sub("", text.strip())  # some models wrap JSON in ```json fences
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        why = f" (finish reason: {finish_reason})" if finish_reason and finish_reason != "STOP" else ""
        raise BadModelAnswer(f"the model did not return JSON{why}: {text[:200]!r}") from error
    try:
        parsed = ModelAnswer.model_validate(data)
    except ValidationError as error:
        raise BadModelAnswer(f"the model's JSON has the wrong shape: {error.errors()[0]['msg']}") from error

    bad = [n for n in parsed.used if n < 1 or n > chunk_count]
    if bad:
        raise BadModelAnswer(f"the model pointed at excerpts that do not exist: {bad}")
    return parsed


def fetch_chunks(request: AskRequest, *, embedder: Embedder, store: ChunkStore) -> list:
    """Step 1 on its own, so the evaluation can see exactly which chunks the model is shown."""
    if store.count() == 0:
        raise IndexEmpty("The index is empty. Run `python -m app.ingest.index` first.")
    chunks = retrieve(request.question, embedder, store, request.top_k, request.repo)
    if not chunks:
        raise IndexEmpty(f"No chunks found for repo '{request.repo}'.")
    return chunks


def answer_from_chunks(request: AskRequest, chunks: list, *, chat: ChatModel) -> AskResponse:
    reply = chat.generate_json(SYSTEM, build_user_prompt(request.question, chunks))
    parsed = parse_model_answer(reply.text, len(chunks), reply.finish_reason)

    references: list[Reference] = []
    for number in dict.fromkeys(parsed.used):  # drop repeats, keep the model's order
        chunk = chunks[number - 1]
        meta = chunk.metadata
        references.append(
            Reference(
                chunk_id=chunk.id,
                repo=meta["repo"],
                path=meta["path"],
                name=meta["name"],
                kind=meta["kind"],
                start_line=meta.get("start_line"),
                end_line=meta.get("end_line"),
                cell=meta.get("cell"),
                owner=meta.get("owner", "me"),
                text=chunk.text,
            )
        )

    return AskResponse(
        answer=parsed.answer,
        references=references,
        chunks_used=[c.id for c in chunks],
        model_used=reply.model,
        fell_back=reply.fell_back,
    )


def answer_question(
    request: AskRequest,
    *,
    embedder: Embedder,
    store: ChunkStore,
    chat: ChatModel,
) -> AskResponse:
    chunks = fetch_chunks(request, embedder=embedder, store=store)
    return answer_from_chunks(request, chunks, chat=chat)
