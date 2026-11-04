"""Ask one question from the terminal, without starting the server.

    python -m app.qa.ask "Where is the discount code created?"
    python -m app.qa.ask "How is the model evaluated?" --top-k 8 --repo credit_risk_ML_model
    python -m app.qa.ask "..." --show-chunks    # also print the chunks the model saw
"""
from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from app.config import load_settings
from app.ingest.embedder import EmbeddingError
from app.ingest.store import StoreMismatch
from app.qa.answer import BadModelAnswer, IndexEmpty, answer_question
from app.qa.chat import ChatError
from app.qa.models import AskRequest
from app.qa.services import build_services


def where(ref) -> str:
    if ref.start_line is not None:
        return f"lines {ref.start_line}-{ref.end_line}"
    if ref.cell is not None:
        return f"cell {ref.cell}"
    return "whole file"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question about the indexed repos.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None, help="how many chunks to retrieve")
    parser.add_argument("--repo", default=None, help="search only this repo")
    parser.add_argument("--show-chunks", action="store_true", help="print the retrieved chunk ids")
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        request = AskRequest(
            question=args.question, top_k=args.top_k or settings.top_k, repo=args.repo
        )
    except ValidationError as error:
        print(f"Bad input: {error.errors()[0]['msg']}")
        return 1

    try:
        embedder, store, chat = build_services(settings)
        response = answer_question(request, embedder=embedder, store=store, chat=chat)
    except (EmbeddingError, ChatError, StoreMismatch, IndexEmpty, BadModelAnswer) as error:
        print(f"Stopped: {error}")
        return 1

    print(response.answer)
    print()
    if response.references:
        print("References:")
        for ref in response.references:
            who = "" if ref.owner == "me" else f" | written by: {ref.owner}"
            print(f"  - {ref.repo}/{ref.path} | {ref.kind} {ref.name} | {where(ref)}{who}")
    else:
        print("References: none (the model did not point at any excerpt)")
    print(f"\nanswered by {response.model_used}" + (" (fallback)" if response.fell_back else ""))
    if args.show_chunks:
        print("\nChunks the model saw, closest first:")
        for chunk_id in response.chunks_used:
            print(f"  {chunk_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
