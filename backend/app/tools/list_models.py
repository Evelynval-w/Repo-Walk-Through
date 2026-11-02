"""List the Gemini models your API key can use.

    python -m app.tools.list_models                    # show models
    python -m app.tools.list_models --try-embed NAME   # embed one sentence with model NAME

The key is read from .env and is never printed.
"""
from __future__ import annotations

import argparse
import sys

import httpx

from app.config import load_settings
from app.ingest.embedder import API_ROOT, EmbeddingError, GeminiEmbedder


def list_models(api_key: str) -> list[dict]:
    models: list[dict] = []
    page_token = ""
    while True:
        params = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        response = httpx.get(f"{API_ROOT}/models", params=params, headers={"x-goog-api-key": api_key}, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini returned {response.status_code}: {response.text[:300]}")
        data = response.json()
        models.extend(data.get("models", []))
        page_token = data.get("nextPageToken", "")
        if not page_token:
            return models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--try-embed", metavar="MODEL", help="embed one sentence with this model")
    args = parser.parse_args()

    settings = load_settings()
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is empty. Put it in the .env file in the project root.")
        return 1

    if args.try_embed:
        try:
            embedder = GeminiEmbedder(settings.gemini_api_key, args.try_embed)
            vector = embedder.embed_query("where is the booking logic?")
        except EmbeddingError as error:
            print(f"Failed: {error}")
            return 1
        print(f"It works. {embedder.name} returned a vector with {len(vector)} numbers.")
        return 0

    try:
        models = list_models(settings.gemini_api_key)
    except RuntimeError as error:
        print(f"Failed: {error}")
        return 1

    for title, method in (("Embedding models (use for GEMINI_EMBED_MODEL)", "embedContent"),
                          ("Chat models (for phase 2)", "generateContent")):
        print(f"\n{title}")
        for model in models:
            if method in model.get("supportedGenerationMethods", []):
                print(f"  {model['name'].removeprefix('models/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
