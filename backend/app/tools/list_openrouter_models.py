"""List OpenRouter models you can use as the fallback.

    python -m app.tools.list_openrouter_models          # free models only
    python -m app.tools.list_openrouter_models --all    # every model

The list is public, so no key is needed to read it. Pick one id and put it in
OPENROUTER_MODEL in your .env file.

We mark models that say they support JSON output, because the fallback asks
for JSON. A model without that mark may still work, but it is less likely to.
"""
from __future__ import annotations

import argparse
import sys

import httpx

MODELS_URL = "https://openrouter.ai/api/v1/models"
JSON_FLAGS = {"response_format", "structured_outputs"}


def is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    return str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"


def supports_json(model: dict) -> bool:
    return bool(JSON_FLAGS & set(model.get("supported_parameters") or []))


def pick(models: list[dict], include_paid: bool) -> list[dict]:
    chosen = [m for m in models if include_paid or is_free(m)]
    # JSON-capable first, then by id
    return sorted(chosen, key=lambda m: (not supports_json(m), m.get("id", "")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List OpenRouter models.")
    parser.add_argument("--all", action="store_true", help="include paid models")
    args = parser.parse_args(argv)

    try:
        response = httpx.get(MODELS_URL, timeout=30)
    except httpx.HTTPError as error:
        print(f"Could not reach OpenRouter: {type(error).__name__}")
        return 1
    if response.status_code != 200:
        print(f"OpenRouter returned {response.status_code}")
        return 1

    models = pick(response.json().get("data", []), include_paid=args.all)
    print("Models (JSON output marked with *):")
    for model in models:
        mark = "*" if supports_json(model) else " "
        print(f"  {mark} {model.get('id')}  (context {model.get('context_length')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
