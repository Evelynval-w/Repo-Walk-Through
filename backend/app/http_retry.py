"""Send a POST request and try again when trying again can help.

Both the embedding client and the chat clients use this, so the retry rule
lives in one place.

- 429 (too many requests) and 5xx (server problem) are temporary. We wait
  a bit longer each time and ask again.
- Anything else, such as 400 (our request is wrong), will fail the same way
  every time. We hand it back at once and let the caller report it.
"""
from __future__ import annotations

import time
from typing import Callable

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


class NetworkFailure(RuntimeError):
    """The request never got an answer (timeout, connection reset, ...)."""


def post_with_retries(
    client: httpx.Client,
    url: str,
    body: dict,
    headers: dict,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_tries: int = 5,
) -> httpx.Response:
    """Return the last response. The caller checks the status code.

    If the last response is still 429 or 5xx, it is returned too, so the
    caller can tell "the service was busy" from "our request was wrong".
    """
    for attempt in range(1, max_tries + 1):
        try:
            response = client.post(url, json=body, headers=headers)
        except httpx.TransportError as error:
            if attempt == max_tries:
                # only the error type, never the message: it can contain the URL
                raise NetworkFailure(f"network error: {type(error).__name__}") from error
            sleep(2**attempt)
            continue
        if response.status_code in RETRY_STATUS and attempt < max_tries:
            sleep(2**attempt)
            continue
        return response
    raise NetworkFailure("too many retries")  # not reached, kept for safety
