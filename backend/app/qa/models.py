"""The shapes of data that go in and out of the question answering part.

Pydantic checks these. If the model sends back something that does not fit
the shape, we find out right away instead of showing a broken answer.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    repo: str | None = None  # search only this repo


class Reference(BaseModel):
    """Where an answer comes from. Built from chunk metadata, never from the model's words."""

    chunk_id: str
    repo: str
    path: str
    name: str
    kind: str
    start_line: int | None = None
    end_line: int | None = None
    cell: int | None = None  # notebook cell number
    text: str  # the chunk itself, so the UI can show it when you click


class AskResponse(BaseModel):
    answer: str
    references: list[Reference]
    chunks_used: list[str]  # ids of every chunk the model was shown
    model_used: str
    fell_back: bool = False  # true if the fallback model answered


class ModelAnswer(BaseModel):
    """What we ask the model to send back."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1)
    used: list[int] = Field(default_factory=list)  # excerpt numbers, starting at 1
