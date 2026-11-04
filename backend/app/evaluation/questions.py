"""Load the evaluation questions from eval/questions.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Question:
    id: str
    repo: str
    kind: str
    question: str
    expected_files: list[str] = field(default_factory=list)
    must_say: list[str] = field(default_factory=list)
    answerable: bool = True
    note: str = ""


def load_questions(path: Path) -> list[Question]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    questions = [Question(**item) for item in data.get("questions", [])]

    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("two questions share an id")
    for q in questions:
        if q.answerable and not (q.expected_files and q.must_say):
            raise ValueError(f"{q.id}: an answerable question needs expected_files and must_say")
        if not q.answerable and (q.expected_files or q.must_say):
            raise ValueError(f"{q.id}: an unanswerable question must have empty expected_files and must_say")
    return questions
