"""A second model call that checks an answer against the facts it must contain.

The judge sees the question, the answer text and the list of facts. It does not
see the code. For each fact it says whether the answer states it. The program
(not the model) decides "correct": every fact must be present.
"""
from __future__ import annotations

import json

from app.qa.answer import _FENCE
from app.qa.chat import ChatModel

JUDGE_SYSTEM = """You grade an answer about a code repository.
You get a question, an answer, and a list of facts. For each fact, say whether
the answer clearly states it (the wording can differ, the meaning must match).
If the answer is vague, wrong about it, or leaves it out, the fact is not present.
Reply with JSON only: {"facts": [true, false, ...]} with one true or false per fact, in order."""

REFUSAL_SYSTEM = """You read an answer to a question about a code repository.
Decide whether the answer says that the repository does not contain this, does not do
this, or that it cannot find the information (a refusal). Saying "it does not do X,
it does Y instead" is still a refusal. An answer that describes how the asked thing
works, even loosely, is not a refusal.
Reply with JSON only: {"refused": true} or {"refused": false}."""


class JudgeError(RuntimeError):
    pass


def _load(text: str) -> dict:
    try:
        data = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError as error:
        raise JudgeError(f"judge did not return JSON: {text[:200]!r}") from error
    if not isinstance(data, dict):
        raise JudgeError("judge JSON is not an object")
    return data


def judge_facts(chat: ChatModel, question: str, answer: str, must_say: list[str]) -> list[bool]:
    facts = "\n".join(f"{i}. {fact}" for i, fact in enumerate(must_say, 1))
    user = f"Question: {question}\n\nAnswer:\n{answer}\n\nFacts:\n{facts}"
    flags = _load(chat.generate_json(JUDGE_SYSTEM, user).text).get("facts")
    if not isinstance(flags, list) or len(flags) != len(must_say) or not all(isinstance(f, bool) for f in flags):
        raise JudgeError(f"judge gave a bad list for {len(must_say)} facts: {flags!r}")
    return flags


def judge_refusal(chat: ChatModel, question: str, answer: str) -> bool:
    user = f"Question: {question}\n\nAnswer:\n{answer}"
    refused = _load(chat.generate_json(REFUSAL_SYSTEM, user).text).get("refused")
    if not isinstance(refused, bool):
        raise JudgeError(f"judge gave a bad refusal flag: {refused!r}")
    return refused
