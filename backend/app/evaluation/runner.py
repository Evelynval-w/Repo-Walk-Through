"""Ask every question and measure three separate things.

1. retrieval hit: an expected file is among the chunks shown to the model
2. citation hit:  an expected file is among the references the model chose
3. answer correct: the judge finds every must_say fact in the answer

Keeping them apart shows where a failure happens. A retrieval miss is a search
problem (phase 4). A retrieval hit with a wrong answer is a model or prompt problem.

Unanswerable questions are not scored yet. Their answers are saved, plus whether
the answer refused. Phase 5 adds the real refusal and compares against this baseline.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable

from app.evaluation.judge import JudgeError, judge_facts, judge_refusal
from app.evaluation.questions import Question
from app.ingest.embedder import Embedder, EmbeddingError
from app.ingest.store import ChunkStore
from app.qa.answer import BadModelAnswer, answer_from_chunks, fetch_chunks
from app.qa.chat import ChatError, ChatModel
from app.qa.models import AskRequest


@dataclass
class RunResult:
    question_id: str
    run: int
    error: str = ""
    answer: str = ""
    retrieved_files: list[str] = field(default_factory=list)
    cited_files: list[str] = field(default_factory=list)
    retrieval_hit: bool | None = None
    evidence_found: list[bool] = field(default_factory=list)
    evidence_hit: bool | None = None
    citation_hit: bool | None = None
    facts_found: list[bool] = field(default_factory=list)
    correct: bool | None = None
    refused: bool | None = None
    judge_error: str = ""
    model_used: str = ""


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def evidence_found(phrases: list[str], chunk_texts: list[str]) -> list[bool]:
    """For each phrase: does it appear in the text of any chunk the model was shown?"""
    shown = [_norm(t) for t in chunk_texts]
    return [any(_norm(p) in t for t in shown) for p in phrases]


def file_of(chunk_id: str) -> str:
    """'repo/path#kind:name@...' -> 'repo/path'"""
    return chunk_id.split("#", 1)[0]


def run_one(
    q: Question,
    run: int,
    *,
    embedder: Embedder,
    store: ChunkStore,
    chat: ChatModel,
    judge_chat: ChatModel | None,
    top_k: int,
) -> RunResult:
    result = RunResult(q.id, run)
    request = AskRequest(question=q.question, top_k=top_k)
    chunks = []
    try:
        chunks = fetch_chunks(request, embedder=embedder, store=store)
        response = answer_from_chunks(request, chunks, chat=chat)
    except (ChatError, BadModelAnswer, EmbeddingError) as error:
        result.error = f"{type(error).__name__}: {error}"
        if q.answerable:  # a failed answer is a miss, not a skipped question
            # Search may have worked even though the answer step failed. Record that honestly.
            result.retrieved_files = sorted({file_of(c.id) for c in chunks})
            result.retrieval_hit = bool(set(q.expected_files) & set(result.retrieved_files))
            result.citation_hit = result.correct = False
            if q.evidence:
                result.evidence_found = evidence_found(q.evidence, [c.text for c in chunks])
                result.evidence_hit = all(result.evidence_found)
        return result

    result.answer = response.answer
    result.model_used = response.model_used
    result.retrieved_files = sorted({file_of(c) for c in response.chunks_used})
    result.cited_files = sorted({f"{r.repo}/{r.path}" for r in response.references})

    if q.answerable:
        wanted = set(q.expected_files)
        result.retrieval_hit = bool(wanted & set(result.retrieved_files))
        result.citation_hit = bool(wanted & set(result.cited_files))
        if q.evidence:
            result.evidence_found = evidence_found(q.evidence, [c.text for c in chunks])
            result.evidence_hit = all(result.evidence_found)

    if judge_chat is None:
        return result
    try:
        if q.answerable:
            result.facts_found = judge_facts(judge_chat, q.question, response.answer, q.must_say)
            result.correct = all(result.facts_found)
        else:
            result.refused = judge_refusal(judge_chat, q.question, response.answer)
    except (JudgeError, ChatError) as error:
        result.judge_error = str(error)
        if q.answerable:
            result.correct = False
    return result


def run_all(
    questions: list[Question],
    *,
    embedder: Embedder,
    store: ChunkStore,
    chat: ChatModel,
    judge_chat: ChatModel | None,
    top_k: int = 6,
    runs: int = 1,
    pause: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] = lambda _msg: None,
) -> list[RunResult]:
    results = []
    for q in questions:
        for run in range(1, runs + 1):
            result = run_one(q, run, embedder=embedder, store=store, chat=chat, judge_chat=judge_chat, top_k=top_k)
            results.append(result)
            progress(f"{q.id} run {run}: " + (result.error or f"retrieval={result.retrieval_hit} evidence={result.evidence_hit} citation={result.citation_hit} correct={result.correct}"))
            if pause:
                sleep(pause)
    return results


def _rate(values: list[bool | None]) -> tuple[int, int]:
    """(true count, counted). None means 'not measured' and is left out."""
    counted = [v for v in values if v is not None]
    return sum(counted), len(counted)


def summarize(questions: list[Question], results: list[RunResult]) -> dict:
    by_id = {q.id: q for q in questions}
    groups: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        q = by_id[r.question_id]
        if not q.answerable:
            continue
        groups["all"].append(r)
        groups[f"repo:{q.repo}"].append(r)
        groups[f"kind:{q.kind}"].append(r)

    def block(rs: list[RunResult]) -> dict:
        out = {}
        for name in ("retrieval_hit", "evidence_hit", "citation_hit", "correct"):
            hit, total = _rate([getattr(r, name) for r in rs])
            out[name] = {"hit": hit, "total": total}
        flags = [f for r in rs for f in r.facts_found]  # one flag per required fact
        out["facts"] = {"hit": sum(flags), "total": len(flags)}
        out["errors"] = sum(1 for r in rs if r.error)
        return out

    unanswerable = [r for r in results if not by_id[r.question_id].answerable]
    refused, judged = _rate([r.refused for r in unanswerable])
    return {
        "groups": {name: block(rs) for name, rs in sorted(groups.items())},
        "unanswerable": {"refused": refused, "judged": judged, "asked": len(unanswerable),
                         "errors": sum(1 for r in unanswerable if r.error)},
    }


def pick_for_hand_check(questions: list[Question], results: list[RunResult], n: int = 10, seed: int = 7) -> list[RunResult]:
    """A fixed random pick of judged answers so the same ones show up every time."""
    answerable = {q.id for q in questions if q.answerable}
    judged = [r for r in results if r.question_id in answerable and r.facts_found]
    return random.Random(seed).sample(judged, min(n, len(judged)))


def format_report(questions: list[Question], results: list[RunResult], summary: dict, meta: dict) -> str:
    by_id = {q.id: q for q in questions}

    def pct(b: dict, name: str) -> str:
        x = b[name]
        return f"{x['hit']}/{x['total']}" + (f" ({100 * x['hit'] // x['total']}%)" if x["total"] else "")

    lines = ["# Baseline evaluation", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items()]
    lines += ["", "## Scores (answerable questions only)", "",
              "| group | retrieval hit (file) | evidence shown (chunk) | citation hit | answer correct (all facts) | facts found | errors |", "|---|---|---|---|---|---|---|"]
    for name, b in summary["groups"].items():
        lines.append(f"| {name} | {pct(b, 'retrieval_hit')} | {pct(b, 'evidence_hit')} | {pct(b, 'citation_hit')} | {pct(b, 'correct')} | {pct(b, 'facts')} | {b['errors']} |")
    u = summary["unanswerable"]
    lines += ["", "## Unanswerable questions (not scored in phase 3)", "",
              f"Asked {u['asked']} times. The judge said the answer refused in {u['refused']} of {u['judged']} judged. Errors: {u['errors']}.",
              "In phase 5 this number should go up. Read the answers below yourself.", ""]
    for r in results:
        if not by_id[r.question_id].answerable:
            lines += [f"- **{r.question_id}** (run {r.run}, refused={r.refused}): {r.error or r.answer}", ""]
    lines += ["## Misses", ""]
    for r in results:
        q = by_id[r.question_id]
        if q.answerable and (r.evidence_hit is False or not (r.retrieval_hit and r.citation_hit and r.correct)):
            lines.append(f"- {r.question_id} run {r.run}: retrieval={r.retrieval_hit} evidence={r.evidence_hit} citation={r.citation_hit} correct={r.correct}"
                         + (f" | error: {r.error}" if r.error else "") + (f" | judge error: {r.judge_error}" if r.judge_error else ""))
    lines += ["", "## Hand check (read these yourself and say if you agree with the judge)", ""]
    for r in pick_for_hand_check(questions, results):
        q = by_id[r.question_id]
        lines += [f"### {r.question_id} run {r.run}: {q.question}", "", f"Answer: {r.answer}", ""]
        for fact, found in zip(q.must_say, r.facts_found):
            lines.append(f"- [{'x' if found else ' '}] {fact}")
        lines.append("")
    return "\n".join(lines)


def results_to_json(results: list[RunResult]) -> list[dict]:
    return [asdict(r) for r in results]
