"""Judge the SAVED answers again against the current gold facts.

Use it after fixing questions.toml. It makes no answer calls and does not touch
search, so only the judge changes. Writes a new report next to the old one.

python -m app.evaluation.rejudge ../eval/reports/baseline-20261105-0956.json --name rejudged
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from app.config import load_settings
from app.evaluation.judge import JudgeError, judge_facts, judge_refusal
from app.evaluation.questions import Question, load_questions
from app.evaluation.runner import RunResult, format_report, results_to_json, summarize
from app.qa.chat import ChatError, ChatModel
from app.qa.services import build_chat

ROOT = Path(__file__).resolve().parents[3]


def rejudge(questions: list[Question], old_results: list[dict], judge_chat: ChatModel) -> list[RunResult]:
    by_id = {q.id: q for q in questions}
    out = []
    for item in old_results:
        r = RunResult(**item)
        q = by_id.get(r.question_id)
        if q is None or r.error or not r.answer:
            out.append(r)
            continue
        r.judge_error, r.facts_found, r.correct, r.refused = "", [], None, None
        try:
            if q.answerable:
                r.facts_found = judge_facts(judge_chat, q.question, r.answer, q.must_say)
                r.correct = all(r.facts_found)
            else:
                r.refused = judge_refusal(judge_chat, q.question, r.answer)
        except (JudgeError, ChatError) as error:
            r.judge_error = str(error)
            r.correct = False if q.answerable else None
        out.append(r)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.toml")
    p.add_argument("--name", default="rejudged")
    args = p.parse_args(argv)

    old = json.loads(args.report.read_text())
    questions = load_questions(args.questions)
    ids = {q.id for q in questions}
    kept = [x for x in old["results"] if x["question_id"] in ids]
    try:
        judge_chat = build_chat(load_settings())
    except ChatError as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return 1

    results = rejudge(questions, kept, judge_chat)
    summary = summarize(questions, results)
    meta = dict(old["meta"], note=f"answers from {args.report.name}, judged again with the current gold facts",
                rejudged=datetime.now().strftime("%Y-%m-%d %H:%M"))
    base = args.report.parent / f"{args.name}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    base.with_suffix(".json").write_text(json.dumps({"meta": meta, "summary": summary, "results": results_to_json(results)}, indent=2))
    report = format_report(questions, results, summary, meta)
    base.with_suffix(".md").write_text(report)
    print(report)
    print(f"\nSaved {base.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
