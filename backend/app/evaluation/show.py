"""Read a saved report and print what went wrong, with no model calls.

python -m app.evaluation.show ../eval/reports/baseline-20261105-0956.json
python -m app.evaluation.show REPORT.json --ids sh-02 gh-05      # only these questions
python -m app.evaluation.show REPORT.json --all-runs             # every run, not just run 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.questions import load_questions

ROOT = Path(__file__).resolve().parents[3]


def diagnose(q, r) -> str:
    """One word for where this run went wrong."""
    if r["error"]:
        return "ERROR"
    if not r["retrieval_hit"]:
        return "SEARCH (expected file not in the top-k)"
    if r.get("evidence_hit") is False:
        return "SEARCH (right file, but the chunk with the answer was not shown)"
    if not r["citation_hit"]:
        return "CITATION (file was shown but not cited)"
    if not r["correct"]:
        return "ANSWER (right file cited, facts missing)"
    return "ok"


def render(questions, report, ids=None, all_runs=False) -> str:
    by_id = {q.id: q for q in questions}
    lines = []
    seen = set()
    for r in report["results"]:
        q = by_id.get(r["question_id"])
        if q is None or not q.answerable or (ids and q.id not in ids):
            continue
        if r["correct"] and not ids:
            continue
        if not all_runs and q.id in seen:
            continue
        seen.add(q.id)
        lines += [f"=== {q.id} run {r['run']}: {diagnose(q, r)}", f"Q: {q.question}",
                  f"expected: {', '.join(q.expected_files)}",
                  f"retrieved: {', '.join(r['retrieved_files']) or '-'}",
                  f"cited:     {', '.join(r['cited_files']) or '-'}",
                  f"A: {r['answer']}"]
        if q.evidence and r.get("evidence_found"):
            lines.append("evidence in the shown chunks:")
            lines += [f"  [{'x' if f else ' '}] {e}" for e, f in zip(q.evidence, r["evidence_found"])]
        lines.append("facts:")
        flags = r["facts_found"] or [False] * len(q.must_say)
        lines += [f"  [{'x' if f else ' '}] {fact}" for fact, f in zip(q.must_say, flags)]
        lines.append("")
    return "\n".join(lines) or "Nothing to show."


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.toml")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--all-runs", action="store_true")
    args = p.parse_args(argv)
    report = json.loads(args.report.read_text())
    print(render(load_questions(args.questions), report, set(args.ids or []), args.all_runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
