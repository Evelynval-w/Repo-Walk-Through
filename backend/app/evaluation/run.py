"""Command line: python -m app.evaluation.run [--runs 3] [--only cr-01 sh-02]

Writes eval/reports/<name>.json (every answer) and <name>.md (the readable report).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from app.config import load_settings
from app.evaluation.questions import load_questions
from app.evaluation.runner import format_report, results_to_json, run_all, summarize
from app.ingest.embedder import EmbeddingError
from app.ingest.store import StoreMismatch
from app.qa.chat import ChatError
from app.qa.services import build_services

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the evaluation questions against the real chatbot.")
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.toml")
    p.add_argument("--runs", type=int, default=1, help="1 for a quick check, 3 for numbers you report")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--only", nargs="*", help="question ids to run")
    p.add_argument("--pause", type=float, default=2.0, help="seconds to wait between calls (free-tier limits)")
    p.add_argument("--no-judge", action="store_true", help="only measure retrieval and citation")
    p.add_argument("--name", default="baseline")
    args = p.parse_args(argv)

    questions = load_questions(args.questions)
    if args.only:
        questions = [q for q in questions if q.id in set(args.only)]
        if not questions:
            print("No question matches --only.", file=sys.stderr)
            return 2

    try:
        embedder, store, chat = build_services(load_settings())
    except (ChatError, EmbeddingError, StoreMismatch) as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return 1

    results = run_all(
        questions, embedder=embedder, store=store, chat=chat,
        judge_chat=None if args.no_judge else chat,  # same model judges: see README limits
        top_k=args.top_k, runs=args.runs, pause=args.pause, progress=print,
    )
    summary = summarize(questions, results)
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chat model": getattr(chat, "name", "?"),
        "embedder": embedder.name,
        "top_k": args.top_k,
        "runs per question": args.runs,
        "questions": len(questions),
        "judge": "none" if args.no_judge else "same chat model",
    }
    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = out_dir / f"{args.name}-{stamp}"
    base.with_suffix(".json").write_text(json.dumps({"meta": meta, "summary": summary, "results": results_to_json(results)}, indent=2))
    report = format_report(questions, results, summary, meta)
    base.with_suffix(".md").write_text(report)
    print("\n" + report)
    print(f"\nSaved {base.with_suffix('.md')} and {base.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
