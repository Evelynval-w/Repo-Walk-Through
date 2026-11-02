"""Command line tool that builds the index.

Run it from the backend folder:

    python -m app.ingest.index --dry-run --show 3   # chunk only, no key needed
    python -m app.ingest.index                      # chunk, embed with Gemini, store
    python -m app.ingest.index --fake               # use fake vectors (for trying it out)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from app.config import load_settings
from app.ingest.embedder import Embedder, EmbeddingError, FakeEmbedder, GeminiEmbedder
from app.ingest.pipeline import RepoResult, chunk_repo
from app.ingest.store import ChunkStore, StoreMismatch
from app.ingest.walker import load_repo_configs

PREVIEW_CHARS = 400


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the code index.")
    parser.add_argument("--only", help="index only this repo (name from repos.toml)")
    parser.add_argument("--dry-run", action="store_true", help="chunk only, do not embed or store")
    parser.add_argument("--fake", action="store_true", help="use fake vectors instead of Gemini")
    parser.add_argument("--reset", action="store_true", help="empty the index first")
    parser.add_argument("--show", type=int, default=0, help="print this many sample chunks per repo")
    return parser.parse_args(argv)


def print_report(results: list[RepoResult]) -> None:
    for result in results:
        kinds = Counter(c.kind for c in result.chunks)
        print(f"\n== {result.repo}")
        print(f"   files indexed: {result.files_indexed}, chunks: {len(result.chunks)}")
        print("   by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        if result.skipped:
            print("   skipped files: " + ", ".join(f"{k}={v}" for k, v in sorted(result.skipped.items())))
        if result.dropped_for_secret:
            print(f"   DROPPED {len(result.dropped_for_secret)} chunk(s) that look like they hold a key:")
            for line in result.dropped_for_secret[:10]:
                print(f"     - {line}")
        if result.redacted:
            print(f"   HIDDEN values of secret-looking settings in {len(result.redacted)} chunk(s):")
            for line in result.redacted[:10]:
                print(f"     - {line}")
        for error in result.errors[:10]:
            print(f"   problem: {error}")


def print_samples(results: list[RepoResult], count: int) -> None:
    for result in results:
        step = max(len(result.chunks) // count, 1) if count else 1
        for chunk in result.chunks[::step][:count]:
            where = f"lines {chunk.start_line}-{chunk.end_line}" if chunk.start_line else f"cell {chunk.cell}"
            print(f"\n--- {chunk.repo}/{chunk.path} | {chunk.kind} {chunk.name} | {where} | owner: {chunk.owner}")
            preview = chunk.text[:PREVIEW_CHARS]
            print(preview + ("..." if len(chunk.text) > PREVIEW_CHARS else ""))


def make_embedder(args: argparse.Namespace, settings) -> Embedder:
    if args.fake:
        return FakeEmbedder()
    return GeminiEmbedder(
        settings.gemini_api_key,
        settings.gemini_embed_model,
        batch_size=settings.embed_batch_size,
        pause_seconds=settings.embed_pause_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    if not settings.repos_config.is_file():
        print(f"Cannot find the repo list at {settings.repos_config}")
        return 1
    configs = load_repo_configs(settings.repos_config)
    if args.only:
        configs = [c for c in configs if c.name == args.only]
        if not configs:
            print(f"No repo called '{args.only}' in {settings.repos_config.name}")
            return 1

    results: list[RepoResult] = []
    for config in configs:
        root = settings.repos_dir / config.name
        if not root.is_dir():
            print(f"\n== {config.name}\n   not found at {root}. Clone it there first.")
            continue
        results.append(chunk_repo(config, root, settings))

    print_report(results)
    if args.show:
        print_samples(results, args.show)

    total_chunks = sum(len(r.chunks) for r in results)
    total_chars = sum(len(c.embed_text()) for r in results for c in r.chunks)
    print(f"\nTotal: {total_chunks} chunks, about {total_chars:,} characters to embed")

    report = {
        r.repo: {
            "files_indexed": r.files_indexed,
            "chunks": len(r.chunks),
            "by_kind": dict(Counter(c.kind for c in r.chunks)),
            "skipped": dict(r.skipped),
            "dropped_for_secret": r.dropped_for_secret,
            "redacted": r.redacted,
            "errors": r.errors,
        }
        for r in results
    }
    report_path = settings.chroma_dir.parent / "index_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.dry_run or not results:
        if args.dry_run:
            print("Dry run: nothing was embedded or stored.")
        return 0

    try:
        embedder = make_embedder(args, settings)
        store = ChunkStore(settings.chroma_dir, embedder.name)
        if args.reset:
            store.reset()
        for result in results:
            if not result.chunks:
                continue
            print(f"Embedding {result.repo}: {len(result.chunks)} chunks ...")
            vectors = embedder.embed_documents([c.embed_text() for c in result.chunks])
            store.delete_repo(result.repo)
            store.upsert(result.chunks, vectors)
    except (EmbeddingError, StoreMismatch) as error:
        print(f"\nStopped: {error}")
        return 1

    print(f"\nDone. The index holds {store.count()} chunks ({embedder.name}).")
    for repo, count in sorted(store.repo_counts().items()):
        print(f"  {repo}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
