"""Turn one repo folder into a list of chunks, with a report of what was skipped."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.config import Settings
from app.ingest.chunkers import chunk_file
from app.ingest.models import Chunk
from app.ingest.secrets import CONFIG_LANGUAGES, find_secret, redact_secret_values
from app.ingest.walker import RepoConfig, list_files, owner_for, read_text, skip_reason


@dataclass
class RepoResult:
    repo: str
    chunks: list[Chunk] = field(default_factory=list)
    files_indexed: int = 0
    skipped: Counter = field(default_factory=Counter)
    dropped_for_secret: list[str] = field(default_factory=list)  # "chunk id (pattern name)"
    redacted: list[str] = field(default_factory=list)  # "chunk id (n values)"
    errors: list[str] = field(default_factory=list)


def chunk_repo(config: RepoConfig, root: Path, settings: Settings) -> RepoResult:
    result = RepoResult(repo=config.name)
    seen_ids: set[str] = set()

    for rel_path in list_files(root):
        full = root / rel_path
        try:
            size = full.stat().st_size
        except OSError:
            result.skipped["unreadable"] += 1
            continue

        reason = skip_reason(rel_path, size, config, settings)
        if reason:
            result.skipped[reason] += 1
            continue

        try:
            text = read_text(full)
            if text is None:
                result.skipped["binary"] += 1
                continue
            chunks = chunk_file(config.name, rel_path, text, settings.chunk_max_chars)
        except Exception as error:  # one bad file must not stop the whole run
            result.errors.append(f"{rel_path}: {type(error).__name__}: {error}")
            continue

        owner = owner_for(rel_path, config)
        kept = 0
        for chunk in chunks:
            chunk = chunk.with_owner(owner)
            label = find_secret(chunk.text)
            if label:
                result.dropped_for_secret.append(f"{chunk.id} ({label})")
                continue
            hidden_text, hidden = redact_secret_values(chunk.text, chunk.language in CONFIG_LANGUAGES)
            if hidden:
                chunk = replace(chunk, text=hidden_text)
                result.redacted.append(f"{chunk.id} ({hidden} value{'s' if hidden > 1 else ''})")
            if chunk.id in seen_ids:
                result.errors.append(f"duplicate chunk id skipped: {chunk.id}")
                continue
            seen_ids.add(chunk.id)
            result.chunks.append(chunk)
            kept += 1
        if kept:
            result.files_indexed += 1

    return result
