"""Helpers shared by all chunkers."""
from __future__ import annotations

from dataclasses import replace

from app.ingest.models import Chunk


def split_lines(text: str) -> list[str]:
    """Split on \\n only, so line numbers match what parsers and editors count."""
    return [line.rstrip("\r") for line in text.split("\n")]


def trim_end(lines: list[str], first: int, last: int) -> int:
    """Move `last` up past blank lines, so a chunk does not end with empty lines."""
    while last > first and not lines[last].strip():
        last -= 1
    return last


def gap_chunks(
    lines: list[str],
    covered: list[tuple[int, int]],
    repo: str,
    path: str,
    language: str,
) -> list[Chunk]:
    """Make chunks from the lines that no function or class covers.

    `covered` holds (first_row, last_row) pairs, 0-based and inclusive.
    What is left is imports, constants, route setup, scripts at the bottom of
    a file. We keep it so no code is silently lost.
    """
    taken = [False] * len(lines)
    for first, last in covered:
        for row in range(max(first, 0), min(last, len(lines) - 1) + 1):
            taken[row] = True

    chunks: list[Chunk] = []
    row = 0
    while row < len(lines):
        if taken[row]:
            row += 1
            continue
        start = row
        while row < len(lines) and not taken[row]:
            row += 1
        end = row - 1
        # trim blank lines at both ends
        while start <= end and not lines[start].strip():
            start += 1
        while end >= start and not lines[end].strip():
            end -= 1
        if start > end:
            continue  # only blank lines
        if not any(ch.isalnum() for line in lines[start : end + 1] for ch in line):
            continue  # only braces and punctuation, like a lone "}"
        chunks.append(
            Chunk(
                repo=repo,
                path=path,
                name="top-level code",
                kind="module",
                language=language,
                text="\n".join(lines[start : end + 1]),
                start_line=start + 1,
                end_line=end + 1,
            )
        )
    return chunks


def split_chunk(chunk: Chunk, max_chars: int) -> list[Chunk]:
    """Cut a chunk that is too long into parts.

    We cut at a blank line when we can, and at any line when we must.
    A single line longer than max_chars is cut in the middle (minified code).
    There is no overlap between parts. That keeps the references simple.
    """
    if len(chunk.text) <= max_chars:
        return [chunk]

    lines = chunk.text.split("\n")
    parts: list[tuple[int, int]] = []  # (first_index, last_index) inside `lines`
    start = 0
    while start < len(lines):
        size = 0
        end = start
        last_blank = None
        while end < len(lines):
            added = len(lines[end]) + 1
            if size + added > max_chars and end > start:
                break
            size += added
            if not lines[end].strip():
                last_blank = end
            end += 1
        if end < len(lines) and last_blank is not None and last_blank > start:
            end = last_blank + 1  # cut right after the last blank line
        parts.append((start, end - 1))
        start = end

    result: list[Chunk] = []
    for first, last in parts:
        text = "\n".join(lines[first : last + 1])
        # a very long single line: hard cut
        pieces = [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [""]
        for piece in pieces:
            result.append(
                replace(
                    chunk,
                    text=piece,
                    part=len(result) + 1,
                    start_line=(chunk.start_line + first) if chunk.start_line is not None else None,
                    end_line=(chunk.start_line + last) if chunk.start_line is not None else None,
                )
            )
    return result
