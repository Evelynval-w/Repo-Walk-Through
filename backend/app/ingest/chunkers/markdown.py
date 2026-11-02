"""Cut a markdown file into sections, one per heading."""
from __future__ import annotations

import re

from app.ingest.chunkers.common import split_lines
from app.ingest.models import Chunk

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


def chunk_markdown(repo: str, path: str, text: str) -> list[Chunk]:
    lines = split_lines(text)

    # find headings, ignoring lines inside fenced code blocks
    headings: list[tuple[int, int, str]] = []  # (row, level, title)
    in_fence = False
    for row, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match:
            headings.append((row, len(match.group(1)), match.group(2)))

    # sections: (first_row, last_row, name)
    sections: list[tuple[int, int, str]] = []
    first_heading_row = headings[0][0] if headings else len(lines)
    if first_heading_row > 0:
        sections.append((0, first_heading_row - 1, "(intro)"))

    stack: list[tuple[int, str]] = []  # (level, title) for names like "A > B"
    for index, (row, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        last = (headings[index + 1][0] - 1) if index + 1 < len(headings) else len(lines) - 1
        sections.append((row, last, " > ".join(t for _, t in stack)))

    chunks: list[Chunk] = []
    for first, last, name in sections:
        body = "\n".join(lines[first : last + 1]).strip()
        has_text = any(
            line.strip() and not HEADING.match(line) for line in lines[first : last + 1]
        )
        if not body or not has_text:
            continue  # a heading with nothing under it
        chunks.append(
            Chunk(repo, path, name, "markdown_section", "markdown", body, first + 1, last + 1)
        )
    return chunks
