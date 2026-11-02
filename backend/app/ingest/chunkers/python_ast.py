"""Cut a Python file into functions, classes and methods with the `ast` module."""
from __future__ import annotations

import ast

from app.ingest.chunkers.common import gap_chunks, split_lines, trim_end
from app.ingest.models import Chunk

FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _first_row(node: ast.AST) -> int:
    """0-based first row of a def, including its decorators."""
    rows = [node.lineno]
    rows += [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(rows) - 1


def _last_row(node: ast.AST) -> int:
    return node.end_lineno - 1


def chunk_python(repo: str, path: str, text: str) -> list[Chunk]:
    lines = split_lines(text)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Not valid Python 3 (or a template). Keep it as one block so it is not lost.
        return [
            Chunk(repo, path, path.rsplit("/", 1)[-1], "file", "python", text, 1, len(lines))
        ]

    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []

    def add(name: str, kind: str, first: int, last: int) -> None:
        last = trim_end(lines, first, last)
        covered.append((first, last))
        chunks.append(
            Chunk(
                repo=repo,
                path=path,
                name=name,
                kind=kind,
                language="python",
                text="\n".join(lines[first : last + 1]),
                start_line=first + 1,
                end_line=last + 1,
            )
        )

    for node in tree.body:
        if isinstance(node, FUNCTION_NODES):
            add(node.name, "function", _first_row(node), _last_row(node))
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, FUNCTION_NODES)]
            class_first = _first_row(node)
            if not methods:
                add(node.name, "class", class_first, _last_row(node))
                continue
            # The class chunk is the part above the first method:
            # decorators, signature, docstring and class attributes.
            header_last = _first_row(methods[0]) - 1
            if header_last >= class_first:
                add(node.name, "class", class_first, header_last)
            for method in methods:
                add(f"{node.name}.{method.name}", "method", _first_row(method), _last_row(method))
        # anything else (imports, constants, scripts) is handled as a gap below

    chunks.extend(gap_chunks(lines, covered, repo, path, "python"))
    chunks.sort(key=lambda c: c.start_line or 0)
    return chunks
