"""Pick the right chunker for a file."""
from __future__ import annotations

import posixpath

from app.ingest.chunkers.common import split_chunk
from app.ingest.chunkers.markdown import chunk_markdown
from app.ingest.chunkers.notebook import chunk_notebook
from app.ingest.chunkers.python_ast import chunk_python
from app.ingest.chunkers.tree_sitter_chunker import EXTENSIONS as TREE_SITTER_EXTENSIONS
from app.ingest.chunkers.tree_sitter_chunker import chunk_with_tree_sitter
from app.ingest.chunkers.whole_file import LANGUAGE_BY_EXTENSION, chunk_whole_file, language_for
from app.ingest.models import Chunk

LOCK_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
}


def file_kind(path: str) -> str | None:
    """Which chunker handles this file? None means: not code or docs, skip it."""
    name = posixpath.basename(path)
    extension = posixpath.splitext(name)[1].lower()

    if name in LOCK_FILES:
        return None
    if extension == ".py":
        return "python"
    if extension in TREE_SITTER_EXTENSIONS:
        return "tree-sitter"
    if extension == ".ipynb":
        return "notebook"
    if extension in (".md", ".markdown"):
        return "markdown"
    if name.startswith("Dockerfile"):
        return "whole"
    if name.startswith(".env") and name.endswith(".example"):
        return "whole"
    if extension == ".txt":
        return "whole" if name.startswith("requirements") else None
    if extension in LANGUAGE_BY_EXTENSION:
        return "whole"
    return None


def chunk_file(repo: str, path: str, text: str, max_chars: int) -> list[Chunk]:
    kind = file_kind(path)
    extension = posixpath.splitext(path)[1].lower()
    name = posixpath.basename(path)

    if kind == "python":
        chunks = chunk_python(repo, path, text)
    elif kind == "tree-sitter":
        chunks = chunk_with_tree_sitter(repo, path, text, extension)
    elif kind == "notebook":
        chunks = chunk_notebook(repo, path, text)
    elif kind == "markdown":
        chunks = chunk_markdown(repo, path, text)
    elif kind == "whole":
        chunks = chunk_whole_file(repo, path, text, language_for(name, extension))
    else:
        return []

    result: list[Chunk] = []
    for chunk in chunks:
        result.extend(split_chunk(chunk, max_chars))
    return result
