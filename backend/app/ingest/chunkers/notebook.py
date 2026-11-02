"""Cut a Jupyter notebook into one chunk per code cell.

We only read the code and markdown cells. The saved outputs are ignored. In
the credit risk notebooks they are mostly base64 images, which would make the
index huge and add nothing to search.
"""
from __future__ import annotations

import json

from app.ingest.models import Chunk

MARKDOWN_CONTEXT_CHARS = 800


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _title_from_markdown(markdown: str) -> str | None:
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or None
    return None


def chunk_notebook(repo: str, path: str, raw: str) -> list[Chunk]:
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError:
        return []

    language = (
        notebook.get("metadata", {}).get("language_info", {}).get("name")
        or notebook.get("metadata", {}).get("kernelspec", {}).get("language")
        or "python"
    )

    chunks: list[Chunk] = []
    pending_markdown: list[str] = []  # markdown cells waiting for the next code cell
    pending_cell = 0

    def flush_markdown() -> None:
        if not pending_markdown:
            return
        text = "\n\n".join(pending_markdown)
        title = _title_from_markdown(text) or f"notes before cell {pending_cell}"
        chunks.append(
            Chunk(repo, path, title, "markdown_section", "markdown", text, cell=pending_cell)
        )
        pending_markdown.clear()

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        kind = cell.get("cell_type")
        source = _source(cell).strip()
        if kind == "markdown":
            if source:
                if not pending_markdown:
                    pending_cell = index
                pending_markdown.append(source)
        elif kind == "code" and source:
            markdown = "\n\n".join(pending_markdown)[:MARKDOWN_CONTEXT_CHARS]
            name = _title_from_markdown(markdown) or f"cell {index}"
            if markdown:
                # keep the explanation next to the code, written as comments
                notes = "# Notes above this cell:\n# " + markdown.replace("\n", "\n# ")
                text = notes + "\n\n" + source
            else:
                text = source
            chunks.append(
                Chunk(repo, path, name, "notebook_cell", language, text, cell=index)
            )
            pending_markdown.clear()
    flush_markdown()
    return chunks
