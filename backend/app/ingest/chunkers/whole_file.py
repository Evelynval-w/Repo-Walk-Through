"""Keep a small config or script file as one chunk (the dispatcher splits long ones)."""
from __future__ import annotations

from app.ingest.chunkers.common import split_lines
from app.ingest.models import Chunk

LANGUAGE_BY_EXTENSION = {
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".xml": "xml",
    ".properties": "properties",
    ".sql": "sql",
    ".cypher": "cypher",
    ".cfg": "ini",
    ".ini": "ini",
    ".sh": "shell",
    ".tpl": "helm-template",
    ".txt": "text",
}


def language_for(file_name: str, extension: str) -> str:
    if file_name.startswith("Dockerfile"):
        return "dockerfile"
    if file_name.startswith(".env"):
        return "env-example"
    return LANGUAGE_BY_EXTENSION.get(extension, "text")


def chunk_whole_file(repo: str, path: str, text: str, language: str) -> list[Chunk]:
    if not text.strip():
        return []
    lines = split_lines(text)
    name = path.rsplit("/", 1)[-1]
    return [Chunk(repo, path, name, "file", language, text.strip("\n"), 1, len(lines))]
