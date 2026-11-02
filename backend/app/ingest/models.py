"""The Chunk: one piece of code or text that we index and later search."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Chunk:
    repo: str
    path: str  # path inside the repo, always with forward slashes
    name: str  # function, class, route, heading or cell title
    kind: str  # function, method, class, interface, route, module, notebook_cell, markdown_section, file
    language: str
    text: str
    start_line: int | None = None  # 1-based, None for notebook cells
    end_line: int | None = None
    cell: int | None = None  # 1-based cell number, only for notebooks
    owner: str = "me"
    part: int = 1  # 2, 3, ... when a long block was split

    @property
    def id(self) -> str:
        where = self.start_line if self.start_line is not None else f"cell{self.cell}"
        return f"{self.repo}/{self.path}#{self.kind}:{self.name}@{where}:{self.part}"

    def embed_text(self) -> str:
        """The text we send to the embedding model.

        A short header with the repo, file and name helps the search match
        questions like "where is the booking logic in timeout-airline".
        """
        header = f"{self.repo}/{self.path}\n{self.kind}: {self.name}"
        return f"{header}\n\n{self.text}"

    def metadata(self) -> dict:
        """Flat metadata for Chroma. It does not accept None values."""
        data = {
            "repo": self.repo,
            "path": self.path,
            "name": self.name,
            "kind": self.kind,
            "language": self.language,
            "owner": self.owner,
            "part": self.part,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "cell": self.cell,
        }
        return {k: v for k, v in data.items() if v is not None}

    def with_owner(self, owner: str) -> "Chunk":
        return replace(self, owner=owner)
