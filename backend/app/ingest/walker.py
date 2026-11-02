"""Find the files of a repo and decide which ones are worth indexing."""
from __future__ import annotations

import fnmatch
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.ingest.chunkers import file_kind

# Folders that never hold my own code, in any repo.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".next",
    ".idea", ".vscode", ".claude", "dist", "build", "target", "coverage",
}
# Notebooks are big because of saved outputs. We drop the outputs later,
# so the size limit for them is much higher.
NOTEBOOK_MAX_BYTES = 20_000_000


@dataclass(frozen=True)
class OwnerRule:
    glob: str
    owner: str


@dataclass(frozen=True)
class RepoConfig:
    name: str
    exclude: tuple[str, ...] = ()
    owners: tuple[OwnerRule, ...] = ()


def load_repo_configs(path: Path) -> list[RepoConfig]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    configs = []
    for entry in data.get("repos", []):
        owners = tuple(OwnerRule(o["glob"], o["owner"]) for o in entry.get("owners", []))
        configs.append(RepoConfig(entry["name"], tuple(entry.get("exclude", [])), owners))
    return configs


def list_files(root: Path) -> list[str]:
    """All files of the repo as relative paths with forward slashes.

    If it is a git repo we ask git, so we only see tracked files and
    .gitignore is respected. Otherwise we walk the folder.
    """
    if (root / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z"],
                check=True, capture_output=True,
            ).stdout.decode("utf-8", errors="replace")
            return sorted(p for p in out.split("\0") if p)
        except (OSError, subprocess.CalledProcessError):
            pass  # no git on this machine, fall back to walking
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            full = Path(folder) / name
            found.append(full.relative_to(root).as_posix())
    return sorted(found)


def skip_reason(rel_path: str, size: int, config: RepoConfig, settings: Settings) -> str | None:
    """Return why a file is skipped, or None if it should be indexed."""
    parts = rel_path.split("/")
    name = parts[-1]
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return "skipped folder"
    if any(rel_path.startswith(prefix) for prefix in config.exclude):
        return "excluded in repos.toml"
    if name.startswith(".env") and not name.endswith(".example"):
        return "env file"
    kind = file_kind(rel_path)
    if kind is None:
        return "not code or docs"
    if kind == "notebook":
        limit = NOTEBOOK_MAX_BYTES
    elif kind == "whole":
        limit = settings.config_file_max_bytes
    else:
        limit = settings.max_file_bytes
    if size > limit:
        return "too large"
    return None


def owner_for(rel_path: str, config: RepoConfig) -> str:
    name = rel_path.rsplit("/", 1)[-1]
    for rule in config.owners:
        if fnmatch.fnmatch(name, rule.glob):
            return rule.owner
    return "me"


def read_text(path: Path) -> str | None:
    """Read a file as text. Returns None for binary files."""
    raw = path.read_bytes()
    if b"\x00" in raw:
        return None
    return raw.decode("utf-8-sig", errors="replace")
