"""Settings for the app.

Everything comes from environment variables. For local work, the values are
read from a `.env` file in the project root. That file is git-ignored, so keys
never end up in the repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# backend/app/config.py -> project root is two folders up from "backend"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a file into os.environ.

    Values that are already set in the real environment win, so a CI job or a
    shell export can override the file.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_embed_model: str
    openrouter_api_key: str
    chroma_dir: Path
    repos_dir: Path
    repos_config: Path
    chunk_max_chars: int
    embed_batch_size: int
    embed_pause_seconds: float
    max_file_bytes: int
    config_file_max_bytes: int
    # Phase 2. These have defaults so the phase 1 tests did not need to change.
    gemini_chat_model: str = "gemini-2.5-flash"
    openrouter_model: str = ""  # no default on purpose, see list_openrouter_models
    top_k: int = 6
    chat_temperature: float = 0.2


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    get = os.environ.get
    return Settings(
        gemini_api_key=get("GEMINI_API_KEY", ""),
        # No default on purpose: the model names change over time.
        # Run `python -m app.tools.list_models` to see what your key can use.
        gemini_embed_model=get("GEMINI_EMBED_MODEL", ""),
        openrouter_api_key=get("OPENROUTER_API_KEY", ""),
        chroma_dir=_path(get("CHROMA_DIR", "data/chroma")),
        repos_dir=_path(get("REPOS_DIR", "data/repos")),
        repos_config=_path(get("REPOS_CONFIG", "repos.toml")),
        chunk_max_chars=int(get("CHUNK_MAX_CHARS", "3000")),
        embed_batch_size=int(get("EMBED_BATCH_SIZE", "50")),
        embed_pause_seconds=float(get("EMBED_PAUSE_SECONDS", "0.5")),
        max_file_bytes=int(get("MAX_FILE_BYTES", "300000")),
        config_file_max_bytes=int(get("CONFIG_FILE_MAX_BYTES", "60000")),
        gemini_chat_model=get("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        openrouter_model=get("OPENROUTER_MODEL", ""),
        top_k=int(get("TOP_K", "6")),
        chat_temperature=float(get("CHAT_TEMPERATURE", "0.2")),
    )
