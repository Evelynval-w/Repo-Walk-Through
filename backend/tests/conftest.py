import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path):
    """Small limits so tests can trigger them with tiny files."""
    return Settings(
        gemini_api_key="",
        gemini_embed_model="",
        openrouter_api_key="",
        chroma_dir=tmp_path / "chroma",
        repos_dir=tmp_path / "repos",
        repos_config=tmp_path / "repos.toml",
        chunk_max_chars=500,
        embed_batch_size=2,
        embed_pause_seconds=0,
        max_file_bytes=2000,
        config_file_max_bytes=1000,
    )


def shape(chunks):
    """(name, kind, start_line, end_line) for each chunk, easy to compare in tests."""
    return [(c.name, c.kind, c.start_line, c.end_line) for c in chunks]
