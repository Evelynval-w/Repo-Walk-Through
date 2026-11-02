import json

import pytest

from app.ingest import index
from app.ingest.embedder import FakeEmbedder
from app.ingest.models import Chunk
from app.ingest.store import ChunkStore, StoreMismatch


def sample_chunks():
    return [
        Chunk("airline", "BookingService.java", "BookingService.book", "method", "java",
              "book a flight, check available seats, create the booking", 10, 30),
        Chunk("airline", "MilesRewardService.java", "MilesRewardService.reward", "method", "java",
              "after three flights in a year create a discount code", 5, 20),
        Chunk("credit", "04_modeling.ipynb", "cell 7", "notebook_cell", "python",
              "train xgboost on the loan data and compare roc auc", cell=7),
    ]


def test_store_finds_the_closest_chunk(tmp_path):
    fake = FakeEmbedder()
    store = ChunkStore(tmp_path / "db", fake.name)
    chunks = sample_chunks()
    store.upsert(chunks, fake.embed_documents([c.embed_text() for c in chunks]))

    assert store.count() == 3
    hits = store.query(fake.embed_query("which service books a flight and checks seats"), k=2)
    assert hits[0]["metadata"]["name"] == "BookingService.book"
    assert hits[0]["metadata"]["start_line"] == 10
    assert dict(store.repo_counts()) == {"airline": 2, "credit": 1}


def test_notebook_chunks_keep_their_cell_number(tmp_path):
    fake = FakeEmbedder()
    store = ChunkStore(tmp_path / "db", fake.name)
    chunks = sample_chunks()
    store.upsert(chunks, fake.embed_documents([c.embed_text() for c in chunks]))
    hit = store.query(fake.embed_query("xgboost roc auc"), k=1)[0]
    assert hit["metadata"]["cell"] == 7
    assert "start_line" not in hit["metadata"]


def test_delete_repo_removes_only_that_repo(tmp_path):
    fake = FakeEmbedder()
    store = ChunkStore(tmp_path / "db", fake.name)
    chunks = sample_chunks()
    store.upsert(chunks, fake.embed_documents([c.embed_text() for c in chunks]))
    store.delete_repo("airline")
    assert dict(store.repo_counts()) == {"credit": 1}


def test_index_built_with_another_embedder_is_refused_until_reset(tmp_path):
    path = tmp_path / "db"
    ChunkStore(path, "fake-hash-64")
    with pytest.raises(StoreMismatch):
        ChunkStore(path, "gemini:other-model")


def test_reset_empties_the_store(tmp_path):
    fake = FakeEmbedder()
    store = ChunkStore(tmp_path / "db", fake.name)
    chunks = sample_chunks()
    store.upsert(chunks, fake.embed_documents([c.embed_text() for c in chunks]))
    store.reset()
    assert store.count() == 0


def test_index_command_end_to_end_with_fake_vectors(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repos" / "demo"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# Demo\nA tiny repo.\n", encoding="utf-8")
    config = tmp_path / "repos.toml"
    config.write_text('[[repos]]\nname = "demo"\n', encoding="utf-8")

    monkeypatch.setenv("REPOS_DIR", str(tmp_path / "repos"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "out" / "chroma"))
    monkeypatch.setenv("REPOS_CONFIG", str(config))

    assert index.main(["--fake", "--show", "1"]) == 0
    output = capsys.readouterr().out
    assert "chunks: 2" in output
    assert "Done. The index holds 2 chunks" in output

    report = json.loads((tmp_path / "out" / "index_report.json").read_text())
    assert report["demo"]["chunks"] == 2


def test_dry_run_does_not_create_the_database(tmp_path, monkeypatch):
    repo = tmp_path / "repos" / "demo"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    config = tmp_path / "repos.toml"
    config.write_text('[[repos]]\nname = "demo"\n', encoding="utf-8")
    monkeypatch.setenv("REPOS_DIR", str(tmp_path / "repos"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "out" / "chroma"))
    monkeypatch.setenv("REPOS_CONFIG", str(config))

    assert index.main(["--dry-run"]) == 0
    assert not (tmp_path / "out" / "chroma").exists()


def test_missing_repo_folder_is_reported_not_crashing(tmp_path, monkeypatch, capsys):
    config = tmp_path / "repos.toml"
    config.write_text('[[repos]]\nname = "ghost"\n', encoding="utf-8")
    monkeypatch.setenv("REPOS_DIR", str(tmp_path / "repos"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "out" / "chroma"))
    monkeypatch.setenv("REPOS_CONFIG", str(config))

    assert index.main(["--dry-run"]) == 0
    assert "not found" in capsys.readouterr().out
