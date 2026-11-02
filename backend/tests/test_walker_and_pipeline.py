import subprocess

from app.ingest.pipeline import chunk_repo
from app.ingest.walker import OwnerRule, RepoConfig

FAKE_GOOGLE_KEY = "AIza" + "B" * 35  # built at run time, so this file holds no key-like text


def write(root, rel, text="x = 1\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_repo(root):
    write(root, "src/app.py", "def run():\n    return 1\n")
    write(root, "src/Plane.java", "public class Plane {\n    int seats;\n}\n")
    write(root, "README.md", "# Demo\nHello there.\n")
    write(root, ".env.example", "API_KEY=\n")
    write(root, ".env", "API_KEY=real-secret-value\n")
    write(root, "node_modules/lib/index.js", "export const a = 1;\n")
    write(root, "data/results.py", "x = 1\n")  # excluded by prefix in the config below
    write(root, "package-lock.json", "{}")
    write(root, "config/big.json", "{" + '"a": 1,' * 400 + '"b": 2}')  # over the config size limit
    write(root, "notes.py", f'KEY = "{FAKE_GOOGLE_KEY}"\n')
    (root / "logo.png").write_bytes(b"\x89PNG\x00\x00")


def test_walker_skips_what_it_should_and_keeps_the_rest(tmp_path, settings):
    root = tmp_path / "demo"
    build_repo(root)
    config = RepoConfig(
        "demo",
        exclude=("data/",),
        owners=(OwnerRule("*Plane*", "Hetvi (teammate)"),),
    )

    result = chunk_repo(config, root, settings)

    paths = {c.path for c in result.chunks}
    assert paths == {"src/app.py", "src/Plane.java", "README.md", ".env.example"}
    # Without git, the folder walk never enters node_modules, so it is not counted.
    assert result.skipped["excluded in repos.toml"] == 1  # data/
    assert result.skipped["env file"] == 1  # .env, but not .env.example
    assert result.skipped["too large"] == 1  # config/big.json
    assert result.skipped["not code or docs"] == 2  # package-lock.json and logo.png


def test_in_a_git_repo_only_tracked_files_are_listed(tmp_path, settings):
    root = tmp_path / "gitrepo"
    write(root, "app.py", "def run():\n    return 1\n")
    write(root, "ignored.py", "x = 1\n")
    write(root, "node_modules/lib/index.js", "export const a = 1;\n")
    write(root, ".gitignore", "ignored.py\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # respects .gitignore

    result = chunk_repo(RepoConfig("gitrepo"), root, settings)

    assert {c.path for c in result.chunks} == {"app.py"}
    assert result.skipped["skipped folder"] == 1  # node_modules was tracked, so it is counted


def test_owner_rules_tag_the_right_files(tmp_path, settings):
    root = tmp_path / "demo"
    build_repo(root)
    config = RepoConfig("demo", exclude=("data/",), owners=(OwnerRule("*Plane*", "Hetvi (teammate)"),))

    owners = {c.path: c.owner for c in chunk_repo(config, root, settings).chunks}

    assert owners["src/Plane.java"] == "Hetvi (teammate)"
    assert owners["src/app.py"] == "me"


def test_chunks_with_a_key_are_dropped_and_the_key_is_not_reported(tmp_path, settings):
    root = tmp_path / "demo"
    build_repo(root)
    result = chunk_repo(RepoConfig("demo", exclude=("data/",)), root, settings)

    assert len(result.dropped_for_secret) == 1
    assert "notes.py" in result.dropped_for_secret[0]
    assert "google api key" in result.dropped_for_secret[0]
    assert FAKE_GOOGLE_KEY not in " ".join(result.dropped_for_secret)
    assert all(FAKE_GOOGLE_KEY not in c.text for c in result.chunks)


def test_words_that_only_contain_sk_dash_are_not_flagged(tmp_path, settings):
    root = tmp_path / "demo"
    write(root, "doc.md", "# Notes\nThis task-based-evaluation-run-with-many-parts is fine.\n")
    result = chunk_repo(RepoConfig("demo"), root, settings)
    assert result.dropped_for_secret == []
    assert len(result.chunks) == 1
