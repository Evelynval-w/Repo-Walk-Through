# Repo-Walk-Through

A chatbot that answers questions about my own GitHub repos and points to the file and function the answer comes from.

**Status: phase 1 of 8 (indexing).** The chatbot itself does not exist yet. Right now the project reads five repos, cuts the code into pieces, turns each piece into a vector and stores it in a local database. The plan for all phases is in [docs/PLAN.md](docs/PLAN.md).

## The five repos

| Repo | Language | What I index |
|---|---|---|
| credit_risk_ML_model | Python (notebooks) | one chunk per code cell, with the markdown above it. Saved outputs are dropped |
| synergyhack | JavaScript, JSX | functions, classes, Express routes, README and docs |
| timeout-airline | Java | classes and methods with their Javadoc. Some files are tagged with my teammate as owner |
| GameHub | Python, TypeScript | the services, frontend, infra and docs. The course exercises in `modules/` are skipped |
| fracton-scanner | Python | the code, README, workflows and Dockerfile. The `data/` folder is never read |

GameHub is a fork of a course template. Only the parts I wrote are meant to be answered as "my code".

## How the cutting works

- Python files use the built-in `ast` module. One chunk per function, class header and method.
- Java, JavaScript, JSX, TypeScript and TSX use tree-sitter. Same idea. Express routes like `router.get("/users", ...)` become their own chunks. Comments right above a definition stay with it.
- Notebooks: one chunk per code cell. I only read the code and markdown cells, because the saved outputs are mostly images.
- Markdown is cut by heading. Config files (YAML, JSON, properties, Dockerfile) are kept whole.
- Anything that is too long is cut into parts at a blank line.
- Code that is not inside a function or class (imports, constants, scripts) becomes a chunk called "top-level code", so nothing is lost.

Each chunk keeps its repo, file, name, kind, language, line numbers (or cell number), and owner.

## Safety

- Only files that git tracks are read, so ignored files are never seen.
- `.env` files, lock files, images, build output and big data files are skipped.
- A chunk that contains a key shape (Google, `sk-`, GitHub, AWS, Slack, private key blocks) is dropped and listed in the report.
- In config files and in code, the value of a setting named like a secret (password, secret, token, api key) is replaced with `<redacted>`. Placeholders such as `${VAR}` or `change-me` are left alone. The report lists which chunks were changed, never the values.
- This is a simple safety net, not a security scanner. I still read the report.

## Setup (macOS)

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then open .env and paste the Gemini key
mkdir -p data/repos && cd data/repos
git clone https://github.com/Evelynval-w/credit_risk_ML_model
git clone https://github.com/Evelynval-w/synergyhack
git clone https://github.com/Evelynval-w/timeout-airline
git clone https://github.com/Evelynval-w/GameHub
git clone https://github.com/Evelynval-w/fracton-scanner
cd ../..
```

The folder names must match the names in `repos.toml`.

## Run it

All commands run from the `backend` folder.

```
cd backend

python -m pytest -q                          # tests, no key needed
python -m app.ingest.index --dry-run --show 2   # cut the repos, print the report and samples, no key needed
python -m app.tools.list_models              # which Gemini models my key can use
python -m app.tools.list_models --try-embed MODEL_NAME   # embed one sentence to check the model works
python -m app.ingest.index                   # cut, embed with Gemini, store in data/chroma
```

Put the embedding model name from `list_models` into `GEMINI_EMBED_MODEL` in `.env`.

If I change the embedding model, the old index cannot be used. The program stops and asks for `--reset`.

## What is tested and what is not

- Tested with `pytest` and a fake embedder: the walker, all chunkers, secret handling, the Chroma store, the index command, and the Gemini client against a fake HTTP server (batching, retry on rate limit, error messages, key in header only).
- Also run on my five real repos with `--dry-run` and with fake vectors.
- **Not tested yet:** a real call to the Gemini API. The client follows the public REST documentation. The first real run is the check.

## Known limits

- A class attribute that sits after the first method is not in the class chunk. It ends up in a "top-level code" chunk.
- Some chunks are very small, for example `app.use('/auth', authRoutes);`. They are kept so every route has a reference.
- The owner tags for `timeout-airline` come from the repo README. They need a check against the real history.
- Parts of a long function are cut with no overlap.
- No tests exist yet for the chat part, because it is not built.

## Files in phase 1

| File | What it does |
|---|---|
| `repos.toml` | the repo list, excluded folders, owner rules |
| `backend/app/config.py` | reads settings from `.env` |
| `backend/app/ingest/models.py` | the `Chunk` object |
| `backend/app/ingest/walker.py` | lists files and decides what to skip |
| `backend/app/ingest/secrets.py` | key detection and redaction |
| `backend/app/ingest/chunkers/` | one chunker per file type, and the dispatcher |
| `backend/app/ingest/pipeline.py` | runs all of the above for one repo |
| `backend/app/ingest/embedder.py` | Gemini client and the fake embedder |
| `backend/app/ingest/store.py` | the Chroma wrapper |
| `backend/app/ingest/index.py` | the command that builds the index |
| `backend/app/tools/list_models.py` | lists Gemini models for my key |
| `backend/tests/` | the tests |
