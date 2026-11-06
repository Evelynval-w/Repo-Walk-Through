# codecite

A chatbot that answers questions about my own GitHub repos and points to the file and function the answer comes from.

**Status: phase 3 of 8 done (evaluation).** The project reads five repos, cuts the code into pieces, stores them in a local database (phase 1), answers questions with the file and function each answer comes from (phase 2), and is now measured on 37 hand-written questions (phase 3). It has no UI yet and no real way to say "I don't know" (the model already says it in prose, but nothing checks it). Search and answer quality are the next work. The plan for all phases is in [docs/PLAN.md](docs/PLAN.md).

## The five repos

| Repo | Language | What I index |
|---|---|---|
| credit_risk_ML_model | Python (notebooks) | one chunk per code cell, with the markdown above it. Saved outputs are dropped |
| synergyhack | JavaScript, JSX | functions, classes, Express routes, README and docs |
| timeout-airline | Java | classes and methods with their Javadoc. Some files are tagged with my teammate as owner |
| GameHub | Python, TypeScript | the services, frontend, infra and docs. The course exercises in `modules/` are skipped. Everything except my `user-service` is tagged `course template`, so answers say who wrote it |
| fracton-scanner | Python | the code, README, workflows and Dockerfile. The `data/` folder is never read |

GameHub is a fork of a course template. Git history shows 39 commits by the course author and 3 by me, all in `services/user-service`. Only that folder is tagged as mine. Every other file is tagged `course template`, the prompt tells the model to say so, and each reference in the answer carries an `owner` field.

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

## Ask a question (phase 2)

Settings for this part go in `.env`. `GEMINI_CHAT_MODEL` defaults to `gemini-2.5-flash`. `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` are optional and turn on the fallback.

From the terminal, in `backend`:

```
python -m app.qa.ask "Where is the discount code created?"
python -m app.qa.ask "How is the model evaluated?" --repo credit_risk_ML_model --top-k 8 --show-chunks
```

As a web API, in `backend`:

```
python -m uvicorn app.main:app --port 8000
```

Use `python -m uvicorn`, not plain `uvicorn`. Plain `uvicorn` can pick up a copy installed outside the virtual environment, which then cannot find `chromadb`.

Then, in a second terminal:

```
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Where is the discount code created?", "top_k": 6}'
```

To pick a fallback model: `python -m app.tools.list_openrouter_models`. Models marked with `*` say they can return JSON, which the fallback needs.

### How an answer is made

1. The question is turned into a vector and the `top_k` closest chunks are fetched (6 by default).
2. The chunks are numbered and sent to Gemini with the rule "answer only from these excerpts" and a request for JSON: `{"answer": "...", "used": [1, 3]}`.
3. Pydantic checks the JSON. If it is not JSON, has the wrong shape, or points at an excerpt number that does not exist, the answer is rejected with a 502 and nothing is shown.
4. The references are built from the metadata of the excerpts the model pointed at (repo, file, function, lines or notebook cell, plus the chunk text). The model never writes a file name, so it cannot invent one.
5. If Gemini is rate limited or down (429, 5xx, no network) after its own retries, the same prompt goes to OpenRouter. A 400 does not trigger the fallback, because a wrong request would fail there too.

## Measure it (phase 3)

`eval/questions.toml` holds 37 questions: 31 whose answer is in the code and 6 whose answer is not. Each answerable one lists the files that hold the answer and the facts a correct answer must contain.

```
python -m app.evaluation.run --runs 1 --only cr-01 sh-01   # quick try
python -m app.evaluation.run --runs 3                      # the numbers to report
```

Four separate numbers: **retrieval hit** (an expected file was among the chunks shown), **evidence shown** (a short exact phrase from the code that holds the answer was in the text of the chunks shown, so it catches "right file, wrong part of it"), **citation hit** (the model cited an expected file), **answer correct** (a second model call finds every required fact in the answer). Reports go to `eval/reports/` as `.md` and `.json`. The report ends with 10 answers to check by hand against the judge.

`python -m app.evaluation.show REPORT.json` prints every miss with its cause. `python -m app.evaluation.rejudge REPORT.json` judges saved answers again after the gold facts change.

Limits: the same chat model judges its own answers, so I read the hand-check list before trusting the number. The 6 unanswerable questions are not scored yet; the report only shows whether the answer refused. That is the starting point for phase 5.

### Results (phase 3 reference run)

Setup: `gemini-3.5-flash` answers and judges, `gemini-embedding-001` for search, top-k 6, 3 runs per question, 31 answerable questions (93 runs) and 6 unanswerable ones.

| What is measured | Result |
|---|---|
| An expected file is among the chunks shown | 86/93 (92%) |
| The chunk with the key phrase is among the chunks shown | 69/93 (74%) |
| The model cites an expected file | 80/93 (86%) |
| Every required fact is in the answer | 66/93 (70%) |
| Required facts found, counted one by one | 184/229 (80%) |
| Unanswerable questions where the answer says it is not in the code | 18/18 |

By type of question: "why" and "numbers" questions are 100% correct. "List" questions are the weakest (3/12 correct), because the items sit in different chunks.

How the correct-answer number moved, on the same questions: 45/93 (48%) in the first run, 56/93 (60%) after I fixed gold facts that were wrong or too vague (same saved answers, only the judging changed), 66/93 (70%) after changing the prompt from "keep the answer short" to "include every condition, number and name". So about 12 points came from fixing my own ruler and about 10 from the prompt.

One extra run with top-k 12 (1 run, so noisy): chunk with the key phrase shown 83%, citation 96%, correct still 70%. More chunks help search and do not help the answer, and one unanswerable question got an answer built from a different repo. The default stays at 6.

What is left, in my reading: search misses on questions that do not name the repo (sh-01) and on list questions (ta-06), and answer misses where the right chunks were shown (gh-05: the model described a docstring as working code, but the function only raises `NotImplementedError`).

How far to trust these numbers:
- 31 questions is small. A change of one or two questions is noise.
- I wrote the questions and gold facts myself, and I fixed some of them once after seeing the first results. The first-run number is not comparable to the later ones for that reason.
- The same model that writes the answers also judges them. I read ten judged answers by hand each run.
- "Chunk with the key phrase shown" is strict. For fs-04 and fs-05 the answers were right without it.
- Two runs in 124 failed because the model returned JSON that was cut off. They count as misses. The error now says the finish reason, so the next one shows the cause.

## What is tested and what is not

- Tested with `pytest` and fakes: the walker, all chunkers, secret handling, the Chroma store, the index command, the Gemini embedding client against a fake HTTP server, the chat clients (Gemini and OpenRouter) against a fake HTTP server, the fallback switch, the answer checks, and the `/ask` and `/health` endpoints.
- I broke the code on purpose in two places (skipping the "excerpt does not exist" check, and letting the fallback switch on a 400) and the tests failed each time.
- Also run on my five real repos with `--dry-run` and with fake vectors, and indexed for real with Gemini embeddings (1,321 chunks).
- **Not tested yet:** a real Gemini chat call and a real OpenRouter call. Both clients follow the public REST documentation. The first real questions are the check.

## Known limits

- A class attribute that sits after the first method is not in the class chunk. It ends up in a "top-level code" chunk.
- Some chunks are very small, for example `app.use('/auth', authRoutes);`. They are kept so every route has a reference.
- The owner tags for `timeout-airline` come from the repo README. Git history only shows my teammate committing the three model files (Plane, Airport, Flight), so only those three are tagged as hers.
- Parts of a long function are cut with no overlap.
- The chatbot tries to answer every question. It cannot say "this is not in the code" yet. That is phase 5.
- A reference proves the chunk was shown to the model. It does not prove the chunk supports the claim. The model can point at the wrong excerpt.
- Retrieval is vector search only. Exact names (a function called `rewardMiles`) can be missed. Hybrid search is phase 4.
- If the model returns broken JSON, the request fails with a 502. There is no second try yet.
- Free OpenRouter models may ignore the JSON request or be slow. The fallback has not been run for real.
- The numbers above come from one model, one judge and 37 questions. See how far to trust them.

## Files in phase 2

| File | What it does |
|---|---|
| `backend/app/http_retry.py` | sends a POST and retries on 429 and 5xx. Used by every Gemini and OpenRouter client |
| `backend/app/qa/models.py` | the Pydantic shapes: request, reference, response, and what the model must return |
| `backend/app/qa/chat.py` | Gemini chat, OpenRouter chat, the fallback switch, and a fake chat for tests |
| `backend/app/qa/retrieve.py` | embeds the question and fetches the closest chunks |
| `backend/app/qa/prompt.py` | the instructions and the numbered excerpts sent to the model |
| `backend/app/qa/answer.py` | runs the steps, checks the reply, builds the references |
| `backend/app/qa/services.py` | builds the real embedder, store and chat from `.env` |
| `backend/app/qa/ask.py` | ask one question from the terminal |
| `backend/app/api.py` | the FastAPI app: `POST /ask`, `GET /health` |
| `backend/app/main.py` | starts the real server |
| `backend/app/tools/list_openrouter_models.py` | lists OpenRouter models for the fallback |

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
