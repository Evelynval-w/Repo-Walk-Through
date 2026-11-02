# Plan: code-aware documentation chatbot

A chatbot that answers questions about my own GitHub repos. Every answer points to the file and the function it came from. If the answer is not in the code, the bot says so.

Status: plan only. Nothing is built yet. A phase is "done" only when I have run it and can explain every file in it.

## Repos to index

| Repo | Language | Notes |
|---|---|---|
| credit_risk_ML_model | Python in notebooks | no .py files, outputs removed before indexing |
| synergyhack | JavaScript, JSX | Express server, React client |
| timeout-airline | Java | team project, teammate files get an `owner` tag |
| GameHub | Python, TypeScript | fork of a course template, `modules/` skipped |
| fracton-scanner | Python | one file (`streamer_finder_v3.py`, about 1,340 lines, about 30 functions), README, 2 GitHub workflows, Dockerfile |

triage-base is not part of the 5.

Skipped everywhere: `node_modules`, lockfiles, images, `.joblib`, `.class`, PDFs, generated Javadoc HTML, `bin/`, `.env` files, anything that looks like a key.

Skipped in fracton-scanner: the whole `data/` folder (CSV, Excel, SQLite database, generated notes). It holds scan results about real streamers, and none of it should be embedded or sent to a model.

## Stack

- Backend: Python 3.11, FastAPI, Pydantic
- Model: Gemini through a Google AI Studio key, OpenRouter as fallback
- Vector store: Chroma (stored on disk)
- Keyword search: BM25 (`rank_bm25`)
- UI: Next.js with TypeScript
- Tests: pytest with a fake model, so no test needs a key or a network
- Keys live only in `.env`, which is git-ignored. `.env.example` has empty names.

## Folder layout (target)

```
backend/app/
  config.py            settings read from .env
  ingest/              walker, chunkers, embeddings, store
  retrieval/           vector search, BM25, fusion, reranker
  llm/                 Gemini client, OpenRouter client, fake client
  api/                 FastAPI routes and Pydantic models
backend/tests/
eval/                  questions.json, runner, results/
ui/                    Next.js app
mcp_server/            MCP server exposing the search
data/repos/            cloned repos (git-ignored)
data/chroma/           vector store (git-ignored)
```

## Rules for every phase

- Stop after the phase. I run it and show the output. Nothing is called "working" before that.
- Git is done by me, under my own name. No tool names or AI markers in commit messages or files.
- Comments and README text are plain and simple English.
- After each phase: 3 questions about the code before moving on.
- Only numbers from my own runs go on my CV.

---

## Phase 1: Repo setup and indexing

**Goal:** turn the repos into chunks, embed them, store them in Chroma.

**Builds**
- `config.py`, `.env.example`, `.gitignore`
- A walker that lists files and applies the skip list
- Chunkers:
  - Python: the built-in `ast` module, one chunk per function and class
  - Java, JS, JSX, TS, TSX: tree-sitter, same idea
  - Notebooks: one chunk per code cell with the markdown cell above it, outputs dropped
  - Markdown: split by heading
  - YAML, Dockerfile, config: whole file
- Each chunk keeps: repo, file path, function or class name, start and end line, language, owner
- Embedding client (Gemini) with batching and retry, plus a fake embedder for tests
- A script that lists which models my key can use
- `index.py` command that builds the store

**I run:** the index command, then show the chunk count per repo and 5 sample chunks.

**Test without a key:** chunkers on small sample files, the skip list, the fake embedder.

**Decisions**
- AST chunks over fixed-size chunks. Why: a function stays whole and the reference can name it. Other way: split by lines with overlap. Simpler, but it cuts functions in half.
- Python `ast` for Python, tree-sitter for the rest. Why: `ast` is built in and easy to explain. Other way: tree-sitter for everything, one code path.
- Drop notebook outputs. Why: most of it is base64 images. Other way: keep short text outputs.

---

## Phase 2: Naive question answering

**Goal:** question in, answer out, with references.

**Builds**
- Top-k vector search
- Prompt that tells the model to answer only from the given chunks
- Gemini client and OpenRouter fallback behind one interface
- Pydantic response: `answer`, `references` (repo, file, function, lines), `chunks_used`
- `POST /ask` in FastAPI

**I run:** 5 real questions through the API and show the answers.

**Test without a key:** fake model returns a fixed JSON. Tests check validation, the fallback switch and bad model output.

**As built**
- The model sends `{"answer", "used": [excerpt numbers]}`. Our code turns the numbers into references. A number that does not exist rejects the whole answer.
- `references` also carry the chunk text, so the UI in phase 6 can show it on click. `chunks_used` lists every chunk the model saw.
- The fallback switches only on 429, 5xx and network errors (after Gemini's own retries). A 400 or 401 is shown as it is.
- One shared retry helper (`http_retry.py`) serves the embedding client and both chat clients.
- No refusal yet. The prompt says "only these excerpts", but there is no "not answerable" field. Phase 5 adds it, after phase 3 measures how often the bot answers when it should not.
- Chat model default is `gemini-2.5-flash`. Temperature is 0.2, so repeated questions give close but not identical answers. Phase 3 should run each question more than once or fix the temperature to 0.

**Decisions**
- Ask the model for structured JSON and validate it with Pydantic. Why: a broken answer is caught, not shown. Other way: free text, then regex for references.
- References come from chunk metadata, not from what the model writes. Why: the model cannot invent a file name. Other way: let the model cite and check afterwards.

---

## Phase 3: Evaluation set and baseline

**Goal:** a fixed set of questions so every change can be measured.

**Builds**
- `eval/questions.json`: about 30 questions across the repos. Each has the file (and function when it applies) where the answer lives. About 6 of the 30 have no answer in the code, for Phase 5.
- Runner that reports retrieval hit rate (file level and function level) at k = 1, 3, 5
- Answer accuracy: a judge prompt plus a hand check by me on a sample
- Baseline saved in `eval/results/baseline.json` and never overwritten

**I run:** the runner and show the table.

**Decisions**
- I write the questions myself from the code. Why: I need to know the answers. Other way: have the model generate them, which is faster but they are easier and may be wrong.
- Accuracy uses a model judge plus my hand check. Why: a judge alone can be too kind. Other way: exact keyword match, which is strict but misses correct answers in other words.

**Honest limit:** 30 questions is a small set. The README will say a change of one or two questions is noise.

---

## Phase 4: Better retrieval

**Goal:** hybrid search and reranking, compared on the same set.

**Builds**
- BM25 index over the same chunks
- Fusion of vector and BM25 ranks with reciprocal rank fusion
- A reranker over the top candidates
- Runner prints baseline, hybrid and hybrid plus rerank side by side

**I run:** the comparison table.

**Decisions**
- Reciprocal rank fusion. Why: no score scaling needed. Other way: weighted sum of normalized scores.
- Reranker: a local cross-encoder or a Gemini call. Chosen after trying both on the eval set. Local costs no quota but is slower to load. Gemini adds cost and latency per question.

The result may show that reranking does not help on this data. If so, the README says so.

---

## Phase 5: Refusal, logging, cost and latency

**Goal:** no silent wrong answers, and every request is measurable.

**Builds**
- Refusal: if the best retrieval score is under a threshold, or the model returns `not_found`, the API answers "not in the indexed code" with no references
- Threshold chosen on the eval set, including the unanswerable questions
- A log line per request (JSONL): question, chunks, latency per stage, tokens, cost, model used, refused or not
- Cost computed from token counts in the model response

**I run:** one answerable and one unanswerable question, then the log file.

**Extension (chosen: yes):** expose the same numbers as Prometheus metrics and add OpenTelemetry traces. This matches the observability line in the NVIDIA posting.

---

## Phase 6: TypeScript UI (meme-worthy, not the usual AI look)

**Goal:** a page I can demo that does not look like every other AI chat app.

**Must-have features (these never get hidden by the fun layer)**
- Question box and answer view
- Clickable references that open the chunk with file path and line numbers
- Controls for top-k and chunk size

**Look and feel**
- No default chat-bubble layout, no purple gradient, no sparkle icons.
- Original 3D characters built in Three.js (through react-three-fiber) from simple shapes in a soft clay style. They are my own designs. No existing meme, mascot or cartoon character is copied.
- The character reacts to the app state:
  - idle: waiting for a question
  - searching: digging through a pile of file cards while the request runs
  - found: holds up the chunk the answer came from
  - refused: a shrug and a joke when the answer is not in the code (this is the best meme moment, and it also shows that the refusal works)
  - fallback: a small change when Gemini fails and OpenRouter answers instead
- A "serious mode" switch turns the jokes and most of the motion down, for a recruiter who just wants to read.

**How the skills are used**
- `design-taste-frontend`: deals three art directions. I pick one that fits the subject (code, search, files), then it checks the page in a real browser and screenshots.
- `animate` and the animation review skills: decide what moves and what does not. The question box and typing never animate, since they are used all the time. Motion goes to rare moments: searching, found, refused.
- The `motion-*` skills (clay 3D, comic book, neo-brutalism and others) make video clips from canvas templates. They are not live Three.js. They can make the README demo clip and the portfolio loop. The live 3D characters are written in Three.js code.

**Rules for the 3D part**
- Respect `prefers-reduced-motion`.
- Load the 3D canvas lazily and fall back to a flat picture when WebGL is not available.
- Keep the page fast on a phone. Measure it, do not guess.

**Note on the controls:** top-k applies at query time. Chunk size applies at index time, so changing it means re-indexing, which uses embedding quota. The UI will either trigger a re-index with a warning or pick from a few pre-built indexes. To be decided in this phase.

**I run:** the UI against the real API, take screenshots on desktop and phone width, and check the load time.

---

## Phase 7: MCP server

**Goal:** the same search available to MCP clients.

**Builds**
- MCP server with a `search_code` tool (query, top-k) returning chunks with references
- Optional second tool that returns one chunk by id
- Tests with the fake store

**I run:** connect it to an MCP client and show one real call.

**Optional extension:** make the answer step call `search_code` as a model tool, so the model decides when to search. This shows tool calling. Costs one more model call per question.

---

## Phase 8: Docker, CI and deploy

**Goal:** anyone can run it, tests run on every push, a public demo exists.

**Builds**
- Dockerfiles for backend and UI, `docker-compose.yml`
- GitHub Actions: lint, pytest with the fake model, and the eval run on a fake embedder
- The real-model eval is run by me locally and its result is committed. CI does not hold my key for that.
- Free-tier deploy. The platform is chosen at this point, after checking current free limits, because they change.

**Open question for this phase:** ship a pre-built Chroma index in the image, or build at start-up. Pre-built is faster to start and uses no quota at boot.

---

## What this project shows, and what it does not

Shows: RAG, hybrid search, evaluation, structured output, MCP, Docker, CI/CD, a cloud deploy, logging of cost and latency, and a custom Three.js front end.

Does not show: Kubernetes and Go. I am a beginner there and this project does not change that. Observability with OpenTelemetry, Prometheus and Grafana only appears if I choose the optional extension in Phase 5.
