# Agent Instructions

## Mission

- Build HCMAI 2026 retrieval for Textual KIS, TRAKE, Q&A/VQA, and the
  working 2026 extensions conversational KIS and VKIS, using Vietnamese and
  English queries.
- Return official video names and `frame_idx` values through the canonical
  frame mapping; optimize the official competition score, ranking,
  reproducibility, warm-query latency, and operator throughput.
- Build a competition system and identify at most two defensible research gaps
  for potential SoICT papers.

## Competition Contract and Evidence

- Primary public baseline:
  [https://www.codabench.org/competitions/10187/](https://www.codabench.org/competitions/10187/), "HCMC AI Challenge 2025
  (Group A)", reviewed 2026-07-28.
- Treat this 2025 contract as evidence about the prior competition, not proof
  of unchanged 2026 rules. New official 2026 rules override this section.
- The official 2025 preliminary round contains exactly three query types:
  Textual KIS, Q&A/VQA, and TRAKE. The page does not list VKIS or
  conversational KIS as preliminary submission types.
- Do not invent missing rules. Record conflicts or unknown 2026 behavior and
  ask the user before encoding it in schemas, scoring, or submission logic.

### Official 2025 Task Semantics

- **Textual KIS**: rank candidate rows `<video_name>,<frame_idx>`. A row is
  correct when the video name matches the ground-truth video and `frame_idx`
  falls inside the accepted ground-truth interval `[s, e]`.
- **Q&A/VQA**: rank `<video_name>,<frame_idx>,<answer>` rows. Credit requires
  the correct video, a frame inside `[s, e]`, and the correct answer. The answer
  is at most 100 characters and may be Vietnamese or English. The public page
  mentions both semantic comparison and exact string comparison; until the
  2026 scorer resolves this, preserve the submitted string and evaluate both
  normalized semantic accuracy and exact match in local experiments.
- **TRAKE**: retrieve and temporally align an ordered sequence of `N` key
  events. Each row is `<video_name>,<frame_1>,...,<frame_N>`; it must contain
  exactly `N` integer frame indices in chronological event order. A wrong
  video gives zero row credit. For the correct video, row credit is the
  fraction of events whose predicted frame falls in that event's accepted
  interval `[s_j, e_j]`.

### Official 2025 Ranking and Submission

- A query accepts at most 100 ranked answer rows.
- For `k in {1, 5, 20, 50, 100}`, `R@k` is the maximum task-specific row
  R-Score among the first `k` rows. Query score is the mean of those five
  `R@k` values. Ranking at all five cutoffs matters; do not optimize only
  Recall@1/5.
- The preliminary round used three query packages. Public leaderboard results
  used 50% of the answers and private ranking used 100%.
- Each package allowed at most three submissions, with the last submission
  used for ranking; a malformed upload still consumed a submission.
- Submission CSV files are UTF-8, comma-delimited, have no header, and use
  integer frame indices. Video names omit `.mp4`.
- Package all per-query CSV files under `submission/` inside one `.zip`.

### Working 2026 Extensions — User-Provided, Not Yet Official

- **VKIS**: the participant watches a target video clip, describes it to the
  system, and retrieves the corresponding video/frame. The initial description
  is a standalone retrieval query and must not require conversation resolution.
- **Conversational KIS**: follow-up turns refine a previous retrieval state.
  Resolve only genuinely context-dependent turns; first-turn, feedback-only,
  and already standalone queries should bypass a generative resolver.
- VKIS and conversational KIS remain product/research requirements until an
  official 2026 competition source confirms their scoring and submission rules.

## DOs and DON'Ts

### DOs

- Inspect the relevant code, tests, artifacts, and official competition source
  before proposing or implementing a change.
- Distinguish confirmed 2025 rules, user-provided 2026 working definitions, and
  unresolved 2026 assumptions in code, reports, and experiments.
- Ask the user when a missing rule, corpus detail, or product decision would
  materially change the implementation or evaluation.
- Preserve the canonical `frame_id` → `video_id` → `frame_idx` mapping through
  retrieval, reranking, temporal alignment, UI display, and submission export.
- Optimize and report the official Mean Top-k R-Score at
  `{1,5,20,50,100}` alongside task metrics, Recall, MRR, and P50/P95 latency.
- Use the `fast` profile for the default competition path and measure
  query-to-first-useful-results plus end-to-end operator throughput.
- Route standalone Textual KIS and initial VKIS descriptions directly to
  retrieval; invoke conversation resolution only for context-dependent turns.
- Handle feedback-only KISC turns deterministically and keep accepted/rejected
  frame state outside generative model control.
- Retrieve temporal windows for TRAKE, preserve same-video and event-order
  constraints, and jointly align the complete event sequence.
- Reuse authoritative schemas and shared `RetrievalCandidate` objects across
  data preparation, retrieval, reranking, orchestration, API, and frontend.
- Keep experiments reproducible with pinned checkpoints, configs, predictions,
  failures, official metrics, and latency measurements under `runs/`.
- Use fake models and tiny fixtures in tests; update focused tests and docs when
  public behavior or competition contracts change.
- Preserve unrelated worktree changes and keep changes within the requested
  files and task scope.
- Treat GLM-class thinking models as offline teachers or measured accurate-mode
  components unless they satisfy the competition latency and reliability budget.

### DON'Ts

- Do not guess competition rules, scorer normalization, dataset structure,
  frame sampling policy, or user intent.
- Do not infer `frame_idx` from timestamps, FPS, filenames, array positions, or
  neighboring frames.
- Do not route every query through KISC, an LLM, a VLM, or `/resolve`.
- Do not place unbounded reasoning, model loading, network calls, or synchronous
  generation on the default critical path without a timeout and deterministic
  fallback.
- Do not treat TRAKE events as unrelated static-image queries or combine event
  frames from different predicted videos in one row.
- Do not allow a reranker, resolver, or VQA provider to rewrite candidate/frame
  identity.
- Do not duplicate authoritative schemas, invent frontend-only fields, or add
  parallel contracts for the same task.
- Do not rewrite stable mapping, embedding, or FAISS artifact foundations
  without an evidence-backed compatibility or performance reason.
- Do not call a component competition-ready solely because its schema, mock
  test, or endpoint exists; require an end-to-end path and recorded metrics.
- Do not optimize only Recall@1/5 or report a retrieval improvement without the
  official score and latency trade-off.
- Do not commit datasets, videos, frames, model weights, embeddings, indexes,
  run outputs, credentials, Cloudflare tokens, or private deployment scripts.
- Do not add auth, databases, microservices, containers, retries, circuit
  breakers, generalized plugin systems, premature factories/DI, or base classes
  without explicit approval and demonstrated need.
- Do not download checkpoints, load the real corpus, or depend on live remote
  services in unit tests.
- Do not expand a research prototype into production code before it has a
  notebook result and a second demonstrated use.

## Package Managers

- Python: use the repository virtual environment at `aic/`; install with
  `aic/bin/python -m pip install -e ".[embedding,dev]"`.
- Frontend: use `npm` in `frontend/`; preserve the existing React application.
- Add supported runtime dependencies to `pyproject.toml`.

## Folder and File Aware

- `src/hcmai/agents/`: KISC and future bounded task orchestration.
- `src/hcmai/data/`: dataset loading, preparation, and canonical frame mapping.
- `src/hcmai/common/schemas/`: authoritative API and task contracts.
- `src/hcmai/embedding/`: embedding modules and services.
- `src/hcmai/retriever/`: retrieval modules, indexes, and benchmarks.
- `src/hcmai/common/utils/`: cross-cutting helpers, not domain logic.

## Subagent Roles

These are responsibility profiles for parallel work, not a requirement to
spawn agents for every task. The lead agent dispatches them only when
multi-agent work is explicitly requested.

### nhuy — Senior SWE, API and Integration

- Owns contracts crossing frontend, FastAPI, and the AI search pipeline.
- Primary paths: `frontend/`, `src/hcmai/app.py`,
  `src/hcmai/common/schemas/`, and API integration tests.
- Checks endpoint request/response compatibility, startup configuration,
  error handling, submission CSV/ZIP compatibility, and complete
  UI → API → search → UI flows.
- Must not invent frontend-only fields or duplicate authoritative schemas.

### khầy — Senior AI Engineer, Conversation and Reranking

- Owns KIS-conversation orchestration, turn state, query reformulation,
  feedback behavior, clarification logic, reranking, and TRAKE joint temporal
  alignment over retrieval candidates.
- Primary paths: `src/hcmai/kisc.py`, `src/hcmai/search.py`, future
  `src/hcmai/agents/`, temporal alignment, or reranker code, and their focused
  tests.
- Consumes shared `RetrievalCandidate` objects and preserves their exact frame
  identifiers when reranking.
- Keeps orchestration research-friendly; no database or generalized agent
  framework unless explicitly approved.

### [fuvo — Senior AI Engineer, Retrieval and Enrichment]()

- Owns metadata preparation, embeddings, FAISS indexing, dense retrieval,
  OCR/ASR/caption/object/action enrichment, temporal windows, per-event TRAKE
  candidates, fusion inputs, and retrieval benchmarks.
- Primary paths: `src/hcmai/data/`, `src/hcmai/embedding/`,
  `src/hcmai/retriever/`, `scripts/`, `notebooks/`, and related tests.
- Produces shared `RetrievalCandidate` outputs for khầy and stable API
  materialization for nhuy.
- Every retrieval experiment must record the configured checkpoint and metrics
  under `runs/`; no real corpus or checkpoint loading in tests.

### Coordination

- Fuvo produces frame/event candidates, khầy may reorder, refine, or jointly
  align them, and nhuy exposes and exports the final shared response.
- Each subagent reports files inspected or changed, tests run, and unresolved
  gaps. For audits, use evidence-backed status tables and do not guess.
- Agents share the worktree: preserve unrelated edits and never revert another
  agent's changes.

## Key Conventions

- `src/hcmai/common/schemas/` is authoritative. Extend existing contracts with
  optional fields before adding models; keep unknown fields rejected.
- Preserve exact `frame_id` → `video_id` → `frame_idx` mappings everywhere.
  Never infer `frame_idx` from timestamps or FPS.
- Competition output uses official video names and integer `frame_idx` values;
  internal `frame_id` values must always materialize through the canonical
  mapping before display or submission.
- TRAKE rows must keep all `N` event frames on one predicted video, preserve
  event order, and contain exactly one mapped frame index per event. Retrieve
  temporal windows and score state transitions; do not treat boundary events
  as independent static-image queries.
- Route standalone Textual KIS and initial VKIS descriptions directly to
  retrieval. Route only context-dependent conversational refinements through a
  resolver. Do not route TRAKE through the KISC agent.
- Treat query-to-first-useful-results and operator throughput as competition
  KPIs. The `fast` profile is the default competition path. Any generative or
  remote stage on that path needs measured warm P50/P95 latency, a bounded
  timeout, and a deterministic fallback before it can be enabled by default.
- Resolve canonical relative `image_path` values against the dataset root
  without rewriting `frames.parquet`.
- Artifact flow is `frames.parquet` → normalized `.npy` + mapping Parquet →
  exact FAISS `visual.index`; YAML/JSON stores provenance, not vectors.
- Join artifacts on `frame_id`; exchange retrieval data via shared schemas.
- Keep checkpoints, candidate counts, and `fast`/`accurate` profiles in config;
  use one pipeline with different values.
- Load models/indexes once, never at import or per request. `DenseEncoder`
  lazily loads on its first non-empty encode call.
- Keep reusable logic in `src/hcmai`, CLIs thin, domain logic out of
  `common/utils`, and the Python/frontend boundary intact.

## Research Rules

- Prove retrieval ideas in a notebook; extract code only for a second use.
- Limits: new module ≤200 lines, smoke test ≤100, function ≤40, change ≤300.
- A new research component uses exactly implementation + smoke-test files
  unless explicitly approved.
- Do not add auth, databases, microservices, containers, retries, circuit
  breakers, runtime checksums, generalized plugins, premature factories/DI,
  or a base class without two implementations.
- Atomic temp-file writes are allowed only in the completed ingestion pipeline.
- Tests use fake models and tiny fixtures; never download checkpoints or load
  the corpus.
- Runs record the official mean Top-k R-Score at `{1,5,20,50,100}`, task row
  R-Scores, Recall@1/5, MRR, P50/P95 latency, predictions, failures, config,
  and checkpoint under `runs/`. TRAKE runs also record video accuracy,
  per-event interval accuracy, and full-sequence accuracy; Q&A runs record
  exact match and the explicitly configured normalization metric. No
  `metrics.json` means no experiment.
- Never commit datasets, weights, embeddings, indexes, or run outputs.
- MUST NOT GUESS the information or intent. If you don't know anything, MUST QA with me to clarify

## File-Scoped Commands

| Task      | Command                                                     |
| --------- | ----------------------------------------------------------- |
| Compile   | `aic/bin/python -m py_compile src/hcmai/<file>.py`        |
| Test      | `PYTHONPATH=src aic/bin/pytest tests/test_<component>.py` |
| Typecheck | `pyright src/hcmai/<file>.py`                             |

## Change and Commit Discipline

- Inspect first and preserve unrelated work. Update tests/docs with public
  behavior or contract changes; record unknown corpus-format assumptions.
- AI commits include `Co-Authored-By: <model name> <noreply@openai.com>`.

## Infrastructure

- We use ThunderCompute with **L40 GPU** or **A6000** **GPU**
