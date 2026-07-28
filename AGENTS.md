# Agent Instructions

## Mission

- Build HCMAI 2026 frame retrieval for KIS, conversational KIS, VQA, and ad-hoc
  Vietnamese/English queries.
- Return exact official `video_id` and `frame_idx`; optimize Recall@K, ranking,
  reproducibility, and warm-query latency with small research-friendly code.
- Beside develop an AI assistant to win the challenge, we also need to find research gaps in this domain to publis papers (max 2) at SoICT conference.

## Package Managers

- Python: use the repository virtual environment at `aic/`; install with
  `aic/bin/python -m pip install -e ".[embedding,dev]"`.
- Frontend: use `npm` in `frontend/`; preserve the existing React application.
- Add supported runtime dependencies to `pyproject.toml`.

## Folder and File Aware

- `agents/`: to find source code of KISC agent
- `data/`: to find functions/classes used to load/modify dataset
- `common/`: to find API schemas, or base models/classes used in the project
- `embeddings/`: to find embedding module/services used in the project
- `retriever/`: to find retrieve module/services used in the project
- `hcmai.common.utils/`: to find helpers functions used in the projects

## Subagent Roles

These are responsibility profiles for parallel work, not a requirement to
spawn agents for every task. The lead agent dispatches them only when
multi-agent work is explicitly requested.

### nhuy — Senior SWE, API and Integration

- Owns contracts crossing frontend, FastAPI, and the AI search pipeline.
- Primary paths: `frontend/`, `src/hcmai/app.py`,
  `src/hcmai/common/schemas/`, and API integration tests.
- Checks endpoint request/response compatibility, startup configuration,
  error handling, and complete UI → API → search → UI flows.
- Must not invent frontend-only fields or duplicate authoritative schemas.

### khầy — Senior AI Engineer, Conversation and Reranking

- Owns KIS-conversation orchestration, turn state, query reformulation,
  feedback behavior, clarification logic, and reranking.
- Primary paths: `src/hcmai/kisc.py`, `src/hcmai/search.py`, future
  `src/hcmai/agents/` or reranker code, and their focused tests.
- Consumes shared `RetrievalCandidate` objects and preserves their exact frame
  identifiers when reranking.
- Keeps orchestration research-friendly; no database or generalized agent
  framework unless explicitly approved.

### fuvo — Senior AI Engineer, Retrieval and Enrichment

- Owns metadata preparation, embeddings, FAISS indexing, dense retrieval,
  OCR/ASR/caption/object enrichment, fusion inputs, and retrieval benchmarks.
- Primary paths: `src/hcmai/data/`, `src/hcmai/embedding/`,
  `src/hcmai/retriever/`, `scripts/`, `notebooks/`, and related tests.
- Produces shared `RetrievalCandidate` outputs for khầy and stable API
  materialization for nhuy.
- Every retrieval experiment must record the configured checkpoint and metrics
  under `runs/`; no real corpus or checkpoint loading in tests.

### Coordination

- Fuvo produces candidates, khầy may reorder or refine them, and nhuy exposes
  the final shared response to the frontend.
- Each subagent reports files inspected or changed, tests run, and unresolved
  gaps. For audits, use evidence-backed status tables and do not guess.
- Agents share the worktree: preserve unrelated edits and never revert another
  agent's changes.

## Key Conventions

- `src/hcmai/common/schemas/` is authoritative. Extend existing contracts with
  optional fields before adding models; keep unknown fields rejected.
- Preserve exact `frame_id` → `video_id` → `frame_idx` mappings everywhere.
  Never infer `frame_idx` from timestamps or FPS.
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
- Runs record Recall@1/5, MRR, P50/P95 latency, predictions, failures, config,
  and checkpoint under `runs/`. No `metrics.json` means no experiment.
- Never commit datasets, weights, embeddings, indexes, or run outputs.

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
