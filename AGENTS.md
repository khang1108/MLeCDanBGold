# AGENTS.md

## Mission

Build a hackathon-oriented AI assistant for the Ho Chi Minh City AI Challenge
2026. Given a Vietnamese or English natural-language query, the system should
retrieve the exact matching video frame and return the official `video_id` and
`frame_idx`.

The project is currently establishing a small, testable Python foundation.
Prioritize correct frame mappings and measurable retrieval experiments before
adding production infrastructure or model-specific complexity.

## Priorities

1. Preserve exact `frame_id`, `video_id`, and `frame_idx` mappings.
2. Keep contracts and experiments reproducible.
3. Improve candidate recall and final ranking accuracy.
4. Reduce warm-query latency after the retrieval baseline is measured.
5. Keep abstractions small and useful for the hackathon workflow.

## Current scope

The current implementation contains:

- Pydantic 2 contracts in `src/hcmai/common/schemas/`.
- A lightweight `SearchEngine` orchestration skeleton in
  `src/hcmai/search.py`.
- Shared configuration scaffolding in `src/hcmai/common/config.py`.
- Generic file, image, timing, and logging helpers in
  `src/hcmai/common/utils/`.
- Contract tests in `tests/`.
- An existing Node.js frontend in `frontend/`.

Concrete embedding, FAISS, enrichment, reranking, evaluation, and FastAPI
components are planned work, not assumptions about the current codebase.
Do not add authentication, microservices, Kubernetes, distributed databases,
or generalized plugin systems unless explicitly requested.

## Repository layout

```text
frontend/                         Existing Node.js UI
src/hcmai/
├── search.py                     Search orchestration
└── common/
    ├── config.py                 Shared settings scaffolding
    ├── schemas/                  Pydantic contracts and enums
    │   └── README.md              Schema reference
    └── utils/                    Generic I/O, image, timing, and logging
        └── README.md              Utility usage guide
configs/                          Experiment configuration
data/                             Local corpus and metadata (not Git)
artifacts/                        Generated embeddings and indexes (not Git)
runs/                             Experiment outputs (not Git)
tests/                            Contract and smoke tests
```

Add directories only when their first real implementation is needed. Reuse
the existing frontend; do not replace it with Streamlit or Gradio.

## Canonical contracts

The source of truth is `src/hcmai/common/schemas/`. Use those models rather
than defining duplicate dictionaries or dataclasses.

Important models include:

- `FrameRecord` and `FrameEnrichment` for frame metadata and offline evidence.
- `RetrievalCandidate` and `SearchScores` for retrieval-stage exchange.
- `SearchRequest`, `SearchFilters`, `SearchResult`, and `SearchResponse` for
  the search boundary.
- `EvaluationQuery` for labelled offline evaluation data.
- `ConversationTurn` and `FrameFeedback` for conversational KIS workflows.

Important identifiers:

- `frame_id` is globally unique and stable across pipeline reruns.
- `video_id` identifies the source video.
- `frame_idx` is the authoritative submission frame index.
- `timestamp_ms` is used for preview and temporal search.

Never infer `frame_idx` from `timestamp_ms * fps`; variable-frame-rate videos
and decoder behavior make that mapping unsafe. Unknown schema fields are
intentionally rejected. If a contract changes, update its tests and related
documentation in the same change. Always align with existing schemas in
`src/hcmai/common/schemas/` by prioritizing extending existing schemas
(e.g., `SearchRequest`, `SearchResponse`) with optional fields over creating
new ones to prevent system bloat.

## Utility conventions

Use `src/hcmai/common/utils/` only for generic helpers:

- `io.py`: YAML, JSON, and Parquet read/write helpers.
- `image.py`: detached Pillow image loading.
- `timing.py`: monotonic millisecond timing and `Timer`.
- `logging.py`: explicit console/file logging configuration and named loggers.

Model, retrieval, and domain-specific logic belongs outside `utils/`. See
`src/hcmai/common/utils/README.md` for examples and optional dependencies.

## Architecture rules

- Keep reusable logic in `src/hcmai`, not only in notebooks or scripts.
- Scripts should parse arguments and call reusable package functions.
- Retrieval stages should exchange `RetrievalCandidate` or another approved
  schema.
- Keep model checkpoints, candidate counts, and search profiles in
  configuration rather than hard-coding them.
- Load online models and indexes once at application startup, not per request.
- Avoid factories, registries, dependency-injection layers, and base classes
  until two real implementations require them.
- Preserve the boundary between the Python package and the existing frontend.

The intended profiles are `accurate` and `fast`, using one orchestration path
with different configuration values. Do not duplicate the search pipeline for
each profile.

## Artifact conventions

When the offline pipeline is implemented, prefer these roles:

| Path | Role |
|---|---|
| `data/metadata/frames.parquet` | Canonical frame metadata |
| `artifacts/enrichment/frame_enrichment.parquet` | Caption/OCR/ASR evidence |
| `artifacts/embeddings/visual_embeddings.npy` | Visual vectors |
| `artifacts/embeddings/frame_mapping.parquet` | Vector-to-frame mapping |
| `artifacts/indexes/visual.index` | FAISS index |

Join artifacts on `frame_id`. Do not commit datasets, embeddings, model
weights, indexes, or experiment output.

## Python standards

- Follow PEP 8 and use four spaces for indentation.
- Use type hints and concise docstrings for public APIs.
- Prefer `pathlib.Path` over manual path concatenation.
- Use `from __future__ import annotations` in new modules.
- Use Pydantic 2 APIs for shared contracts.
- Keep imports ordered: standard library, third party, then local modules.
- Do not download models, allocate GPUs, or load the corpus at import time.
- Keep lines at or below 79 characters when practical.

The project metadata currently declares only the core Pydantic dependency.
When adding a runtime dependency, update `pyproject.toml` and document the
installation or usage impact.

## Evaluation and testing

Meaningful retrieval experiments should record candidate Recall@K, final
Recall@K, MRR, P50/P95 latency, per-query predictions, failure categories,
configuration, and checkpoint names under `runs/<experiment_name>/`.

Before declaring a Python change complete:

1. Compile or import the modified modules.
2. Run relevant tests.
3. Exercise a small fixture instead of the full corpus when possible.
4. Verify frame mappings remain valid.
5. Run configured formatting or lint checks.

Useful checks are:

```bash
python -m compileall src
PYTHONPATH=src pytest
```

Tests must not download large models. Use fake retrievers, rerankers, and
small in-memory frame records for orchestration tests.

## Research-first discipline

This is a hackathon research project. Every hour spent on defensive engineering
is an hour not spent on improving Recall@K or writing the paper. These rules are
binding for all team members and all AI-generated code.

### The single question before writing any code

> "Does this line help us get a better number in the paper, or does it only make
> the codebase more robust?"

If the answer is only the latter, do not write it.

### Hard size limits

| Artifact | Hard limit | Rationale |
|---|---|---|
| Any new `.py` module | ≤ 200 lines | A teammate must understand it in ≤ 10 min |
| Test file for that module | ≤ 100 lines | Smoke tests only; no production-grade coverage |
| New function | ≤ 40 lines | One responsibility, readable at a glance |
| PR / single change | ≤ 300 lines total | Keep reviews fast |

Exceeding a limit requires an explicit written reason in the PR description.
"AI generated it this way" is not a valid reason.

### Experiment before module

1. Prove an idea works in a Jupyter notebook first.
2. Extract the working code into a `.py` module only when a second experiment
   needs to reuse it.
3. Never write a module in anticipation of future use.

### Banned patterns in this codebase

The following patterns are forbidden unless Pkhanggg approves them in writing:

- Atomic temp-file writes (`write to .tmp then rename`) outside the data
  ingestion pipeline that is already complete.
- Per-file SHA-256 checksums computed at runtime.
- Factories, registries, or plugin systems.
- Abstract base classes with fewer than two concrete implementations.
- Retry loops, circuit breakers, or backoff logic.
- Docker, Kubernetes, or any container orchestration.
- Authentication or user management of any kind.
- Any new database (SQL or NoSQL). `frames.parquet` is the database.

### The "two-file rule" for new research components

Every new research component (embedder, retriever, reranker, evaluator) must fit
in exactly two files:

```text
src/hcmai/<component>.py        ← implementation, ≤ 200 lines
tests/test_<component>.py       ← smoke tests with tiny fixture, ≤ 100 lines
```

If you need a third file, you have over-engineered the component.

### Experiment output contract

Every experiment run must write a `runs/<name>/metrics.json` with at least:

```json
{
  "recall_at_1": 0.0,
  "recall_at_5": 0.0,
  "mrr": 0.0,
  "p50_latency_ms": 0.0,
  "p95_latency_ms": 0.0,
  "config": "<path to config file used>"
}
```

No `metrics.json` means the experiment did not happen.

## Change discipline

- Inspect existing files before editing.
- Preserve unrelated user changes and the existing frontend.
- Keep changes scoped to the requested component.
- Update documentation when public behavior or contracts change.
- Record assumptions when corpus formats or official mappings are unknown.
- Prefer a working baseline and measured ablation over broad unfinished work.
- If AI generated the change, the human author is still responsible for
  understanding every line before merging.
