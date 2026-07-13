# AGENTS.md

## Mission

Build a hackathon-oriented AI assistant for the Ho Chi Minh City AI Challenge
2026. Given a Vietnamese or English natural-language query, retrieve the exact
matching image frame from an approximately 80–100 GB video corpus and return
the official `video_id` and `frame_idx`.

The system must support rapid research, evaluation, and model replacement.
Accuracy is the first priority for the preliminary round; low retrieval
latency becomes equally important for the live final round.

## Priority order

When trade-offs are necessary, use this order:

1. Preserve correct frame identifiers and source mappings.
2. Keep experiments measurable and reproducible.
3. Improve candidate recall and final ranking accuracy.
4. Reduce warm-query latency for the final round.
5. Improve code elegance only when it accelerates the work above.

## Scope

The current MVP is text-to-frame retrieval using:

- Multilingual image-text embeddings.
- FAISS candidate retrieval.
- Offline caption grounding.
- Optional OCR and ASR evidence.
- Multimodal candidate reranking.
- A FastAPI backend and an existing Node.js frontend.

Do not expand the MVP into a production platform. Authentication,
microservices, distributed databases, Kubernetes, generalized plugin systems,
and enterprise abstractions are out of scope unless explicitly requested.

## Repository layout

```text
frontend/                   Existing Node.js UI
backend/                    FastAPI entry point
src/aic/schemas.py          Shared Pydantic contracts
src/aic/search.py           Search orchestration
src/aic/data/               Extraction and frame metadata
src/aic/retriever/          Embeddings, FAISS, and score fusion
src/aic/enrichment/         Captioning, OCR, and ASR
src/aic/reranking/          Multimodal rerankers
src/aic/evaluation/         Metrics and evaluation runner
src/aic/utils/              Small generic helpers only
scripts/                    Thin command-line entry points
configs/                    Experiment and search configuration
data/                       Local corpus and metadata
artifacts/                  Generated embeddings and indexes
runs/                       Experiment results
tests/                      Contract, unit, and smoke tests
```

This is a target structure. Do not create unused directories or placeholder
modules. Keep each folder small and add files only when an implementation
requires them.

## Current state

- `src/aic/schemas.py` is implemented with Pydantic 2 contracts.
- `src/aic/__init__.py` exists.
- Other components may not exist yet; inspect the repository before editing.
- The user has an existing Node.js UI intended to live in `frontend/`.
  Preserve it and do not replace it with Streamlit or Gradio.

## Ownership boundaries

| Area | Primary owner |
|---|---|
| Contracts, orchestration, evaluation | AI Tech Lead |
| Frame extraction and metadata | Data Engineer |
| Dense retrieval and FAISS | AI Engineer 1 |
| Enrichment and reranking | AI Engineer 2 |
| FastAPI and Node.js UI integration | Software Engineer |

Avoid editing another owner's active component unless the task explicitly
requires it. Shared-contract changes require Tech Lead approval.

## Canonical contracts

`src/aic/schemas.py` is the source of truth for Python and API data shapes.
Reuse its models instead of redefining dictionaries or duplicate dataclasses.

Important identifiers:

- `frame_id`: globally unique and stable across pipeline reruns.
- `video_id`: identifier of the source video.
- `frame_idx`: authoritative frame index used for submission.
- `timestamp_ms`: presentation timestamp used for preview and temporal search.

Never infer `frame_idx` as `timestamp * fps`. Variable-frame-rate videos and
decoder behavior can make that mapping incorrect.

Unknown schema fields are intentionally rejected. If a new field is required,
update the canonical schema, its tests, API examples, and affected artifact
documentation together.

## Offline artifact contracts

Use these default artifact roles unless the current code or task specifies an
approved replacement:

| Path | Producer | Consumers |
|---|---|---|
| `data/metadata/frames.parquet` | Data pipeline | Retrieval and enrichment |
| `artifacts/enrichment/frame_enrichment.parquet` | Enrichment | Retrieval |
| `artifacts/embeddings/visual_embeddings.npy` | Encoder | Index builder |
| `artifacts/embeddings/frame_mapping.parquet` | Encoder | Vector search |
| `artifacts/indexes/visual.index` | Index builder | Online search |

Use `frame_id` as the join key. Images remain JPEG/WebP files, vectors remain
NumPy arrays, and FAISS indexes remain FAISS artifacts. Do not store large
binary data inside Git or force every artifact into Parquet.

## Python standards

All Python code must follow PEP 8.

- Use four spaces for indentation.
- Keep code lines at or below 79 characters when practical.
- Use descriptive `snake_case` names for functions and variables.
- Use `PascalCase` for classes and `UPPER_CASE` for constants.
- Add type hints to public functions, methods, and return values.
- Add concise docstrings to public modules, classes, and non-obvious methods.
- Prefer `pathlib.Path` over manual path concatenation.
- Use `from __future__ import annotations` in new Python modules.
- Use Pydantic 2 APIs for shared schemas.
- Avoid mutable default arguments; use `Field(default_factory=...)` or `None`.
- Do not perform model downloads, GPU allocation, or corpus loading at import
  time.

Keep imports ordered as standard library, third-party packages, then local
project modules. Use a formatter/linter when the project adds one, but do not
introduce a large tooling stack solely for a small change.

## Architecture rules

- `frontend/` communicates with `backend/` through the documented HTTP API.
- `backend/` may import `aic`; `aic` must not import `backend` or `frontend`.
- Scripts should parse arguments and call reusable functions from `aic`.
- Notebooks may import `aic`, but reusable logic must not live only in a
  notebook.
- Retrieval stages exchange `RetrievalCandidate` or another approved schema.
- Model checkpoints and candidate counts belong in configuration, not code.
- Load online models and indexes once at application startup, not per request.
- Keep `utils/` limited to generic I/O, image, and timing helpers. Model or
  retrieval logic belongs in its domain folder.

Avoid base classes, factories, registries, and dependency-injection layers
until at least two real implementations demonstrate the need.

## Search profiles

Use configuration-driven profiles:

- `accurate`: larger candidate pool and deeper reranking.
- `fast`: smaller candidate pool and latency-focused reranking.

Do not duplicate the search pipeline for each profile. Both profiles must use
the same orchestration with different configuration values.

## Evaluation requirements

Every meaningful retrieval experiment should record:

- Candidate Recall@K before reranking.
- Final Recall@K after reranking.
- MRR.
- P50 and P95 latency.
- Per-query predictions and failure categories.
- The exact configuration and model checkpoint names.

Write small experiment outputs under `runs/<experiment_name>/`. Do not report
only aggregate accuracy; preserving per-query results is necessary for failure
analysis and paper ablations.

## Testing and verification

Before declaring a Python change complete:

1. Compile or import the modified modules.
2. Run the relevant unit or contract tests if they exist.
3. Test a small fixture rather than the full corpus when possible.
4. Verify that frame mappings remain valid.
5. Run linting or formatting checks if configured in the repository.

Useful lightweight checks include:

```bash
python -m compileall src
PYTHONPATH=src python -c "import aic"
```

Unit tests must not require downloading large models. Use fake retrievers,
rerankers, and small in-memory frame records for orchestration tests.

## Change discipline

- Inspect existing files before editing.
- Preserve user changes and the existing frontend.
- Keep changes scoped to the requested component.
- Do not rename shared fields without updating every consumer and test.
- Do not commit data, embeddings, model weights, or FAISS indexes.
- Record assumptions when the corpus format or official frame mapping is not
  known.
- Prefer a working baseline and a measured ablation over a broad unfinished
  implementation.

## Definition of done

A component change is complete when:

- Its inputs and outputs match the shared contracts.
- It can run on a small representative fixture.
- Errors include enough context to identify the video, frame, or query.
- Relevant tests or smoke checks pass.
- Configuration and experiment metadata are recorded when model behavior
  changes.
- The README or contract documentation is updated when public behavior changes.
