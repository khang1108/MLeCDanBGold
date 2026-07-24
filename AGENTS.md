# Agent Instructions

## Mission

- Build HCMAI 2026 frame retrieval for KIS, conversational KIS, VQA, and ad-hoc
  Vietnamese/English queries.
- Return exact official `video_id` and `frame_idx`; optimize Recall@K, ranking,
  reproducibility, and warm-query latency with small research-friendly code.

## Package Managers

- Python: use the repository virtual environment at `aic/`; install with
  `aic/bin/python -m pip install -e ".[embedding,dev]"`.
- Frontend: use `npm` in `frontend/`; preserve the existing React application.
- Add supported runtime dependencies to `pyproject.toml`.

## Current System

- Implemented: Pydantic contracts, frame preparation/`FrameStore`, embeddings,
  lazy dense encoder, FAISS retrieval/benchmark, KISC, search, and FastAPI.
- Planned: caption/OCR/ASR enrichment, multimodal reranking, measured runs.
- See `README.md` and component READMEs for layout, APIs, and artifact details.

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

| Task | Command |
|---|---|
| Compile | `aic/bin/python -m py_compile src/hcmai/<file>.py` |
| Test | `PYTHONPATH=src aic/bin/pytest tests/test_<component>.py` |
| Typecheck | `pyright src/hcmai/<file>.py` |

## Change and Commit Discipline

- Inspect first and preserve unrelated work. Update tests/docs with public
  behavior or contract changes; record unknown corpus-format assumptions.
- AI commits include `Co-Authored-By: <model name> <noreply@openai.com>`.
