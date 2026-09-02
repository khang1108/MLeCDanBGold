# Final Review Fix Report

**Date:** 2026-09-02  
**Status:** Critical/Important findings complete  
**Implementation commit:** `55b1583`

## Findings resolved

- **C1 — self-contained tree:** committed the previously missing BM25, Dense,
  Hybrid, and normalization modules; the offline BM25 builder and SciPy runtime
  dependency; Hybrid request/response, orchestration, workflow, setup, and
  retriever bindings; their regression tests; and the coordinated baseline
  repairs (removed Filter router inventory, retained JSON error middleware, and
  updated the empty-alignment fixture).
- **I1 — Pydantic v2 validator:** `IndexConfig.text_embedding_filenames` now
  uses `@field_validator` followed by `@classmethod`. Tests reject missing
  modalities, path traversal/nested paths, and non-`.npy` filenames.
- **I2 — truthful Dense health:** Context/ASR dimension incompatibility marks
  ASR Dense unready, projector failure marks ASR unready, and deterministic
  scorer identity errors mark their named Context or ASR source unready.
  `dense_temporal` remains false whenever the combined scorer is unavailable.
- **I3 — projected identity:** `SegmentProjectedASRIndex` validates the returned
  `(video_id, frame_idx, timestamp_ms)` against the canonical row selected by
  `frame_id`; stale projector results fail construction.
- **I4 — event ceiling:** the conservative hard ceiling is **32** events.
  `SearchConfig.max_temporal_event_count` can be tightened to `1..32`, the
  baseline declares `32`, API event arrays are capped at 32, and KIS/TRAKE plus
  the shared temporal service reject configured-limit violations before
  translation, encoding, or evidence scoring.

## Committed files

- Configuration/dependency: `configs/baseline.yaml`, `pyproject.toml`,
  `src/hcmai/common/config.py`.
- API/baseline repair: `src/hcmai/api/contracts/search.py`,
  `src/hcmai/api/contracts/trake.py`, `src/hcmai/api/routers/__init__.py`,
  `src/hcmai/api/routers/filter.py` (removed), `src/hcmai/app.py`.
- Runtime closure: `src/hcmai/orchestration/pipeline.py`, `setup.py`,
  `temporal_search.py`, `workflows/kis.py`, `workflows/trake.py`,
  `src/hcmai/retrieval/retriever/pipeline.py`, and
  `src/hcmai/retrieval/evidence/{asr_projected,bm25,dense,hybrid,normalization}.py`.
- Offline closure: `offline/indexes/bm25.py`.
- Tests: updated API, configuration, orchestration, projection, and broad API
  fixtures plus new `tests/offline/indexes/test_bm25_builder.py`,
  `tests/api/{test_hybrid_search_contracts,test_router_inventory}.py`,
  `tests/orchestration/{test_hybrid_query_routing,test_hybrid_temporal_search}.py`,
  and `tests/retrieval/evidence/{test_bm25,test_hybrid,test_normalization}.py`;
  obsolete `tests/api/test_filter_routes.py` was removed.

The two untracked design-plan drafts were intentionally excluded. No mounted
artifact was edited.

## TDD and verification

### RED

Before the implementation, the config/API slice produced **6 failures, 11
passes**: the broken decorator accepted incomplete/unsafe filename mappings,
and no event-limit configuration or API cap existed.

### GREEN

```text
PYTHONPATH=.:src aic/bin/pytest -q \
  tests/retrieval/evidence \
  tests/orchestration/test_asr_projected_loading.py \
  tests/orchestration/test_hybrid_temporal_search.py \
  tests/orchestration/test_hybrid_health.py tests/temporal \
  -k 'not production_artifacts'
56 passed, 2 deselected in 3.36s

PYTHONPATH=.:src aic/bin/pytest -q \
  tests/orchestration/test_kis_pipeline.py \
  tests/orchestration/test_trake_pipeline.py \
  tests/api/test_hybrid_search_contracts.py \
  tests/api/test_router_inventory.py tests/test_api.py tests/test_api_contracts.py
26 passed in 3.85s

PYTHONPATH=.:src aic/bin/pytest -q \
  tests/retrieval/test_segment_dense_index.py \
  tests/unit/retriever/test_dense_index_score_subset.py \
  tests/data/custom_pipeline/test_asr.py tests/data/custom_pipeline/test_lineage.py
45 passed in 5.19s
```

The new-finding slice passed **57 tests** with the two production-artifact tests
deselected. A detached worktree at committed `55b1583` ran the same slice with
**57 passed, 2 deselected**, proving the committed dependency closure imports
and runs without worktree-only source files. `compileall` and `git diff --check`
also passed.

## Known external limitation

The mounted production lineage test still fails truthfully because the mounted
Context revision and ASR model do not match the configured evidence encoder.
Those artifacts are external to this commit and were not modified. No
performance benchmark was run and no performance claim is made. This was a
deterministic bug-fix wave, so `KNOWLEDGE.md` was not updated.
