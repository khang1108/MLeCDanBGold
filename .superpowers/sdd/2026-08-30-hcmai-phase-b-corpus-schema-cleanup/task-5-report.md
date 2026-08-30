## Task 5 report — 2026-08-30

### Completed migration

- Deleted `src/hcmai/data/pipeline.py`; runtime composition now enters through
  the read-only `Corpus` facade.
- Updated setup, search orchestration, temporal validation, KIS materialization,
  API frame routes, retrieval composition, reranking, and segment projection to
  consume Corpus methods or runtime `Frame` values rather than the removed
  mutable facade.
- Kept KIS ASR metadata point-contained by reading the half-open
  `[timestamp_ms, timestamp_ms + 1)` transcript range.
- Preserved canonical identity validation and restored canonical manifest
  lineage checks in `Corpus.open` for text evidence and object-count artifacts.
- Moved BTC frame-store preparation and offline text/index construction to
  direct corpus stores, avoiding a runtime dependency from offline builders.

### Tests and scans

- `PYTHONPATH=.:src aic/bin/pytest tests/api tests/orchestration tests/temporal tests/retrieval tests/corpus tests/architecture -v`
  — **157 passed**.
- Focused migration-adjacent legacy tests — **54 passed**.
- `rg -n 'DataService|hcmai\\.data\\.pipeline' src/hcmai tests` — no matches.
- `python -m compileall -q src tests` and `git diff --check` passed.

### Follow-up note

A scoped FAISS-stub setup test reloads retrieval modules. The visual-scoring
test now patches the concrete service method's globals, preventing it from
patching a later reloaded module instance. This was a deterministic test-order
issue, not a runtime behavior change.

### Review-fix round

- Kept API and orchestration on the public `Corpus` boundary: frame asset
  errors are now handled through the standard `OSError` base class, while
  asset sampling and evidence availability are explicit `Corpus` methods.
- Made a missing or empty canonical frame artifact fail startup immediately;
  optional artifacts and invalid optional evidence still produce diagnostics.
- Restored sample-based frame-asset health and made `evidence_stores`
  consistent between initialized and unavailable services.
- Added architecture coverage preventing API, orchestration, and temporal code
  from importing private `corpus.assets` or `corpus.stores` modules.
- Regression: `PYTHONPATH=.:src aic/bin/pytest tests/api tests/orchestration
  tests/temporal tests/retrieval tests/corpus tests/architecture
  tests/test_frame_assets.py -v` — **163 passed**.

### Review-fix round 2

- Added `CorpusFrameLoadError` to distinguish failures opening required
  canonical frame metadata from optional artifact failures.
- `_load_corpus` now re-raises required frame failures, while retaining its
  existing diagnostic-and-degrade path for optional artifact errors surfaced by
  `Corpus.open`.
- Added a regression for a present but malformed `frames.parquet` artifact.
- Validation: `PYTHONPATH=.:src aic/bin/pytest
  tests/orchestration/test_fast_track_setup.py tests/corpus/test_corpus.py -v`
  — **21 passed**; `python -m compileall -q src tests` and `git diff --check`
  also passed.
