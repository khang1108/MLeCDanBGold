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
