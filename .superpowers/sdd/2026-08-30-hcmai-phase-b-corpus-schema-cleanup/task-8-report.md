# Task 8 report — move index construction offline

## Status

Completed. Retrieval runtime now loads and searches published bundles, while
all corpus construction and publication entry points live under
`offline.indexes`.

## Changed ownership

- Added `offline.indexes.text` for Caption/OCR/frame-ASR and FrameContext
  index construction, preserving the established supplemental-vector bundle
  behavior.
- Added `offline.indexes.asr_segment` for segment-native ASR corpus/index
  construction and corrected its transcript source to
  `offline.enrichment.transcripts.artifacts`.
- Added `offline.indexes.text_embeddings` for the standalone text-vector
  writer and `offline.indexes.visual.build_index` as the offline
  `DenseIndex.build` adapter.
- Removed runtime text/segment artifact modules and all five builder
  convenience methods from `RetrievalService`. `DenseIndex`,
  `SegmentDenseIndex`, their `load`/search APIs, and runtime retrievers remain
  under `hcmai.retrieval`.
- Updated corpus preparation and the offline index CLI to import the new
  offline builders directly.

## Compatibility coverage

- Added architecture checks preventing runtime retrieval modules from defining
  or importing offline index builders.
- Extended Context and ASR build tests to assert all current bundle member
  filenames, metadata schema version, checksum-manifest keys, and successful
  round-trip loading. The tests intentionally do not snapshot FAISS bytes.

## Validation

- `PYTHONPATH=.:src aic/bin/pytest tests/retrieval tests/compat tests/architecture/test_index_ownership.py -v`
  — **95 passed**.
- `PYTHONPATH=.:src aic/bin/python -m compileall -q src/hcmai/retrieval offline/indexes offline/ingestion/corpus_build scripts/build_retrieval_indexes.py`
  — passed.
- `git diff --check` — passed.

The system Python environment lacks `faiss` and `pyarrow`; the project `aic`
environment provides both and was used for the full prescribed suite.

## Research

No research was needed: this is a mechanical ownership split with no new
retrieval or scoring algorithm.
