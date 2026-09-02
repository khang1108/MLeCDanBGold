# Task 6 Artifact-Lineage Remediation Report

**Date:** 2026-09-02
**Status:** Complete — fixes future custom-pipeline builds only.

## Scope and root cause

The custom batch builder previously omitted the configured Visual and Context
encoder revisions, and rebuilt reused ASR subsets with the synthetic model
label `reused-asr-vectors`. Finalization then faithfully compacted that bad
metadata. No vectors, canonical identity, segment identity, projection,
scoring, BM25, Hybrid, or DP behavior was changed.

## Changes

- `DenseIndex.build` and `SegmentDenseIndex.build` now accept optional
  keyword-only `model_revision` values and persist them in `IndexMetadata`.
- `prepare_custom_pipeline.py` now loads the pinned Visual and evidence
  `EncoderConfig` values and threads both model name and revision into the
  batch runner. Its reusable-ASR factory validates against the configured
  evidence encoder before a batch is built.
- Batch construction persists the configured Visual and Context lineages.
  Reused ASR subsets retain the source `SegmentDenseIndex` model name and
  revision; the synthetic `reused-asr-vectors` label is no longer written.
- Both the source validator and batch builder reject ASR model/revision
  mismatches against the configured evidence encoder before batch publication.
- The active disk-backed finalizer already validated and preserved revisions;
  focused regression coverage now protects that behavior. The legacy
  in-memory compactor now validates a shared revision and returns it alongside
  the model name. Its two precomputed-vector builders accept and propagate an
  optional revision.

## TDD evidence

### RED

Before implementation:

```text
PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/data/custom_pipeline/test_lineage.py
6 failed in 1.49s
```

The failures were the expected missing `model_revision` keyword support in
both index builders and legacy finalization builders.

### GREEN

```text
PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/data/custom_pipeline/test_lineage.py
6 passed in 1.12s

PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/retrieval/test_segment_dense_index.py \
  tests/unit/retriever/test_dense_index_score_subset.py \
  tests/data/custom_pipeline/test_asr.py \
  tests/data/custom_pipeline/test_lineage.py
41 passed in 1.59s

PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/retrieval/evidence -k 'not production_artifacts_satisfy_fast_track_lineage'
31 passed in 1.50s

PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/orchestration/test_asr_projected_loading.py \
  tests/orchestration/test_hybrid_temporal_search.py \
  tests/orchestration/test_hybrid_health.py \
  -k 'not production_artifacts'
9 passed, 2 deselected in 1.07s

PYTHONPATH=.:src aic/bin/python -m pytest -q tests/temporal
9 passed in 0.11s
```

`compileall` passed for every changed production module and `git diff --check`
was clean.

## Known external test limitations

The broader existing custom-pipeline test invocation cannot currently collect
for unrelated baseline issues:

- `tests/data/custom_pipeline/test_shards.py` has a statement before its
  `from __future__ import annotations` line, producing a syntax error.
- `tests/data/custom_pipeline/test_finalize.py` imports the nonexistent
  `hcmai.data.custom_pipeline` package. Its only Task 6 update is the expected
  fourth return value (`model_revision`) from the legacy compactor.

These paths were already invalid at `HEAD`; no compatibility package or test
layout migration was added because it is outside this lineage remediation.

Mounted-artifact smoke tests were intentionally excluded. Current mounted
artifacts retain the old lineage and this task must not edit or migrate them.

## Research and knowledge record

No scholarly research or `KNOWLEDGE.md` update was needed: this is a
deterministic provenance propagation and validation correction, not an
algorithmic design decision.
