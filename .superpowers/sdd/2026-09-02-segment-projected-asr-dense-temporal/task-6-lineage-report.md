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
  in-memory compactor validates a shared revision while retaining its original
  three-value public result. Its two precomputed-vector builders accept and
  propagate an optional revision.

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
  `hcmai.data.custom_pipeline` package.

These paths were already invalid at `HEAD`; no compatibility package or test
layout migration was added because it is outside this lineage remediation.

Mounted-artifact smoke tests were intentionally excluded. Current mounted
artifacts retain the old lineage and this task must not edit or migrate them.

## Research and knowledge record

No scholarly research or `KNOWLEDGE.md` update was needed: this is a
deterministic provenance propagation and validation correction, not an
algorithmic design decision.

## Fix Round 1

### Reviewer findings addressed

- Restored `compact_batch_embeddings` to its original public
  `(vectors, mapping, model_name)` result. It still rejects model-revision
  drift internally.
- Batch ASR indexes now preserve the source `SegmentDenseIndex`
  `source_fingerprint` and `config_fingerprint`. Active global compaction
  validates both fields across batch metadata and writes the shared values to
  the final ASR index metadata.
- `ASRReuseBundle.transcript_fingerprint` and `index_fingerprint` remain on
  their existing validation/batch-scope contract. They are deliberately not
  copied into `IndexMetadata.source_fingerprint` or `config_fingerprint`:
  those fields retain the actual source-index metadata semantics.
- Added runnable regression coverage for source/config fingerprint propagation,
  fingerprint drift rejection, CLI factory validation, runner forwarding, and
  process-command encoder revision forwarding. The stale, uncollectable
  `test_finalize.py` assertion change was reverted.
- Updated the ASR factory type annotation to match the runner's
  `Callable[[Sequence[str]], ASRReuseBundle]` contract.

### RED

Before the round-1 implementation:

```text
PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/data/custom_pipeline/test_lineage.py
5 failed, 4 passed in 1.81s
```

The product failures showed the incompatible public four-tuple result, dropped
batch/global fingerprints, and missing fingerprint-drift validation. The
initial runner-test mock also omitted its synthetic video shard; that test seam
was corrected before the production fix.

### GREEN

```text
PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/data/custom_pipeline/test_lineage.py
10 passed in 1.54s

PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/retrieval/test_segment_dense_index.py \
  tests/unit/retriever/test_dense_index_score_subset.py \
  tests/data/custom_pipeline/test_asr.py \
  tests/data/custom_pipeline/test_lineage.py
45 passed in 2.36s
```

`compileall` passed for all remediation production modules and `git diff
--check` was clean.
