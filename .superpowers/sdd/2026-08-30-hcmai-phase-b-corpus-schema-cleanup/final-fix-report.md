# Phase B Final Fix Report

**Date:** 2026-08-31

## Scope

This final review wave closes the remaining Phase B offline/runtime boundary
and API-validation gaps without changing artifact paths or index layouts.

### Offline artifact construction

- Added `offline.artifact_readers` as the offline-owned projection layer for
  canonical frames, caption/OCR/FrameContext evidence, legacy ASR text,
  transcript timeline inspection, and frame asset resolution.
- Removed every direct `offline -> hcmai.corpus` import, including index builds,
  caption/OCR generation, object detection, transcript materialization, custom
  frame publication, and the corpus-build compatibility writer.
- Updated the AST dependency guard so `offline` may not import
  `hcmai.corpus`, `hcmai.api`, or `hcmai.orchestration`.

### Index integrity

- Caption, OCR, and FrameContext joins now validate every persisted specialist
  row against `video_id`, `frame_idx`, and `timestamp_ms` from its canonical
  frame before ignoring unavailable text.
- Caption, OCR, and FrameContext index inputs require matching
  `frame_store_id` lineage with the canonical frame artifact.
- Added regression coverage for identity and lineage mismatches, including
  rows with empty text that would previously have been skipped.

### API boundary

- KIS requests require a nonblank query and `top_k >= 1`.
- TRAKE requests require a nonempty list of nonblank events and `top_k >= 1`.
- Invalid request values now fail in Pydantic/FastAPI validation with HTTP 422,
  before service/workflow code can raise an unhandled `ValueError`.

### Documentation

- Clarified offline-contract and runtime-evidence ownership descriptions.
- Repaired the stale custom-pipeline module path in the bootstrap runbook.
- Clarified the organizer-provided BTC Keyframes wording and corrected the
  Task 12 boundary claim.

## Validation

```text
PYTHONPATH=.:src aic/bin/python -m pytest \
  tests/architecture tests/compat/test_runtime_loaders.py \
  tests/api/test_contracts.py tests/api/test_search_routes.py \
  tests/api/test_trake_routes.py tests/orchestration/test_kis_pipeline.py \
  tests/retrieval/test_context_index.py tests/test_caption.py tests/test_ocr.py \
  tests/test_transcript_reliability.py tests/data/test_custom_frames.py \
  tests/scripts/test_detect_objects.py -q
```

Result: **132 passed**.

Additional focused producer/index coverage passed:

```text
PYTHONPATH=.:src aic/bin/python -m pytest \
  tests/test_caption.py tests/test_ocr.py tests/test_transcript_reliability.py \
  tests/data/enrichment/test_caption_evidence.py \
  tests/data/enrichment/test_ocr_evidence.py \
  tests/data/enrichment/test_asr_segment_evidence.py \
  tests/data/test_custom_frames.py tests/scripts/test_custom_extraction_cli.py \
  tests/corpus_build/test_index_alignment.py -q
```

Result: **83 passed**.

`git diff --check` and the offline corpus-import scan pass. The repository
environment does not install `ruff`, so lint was not run.

## Compatibility and remaining concern

Artifact filenames, Parquet layouts, and dense-index publication are unchanged.
The only intentionally visible API change is rejection of formerly accepted
blank/nonpositive KIS and TRAKE inputs at the documented HTTP boundary.
