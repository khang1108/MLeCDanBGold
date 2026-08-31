# Task 10 report — 2026-08-31

## Completed migration

- Deleted the generic `hcmai.common.schemas` package after rewriting every
  Python caller to a domain owner.
- Kept the six-field frozen `hcmai.corpus.models.Frame` as the sole runtime
  frame contract. Provenance-heavy frame artifact validation now belongs to
  `offline.ingestion.models.FrameArtifact`; no `FrameRecord` replacement was
  added to the runtime.
- Moved retrieval candidates/results/sources and request traces to frozen
  runtime dataclasses under retrieval and observability ownership.
- Moved Caption, OCR, Object, FrameContext, transcript, legacy frame
  enrichment, and processing-state artifact models under their offline
  producer owners. Runtime corpus stores validate private read projections and
  do not import offline producers.
- Moved catalog/submission HTTP models to API contracts, evaluation models to
  the retrieval benchmark, and remote embedding responses to the retrieval
  embedding boundary.
- Moved enrichment worker HTTP contracts to
  `offline.enrichment.inference_contracts` and retained the four live private
  ThunderCompute contracts (`TextEmbeddingRequest`, `BoundaryScoreResponse`,
  `RerankItem`, and `RerankResponse`) in `thundercompute.contracts`, per the
  ledger ruling.
- Added the schema-ownership architecture regression and updated obsolete
  package-layout/absence tests to assert the retired package itself.

## Validation

- `PYTHONPATH=.:src python -m compileall -q src/hcmai offline thundercompute scripts`
  — passed.
- Task 10 architecture, contract, evidence, telemetry, and KIS focused suite —
  **46 passed**.
- Readiness, API, Corpus-test-double, and temporal service focused suite —
  **37 passed** before the final KIS fixture correction; the corrected KIS
  test is included in the 46-test passing suite above.
- Initial complete suite — **851 passed, 14 failed**. Eight migration-adjacent
  fixture failures found by that run were corrected.
- Rechecked the remaining failures directly — **6 failed**, all outside Task
  10: five legacy flat-CLI expectations in
  `tests/scripts/test_custom_pipeline.py`, plus the Task 9 native C++ tree
  being treated as Python packages by
  `tests/architecture/test_offline_boundary.py`.
- `git diff --check` and the source import scan passed. No Python production or
  test import of `hcmai.common.schemas` remains; the only matches are deliberate
  package-absence assertions.

## Compatibility and concerns

- Published artifact columns and API field semantics are preserved; Pydantic
  remains at external HTTP and artifact-validation boundaries.
- Runtime ranking and observability values are now frozen dataclasses, so
  callers use `dataclasses.replace` instead of Pydantic `model_copy`.
- The six pre-existing/out-of-scope failures above are intentionally not
  changed by this schema ownership commit.
- No research decision or algorithm changed, so `KNOWLEDGE.md` was not updated.
