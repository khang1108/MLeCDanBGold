# Task 6 report: offline artifact producers

## Status

Implemented. Artifact-producing ingestion, preprocessing, enrichment, corpus
build, custom pipeline, and S3 modules now live under `offline/`. Runtime
transcript lookup remains owned by `hcmai.corpus.stores.transcript`.

## Changes

- Moved the producer packages with `git mv`, preserving artifact layouts and
  package exports.
- Rewrote moved-package, script, test, and ThunderCompute imports to the new
  `offline.*` paths; no compatibility wrappers were added.
- Preserved repository-root path resolution after the move.
- Kept the caption CLI available as `python -m offline.enrichment.caption`.
- Added the offline/runtime import-boundary and package-initializer tests.
- Included `offline*` in setuptools package discovery.

## Validation

- `PYTHONPATH=.:src python -m compileall -q src/hcmai offline` — passed.
- `PYTHONPATH=.:src python -m offline.enrichment.caption --help` — passed.
- Focused architecture, BTC keyframe-map, custom manifest/state, and config
  tests — 43 passed.

The custom extraction CLI tests requiring the prebuilt native extractor and
Parquet tests requiring `pyarrow` were not fully runnable in this environment.
The existing retrieval package-layout test also requires the unavailable
`faiss` dependency. A remaining import from the pre-Task-8 retrieval segment
artifact builder still points at the old transcript artifact path and should
be resolved by Task 8 when that builder moves to `offline/indexes`.

## Fix round 1

- Restored `offline.enrichment.caption.__main__` to delegate through
  `EnrichmentService.run_caption_cli()`, preserving the moved entry point's
  original execution path and options.
- Updated `src/hcmai/data/WORKFLOW.md` with the new `offline.*` producer
  ownership and `hcmai.corpus.stores` runtime boundary.
- Kept the small import deferrals in the aggregate enrichment service and
  object detector so the required `--help` path does not require optional
  Parquet dependencies; generation behavior and call signatures are unchanged.

Fix-round validation: compileall, caption `--help`, and the focused suite all
passed (`43 passed`).
