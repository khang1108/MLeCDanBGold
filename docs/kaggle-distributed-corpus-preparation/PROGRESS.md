# Kaggle Distributed Corpus Preparation Progress

## Status: Implementation Complete; Real-Endpoint Smoke Pending

## Delivered

- Extended the existing inference API/client with OCR, image/DINO embeddings,
  shot/GEBD scoring, ASR, and diarization contracts.
- Added least-busy endpoint pools with bounded gateway failover.
- Added remote domain adapters with model/revision, identity/order, shape,
  finiteness, and actual L2-normalization validation.
- Added deterministic scoped S3 audio references and exact cleanup.
- Added local `Lxx` inventory verification, resumable group orchestration,
  artifact-only embedding stages, immutable `COMMITTED.json` publication, and
  deterministic committed-group index reduction.
- Added the Kaggle capability runtime, notebook template, operator runbook,
  typed configuration examples, and group coordinator CLI.

## Verification

- Focused implementation suite: **60 passed**.
- Expanded preprocessing/corpus/transcript suite: **103 passed, 8 existing
  failures** in legacy APIs/fixtures outside the changed code.
- Full repository collection is additionally blocked in the minimal test
  environment by missing `torch`/`sentence-transformers` and the already-missing
  `hcmai.data.prepare` module.
- `compileall` and `git diff --check` pass.

The eight existing failures are in legacy `CandidateFrame`/`deduplicate`, OCR
factory, `DataService.prepare_adaptive`, and one nested-prefix fixture. None of
the corresponding implementation files were modified by this work.

## Environment Constraint

The full project extra set cannot currently resolve as one environment because
the reranking extra pins `transformers < 5` while the transcripts extra requires
`transformers >= 5.13`. Verification therefore used a minimal isolated `uv`
environment containing only dependencies needed by the exercised paths.

## Production Gate Still Required

Before a full corpus run, supply real Kaggle/Cloudflare endpoints and immutable
TransNet/EfficientGEBD pins, then run a five-video smoke group without cleanup.
Record stage throughput, P50/P95 latency, endpoint failover, the committed S3
manifest, reducer validation, and the projected 24-hour completion time.

## Session Log

### 2026-08-16

- Validated the architecture against active repository contracts.
- Implemented all five phases while preserving legacy S3 preparation entrypoints.
- Added fault/invariant tests and completed local verification.

## Architectural Decisions

- Existing `InferenceClient`, gateway, schemas, and domain services are extended
  in place; there is no parallel HTTP stack.
- `Lxx` is the transaction, resume, publication, and cleanup boundary.
- Workers are stateless inference providers; canonical identity and publication
  remain coordinator-owned.
- GEBD sampling/window overlap remains local; only fixed-window scoring is remote.
- Per-group vectors are published without indexes; global indexes are built only
  from checksum-verified committed bundles.
