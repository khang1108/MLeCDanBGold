# Kaggle Distributed Corpus Preparation Implementation Plan

## Overview

Implement the validated group-scoped offline preparation architecture. The
coordinator accepts an already-local `Lxx/videos` directory plus an S3 source
inventory, offloads pinned GPU inference through the existing resilient HTTP
gateway, publishes one immutable group bundle, and reduces committed group
embedding artifacts into global indexes.

## Prerequisites

- Validated architecture design in `docs/superpowers/specs/`.
- Existing canonical frame/enrichment/embedding contracts remain authoritative.
- Unit tests use injected fake runtimes and mocked S3; no network/model download.
- Exact model and artifact revisions remain pinned in configuration.

## Phase Summary

1. Remote contracts, client methods, and injectable API routes.
2. Domain adapters and configuration/composition wiring.
3. Local-group orchestration, validation, immutable publication, and reduction.
4. Kaggle capability runtime and notebook/runbook assets.
5. Integration, fault-injection, regression, and documentation verification.

---

## Phase 1: Remote Inference Foundation

### Objective

Extend the existing inference schemas, `InferenceClient`, and FastAPI service
with preprocessing, image embedding, OCR, ASR, and diarization capabilities.

### Tasks

- [x] Promote shared embedding response semantics without breaking text clients.
- [x] Add bounded request/response contracts with caller-owned identity.
- [x] Add multipart/binary client methods to the existing transport.
- [x] Add injectable FastAPI routes and strict payload validation.
- [x] Add contract and identity/order tests.

### Success Criteria

- Existing inference tests remain green.
- Every new capability has a fake-runtime API/client round trip.
- Invalid shape, identity, revision, or result cardinality is rejected.

### Files Likely Affected

- `src/hcmai/common/schemas/inference.py`
- `src/hcmai/common/schemas/__init__.py`
- `src/hcmai/llm/adapters/http.py`
- `src/hcmai/llm/server/api.py`
- `tests/test_llm_api.py`

---

## Phase 2: Domain Adapters and Composition

### Objective

Implement remote detector/encoder/enrichment/transcript adapters and select
them explicitly from typed preparation endpoint configuration.

### Tasks

- [x] Add remote TransNet, GEBD window scorer, and DINO adapters.
- [x] Add remote OCR and image-embedding adapters.
- [x] Add remote ASR/diarization adapters using scoped audio references.
- [x] Add endpoint-pool/config models and environment secret handling.
- [x] Inject adapters through existing sessions/services.
- [x] Add parity/invariant unit tests with fake clients.

### Success Criteria

- Offline services can run with injected remote adapters and no local GPU model.
- Model revision, shape, normalized-vector, and canonical ordering checks pass.
- Local adapters and existing configs remain backward compatible.

---

## Phase 3: Group Orchestration and Publication

### Objective

Add the local-group input contract, deterministic run identity, resumable state,
bundle validation, S3 commit publication, and global embedding reduction seams.

### Tasks

- [x] Validate local videos against a source inventory manifest.
- [x] Refactor preparation paths and operations to one `Lxx` group.
- [x] Persist atomic lifecycle and stage checkpoints.
- [x] Generate per-group embedding artifacts separately from indexes.
- [x] Publish immutable run manifests and final `COMMITTED.json`.
- [x] Reduce committed group mappings/vectors deterministically.
- [x] Preserve the legacy S3 preparation entry point where tests require it.

### Success Criteria

- Restart does not duplicate or change canonical results.
- Incomplete uploads are invisible.
- A committed group can be validated and reduced without local payloads.

---

## Phase 4: Kaggle Runtime and Operations

### Objective

Provide a configurable capability server and notebook/runbook assets for pinned
Kaggle T4 workers behind Cloudflare Access.

### Tasks

- [x] Add capability runtime composition controlled by environment variables.
- [x] Add server entry point and notebook template.
- [x] Document endpoint registration, model pins, Access tokens, and rotation.
- [x] Keep corpus identity and S3 publication out of the worker.

### Success Criteria

- Server starts with fake/disabled capabilities without model downloads.
- Readiness advertises exact configured capabilities and model provenance.
- Notebook steps contain no committed credentials or endpoint URLs.

---

## Phase 5: Verification

### Objective

Verify contracts, recovery, compatibility, and smoke-run readiness.

### Tasks

- [x] Run focused unit and integration suites after every phase.
- [x] Add endpoint-loss, revision-mismatch, partial-upload, and resume tests.
- [x] Run broader repository tests and static/diff checks.
- [x] Update configuration examples and operator documentation.
- [x] Record unverified performance risks and smoke benchmark command.

### Success Criteria

- Focused and broader tests pass or unrelated pre-existing failures are recorded.
- No secrets, raw videos, model weights, or generated artifacts are committed.
- The smoke workflow is executable when real Kaggle endpoints are supplied.

## Post-Implementation

- [ ] Run five-video real-endpoint smoke benchmark.
- [ ] Record per-stage throughput and P50/P95 latency.
- [ ] Validate the 24-hour projection before a full production run.
