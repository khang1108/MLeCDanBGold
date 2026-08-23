# HCMAI Roadmap Implementation Program Design

> **Status:** Approved design awaiting written-spec review
> **Date:** 2026-08-13
> **Source specification:**
> `docs/HCMAI_SYSTEM_ARCHITECTURE_STATUS_AND_ROADMAP.md`
> **Implementation planning workflow:** Superpowers `writing-plans`

## 1. Purpose

This design converts the approved HCMAI architecture roadmap into a bounded
implementation program. It covers repository stabilization, product contract
correctness, data/evidence reliability, and convergence on a shared temporal
package for KIS, VQA, and TRAKE.

The program deliberately excludes current corpus/index synchronization,
evaluation framework construction, quality experiments, and distributed
progressive state. Those workstreams will be designed later.

The outcome of this design is four independently reviewable implementation
plans. Each plan must produce working, tested software before the next dependent
plan begins.

## 2. Fixed architectural decisions

The implementation must preserve these project invariants:

1. `FrameRecord` remains the canonical frame identity.
2. KIS and VQA use the same progressive scene-localization behavior.
3. VQA hints localize; the question selects evidence and asks for an answer.
4. TRAKE preserves hard ordered-event semantics and its existing monotonic DP.
5. A missing retrieval hit is not automatically negative evidence.
6. Hint reveal order is not automatically video event order.
7. Online startup and requests never rebuild offline artifacts.
8. Providers may only select canonical frame IDs supplied by HCMAI.
9. New contracts are introduced only when active contracts cannot safely own
   the required semantics.
10. No quality or latency improvement is claimed without a later recorded
    experiment.

## 3. Program scope

### 3.1 Included

- restore a trustworthy tracked backend/frontend test baseline;
- resolve stale schema, observability, and configuration expectations;
- fix caption configuration root resolution;
- fix VQA answerability, event context, and evidence provenance contracts;
- make browser-to-backend TRAKE requests use the dedicated typed API;
- isolate progressive browser state by task/session fingerprint;
- remove dead and unsafe frontend client behavior;
- preserve exact decoded audio timeline information;
- make diarization optional;
- add transcript resume/model/config fingerprinting;
- materialize transcript segments into frame-aligned ASR evidence;
- validate and atomically publish local transcript artifacts;
- converge all task pipelines on one shared temporal package with distinct
  scene and ordered-path aligners.

### 3.2 Deferred

- rebuilding or synchronizing the current visual index;
- publishing or downloading current frame images;
- corpus-scale OCR, ASR, caption, embedding, or index jobs;
- evaluation datasets, official metrics, and benchmark runners;
- P2 fusion, reranking, frame-selection, OCR, motion, relation, or TRAKE
  experiments;
- persistent observability expansion beyond directly affected stages;
- S3 upload implementation and cloud execution;
- distributed progressive state;
- replacement of the existing TRAKE DP;
- model or threshold ablations.

## 4. Program structure and dependency order

```text
Prerequisite: owner checkpoints the current dirty migration
        |
        v
Plan 01: Repository and Test Baseline
        |
        v
Plan 02: Task, API, and Frontend Contract Correctness
        |
        v
Plan 03: Data and Evidence Reliability
        |
        v
Plan 04: Shared Temporal/Progressive Convergence
```

Plans 02 and 03 are technically separable after Plan 01, but the default
execution order remains serial. This minimizes concurrent edits to common
configuration, schemas, and integration fixtures. Plan 04 runs last so parity
tests protect the architecture migration.

The current working tree contains extensive uncommitted migration work. Before
implementation, the repository owner must create an intentional checkpoint.
Executors must not use reset, checkout, or bulk deletion to manufacture a clean
state.

## 5. Plan 01 — Repository and test baseline

### 5.1 Goal

Produce a tracked, repeatable backend/frontend validation baseline that detects
regressions in the active architecture.

### 5.2 Owned files

- `.gitignore`
- `src/hcmai/data/enrichment/caption/config.py`
- active tests under `tests/`
- observability tests importing the old package location
- configuration/schema tests containing retired expectations
- repository validation documentation or existing command entrypoint

### 5.3 Design

The entire `tests/` directory must no longer be ignored. Existing staged
deletions are reconciled deliberately:

- tests for active behavior are restored and migrated;
- tests for removed KISC/conversation behavior are retired with an explicit
  removal rationale;
- the orphan VQA evaluator test is retired because evaluator implementation is
  explicitly deferred and there is no runtime consumer;
- observability tests import `hcmai.common.observability`;
- configuration assertions use the active artifact-root configuration;
- TRAKE tests assert the implemented pipeline rather than the historical
  unimplemented state.

`CaptionJobConfig` must derive the repository root correctly from its module
path. Tests use an explicit temporary configuration when verifying relative
path resolution and must not depend on an arbitrary process working directory.

The validation command must execute, in order:

1. focused temporal/VQA/TRAKE tests;
2. the complete tracked backend suite;
3. frontend tests;
4. frontend production build;
5. whitespace/diff validation.

### 5.4 Failure behavior

- Collection failure is a release-gate failure.
- A stale test is migrated or explicitly retired, never silently skipped.
- Tests requiring unavailable external networks/models use deterministic fakes.
- No test may trigger corpus reconstruction or remote inference.

### 5.5 Exit gate

- `tests/` is tracked and visible in `git status`.
- Backend collection completes.
- The full tracked backend suite completes with documented deterministic
  results.
- Frontend tests and build pass.
- Caption root resolution tests pass from repository and non-repository working
  directories.
- `git diff --check` passes for files owned by the plan.

## 6. Plan 02 — Task, API, and frontend contract correctness

### 6.1 Goal

Make KIS, VQA, and TRAKE product contracts correct from browser request through
backend response without fake competition results or cross-task state leakage.

### 6.2 Owned files

- `src/hcmai/common/schemas/vqa.py`
- `src/hcmai/pipelines/vqa/reasoning/answerer.py`
- `src/hcmai/pipelines/vqa/reasoning/evidence.py`
- `src/hcmai/pipelines/vqa/domain/models.py` when needed for internal evidence
- `src/hcmai/llm/adapters/vqa.py`
- local/HTTP VQA adapter contracts affected by the shared schema
- `frontend/src/api/search.js`
- `frontend/src/api/search.test.js`
- `frontend/src/features/vqa/components/VqaSearchWorkspace.jsx`
- its focused frontend tests
- affected VQA and TRAKE API/integration tests

### 6.3 VQA response contract

The answer stage reads `answerable`, matching the unified
`VQAInferenceResponse` and provider output. The compatibility parser
may accept the retired `answerability` spelling only as a lower-priority legacy
alias during this migration. When `answerable` is false, the candidate is
rejected with `provider_returned_unanswerable` even if `grounded` is true.

Provider-selected frame identity remains constrained to the ordered frame IDs
supplied for inference. An unknown frame ID rejects the candidate.

### 6.4 VQA evidence contract

The existing `VQAInferenceEvidence` is extended instead of introducing a
parallel VQA evidence wrapper. It gains bounded structured items containing:

```text
source
value
frame_id
start_ms
end_ms
confidence
provenance
```

Legacy aggregated caption/OCR/ASR fields remain during migration for one-frame
providers. Multi-frame prompts consume structured items and render their
canonical frame/time association explicitly.

The inference call also receives the localized event description. The prompt
separates it from the question:

```text
Scene context: <event description>
Question: <question>
Evidence: <bounded structured evidence>
```

Scene context cannot select evidence outside the already localized scene.

### 6.5 Frontend task contracts

The frontend gains a dedicated `searchTrake()` API function using
`POST /api/v1/trake` and a payload containing explicit ordered `events`. The UI
must require at least two non-empty events and render ordered path submissions,
not frame-search results.

Progressive search IDs are stored under a key derived from task type and stable
session fingerprint. A KIS ID cannot be sent to VQA, VKIS, or TRAKE. TRAKE is
stateless and does not use a progressive search ID under the current contract.

The following behavior is removed:

- dead KISC/MiniChallenge client requests;
- Suggest UI/client calls unless an active backend route exists;
- backend-unreachable fallbacks that return plausible mock frames or answers.

Development fixtures remain available only through explicit tests or an
explicit development-only mode that cannot activate in a production build.

### 6.6 Failure behavior

- unknown/expired progressive ID: HTTP 410 and local key removal;
- incompatible task/fingerprint state: HTTP 409 and visible reset instruction;
- malformed TRAKE events: HTTP 422 and no request retry;
- backend unreachable: visible error, no results;
- VQA unanswerable: deterministic warning/evidence fallback, no fabricated
  answer;
- missing image/provider failure: existing bounded warning/fallback behavior.

### 6.7 Exit gate

- negative tests prove `answerable=false` cannot become a submission;
- text evidence reaches the provider with frame/time provenance;
- event context and question are distinguishable in provider requests;
- provider frame-ID validation remains intact;
- frontend TRAKE requests use `/api/v1/trake` with ordered events;
- task switching cannot reuse incompatible progressive IDs;
- no production API helper returns mock competition results;
- focused backend/frontend and end-to-end contract suites pass.

## 7. Plan 03 — Data and evidence reliability

### 7.1 Goal

Make transcript production reproducible, timeline-correct, resumable, and
compatible with online frame-aligned ASR retrieval.

### 7.2 Owned files

- `src/hcmai/common/config.py`
- `configs/enrichment.yaml` when transcript settings belong there
- `thundercompute/config.yaml` for immutable model revisions
- `src/hcmai/data/enrichment/transcripts/adapters/asr.py`
- `src/hcmai/data/enrichment/transcripts/pipeline.py`
- `src/hcmai/data/enrichment/transcripts/prepare.py`
- `src/hcmai/data/enrichment/transcripts/store.py`
- new focused transcript manifest/resume module
- new `src/hcmai/data/enrichment/transcripts/materialize.py`
- `scripts/prepare_transcripts.py`
- affected transcript/config/frame-enrichment tests

### 7.3 Decoded audio timeline

Audio decoding returns an internal immutable value containing:

```text
samples: float32 mono waveform
sample_rate: int
start_ms: int
```

`start_ms` is derived from the first valid audio PTS/time base, with the stream
start time used only as a documented fallback. VAD/ASR segment offsets are
translated into media time before constructing `TranscriptSegment`.

The adapter must reject non-monotonic or negative final segment intervals.

### 7.4 Reproducibility and optional diarization

`ASRConfig` and `DiarizationConfig` include immutable model revisions. A
transcript manifest records source fingerprint, relevant configuration hash,
resolved model revisions, schema/pipeline version, segment count, and completion
status.

Resume reuses an output only when its manifest matches all relevant inputs.
Changed source, model revision, configuration, or schema invalidates reuse.

Diarization becomes optional. When disabled, transcript segments retain
`speaker_id=None`; ASR output remains valid and retrievable.

### 7.5 Segment-to-frame materialization

`materialize.py` consumes canonical `FrameRecord` values and
`TranscriptSegment` values. For each frame it selects transcript segments whose
half-open intervals overlap the configured frame evidence window. It emits
existing `FrameEnrichment` rows with:

- canonical `frame_id`;
- stable chronological, deduplicated `asr_text`;
- configured enrichment version;
- immutable model/pipeline identity;
- completed or explicit no-evidence status according to the existing schema.

Materialization must never create frame IDs, derive video identity from
filenames, or turn an unevaluated video into negative evidence.

### 7.6 Publication seam

The local implementation writes to a sibling staging path, validates the full
table and manifest, then atomically replaces the target. A failed validation or
write leaves the previous valid artifact unchanged.

The plan defines a narrow storage/publication protocol only if an existing
contract cannot represent staging and atomic promotion. S3 upload and cloud
credentials remain deferred.

### 7.7 Failure behavior

- missing/invalid PTS uses an explicit documented fallback or fails the video;
- model/config/source mismatch invalidates resume;
- one failed video is recorded without corrupting completed videos;
- invalid transcript intervals fail validation;
- unknown frame foreign keys fail materialization;
- publication failure preserves the previous valid artifact;
- no secret or raw provider error is written into the manifest.

### 7.8 Exit gate

- hand-calculable audio-offset tests pass;
- transcript timestamps remain in media time after VAD segmentation;
- optional diarization tests pass with and without a diarizer;
- every resume fingerprint field has a mismatch test;
- segment/frame overlap and text deduplication fixtures pass;
- emitted rows validate as `FrameEnrichment`;
- atomic publication failure tests preserve the old artifact;
- online `DataService` can load the materialized ASR artifact.

## 8. Plan 04 — Shared temporal/progressive convergence

### 8.1 Goal

Make KIS, VQA, and TRAKE consume one shared temporal package for query planning,
canonical evidence acquisition, alignment composition, and diagnostics while
preserving their different localization semantics and output heads.

### 8.2 Architectural choice

The program adopts pluggable aligners rather than forcing all tasks through one
scene algorithm.

```text
Task adapter
  -> TemporalQueryPlan
  -> ProgressiveEvidenceProvider | OrderedEvidenceProvider
  -> SceneAligner | OrderedPathAligner
  -> shared temporal result contract
  -> thin task head
```

KIS and VQA use sparse progressive evidence and scene coverage alignment.
TRAKE uses dense ordered evidence and monotonic path alignment.

### 8.3 Shared contracts

`src/hcmai/common/schemas/temporal.py` is extended with:

- `TemporalAlignmentMode` with `progressive_scene` and `ordered_path`;
- `TemporalQueryPlan` containing task type, query units, constraints, filters,
  and alignment mode;
- `OrderedPathCandidate`, promoted from the semantics currently owned by
  `TrakePath`, containing `path_id`, `video_id`, ordered canonical
  `FrameRecord` values, ordered query-unit IDs, score, and reason labels;
- validators that guarantee one video, canonical ordered frames, unique query
  unit IDs, and mode-compatible constraints.

`SceneCandidate` remains the scene result. The ordered-path contract is not
misrepresented as a scene.

Progressive `search_id`, snapshot, session fingerprint, and state version remain
state/workflow inputs rather than semantic query-plan fields. After snapshot
differencing, the progressive workflow creates a `TemporalQueryPlan` from the
committed query units. TRAKE creates the same plan type directly from explicit
ordered events.

### 8.4 Ports and implementations

Create focused ports under `src/hcmai/temporal/ports.py`. Sparse and dense
evidence deliberately use separate typed interfaces:

```text
ProgressiveEvidenceProvider.acquire(state, unit, filters)
  -> ProgressiveAcquisition

OrderedEvidenceProvider.acquire(plan)
  -> tuple[VideoEventScores, ...]

SceneAligner.align(plan, progressive_evidence)
  -> tuple[SceneCandidate, ...]

OrderedPathAligner.align(plan, video_scores)
  -> tuple[OrderedPathCandidate, ...]
```

`ProgressiveAcquisition` is an internal immutable result containing the proposed
evidence state, candidate video IDs, warnings, and retrieval trace. It does not
commit progressive state; the facade remains the transaction owner.

Concrete implementations are organized by responsibility:

- `temporal/providers/sparse.py` — current global/local/backfill progressive
  evidence logic;
- `temporal/providers/dense.py` — adapter over current TRAKE candidate-video
  dense event scoring;
- `temporal/aligners/scene.py` — current clustering and scene scoring;
- `temporal/aligners/monotonic.py` — compatibility adapter over current TRAKE
  `align_video()`/`rank_paths()` behavior that materializes
  `OrderedPathCandidate` through canonical frame records.

Multiple real implementations justify the ports. No speculative base hierarchy
is introduced.

### 8.5 Facade and state

`TemporalEvidenceCore` remains the public composition facade during migration.
Its existing KIS/VQA `localize()` behavior remains compatible while internals
move behind the sparse provider and scene aligner.

The facade gains an explicit ordered-path operation used by TRAKE. It does not
infer which behavior to run from arbitrary text; the task adapter supplies a
validated `TemporalQueryPlan`.

Progressive state remains limited to cumulative KIS/VQA snapshots. TRAKE remains
stateless because its active contract is an explicit ordered event list and no
official progressive TRAKE behavior is defined.

### 8.6 Task heads

- KIS consumes ranked `SceneCandidate` values and selects a representative
  canonical frame.
- VQA consumes the same ranked scenes, then performs question-conditioned
  evidence selection and grounded answering.
- TRAKE consumes `OrderedPathCandidate` values and shapes the existing response and
  CSV submission.

No task head performs low-level index search, canonical identity construction,
or temporal alignment.

### 8.7 Rollout

1. Add shared contracts and validators without changing runtime routing.
2. Extract the scene aligner and prove byte/field-equivalent KIS/VQA fixtures.
3. Wrap the TRAKE DP and prove identical paths/scores on existing fixtures.
4. Extract sparse and dense providers behind the ports.
5. Construct one temporal facade in `SearchService` and inject it into all task
   heads.
6. Route TRAKE through the compatibility operation behind validated
   configuration.
7. Flip the shared route default only after parity and integration suites pass.
8. Remove superseded direct orchestration only after the full contract suite
   passes.

### 8.8 Failure behavior

- invalid plan/mode combinations fail before retrieval;
- progressive state failure preserves the previous committed version;
- sparse optional-modality failure preserves successful sources;
- dense provider failure does not fall back to unordered scene alignment;
- monotonic alignment with insufficient frames returns no path;
- provider/aligner results with mixed video identity fail validation;
- compatibility rollout can revert routing without reverting contracts or data.

### 8.9 Exit gate

- KIS and VQA share one scene aligner and retain existing progressive behavior;
- VQA question remains absent from localization query units;
- TRAKE Top-1 output matches the existing monotonic implementation on all
  focused and hand-calculable fixtures;
- all three pipelines receive the temporal facade from the composition root;
- public API response schemas remain unchanged;
- canonical frame identity is unchanged through each task head;
- no duplicate low-level retrieval or temporal alignment remains in task heads;
- full Plan 01–03 contract suites pass after the migration.

## 9. Cross-plan interfaces

The plans communicate through existing or explicitly promoted contracts:

| Producer | Interface | Consumer |
|---|---|---|
| Plan 01 | tracked deterministic test baseline | Plans 02–04 |
| Plan 02 | structured `VQAInferenceEvidence` | VQA providers and Plan 04 VQA head |
| Plan 03 | frame-aligned `FrameEnrichment.asr_text` | retrieval/data service and VQA evidence |
| Plan 04 | `TemporalQueryPlan` | all task adapters |
| Plan 04 | `SceneCandidate` | KIS and VQA heads |
| Plan 04 | `OrderedPathCandidate` | TRAKE head |

Plan 04 must not redefine the evidence schema produced by Plan 02 or the frame
enrichment schema produced by Plan 03.

## 10. Testing strategy

Every implementation task uses red-green-refactor in small reviewable commits:

```text
write one focused failing test
run it and observe the expected failure
implement the minimum behavior
rerun the focused test
run the affected package/integration suite
commit only the independently reviewable change
```

Test categories:

- contract validation and serialization;
- deterministic query/state transitions;
- hand-calculable temporal fixtures;
- canonical identity preservation;
- provider failure and abstention;
- transcript timing and overlap;
- frontend request/response contracts;
- parity tests before architecture routing changes;
- full tracked suite at every plan gate.

Network calls, real model loading, S3, and corpus-scale artifacts are replaced by
deterministic fakes in unit/integration tests.

## 11. Operational and security rules

- Do not log prompts, credentials, access keys, raw provider failures, or image
  payloads.
- Keep retries bounded and limited to transient idempotent work.
- Use atomic local publication for reusable artifacts.
- Never infer identity from path naming when canonical metadata is available.
- Never make a production frontend outage look like a successful search.
- Never run destructive git cleanup as part of a plan task.
- Offline generation and current artifact deployment are separate activities.

## 12. Definition of Done

This implementation program is complete when:

1. the repository owns a tracked, repeatable validation baseline;
2. browser-to-backend KIS, VQA, and TRAKE contracts are correct;
3. VQA answerability and evidence provenance are enforced;
4. transcript artifacts are timeline-correct, reproducible, optionally
   diarized, and materialized into frame-aligned ASR evidence;
5. KIS, VQA, and TRAKE consume one shared temporal package through pluggable
   evidence providers and aligners;
6. KIS/VQA progressive and TRAKE monotonic semantics remain intact;
7. the full tracked test suite and frontend build pass;
8. deferred evaluator, artifact deployment, and P2 work remain outside the
   implementation diff.

## 13. Planning deliverables after spec approval

After written-spec approval, Superpowers `writing-plans` will create:

```text
docs/superpowers/plans/2026-08-13-hcmai-01-repository-test-baseline.md
docs/superpowers/plans/2026-08-13-hcmai-02-product-contract-correctness.md
docs/superpowers/plans/2026-08-13-hcmai-03-data-evidence-reliability.md
docs/superpowers/plans/2026-08-13-hcmai-04-temporal-progressive-convergence.md
```

Each plan will specify exact files, interfaces, failing tests, commands,
minimal implementations, validation expectations, and frequent commit points.
