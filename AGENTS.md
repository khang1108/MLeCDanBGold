# HCMAI Coding Agent Instructions

This file is the primary repository-level guardrail for agentic coding on the
**HCMAI / AIC HCMC 2026 Multimodal Video Retrieval** project.

The purpose of these instructions is to keep coding agents aligned with the
actual competition objective, the current repository, and the approved KIS/VQA
optimization roadmap. Do not infer a different mission from a paper, an old
TODO, a stale README sentence, or an unused module.

---

## 1. Mission and active scope

This workstream owns:

1. shared multimodal retrieval infrastructure used by KIS and VQA;
2. Textual Known Item Search — KIS;
3. Competition Q&A / Video Question Answering — Q&A/VQA;
4. KIS/VQA reranking, localization, evidence selection, answering, ranking;
5. resilience, caching, observability, evaluation, and integration required by
   KIS/VQA.

The system must support Vietnamese and English queries.

### Product objective

Build a competition system that:

- returns correct canonical video/frame identities as early as possible;
- produces ranked alternatives up to Top-100;
- grounds every VQA answer in retrieved video evidence;
- preserves useful OCR/ASR/visual/caption evidence through reranking;
- handles temporal VQA with bounded, ordered visual context;
- degrades deterministically when optional modalities or remote inference fail;
- records enough telemetry to explain both quality and latency failures;
- remains backward compatible with shared interfaces used by the TRAKE owner.

### Current optimization program

The approved optimization program is defined in:

```text
KIS_VQA_V2_PLAN.md
```

Do **not** use an old hard-coded "next task" such as `S2-T06` as the current
mission. Before coding, determine the active task from:

1. the user's latest instruction;
2. the current branch/repository state;
3. `KIS_VQA_V2_PLAN.md` and its dependency order.

If the user does not name a task, inspect the repository and choose the first
approved task whose dependencies are complete and whose behavior is not already
implemented. Report that choice before editing.

### Priority order

Unless the user explicitly changes priorities:

```text
P0 measurement/correctness
    -> P0 KIS/VQA architecture fixes
    -> P1 quality and latency
    -> P2 research/model experiments
```

Do not skip P0 in order to replace models or add a research architecture.

---

## 2. Hard ownership boundaries

### TRAKE is externally owned

TRAKE is implemented by a separate teammate.

This agent must not implement, refactor, optimize, benchmark, or review:

- TRAKE query/event parsing;
- TRAKE per-event posting generation;
- candidate-video ranking specific to TRAKE;
- monotonic/exhaustive/sparse temporal alignment for TRAKE;
- TRAKE gap/shot-transition penalties;
- TRAKE original-frame refinement;
- TRAKE k-best path generation;
- TRAKE-specific evaluation or ablations.

Do not delete or rewrite existing TRAKE contracts, routers, registrations,
interfaces, tests, or integration seams.

Treat TRAKE as an opaque pipeline:

```text
TaskRequest
  -> PipelineRegistry
  -> externally owned TRAKEPipeline
  -> TaskResponse / TRAKEResponse
```

Shared-kernel changes must remain backward compatible when practical.

Before changing a shared contract that may affect TRAKE:

1. describe the interface diff;
2. identify compatibility impact;
3. preserve compatibility if reasonable;
4. coordinate migration with the TRAKE owner before merge;
5. never edit TRAKE internals merely to make the shared change compile.

### Other out-of-scope areas

KISC, conversational KIS, and VKIS are out of scope unless explicitly restored
by the user.

---

## 3. Source-of-truth order

Use this precedence:

1. user's latest explicit instruction;
2. latest official AIC HCMC 2026 specification/scorer/organizer notice;
3. current repository, tests, artifacts, configuration, and active branch;
4. `KIS_VQA_V2_PLAN.md`;
5. `README.md`;
6. official AIC 2025 material as historical evidence only;
7. peer-reviewed papers and official implementations;
8. engineering hypotheses requiring validation.

When two sources conflict:

1. identify the conflict explicitly;
2. state what behavior differs;
3. do not silently choose based on preference;
4. ask the user only when the unresolved decision materially changes public
   contracts, competition semantics, corpus assumptions, or research claims.

### Never guess

Do not guess:

- competition rules or scoring normalization;
- data layout or frame sampling policy;
- whether an index/artifact exists;
- whether a module is used in the active online path;
- model multimodal or multi-image capabilities;
- provider timeout/retry behavior;
- current config values when configuration can override code defaults;
- `frame_idx` from timestamps/FPS/filenames/array positions;
- performance or quality improvements without measurements.

Inspect before asserting.

---

## 4. Evidence discipline: SOURCE / PAPER / PROPOSED

Every non-trivial technical claim should mentally belong to one of these
categories.

### SOURCE

Verified from current:

- code;
- tests;
- artifacts;
- configuration;
- official competition rules;
- measured experiment output.

Example:

```text
SOURCE: current VQA orchestration calls answer_vqa(...) with one image.
```

Only call something a current bug or current behavior after verifying the
active branch.

### PAPER

Explicitly supported by a paper or official implementation.

Examples of research directions approved as literature grounding:

- QD-DETR: query-dependent video relevance/representation;
- CG-DETR: query-conditioned calibration of relevant clips;
- ChatVTG: coarse-to-fine temporal localization using multi-granularity video
  descriptions;
- NumPro: explicit numbered/ordered frame identity for temporal grounding;
- VideoQA-TA: explicit temporal information and temporal-aware aggregation for
  VideoQA.

A paper result is **not** evidence that the adaptation improves HCMAI.

### PROPOSED

An engineering/research hypothesis that must be tested on HCMAI.

Examples:

- dynamic modality routing will improve OCR KIS;
- second-stage rank fusion will preserve ASR evidence better than Qwen score
  replacement;
- BGE semantic localization will beat lexical overlap;
- four ordered frames will improve temporal VQA over one frame.

Never describe PROPOSED behavior as proven until the frozen benchmark and
ablation support it.

---

## 5. Competition semantics

### 5.1 Textual KIS

Input:

- natural-language event description.

Output row:

```text
<video_name>,<frame_idx>
```

A row is correct when the official scorer accepts both the video and frame
interval.

### 5.2 Competition VQA

Input:

- event description;
- question about that event.

Output row:

```text
<video_name>,<frame_idx>,<answer>
```

A grounded VQA result must keep:

- canonical video/frame identity;
- raw answer;
- normalized answer for local evaluation;
- evidence/provenance required to explain the answer.

A plausible answer without supporting accepted video/frame evidence is not a
valid grounded result.

### 5.3 Ranking

When official Top-k scoring is available, optimize all official cutoffs rather
than only Top-1.

Preserve canonical integer `frame_idx` from authoritative metadata.

Never infer `frame_idx` from:

- timestamp;
- FPS;
- filename;
- array index;
- keyframe order;
- neighboring frame identity.

---

## 6. Current model strategy

The current stack uses model families such as:

- Florence-2 for frame captioning;
- SigLIP2 for visual retrieval;
- BGE-M3 for caption/OCR/ASR text embeddings;
- Qwen3-VL reranking;
- Qwen2.5-VL VQA.

Configuration is authoritative for the exact checkpoints.

### Model-replacement rule

Do not replace the model stack as a first response to poor quality.

Before proposing a replacement, isolate the failure stage:

```text
retrieval
  -> video selection
  -> temporal localization
  -> frame/evidence selection
  -> VLM reasoning
  -> final ranking
```

Examples:

- high correct-video recall + low window recall => localization problem;
- high correct-window recall + low answer accuracy => evidence/VLM problem;
- poor OCR-only queries after image reranking => reranking/fusion problem.

Alternative backbones are P2 experiments unless a verified blocker requires
otherwise.

---

## 7. Target architecture

Build one shared retrieval kernel with thin task-specific KIS and VQA
orchestration.

```text
HTTP API
   |
   v
SearchService
   |
   v
PipelineRegistry
   |-------------------------------|
   v                               v
KISPipeline                     VQAPipeline
   |                               |
   +---------------+---------------+
                   |
                   v
              QueryPlanner
              - intent
              - modalities
              - temporal need
              - controlled subqueries
                   |
                   v
         Shared Retrieval Kernel
         - normalization
         - batched encoding
         - concurrent modality search
         - query-conditioned fusion
         - provenance
         - caching / telemetry
                   |
         +---------+----------+
         |                    |
         v                    v
       KIS V2                VQA V2
   evidence-preserving     video aggregation
   reranking               temporal peaks
   second fusion           bounded windows
   shot-aware NMS          frame selection
                           frame-bound evidence
                           semantic localization
                           adaptive single/multi-frame VLM
                           grounded joint ranking
```

### Architecture rules

- `SearchService` remains the online application facade.
- Use explicit task pipeline dispatch.
- FastAPI routers remain thin.
- KIS and VQA reuse the shared retrieval kernel.
- VQA-specific logic must stay outside generic retrieval internals.
- Shared contracts belong in `src/hcmai/common/schemas/`.
- Do not create parallel contracts for an existing concept.
- Cross-component production imports target a public service `pipeline.py` or
  authoritative `common` contracts.
- Do not import another component's private adapter/store/config internals.
- Rerankers/providers may reorder or score candidates but must not mutate
  canonical identity.

---

## 8. Query planning rules

V2 introduces a query-planning concept, but it is not a generalized agent.

The planner may classify/query-plan:

```text
VISUAL
OCR
SPEECH
MIXED
TEMPORAL
```

For VQA it may additionally classify answer type:

```text
COLOR
COUNT
OCR
SPEECH
IDENTITY
OBJECT
ACTION
TEMPORAL
CAUSAL
GENERAL
```

### Planner constraints

- deterministic/rule-based baselines are acceptable and preferred first;
- generated subqueries must be bounded and auditable;
- original query/event description must be preserved;
- do not allow an LLM to invent arbitrary production fusion weights;
- map intent to configured retrieval policies;
- do not route every query through an LLM;
- log planner output and prompt/model version when an LLM is used.

---

## 9. Shared retrieval requirements

### 9.1 Batched encoding

The retrieval kernel must:

- deduplicate identical query text;
- encode each unique text once per compatible encoder;
- reuse BGE embeddings across caption/OCR/ASR when compatible;
- preserve query order;
- reject vector-dimension mismatches;
- record encoder/version/cache/latency metadata.

### 9.2 Concurrent modality search

Visual, caption, OCR, and ASR search should run concurrently where safe.

Support:

- bounded concurrency;
- deadline propagation;
- per-source timeout;
- partial success;
- deterministic merging;
- per-source warnings/latency/result counts.

Do not discard successful sources because one optional source failed.

### 9.3 Query-conditioned fusion

Fusion must preserve:

- source ranks;
- source scores where available;
- query branch;
- modality provenance;
- final fused rank.

Start from query-aware weighted RRF because embedding score spaces are not
assumed calibrated.

Raw-score fusion requires explicit calibration experiments.

Do not hide fusion constants inside task code.

### 9.4 Local filtering

Do not use unbounded full-index scans merely to filter by video/time.

Prefer measured solutions such as:

- per-video postings/ranges;
- FAISS ID selectors;
- subset-vector search;
- exact narrowed fallback.

Keep exact behavior for correctness tests.

---

## 10. KIS V2 rules

Target flow:

```text
query
  -> validate/normalize
  -> QueryPlan
  -> controlled branches/variants
  -> batch encode
  -> concurrent visual/caption/OCR/ASR retrieval
  -> query-conditioned RRF
  -> preserve candidate provenance
  -> bounded/gated reranking
  -> second-stage rank fusion
  -> shot-aware dedup/diversity
  -> canonical Top-100
```

### KIS invariants

- query expansion must not overwhelm the original query;
- OCR/speech evidence must survive an image-only reranker when relevant;
- reranking failure falls back to fused retrieval;
- reranking must not change frame/video identity;
- avoid one video monopolizing Top-100 without strong evidence;
- avoid time-only duplicate suppression when shots are distinct;
- preserve a deterministic golden path for regression testing.

### KIS current-audit hypotheses to verify before editing

The approved plan is based on prior code audit that observed potential issues
including:

- static/equal modality weighting;
- image-only reranker replacing fused retrieval score;
- fixed rerank depth;
- fixed time-window deduplication.

These are **not license to edit blindly**. Verify the active branch and tests
before each change because a teammate may already have addressed them.

---

## 11. VQA V2 rules

Target flow:

```text
event description + question
  -> validate
  -> VQA QueryPlan
  -> event/contextual question branches
  -> multimodal retrieval
  -> query-aware candidate merge
  -> video aggregation
  -> local temporal peak selection
  -> fixed bounded windows
  -> question-aware frame selection
  -> frame-bound caption/OCR/ASR evidence
  -> semantic localization
  -> adaptive single/multi-frame answering
  -> answer normalization
  -> confidence-aware grounded joint ranking
  -> bounded fallback when evidence is insufficient
  -> canonical Top-100
```

### VQA candidate contract

A grounded candidate should retain at least:

```text
video_id
frame_id / selected_frame_ids
frame_idx
timestamp_ms
temporal_window
answer
normalized_answer
retrieval provenance
video score
localization score
answerability
answer confidence/system confidence
joint score
evidence IDs
warnings
```

Use existing authoritative schemas where possible rather than creating a
parallel contract.

### VQA invariants

- retrieve/localize before expensive VLM calls;
- never run VLM reasoning over the full corpus;
- never rank only by answer confidence;
- every answer remains attached to evidence;
- preserve event description when answering a question with pronouns/context;
- temporal windows must remain bounded by configured policy;
- selected frames are chronological when passed to a temporal VLM prompt;
- evidence remains associated with frame ID and timestamp;
- a provider may only select frame IDs supplied to it;
- reject/degrade responses that invent frame identity;
- verify multi-image capability before sending multiple images;
- maintain a text-only blind baseline to detect language-prior leakage.

### VQA current-audit hypotheses to verify before editing

Prior code audit found potential issues such as:

- required OCR/ASR boost computed but not used in primary candidate score;
- video aggregation heuristic terms on incompatible scales;
- transitive overlap merging creating oversized windows;
- earliest-frame truncation after window merging;
- lexical localization fragile across Vietnamese queries / English captions;
- flattened evidence losing frame/timestamp association;
- multi-frame capability existing while orchestration may still answer from one
  selected frame;
- weak/hard-coded answerability/confidence behavior.

Verify current behavior before implementing each planned fix.

---

## 12. Bounded temporal-window rules

A configured window size is a hard semantic bound unless the user explicitly
approves another policy.

For a window profile with duration `W`:

```text
end_ms - start_ms <= W
```

must remain true after window construction.

Do not use transitive overlap merging that silently turns multiple local
windows into one long segment.

Prefer:

1. relevance peak selection;
2. fixed window around the peak;
3. temporal NMS among windows;
4. question-aware frame sampling within each bounded window.

Frame selection should preserve:

- at least one retrieval/evidence anchor;
- high query relevance;
- required modality evidence;
- temporal diversity before/after when useful;
- chronological ordering in the final VLM input.

---

## 13. Evidence and localization rules

Evidence is structured, not a bag of concatenated strings.

Conceptually prefer:

```text
FrameEvidence
  frame_id
  timestamp_ms
  caption
  OCR
  ASR / transcript interval
  source/provenance scores
```

Do not flatten all captions/OCR/ASR in a way that makes temporal ownership
ambiguous before VLM reasoning.

### Semantic localization

Lexical overlap may be retained as a diagnostic feature, but should not be the
only semantic localizer for multilingual queries.

BGE-M3 or another configured compatible text encoder may be used for a
training-free semantic-localization baseline.

Any combined localizer score must have documented/calibrated feature scales or
use rank fusion.

---

## 14. Adaptive VQA compute

Do not give every question the same image/VLM budget.

Reasonable baseline policy classes:

```text
OCR/COLOR        -> 1-2 strong frames
COUNT            -> 1-3 frames when needed
SPEECH           -> 1-2 frames + aligned ASR evidence
IDENTITY/ACTION  -> 2-4 frames
GENERAL          -> 2-4 frames
TEMPORAL/CAUSAL  -> ordered multi-frame context, e.g. 4-8 within budget
```

Exact budgets are configuration/benchmark decisions, not immutable constants.

Use bounded fallback when:

- retrieval/localizer margin is low;
- answerer reports unanswerable;
- independent evidence disagrees;
- the required modality is absent;
- the selected window lacks adequate coverage.

Do not trigger fallback from a fake constant confidence value.

---

## 15. Evaluation gates

No optimization is complete without a reproducible experiment.

### Gate A — Retrieval

KIS:

- official Mean Top-k R-Score where available;
- Recall/Hit@1/5/20/50/100;
- MRR;
- category breakdown: visual/OCR/speech/mixed/temporal/hard-negative.

VQA:

- correct-video Recall@K.

### Gate B — Localization

VQA:

- correct-window Recall@1/3/8;
- selected-frame/evidence Recall@budget.

### Gate C — Reasoning

Run oracle-evidence/oracle-window evaluation:

```text
answer accuracy | correct evidence
```

This separates retrieval/localization failures from VLM reasoning failures.

### Gate D — End-to-end

Record:

- official task metric where available;
- joint video-frame-answer accuracy for VQA;
- warm P50/P95;
- time to first useful/correct result when measurable;
- remote calls per query;
- VLM calls/images per query;
- GPU/API seconds per query.

### Experiment record

Every completed experiment records:

- task;
- corpus/query-set version;
- config;
- checkpoints;
- index version;
- predictions;
- failures/warnings;
- metrics;
- per-stage latency;
- hardware/provider;
- git commit;
- timestamp.

Store records under `runs/`.

No recorded metrics means no verified improvement.

---

## 16. Performance optimization rules

Optimize measured bottlenecks only.

Approved P1/P2 candidates include:

- adaptive KIS rerank depth;
- thumbnail/image decode cache;
- immutable evidence cache;
- query embedding cache keyed by model/index/corpus/config version;
- BF16/fp16 inference benchmark where numerically safe;
- FAISS HNSW/IVF/GPU benchmark at large corpus scale;
- local subset search rather than full-index filtering.

Do not trade away accepted-frame recall merely to reduce latency without
showing the quality/latency frontier.

---

## 17. Remote inference reliability

Approved mechanisms for remote inference:

- connect/read/write/pool timeout;
- total request deadline;
- bounded retry for transient idempotent failures;
- exponential backoff with jitter;
- circuit breaker;
- bounded concurrency/bulkhead semaphore;
- capability discovery;
- deterministic fallback.

Rules:

- do not retry deterministic client errors;
- do not retry non-idempotent operations without an idempotency strategy;
- do not begin a retry when insufficient deadline remains;
- do not place unbounded generation on the critical path;
- record backend, attempts, timeout, fallback, and failure category;
- unit tests use mocked HTTP/fake time;
- remote providers must not rewrite canonical identity.

---

## 18. Request-scoped state and observability

Do not store request-specific intermediate state in mutable service singletons.

Use request-scoped tracing/context.

Trace stages when applicable:

```text
validation
normalization
query planning
expansion/subquery generation
encoding
visual search
caption search
OCR search
ASR search
fusion
KIS reranking
second fusion
video aggregation
temporal peak selection
window construction
frame selection
evidence construction
semantic localization
VQA answering
VQA joint ranking
fallback
materialization
submission export
```

Each stage should be able to report:

- duration;
- input/output count;
- backend;
- cache status;
- fallback;
- warnings/error category.

Do not log secrets, credentials, private tokens, or sensitive full prompts.

---

## 19. Canonical identity and artifacts

`src/hcmai/common/schemas/` is authoritative for shared contracts.

Preserve:

```text
frame_id -> video_id -> frame_idx
```

through preparation, indexing, retrieval, fusion, reranking, localization,
answering, API, evaluation, and export.

Artifact flow should remain explicit and versioned.

Embedding generation and index persistence are offline batch operations run on
a remote GPU VM. The local KIS, TRAKE, and VQA serving process treats deployed
retrieval artifacts as immutable and read-only: it may validate and memory-map
them, but it must never generate, reconstruct, migrate, or save embeddings at
startup or during a request. Missing or inconsistent files make the affected
retrieval capability unavailable until the complete versioned bundle is rebuilt
or resynchronized from the offline pipeline.

Do not commit:

- datasets/videos/extracted frames;
- model weights;
- embeddings/indexes;
- run outputs unless repository policy explicitly allows selected summaries;
- credentials/tokens;
- private deployment scripts.

---

## 20. Repository ownership awareness

Typical ownership:

- `src/hcmai/api/routers/`: thin transport only;
- `src/hcmai/orchestration/`: task dispatch/composition;
- `src/hcmai/data/`: canonical mapping/evidence stores;
- `src/hcmai/common/schemas/`: authoritative shared contracts;
- `src/hcmai/embedding/`: embedding contracts/adapters;
- `src/hcmai/enrichment/`: caption/OCR/object enrichment;
- `src/hcmai/retriever/`: search/index/fusion/cache/evaluation;
- `src/hcmai/reranking/`: reranking service/adapters;
- `src/hcmai/transcripts/`: ASR/diarization;
- `src/hcmai/llm/`: inference service/adapters;
- `src/hcmai/vqa/`: VQA candidates/windows/evidence/localization/answering;
- `src/hcmai/common/utils/`: cross-cutting helpers only.

Existing TRAKE-owned folders are out of scope.

Service-owning packages expose a public `pipeline.py` facade. Do not duplicate
service composition in routers, CLIs, notebooks, or task pipelines.

---

## 21. Team responsibility profiles

These roles are coordination profiles, not an instruction to spawn subagents.

### nhuy — API / integration

Owns:

- frontend/API contracts;
- startup wiring;
- endpoint compatibility;
- integration tests;
- submission export;
- health/readiness.

Must not place retrieval/ranking logic in routers or modify TRAKE internals.

### khầy — VQA / reranking

Owns:

- VQA query decomposition/planning;
- evidence localization;
- multi-frame/multi-candidate answering;
- grounded joint ranking;
- reranking;
- bounded fallback;
- related tests.

Must preserve exact frame identity and shared contracts.

### fuvo — retrieval / enrichment

Owns:

- metadata preparation;
- embeddings and FAISS;
- dense retrieval;
- caption/OCR/ASR/object enrichment;
- shared fusion inputs;
- video aggregation/local retrieval support;
- caches and retrieval benchmarks.

Shared changes that affect TRAKE require coordination.

### Coordination rule

Each agent reports:

- files inspected;
- files changed;
- tests run/results;
- benchmark results when applicable;
- assumptions/limitations;
- shared-interface impact.

Preserve unrelated edits and never revert another teammate's work without
explicit approval.

---

## 22. Required execution protocol

### Step 1 — inspect repository state

Run:

```bash
git status
git branch --show-current
```

Identify:

- uncommitted work;
- teammate/agent edits;
- active config;
- current plan status.

### Step 2 — read before editing

Inspect:

1. `README.md`;
2. `KIS_VQA_V2_PLAN.md`;
3. relevant modules/call sites;
4. shared schemas;
5. tests;
6. configuration;
7. any duplicate/legacy implementation.

Verify the actual online path. A module existing in the repository does not
prove it is used.

### Step 3 — restate the task

Before coding report briefly:

- objective/task ID;
- current verified behavior;
- components/files likely affected;
- dependencies already satisfied;
- acceptance criteria;
- assumptions/ambiguities.

Do not repeat a question already answered by the user or repository.

### Step 4 — ownership check

Classify each target file as:

- KIS/VQA owned;
- shared infrastructure;
- TRAKE owned;
- unrelated.

Do not edit TRAKE-owned implementation files.

### Step 5 — smallest coherent design

Specify:

- files to create/modify;
- public interfaces;
- compatibility/fallback;
- tests;
- experiment required;
- risks.

Prefer one coherent task from the roadmap rather than a broad rewrite.

### Step 6 — implement

- follow existing style where sound;
- use typed contracts;
- keep routers thin;
- keep VQA domain logic outside generic retrieval;
- avoid hidden mutable global state;
- preserve deterministic ordering;
- reject invalid input explicitly;
- categorize failures;
- avoid broad silent exceptions;
- do not leave fake/dead components.

### Step 7 — test

At minimum cover relevant cases among:

- success;
- invalid/empty input;
- no results;
- duplicate candidates;
- optional modality unavailable;
- remote timeout/partial failure;
- deterministic fallback/order;
- boundary conditions;
- regression behavior.

For scoring/window algorithms include hand-computed fixtures and oracle
comparison when feasible.

Unit tests must not download checkpoints, load the real corpus, or call live
remote services.

### Step 8 — validate

Use the repository environment, for example:

```bash
aic/bin/python -m py_compile src/hcmai/<changed>.py
PYTHONPATH=src aic/bin/pytest tests/test_<component>.py
pyright src/hcmai/<changed>.py
```

Run broader integration/regression tests when public behavior changes.

### Step 9 — benchmark when task requires it

For optimization tasks, run the matching frozen evaluation/ablation from
`KIS_VQA_V2_PLAN.md`.

Do not claim quality/latency improvement from unit tests alone.

### Step 10 — review diff

Run:

```bash
git diff --check
git diff --stat
git diff
```

Confirm:

- no unrelated changes;
- no secrets/generated corpus/model artifacts;
- no accidental shared-contract break;
- no TRAKE algorithm edits;
- docs/tests match behavior.

### Step 11 — report

Report:

- task ID;
- files inspected/changed;
- behavior implemented;
- tests and results;
- benchmark results if applicable;
- limitations/unresolved assumptions;
- shared-interface impact;
- recommended next dependency-ready task.

Do not hide failed tests.

### Step 12 — commit/push only when requested/approved

Use explicit files rather than `git add .` when unrelated work exists.

Do not push directly to a protected branch.

---

## 23. Code-quality guidelines

Prefer:

- focused modules/functions;
- explicit interfaces;
- small fixtures;
- deterministic behavior;
- one responsibility per module;
- configuration over hidden constants.

Do not create:

- a generic agent framework for query planning;
- a base class with no real alternate implementation;
- a plugin/factory system without concrete need;
- a new microservice merely to separate a small module;
- new dependencies when existing libraries cover the need.

Before adding a dependency document:

- why it is required;
- runtime scope;
- test impact;
- why current dependencies are insufficient.

---

## 24. Final working principle

For every KIS/VQA change:

1. verify current behavior;
2. make the baseline correct;
3. keep evidence and identity intact;
4. make the behavior observable;
5. make experiments reproducible;
6. measure the correct stage;
7. optimize the measured bottleneck;
8. preserve shared compatibility;
9. separate SOURCE, PAPER, and PROPOSED claims;
10. do not optimize a paper idea instead of the competition system.

When information is genuinely missing and materially affects correctness or a
public interface, ask the user. Otherwise, use the repository and approved plan
to make the smallest defensible implementation decision.
