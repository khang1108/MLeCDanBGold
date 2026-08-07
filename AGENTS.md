# Agent Instructions

## 1. Current Scope and Ownership

You are working on the **HCMAI / AIC HCMC 2026 Multimodal Video Retrieval**
repository.

This workstream owns:

1. the shared retrieval infrastructure;
2. Textual Known Item Search — KIS;
3. Competition Question Answering / Video Question Answering — Q&A/VQA;
4. API, resilience, caching, observability, evaluation, and integration needed
   by KIS and VQA.

The system must support Vietnamese and English queries.

### Current progress

The repository has completed implementation through:

- S2-T01 — task-pipeline registry;
- S2-T02 — task-specific API contracts;
- S2-T03 — request-scoped traces;
- S2-T04 — batched query encoding;
- S2-T05 — concurrent modality retrieval.

The next active task is:

> **S2-T06 — resilient remote inference gateway**

Do not redo S2-T01 through S2-T05 unless a regression is demonstrated or the
user explicitly asks for rework.

### TRAKE ownership boundary

TRAKE is implemented by a separate teammate.

This agent must not implement, refactor, optimize, benchmark, or review:

- TRAKE query/event parsing;
- per-event posting generation;
- TRAKE candidate-video ranking;
- monotonic, exhaustive, sparse, or other temporal alignment;
- gap or shot-transition penalties;
- original-frame refinement for TRAKE;
- TRAKE k-best path generation;
- TRAKE-specific metrics, datasets, experiments, or ablations.

Do not delete existing TRAKE contracts, routers, pipeline registrations, or
integration seams that have already been merged.

Treat the TRAKE pipeline as an externally owned black box:

```text
TaskRequest
  -> PipelineRegistry
  -> externally owned TRAKEPipeline
  -> TaskResponse / TRAKEResponse
```

Shared-kernel changes must remain backward compatible with public interfaces
used by the TRAKE teammate.

Before changing any shared schema or retrieval contract that may affect their
branch:

1. describe the interface diff;
2. identify the compatibility impact;
3. coordinate the migration with the TRAKE owner;
4. do not edit their algorithm to make the shared change compile.

KISC, conversational KIS, and VKIS are also out of scope unless the user
explicitly restores them.

---

## 2. Mission

### Product mission

Build a reliable competition system that:

- accepts stable KIS and VQA requests;
- reuses one shared multimodal retrieval kernel;
- returns the earliest accurate result possible;
- continues producing ranked alternatives up to Top-100;
- degrades gracefully when optional modalities or remote services fail;
- preserves canonical frame identity through every stage;
- records enough telemetry to diagnose failures and latency;
- remains compatible with the separately owned TRAKE integration boundary.

### Research mission

Establish reproducible baselines for:

- KIS retrieval quality and latency;
- VQA retrieval and correct-video ranking;
- VQA evidence localization;
- VQA answer generation;
- grounded joint video-frame-answer ranking;
- VQA accuracy-latency-cost trade-offs;
- anytime Top-100 generation.

Identify at most two defensible research gaps for potential SoICT work.

Do not promote a research idea into a paper claim without a frozen benchmark,
ablations, and recorded evidence.

### Operational objective

There is no fixed latency threshold.

Optimize:

- official Mean Top-k R-Score at `{1, 5, 20, 50, 100}`;
- task-specific accuracy;
- query-to-first-useful-result;
- time to first correct-video result;
- time to first grounded correct answer when labels exist;
- warm P50/P95 latency;
- operator throughput;
- remote API calls and GPU time per query.

Do not optimize only Recall@1 or Recall@5.

---

## 3. Deployment Assumptions

- Videos, keyframes, metadata, evidence stores, and FAISS indexes run locally.
- Expensive model inference may run:
  - on ThunderCompute;
  - on an L40 or A6000 GPU;
  - through an external API;
  - or through a configurable local backend.
- Models and indexes are loaded once and reused.
- Do not load models at import time or once per request.
- Embedding adapters may lazily load on the first non-empty encode call.
- Unit tests use fake models and tiny fixtures.
- Unit tests must not download checkpoints, load the real corpus, or call live
  remote services.

---

## 4. Source-of-Truth Order

Use this precedence:

1. the user's latest explicit instruction;
2. the latest official AIC HCMC 2026 specification, scorer, or organizer notice;
3. the current repository, tests, artifacts, and active branch;
4. the current approved implementation plan;
5. official AIC 2025 material as historical evidence only;
6. peer-reviewed papers and official implementations;
7. engineering hypotheses requiring validation.

When sources conflict:

1. identify the conflict;
2. state the practical consequences;
3. do not silently choose;
4. ask the user when the decision changes API contracts, scoring semantics,
   corpus assumptions, or research claims.

Do not guess:

- competition rules;
- scorer normalization;
- dataset structure;
- frame sampling policy;
- model capabilities;
- private provider behavior;
- user intent.

---

## 5. Competition Semantics

### 5.1 Textual KIS

Input:

- a natural-language event description.

Output row:

```text
<video_name>,<frame_idx>
```

A row is correct when:

- the video name is correct; and
- `frame_idx` lies inside the accepted interval `[s, e]`.

### 5.2 Competition Q&A / VQA

Input:

- an event description;
- a question about information in that event.

Output row:

```text
<video_name>,<frame_idx>,<answer>
```

Credit requires:

- the correct video;
- a frame inside the accepted interval `[s, e]`;
- the correct answer under the official scorer.

Preserve both:

- raw submitted answer;
- normalized answer used for local evaluation.

For local experiments record:

- exact match;
- normalized exact match;
- configured semantic or alias-based metric;
- correct-video accuracy;
- frame-interval accuracy;
- joint video-frame-answer accuracy.

A plausible answer without supporting evidence is not a valid grounded VQA
result.

### 5.3 Ranking and submission

- A query may contain at most 100 ranked answer rows.
- For `k in {1, 5, 20, 50, 100}`, `R@k` is the maximum row R-Score among the
  first `k` rows.
- Query score is the mean of those five `R@k` values.
- Ranking at all five cutoffs matters.
- Submission video names omit `.mp4` when required by the official exporter.
- Submission rows use canonical integer `frame_idx`.

Never infer `frame_idx` from:

- timestamps;
- FPS;
- filenames;
- array positions;
- keyframe order;
- neighboring frames.

---

## 6. Target Architecture

Build a shared retrieval kernel with thin KIS and VQA pipelines.

```text
HTTP API
    |
    v
SearchService
    |
    v
PipelineRegistry
    |-------------------------|
    v                         v
KISPipeline               VQAPipeline
    |                         |
    +-------------------------+
                |
                v
       Shared Retrieval Kernel
       - normalization
       - controlled expansion
       - batched encoding
       - concurrent modality search
       - fusion and filtering
       - caching
       - video aggregation
       - telemetry
                |
       +--------+---------+
       |                  |
       v                  v
Local data + FAISS    Remote inference gateway
evidence stores       rerank / VLM / parser
```

The same registry may contain an externally owned `TRAKEPipeline`. Treat it as
opaque and do not import its private implementation.

### Architecture rules

- `SearchService` is the online application facade.
- Task-specific logic must not accumulate in one large `if/elif` chain.
- Use explicit pipeline dispatch.
- KIS and VQA reuse the shared retrieval kernel.
- FastAPI routers remain thin.
- Routers must not compose retrieval internals.
- VQA-specific logic stays outside the shared retrieval kernel.
- Shared contracts belong in `src/hcmai/common/schemas/`.
- Do not create parallel contracts for the same concept.
- Cross-component production imports may target:
  - another component's public `pipeline.py`; or
  - authoritative contracts under `common`.
- Do not import another component's private adapters, stores, provider config,
  or implementation modules.

---

## 7. KIS Workflow

```text
text query
  -> validate and normalize
  -> preserve original query
  -> optional controlled query variants
  -> batch encode each unique variant once
  -> search visual/caption/OCR/ASR concurrently
  -> fuse modality and variant evidence
  -> bounded reranking
  -> deterministic fallback to fused ranking
  -> temporal deduplication
  -> video diversity
  -> canonical frame materialization
  -> ranked Top-100
```

Rules:

- Query expansion is controlled, auditable, and configurable.
- Generated variants must not overwhelm the original query.
- Reuse compatible text embeddings across caption, OCR, and ASR.
- Avoid many adjacent duplicate frames.
- Avoid one video occupying the whole Top-100 without strong evidence.
- Preserve per-modality scores, ranks, and provenance.
- Maintain a deterministic golden path for regression testing.
- A failed optional expansion or reranker must not block KIS.

---

## 8. Competition VQA Workflow

```text
event description + question
  -> validate request
  -> decompose retrieval and answer intent
  -> event-aware retrieval branch
  -> question-aware retrieval branch
  -> multimodal fusion
  -> video aggregation
  -> temporal deduplication
  -> bounded evidence-window construction
  -> caption/OCR/ASR/object/visual evidence bundle
  -> question-aware localization
  -> answer multiple shortlisted candidates
  -> deterministic answer normalization
  -> grounded joint ranking
  -> bounded neighbor-window fallback
  -> canonical frame materialization
  -> ranked Top-100
```

A VQA candidate is a grounded object, not only an answer string.

It should contain at least:

```text
video_id
frame_id
frame_idx
timestamp_ms
temporal_window
answer
normalized_answer
retrieval_score
localization_score
answer_confidence
evidence_consistency_score
joint_score
provenance
warnings
```

Rules:

- Retrieve and localize before calling an expensive VLM.
- Never call a VLM over the full corpus.
- Never rank only by answer confidence.
- Every answer remains attached to supporting evidence.
- Answer multiple shortlisted candidates, not only the first frame.
- Keep Localizer and Answerer as separate interfaces.
- Verify that the configured checkpoint supports image or multi-image input.
- Do not treat a generic causal language model as multimodal.
- Use configurable temporal windows.
- Start with a training-free localizer.
- Do not train a selector without a validated internal benchmark.
- Add a text-only blind baseline to detect language-prior leakage.
- Evaluate video, frame, answer, and joint correctness separately.
- A provider may select only frame IDs supplied in its evidence set.
- Reject or degrade a response that invents a frame identity.

---

## 9. Shared Retrieval Requirements

### 9.1 Batched query encoding

The kernel must:

- accept multiple queries and variants;
- deduplicate identical text;
- encode each unique text once per encoder;
- reuse compatible embeddings across indexes;
- preserve query order;
- return explicit text-to-vector mappings;
- record model version, cache status, and latency;
- reject dimension mismatches.

Do not independently encode the same BGE text for caption, OCR, and ASR when
one embedding is compatible with all three indexes.

### 9.2 Concurrent modality search

Visual, caption, OCR, and ASR searches should execute concurrently where safe.

Requirements:

- bounded concurrency;
- request deadline propagation;
- per-modality timeout;
- partial success support;
- successful sources are not discarded because one optional source failed;
- deterministic merge order;
- per-modality status, warning, latency, and result count.

### 9.3 Fusion

Fusion must:

- preserve modality scores and ranks;
- support task-specific weights;
- use configuration rather than hidden constants;
- validate score direction and normalization;
- be deterministic;
- be unit-testable;
- expose provenance.

### 9.4 Filtering and local search

Do not use an unbounded full-index search merely to filter by video or time.

Do not default to:

```python
search_k = index.ntotal
```

for candidate-local refinement.

Use a measured solution such as:

- per-video postings;
- per-video indexes;
- FAISS ID selectors;
- subset vector search;
- sorted mapping ranges;
- exact narrowed fallback.

Preserve exact-search behavior for correctness tests.

### 9.5 Caching

Approved caches include:

- normalized query embeddings;
- repeated query variants;
- thumbnails;
- immutable evidence bundles;
- local frame-window materialization.

Cache keys include relevant:

- encoder/model version;
- index version;
- corpus version;
- normalization version;
- modality;
- normalized query;
- prompt version;
- configuration affecting output.

Define TTL, size limits, and eviction.

Never return an entry from an incompatible model, index, or corpus version.

### 9.6 Reranking fallback

- Use reranked output when successful.
- Fall back to fused ranking on timeout or failure.
- Record warning and error category.
- Do not lose valid retrieval results because an optional reranker failed.
- A reranker may reorder but must not rewrite candidate identity.

---

## 10. Remote Inference Reliability

The following are explicitly approved for remote model and API calls:

- connect timeout;
- read timeout;
- write timeout;
- pool timeout;
- total request deadline;
- bounded retry for transient idempotent failures;
- exponential backoff with jitter;
- circuit breaker;
- bounded concurrency / bulkhead semaphore;
- capability discovery;
- deterministic fallback.

Rules:

- Do not retry deterministic client errors.
- Do not retry a non-idempotent operation without an idempotency strategy.
- Do not begin a retry when too little request deadline remains.
- Do not place unbounded generation or network calls on the critical path.
- Do not turn resilience into a generalized infrastructure framework.
- Scope it to remote inference.
- Record backend, attempts, timeout, fallback, and circuit state.
- A remote provider must not rewrite candidate identity.
- Unit tests use mocked HTTP responses and fake time.

---

## 11. Request-Scoped State and Observability

Do not store request-specific timing, warnings, or intermediate results in
mutable singleton fields.

Use a request-scoped context conceptually similar to:

```python
class RequestContext:
    request_id: str
    task_type: TaskType
    deadline_ms: int | None
    started_at: float
    warnings: list[str]
    trace: PipelineTrace
    cache_policy: CachePolicy
    cancellation_token: CancellationToken | None
```

Each stage records:

- stage name;
- start and end time;
- duration;
- input count;
- output count;
- cache status;
- backend;
- fallback;
- warnings;
- error category.

Trace stages when applicable:

- validation;
- normalization;
- parsing;
- expansion;
- encoding;
- each modality search;
- fusion;
- filtering;
- video aggregation;
- temporal deduplication;
- reranking;
- evidence construction;
- VQA localization;
- VQA answering;
- VQA joint ranking;
- materialization;
- submission export.

Health/readiness may report an externally registered TRAKE capability, but this
agent must not inspect or assert its internal stages.

Logs must be structured.

Do not log:

- API keys;
- credentials;
- private tokens;
- full sensitive prompts;
- private deployment details;
- raw provider secrets.

---

## 12. Canonical Identity and Artifacts

`src/hcmai/common/schemas/` is authoritative.

Preserve:

```text
frame_id -> video_id -> frame_idx
```

through:

- data preparation;
- indexing;
- retrieval;
- fusion;
- reranking;
- VQA localization;
- VQA answering;
- API responses;
- UI display;
- evaluation;
- submission export.

Never infer `frame_idx`.

Expected artifact flow:

```text
frames.parquet
  -> normalized .npy + mapping Parquet
  -> FAISS index
```

YAML and JSON store configuration and provenance, not vector arrays.

Join artifacts on `frame_id`.

Do not commit:

- datasets;
- videos;
- extracted frames;
- weights;
- embeddings;
- indexes;
- run outputs;
- credentials;
- Cloudflare tokens;
- private deployment scripts.

---

## 13. Repository and Folder Awareness

- `src/hcmai/api/routers/`
  - thin FastAPI adapters;
  - no retrieval, answering, or ranking logic.

- `src/hcmai/orchestration/pipeline.py`
  - public `SearchService` facade;
  - explicit task dispatch;
  - delegate to pipelines.

- `src/hcmai/orchestration/setup.py`
  - single composition root;
  - create and connect long-lived services.

- `src/hcmai/data/`
  - dataset preparation;
  - canonical mapping;
  - frame and evidence stores.

- `src/hcmai/common/schemas/`
  - authoritative cross-component contracts.

- `src/hcmai/embedding/`
  - embedding contracts, artifacts, and adapters.

- `src/hcmai/enrichment/`
  - caption, OCR, object, and enrichment adapters.

- `src/hcmai/retriever/`
  - retrieval service, indexes, fusion, filtering, cache, and benchmarks.

- `src/hcmai/reranking/`
  - local and remote scoring adapters.

- `src/hcmai/transcripts/`
  - ASR and diarization stores and adapters.

- `src/hcmai/llm/`
  - private inference service and local/HTTP adapters.

- `src/hcmai/vqa/`
  - VQA parser, candidates, windows, evidence, localization, answering,
    normalization, and joint ranking.

- `src/hcmai/query_suggestions/`
  - controlled expansion providers.

- `src/hcmai/common/utils/`
  - cross-cutting helpers only;
  - no task domain logic.

Existing TRAKE-owned folders may remain. Do not modify them unless the user and
TRAKE owner explicitly assign a shared-interface migration.

### Service boundary convention

Service-owning packages expose one public `pipeline.py` containing a
`*Service` facade.

Concrete providers belong in the owning component's `adapters/`.

`models/` contains contracts, entities, metadata, statistics, and value
objects, not provider implementations.

Do not duplicate service composition in routers, CLIs, notebooks, or task
pipelines.

---

## 14. Subagent Roles

These are responsibility profiles for parallel work, not a requirement to
spawn agents for every task.

### nhuy — Senior SWE, API and Integration

Owns:

- frontend and FastAPI contracts;
- KIS/VQA request and response schemas;
- startup wiring;
- endpoint compatibility;
- API integration tests;
- submission export;
- health and readiness;
- UI -> API -> pipeline -> UI flows.

Primary paths:

- `frontend/`
- `src/hcmai/app.py`
- `src/hcmai/api/routers/`
- `src/hcmai/common/schemas/`
- API integration tests

Must not:

- invent frontend-only fields;
- duplicate schemas;
- put retrieval logic in routers;
- modify TRAKE internals.

### khầy — Senior AI Engineer, VQA and Reranking

Owns:

- VQA query decomposition;
- VQA evidence localization;
- VQA multi-candidate answering;
- VQA grounded joint ranking;
- reranking;
- temporal-window fallback;
- related tests.

Primary paths:

- `src/hcmai/orchestration/`
- `src/hcmai/reranking/`
- `src/hcmai/vqa/`
- related tests

Must:

- consume shared candidate objects;
- preserve exact frame identity;
- avoid generalized agent frameworks;
- not implement or review TRAKE algorithms.

### fuvo — Senior AI Engineer, Retrieval and Enrichment

Owns:

- metadata preparation;
- embeddings;
- FAISS indexing;
- dense retrieval;
- caption/OCR/ASR/object/action enrichment;
- KIS/VQA temporal windows;
- video aggregation;
- fusion inputs;
- filtered retrieval;
- caches;
- retrieval benchmarks.

Primary paths:

- `src/hcmai/data/`
- `src/hcmai/embedding/`
- `src/hcmai/enrichment/`
- `src/hcmai/retriever/`
- `src/hcmai/transcripts/`
- `scripts/`
- `notebooks/`
- related tests

Must:

- produce stable shared candidate outputs;
- record experiments under `runs/`;
- not load the real corpus in unit tests;
- coordinate shared-contract changes with the TRAKE owner.

### Coordination

- Fuvo produces frame, window, and video candidates.
- Khầy may rerank, localize, answer, and refine VQA evidence.
- Nhuy exposes and exports final responses.
- The TRAKE teammate consumes only stable shared contracts and owns TRAKE
  internals.
- Each agent reports files inspected, files changed, tests run, unresolved
  assumptions, and limitations.
- Preserve unrelated edits.
- Never revert another agent's work without explicit approval.

---

## 15. Evidence and Research Discipline

Classify technical claims as:

- **SOURCE**
  - verified from code, tests, artifacts, or official rules.

- **PAPER**
  - explicitly supported by a cited paper or official implementation.

- **PROPOSED**
  - an engineering or research hypothesis requiring experiments.

Do not:

- present a proposal as proven;
- invent benchmark or latency values;
- invent paper claims;
- infer unseen implementation details from an abstract;
- call a component competition-ready because a schema or endpoint exists.

Competition-ready requires:

- an end-to-end path;
- deterministic fallback;
- unit tests;
- integration tests;
- recorded metrics;
- recorded latency;
- failure analysis;
- reproducible configuration.

### Baseline-first VQA rule

1. retrieve candidate frames/videos;
2. build bounded temporal windows;
3. answer one baseline candidate;
4. add question-aware localization;
5. answer multiple candidates;
6. add grounded joint ranking;
7. add bounded temporal fallback;
8. generate Top-100;
9. measure before training a selector.

---

## 16. Evaluation and Reproducibility

Every experiment records:

- task;
- dataset/corpus version;
- query-set version;
- configuration;
- checkpoints;
- index version;
- predictions;
- failures;
- warnings;
- official metrics;
- task metrics;
- P50/P95 latency;
- per-stage latency;
- hardware;
- remote provider;
- remote call count;
- timestamp;
- git commit.

Store experiment records under `runs/`.

No `metrics.json` means no completed experiment.

### KIS metrics

- official Mean Top-k R-Score;
- Recall@1/5/20/50/100;
- MRR;
- accepted-frame accuracy;
- video diversity;
- duplicate-frame rate;
- P50/P95 latency;
- time to first useful result;
- degraded-mode behavior.

### VQA metrics

- official Mean Top-k R-Score;
- correct-video recall;
- frame-interval accuracy;
- raw answer exact match;
- normalized answer match;
- configured semantic/alias metric;
- joint video-frame-answer accuracy;
- grounded accuracy;
- text-only blind baseline;
- P50/P95 latency;
- VLM calls and GPU/API seconds per query.

TRAKE metrics belong to the separate TRAKE workstream.

Use frozen development and test partitions when an internal benchmark exists.

Manual inspection is useful for debugging, but not enough for a publishable
accuracy claim.

---

## 17. DOs and DON'Ts

### DO

- Inspect code, tests, artifacts, and official rules before changes.
- Preserve canonical frame mapping.
- Reuse the shared retrieval kernel.
- Use KIS/VQA-specific contracts and pipelines.
- Batch query encoding.
- Run independent modality retrieval concurrently when safe.
- Use expensive inference only after corpus pruning.
- Keep deterministic fallbacks.
- Preserve candidate identity.
- Add focused tests for public behavior.
- Record benchmark config and failures.
- Preserve unrelated worktree changes.
- Coordinate shared-interface changes with the TRAKE owner.
- Ask the user when a missing decision affects correctness.

### DON'T

- Do not guess.
- Do not implement KISC, conversational KIS, or VKIS.
- Do not implement or modify TRAKE internals.
- Do not route every query through an LLM or VLM.
- Do not call a VLM on the full corpus.
- Do not infer frame indices.
- Do not let rerankers or providers rewrite identity.
- Do not duplicate schemas.
- Do not hide failure with broad silent exceptions.
- Do not add auth, databases, microservices, containers, generalized plugins,
  or premature dependency-injection frameworks without approval.
- Do not download checkpoints or use live services in unit tests.
- Do not commit datasets, weights, indexes, embeddings, credentials, or run
  outputs.
- Do not claim improvement without accuracy and latency evidence.
- Do not rewrite another teammate's files to avoid a shared-contract migration.

---

## 18. Required Execution Protocol

### Step 1 — Inspect repository state

Run:

```bash
git status
git branch --show-current
```

Then:

- identify uncommitted changes;
- identify work by other agents;
- inspect configuration;
- preserve unrelated edits.

### Step 2 — Read before editing

Inspect:

- relevant modules;
- service boundaries;
- call sites;
- authoritative schemas;
- tests;
- configuration;
- documentation;
- duplicate or legacy implementations.

Verify actual behavior.

### Step 3 — Restate the task

Before coding report:

- objective;
- current task ID;
- affected components;
- assumptions;
- ambiguities;
- expected public behavior;
- acceptance criteria;
- likely files.

Ask the user when required behavior cannot be inferred safely.

### Step 4 — Check ownership

Before modifying a file determine whether it is:

- owned by this KIS/VQA workstream;
- shared infrastructure;
- owned by the TRAKE teammate;
- unrelated.

For shared contracts that may affect TRAKE:

1. document the proposed interface change;
2. preserve compatibility where practical;
3. coordinate before merge.

Do not modify TRAKE-owned implementation files.

### Step 5 — Design the smallest coherent change

Specify:

- files to create;
- files to modify;
- interfaces;
- compatibility;
- fallback;
- tests;
- risks.

Prefer a small coherent change over a rewrite.

### Step 6 — Implement

- follow existing style where sound;
- use typed contracts;
- reject invalid input explicitly;
- avoid hidden global state;
- keep VQA logic outside retrieval;
- keep routers thin;
- avoid dead code and fake components;
- do not use silent broad exceptions;
- categorize and trace failures;
- preserve deterministic ordering;
- preserve public compatibility.

### Step 7 — Add tests

At minimum cover:

- success;
- invalid input;
- empty results;
- duplicate candidates;
- optional modality unavailable;
- remote timeout;
- partial failure;
- fallback;
- deterministic ordering;
- boundary conditions;
- existing behavior.

For algorithms also cover:

- hand-computed cases;
- randomized small cases;
- oracle comparison where available;
- regression fixtures.

### Step 8 — Run validation

Compile changed files:

```bash
aic/bin/python -m py_compile src/hcmai/<file>.py
```

Run focused tests:

```bash
PYTHONPATH=src aic/bin/pytest tests/test_<component>.py
```

Run type checking:

```bash
pyright src/hcmai/<file>.py
```

Run broader tests when public behavior changes.

Unit tests must not load real models or corpus data.

### Step 9 — Review diff

Run:

```bash
git diff --check
git diff --stat
git diff
```

Confirm:

- no unrelated changes;
- no secrets;
- no generated artifacts;
- no dataset/model files;
- no accidental contract break;
- no modification to TRAKE-owned algorithm files;
- tests and docs match behavior.

### Step 10 — Report

Report:

- files inspected;
- files created;
- files modified;
- behavior implemented;
- tests and results;
- benchmark results when applicable;
- limitations;
- unresolved assumptions;
- shared-contract impact;
- follow-up work.

Do not hide failed tests.

### Step 11 — Commit and push

Commit only after validation passes.

```bash
git add <explicit-files>
git commit -m "<conventional commit message>"
git push -u origin <approved-feature-branch>
```

Do not use `git add .` when unrelated changes exist.

Do not push directly to a protected branch.

AI-authored commits include:

```text
Co-Authored-By: <model name> <noreply@openai.com>
```

If credentials or permissions are unavailable, stop after the local commit and
report the blocker.

---

## 19. Package Managers

### Python

Use:

```bash
aic/bin/python -m pip install -e ".[embedding,dev]"
```

Add runtime dependencies to `pyproject.toml`.

Before adding a dependency explain:

- why it is required;
- runtime scope;
- test impact;
- whether an existing dependency already covers it.

### Frontend

Use `npm` in:

```text
frontend/
```

Preserve the React application and API boundary.

---

## 20. Code-Quality Guidelines

Prefer:

- module under 200 lines;
- function under 40 lines;
- focused change under 300 lines;
- explicit interfaces;
- small fixtures;
- one responsibility per module.

These are guidelines, not reasons to fragment cohesive code.

Exceeding a guideline is allowed when splitting damages cohesion. Explain the
exception in the task report.

Do not create:

- a base class without two implementations;
- a factory without a concrete need;
- a plugin system for one provider;
- a generic framework for one task;
- a new service boundary without ownership and tests.

---

## 21. Final Working Principle

For every active KIS/VQA task:

1. make the baseline executable;
2. make it correct;
3. make it observable;
4. make it reproducible;
5. measure it;
6. optimize the measured bottleneck;
7. preserve compatibility;
8. document source-derived, paper-derived, and proposed decisions.

When information is missing and materially affects correctness:

> Stop and ask the user. Do not guess.
