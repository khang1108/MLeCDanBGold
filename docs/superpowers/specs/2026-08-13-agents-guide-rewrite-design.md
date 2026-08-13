# HCMAI Root AGENTS.md Rewrite Design

> **Status:** Approved design awaiting written-spec review
> **Date:** 2026-08-13
> **Target:** `/AGENTS.md`

## 1. Goal

Rewrite the repository-root `AGENTS.md` as a concise, self-contained orientation
and guardrail for agentic coding tools working on HCMAI. The guide must explain
the project mission, competition semantics, active and target architecture,
repository ownership, correctness invariants, implementation-program context,
and the required coding/verification workflow.

The rewrite replaces the current 1,553-line guide. It preserves durable rules,
removes repeated explanations, and adds the approved shared temporal architecture
and current implementation-program boundaries.

## 2. Audience and language

The primary audience is an automated coding agent with no prior HCMAI context
but with access to the repository. The guide will be written in direct technical
English so it works consistently across agentic coding tools. Vietnamese query
and competition examples may be retained only where they clarify task semantics.

The guide should be scannable and generally remain within 700–900 lines. This is
a content budget, not a reason to omit a correctness-critical invariant.

## 3. Authority and document roles

The guide will define this decision precedence:

1. latest explicit user instruction;
2. latest official competition rule/scorer/organizer notice;
3. active code, tests, configuration, and deployed artifact contract;
4. root `AGENTS.md`;
5. `docs/HCMAI_SYSTEM_ARCHITECTURE_STATUS_AND_ROADMAP.md`;
6. approved Superpowers specifications and implementation plans;
7. papers and engineering hypotheses.

Document roles will be explicit:

- `AGENTS.md` — durable repository orientation and guardrails;
- consolidated roadmap — current architecture/status and deferred roadmap;
- Superpowers specification — approved design boundary;
- Superpowers plan — executable TDD task sequence.

The guide will state that active code must be traced before editing because the
shared temporal migration can be partially complete.

## 4. Content structure

### 4.1 Project mission and mental model

Explain HCMAI as a multimodal long-video retrieval and reasoning system using
visual, caption, OCR, ASR, motion, temporal, and bounded model evidence.

Define the core abstraction:

```text
query units -> canonical frame evidence -> temporal scene/path -> thin task head
```

Distinguish a retrieval hit from a coherent scene/path.

### 4.2 Competition task semantics

Keep concise but complete sections for:

- KIS cumulative snapshots and multi-frame scene evidence;
- VQA separation of localization hints from the answer question;
- TRAKE hard ordered-event paths.

The guide will explicitly forbid forcing TRAKE through soft KIS/VQA scene
semantics.

### 4.3 Current and target architecture

Describe the current shared retrieval stack and approved migration target:

```text
Task adapter
  -> TemporalQueryPlan
  -> ProgressiveEvidenceProvider | OrderedEvidenceProvider
  -> SceneAligner | OrderedPathAligner
  -> SceneCandidate | OrderedPathCandidate
  -> KIS | VQA | TRAKE head
```

KIS/VQA share progressive sparse evidence and scene alignment. TRAKE shares
query/evidence/orchestration contracts but retains dense ordered evidence and a
compatibility wrapper around the existing monotonic DP.

Progressive state applies to cumulative KIS/VQA input. TRAKE remains stateless
until an official progressive TRAKE contract exists.

### 4.4 Repository map and ownership

Document the current responsibilities of:

- `src/hcmai/api`;
- `src/hcmai/common` and `common/schemas`;
- `src/hcmai/common/observability`;
- `src/hcmai/data` and preprocessing/enrichment;
- `src/hcmai/retrieval` and reranking;
- `src/hcmai/temporal`;
- `src/hcmai/orchestration`;
- `src/hcmai/pipelines/vqa`;
- `src/hcmai/pipelines/trake`;
- `src/hcmai/llm`;
- `frontend`;
- `tests`, `configs`, `scripts`, `artifacts`, and `docs/superpowers`.

The map will describe responsibilities rather than promise that every target
file already exists.

### 4.5 Non-negotiable contracts

Preserve these rules:

- `FrameRecord` is canonical identity;
- decode, media-time, and organizer coordinates are distinct;
- canonical IDs survive every pipeline stage;
- missing evidence is not negative evidence;
- progressive state commits only after success;
- task/filter/session context cannot mutate within a progressive search;
- reveal order is not event order;
- provider/reranker output cannot invent identity;
- offline artifacts are immutable and never rebuilt online;
- new contracts require an immediate consumer and reuse search;
- task heads own only task-specific output behavior.

### 4.6 Runtime and offline flows

Include small text or Mermaid flows for:

- KIS progressive scene localization;
- VQA localization then evidence-conditioned answering;
- TRAKE dense scoring then monotonic alignment;
- raw video through canonical frames/enrichment/indexes;
- API through orchestration, retrieval, temporal result, and materialization.

### 4.7 Current implementation program

Summarize the approved order:

1. repository and test baseline;
2. task/API/frontend contract correctness;
3. data/evidence reliability;
4. temporal/progressive convergence.

Mark corpus/index deployment, evaluator construction, P2 experiments, cloud/S3
execution, and distributed state as deferred. The guide will link to the
approved program spec rather than repeat its task-level details.

### 4.8 Agent working protocol

Require every non-trivial change to inspect:

1. the active public entry point and real runtime path;
2. relevant schemas and configuration;
3. nearby tests;
4. overlapping contracts/services;
5. cross-task and canonical-identity impact;
6. offline artifact compatibility;
7. appropriate verification scope.

Require preservation of unrelated dirty-worktree changes. Agents may not reset,
checkout, delete, or rewrite unrelated work to create a clean state.

Use TDD for behavior changes, focused modules, configuration for tunables,
bounded failure handling, and small independently reviewable commits when
commits are authorized.

### 4.9 Validation and completion reporting

List repository commands using the active `aic` environment:

```bash
PYTHONPATH=src aic/bin/python -m pytest -q
CI=true npm test -- --runInBand
npm run build
git diff --check
```

Frontend commands run from `frontend/`. Agents should run focused tests first,
then the largest proportionate suite. Tests must mock external networks/models
unless an integration task explicitly authorizes them.

Completion reports must state files changed, behavior before/after,
compatibility impact, verification results, known limitations, assumptions, and
whether shared temporal/TRAKE behavior changed.

### 4.10 Compact Do/Don't rules

Retain high-value prohibitions and remove repeated wording. The final rules must
cover identity, evidence provenance, cumulative hints, temporal ordering,
bounded compute, provider safety, artifact immutability, reuse-before-create,
measured claims, thin routers, and preservation of teammate work.

## 5. Information intentionally excluded

The root guide will not contain:

- volatile test pass counts;
- current artifact row counts or missing-file inventories;
- sprint ticket details;
- code-level implementation steps from Superpowers plans;
- unverified performance claims;
- long paper surveys;
- credentials, local machine details, or private service diagnostics;
- duplicated copies of roadmap tables.

These exclusions keep the guide durable and reduce agent context overhead.

## 6. Consistency and maintenance rules

The rewritten guide must not contradict the consolidated roadmap or approved
program specification. If the target architecture differs from current code,
it must be labeled as a migration target.

Future maintainers update `AGENTS.md` when task semantics, package ownership,
canonical contracts, or agent workflow changes. Volatile runtime status belongs
in the consolidated roadmap, and task-level instructions belong in a
Superpowers plan.

## 7. Acceptance criteria

The rewrite is complete when:

1. a new coding agent can explain the mission and all three task semantics;
2. the guide shows current package ownership and where to inspect first;
3. the shared temporal facade with pluggable aligners is represented accurately;
4. KIS/VQA progressive and TRAKE ordered semantics remain distinct;
5. canonical identity, evidence, state, artifact, and provider invariants are
   explicit;
6. the approved four-plan program and deferred work are discoverable;
7. working/testing/reporting instructions are actionable;
8. repeated content from the old guide is removed;
9. references point to files that exist in the working tree;
10. Markdown structure and `git diff --check` pass;
11. no unrelated repository file is modified as part of the rewrite.
