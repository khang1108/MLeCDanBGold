# Simplify Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove obsolete orchestration compatibility paths and make runtime composition, health reporting, and temporal exploration easier to inspect.

**Architecture:** `setup.py` coordinates configuration and final composition. `corpus_setup.py` owns canonical corpus/artifact loading and `retrieval_setup.py` owns online index/model composition. `SearchService` only composes request workflows. A pure health-report function reads those dependencies without adding a new runtime service or mutable state.

**Tech Stack:** Python 3, unittest, existing HCMAI services.

**Spec:** User-approved orchestration simplification items 1–5 from the 2026-09-14 review.

## Global Constraints

- Preserve canonical frame identity and existing KIS, TRAKE, image, filter, and health response behavior.
- Remove compatibility behavior only after production composition provides the current dependency explicitly.
- Do not add dependency injection, registries, or generic loader frameworks.
- Keep experiment logic in retrieval/temporal; orchestration only composes it.

---

### Task 1: Remove implicit temporal and image dependency discovery

**Files:**

- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/workflows/temporal_search.py`
- Test: `tests/orchestration/test_pipeline.py`

- [x] Write a failing construction test proving `SearchService` does not inspect a retrieval service to synthesize an image encoder or temporal evidence.
- [x] Remove `_LegacyDenseEvidenceAdapter`, `_UNSET`, retrieval introspection, and the legacy `score_event_videos` branch.
- [x] Run the focused orchestration test.

### Task 2: Keep runtime dependency discovery in setup

**Files:**

- Modify: `src/hcmai/orchestration/setup.py`
- Test: `tests/orchestration/test_setup.py`

- [x] Confirm setup already constructs the image encoder and temporal evidence, then pass the selected visual retriever through that same composition boundary.
- [x] Pass the selected optional image encoder and temporal evidence directly to `SearchService`; leave request construction free of retrieval discovery.
- [x] Run focused orchestration tests and audit that `source_retriever(...)` occurs only in `setup.py`.

### Task 3: Extract health reporting

**Files:**

- Create: `src/hcmai/orchestration/health.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Test: `tests/orchestration/test_health.py`

- [x] Write a failing test for a health report built from explicit dependencies.
- [x] Move readiness and capability calculation into `build_health_report`; retain `SearchService.health` as a compatibility delegate for current routers and application startup.
- [x] Run focused health and API import tests.

### Verification

- [x] Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src aic/bin/python -m unittest discover -s tests -v`.
- [x] Run `git diff --check`.

### Task 4: Split corpus and retrieval startup ownership

**Files:**

- Create: `src/hcmai/orchestration/corpus_setup.py`
- Create: `src/hcmai/orchestration/retrieval_setup.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Test: `tests/orchestration/test_setup_modules.py`

- [x] Write failing tests for the extracted corpus loader's required-metadata failure and the retrieval module's selected-visual-retriever boundary.
- [x] Move canonical corpus/artifact loading into `corpus_setup.py` without changing required metadata failures or optional artifact diagnostics.
- [x] Move retrieval indexes, query encoders, visual selection, and temporal-evidence composition into `retrieval_setup.py` without changing index selection or required-source behavior.
- [x] Reduce `setup.py` to configuration, remote inference, query preparation, and final `SearchService` composition.
- [x] Run focused setup-module tests and source compilation.

### Task 5: Flatten active temporal-exploration evaluation

**Files:**

- Modify: `src/hcmai/orchestration/workflows/temporal_exploration.py`
- Test: `tests/orchestration/test_temporal_exploration.py`

- [x] Add a behavior test for active-branch re-evaluation under a changed window.
- [x] Replace the `_evaluate_available` → `_evaluate` → `_evaluate_video` chain with an active-branch evaluator and one shared decoder helper.
- [x] Preserve `OSError` translation, evaluate-before-publish behavior, revision guards, and undo semantics.
- [x] Run focused temporal-exploration tests.
