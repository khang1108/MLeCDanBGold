# Residual Transition Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-training offline probe that decides whether score-space residuals discriminate correct adjacent transitions, and integrate a separate Residual-DP baseline only if that gate passes.

**Architecture:** Phase 1 operates on existing `VideoEventScores.scores` and an explicit gold-derived pair manifest; it computes raw and multiscale residual variants and publishes edge metrics. Phase 2 is conditionally implemented as a new baseline method with pairwise edge scores in sparse Viterbi decoding. Existing baseline methods remain unchanged.

**Tech Stack:** Python 3.11, NumPy, standard-library JSON/CSV, pytest

**Spec:** `docs/superpowers/specs/2026-09-11-residual-transition-probe-design.md`

## Global Constraints

- No model training, new model dependency, or embedding rebuild.
- No expected-duration, maximum-gap, or linear-gap term in the residual method.
- Phase 1 must pass the predeclared gate before Task 4 begins.
- All edge comparisons occur within one video and use strict timestamp order for forward edges.
- Existing unary baseline outputs and implementations are controls and must not change.

---

## File map

- `baseline/residual.py`: pure residual and robust-normalization functions.
- `baseline/probe_contracts.py`: validated manifest and report records.
- `scripts/evaluation/build_residual_probe_manifest.py`: gold-to-manifest conversion and deterministic hard-negative generation.
- `baseline/probe_residual.py`: Phase 1 CLI, pair evaluation, bootstrap, JSON/CSV publication.
- `baseline/methods/residual_lattice.py`: Phase 2 pairwise Viterbi method, created only after the gate passes.
- `baseline/run_residual_lattice.py`: fixed-method CLI wrapper, created only after the gate passes.
- `tests/baseline/`: new focused unit and integration tests because the supplied archive contains no tests.
- `artifacts/probes/residual_transition/`: generated manifests and reports; never imported by runtime code.

### Task 1: Implement and verify pure residual scoring

**Files:**
- Create: `baseline/residual.py`
- Test: `tests/baseline/test_residual.py`

**Interfaces:**
- Produces: `score_residual(scores: np.ndarray, event_index: int, a: int, b: int) -> float`
- Produces: `score_multiscale_residual(scores: np.ndarray, event_index: int, a: int, b: int, radii: tuple[int, ...] = (0, 1, 2)) -> float`
- Produces: `robust_z(values: np.ndarray, eps: float = 1e-8) -> np.ndarray`

- [x] **Step 1: Write failing algebra, direction, boundary, and MAD-zero tests.** Use a `2 x 4` matrix where the exact four-term result is hand-computable; assert swapping endpoints negates the raw residual; assert multiscale windows clip to array bounds; assert constant input normalizes to zeros.
- [x] **Step 2: Run `PYTHONPATH=.:src pytest tests/baseline/test_residual.py -v`; expect import failure.**
- [x] **Step 3: Implement strict validation and the three pure functions.** Reject non-2D/non-finite matrices, invalid adjacent event index, equal/out-of-range endpoints, empty radii, negative radii, and non-finite epsilon. Use endpoint neighborhood means and `np.median` across scale residuals.
- [x] **Step 4: Rerun the test file; expect all tests to pass.**
- [x] **Step 5: Commit `test: add score-space residual primitives`.**

### Task 2: Define the leak-proof probe manifest

**Files:**
- Create: `baseline/probe_contracts.py`
- Test: `tests/baseline/test_probe_contracts.py`

**Interfaces:**
- Produces: immutable `TransitionPair` with `query_id`, `split`, `event_index`, `video_id`, `positive_frame_idx_a`, `positive_frame_idx_b`, `negative_frame_idx_a`, `negative_frame_idx_b`, and `negative_type`.
- Produces: `load_transition_pairs(path: Path) -> tuple[TransitionPair, ...]`.

- [ ] **Step 1: Write failing JSONL validation tests.** Cover a valid row, duplicate row, missing field, negative index, equal positive endpoints, unsupported negative type, blank ID, and malformed JSON with line number in the error.
- [ ] **Step 2: Run the test file and verify failure.**
- [ ] **Step 3: Implement the dataclass, the literal negative-type set, deterministic loading, duplicate rejection, and contextual errors.** Do not import retrieval services or ground-truth loaders into this module.
- [ ] **Step 4: Rerun the tests; expect all pass.**
- [ ] **Step 5: Commit `feat: define residual probe manifest contract`.**

### Task 3: Build the offline transition-pair manifest

**Files:**
- Create: `scripts/evaluation/build_residual_probe_manifest.py`
- Test: `tests/scripts/test_build_residual_probe_manifest.py`

**Interfaces:**
- Consumes: `hcmai-query-test-set-v2` fixtures with `event_windows`, per-query `VideoEventScores`, and `select_event_candidates(..., limit=32, ...)`.
- Produces: deterministic canonical-identity `transition_pairs.jsonl` and `manifest_summary.json`.

- [ ] **Step 1: Write failing fixture tests.** Cover chronological adjacent events, a non-chronological event such as `E3 -> E4`, missing `event_windows`, duplicate canonical `frame_idx`, unavailable negative families, and deterministic output ordering.
- [ ] **Step 2: Run `PYTHONPATH=.:src pytest tests/scripts/test_build_residual_probe_manifest.py -v`; expect import failure.**
- [ ] **Step 3: Implement positive extraction.** Resolve every representative `frame_idx` through `VideoEventScores.frame_idx`; require exactly one match; retain only adjacent pairs with strictly increasing timestamps; report every excluded pair with a reason.
- [ ] **Step 4: Implement deterministic negative generation from top-32 candidates.** Generate `reverse` diagnostically; choose `wrong_next` and `wrong_previous` by high unary score outside the corresponding gold event window; choose `same_video_unrelated` outside both event windows while preserving forward time. Never manufacture a family with no valid candidate.
- [ ] **Step 5: Write canonical JSONL plus counts by split/query/video/type and excluded-pair reasons.** Multiple tolerance paths for one query must share one independence-group ID and must not inflate the research sample count.
- [ ] **Step 6: Rerun tests; expect all pass.**
- [ ] **Step 7: Commit `feat: build canonical residual probe manifest`.**

### Task 4: Build the Phase 1 evaluator and decision report

**Files:**
- Create: `baseline/probe_residual.py`
- Test: `tests/baseline/test_probe_residual.py`

**Interfaces:**
- Consumes: `TransitionPair`, existing per-query `VideoEventScores`, and residual functions.
- Produces: `evaluate_pairs(...) -> dict[str, object]` and CLI files `edge_scores.csv`, `probe_report.json`, `probe_summary.md`.

- [ ] **Step 1: Write failing tests with two synthetic videos and multiple queries.** Assert exact canonical `frame_idx` lookup, rejection of missing/duplicate identities, same-video enforcement, timestamp-order validation for positive pairs, diagnostic reverse handling, tie credit `0.5`, per-type metrics, query-macro metrics, query/event-macro ROC-AUC, deterministic query-level bootstrap, and atomic output files.
- [ ] **Step 2: Run the test and verify failure.**
- [ ] **Step 3: Implement evaluation for `raw`, `robust_z`, `multiscale_median`, and `multiscale_robust_z`.** Fit robust normalization only on strict chronological edges between the top-32 candidates of the two events in the same query/event/video lattice, never on gold labels. Record missing frame/video failures rather than silently skipping them.
- [ ] **Step 4: Implement fixed gate evaluation exactly as specified.** The report must contain each boolean condition and `passed`; no automatic selection using end-to-end benchmark metrics.
- [ ] **Step 5: Implement CLI wiring to load the normal service/query cases once, score events once, join manifest rows, and atomically write the three artifacts.** Add `--bootstrap-repeats` default `2000` and `--seed` default `20260911`.
- [ ] **Step 6: Run unit tests and a one-query smoke probe.** Expected: deterministic byte-stable numeric content except explicit creation timestamps, zero retrieval calls per manifest row beyond the one call per query.
- [ ] **Step 7: Commit `feat: add residual transition signal probe`.**

### Task 5: Review Phase 1 gate

**Files:**
- Read: `artifacts/probes/residual_transition/probe_report.json`
- Update: `artifacts/probes/residual_transition/probe_summary.md`

- [ ] **Step 1: Count independent query/video transition groups, not alternative tolerance paths.** If fewer than 30, label the result `PILOT_INCONCLUSIVE` regardless of accuracy and inspect missing-row failures.
- [ ] **Step 2: Check overall, reverse, per-type, macro, and bootstrap conditions against the spec.**
- [ ] **Step 3: Inspect margin histograms by query/event without changing thresholds or variants.**
- [ ] **Step 4: Record one decision: `STOP_RESIDUAL`, `COLLECT_MORE_PAIRS`, or `PROCEED_TO_DP`.**
- [ ] **Step 5: Append the result and evidence paths to `KNOWLEDGE.md` with status `REJECTED`, `INCONCLUSIVE`, or `VERIFIED`.**
- [ ] **Step 6: Stop the plan unless the recorded decision is `PROCEED_TO_DP`.**

### Task 6: Implement Residual-DP as a separate baseline

**Files:**
- Create: `baseline/methods/residual_lattice.py`
- Create: `baseline/run_residual_lattice.py`
- Modify: `baseline/contracts.py`
- Modify: `baseline/methods/__init__.py`
- Modify: `baseline/cli.py`
- Test: `tests/baseline/test_residual_lattice.py`

**Interfaces:**
- Produces: `ResidualCandidateLatticeMethod(options: MethodOptions)` with the existing `rank(videos, *, top_k) -> list[BaselinePath]` protocol.
- Adds: `residual_beta: float = 0.0`, `residual_variant: Literal[...] = "raw"` to `MethodOptions`.
- Adds: `"residual_candidate_lattice"` to `MethodName` and `BASELINE_METHODS`.

- [ ] **Step 1: Write a failing brute-force equivalence test.** Enumerate every valid path in a tiny `3 events x 5 frames` lattice and assert Viterbi returns the exact maximum of unary plus adjacent residuals.
- [ ] **Step 2: Add failing controls.** Assert `beta=0` is path-and-score equivalent to `UnaryCandidateLatticeMethod` with `lambda_gap=0`; assert a constructed high-residual path beats a higher-unary incoherent path; assert separate videos never form a mixed path.
- [ ] **Step 3: Run tests and verify failure.**
- [ ] **Step 4: Implement pairwise `O(MK^2)` recurrence in the new module.** Reuse `select_event_candidates`; do not alter `_decode_order` or existing methods. Require both increasing candidate position and strictly increasing timestamp. Add a corpus preflight/report for duplicate or non-monotonic timestamps; the beta-zero equivalence gate applies only to videos passing this invariant.
- [ ] **Step 5: Register only the new method and expose `--residual-beta` plus `--residual-variant`.** Reject negative/non-finite beta and unsupported variants.
- [ ] **Step 6: Run all `tests/baseline` tests; expect pass.**
- [ ] **Step 7: Commit `feat: add residual candidate-lattice baseline`.**

### Task 7: Run fixed-beta end-to-end ablation

**Files:**
- Create: `artifacts/probes/residual_transition/runs/`
- Update: `artifacts/probes/residual_transition/probe_summary.md`

- [ ] **Step 1: Run the unary control with `lambda_gap=0` and residual runs at beta `0`, `0.1`, `0.25`, `0.5`, and `1.0` on the identical query set and evidence settings.**
- [ ] **Step 2: Evaluate every run with the existing benchmark evaluator.** Preserve raw run and metrics JSON for each beta.
- [ ] **Step 3: Verify residual beta `0` matches the unary control.** Any mismatch is an implementation bug; stop before interpreting metrics.
- [ ] **Step 4: Create one comparison table containing video R@1/R@5, representative R@1@5s, path MRR@5s, exact path, alignment latency, and Phase 1 edge metrics.**
- [ ] **Step 5: Apply the Phase 2 gate without selecting on a hidden/test split.** Record whether the residual hypothesis is supported, rejected, or inconclusive.
- [ ] **Step 6: Append the Phase 2 result, fixed beta grid, metrics, and evidence paths to `KNOWLEDGE.md` with the appropriate research status.**
- [ ] **Step 7: Commit `eval: report residual transition probe`.**

## Self-review

- Spec coverage: raw/multiscale/robust variants, canonical identity, manifest generation, four negative families, bootstrap gate, conditional DP, beta control, latency, and research-memory updates all map to Tasks 1–7.
- Placeholder scan: no deferred implementation markers remain.
- Type consistency: residual functions, manifest records, report interface, method name, and `MethodOptions` fields are consistent across tasks.
