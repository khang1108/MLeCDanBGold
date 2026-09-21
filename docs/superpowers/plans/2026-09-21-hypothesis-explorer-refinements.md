# Hypothesis Explorer Refinements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine 4 targeted correctness and usability aspects of Hypothesis Explorer before feature freeze for paper writing and demo recording.

**Architecture:** Exact frame identity pinning in temporal constraint decoding; transactional UseAlternative semantics; disambiguated Reject occurrence in frontend inspector and path preview; auto-focus and informative empty state in alternative occurrences view.

**Tech Stack:** Python (FastAPI, NumPy, pytest), React (Jest, React Testing Library).

---

### Task 1: Pin exact `frame_id` for anchors in temporal decoding

**Files:**
- Modify: `src/hcmai/event_trail/decoding/decoder.py`
- Test: `tests/event_trail/test_exact_frame_anchor.py`

- [ ] **Step 1: Write failing test for exact frame_id anchor pinning**
Create `tests/event_trail/test_exact_frame_anchor.py` with two candidate frames sharing the exact same timestamp (e.g. 5000ms) but distinct frame_ids (`f_target` and `f_other`). Verify that setting anchor to `f_target` strictly decodes `f_target`.

- [ ] **Step 2: Update `decode()` and `alternatives()` in `decoder.py`**
After `build_mask`, apply:
```python
for i, frame_id in enumerate(constraints.anchors):
    if frame_id is not None:
        mask[i] &= (video.frame_ids == frame_id)
```
Handle empty mask cases returning `contradictory_conditions` in `decode()` or empty tuple `()` in `alternatives()`.

- [ ] **Step 3: Run pytest on decoder tests**
`aic/bin/pytest tests/event_trail/test_exact_frame_anchor.py tests/event_trail/test_decoder.py`

---

### Task 2: Make `UseAlternative` strictly transactional

**Files:**
- Modify: `src/hcmai/event_trail/actions/transitions.py`
- Test: `tests/event_trail/test_hypothesis_actions.py`

- [ ] **Step 1: Write failing test for transactional `UseAlternative`**
Add test in `tests/event_trail/test_hypothesis_actions.py` showing that if an alternative anchor contradicts other constraints (causing decode failure), `apply_use_alternative` raises `EventTrailError("CONSTRAINT_CONFLICT")` and leaves the session intact (no revision increment, not exhausted).

- [ ] **Step 2: Update `apply_use_alternative` in `transitions.py`**
If `outcome.status != "ok"`, raise `EventTrailError("CONSTRAINT_CONFLICT", ...)`.

- [ ] **Step 3: Run pytest on action tests**
`aic/bin/pytest tests/event_trail/test_hypothesis_actions.py`

---

### Task 3: Disambiguate `Reject occurrence` UX in frontend

**Files:**
- Modify: `frontend/src/features/event-trail/components/EventTrailPanel.jsx`
- Modify: `frontend/src/features/event-trail/components/EvidenceInspector.jsx`
- Modify: `frontend/src/features/event-trail/components/HypothesisPathPreview.jsx`
- Test: `frontend/src/features/event-trail/components/EventTrailPanel.test.jsx`
- Test: `frontend/src/features/event-trail/components/HypothesisPathPreview.test.jsx`

- [ ] **Step 1: Write failing tests for reject disambiguation**
Verify Reject button on inspector is only enabled when `is_current: true` mode exists; verify `HypothesisPathPreview` exposes a direct "Reject this alternative" button.

- [ ] **Step 2: Implement disambiguation in EventTrailPanel, EvidenceInspector, and HypothesisPathPreview**
Remove fallback to `alternatives[0]`. Only set `currentModeId` if mode has `is_current: true`. Wire direct alternative rejection from `HypothesisPathPreview`.

- [ ] **Step 3: Run frontend tests**
`CI=true npm test -- --testPathPattern="EventTrailPanel|HypothesisPathPreview" --runInBand`

---

### Task 4: Auto-focus selected event on Explorer open & empty state UX

**Files:**
- Modify: `frontend/src/features/event-trail/components/EventTrailPanel.jsx`
- Modify: `frontend/src/features/event-trail/components/HypothesisAlternatives.jsx`
- Test: `frontend/src/features/event-trail/components/HypothesisAlternatives.test.jsx`

- [ ] **Step 1: Write failing tests for empty state message and auto-focus**
Verify `HypothesisAlternatives` displays "No distinct alternative occurrences found for {eventId} under the current constraints." when empty and loaded.

- [ ] **Step 2: Implement auto-focus in `EventTrailPanel.jsx` and empty message in `HypothesisAlternatives.jsx`**
Add `useEffect` to trigger `onFocusEvent(activeEventId)` when session is active and event is not yet focused. Render empty message in `HypothesisAlternatives`.

- [ ] **Step 3: Run all frontend tests**
`CI=true npm test -- --runInBand`
