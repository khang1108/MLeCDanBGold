# EventTrail Full Frontend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the already-implemented EventTrail backend in `src_v1.5` to the live KIS result UI end-to-end, expose Event Rail + evidence correction controls, integrate explicit Use-based submission, preserve grid/search state, add query/session correlation, and remove the legacy TemporalExploration product path after migration.

**Architecture:** Keep KIS search as the fast path. A successful live KIS response carries `evidence_snapshot_id` and each result carries `result_id`; opening a promising result passes those opaque IDs plus the committed KIS revision/events into one App-owned `useEventTrail` session. EventTrail mutations remain selected-video-only and call the existing `/api/v1/event-trail` backend; the frontend never reruns retrieval or computes path diffs. `Use` resolves the current playback timestamp to a canonical frame via the existing `/api/v1/frames/resolve` endpoint, then anchors that frame through `use_frame`. After all frontend callers move, delete the legacy `/api/v1/exploration` route, `TemporalExploration`, `exploration_seed`, and legacy UI.

**Tech Stack:** React 19, React Testing Library/Jest via `react-scripts`, FastAPI/Pydantic, existing `EventTrailService`, existing canonical frame resolver, existing DRES direct-submission flow.

**Spec:** `docs/superpowers/specs/2026-09-16-event-trail-design.md`

## Global Constraints

- EventTrail is an escalation path from a live ranked KIS result; it does not replace the ranked grid or Query Composer.
- EventTrail opens only from the exact `evidence_snapshot_id + result_id + KIS revision` that produced the selected result.
- `TrailPath_0` must equal the ranked result path exactly; frontend must not synthesize or re-decode it.
- The hot Trail loop must not call the LLM, embedding model, BM25 corpus retrieval, FAISS corpus search, or full-corpus temporal scoring.
- `Approve` anchors the backend's current candidate for one event; client sends only `event_id`.
- `Decline` rejects the backend's current candidate occurrence for one event; client sends only `event_id`.
- `Use` anchors a canonical frame explicitly selected from the current player time and makes that frame the Trail submission selection.
- Continuous playback time is not a frame ID. `Use` must call the existing canonical frame resolver before sending `use_frame`.
- Trail revisions are monotonic, including Undo.
- Contradictory hard actions are transactional; an exhausting Decline is committed and preserves `last_valid_path`.
- Direct and indirect changes come from backend `transition`; frontend must not infer them by diffing response trees.
- A new committed KIS result world invalidates the active Trail. A closed inspector may preserve one active Trail only when reopening the same result in the same snapshot.
- Replay/filter results do not fabricate EventTrail context. EventTrail is available only for live KIS results with a non-null evidence snapshot.
- No global reranking/relevance feedback, path carousel, Redis, persistent Trail sessions, confidence model, or new retrieval algorithm in this plan.
- Do not keep the legacy TemporalExploration product API as a fallback after migration.

---

## File Structure / Ownership

### Frontend: create

- `frontend/src/api/eventTrail.js` — thin EventTrail transport + response validation.
- `frontend/src/api/eventTrail.test.js` — transport/contract tests.
- `frontend/src/features/event-trail/hooks/useEventTrail.js` — one App-owned Trail session state machine.
- `frontend/src/features/event-trail/hooks/useEventTrail.test.js` — lifecycle, stale-response, conflict, expiry tests.
- `frontend/src/features/event-trail/components/EventTrailPanel.jsx` — Trail shell and toolbar.
- `frontend/src/features/event-trail/components/EventTrailPanel.test.jsx` — integrated panel behavior.
- `frontend/src/features/event-trail/components/EventRail.jsx` — ordered event/candidate rail.
- `frontend/src/features/event-trail/components/EvidenceInspector.jsx` — selected-event actions and diff summary.
- `frontend/src/features/event-trail/components/TrailWindowControls.jsx` — collapsed global range controls.
- `frontend/src/features/event-trail/index.js` — feature exports.
- `frontend/src/styles/event-trail.css` — focused EventTrail styles imported by `styles/index.css`.

### Frontend: modify

- `frontend/src/api/client.js` — preserve all structured backend `detail.code` values on thrown errors.
- `frontend/src/api/client.test.js` — structured error regression.
- `frontend/src/features/search/components/SearchWorkspace.jsx` — retain live EventTrail handoff context instead of `exploration_seed`.
- `frontend/src/features/search/components/SearchWorkspace.test.jsx` — exact snapshot/result/revision handoff and invalidation tests.
- `frontend/src/features/frames/components/ImageModal.jsx` — replace legacy Explore panel with EventTrail launch/panel integration.
- `frontend/src/features/frames/components/ImageModal.test.jsx` — launch, player seek, Use/Submit behavior.
- `frontend/src/App.jsx` — replace `useTemporalExploration` with `useEventTrail`, one active selected-result session, Back/close semantics.
- `frontend/src/App.test.jsx` — same-result preserve / different-result discard / new-search invalidation.
- `frontend/src/styles/index.css` — import `event-trail.css`.

### Backend: modify

- `src/hcmai/api/contracts/event_trail.py` — optional `search_session_id` on open; add close request/response contract only if required by the route signature below.
- `src/hcmai/api/routers/event_trail.py` — explicit DELETE close route.
- `src/hcmai/event_trail/models.py` — retain optional search-session correlation inside Trail session.
- `src/hcmai/event_trail/service.py` — carry correlation to every log call; explicit close lifecycle.
- `src/hcmai/event_trail/logging.py` — no schema redesign; consume the correlation already supported by `search_session_id`.
- `src/hcmai/api/contracts/kis.py` — remove `KISExplorationSeed` and `exploration_seed` only in the final legacy-deletion task.
- `src/hcmai/orchestration/pipeline.py` — stop materializing `exploration_seed` only in the final legacy-deletion task.
- `src/hcmai/app.py` — remove legacy exploration registry/router in final migration task.
- `src/hcmai/api/contracts/__init__.py`, `src/hcmai/api/routers/__init__.py` — remove legacy exports in final migration task.

### Delete after migration

- `frontend/src/api/exploration.js`
- `frontend/src/api/exploration.test.js`
- `frontend/src/features/alignment/hooks/useTemporalExploration.js`
- `frontend/src/features/alignment/hooks/useTemporalExploration.test.js`
- `frontend/src/features/alignment/components/ExplorationPanel.jsx`
- `frontend/src/features/alignment/components/ExplorationPanel.test.jsx`
- `src/hcmai/api/contracts/exploration.py`
- `src/hcmai/api/routers/exploration.py`
- `src/hcmai/orchestration/workflows/temporal_exploration.py`
- legacy `.exploration-*` CSS selectors from `frontend/src/styles/modal.css`

### Backend tests: create

The archive currently has no Python `tests/` tree. Create only the focused tests needed for this migration:

- `tests/api/test_event_trail_close.py`
- `tests/event_trail/test_search_session_logging.py`
- `tests/api/test_removed_exploration_routes.py`

---

### Task 1: Preserve EventTrail Error Codes and Add a Typed Frontend Transport

**Files:**
- Create: `frontend/src/api/eventTrail.js`
- Create: `frontend/src/api/eventTrail.test.js`
- Modify: `frontend/src/api/client.js`
- Create or modify: `frontend/src/api/client.test.js`

**Interfaces:**
- Consumes backend routes:
  - `POST /api/v1/event-trail/open`
  - `GET /api/v1/event-trail/{session_id}`
  - `POST /api/v1/event-trail/{session_id}/actions`
  - DELETE route added in Task 2.
- Produces:
  - `openEventTrail({ snapshotId, resultId, expectedKisRevision, searchSessionId?, signal })`
  - `getEventTrail(sessionId, { signal })`
  - `actOnEventTrail(sessionId, { expectedTrailRevision, action }, { signal })`
  - `closeEventTrail(sessionId, expectedTrailRevision, { signal })`
- Every thrown HTTP error preserves `error.status` and, when backend returns `detail.code`, `error.code`.

- [ ] **Step 1: Write a failing generic structured-error test**

Add a test proving a non-DRES code such as `TRAIL_REVISION_CONFLICT` survives `requestJson`:

```javascript
fetch.mockResolvedValueOnce({
  ok: false,
  status: 409,
  headers: { get: () => null },
  json: async () => ({
    detail: {
      code: 'TRAIL_REVISION_CONFLICT',
      message: 'Expected trail revision 2 but session is at 3',
    },
  }),
});

await expect(requestJson('/api/v1/event-trail/s1/actions')).rejects.toMatchObject({
  status: 409,
  code: 'TRAIL_REVISION_CONFLICT',
  message: 'Expected trail revision 2 but session is at 3',
});
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cd frontend
CI=true npm test -- --runInBand src/api/client.test.js
```

Expected: FAIL because `client.js` currently attaches `.code` only for two DRES codes.

- [ ] **Step 3: Make structured error-code propagation generic**

Replace the allowlist behavior with:

```javascript
const code = payload?.detail?.code;
if (typeof code === 'string' && code.trim()) error.code = code;
```

Do not change message formatting or DRES status event behavior.

- [ ] **Step 4: Write failing EventTrail transport tests**

Cover exact request bodies and validation:

```javascript
await openEventTrail({
  snapshotId: 'snap_1',
  resultId: 'r_1',
  expectedKisRevision: 3,
  searchSessionId: 'query_1',
});

expect(fetch).toHaveBeenCalledWith(
  expect.stringMatching(/\/api\/v1\/event-trail\/open$/),
  expect.objectContaining({
    method: 'POST',
    body: JSON.stringify({
      snapshot_id: 'snap_1',
      result_id: 'r_1',
      expected_kis_revision: 3,
      search_session_id: 'query_1',
    }),
  }),
);
```

Also assert action mapping is sent unchanged:

```javascript
await actOnEventTrail('trail_1', {
  expectedTrailRevision: 4,
  action: { type: 'decline', event_id: 'E2' },
});
```

and reject malformed responses missing `session_id`, integer `trail_revision`, valid `status`, or a path/last-valid-path shape.

- [ ] **Step 5: Run transport tests and verify RED**

```bash
cd frontend
CI=true npm test -- --runInBand src/api/eventTrail.test.js src/api/client.test.js
```

Expected: EventTrail module missing; generic error-code test fails before Step 3 and passes after Step 3.

- [ ] **Step 6: Implement the thin EventTrail API client**

Use `requestJson` only. Do not put session state or UI decisions in this module. Normalize request field names exactly to backend snake_case and return the backend state unchanged after structural validation.

- [ ] **Step 7: Run focused tests GREEN**

```bash
cd frontend
CI=true npm test -- --runInBand src/api/eventTrail.test.js src/api/client.test.js
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/api/client.js frontend/src/api/client.test.js frontend/src/api/eventTrail.js frontend/src/api/eventTrail.test.js
git commit -m "feat: add EventTrail frontend transport"
```

---

### Task 2: Add Explicit Trail Close and Query-Session Correlation

**Files:**
- Modify: `src/hcmai/api/contracts/event_trail.py`
- Modify: `src/hcmai/api/routers/event_trail.py`
- Modify: `src/hcmai/event_trail/models.py`
- Modify: `src/hcmai/event_trail/service.py`
- Create: `tests/api/test_event_trail_close.py`
- Create: `tests/event_trail/test_search_session_logging.py`

**Interfaces:**
- Extend `EventTrailOpenRequest` with `search_session_id: str | None = None`.
- Extend `EventTrailSession` with `search_session_id: str | None`.
- Add:

```python
EventTrailService.close(
    session_id: str,
    expected_trail_revision: int,
) -> None
```

- Add HTTP:

```text
DELETE /api/v1/event-trail/{session_id}?expected_trail_revision=N
```

returns `204`.
- `close()` logs `trail_close` before removing the session.
- Every existing Trail log event passes `search_session_id=session.search_session_id`; `trail_open` uses the supplied open correlation ID.

- [ ] **Step 1: Write failing close lifecycle tests**

Use a minimal fake/store fixture and assert:

```python
service.close("trail_1", expected_trail_revision=2)
with pytest.raises(EventTrailError) as exc:
    service.get("trail_1")
assert exc.value.code == "TRAIL_SESSION_NOT_FOUND"
```

and stale close is transactional:

```python
with pytest.raises(EventTrailError) as exc:
    service.close("trail_1", expected_trail_revision=1)
assert exc.value.code == "TRAIL_REVISION_CONFLICT"
assert service.get("trail_1").trail_revision == 2
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src pytest tests/api/test_event_trail_close.py -q
```

Expected: FAIL because `close()` / DELETE route do not exist.

- [ ] **Step 3: Write failing correlation-log test**

Open with `search_session_id="query_123"`, perform one action, capture `hcmai.event_trail.interactions`, and assert both JSON records carry:

```json
{"search_session_id":"query_123"}
```

- [ ] **Step 4: Implement optional correlation ownership**

Add `search_session_id` only as metadata. It must never participate in path decoding, revision checks, snapshot identity, or retrieval.

- [ ] **Step 5: Implement `close()` and DELETE route**

Pseudo-order inside `close()`:

```python
with session_store.locked(session_id) as slot:
    session = slot.session
    if session.trail_revision != expected_trail_revision:
        raise EventTrailError("TRAIL_REVISION_CONFLICT", ...)
    log_trail_event(event_type="trail_close", ..., search_session_id=session.search_session_id)
session_store.remove(session_id)
```

Do not silently accept stale revisions.

- [ ] **Step 6: Thread correlation into every EventTrail log call**

For existing `trail_approve`, `trail_decline`, `trail_use`, `submission_select`, `trail_undo`, `trail_window`, `trail_clear_anchor`, and `trail_exhausted`, pass the stored `search_session_id`.

- [ ] **Step 7: Run focused backend tests GREEN**

```bash
PYTHONPATH=src pytest tests/api/test_event_trail_close.py tests/event_trail/test_search_session_logging.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hcmai/api/contracts/event_trail.py src/hcmai/api/routers/event_trail.py src/hcmai/event_trail/models.py src/hcmai/event_trail/service.py tests/api/test_event_trail_close.py tests/event_trail/test_search_session_logging.py
git commit -m "feat: add EventTrail close lifecycle and correlation"
```

---

### Task 3: Build the App-Owned `useEventTrail` Session State Machine

**Files:**
- Create: `frontend/src/features/event-trail/hooks/useEventTrail.js`
- Create: `frontend/src/features/event-trail/hooks/useEventTrail.test.js`
- Create: `frontend/src/features/event-trail/index.js`

**Interfaces:**
- Consumes Task 1 EventTrail API client.
- Produces:

```javascript
useEventTrail()
// => {
//   session,
//   sessionKey,
//   pending,
//   error,
//   unsynced,
//   open(context),
//   act(action),
//   undo(),
//   refresh(),
//   close({ suppressError } = {}),
//   clearLocal(),
// }
```

where `context` is:

```javascript
{
  snapshotId,
  resultId,
  kisRevision,
  searchSessionId,
}
```

and `sessionKey = JSON.stringify([snapshotId, resultId, kisRevision])`.

- [ ] **Step 1: Write failing open/reuse tests**

Test:

```javascript
await result.current.open({
  snapshotId: 'snap_1',
  resultId: 'r_1',
  kisRevision: 2,
  searchSessionId: 'q1',
});
expect(openEventTrail).toHaveBeenCalledTimes(1);

await result.current.open({
  snapshotId: 'snap_1',
  resultId: 'r_1',
  kisRevision: 2,
  searchSessionId: 'q1',
});
expect(openEventTrail).toHaveBeenCalledTimes(1);
```

Same result reuses the active session; a different key explicitly closes the prior session before opening the next one.

- [ ] **Step 2: Add revision-guarded action tests**

Given session rev 3:

```javascript
await result.current.act({ type: 'approve', event_id: 'E2' });
expect(actOnEventTrail).toHaveBeenCalledWith(
  'trail_1',
  { expectedTrailRevision: 3, action: { type: 'approve', event_id: 'E2' } },
  expect.any(Object),
);
```

Block double mutation while pending.

- [ ] **Step 3: Add conflict/expiry tests**

Behavior:

- `TRAIL_REVISION_CONFLICT` → one GET refresh, update state, do **not** replay action automatically.
- `CONSTRAINT_CONFLICT` → keep current state, show backend message, no GET required.
- `TRAIL_SESSION_EXPIRED` / HTTP 410 during action/refresh → clear session and show `EventTrail session expired. Reopen it from this result.`
- `SNAPSHOT_EXPIRED` / HTTP 410 during open → no session; show `EventTrail snapshot expired. Rerun the current search to create a fresh snapshot.`
- 404 session missing → clear local session.

- [ ] **Step 4: Add stale-response generation tests**

Open A, then invalidate/open B; a late A response must not replace B. `AbortController` is not the only guard: compare generation + session key.

- [ ] **Step 5: Run tests RED**

```bash
cd frontend
CI=true npm test -- --runInBand src/features/event-trail/hooks/useEventTrail.test.js
```

Expected: module missing.

- [ ] **Step 6: Implement the minimal hook**

Reuse the proven generation/pending/reconciliation pattern from legacy `useTemporalExploration`, but rewrite around EventTrail's simpler revision contract. Do not copy interval draft logic, event-version logic, or scoring revision fields from the legacy hook.

- [ ] **Step 7: Run tests GREEN**

```bash
cd frontend
CI=true npm test -- --runInBand src/features/event-trail/hooks/useEventTrail.test.js
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/features/event-trail
git commit -m "feat: add EventTrail session hook"
```

---

### Task 4: Hand Off Exact KIS Snapshot/Result Context From the Ranked Grid

**Files:**
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/api/kis.test.js`

**Interfaces:**
- Replace `liveKisSnapshotRef` with `liveEventTrailContextRef` holding only the live search handoff needed by the UI:

```javascript
{
  snapshotId: response.evidence_snapshot_id,
  kisRevision: response.intent.revision,
  events: response.intent.events.map(({ id, text, images }) => ({ id, text, images })),
  searchSessionId: activeQueryIdOrNull,
}
```

- Every live KIS result already carries `result_id`; clicked selection adds:

```javascript
{
  frame,
  eventTrailContext: {
    snapshotId,
    resultId: frame.result_id,
    kisRevision,
    events,
    searchSessionId,
  },
}
```

- Filter/replay results omit `eventTrailContext`.
- If `evidence_snapshot_id` is null, selection omits EventTrail context and the search warning remains visible.

- [ ] **Step 1: Update `mockKisResponse` contract fixture first**

Add `evidence_snapshot_id` and `result_id` to the default live temporal response fixture. Keep explicit no-snapshot tests for degraded EventTrail availability.

- [ ] **Step 2: Write failing exact-handoff test**

```javascript
expect(onFrameClick).toHaveBeenCalledWith({
  frame: expect.objectContaining({ result_id: 'r_1', video_id: 'V01' }),
  eventTrailContext: {
    snapshotId: 'snap_1',
    resultId: 'r_1',
    kisRevision: 3,
    events: [
      { id: 'E1', text: 'woman enters', images: [] },
      { id: 'E2', text: 'woman sits', images: [] },
    ],
    searchSessionId: expect.anything(),
  },
});
```

Do not derive `resultId` from `video_id`, rank, frame ID, or array index.

- [ ] **Step 3: Add no-snapshot/replay tests**

Assert no `eventTrailContext` when:

```text
evidence_snapshot_id == null
resultType is replay
resultType is filter
```

- [ ] **Step 4: Add invalidation test for every committed new live KIS response**

After any successful live KIS search/search-only response that replaces the grid, call the renamed callback `onEventTrailInvalidated`. This is necessary even when semantic revision is unchanged by `SearchOnly`, because the evidence snapshot/result IDs define a new result world.

- [ ] **Step 5: Run tests RED**

```bash
cd frontend
CI=true npm test -- --runInBand src/features/search/components/SearchWorkspace.test.jsx src/api/kis.test.js
```

Expected: current code still forwards `explorationSnapshot` and ignores `evidence_snapshot_id`.

- [ ] **Step 6: Implement exact live EventTrail handoff**

Remove all reads of `response.exploration_seed` in `SearchWorkspace`. Do not remove the backend field yet; final legacy deletion is Task 9.

`searchSessionId` should use the active query-history query ID only when one exists. If the user is not connected to local history identity, pass `null`; never fabricate a query ID solely for EventTrail logging.

- [ ] **Step 7: Run focused tests GREEN**

```bash
cd frontend
CI=true npm test -- --runInBand src/features/search/components/SearchWorkspace.test.jsx src/api/kis.test.js
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/features/search/components/SearchWorkspace.jsx frontend/src/features/search/components/SearchWorkspace.test.jsx frontend/src/api/kis.test.js
git commit -m "feat: hand off KIS results to EventTrail"
```

---

### Task 5: Implement Event Rail and Evidence Inspector UI

**Files:**
- Create: `frontend/src/features/event-trail/components/EventRail.jsx`
- Create: `frontend/src/features/event-trail/components/EvidenceInspector.jsx`
- Create: `frontend/src/features/event-trail/components/TrailWindowControls.jsx`
- Create: `frontend/src/features/event-trail/components/EventTrailPanel.jsx`
- Create: `frontend/src/features/event-trail/components/EventTrailPanel.test.jsx`
- Create: `frontend/src/styles/event-trail.css`
- Modify: `frontend/src/styles/index.css`

**Interfaces:**
- `EventTrailPanel` props:

```javascript
{
  events,
  state,
  pending,
  error,
  selectedEventId,
  onSelectEvent,
  onExplore,
  onUse,
  onApprove,
  onDecline,
  onClearAnchor,
  onUndo,
  onSetWindow,
  onClearWindow,
  onBack,
  onSubmit,
}
```

- Current candidate source:
  - active state → `state.path`
  - exhausted state → `state.last_valid_path`, rendered dimmed/read-only.
- Event semantic text comes from the committed `eventTrailContext.events`, never from Trail backend inference.

- [ ] **Step 1: Write failing rail rendering test**

Given E1/E2/E3 + active path, assert all event IDs/texts, canonical timestamps, and keyframe thumbnails render. Clicking E2 calls `onSelectEvent('E2')` and its candidate seek control calls `onExplore` with E2 candidate.

- [ ] **Step 2: Write failing action-state tests**

Assert:

- Approve disabled when selected event is already approved or `state.status === 'exhausted'`.
- Decline disabled when selected event is approved or exhausted.
- Use disabled while exhausted; v1 exhausted recovery is Undo or Back, matching the approved design.
- Clear anchor shown only for approved event.
- Undo disabled while pending; server remains source of truth for whether Undo is valid.
- Submit disabled when exhausted or when `state.submission_selection` is absent.

- [ ] **Step 3: Write failing diff/exhaustion tests**

For transition:

```javascript
{
  action_event_id: 'E2',
  direct_changed_event_ids: ['E2'],
  indirect_changed_event_ids: ['E1', 'E3'],
  candidate_diffs: [...],
}
```

assert the UI shows E2 as the acted-on event plus `2 other events updated`, and renders `before → after` timestamps only for the latest transition. In exhausted state, show `No valid path remains in this video`, keep the last valid rail dimmed, and keep Undo + Back available.

- [ ] **Step 4: Write failing secondary-window test**

`TrailWindowControls` is collapsed by default. Opening it allows integer millisecond start/end entry and emits exactly:

```javascript
onSetWindow({ type: 'set_window', start_ms: 10_000, end_ms: 40_000 })
```

Clear emits `{ type: 'clear_window' }`. This is a Trail-global constraint, not an event feedback interval.

- [ ] **Step 5: Run component test RED**

```bash
cd frontend
CI=true npm test -- --runInBand src/features/event-trail/components/EventTrailPanel.test.jsx
```

Expected: components missing.

- [ ] **Step 6: Implement focused components**

Reuse `formatTimestampMs` and `keyframeUrl`; do not copy result-card accordion behavior. Keep the rail always visible once Trail is open. Use English labels already approved:

```text
EventTrail
Explore
Use
Approve
Decline
Clear anchor
Search range
Undo
Back to results
Submit
```

- [ ] **Step 7: Implement styles without changing the global design system**

`event-trail.css` owns `.event-trail-*` selectors only. Fit inside the existing 380px inspector column; allow rail scrolling rather than widening the modal.

- [ ] **Step 8: Run tests GREEN**

```bash
cd frontend
CI=true npm test -- --runInBand src/features/event-trail/components/EventTrailPanel.test.jsx
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add frontend/src/features/event-trail/components frontend/src/styles/event-trail.css frontend/src/styles/index.css
git commit -m "feat: add EventTrail rail and evidence inspector"
```

---

### Task 6: Integrate EventTrail With the Video Player, Canonical `Use`, and Submission

**Files:**
- Modify: `frontend/src/features/frames/components/ImageModal.jsx`
- Modify: `frontend/src/features/frames/components/ImageModal.test.jsx`
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/App.test.jsx`
- Modify: `frontend/src/features/event-trail/index.js`
- Reuse without changing unless a bug is found: `frontend/src/api/frames.js`

**Interfaces:**
- `ImageModal` replaces prop `exploration` with:

```javascript
eventTrail?: {
  context,
  state,
  pending,
  error,
  open,
  act,
  undo,
  refresh,
  back,
}
```

- `Use` flow:

```text
videoRef.current.currentTime
 -> round to requested_timestamp_ms
 -> resolveFrameAtTimestamp({ videoId, timestampMs })
 -> verify resolved.video_id == Trail video
 -> act({ type: 'use_frame', event_id, frame_id: resolved.frame_id })
```

- Trail `Submit` uses `state.submission_selection.timestamp_ms` and `state.video_id` through existing `onOpenSubmission`.

- [ ] **Step 1: Write failing launch test matching the screenshot problem**

Render a modal with live EventTrail context but no active Trail state. Assert a visible `Open EventTrail` button under Event Alignment / inspector content. Opening Trail must **not** depend on video duration or successful MP4 streaming.

This specifically fixes the current behavior where legacy Explore is hidden/disabled when no `exploration_seed` or duration is available.

- [ ] **Step 2: Write failing open-panel/player-focus test**

Once Trail state exists:

- legacy `Explore` interval panel is absent;
- Event Rail is visible;
- selecting E2 + Explore seeks player to E2's current candidate timestamp;
- indirect path changes never auto-seek the player.

- [ ] **Step 3: Write failing Decline focus rule test**

When a successful Decline on selected E2 returns a new E2 candidate, auto-seek only to the new E2 timestamp. Approve and Use keep current playback position. If Decline exhausts the path, do not auto-seek to another event/video.

- [ ] **Step 4: Write failing canonical Use test**

At player time `12.345s`, mock:

```javascript
resolveFrameAtTimestamp.mockResolvedValue({
  frame_id: 'f12',
  video_id: 'V01',
  requested_timestamp_ms: 12345,
  frame_idx: 300,
  timestamp_ms: 12000,
  metadata: {},
});
```

Click Use for E2 and assert:

```javascript
eventTrail.act({
  type: 'use_frame',
  event_id: 'E2',
  frame_id: 'f12',
});
```

The canonical `12000` ms returned by the backend, not arbitrary `12345`, becomes the eventual Trail submission selection through the EventTrail action response.

- [ ] **Step 5: Write failing Trail Submit test**

With no `submission_selection`, Submit is disabled and UI says `Use a frame before submitting from EventTrail.` If `state.status === 'exhausted'`, Submit stays disabled even when an older selection exists.

With:

```javascript
submission_selection: {
  event_id: 'E2', frame_id: 'f12', frame_idx: 300, timestamp_ms: 12000,
}
```

click Submit and assert:

```javascript
onOpenSubmission({ videoId: 'V01', startMs: 12000, endMs: 12000 })
```

While Trail is active, hide the modal header's generic current-time Submit button so it cannot bypass explicit Trail selection. Outside Trail, preserve the existing fast-path Submit behavior.

- [ ] **Step 6: Write failing App lifecycle tests**

App owns the hook once. Assert:

- closing modal with X/Escape preserves the active same-result Trail session;
- reopening the same `snapshotId/resultId/revision` shows the existing Trail state without another open call;
- selecting a different result closes/discards the old Trail before presenting the new result;
- `Back to results` explicitly closes Trail and modal;
- successful new KIS result world / New Search / Replay calls `eventTrail.close({ suppressError: true })`.

- [ ] **Step 7: Run tests RED**

```bash
cd frontend
CI=true npm test -- --runInBand \
  src/features/frames/components/ImageModal.test.jsx \
  src/App.test.jsx
```

Expected: current App/ImageModal still use TemporalExploration.

- [ ] **Step 8: Implement ImageModal and App wiring**

Create a deterministic selection key:

```javascript
const eventTrailSelectionKey = (selection) => {
  const c = selection?.eventTrailContext;
  return c ? JSON.stringify([c.snapshotId, c.resultId, c.kisRevision]) : null;
};
```

Use one `useEventTrail()` in `AppShell`. Do not put EventTrail session state inside `ImageModal`.

- [ ] **Step 9: Run focused tests GREEN**

```bash
cd frontend
CI=true npm test -- --runInBand \
  src/features/frames/components/ImageModal.test.jsx \
  src/App.test.jsx \
  src/features/event-trail/components/EventTrailPanel.test.jsx \
  src/features/event-trail/hooks/useEventTrail.test.js
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add frontend/src/App.jsx frontend/src/App.test.jsx frontend/src/features/frames/components/ImageModal.jsx frontend/src/features/frames/components/ImageModal.test.jsx frontend/src/features/event-trail frontend/src/api/frames.js
git commit -m "feat: connect EventTrail to video inspection and submission"
```

---

### Task 7: Complete Diff-Aware UX and Per-Result Exploration State

**Files:**
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/features/frames/components/FrameCard.jsx`
- Modify: `frontend/src/features/frames/components/FrameCard.test.jsx`
- Modify: `frontend/src/features/event-trail/components/EventTrailPanel.jsx`
- Modify: `frontend/src/features/event-trail/components/EventTrailPanel.test.jsx`

**Interfaces:**
- Search result ordering and retrieval scores never change from Trail feedback.
- `AppShell` owns an `eventTrailAnnotations` map because the Trail hook also lives in App. Keys are `${snapshotId}:${resultId}` and values are the UI-only states below. `AppShell` passes the map into `SearchWorkspace`; `SearchWorkspace` passes the matching value into `FrameCard`.
- Add UI-only result annotations keyed by the exact snapshot/result pair:

```javascript
'unvisited' | 'explored' | 'exhausted'
```

- successful `trail_open` → explored.
- active Trail state `status === 'exhausted'` → exhausted.
- Undo that restores an active path returns annotation to explored, never unvisited.
- `onEventTrailInvalidated` clears the annotation map when a new live result world replaces the grid; modal X does not clear the annotation.

- [ ] **Step 1: Write failing result-order preservation test**

Open/decline/exhaust one result and assert the same result array/order remains rendered. No Trail action may call `searchKis` or mutate frame scores/ranks.

- [ ] **Step 2: Write failing annotation tests**

FrameCard receives a non-ranking Trail state badge:

```text
Explored
Exhausted
```

Do not hide or reorder exhausted results.

- [ ] **Step 3: Write failing changed-event review behavior**

Latest transition only:

- direct target visually marked;
- indirect events visually marked separately;
- `N other events updated` button cycles selection through `indirect_changed_event_ids` without changing playback unless user chooses Explore/candidate timestamp.

- [ ] **Step 4: Run RED**

```bash
cd frontend
CI=true npm test -- --runInBand \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/frames/components/FrameCard.test.jsx \
  src/features/event-trail/components/EventTrailPanel.test.jsx
```

- [ ] **Step 5: Implement UI-only annotations and review queue**

Keep annotation state scoped to the current live snapshot. Clear it when a new KIS response replaces that snapshot. Do not persist it into query replay in v1.

- [ ] **Step 6: Run GREEN**

Run the same command; expect PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/features/search/components/SearchWorkspace.jsx frontend/src/features/search/components/SearchWorkspace.test.jsx frontend/src/features/frames/components/FrameCard.jsx frontend/src/features/frames/components/FrameCard.test.jsx frontend/src/features/event-trail
git commit -m "feat: add EventTrail result and diff annotations"
```

---

### Task 8: Verify Interaction Logging and Submission Correlation

**Files:**
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx` only if correlation handoff needs coverage.
- Modify: `tests/event_trail/test_search_session_logging.py`
- No query-history schema migration in this task.

**Interfaces:**
- Backend EventTrail JSON logs are the authoritative detailed Trail action log.
- `search_session_id` is the existing query-history query ID when available.
- Existing frontend `result_open` and `submission` query-history events remain unchanged.
- A Trail `Use` already produces backend `submission_select`; final DRES opening still goes through existing `handleOpenSubmission`, which records the existing query-history `submission` event.

- [ ] **Step 1: Add one end-to-end correlation test around backend logs**

Capture records for:

```text
trail_open
trail_approve
trail_decline
trail_use
submission_select
trail_undo
trail_close
```

and assert every record from one session has the same:

```text
search_session_id
snapshot_id
result_id
trail_session_id
```

with monotonic trail revisions on committed mutations.

- [ ] **Step 2: Add frontend regression proving the active query ID is passed only when real**

If `activeQuerySession.queryId === 'q1'`, EventTrail context has `searchSessionId: 'q1'`. Without a persisted history session, it is `null`, not a generated surrogate.

- [ ] **Step 3: Run focused tests**

```bash
PYTHONPATH=src pytest tests/event_trail/test_search_session_logging.py -q
cd frontend
CI=true npm test -- --runInBand src/features/search/components/SearchWorkspace.test.jsx
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/event_trail/test_search_session_logging.py frontend/src/features/search/components/SearchWorkspace.test.jsx
git commit -m "test: verify EventTrail interaction correlation"
```

---

### Task 9: Delete Legacy TemporalExploration and `exploration_seed`

**Files:**
- Delete: `frontend/src/api/exploration.js`
- Delete: `frontend/src/api/exploration.test.js`
- Delete: `frontend/src/features/alignment/hooks/useTemporalExploration.js`
- Delete: `frontend/src/features/alignment/hooks/useTemporalExploration.test.js`
- Delete: `frontend/src/features/alignment/components/ExplorationPanel.jsx`
- Delete: `frontend/src/features/alignment/components/ExplorationPanel.test.jsx`
- Delete: `src/hcmai/api/contracts/exploration.py`
- Delete: `src/hcmai/api/routers/exploration.py`
- Delete: `src/hcmai/orchestration/workflows/temporal_exploration.py`
- Modify: `src/hcmai/api/contracts/kis.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py`
- Modify: `frontend/src/styles/modal.css`
- Create: `tests/api/test_removed_exploration_routes.py`

**Interfaces:**
- `KISSearchResponse` after migration contains:

```text
intent
operation_summary
use_dense
use_bm25
results[] with result_id
latency
evidence_snapshot_id | null
warnings[]
```

and no `exploration_seed`.
- `/api/v1/exploration*` returns 404.
- `/api/v1/event-trail*` remains registered.

- [ ] **Step 1: Write the failing removed-route test before deletion**

Build the app/test client and assert current `/api/v1/exploration` exists; then write the desired test:

```python
response = client.post("/api/v1/exploration", json={})
assert response.status_code == 404
```

Also assert EventTrail `/open` is still routed (empty invalid body should be 422/503, not 404).

- [ ] **Step 2: Add KIS frontend/backend contract tests requiring no `exploration_seed`**

Frontend `searchKis` must accept the new response without that field. Backend contract serialization must not emit it.

- [ ] **Step 3: Run route/contract tests RED**

```bash
PYTHONPATH=src pytest tests/api/test_removed_exploration_routes.py -q
cd frontend
CI=true npm test -- --runInBand src/api/kis.test.js
```

Expected: old route still exists before deletion.

- [ ] **Step 4: Delete all legacy callers and exports**

Remove:

```python
ExplorationRegistry()
create_exploration_router(...)
KISExplorationSeed
KISExplorationEventSeed
exploration_seed=...
```

from production code. Delete `TemporalExploration` rather than keeping it as a fallback.

- [ ] **Step 5: Remove legacy CSS**

Delete `.exploration-*` rules from `frontend/src/styles/modal.css`. Do not remove `AlignmentAccordion`; it is still useful as compact read-only alignment on result cards.

- [ ] **Step 6: Run dead-reference scans**

```bash
rg -n "TemporalExploration|ExplorationRegistry|KISExplorationSeed|exploration_seed|create_exploration_router|/api/v1/exploration|useTemporalExploration|ExplorationPanel" src frontend \
  --glob '!frontend/build/**'
```

Expected: no production/test references except historical docs, if any.

- [ ] **Step 7: Run focused removal tests GREEN**

```bash
PYTHONPATH=src pytest tests/api/test_removed_exploration_routes.py -q
cd frontend
CI=true npm test -- --runInBand src/api/kis.test.js
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -A src/hcmai frontend/src tests/api/test_removed_exploration_routes.py
git commit -m "refactor: remove legacy temporal exploration"
```

---

### Task 10: Full EventTrail Acceptance and Production Build

**Files:**
- Modify only files that fail a verified acceptance case; no opportunistic refactor.

**Interfaces:**
- This task adds no new API or component contract.

- [ ] **Step 1: Run backend syntax/tests**

```bash
python -m compileall -q src/hcmai
PYTHONPATH=src pytest tests/api/test_event_trail_close.py tests/event_trail/test_search_session_logging.py tests/api/test_removed_exploration_routes.py -q
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend suites**

```bash
cd frontend
CI=true npm test -- --runInBand \
  src/api/client.test.js \
  src/api/eventTrail.test.js \
  src/api/kis.test.js \
  src/features/event-trail/hooks/useEventTrail.test.js \
  src/features/event-trail/components/EventTrailPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/frames/components/FrameCard.test.jsx \
  src/features/frames/components/ImageModal.test.jsx \
  src/App.test.jsx
```

Expected: PASS.

- [ ] **Step 3: Build the frontend**

```bash
cd frontend
npm run build
```

Expected: build exits 0. Do not hand-edit files under `frontend/build`.

- [ ] **Step 4: Manual live KIS → EventTrail walkthrough**

Use a multi-event query that produces at least E1/E2/E3 and verify in order:

```text
1. Search returns ranked cards with Alignment and EventTrail-capable result IDs.
2. Open one promising result.
3. `Open EventTrail` is visible even if MP4 playback fails and only keyframe preview is available.
4. Open Trail; revision 0 rail exactly matches the result card alignment.
5. Explore E2 seeks to E2 candidate without mutation.
6. Approve E1: E1 becomes anchored; player does not jump.
7. Decline E2: candidate changes or video becomes exhausted; only selected E2 auto-seeks on successful replacement.
8. If E1/E3 move indirectly, UI marks them but does not steal player focus.
9. Scrub player to a better E2 moment; Use resolves a canonical frame and anchors it.
10. Trail Submit becomes enabled and opens existing DRES dialog with that canonical timestamp.
11. Undo increments Trail revision and restores previous constraints.
12. Close modal with X, reopen same result: same active Trail session remains.
13. Back to results: Trail closes and ranked grid scroll/order remains intact.
14. New KIS search: old Trail is invalidated; new snapshot world is used.
15. Exhaust a result: card stays in the same rank with Exhausted annotation; no auto-jump to another result.
```

- [ ] **Step 5: Manual expiry/conflict checks**

Temporarily shorten TTL in development config or use a test harness and verify:

```text
expired unopened snapshot -> clear error asking to rerun search
expired active session -> clear error asking to reopen Trail
revision conflict -> state refresh, no automatic action retry
constraint conflict -> existing Trail state unchanged
```

- [ ] **Step 6: Verify no expensive work enters hot actions**

Capture backend logs around one open + Approve + Decline + Use + Undo. Confirm only KIS search creates the evidence snapshot; action calls do not invoke full-corpus scoring/retrieval or intent resolution.

- [ ] **Step 7: Final dead-code scan**

```bash
rg -n "TemporalExploration|ExplorationRegistry|KISExplorationSeed|exploration_seed|/api/v1/exploration|useTemporalExploration|ExplorationPanel" src frontend \
  --glob '!frontend/build/**'
```

Expected: zero live-code matches.

- [ ] **Step 8: Commit verification-only fixes if any**

If Steps 1–7 required a real code correction, commit only that verified correction. If no correction was needed, do not create an empty commit.

---

## Self-Review / Coverage Map

- **Exact KIS handoff:** Tasks 4 and 6.
- **EventTrail API + revisions/errors:** Tasks 1–3.
- **Event Rail + Evidence Inspector:** Task 5.
- **Explore / Use / Approve / Decline / ClearAnchor / Window / Undo:** Tasks 5–6.
- **Canonical frame resolution for Use:** Task 6.
- **Explicit Use-based submission:** Task 6.
- **Diff-aware focus behavior:** Tasks 5–7.
- **Exhaustion / last valid path:** Tasks 5–7.
- **Grid state and no reranking:** Task 7.
- **Search-session correlation and close lifecycle:** Tasks 2 and 8.
- **No EventTrail on replay/filter/no-snapshot:** Task 4.
- **Legacy TemporalExploration removal:** Task 9.
- **No full-corpus work in hot Trail loop:** Task 10 acceptance.
- **Production build / manual competition workflow:** Task 10.

