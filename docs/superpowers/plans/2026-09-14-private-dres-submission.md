# Direct Per-Participant DRES Submission Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every HCMAI answer workspace and answer-storage path so one edited KIS, AVS, or VQA answer travels directly from the participant's popup to the backend and then to DRES.

**Architecture:** The frontend owns one ephemeral in-memory dialog draft only; it never persists answers and has no bulk submission action. The backend accepts exactly one answer, resolves the live DRES task, validates the answer kind, checks the caller's expected task scope, and forwards one DRES submission through the private session cached for that `user_id`. Query history remains in SQLite, while answer candidates, submission attempts, shared workspace routes, and WebSockets are retired.

**Tech Stack:** React 19, Jest/React Testing Library, FastAPI, Pydantic v2, HTTPX DRES client, SQLite, pytest.

---

## Final agreed behavior

- There is no Answer Workspace for KIS, AVS, or VQA.
- There is no answer draft/result storage in backend SQLite, browser `localStorage`, browser `sessionStorage`, IndexedDB, or another cache.
- There is no `Submit all`, batch submission, answer queue, collaborative answer state, or submitted-answer history in HCMAI.
- `FrameCard` and `ImageModal` each expose a single submission action only when a VBS participant is connected.
- Clicking that action opens one editable popup. It does not store or send before the participant clicks `Submit` in the popup.
- The popup freezes the opaque task scope resolved when it opens. Clicking popup `Submit` sends that frozen `expected_task_scope_key` and does not refresh or replace it first; a task transition therefore fails safely at the backend with zero DRES calls.
- The popup defaults to `video_id,start_ms,end_ms` for every task type.
- `FrameCard` defaults to canonical `frame.video_id` and `frame.timestamp_ms`.
- `ImageModal` defaults to `frame.video_id` and the current player time rounded to whole milliseconds.
- A valid three-field temporal line is detected as `TEMPORAL`; every other non-blank single line is detected as `TEXT`. Blank or multiline input is rejected.
- KIS and AVS both accept exactly one temporal answer per request. VQA accepts exactly one text answer per request.
- Clicking popup `Submit` makes one HCMAI request and at most one DRES submission POST. There is no second confirmation.
- The browser sends `user_id`, an opaque `expected_task_scope_key`, and one answer. It never receives or sends a raw DRES `session_id`.
- The backend maps `user_id` to that participant's private cached DRES session. Sessions are never shared.
- HCMAI retains no durable record of the answer or DRES outcome. Closing/reloading the popup loses its draft by design.
- An `UNKNOWN` outcome stays visible in the open popup and is never retried automatically.

## VBS/DRES basis

- [Official VBS task description](https://videobrowsershowdown.org/about-vbs/) defines AVS as finding many independently scored instances and considers time, false submissions, and instance diversity.
- [Results of VBS 2025](https://arxiv.org/abs/2509.12000) reports many AVS submissions per participant, consistent with iterative instance-by-instance submission.
- [Official DRES repository](https://github.com/dres-dev/DRES) documents the flexible POST transport. Its array-shaped wire schema does not require HCMAI to bundle independent AVS instances.
- **Decision:** every HCMAI submission contains exactly one answer. AVS uses the same one-temporal-answer interaction as KIS; there is no client-side accumulation or bulk action.

## File map

### Backend: modify

- `src/hcmai/api/contracts/vbs.py` — replace workspace/revision contracts with safe current-task and one-answer submission contracts.
- `src/hcmai/api/contracts/__init__.py` — remove answer-workspace exports and export direct VBS contracts.
- `src/hcmai/api/routers/vbs.py` — replace workspace-coupled KIS/VQA/AVS routes with task lookup and one stateless direct route.
- `src/hcmai/vbs/service.py` — add temporal range construction and forbid replaying a submission POST after DRES 401.
- `src/hcmai/api/history.py` — migrate SQLite to query-history-only schema v3.
- `src/hcmai/api/routers/__init__.py` — remove answer-workspace router export.
- `src/hcmai/app.py` — stop registering shared answer routes/WebSocket; retain `WorkspaceStore` for query history.

### Backend: delete

- `src/hcmai/api/contracts/workspace.py`
- `src/hcmai/api/routers/workspace.py`
- `tests/test_submission_reservations.py`
- `tests/api/test_answer_workspace_routes.py`
- `tests/api/test_answer_workspace_broadcast.py`

### Frontend: create

- `frontend/src/api/submissions.js`
- `frontend/src/api/submissions.test.js`
- `frontend/src/api/client.test.js`
- `frontend/src/features/submission/answerFormat.js`
- `frontend/src/features/submission/answerFormat.test.js`
- `frontend/src/features/submission/components/SubmissionDialog.jsx`
- `frontend/src/features/submission/components/SubmissionDialog.test.jsx`
- `frontend/src/features/submission/hooks/useDirectSubmission.js`
- `frontend/src/features/submission/hooks/useDirectSubmission.test.jsx`
- `frontend/src/features/submission/index.js`

### Frontend: modify

- `frontend/src/api/client.js` — preserve allowlisted structured backend error codes.
- `frontend/src/App.jsx` and `frontend/src/App.test.jsx` — own the ephemeral popup and direct-submit flow.
- `frontend/src/features/vbs/contexts/VbsSessionContext.jsx` and `.test.jsx` — support explicit invalidation after DRES rejects a cached session.
- `frontend/src/features/frames/components/FrameCard.jsx` and `.test.jsx` — remove candidate/submitted state and open direct submission.
- `frontend/src/features/frames/components/ImageModal.jsx` and `.test.jsx` — open direct submission at the current player time.
- `frontend/src/features/frames/components/FramesBox.jsx` — pass the direct action.
- `frontend/src/features/search/components/SearchWorkspace.jsx` and `.test.jsx` — replace candidate props with direct-submit props.
- `frontend/src/styles/workspace.css`, `frame-interactions.css`, and `modal-inspector.css` — remove shared workspace styles and style only the popup/action/status.
- `frontend/src/features/docs/components/ApiDocsModal.jsx` and related tests — document direct submission and removed persistence.

### Frontend: delete

- `frontend/src/api/answerWorkspace.js`
- `frontend/src/api/answerWorkspace.test.js`
- `frontend/src/features/answer-workspace/contexts/AnswerWorkspaceContext.jsx`
- `frontend/src/features/answer-workspace/contexts/AnswerWorkspaceContext.test.jsx`
- `frontend/src/features/answer-workspace/components/AnswerWorkspace.jsx`
- `frontend/src/features/answer-workspace/components/AnswerWorkspace.test.jsx`
- `frontend/src/features/answer-workspace/components/AnswerCandidateDialog.jsx`
- `frontend/src/features/answer-workspace/components/AnswerCandidateDialog.test.jsx`

## Chunk 1: One-answer backend contract and forwarding

### Task 1: Define strict current-task and one-answer contracts

**Files:**

- Modify: `src/hcmai/api/contracts/vbs.py`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Test: `tests/api/test_vbs_submission_routes.py`

- [ ] **Step 1: Write failing model tests**

Valid requests:

```python
{
    "user_id": "member-1",
    "expected_task_scope_key": "dres-task-v1:opaque",
    "answer": {
        "kind": "TEMPORAL",
        "video_id": "L21_V001",
        "start_ms": 1234,
        "end_ms": 1234,
    },
}

{
    "user_id": "member-1",
    "expected_task_scope_key": "dres-task-v1:opaque",
    "answer": {"kind": "TEXT", "text": "three"},
}
```

Reject blank fields, negative/non-integer timestamps, `end_ms < start_ms`, extra fields, missing `answer`, and any plural/bulk `answers` array.

- [ ] **Step 2: Run RED test**

Run: `aic\Scripts\python.exe -m pytest tests/api/test_vbs_submission_routes.py -q`

Expected: direct contracts do not yet exist.

- [ ] **Step 3: Implement discriminated request and response models**

Implement the equivalent of:

```python
class VbsTemporalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["TEMPORAL"]
    video_id: NonBlank
    start_ms: int = Field(ge=0, strict=True)
    end_ms: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_range(self) -> "VbsTemporalAnswer":
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self

class VbsTextAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["TEXT"]
    text: NonBlank

class VbsDirectSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: NonBlank
    expected_task_scope_key: NonBlank
    answer: Annotated[VbsTemporalAnswer | VbsTextAnswer, Field(discriminator="kind")]
```

Return a discriminated union with these invariants:

```text
RECORDED     -> recorded=true,  official verdict required
NOT_RECORDED -> recorded=false, verdict=null, reason required
UNKNOWN      -> recorded=null,  verdict=null
```

Official verdicts are `CORRECT`, `WRONG`, `INDETERMINATE`, and `UNDECIDABLE`. Reasons include `DRES_REJECTED` and `DRES_AUTH_REJECTED`. Add tests that contradictory combinations cannot validate.

- [ ] **Step 4: Run GREEN model tests**

Run the same pytest command. Expected: model cases pass; route cases remain red.

- [ ] **Step 5: Commit**

```text
git add src/hcmai/api/contracts/vbs.py src/hcmai/api/contracts/__init__.py tests/api/test_vbs_submission_routes.py
git commit -m "refactor: define one-answer DRES contracts"
```

### Task 2: Implement live-task lookup and stateless direct submission

**Files:**

- Modify: `src/hcmai/api/routers/vbs.py`
- Modify: `src/hcmai/vbs/service.py`
- Test: `tests/api/test_vbs_submission_routes.py`
- Test: `tests/vbs/test_service.py`
- Test: `tests/vbs/test_client.py`

- [ ] **Step 1: Write failing route/service tests**

Cover:

- `GET /api/v1/vbs/task/{user_id}` returns safe evaluation/task metadata without credentials or session token.
- `POST /api/v1/vbs/submit` sends exactly one answer for KIS, AVS, or VQA.
- KIS/AVS reject text; VQA rejects temporal; unknown task types fail closed.
- Backend maps `video_id` through `media_item_name` and preserves exact integer `start_ms/end_ms`.
- Stale `expected_task_scope_key` returns HTTP 409 with `detail: {code: "TASK_SCOPE_MISMATCH", message: "..."}` and zero DRES submit calls.
- Disconnected participant errors have the same structured `detail.code/message` shape.
- The `user_id` passed to the route is the one passed to `DresService.submit`; two users use separate cached sessions.
- A DRES 2xx response becomes `RECORDED` regardless of judge verdict.
- A definitive rejection becomes structured `NOT_RECORDED`; ambiguous delivery becomes structured `UNKNOWN`.
- A DRES submission 401 makes exactly one DRES POST, evicts only that participant session, and returns `NOT_RECORDED/DRES_AUTH_REJECTED` without replay.

- [ ] **Step 2: Run RED tests**

Run: `aic\Scripts\python.exe -m pytest tests/api/test_vbs_submission_routes.py tests/vbs/test_service.py tests/vbs/test_client.py -q`

- [ ] **Step 3: Implement the routes**

Replace `/submit/kis`, `/submit/vqa`, `/submit/avs`, and attempt-resolution routes with:

```text
GET  /api/v1/vbs/task/{user_id}
POST /api/v1/vbs/submit
```

Submission order:

1. Require a connected `user_id`.
2. Resolve live DRES scope.
3. Compare the expected opaque task scope; fail before DRES on mismatch.
4. Compare detected answer kind with normalized live `taskType`.
5. Build exactly:

```python
ApiClientSubmission(
    answer_sets=[
        ApiClientAnswerSet(
            task_name=scope.task_name,
            answers=[mapped_answer],
        )
    ]
)
```

6. Call `DresService.submit(user_id, scope.evaluation_id, submission)` once.
7. Return a secret-free structured outcome with no workspace or attempt ID.

Use 2xx HCMAI responses for `RECORDED`, `NOT_RECORDED`, and `UNKNOWN`, allowing the existing JSON client to retain semantic outcomes. Use non-2xx only for malformed input, task mismatch, disconnected state, or backend misconfiguration.

Add `temporal_range_answer(video_id, start_ms, end_ms)`. Submission POSTs must not use the read-operation 401 replay behavior; safe read operations may retain one-time reauthentication.

- [ ] **Step 4: Run GREEN backend tests**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add src/hcmai/api/routers/vbs.py src/hcmai/vbs/service.py tests/api/test_vbs_submission_routes.py tests/vbs/test_service.py tests/vbs/test_client.py
git commit -m "feat: forward one participant answer directly to DRES"
```

## Chunk 2: Remove backend answer persistence and sharing

### Task 3: Migrate SQLite to query-history-only schema v3

**Files:**

- Modify: `src/hcmai/api/history.py`
- Rename/modify: `tests/test_answer_workspace_store.py` to `tests/test_workspace_store_migrations.py`
- Delete: `tests/test_submission_reservations.py`

- [ ] **Step 1: Write failing migration tests**

Create v0, v1, and v2 fixture databases. After initialization assert:

- `query_history` and viewed-frame values remain unchanged;
- `answer_workspace_state`, `answer_candidates`, `submission_attempts`, and legacy `submission_files` do not exist;
- `PRAGMA user_version == 3`;
- database browsing allowlists only `query_history`;
- repeated initialization is idempotent.

- [ ] **Step 2: Run RED migration tests**

Run: `aic\Scripts\python.exe -m pytest tests/test_workspace_store_migrations.py -q`

- [ ] **Step 3: Remove answer-store methods and migrate atomically**

Set `_DATABASE_VERSION = 3`, remove answer dataclasses/helpers/mutations/reservations, reduce `_DATABASE_TABLE_ORDER` to query history, and drop answer-only tables within the existing `BEGIN IMMEDIATE` migration. Do not drop or semantically rewrite `query_history`.

- [ ] **Step 4: Run GREEN history/database tests**

Run: `aic\Scripts\python.exe -m pytest tests/test_workspace_store_migrations.py tests/api -q`

- [ ] **Step 5: Commit**

```text
git add src/hcmai/api/history.py tests/test_workspace_store_migrations.py
git rm tests/test_answer_workspace_store.py tests/test_submission_reservations.py
git commit -m "refactor: remove persisted answers"
```

### Task 4: Remove shared workspace API and WebSocket

**Files:**

- Delete: `src/hcmai/api/contracts/workspace.py`
- Delete: `src/hcmai/api/routers/workspace.py`
- Delete: `tests/api/test_answer_workspace_routes.py`
- Delete: `tests/api/test_answer_workspace_broadcast.py`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py`
- Modify: `tests/api/test_removed_routes.py`

- [ ] **Step 1: Extend removed-route tests**

Assert shared answer HTTP/WebSocket endpoints, old mode-specific submit routes, and attempt resolution are absent. Assert VBS session/task/direct-submit and query-history routes still exist.

- [ ] **Step 2: Run RED route-removal test**

Run: `aic\Scripts\python.exe -m pytest tests/api/test_removed_routes.py -q`

- [ ] **Step 3: Delete legacy code and registrations**

Remove answer events, socket hub, broadcasts, workspace route exports, and app registration. Keep `WorkspaceStore` initialization because query history still uses it.

- [ ] **Step 4: Run GREEN API suite**

Run: `aic\Scripts\python.exe -m pytest tests/api -q`

- [ ] **Step 5: Commit**

```text
git add src/hcmai/app.py src/hcmai/api/contracts/__init__.py src/hcmai/api/routers/__init__.py tests/api/test_removed_routes.py
git rm src/hcmai/api/contracts/workspace.py src/hcmai/api/routers/workspace.py tests/api/test_answer_workspace_routes.py tests/api/test_answer_workspace_broadcast.py
git commit -m "refactor: remove shared answer workspace API"
```

## Chunk 3: Ephemeral frontend direct submission

### Task 5: Implement deterministic one-line answer parsing

**Files:**

- Create: `frontend/src/features/submission/answerFormat.js`
- Create: `frontend/src/features/submission/answerFormat.test.js`

- [ ] **Step 1: Write failing pure-function tests**

```javascript
formatTemporalAnswer({ videoId: 'L21_V001', startMs: 12346, endMs: 12346 })
// 'L21_V001,12346,12346'

parseAnswerLine('L21_V001,12346,12346')
// { kind: 'TEMPORAL', video_id: 'L21_V001', start_ms: 12346, end_ms: 12346 }

parseAnswerLine('three')
// { kind: 'TEXT', text: 'three' }
```

Only blank/multiline input is generically invalid. A malformed temporal-looking single line, including extra commas or invalid numbers, becomes TEXT. A temporal answer requires exactly three fields, a non-blank video ID, safe non-negative integers, and `end >= start`.

- [ ] **Step 2: Run RED Jest test**

Run from `frontend`: `npm test -- --watchAll=false --runTestsByPath src/features/submission/answerFormat.test.js`

- [ ] **Step 3: Implement side-effect-free formatter/parser**

Return API-ready snake_case objects. Do not read or write browser storage.

- [ ] **Step 4: Run GREEN Jest test**

Run the same command. Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add frontend/src/features/submission/answerFormat.js frontend/src/features/submission/answerFormat.test.js
git commit -m "feat: parse direct DRES answer lines"
```

### Task 6: Add safe browser task/submission APIs

**Files:**

- Create: `frontend/src/api/submissions.js`
- Create: `frontend/src/api/submissions.test.js`
- Create: `frontend/src/api/client.test.js`
- Modify: `frontend/src/api/client.js`
- Modify: `frontend/src/api/vbs.test.js`

- [ ] **Step 1: Write failing API tests**

Assert safe task lookup, exact one-answer POST JSON, bulk-array rejection, strict discriminated outcome normalization, and no secret leakage. Extend `requestJson` to copy an allowlisted FastAPI `detail.code` into `error.code` while retaining status/message.

- [ ] **Step 2: Run RED API tests**

Run from `frontend`: `npm test -- --watchAll=false --runTestsByPath src/api/submissions.test.js src/api/client.test.js src/api/vbs.test.js`

- [ ] **Step 3: Implement only two public functions**

```javascript
getCurrentDresTask(userId, { signal } = {})
submitDresAnswer({ userId, expectedTaskScopeKey, answer, signal } = {})
```

Do not create answer cache, history, queue, retry, or bulk helpers.

- [ ] **Step 4: Run GREEN API tests**

Run the same command. Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add frontend/src/api/client.js frontend/src/api/client.test.js frontend/src/api/submissions.js frontend/src/api/submissions.test.js frontend/src/api/vbs.test.js
git commit -m "feat: add one-answer DRES browser client"
```

### Task 7: Build the ephemeral popup and direct-submit hook

**Files:**

- Create: `frontend/src/features/submission/components/SubmissionDialog.jsx`
- Create: `frontend/src/features/submission/components/SubmissionDialog.test.jsx`
- Create: `frontend/src/features/submission/hooks/useDirectSubmission.js`
- Create: `frontend/src/features/submission/hooks/useDirectSubmission.test.jsx`
- Create: `frontend/src/features/submission/index.js`
- Modify: `frontend/src/features/vbs/contexts/VbsSessionContext.jsx`
- Modify: `frontend/src/features/vbs/contexts/VbsSessionContext.test.jsx`
- Modify: `frontend/src/styles/workspace.css`

- [ ] **Step 1: Write failing dialog/hook tests**

Cover:

- popup opens with editable temporal line for KIS, AVS, and VQA;
- edited valid temporal line becomes TEMPORAL; replacement text becomes TEXT;
- blank/multiline input is blocked;
- opening the popup resolves and freezes its expected task scope; popup `Submit` performs only the one submission POST using that frozen scope;
- if DRES changes task after the popup opens, the POST still carries the original scope, backend returns `TASK_SCOPE_MISMATCH`, the edited value remains visible, and DRES receives zero calls;
- only `RECORDED` closes and discards the draft;
- `NOT_RECORDED`, `UNKNOWN`, HTTP errors, and network ambiguity keep the popup/value visible with an error/status;
- no automatic retry under any outcome;
- closing the popup or reloading loses the draft and writes no browser storage;
- DRES auth rejection invalidates the VBS session, clears `VBS_USER_ID_STORAGE_KEY`, and requires explicit Connect; remount does not auto-connect.

- [ ] **Step 2: Run RED UI tests**

Run from `frontend`: `npm test -- --watchAll=false --runTestsByPath src/features/submission/components/SubmissionDialog.test.jsx src/features/submission/hooks/useDirectSubmission.test.jsx src/features/vbs/contexts/VbsSessionContext.test.jsx`

- [ ] **Step 3: Implement ephemeral state only**

The hook may hold current task, the popup's frozen task scope, popup draft, busy state, error, and last visible response in React memory. Resolve/refresh task metadata before opening, then freeze it with the draft. Popup `Submit` must not perform another task lookup; it issues only `submitDresAnswer` with the frozen key. The hook must not call `localStorage`, `sessionStorage`, IndexedDB, or backend answer APIs. Disable duplicate clicks while a request is in flight and abort only obsolete read-only task lookups; do not abort/retry a submission merely because the popup closes.

- [ ] **Step 4: Run GREEN UI tests**

Run the same command. Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add frontend/src/features/submission frontend/src/features/vbs/contexts/VbsSessionContext.jsx frontend/src/features/vbs/contexts/VbsSessionContext.test.jsx frontend/src/styles/workspace.css
git commit -m "feat: add ephemeral direct submission popup"
```

### Task 8: Wire FrameCard and ImageModal directly to the popup

**Files:**

- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/App.test.jsx`
- Modify: `frontend/src/features/frames/components/FrameCard.jsx`
- Modify: `frontend/src/features/frames/components/FrameCard.test.jsx`
- Modify: `frontend/src/features/frames/components/ImageModal.jsx`
- Modify: `frontend/src/features/frames/components/ImageModal.test.jsx`
- Modify: `frontend/src/features/frames/components/FramesBox.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/styles/frame-interactions.css`
- Modify: `frontend/src/styles/modal-inspector.css`

- [ ] **Step 1: Rewrite entry-point tests first**

Assert:

- disconnected participants cannot submit and receive a clear Connect requirement;
- FrameCard freezes canonical `frame.video_id/frame.timestamp_ms` even if display detail differs;
- ImageModal freezes `Math.round(video.currentTime * 1000)` synchronously;
- both actions only open the popup and stop card click propagation;
- one popup Submit produces one HCMAI request;
- no AVS-specific add/workspace/bulk path exists;
- no candidate/submitted CSS class derives from answer state.

- [ ] **Step 2: Run RED focused tests**

Run from `frontend`: `npm test -- --watchAll=false --runTestsByPath src/App.test.jsx src/features/frames/components/FrameCard.test.jsx src/features/frames/components/ImageModal.test.jsx src/features/search/components/SearchWorkspace.test.jsx`

- [ ] **Step 3: Replace candidate props/context with direct submission**

Render one dialog at app-shell level. Use intent-revealing props such as `submissionAction/onOpenSubmission`. Keep search/exploration behavior unchanged.

- [ ] **Step 4: Run GREEN focused tests**

Run the same command. Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add frontend/src/App.jsx frontend/src/App.test.jsx frontend/src/features/frames frontend/src/features/search frontend/src/styles/frame-interactions.css frontend/src/styles/modal-inspector.css
git commit -m "feat: submit frames directly through DRES popup"
```

## Chunk 4: Delete legacy frontend and verify

### Task 9: Delete shared/local answer workspace code and stale styling

**Files:**

- Delete every file under `frontend/src/features/answer-workspace/`.
- Delete `frontend/src/api/answerWorkspace.js` and `.test.js`.
- Modify related imports/styles/tests found by repository search.

- [ ] **Step 1: Search before deletion**

Run:

```text
rg -n "AnswerWorkspace|answerWorkspace|answer-workspace|Submit all|localAvsWorkspace|AvsWorkspace|candidate_id|expected_workspace_revision|submission-attempts" frontend/src
```

- [ ] **Step 2: Delete legacy files and remove all runtime consumers**

Remove shared labels, WebSocket status, candidate mutation UI, AVS mode toggles, confirmation UI, submission attempt resolution, collapse key, and answer-derived frame styles. Do not introduce a replacement workspace.

- [ ] **Step 3: Run full frontend suite**

Run from `frontend`: `npm test -- --watchAll=false`

Expected: PASS without WebSocket/open-handle warnings.

- [ ] **Step 4: Build frontend**

Run from `frontend`: `npm run build`

Expected: production build succeeds.

- [ ] **Step 5: Commit**

```text
git rm frontend/src/api/answerWorkspace.js frontend/src/api/answerWorkspace.test.js frontend/src/features/answer-workspace
git add frontend/src/styles
git commit -m "refactor: delete answer workspace frontend"
```

### Task 10: Update docs and perform end-to-end verification

**Files:**

- Modify: `frontend/src/features/docs/components/ApiDocsModal.jsx`
- Modify: `frontend/src/features/docs/components/ApiDocsModal.test.jsx`
- Modify: `frontend/src/features/docs/components/OperatorRunbook.test.js`
- Modify: `README.md` only where it mentions removed answer routes/workspaces.
- Modify: `.env.example` only if it names answer-specific storage settings; retain `HCMAI_WORKSPACE_DB` for query history.

- [ ] **Step 1: Update documentation tests and content**

Document the one-answer route, ephemeral popup, no answer persistence, no Submit all, session isolation, task-scope protection, and UNKNOWN warning.

- [ ] **Step 2: Run documentation tests**

Run from `frontend`: `npm test -- --watchAll=false --runTestsByPath src/features/docs/components/ApiDocsModal.test.jsx src/features/docs/components/OperatorRunbook.test.js`

- [ ] **Step 3: Run all backend verification**

```text
aic\Scripts\python.exe -m pytest -q
aic\Scripts\python.exe -m compileall -q src/hcmai
```

Expected: PASS.

- [ ] **Step 4: Run all frontend verification**

From `frontend`:

```text
npm test -- --watchAll=false
npm run build
```

Expected: PASS.

- [ ] **Step 5: Prove removed surfaces are absent**

Run:

```text
rg -n "AnswerWorkspace|answer_workspace_state|answer_candidates|submission_attempts|create_answer_workspace_router|Submit all|localAvsWorkspace|AvsWorkspace" src tests frontend/src
```

Expected: only intentional schema migration/removed-route assertions, or no matches.

- [ ] **Step 6: Hand-check with `dres-mock-server` and two users**

1. Connect `member-1` and `member-2` in separate browser profiles.
2. Submit KIS from FrameCard; verify one DRES request with one temporal answer and member-1's session fingerprint.
3. Seek in ImageModal and submit; verify exact rounded player timestamp.
4. Switch mock to AVS and submit three different frames separately; verify three independent DRES requests, never one multi-answer request.
5. Switch mock to VQA, replace the temporal line with text, and verify one text-answer request.
6. Simulate `CORRECT`, `WRONG`, `INDETERMINATE`, `UNDECIDABLE`, rejection, 401, malformed response, and timeout.
7. Verify only `RECORDED` closes the popup; all other outcomes retain the draft and never retry.
8. Verify submission 401 makes one DRES POST, evicts only that user's session, clears browser restoration identity, and requires explicit reconnect.
9. Reload and inspect browser storage plus SQLite; verify no answer or submission result was persisted.

- [ ] **Step 7: Verify migration on a disposable v2 database copy**

Confirm query history remains and answer tables disappear. Never use the only production database copy for this check.

- [ ] **Step 8: Commit**

```text
git add frontend/src/features/docs README.md .env.example
git commit -m "docs: describe direct per-participant DRES submission"
```

## Completion criteria

- Every popup submission creates one HCMAI request and at most one DRES POST.
- Every DRES payload contains one answer only.
- KIS and AVS accept one temporal answer; VQA accepts one text answer.
- No Submit all or AVS answer accumulation exists.
- No HCMAI answer draft/result persistence exists on frontend or backend.
- Query history remains intact.
- Raw DRES session tokens never cross the backend boundary.
- Task scope mismatch makes zero DRES calls.
- DRES submission 401 is never replayed.
- UNKNOWN is visible and never automatically retried.
- Backend tests, frontend tests, frontend build, compile checks, migration checks, and two-user DRES mock checks pass.

## Compatibility and risk notes

- Schema v3 intentionally and irreversibly deletes old answer candidates and submission attempts. Back up a production SQLite file before first deployment if those records may still be useful.
- Closing/reloading the popup discards its draft because answer persistence is explicitly forbidden.
- A timeout may mean DRES recorded an answer even though HCMAI reports `UNKNOWN`. The participant must check DRES before manually submitting again.
- `user_id` remains a browser-supplied selector backed by server-side credential mapping. This plan guarantees session separation but does not add a new participant authentication system.
- `KNOWLEDGE.md` records the VBS evidence and final direct one-instance-per-request decision under `VBS AVS submissions are scored as individual instances`.
