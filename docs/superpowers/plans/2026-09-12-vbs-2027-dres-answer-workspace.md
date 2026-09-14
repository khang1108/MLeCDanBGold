# VBS 2027 DRES and Answer Workspace Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuyển giao diện thi AIC hiện tại thành giao diện VBS 2027 chỉ còn các luồng KIS/image/filter hiện hữu, bổ sung VQA text answer, AVS multi-frame answer workspace, DRES v2 authentication/submission/result logging, và loại bỏ toàn bộ submission-file workflow cũ.

**Architecture:** FE vẫn gọi duy nhất HCMAI BE. BE giữ credential và session DRES theo từng VBS `user_id`, chuyển `video_id` cùng timestamp được chọn chính xác thành payload DRES v2, đồng thời tiếp tục dùng SQLite + WebSocket để đồng bộ một answer workspace có cấu trúc giữa các thành viên. Retrieval không đổi; chỉ transport/API adapters, collaboration state và submission UI thay đổi.

**Tech Stack:** React 18 + Jest/Testing Library; FastAPI + Pydantic v2 + httpx; SQLite; WebSocket; DRES Client API v2.

---

## Chunk 1: Decisions, boundaries, and source map

### Confirmed external contracts

- VBS 2027 has KIS (visual/text/chat), AVS, and VQA; TRAKE is not a VBS task.
- DRES v2 login: `POST /api/v2/login` with `{ "username", "password" }`; response contains `sessionId`.
- Active evaluations: `GET /api/v2/client/evaluation/list?session=...`.
- Current task: `GET /api/v2/client/evaluation/currentTask/{evaluationId}?session=...`.
- Submission: `POST /api/v2/submit/{evaluationId}?session=...`.
- Submission body:

```json
{
  "answerSets": [
    {
      "taskId": "optional-current-task-id",
      "answers": [
        {
          "mediaItemName": "organizer-media-id",
          "start": 123000,
          "end": 123000
        }
      ]
    }
  ]
}
```

- VQA uses an answer with only `{ "text": "..." }`.
- AVS uses one answer set containing every selected temporal answer. KIS uses one answer set containing exactly one temporal answer.
- Result logging: `POST /api/v2/log/result/{evaluationId}?session=...` with `QueryResultLog`.
- `QueryResultLog.timestamp` and `QueryEvent.timestamp` are interaction wall-clock epoch milliseconds. `answer.start`/`answer.end` are media timeline milliseconds. These are distinct coordinate systems.
- DRES permits nullable `end`, but this implementation intentionally submits point answers with `start === end === timestamp_ms`. Do not add an end-policy setting until VBS requires another shape.

### Scope decisions for this plan

1. Remove TRAKE from FE only. Do not remove `src/hcmai/orchestration/workflows/trake.py`, `/api/v1/trake`, temporal algorithms, configs, or backend TRAKE tests in this change; the user explicitly scoped TRAKE removal to FE and asked that other flows remain unchanged.
2. Delete the old submission-file feature in both FE and BE, including CSV editing/download, validation, optimistic file revisions, related query-history fields, `/api/v1/submission-files`, and old file WebSocket commands.
3. Keep WebSocket collaboration infrastructure, but replace file payloads with typed answer candidates. This is a replacement feature, not a compatibility layer.
4. Keep Query History and frame-view tracking. Remove only submission-file associations from history. Query History becomes KIS-only on FE; legacy TRAKE snapshots may render as unsupported instead of keeping TRAKE rendering code.
5. Do not use `displayVideoId()` or `frame_idx` for DRES. Add a dedicated, testable mapping from `video_id` to the organizer's exact `mediaItemName`; use the selected media `timestamp_ms` unchanged for both `start` and `end`.
6. The AVS switch is shared workspace state. When OFF, frame and text candidates may coexist and each row has its own `Submit KIS` or `Submit VQA` action. When ON, only frame candidates are eligible and the global action submits all non-submitted frame candidates.
7. A successful search result must remain visible even when optional DRES logging fails. Logging failure is reported as a non-blocking UI status and backend warning.
8. DRES secrets and session tokens never reach browser state, localStorage, API JSON, WebSocket events, or application logs. FE sends only VBS `user_id` to HCMAI BE.
9. AVS submits every current eligible frame in exactly one DRES request: one `answerSet` containing multiple `answers`. Do not implement a sequential submission mode.
10. Adding an answer to the workspace and submitting it to DRES are separate actions. Result/inspector controls open an editable confirmation dialog and Enter saves the answer candidate; only workspace submit controls call DRES.
11. An inspector candidate captures `Math.round(video.currentTime * 1000)` at click time. A result `FrameCard` candidate uses that card's exact `timestamp_ms`. Neither path derives a DRES timestamp from `frame_id` or `frame_idx`, and the live value is not snapped to the nearest keyframe.

### Credential placement decision

Do **not** put DRES username/password in the shared Workspace tab. Workspace is
team-visible state, while a DRES credential belongs to one participant/session.

An operator loads the small `user_id -> DRES username/password` map into BE
secrets before the session. Each browser enters only its own `user_id` in the
existing header; BE resolves the credential, logs into DRES, caches the session
in memory, and returns only `OK`. There is no username/password form or login
popup in FE. A missing mapping is a BE configuration error that must be shown
clearly without exposing secret values.

### VBS 2026 compatibility baseline

- The VBS 2026 public task contract is a valid provisional baseline: KIS submits
  exactly one segment, AVS submits many segments, Q/A submits plaintext, and
  KIS/AVS identify a segment by video ID plus timestamp.
- The official VBS DRES integration page still demonstrates client generation
  from DRES `2.0.1`; the current official OpenAPI is `2.0.5-SNAPSHOT`. Implement
  against the minimal overlapping v2 subset and run contract tests against both
  specifications rather than depending on master-only fields.
- V3C assigns sequential numeric IDs across its three partitions and its source
  files look like `00001.mp4`. This supports identity mapping as a useful test
  hypothesis, but it does **not** prove that the VBS campaign imported the DRES
  media item under exactly `00001`.
- Public VBS 2026 material does not prove the campaign-specific DRES
  `mediaItemName`. Therefore 2026 can unblock implementation and local tests,
  but cannot remove that DRES staging gate.
  DRES v2 explicitly models `answerSet.answers` as an array, so this plan treats
  one-request AVS batch submission as decided rather than optional.

### Data and UI invariants

```text
selected video moment
  video_id       -> dedicated VBS media-name mapper -> mediaItemName
  timestamp_ms   -> DRES start and end (equal point timestamp)
  frame_id       -> optional source provenance; never a submission coordinate
  frame_idx      -> display/debug only; never DRES submission coordinate
```

- Workspace frame uniqueness key for the currently displayed workspace: `(video_id, timestamp_ms)`.
- A FRAME candidate may originate from an exact result frame or an arbitrary live video moment. Therefore `frame_id` is nullable provenance: card actions retain it, while inspector actions preserve the exact playback timestamp without resolving to a nearby canonical frame.
- Workspace text uniqueness is not forced; two team members may propose the same text independently.
- Each candidate stores contributor `user_id`, creation time, and revision.
- Submission always resolves the session from the user who clicked Submit, not the candidate contributor.
- Workspace hydration, row rendering, AVS confirmation, and AVS payload use one deterministic order: `(created_at_ms ASC, candidate_id ASC)`.
- A mode change is revision-checked and broadcast so every browser sees the same AVS state.
- At most one DRES submission reservation exists for the shared workspace. A `FORWARDING` or unresolved `UNKNOWN` reservation blocks edits, mode/task changes, clear, and further submits so the reviewed snapshot remains immutable.
- Workspace state is bound to the active DRES `(evaluation_id, task_id)`. If DRES moves to another task while answers remain, BE rejects mutation/submission with `TASK_SCOPE_MISMATCH`; FE requires an explicit `Clear and switch task` confirmation. If the workspace is empty, BE may bind it to the newly active task automatically. Never submit retained answers against a newly resolved task silently.
- Switching AVS OFF does not delete frame candidates. Text candidates are hidden/ineligible while AVS is ON. Nothing is silently destroyed.
- Workspace rows visibly contain only the answer that will be submitted: FRAME renders `<video_id>,<timestamp_ms>,<timestamp_ms>` (`start == end`); TEXT renders its answer text. Operational metadata remains in state for synchronization/audit but is not shown in the row body.
- Clicking a row opens the same typed answer editor. Enter validates and updates workspace state; Escape cancels. Row action buttons must stop propagation.
- In non-AVS mode, every row has a compact submit icon and a red delete icon. FRAME rows additionally have an open-viewer icon that reuses the existing video viewer at the candidate's exact timestamp. TEXT rows have no viewer action.
- In AVS mode, frame rows have open-viewer and red delete icons but no per-row submit icon; one `Submit all` action submits the entire eligible list.
- Accepted submission candidates are marked submitted with DRES status; they are not immediately erased. `Clear workspace` is explicit and applies to the current task scope.

### File structure

**Create**

- `src/hcmai/vbs/__init__.py` — VBS/DRES public service exports.
- `src/hcmai/vbs/config.py` — DRES URL, credential map, timeout, evaluation selection, logging flag, media-name mapping settings.
- `src/hcmai/vbs/models.py` — minimal validated subset of the official DRES v2 request/response schemas.
- `src/hcmai/vbs/client.py` — raw httpx adapter; owns DRES HTTP paths and error translation only.
- `src/hcmai/vbs/service.py` — session cache, active evaluation/task resolution, payload mapping, submit and result-log orchestration.
- `src/hcmai/api/contracts/vbs.py` — HCMAI FE-to-BE session, submission, and log-status contracts.
- `src/hcmai/api/contracts/workspace.py` — answer workspace records and WebSocket commands/events.
- `src/hcmai/api/routers/vbs.py` — session and submission endpoints.
- `src/hcmai/api/routers/workspace.py` — answer hydration and WebSocket collaboration endpoints.
- `tests/vbs/test_config.py`, `tests/vbs/test_models.py`, `tests/vbs/test_client.py`, `tests/vbs/test_service.py`.
- `tests/api/test_vbs_routes.py`, `tests/api/test_answer_workspace_routes.py`.
- `frontend/src/api/vbs.js`, `frontend/src/api/vbs.test.js`.
- `frontend/src/api/history.js`, `frontend/src/api/history.test.js`.
- `frontend/src/api/answerWorkspace.js`, `frontend/src/api/answerWorkspace.test.js`.
- `frontend/src/features/vbs-session/contexts/VbsSessionContext.jsx` and test.
- `frontend/src/features/vbs-session/components/VbsUserControl.jsx` and test.
- `frontend/src/features/answer-workspace/contexts/AnswerWorkspaceContext.jsx` and test.
- `frontend/src/features/answer-workspace/components/AnswerWorkspace.jsx` and test.
- `frontend/src/features/answer-workspace/components/AnswerCandidate.jsx` and test.
- `frontend/src/features/answer-workspace/components/AnswerCandidateDialog.jsx` and test — editable confirmation before adding/updating workspace state.
- `frontend/src/features/answer-workspace/components/DresSubmissionConfirmDialog.jsx` and test — immutable final confirmation before the BE forwards to DRES.
- `frontend/src/styles/vbs.css`, `frontend/src/styles/answer-workspace.css`.

**Modify**

- `.env.example`, `README.md`.
- `src/hcmai/app.py`, `src/hcmai/api/contracts/__init__.py`, `src/hcmai/api/routers/__init__.py`.
- `src/hcmai/api/history.py`, `src/hcmai/api/contracts/history.py`, `src/hcmai/api/routers/history.py`.
- `src/hcmai/api/routers/search.py` to schedule result logging after successful text/image/filter responses.
- `src/hcmai/api/contracts/search.py` and `filter.py` only if a small non-ranking log-status field is selected; prefer response headers to avoid changing retrieval payloads.
- `frontend/src/App.jsx`, `App.test.jsx`.
- `frontend/src/features/header/AppHeader.jsx` and tests.
- `frontend/src/api/search.js`, `filter.js` and tests to attach `X-VBS-User-ID`.
- `frontend/src/features/search/components/SearchWorkspace.jsx` and tests.
- `frontend/src/features/search/components/ImageSearchWorkspace.jsx` and tests.
- `frontend/src/features/filter/components/FilterWorkspace.jsx` and tests.
- `frontend/src/features/search-controls/components/ToolBox.jsx` and tests.
- `frontend/src/features/frames/components/FramesBox.jsx`, `FrameCard.jsx`, `ImageModal.jsx` and tests: rename submit intent to add-to-workspace intent.
- `frontend/src/features/workspace/queryHistory.js`, `ReplayResults.jsx`, `WorkspacePage.jsx` and tests.
- `frontend/src/features/docs/components/ApiDocsModal.jsx` and tests.
- `frontend/src/styles/index.css`, `workspace.css`, `frames-status.css`, and any submission-only selectors found by `rg`.
- `tests/test_query_history.py`, `tests/api/test_database_routes.py`, `tests/api/test_router_inventory.py`, `tests/api/test_search_routes.py`, `tests/api/test_filter_routes.py`, `tests/test_frame_api.py`.

**Delete**

- `frontend/src/features/search/components/TrakeResults.jsx`.
- `frontend/src/features/search/components/TrakePathCard.jsx`.
- Their two test files.
- `frontend/src/features/submission/` and all tests under it.
- `frontend/src/features/submission/submissionArchive.js` and test.
- `src/hcmai/api/contracts/submission.py`.
- `tests/unit/common/test_submission_contract.py`.
- The legacy `/api/v1/submit` handler from `src/hcmai/api/routers/frames.py` and its tests. Keep frame metadata and image-serving routes.

## Chunk 2: Backend implementation

### Task 1: Pin the DRES v2 boundary and backend-only configuration

**Files:** create the `src/hcmai/vbs/` package, its tests, and modify `.env.example`.

- [ ] Write failing config tests for disabled-by-default logging, required HTTPS-capable base URL, bounded timeout, optional explicit evaluation ID, and a backend-only JSON credential map keyed by VBS user ID.
- [ ] Add example configuration without real secrets:

```dotenv
HCMAI_DRES_BASE_URL=https://vbs.videobrowsershowdown.org
HCMAI_DRES_TIMEOUT_SECONDS=3
HCMAI_DRES_EVALUATION_ID=
HCMAI_DRES_LOGGING_ENABLED=false
HCMAI_DRES_USERS_JSON={"member-1":{"username":"member-1","password":"change-me"}}
HCMAI_DRES_MEDIA_ID_PREFIX_TO_STRIP=
```

- [ ] Parse credentials at startup; fail with a redacted configuration error for malformed JSON. Never include passwords in `repr`, exceptions, or logs.
- [ ] Write failing Pydantic tests for `ApiClientAnswer`, `ApiClientAnswerSet`, `ApiClientSubmission`, `QueryEvent`, `QueryResultLog`, login response, evaluation summary, and DRES status/error responses.
- [ ] Implement only fields used by HCMAI, with `extra="forbid"`, matching DRES OpenAPI property casing through aliases.
- [ ] Add a checked-in comment containing the OpenAPI source URL and observed schema version `2.0.5-SNAPSHOT`; do not generate or vendor the entire DRES client.
- [ ] Run:

```bash
uv run pytest tests/vbs/test_config.py tests/vbs/test_models.py -q
```

Expected: all tests pass; malformed config output contains no password.

- [ ] Commit: `feat(vbs): define DRES v2 contracts and secure config`.

### Task 2: Implement and test the DRES HTTP client

**Files:** `src/hcmai/vbs/client.py`, `tests/vbs/test_client.py`.

- [ ] Write MockTransport tests for exact paths, query-string session propagation, JSON body casing, timeout/network failures, and statuses 200, 202, 400, 401, 404, 412.
- [ ] Implement one reusable `httpx.AsyncClient` with methods:

```python
async def login(self, username: str, password: str) -> DresUser: ...
async def list_evaluations(self, session_id: str) -> list[DresEvaluation]: ...
async def get_current_task(self, evaluation_id: str, session_id: str) -> DresTask: ...
async def submit(self, evaluation_id: str, session_id: str,
                 submission: ApiClientSubmission) -> DresSubmissionStatus: ...
async def log_results(self, evaluation_id: str, session_id: str,
                      payload: QueryResultLog) -> DresSuccessStatus: ...
```

- [ ] Translate upstream failures into typed `DresAuthenticationError`, `DresNoActiveTaskError`, `DresRejectedSubmissionError`, and `DresUnavailableError`; retain safe response descriptions but redact session query values.
- [ ] Ensure `aclose()` is called from FastAPI lifespan.
- [ ] Run `uv run pytest tests/vbs/test_client.py -q`.
- [ ] Commit: `feat(vbs): add DRES v2 HTTP adapter`.

### Task 3: Implement per-user session and task resolution

**Files:** `src/hcmai/vbs/service.py`, `tests/vbs/test_service.py`.

- [ ] Write failing tests proving `connect(user_id)` looks up credentials server-side, logs in once, caches the token only in memory, and returns only `{user_id, connected}` without DRES username or session token.
- [ ] Write tests for unknown user ID, bad credential, concurrent connect deduplication, explicit evaluation ID, exactly-one-ACTIVE fallback, zero/multiple active evaluations, and one re-login retry after DRES 401.
- [ ] Implement a lock-per-user cache rather than one global lock.
- [ ] Resolve evaluation in this order: configured `HCMAI_DRES_EVALUATION_ID`; otherwise exactly one `ACTIVE` evaluation from the user's visible list. Never guess when multiple runs are active.
- [ ] Resolve the current DRES task immediately before submission and attach `taskId`. For logs, use the same evaluation and allow DRES to associate the current task.
- [ ] Add `disconnect(user_id)` to evict only the local cached token; DRES logout is optional and must not invalidate another browser using the same configured user.
- [ ] Run `uv run pytest tests/vbs/test_service.py -q`.
- [ ] Commit: `feat(vbs): manage per-member DRES sessions`.

### Task 4: Centralize canonical frame-to-DRES mapping

**Files:** `src/hcmai/vbs/service.py`, `tests/vbs/test_service.py`.

- [ ] Add failing tests using representative V3C, marine, and GynSurg IDs once real metadata samples are available. Until then, require an explicit expected mapping fixture; do not infer from filenames during implementation.
- [ ] Implement `media_item_name(video_id)` at exactly one boundary. Default is identity. An optional configured prefix strip must be exact, anchored, and tested; never call FE `displayVideoId()` semantics.
- [ ] Implement temporal answer mapping:

```python
ApiClientAnswer(
    media_item_name=media_item_name(frame.video_id),
    start=frame.timestamp_ms,
    end=frame.timestamp_ms,
)
```

- [ ] Reject blank media ID and negative/non-integer timestamp. Ignore `frame_id` and `frame_idx` when constructing DRES coordinates; assert `start == end` in contract tests.
- [ ] Add an integration fixture that round-trips one known organizer media ID and millisecond timestamp against the DRES test server before competition cut-over.
- [ ] Commit: `feat(vbs): map canonical frames to DRES temporal answers`.

### Task 5: Replace submission-file persistence with structured answer workspace persistence

**Files:** `src/hcmai/api/history.py`, new `contracts/workspace.py`, new router, associated tests.

- [ ] Characterize and keep current query-history and viewed-frame behavior in tests before editing.
- [ ] Add a SQLite schema migration test starting from the current DB shape. Use `PRAGMA user_version`; the migration must run transactionally.
- [ ] Migration behavior:

```sql
DROP TABLE submission_files;
-- Rebuild query_history without submission_file_names_json and
-- submitted_frame_ids_json.

CREATE TABLE answer_workspace_state (
  singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
  avs_enabled INTEGER NOT NULL CHECK (avs_enabled IN (0, 1)),
  evaluation_id TEXT,
  task_id TEXT,
  revision INTEGER NOT NULL,
  pending_submission_id TEXT,
  pending_submission_state TEXT CHECK (
    pending_submission_state IS NULL
    OR pending_submission_state IN ('FORWARDING', 'UNKNOWN')
  ),
  updated_by_user_id TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL
);

CREATE TABLE answer_candidates (
  candidate_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('FRAME', 'TEXT')),
  source_frame_id TEXT,
  video_id TEXT,
  timestamp_ms INTEGER,
  text TEXT,
  contributed_by_user_id TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL,
  revision INTEGER NOT NULL,
  submitted_at_ms INTEGER,
  submitted_by_user_id TEXT,
  dres_status TEXT,
  evaluation_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  CHECK (
    (kind = 'FRAME' AND video_id IS NOT NULL
      AND timestamp_ms IS NOT NULL AND text IS NULL)
    OR
    (kind = 'TEXT' AND text IS NOT NULL AND source_frame_id IS NULL
      AND video_id IS NULL AND timestamp_ms IS NULL)
  )
);
CREATE UNIQUE INDEX answer_frame_unique
ON answer_candidates(video_id, timestamp_ms)
WHERE kind = 'FRAME';

CREATE TABLE submission_attempts (
  attempt_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('KIS', 'VQA', 'AVS')),
  evaluation_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  submitted_by_user_id TEXT NOT NULL,
  workspace_revision INTEGER NOT NULL,
  snapshot_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK (
    state IN ('FORWARDING', 'UNKNOWN', 'ACCEPTED', 'NOT_ACCEPTED')
  ),
  created_at_ms INTEGER NOT NULL,
  resolved_at_ms INTEGER
);
```

- [ ] Before destructive migration, document that old submission-file data is intentionally retired and advise copying `runtime/workspace.sqlite3` if archival recovery matters.
- [ ] Add store methods to hydrate, add/edit a frame moment, add/edit text, delete candidate, switch AVS mode, mark a submitted set, and clear candidates. Validate `video_id` and `timestamp_ms` directly; do not resolve or replace an inspector timestamp through `SearchService`.
- [ ] Preserve optional `source_frame_id` only as provenance/history linkage when the candidate came from a `FrameCard`. Never require it for live inspector moments and never derive DRES time from it.
- [ ] Split existing `routers/history.py`: keep query-history HTTP routes there; move WebSocket collaboration to `routers/workspace.py`.
- [ ] New transport:

```text
GET /api/v1/answer-workspace
WS  /api/v1/answer-workspace/ws

answer.add_frame
answer.add_text
answer.update_frame
answer.update_text
answer.delete
answer.clear { expected_workspace_revision }
answer.mode.set
answer.task.clear_and_switch {
  expected_workspace_revision,
  expected_old_evaluation_id,
  expected_old_task_id,
  target_evaluation_id,
  target_task_id
}

answer.added
answer.updated
answer.deleted
answer.cleared
answer.mode.changed
answer.submitted
answer.conflict
answer.error
```

- [ ] Preserve optimistic revisions and hydrate-then-replay ordering from the current provider; generalize event names, not file blobs.
- [ ] Bind every candidate and workspace revision to the active evaluation/task. Add tests for empty-workspace automatic rebind and non-empty `TASK_SCOPE_MISMATCH` requiring explicit clear before switching.
- [ ] Implement `answer.task.clear_and_switch` as one transaction: compare workspace revision and old scope, delete candidates, set the exact BE-verified target scope, increment revision, and broadcast one replacement snapshot. Reject a client-supplied target that is not the active DRES task.
- [ ] Run:

```bash
uv run pytest tests/test_query_history.py tests/api/test_answer_workspace_routes.py -q
```

- [ ] Commit: `refactor(workspace): replace submission files with answer candidates`.

### Task 6: Add HCMAI VBS session and submission routes

**Files:** `api/contracts/vbs.py`, `api/routers/vbs.py`, exports, `app.py`, tests.

- [ ] Define FE-facing contracts:

```text
POST /api/v1/vbs/session/connect    { user_id }
DELETE /api/v1/vbs/session/{user_id}
GET /api/v1/vbs/session/{user_id}
POST /api/v1/vbs/submit/kis        { user_id, task_id, expected_workspace_revision, candidate_id, expected_revision }
POST /api/v1/vbs/submit/vqa        { user_id, task_id, expected_workspace_revision, candidate_id, expected_revision }
POST /api/v1/vbs/submit/avs        { user_id, task_id, expected_workspace_revision, candidates: [{ candidate_id, expected_revision }] }
POST /api/v1/vbs/submission-attempts/{attempt_id}/resolve
                                      { user_id, outcome: "accepted" | "not_accepted" }
```

- [ ] KIS route loads the exact revision of one FRAME candidate and submits exactly one answer. VQA route does the same for one TEXT candidate. AVS verifies the ordered candidate ID/revision snapshot exactly matches all current eligible FRAME candidates, then sends exactly one DRES POST whose single answer set contains that frozen order. Do not add a sequential adapter.
- [ ] Immediately before forwarding, verify the current DRES evaluation/task still equals the workspace-bound scope. KIS/VQA additionally require the captured workspace revision and `avs_enabled == false`; AVS requires `avs_enabled == true`. Reject type mismatch, empty AVS list, stale candidate/workspace revision, snapshot membership/order mismatch, task-scope mismatch, disconnected user, and duplicate submit while any workspace submission is in flight.
- [ ] Reserve the validated snapshot transactionally before the network call: insert a durable `submission_attempts` row containing kind, ordered candidate IDs/revisions, exact DRES answer payload, workspace revision, evaluation/task scope, and clicker; then point workspace `pending_submission_id` at it with `FORWARDING` state. While reserved, every candidate mutation, mode change, clear, task switch, or second submission returns `SUBMISSION_IN_FLIGHT`. Construct the DRES request only from this immutable stored snapshot.
- [ ] On definitive DRES success, transactionally mark only the reserved candidate revisions submitted, clear the reservation, increment workspace revision, and broadcast the result. On definitive rejection, clear the reservation without marking candidates. This prevents an edited value from inheriting the status of an older submitted value.
- [ ] On an ambiguous timeout/network outcome, change the reservation to `UNKNOWN`, keep mutations and further submissions blocked, and expose a redacted operator action to resolve the attempt as `accepted` or `not accepted` after checking DRES. Do not auto-release or retry an unknown attempt.
- [ ] During BE startup, transactionally change every persisted `FORWARDING` attempt and matching workspace reservation to `UNKNOWN`; a restarted process cannot know whether DRES accepted the request. Hydrate the locked attempt summary to FE and require explicit resolution.
- [ ] Resolve an unknown attempt using only its durable `snapshot_json`, never by reconstructing from current candidates. `accepted` marks exactly the snapshot revisions submitted; `not_accepted` leaves them unsubmitted. Both outcomes update the attempt, clear the matching workspace reservation, increment revision, and broadcast atomically. Reject attempts that are not the currently reserved `UNKNOWN` attempt.
- [ ] Use the clicker's `user_id` to resolve DRES session. Candidate `contributed_by_user_id` is audit metadata only.
- [ ] On DRES 200 or 202, commit submission status and broadcast `answer.submitted`. On 400/401/404/412, keep candidates and return a safe actionable error.
- [ ] Do not automatically retry submission after ambiguous timeout: retrying can create duplicate/wrong submissions. Show `unknown outcome` and require the explicit resolution flow above.
- [ ] Update application title/version to VBS 2027 and inject/close VBS service in lifespan.
- [ ] Run `uv run pytest tests/api/test_vbs_routes.py tests/api/test_router_inventory.py -q`.
- [ ] Commit: `feat(vbs): forward KIS VQA and AVS submissions to DRES`.

### Task 7: Forward successful search result logs to DRES

**Files:** `api/routers/search.py`, VBS service/config, route tests.

- [ ] Extend text, image, and filter endpoints to read optional `X-VBS-User-ID`. Do not add `user_id` to retrieval-domain Pydantic request models.
- [ ] Capture wall-clock `int(time.time() * 1000)` and construct one event:

```text
text search  -> category=TEXT,   type=SEARCH, value=trimmed query
image search -> category=IMAGE,  type=SEARCH, value=filename + sha256 digest
filter       -> category=FILTER, type=SEARCH, value=stable compact JSON predicates
```

- [ ] After retrieval succeeds, build `QueryResultLog` with 1-based ranks and one `RankedAnswer` per result using exact canonical `video_id` mapping and `timestamp_ms`.
- [ ] Use `sortType="ranked-list"`; set `resultSetAvailability` to a stable value confirmed by the DRES test server (start with the official example's empty string only in test mode).
- [ ] For filter pagination, log the exact returned page on every successful page request and preserve ranks as `(page_id - 1) * frames_per_pages + local_rank`.
- [ ] Send one `/log/result` request, with the query event embedded in its `events`; do not also send `/log/query` for the same search because that duplicates the interaction.
- [ ] Logging policy:
  - disabled config or no connected user: search succeeds with `X-DRES-Log-Status: skipped`;
  - DRES accepted log: `sent`;
  - DRES/network failure: search succeeds with `failed`, logs a redacted warning, and FE shows a non-blocking badge/toast.
- [ ] Add `X-DRES-Log-Status` to the FastAPI CORS `expose_headers` list and update the FE request helper to return this header alongside parsed search data without changing ranking contracts.
- [ ] Await logging with its own short timeout after retrieval result materialization so status is known, but never convert a logging error into a search error. Measure added P50/P95 latency; if unacceptable, replace only this stage with a bounded outbox in a follow-up, not an unbounded background task.
- [ ] Verify image bytes are never included in log JSON or logs.
- [ ] Run search/filter route tests and assert exact DRES payloads and failure isolation.
- [ ] Commit: `feat(vbs): report successful retrieval interactions to DRES`.

### Task 8: Remove legacy backend submission-file and local frame-submit APIs

**Files:** contracts exports, `frames.py`, history/store/database tests and docs.

- [ ] Remove `SubmissionResult`, `SearchService.submission()` only if `rg` confirms no non-legacy runtime consumer, and `/api/v1/submit`.
- [ ] Remove all submission-file classes, methods, routes, events, table allowlisting, query-history fields, and imports.
- [ ] Keep the `submission_files` identifier only inside the one-time migration test/SQL. `rg -n "submission_file|SubmissionFile|/api/v1/submit" src tests` must otherwise be empty.
- [ ] Update Database page backend allowlist so it shows `query_history`, `answer_workspace_state`, and `answer_candidates` if database inspection remains enabled.
- [ ] Run:

```bash
uv run pytest tests/api tests/test_query_history.py tests/test_frame_api.py -q
```

- [ ] Commit: `refactor(api): remove legacy submission file endpoints`.

## Chunk 3: Frontend implementation

### Task 9: Remove all FE TRAKE paths without touching KIS retrieval

**Files:** `api/search.js`, SearchWorkspace, Replay/query history, docs, styles, delete TRAKE components/tests.

- [ ] First update tests to state the new behavior: `E1:` text is ordinary KIS text and Enter always executes text search unless Shift+Enter inserts a newline.
- [ ] Remove `searchTrake`, `parseTrakeEvents`, `buildTrakeSnapshot`, `TrakeResults`, `TrakePathCard`, TRAKE states/branches/submission callbacks, exports, CSS, API docs and test fixtures.
- [ ] Simplify snapshot classification to one KIS `results` shape. Legacy `paths` snapshots render the existing unsupported-history message without importing TRAKE UI.
- [ ] Change placeholder from `Describe the event, or add E1, E2, ... for TRAKE` to VBS-focused KIS/AVS search wording.
- [ ] Run:

```bash
cd frontend
npm test -- --runInBand src/api/search.test.js src/features/search/components/SearchWorkspace.test.jsx src/features/workspace
```

- [ ] Assert `rg -n -i "trake" frontend/src` is empty.
- [ ] Commit: `refactor(frontend): remove TRAKE task flow`.

### Task 10: Add the locked VBS user/session control

**Files:** new VBS session context/control, `AppHeader.jsx`, `App.jsx`, CSS and tests.

- [ ] Replace direct editable input behavior with state machine `editing -> connecting -> connected | error`.
- [ ] On Enter, call `POST /api/v1/vbs/session/connect`. Only after success persist `user_id` to localStorage, lock the input, and display a green `OK` button/status beside it.
- [ ] Keep this control in the application header, not the shared Workspace. It contains only `user_id`; never render a DRES username/password input or login popup. Show a clear configuration error when BE has no credential mapping for that ID.
- [ ] Clicking `OK` evicts local connected state, unlocks and focuses the field. Editing does not change active identity until Enter succeeds. If reconnect fails, remain editable with the old session no longer used by FE.
- [ ] On reload, stored `user_id` is not trusted as connected: call session status/reconnect and show `Connecting…`; lock only after BE confirms.
- [ ] Propagate connected user ID to text/image/filter calls through `X-VBS-User-ID`.
- [ ] Disable DRES Submit actions while disconnected, but do not disable local retrieval or browsing solely because DRES is down. This avoids losing the search UI during an optional integration outage.
- [ ] Never store session token/password in localStorage or React context.
- [ ] Run focused context, header and App tests.
- [ ] Commit: `feat(frontend): lock user ID after DRES session handshake`.

### Task 11: Replace SubmissionProvider with AnswerWorkspaceProvider

**Files:** new answer workspace API/context, `App.jsx`, tests.

- [ ] Port and rename the proven WebSocket lifecycle: initial HTTP hydration, buffered events during hydration, exponential reconnect, pending mutation timeout, optimistic revision conflict, cleanup on unmount.
- [ ] Normalize typed frame/text candidate contracts; reject malformed mixed candidates client-side.
- [ ] Expose minimal context actions: `addFrame`, `addText`, `updateFrame`, `updateText`, `remove`, `clear`, `setAvsEnabled`.
- [ ] Use current connected `user_id` as contributor on every mutation. Disable mutation when user is not connected.
- [ ] Delete `SubmissionProvider`, `SubmissionDialogProvider`, archive logic, and all old modal/file tests.
- [ ] Run provider and transport tests, including reconnect and out-of-order revision events.
- [ ] Commit: `refactor(frontend): synchronize structured answer workspace`.

### Task 12: Build the KIS/VQA/AVS answer workspace UI

**Files:** new components and styles, ToolBox, search/image/filter workspaces, frame components/tests.

- [ ] Replace `SubmissionWorktree` in the sidebar with `AnswerWorkspace` on Query, Image Search, and Filter pages.
- [ ] Header of panel: shared `AVS` switch, connection indicator, answer count, explicit `Clear` with confirmation.
- [ ] Replace the old submission-file picker/editor with one typed `AnswerCandidateDialog`. Opening it from a result or inspector means “review and save to workspace”, never direct DRES submission. FRAME drafts expose editable `video_id` and integer `timestamp_ms`; TEXT drafts expose editable answer text. Enter validates/saves, Escape cancels, and saving a duplicate frame focuses the existing row.
- [ ] For inspector actions, read `videoRef.current.currentTime` synchronously on click and freeze `Math.round(currentTime * 1000)` into the dialog draft; use React playback state only as a guarded fallback. This protects against a final `timeupdate` lag. For `FrameCard` actions, copy `frame.video_id` and exact `frame.timestamp_ms`; do not use `frame_id`, `frame_idx`, FPS conversion, or `liveFrameIdx`.
- [ ] OFF mode:
  - frame cards and the video inspector show a compact add-to-workspace action;
  - panel has a text field + `Add VQA answer`;
  - each FRAME row displays only `<video_id>,<timestamp_ms>,<timestamp_ms>`, with submit, open-viewer, and red delete icons;
  - each TEXT row displays only its answer, with submit and red delete icons;
  - clicking the row body opens `AnswerCandidateDialog`; Enter writes the edited value back through the revision-checked WebSocket mutation;
  - there is no ambiguous global submit.
- [ ] ON mode:
  - text input and text candidates are hidden but retained;
  - frame cards and the video inspector keep the same add-to-workspace action, so members can collect AVS moments without switching AVS OFF;
  - FRAME rows use the same answer-only display and row editor, retain open-viewer and red delete icons, and omit the per-row submit icon;
  - one `Submit all N AVS answers` button opens a confirmation summary and then calls AVS submission once.
- [ ] Keep a confirmation dialog before every actual DRES call: a row submit confirms the single KIS/VQA payload and captures both candidate and workspace revisions, while AVS Submit All confirms the deterministic ordered list and captures candidate plus workspace revisions. Confirmation is read-only at this stage; edits happen by closing it and editing the workspace row. BE rejects the submit if any captured revision, membership, order, mode, or task scope changed after the dialog opened.
- [ ] When BE returns `TASK_SCOPE_MISMATCH`, disable answer mutation/submission and show the old/new task IDs with an explicit `Clear and switch task` action. Confirmation clears old candidates and binds the empty workspace to the new active task; cancellation leaves old answers visible but ineligible.
- [ ] Reuse the existing `resolveFrameAtTimestamp` plus `ImageModal` path for the FRAME open-viewer icon. Pass the candidate timestamp as `initialTimestampMs` so playback opens at that exact moment even when the metadata endpoint resolves a nearby canonical frame.
- [ ] Rename `submissionMode` and `onSubmit` props on frame/modal components to `workspaceAction`/`onAddCandidate` so “add for team review” cannot be confused with “send to DRES”.
- [ ] After 200/202, render DRES status and disable already-submitted rows. On wrong/rejected/unknown outcome, preserve candidates.
- [ ] Add keyboard behavior only where unambiguous; do not bind a single key to direct DRES submission without confirmation.
- [ ] Run focused tests across Query, Image Search, Filter, frame modal and ToolBox.
- [ ] Test both timestamp sources explicitly: inspector at `12.3456s` saves `12346ms`, while a card with `timestamp_ms=12000` saves exactly `12000ms`. Assert neither path reads `frame_idx` and both DRES payloads contain equal `start`/`end`.
- [ ] Test row interaction boundaries: row click edits, Enter updates through `answer.update_frame`/`answer.update_text`, action icons do not open the editor, delete is red, FRAME viewer opens the exact timestamp, TEXT has no viewer icon, non-AVS has row submit icons, and AVS has only Submit All while retaining add controls.
- [ ] Test a collaborator mutation after confirmation opens: FE sends the frozen revision snapshot, BE returns conflict without calling DRES, and the dialog refreshes to the latest deterministic candidate order.
- [ ] Test toggling AVS after a KIS/VQA confirmation opens: stale row submission is rejected because workspace revision/mode changed. Test that all mutations remain blocked during `FORWARDING` and `UNKNOWN`, and that only explicit attempt resolution releases or commits the reservation.
- [ ] Test restart recovery: persisted `FORWARDING` hydrates as locked `UNKNOWN`; explicit resolution uses the stored immutable snapshot and never rebuilds answers from current workspace rows.
- [ ] Commit: `feat(frontend): add collaborative KIS VQA and AVS answer workspace`.

### Task 13: Simplify query history and activity semantics

**Files:** history API/queryHistory/WorkspacePage/ReplayResults and backend history contracts/store.

- [ ] Remove `submission_files` and `submitted_frame_ids` from FE validation and display.
- [ ] Keep `viewed_frame_ids` and KIS snapshots/replay.
- [ ] Do not mark a frame as submitted merely when added to answer workspace. Add a separate visual class such as `candidate` by deriving membership from AnswerWorkspaceContext.
- [ ] Mark DRES-submitted status from workspace events, not query-history patches.
- [ ] Split `frontend/src/api/workspace.js` into `history.js` and `answerWorkspace.js`; update imports and delete the old file when `rg` confirms no consumers.
- [ ] Run all workspace/history tests.
- [ ] Commit: `refactor(history): detach replay history from submissions`.

### Task 14: Update operator-facing docs and status feedback

**Files:** API docs modal, README, env examples, styles/tests.

- [ ] Replace AIC/HCMAI 2026 user-facing title with VBS 2027 without renaming Python packages.
- [ ] Document connect, KIS/VQA/AVS submission, answer workspace and DRES-log status endpoints; remove submission CSV and TRAKE examples from FE docs.
- [ ] Add a compact DRES state indicator: disconnected, connecting, connected, log sent, last log failed. Do not expose session/evaluation tokens.
- [ ] Add a competition runbook section covering: configure users, verify media ID mapping, connect each browser, test KIS/VQA/AVS against test DRES, confirm timestamp, enable logging, and freeze config.
- [ ] Commit: `docs(vbs): document DRES competition workflow`.

## Chunk 4: Verification and release gates

### Task 15: Contract, failure, and regression verification

- [ ] Backend focused suite:

```bash
uv run pytest tests/vbs tests/api/test_vbs_routes.py tests/api/test_answer_workspace_routes.py tests/api/test_search_routes.py tests/api/test_filter_routes.py tests/test_query_history.py -q
```

- [ ] Frontend suite:

```bash
cd frontend
npm test -- --runInBand
npm run build
```

- [ ] Broad backend suite:

```bash
uv run pytest -q
```

- [ ] Static dead-code checks:

```bash
rg -n -i "trake" frontend/src
rg -n "submission_file|SubmissionFile|SubmissionWorktree|SubmissionDialog|/api/v1/submit" frontend/src src tests
```

Expected: first command empty; second contains only intentional migration compatibility references, if any.

- [ ] Verify unrelated text retrieval, image retrieval, filter, frame inspector, video playback, database browser, query history, health, and Vim controls behave as before.
- [ ] Commit: `test(vbs): cover DRES and answer workspace release gates`.

### Task 16: DRES staging rehearsal

- [ ] Use the organizer-provided or local DRES v2 test deployment; do not test first against the live scoring run.
- [ ] For each team member, connect by VBS user ID and verify the DRES response identifies the expected account/team.
- [ ] Submit one KIS candidate captured from live playback and one from a frame card. Inspect captured JSON: exact organizer `mediaItemName`, exact selected `timestamp_ms` as equal `start` and `end`, one answer only per request.
- [ ] Submit one VQA string and verify the answer contains `text` only.
- [ ] Add three AVS candidates from two browsers; verify WebSocket convergence, duplicate suppression, and one DRES request with three answers.
- [ ] Open AVS confirmation, edit one candidate from another browser, and verify the stale snapshot is rejected before DRES receives a request. Reopen and verify deterministic ordering on both clients.
- [ ] Simulate a DRES task transition with a non-empty workspace. Verify submission/mutation is blocked until `Clear and switch task` is explicitly confirmed; then verify new candidates bind to the new task.
- [ ] Run text, image, filter page 1 and filter page 2; verify four result-log requests, correct event categories, global ranks for filter pagination, and no raw image bytes.
- [ ] Simulate 401, timeout, 412, WebSocket reconnect and concurrent candidate edits. Confirm retrieval never disappears because optional logging failed and ambiguous submissions are not auto-retried.
- [ ] Record P50/P95 DRES login, submit and log latency. Enable `HCMAI_DRES_LOGGING_ENABLED=true` for competition only after the rehearsal passes.
- [ ] Save sanitized request/response fixtures (no credentials/session IDs) under `tests/fixtures/dres/` and rerun all contract tests.
- [ ] Commit: `test(vbs): validate DRES staging integration`.

## Open decisions requiring organizer/team confirmation

1. **Exact `mediaItemName`:** must be validated from VBS 2027 metadata. Current AIC `displayVideoId()` drops dotted prefixes and is unsafe as an unverified DRES mapping.
2. **Logging vocabulary:** DRES types are free strings, but `resultSetAvailability` semantics are not constrained by OpenAPI. Confirm preferred VBS 2027 values during staging.

## Suggested VBS-specific improvements (kept within this design)

- Prefer a shared three-state task label in the UI (`KIS`, `VQA`, `AVS`) even if AVS remains a binary switch internally; it eliminates ambiguity during a live task. If the binary switch must remain, use row-specific KIS/VQA submit buttons as planned.
- Keep a visible confirmation modal for every DRES submission, especially AVS Submit All because false AVS submissions are strongly penalized and temporally close examples contribute less than diverse instances.
- Display duplicate-video and near-duplicate-time warnings in AVS workspace, without automatically deleting choices. This helps select diverse instances while preserving human control.
- Add an explicit dry-run switch for local/staging DRES, disabled in production, to inspect exact JSON before competition day.

## Definition of done

- FE contains no TRAKE code, copy, tests or imports.
- Legacy submission-file FE/BE APIs, components, persistence and query-history coupling are gone.
- Every browser can connect a VBS user ID, becomes locked only after successful DRES login, and can unlock/reconnect through the adjacent OK control.
- Team members see the same structured candidates and AVS mode in near real time.
- Workspace answers are bound to one active DRES evaluation/task; a task transition cannot silently reuse or submit stale candidates.
- Inspector additions preserve the exact realtime player timestamp; frame-card additions preserve the exact result `timestamp_ms`. Workspace edits never convert either value through `frame_id` or `frame_idx`.
- KIS submits exactly one point with `start == end`; VQA submits exactly one text; AVS submits all eligible point timestamps in one request using the clicker's DRES session.
- Workspace rows show only their actual answer plus compact actions: edit by row click, red delete, per-row KIS/VQA submit, FRAME open viewer, and AVS-only global Submit All.
- Text, image and filter results generate DRES result logs immediately after successful retrieval without making retrieval fail when logging fails.
- DRES payloads use exact organizer media IDs and media timeline milliseconds, never AIC `frame_idx`.
- Existing non-TRAKE retrieval, inspection, playback, history, database and controls pass regression tests.
