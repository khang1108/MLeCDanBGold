# VBS DRES Contract Corrections Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the VBS integration so current-task resolution and successful submission responses match the official DRES Client API v2 contract, while preserving answer-workspace safety and restoring a green frontend test suite.

**Architecture:** Treat the DRES current-task response as task-template information, not an instantiated task ID. Build a deterministic backend-only `task_scope_key` from the evaluation and current-task fields for workspace isolation, submit the official `taskName` instead of a fabricated `taskId`, and parse submission success with its dedicated verdict-bearing response model. Keep retrieval, result logging, timestamps, KIS/VQA/AVS behavior, and WebSocket collaboration otherwise unchanged.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, httpx MockTransport, SQLite, pytest, React, Jest/Testing Library.

---

## Chunk 1: DRES wire contract and active-task scope

### Source-of-truth decisions

- The official DRES Client API OpenAPI is the wire-contract source of truth:
  `https://github.com/dres-dev/DRES/blob/master/doc/oas-client.json`.
- `GET /api/v2/client/evaluation/currentTask/{evaluationId}` returns
  `ApiClientTaskTemplateInfo` with `name`, `taskGroup`, `taskType`, and optional
  `duration`. It does not return `taskId` or `id`.
- `POST /api/v2/submit/{evaluationId}` accepts an `ApiClientAnswerSet` whose
  `taskId` and `taskName` are optional. This implementation sends `taskName`
  from the freshly resolved task so DRES can reject a stale named task instead
  of silently targeting a different current task.
- A successful submit (HTTP 200 or 202) returns
  `SuccessfulSubmissionsStatus`: `status`, `submission`, and `description`.
  `submission` is one of `CORRECT`, `WRONG`, `INDETERMINATE`, or `UNDECIDABLE`.
- `accepted=True` in HCMAI means DRES accepted and processed the request. It
  does not mean the answer verdict is `CORRECT`; preserve the separate verdict.
- The internal workspace key is not sent to DRES. It is:

```text
task_scope_key = "dres-task-v1:" + sha256(canonical JSON of
  evaluation_id, name, taskGroup, taskType, duration).hexdigest()
```

- Canonical JSON must use sorted keys, UTF-8, and compact separators so all BE
  processes derive the same key. Normalize no task values other than validating
  required strings as non-blank.
- DRES cannot distinguish two consecutive task instances having identical
  current-task fields through this client endpoint. Record this as a staging
  limitation; do not call an undocumented/admin endpoint to manufacture a
  stronger identity.

### File map

**Modify**

- `src/hcmai/vbs/models.py` — exact current-task and successful-submit schemas;
  internal active-scope value object.
- `src/hcmai/vbs/client.py` — parse submit success independently from ordinary
  success/error statuses.
- `src/hcmai/vbs/service.py` — derive and return the internal task scope.
- `src/hcmai/api/contracts/workspace.py` — rename misleading task-ID fields to
  task-scope fields and expose the current task name.
- `src/hcmai/api/history.py` — persist/migrate the internal scope key and task
  name; validate frozen payloads using `taskName`.
- `src/hcmai/api/routers/workspace.py` — use named scope attributes rather than
  tuple-position task IDs.
- `src/hcmai/api/routers/vbs.py` — build `taskName` answer sets and preserve the
  DRES verdict.
- `frontend/src/api/answerWorkspace.js` — consume renamed workspace scope
  fields.
- `frontend/src/features/answer-workspace/contexts/AnswerWorkspaceContext.jsx`
  — compare and switch internal scope keys.
- `frontend/src/features/answer-workspace/components/AnswerWorkspace.jsx` —
  display task name without presenting a hash as a DRES task ID.
- `frontend/src/features/docs/components/ApiDocsModal.jsx` — update workspace
  request/response examples to the renamed internal scope contract.
- `frontend/src/App.test.jsx`,
  `frontend/src/features/docs/components/ApiDocsModal.test.jsx`, and
  `tests/api/test_database_routes.py` — update known integration, docs, and
  database-browser consumers of workspace column/response names.
- All directly corresponding backend and frontend tests listed below.

**Do not modify**

- Search ranking/retrieval logic.
- Result-log payload shape in `src/hcmai/api/routers/search.py`.
- Millisecond conversion or `start == end` temporal-answer behavior.
- TRAKE backend code.
- KIS/VQA/AVS candidate selection semantics.

### Task 1: Replace the inaccurate DRES response models

**Files:**

- Modify: `src/hcmai/vbs/models.py`
- Test: `tests/vbs/test_models.py`

- [ ] **Step 1: Replace fake current-task fixtures with the official response.**

```python
task = DresTaskTemplateInfo.model_validate({
    "name": "KIS task",
    "taskGroup": "KIS",
    "taskType": "KIS",
    "duration": 300,
})
assert task.name == "KIS task"
```

- [ ] **Step 2: Add a regression test proving `taskId` is not accepted as part
  of the current-task response.**

```python
with pytest.raises(ValidationError):
    DresTaskTemplateInfo.model_validate({
        "taskId": "invented-id",
        "name": "KIS",
        "taskGroup": "KIS",
        "taskType": "KIS",
    })
```

- [ ] **Step 3: Add tests for the exact successful-submit response.**

```python
status = DresSubmissionStatus.model_validate({
    "status": True,
    "submission": "CORRECT",
    "description": "accepted",
})
assert status.submission == "CORRECT"

with pytest.raises(ValidationError):
    DresSubmissionStatus.model_validate({
        "status": True,
        "description": "missing verdict",
    })
```

- [ ] **Step 4: Run the focused model tests and confirm the new tests fail for
  the old implementation.**

Run:

```bash
python -m pytest tests/vbs/test_models.py -q
```

Expected before implementation: failures caused by missing
`DresTaskTemplateInfo`/`DresSubmissionStatus` and acceptance of the fake ID.

- [ ] **Step 5: Implement the minimal exact models.**

```python
DresVerdict = Literal["CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"]

class DresTaskTemplateInfo(_DresModel):
    """Official current-task response; it contains no instantiated task ID."""

    name: NonBlank
    task_group: NonBlank = Field(alias="taskGroup")
    task_type: NonBlank = Field(alias="taskType")
    duration: int | None = Field(default=None, ge=0)


class DresSubmissionStatus(_DresModel):
    """Verdict-bearing status returned by a successful DRES submission."""

    status: bool
    submission: DresVerdict
    description: str
```

Keep the ordinary `DresStatus` model for result-log success/error bodies. Remove
`DresTask.task_id`, `DresTask.id`, and `resolved_id()` rather than retaining a
compatibility path that contradicts the official schema. Since HCMAI never
sends `taskId`, also remove `task_id` from its deliberately restricted outbound
`ApiClientAnswerSet`; keep only optional `task_name` plus required `answers`.

- [ ] **Step 6: Run the model tests.**

Expected: all `tests/vbs/test_models.py` tests pass.

- [ ] **Step 7: Commit.**

```bash
git add src/hcmai/vbs/models.py tests/vbs/test_models.py
git commit -m "fix(vbs): match DRES current-task and submit status schemas"
```

### Task 2: Parse verdict-bearing submission responses

**Files:**

- Modify: `src/hcmai/vbs/client.py`
- Test: `tests/vbs/test_client.py`

- [ ] **Step 1: Change the MockTransport current-task fixture to omit
  `taskId`.**

- [ ] **Step 2: Add parametrized submit tests for HTTP 200 and 202 using exact
  official bodies.**

```python
@pytest.mark.parametrize("status_code", [200, 202])
def test_submit_parses_official_success_status(status_code):
    body = {
        "status": True,
        "submission": "INDETERMINATE" if status_code == 202 else "CORRECT",
        "description": "accepted",
    }
    # Mock /api/v2/submit/eval-1, call client.submit(), assert all fields.
```

- [ ] **Step 3: Add malformed-response tests for a missing `submission`, an
  unknown verdict, and an empty 202 body.** These responses must raise
  `DresUnavailableError`; the request may already have reached DRES, so the
  router must later retain its existing `UNKNOWN` delivery state.

- [ ] **Step 4: Run the client tests and verify failure against the old generic
  parser.**

Run: `python -m pytest tests/vbs/test_client.py -q`

- [ ] **Step 5: Split response parsing by endpoint.**

```python
def _submission_status_body(response: httpx.Response) -> DresSubmissionStatus:
    try:
        result = DresSubmissionStatus.model_validate(response.json())
    except (ValueError, ValidationError):
        raise DresUnavailableError(
            "DRES submission returned an invalid status"
        ) from None
    if not result.status:
        raise DresRejectedSubmissionError(
            f"DRES rejected submission: {_redact(result.description)}"
        )
    return result
```

`submit()` uses `_submission_status_body`; `log_results()` keeps the ordinary
success-status parser. Do not treat an empty 202 body as confirmed success.

- [ ] **Step 6: Run `tests/vbs/test_client.py` and `tests/vbs/test_models.py`.**

Expected: all pass.

- [ ] **Step 7: Commit.**

```bash
git add src/hcmai/vbs/client.py tests/vbs/test_client.py
git commit -m "fix(vbs): parse DRES submission verdict responses"
```

### Task 3: Introduce a deterministic internal task scope

**Files:**

- Modify: `src/hcmai/vbs/models.py`
- Modify: `src/hcmai/vbs/service.py`
- Test: `tests/vbs/test_service.py`

- [ ] **Step 1: Add service tests using only official current-task fixtures.**
  Cover deterministic output, a changed task name/type producing a different
  key, a changed evaluation producing a different key, and no active task
  remaining a typed `DresNoActiveTaskError` from the client.

- [ ] **Step 2: Define an internal value object.**

```python
class DresTaskScope(BaseModel):
    """Backend workspace scope derived from official current-task fields."""

    evaluation_id: NonBlank
    task_scope_key: NonBlank
    task_name: NonBlank
    task_group: NonBlank
    task_type: NonBlank
    duration: int | None = None
```

This is an HCMAI model, not a DRES wire model; do not inherit `_DresModel` or
serialize it into a DRES request.

- [ ] **Step 3: Implement one private `_task_scope_key()` helper using canonical
  JSON and SHA-256.** Include a version prefix so its input can evolve without
  silently colliding with stored scopes.

- [ ] **Step 4: Change `resolve_task()` to return
  `DresTaskTemplateInfo` without requiring an ID, and `resolve_scope()` to
  return `DresTaskScope` rather than a positional tuple.**

- [ ] **Step 5: Update service fake return types.** Submission fake results use
  `DresSubmissionStatus`, not `DresStatus`.

- [ ] **Step 6: Run service tests.**

Run:

```bash
python -m pytest tests/vbs/test_service.py -q
```

Expected: all pass; no test fixture contains a positive-path `taskId`.

- [ ] **Step 7: Commit.**

```bash
git add src/hcmai/vbs/models.py src/hcmai/vbs/service.py tests/vbs/test_service.py
git commit -m "fix(vbs): derive workspace scope from official task metadata"
```

## Chunk 2: Workspace, submission payload, and API migration

### Task 4: Rename internal task identity throughout the workspace

**Files:**

- Modify: `src/hcmai/api/contracts/workspace.py`
- Modify: `src/hcmai/api/history.py`
- Modify: `src/hcmai/api/routers/workspace.py`
- Modify: `frontend/src/api/answerWorkspace.js`
- Modify: `frontend/src/features/answer-workspace/contexts/AnswerWorkspaceContext.jsx`
- Modify: `frontend/src/features/answer-workspace/components/AnswerWorkspace.jsx`
- Modify: `frontend/src/features/docs/components/ApiDocsModal.jsx`
- Test: `tests/test_answer_workspace_store.py`
- Test: `tests/test_submission_reservations.py`
- Test: `tests/api/test_answer_workspace_routes.py`
- Test: `tests/api/test_answer_workspace_broadcast.py`
- Test: `tests/api/test_database_routes.py`
- Test: `frontend/src/api/answerWorkspace.test.js`
- Test: `frontend/src/features/answer-workspace/contexts/AnswerWorkspaceContext.test.jsx`
- Test: `frontend/src/features/answer-workspace/components/AnswerWorkspace.test.jsx`
- Test: `frontend/src/features/docs/components/ApiDocsModal.test.jsx`
- Test: `frontend/src/App.test.jsx`

- [ ] **Step 1: Rename public/internal contract fields before implementation and
  run tests to expose every dependent call site.**

```text
task_id                    -> task_scope_key
active_task_id             -> active_task_scope_key
expected_old_task_id       -> expected_old_task_scope_key
target_task_id             -> target_task_scope_key
```

Add `task_name` and `active_task_name` to workspace snapshots so the UI displays
human-readable task information rather than the SHA-256 key.

- [ ] **Step 2: Bump the SQLite schema from version 1 to version 2 and add
  transactional migration tests for both an idle and a populated database.**
  Rebuild these tables with clear columns:

```text
answer_workspace_state.task_scope_key, task_name
answer_candidates.task_scope_key
submission_attempts.task_scope_key, task_name
```

Copy existing `task_id` values into `task_scope_key` and use a visible legacy
task name such as `legacy-unverified`. Preserve candidates and attempt audit
records. The first new official scope will mismatch and require the existing
explicit “clear and switch task” flow; do not silently bind legacy answers.

The populated-v1 fixture must include candidates, revisions, an immutable
attempt snapshot, and a linked pending `FORWARDING` or `UNKNOWN` reservation.
After migration assert:

- candidate values, ordering, revisions, and submitted state are unchanged;
- `snapshot_json` bytes are unchanged;
- pending attempt ID/state still links workspace state to the same attempt;
- startup recovery still converts interrupted `FORWARDING` to `UNKNOWN`;
- resolving the preserved attempt still performs the same accepted/not-accepted
  transition and unlocks the workspace;
- `PRAGMA index_list(answer_candidates)` contains the unique frame index, and a
  duplicate `(video_id, timestamp_ms)` frame insert still fails.

When rebuilding tables, explicitly drop or uniquely rename the old
`answer_frame_unique` index before creating the new index. Do not rely on
`CREATE INDEX IF NOT EXISTS`: an index name attached to a renamed legacy table
can otherwise prevent creation on the replacement table.

- [ ] **Step 3: Update store method parameters, exception attributes,
  snapshots, reservation validation, and WebSocket commands to the new names.**
  Keep revision checks, durable `FORWARDING`/`UNKNOWN`, candidate ordering, and
  shared-socket behavior unchanged.

- [ ] **Step 4: Update routers to use named `DresTaskScope` attributes.** Avoid
  destructuring positional tuples:

```python
scope = await service.resolve_scope(user_id)
workspace = store.get_answer_workspace(
    scope.evaluation_id,
    scope.task_scope_key,
    scope.task_name,
)
```

- [ ] **Step 5: Update FE parsing and task-switch commands.** The mismatch panel
  displays `task_name`/`active_task_name`; hashes remain state values used only
  for optimistic scope comparison. Update `ApiDocsModal` examples and their
  tests, `App.test.jsx`, and database-browser expectations so no known consumer
  documents or asserts the retired `task_id` workspace contract.

- [ ] **Step 6: Run focused store/API/FE tests.**

```bash
python -m pytest tests/test_answer_workspace_store.py tests/test_submission_reservations.py tests/api/test_answer_workspace_routes.py tests/api/test_answer_workspace_broadcast.py tests/api/test_database_routes.py -q
cd frontend
npm test -- --watchAll=false --runInBand src/api/answerWorkspace.test.js src/features/answer-workspace/contexts/AnswerWorkspaceContext.test.jsx src/features/answer-workspace/components/AnswerWorkspace.test.jsx src/features/docs/components/ApiDocsModal.test.jsx src/App.test.jsx
```

Expected: all pass, including migration from database version 1.

- [ ] **Step 7: Commit.**

```bash
git add src/hcmai/api/contracts/workspace.py src/hcmai/api/history.py src/hcmai/api/routers/workspace.py frontend/src/api/answerWorkspace.js frontend/src/features/answer-workspace frontend/src/features/docs/components/ApiDocsModal.jsx frontend/src/features/docs/components/ApiDocsModal.test.jsx frontend/src/App.test.jsx tests/test_answer_workspace_store.py tests/test_submission_reservations.py tests/api/test_answer_workspace_routes.py tests/api/test_answer_workspace_broadcast.py tests/api/test_database_routes.py
git commit -m "refactor(vbs): distinguish workspace scope from DRES task IDs"
```

### Task 5: Submit `taskName` and preserve the DRES verdict

**Files:**

- Modify: `src/hcmai/api/contracts/vbs.py`
- Modify: `src/hcmai/api/routers/vbs.py`
- Test: `tests/api/test_vbs_submission_routes.py`
- Test: `tests/test_submission_reservations.py`

- [ ] **Step 1: Replace all positive-path submit fixtures with official current
  task and response bodies.** Expected KIS payload:

```json
{
  "answerSets": [{
    "taskName": "KIS task",
    "answers": [{
      "mediaItemName": "video.with.dots",
      "start": 12346,
      "end": 12346
    }]
  }]
}
```

VQA still contains one text answer. AVS still contains all selected temporal
answers in one `answerSet` and one HTTP request.

- [ ] **Step 2: Add route assertions for each verdict.** HTTP 200/202 with a
  valid `DresSubmissionStatus` completes the reservation as `ACCEPTED`, marks
  selected candidates submitted, and returns the upstream verdict separately.

```python
class VbsSubmissionResponse(BaseModel):
    # Existing fields remain.
    verdict: Literal[
        "CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"
    ] | None = None
```

An accepted `WRONG` answer is still delivered and must not become retryable;
`accepted=True`, `state="ACCEPTED"`, `verdict="WRONG"`.

- [ ] **Step 3: Change `_build_submission_payload()` to accept the task name,
  not the internal scope key.**

```python
ApiClientSubmission(answer_sets=[
    ApiClientAnswerSet(task_name=scope.task_name, answers=answers),
])
```

- [ ] **Step 4: Update `_validate_submission_payload()` in the store.** Require
  exactly one answer set with `taskName == reserved task_name` and non-empty
  `answers`. Explicitly reject a payload containing `taskId` so an internal
  scope key can never escape onto the DRES wire.

- [ ] **Step 5: Keep all three pre-send scope checks, but compare
  `(evaluation_id, task_scope_key)` and reserve `task_name` with the immutable
  attempt.** If resolution changes before send, release as `NOT_ACCEPTED` as it
  does today.

- [ ] **Step 6: Run submission tests.**

```bash
python -m pytest tests/api/test_vbs_submission_routes.py tests/test_submission_reservations.py tests/vbs -q
```

Expected: all pass; KIS/VQA use one answer, AVS uses one request containing all
answers, and successful responses never become `UNKNOWN` merely because they
contain `submission`.

- [ ] **Step 7: Run contract residue checks.**

```bash
rg -n 'resolved_id|live_task_id|active_task_id|expected_old_task_id|target_task_id|\btask_id\b' src/hcmai/vbs src/hcmai/api frontend/src tests
rg -n '"taskId"' src/hcmai/vbs src/hcmai/api/routers/vbs.py src/hcmai/api/history.py
```

Expected: the first command has no active workspace/submission runtime matches;
unrelated DRES/admin or legacy migration records must be inspected explicitly.
The second has no production model or wire-construction match because the
restricted outbound answer-set model no longer exposes `taskId`; it may remain
only in an intentional negative regression test or migration comment.

- [ ] **Step 8: Commit.**

```bash
git add src/hcmai/api/contracts/vbs.py src/hcmai/api/routers/vbs.py src/hcmai/api/history.py tests/api/test_vbs_submission_routes.py tests/test_submission_reservations.py
git commit -m "fix(vbs): submit named DRES tasks and expose verdicts"
```

## Chunk 3: Test portability and release gates

### Task 6: Make URL tests independent of developer `.env`

**Files:**

- Modify: `frontend/src/api/keyframes.test.js`
- Modify: `frontend/src/features/frames/videoSource.test.js`
- Modify: `frontend/src/features/alignment/components/AlignmentAccordion.test.jsx`
- Optionally modify: `frontend/src/api/client.test.js` only if a separate test
  is needed for the localhost production default.

- [ ] **Step 1: Reproduce the three failures with the checked developer
  `.env`.**

Run:

```bash
cd frontend
npm test -- --watchAll=false --runInBand src/api/keyframes.test.js src/features/frames/videoSource.test.js src/features/alignment/components/AlignmentAccordion.test.jsx
```

Expected before changes: host assertions differ because `.env` configures
`https://backend.iamphuckhang.dev`.

- [ ] **Step 2: Test the invariant owned by each module, not a machine-specific
  deployment host.**

- `keyframes.test.js`: build the expected value from imported `API_BASE_URL` and
  assert encoded frame ID/path.
- `AlignmentAccordion.test.jsx`: compare the rendered image URL with
  `keyframeUrl("f1")`.
- `videoSource.test.js`: parse the returned URL and assert its pathname is
  `/api/v1/videos/L21_V001/stream`, proving the canonical leaf-ID mapping
  independently of the configured host.

Do not edit or commit `frontend/.env`. If default-host behavior needs coverage,
test `resolveDefaultBaseUrl` in an isolated module with the relevant environment
explicitly controlled.

- [ ] **Step 3: Run the three tests under both configured and empty API env
  values.** On Windows/CI, use the project's normal environment mechanism; do
  not add a dependency solely to set one variable.

- [ ] **Step 4: Run the full frontend suite and build.**

```bash
cd frontend
npm test -- --watchAll=false --runInBand
npm run build
```

Expected: the complete discovered frontend suite passes; report its resulting
test count rather than relying on the previous count. The production build
compiles successfully.

- [ ] **Step 5: Commit.**

```bash
git add frontend/src/api/keyframes.test.js frontend/src/features/frames/videoSource.test.js frontend/src/features/alignment/components/AlignmentAccordion.test.jsx
git commit -m "test(frontend): make asset URL assertions environment independent"
```

### Task 7: Run regression and DRES staging gates

**Files:**

- Modify if behavior changed: `README.md`
- Modify if research/contract assumptions changed: `KNOWLEDGE.md`
- Test: all VBS, API, workspace, frontend suites

- [ ] **Step 1: Run focused backend regression.**

```bash
python -m pytest tests/vbs tests/api/test_vbs_submission_routes.py tests/api/test_answer_workspace_routes.py tests/api/test_answer_workspace_broadcast.py tests/api/test_dres_result_logging.py tests/test_answer_workspace_store.py tests/test_submission_reservations.py -q
```

Expected: all pass.

- [ ] **Step 2: Run full backend tests in the complete project environment.**

```bash
python -m pytest -q
```

Expected: all pass. Install/use the repository's encoder and reranker test
dependencies first; do not mark the gate green while collection errors remain.

- [ ] **Step 3: Reconfirm unchanged result logging.** Capture one TEXT, IMAGE,
  and FILTER request and assert `/api/v2/log/result/{evaluationId}`, epoch-ms
  event time, global 1-based rank, `sortType="list"`,
  `resultSetAvailability=""`, exact `mediaItemName`, and
  `start == end == timestamp_ms`.

- [ ] **Step 4: Run a DRES staging smoke test with a dedicated test user.**

1. Connect by VBS `user_id`; confirm FE shows `OK` without receiving the token.
2. Hydrate an empty workspace from an official current-task response with no
   `taskId`.
3. Submit one KIS temporal answer and verify the request contains `taskName`.
4. Submit one VQA text answer.
5. Submit at least two AVS temporal answers in one request.
6. Verify 200 and, if available, 202 verdict bodies are parsed and candidates
   leave `FORWARDING` without becoming false `UNKNOWN`.
7. Change to a differently named task and verify stale workspace mutation is
   blocked until explicit clear-and-switch.
8. Verify search failures in optional result logging remain non-blocking.

- [ ] **Step 5: Validate campaign-specific media identity.** Use several
  organizer-provided media IDs. Confirm the exact `mediaItemName`; do not remove
  prefixes with display helpers. This remains a release blocker until VBS 2027
  publishes or exposes its collection.

- [ ] **Step 6: Exercise the repeated-identical-task limitation.** If staging
  can run two consecutive tasks with identical `name`, `taskGroup`, `taskType`,
  and `duration`, determine whether VBS guarantees an observable gap or unique
  task name. If neither is guaranteed, escalate to the organizer for a supported
  client-visible instance identity; do not switch to an undocumented DRES API.

- [ ] **Step 7: Update documentation with verified facts only.** Record the
  tested DRES version, observed task response, accepted payloads, response
  verdicts, media ID examples, and any remaining VBS 2027 unknowns. Label
  staging observations `VERIFIED` and untested VBS 2027 assumptions `PROPOSED`.

- [ ] **Step 8: Final diff and secret checks.**

```bash
git diff --check
rg -n 'sessionId|password|HCMAI_DRES_USERS_JSON' frontend/src
rg -ni 'trake' frontend/src
rg -ni 'submission[_ -]?file' frontend/src src/hcmai tests
```

Expected: clean diff; no FE secret/session fields; no TRAKE FE runtime; old
submission-file matches only in intentional migration tests/code.

- [ ] **Step 9: Commit final verified documentation.**

```bash
git add README.md KNOWLEDGE.md
git commit -m "docs(vbs): record verified DRES contract and staging gates"
```

## Completion criteria

- Official current-task JSON without `taskId` hydrates and mutates the answer
  workspace successfully.
- No internal workspace key is serialized as DRES `taskId`.
- KIS and VQA submit one answer; AVS submits all eligible frames in one request.
- Every temporal answer preserves exact integer milliseconds with `start == end`.
- HTTP 200/202 official submission responses retain their verdict and complete
  the durable attempt as delivered.
- Malformed/ambiguous responses remain `UNKNOWN`, preventing unsafe automatic
  retries.
- Result logging remains schema-correct and non-blocking.
- Focused backend tests, full backend tests, all frontend tests, and frontend
  production build are green.
- DRES staging verifies `taskName`, AVS batching, verdict parsing, and real
  `mediaItemName` values before competition use.
