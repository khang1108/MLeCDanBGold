# EventTrail Design Specification

**Date:** 2026-09-16  
**Status:** Approved for implementation planning  
**Scope:** Event-level temporal evidence inspection and correction for KIS results  
**Depends on:** Unified Multimodal Progressive KIS stabilization and acceptance gate

## 1. Purpose

EventTrail is an interactive evidence-correction layer for multi-event temporal video retrieval. It does not replace the ranked-result grid, the KIS Query Composer, or the underlying retrieval stack. It is an escalation path for difficult cases where a retrieved temporal hypothesis is partially wrong.

The core interaction model is:

```text
KIS query
  -> ranked temporal result
  -> EventTrail opens exact retrieved path
  -> user inspects event-level evidence
  -> user Approves, Declines, or Uses evidence
  -> constrained temporal decoder re-aligns the remaining path
  -> UI exposes direct and propagated changes
```

The central design principle is:

> When retrieval intent is represented as ordered events, corrective feedback should operate on event-level evidence rather than requiring the user to reformulate the entire query.

EventTrail is an interaction and systems contribution, not a new embedding model, a new multimodal fusion method, or a generic relevance-feedback framework.

---

## 2. Implementation Prerequisite Gate

EventTrail implementation must not begin until the current S0/S1 foundation passes its acceptance gate. In particular, the implementation plan must verify and, if still present, correct the following before EventTrail source changes are introduced:

1. progressive append of image-only events constructs a valid multimodal event atomically;
2. production KIS/temporal workflows contain no `unittest.mock.Mock`-dependent behavior;
3. configured remote embedding failures do not silently fall back to a different local encoder;
4. incoming image references are canonicalized/validated against the image asset store before retrieval;
5. scoped creation of new textual events preserves validated semantic bindings;
6. image-only retrieval source validation accepts image evidence even when text dense/BM25 toggles are disabled;
7. the real production retrieval configuration gives `visual_image` non-zero weight;
8. backend tests, frontend tests/build, and dead-code checks pass for the unified KIS flow.

These are prerequisites, not EventTrail features. EventTrail must not add compatibility shims around unfinished KIS contracts.

---

## 3. Scope and Non-Goals

### 3.1 In scope

EventTrail v1 provides:

- opening an exact ranked temporal result as an interactive event path;
- reuse of the exact fused per-video evidence that produced the result;
- one selected-video interaction session at a time;
- event-level `Explore`, `Use`, `Approve`, and `Decline` interactions;
- optional session-level temporal window constraints;
- transactional constrained temporal re-decoding;
- undo with monotonic trail revisions;
- diff-aware visualization of direct and indirect path changes;
- preserving ranked-grid state while entering/leaving EventTrail;
- per-result interaction annotations such as `explored` and `exhausted`;
- submission from the EventTrail workspace using an explicitly selected frame;
- interaction and latency logging sufficient for later evaluation.

### 3.2 Explicit non-goals

EventTrail v1 does **not** add:

- global relevance-feedback reranking across videos;
- path carousels or manual `Path #1/#2/#3` switching;
- learned feedback models;
- new embedding fusion research;
- confidence-model training or user-facing numeric confidence scores;
- automatic query-semantic mutation from Trail actions;
- event reordering, splitting, or merging;
- Redis or persistent EventTrail state;
- distributed session state;
- automatic transfer of constraints from one video to another;
- automatic switching to the next ranked video after a video is exhausted.

Semantic corrections remain the responsibility of the Query Composer and create a new KIS revision.

---

## 4. Architectural Boundary

The system separates three different state domains.

### 4.1 KIS semantic state

KIS owns what the user means:

```text
KISIntent revision N
E1 < E2 < ... < En
```

A semantic update through `E#:` or `/llm-rewrite` creates a new KIS revision and a new retrieval world.

### 4.2 Retrieval evidence state

Each successful KIS search may produce one immutable `EvidenceSnapshot` containing the evidence needed to reopen the returned temporal results without rescoring the corpus.

### 4.3 EventTrail interaction state

An `EventTrailSession` is bound permanently to:

- one `EvidenceSnapshot`;
- one ranked `result_id`;
- one `video_id`;
- one KIS semantic revision;
- one frozen per-video event-score matrix.

Trail actions change evidence constraints and Trail revision only. They never increment the KIS semantic revision.

The distinction is:

```text
Query Composer = what happened?
EventTrail     = where did each event happen in this video?
```

---

## 5. ET-0: Evidence Handoff

### 5.1 Problem

The current `TemporalExploration.open()` reacquires selected-video scores through `score_plan()`/legacy scoring. This repeats expensive retrieval work after the KIS result was already produced.

### 5.2 Decision

A successful KIS temporal search creates a short-lived backend `EvidenceSnapshot`. The KIS response exposes only an opaque `evidence_snapshot_id`.

Opening EventTrail uses:

```text
snapshot_id + result_id
```

and must not rerun full-corpus embedding, BM25, fusion, or scoring.

### 5.3 Snapshot content

Conceptually:

```python
EvidenceSnapshot:
    snapshot_id: str
    kis_revision: int
    scoring_revision: str
    event_ids: tuple[str, ...]
    decoder_config: DecoderConfigSnapshot
    results: dict[str, SnapshotResult]
    video_evidence: dict[str, VideoEvidence]
    created_at: datetime
    expires_at: datetime

SnapshotResult:
    result_id: str
    video_id: str
    initial_path: tuple[str, ...]   # frame IDs in event order
    path_score: float

VideoEvidence:
    video_id: str
    frame_ids: tuple[str, ...]
    timestamps_ms: tuple[int, ...]
    event_scores: ndarray           # shape [event_count, frame_count]
```

The snapshot stores the final fused event score matrix used by temporal decoding. Stored score arrays and canonical frame metadata are immutable/read-only after publication; a Trail session must receive its own immutable ownership or an independently retained immutable reference whose lifetime is not tied to snapshot eviction. The snapshot does not retain raw embeddings, raw modality matrices, BM25 postings, or LLM outputs.

### 5.4 Returned-result coverage

For every result card returned to the client, EventTrail must be openable from the same snapshot. Every `result_id` is opaque and unique within its snapshot. Evidence is stored per distinct video, not duplicated per result.

If multiple results reference the same video, each result keeps its own initial path while sharing one `VideoEvidence` matrix.

### 5.5 Search degradation rule

Snapshot creation is an enhancement, not a prerequisite for using KIS results. If snapshot materialization fails because of an EventTrail-specific resource problem, the KIS search result remains valid and returns:

```text
evidence_snapshot_id = null
warning = EVENT_TRAIL_UNAVAILABLE
```

Retrieval correctness errors must still fail the search normally; only snapshot-store/materialization failure is degradable.

---

## 6. ET-1: Event Rail and Evidence Inspector

### 6.1 Workspace model

The ranked grid remains the primary fast path. EventTrail opens on demand from a result card through `Explore Path`.

The Trail workspace contains:

1. the selected video player;
2. a compact ordered Event Rail showing the current temporal hypothesis;
3. a single selected-event Evidence Inspector;
4. Trail-level controls for Undo, Back to Results, optional range/window, and Submit.

### 6.2 Event Rail

Each event node shows only user-relevant evidence:

- event ID (`E1`, `E2`, ...);
- short event text when available;
- modality marker(s) when useful;
- aligned thumbnail;
- timestamp;
- long-lived state: proposed or approved;
- transient change indication after re-decoding.

The rail does not show:

- entity graphs;
- raw score matrices;
- decoder costs;
- raw confidence values;
- BM25/dense component details.

Event positions remain fixed in semantic order. Re-decoding changes the candidate inside a node, never the semantic ordering of nodes.

### 6.3 Evidence Inspector

Selecting an event seeks the player to that event candidate and opens a local filmstrip/context view around the current proposal.

The four primary event actions are:

- `Explore`: non-mutating inspection/scrubbing;
- `Use`: anchor the event at the user-selected canonical frame and select that frame for submission;
- `Approve`: anchor the event at its current Trail candidate;
- `Decline`: reject the current candidate occurrence and request a new constrained path.

`Explore` is client-side inspection and does not create a Trail revision.

---

## 7. ET-2: Constraint Semantics and Re-Decoding

### 7.1 Constraint state

A Trail session is modeled as:

```text
C = (A, R, W)
```

where:

- `A_i` is an optional hard anchor frame for event `i`;
- `R_i` is the accumulated set of rejected temporal cells for event `i`;
- `W` is an optional selected-video temporal window applying to every event.

The decoder computes:

```text
P*(C) = argmax Score(P), P in valid paths under C
```

using the frozen selected-video event-score matrix.

### 7.2 Approve and Use

`Approve(Ei)` creates a hard anchor at the event's current canonical candidate.

`Use(Ei, frame_id)` creates a hard anchor at the explicitly selected canonical frame.

Hard anchors never move during later re-decoding unless the user explicitly clears/undoes the anchor.

An approved event returned in `changed_event_indices` without a corresponding unlock/undo is an invariant violation.

### 7.3 Decline

`Decline(Ei)` rejects the current occurrence, not the semantic event itself.

The backend derives the rejected interval from the canonical indexed frame neighborhood. If adjacent canonical timestamps are available, the rejected cell uses midpoint boundaries around the selected frame. For the first canonical frame, the left boundary is the selected video's first canonical timestamp; for the last canonical frame, the right boundary is the selected video's last canonical timestamp. A single-frame video rejects that one canonical timestamp. An authoritative segment boundary may be used instead only when the canonical index exposes that boundary as part of the same scoring generation.

The client never sends arbitrary decline interval boundaries.

Repeated declines accumulate rejected cells for that event.

### 7.4 Approved event conflicts

An approved event cannot simultaneously be declined. The UI disables Decline for a hard-anchored event until the anchor is explicitly cleared or undone.

### 7.5 Transactional contradictions

For `Approve`, `Use`, `SetWindow`, and equivalent hard constraints:

- construct the candidate constraint state;
- run constrained decoding;
- if no valid path exists, reject the action transactionally;
- preserve the existing Trail state and Trail revision.

The system never silently moves or removes an existing hard anchor.

### 7.6 Exhaustion after Decline

A Decline is intentional negative evidence and may be committed even if it removes the last valid path in the selected video.

In that state:

```text
status = exhausted
current_path = null
last_valid_path = previous valid path
```

The UI preserves the dimmed last valid path and offers Undo or Back to Results. It does not auto-open another video.

### 7.7 Temporal window

The optional advanced window is a selected-video hard constraint applying to all events. A window action that contradicts an existing anchor is rejected transactionally.

### 7.8 Semantic order

EventTrail cannot reorder events. The KIS event topology remains authoritative. If the temporal/narrative order is semantically wrong, the user returns to Query Composer and creates a new KIS revision.

---

## 8. Trail Revision and Undo Semantics

### 8.1 Independent revision domain

Trail revisions are independent from KIS revisions.

Example:

```text
KIS Rev 5
  -> Trail Rev 0
  -> Approve E1: Trail Rev 1
  -> Decline E2: Trail Rev 2
```

### 8.2 Monotonic revisions

Trail revisions are protocol versions and are strictly monotonic.

Undo restores an earlier constraint state but creates a new revision:

```text
Rev 3 --Undo--> Rev 4
```

It never decrements to Rev 2.

This prevents stale requests from becoming valid again.

### 8.3 Optimistic concurrency

Every mutating Trail request includes `expected_trail_revision`. A stale request returns `409 TRAIL_REVISION_CONFLICT` and does not mutate the session.

---

## 9. ET-3: Diff-Aware Visualization

### 9.1 Stable positions, changing evidence

After a Trail action, the UI does not rebuild/reorder the rail. It updates only candidate evidence inside affected event nodes.

### 9.2 Direct versus indirect changes

The backend response distinguishes:

- directly affected event(s): targeted by the user action;
- indirectly affected events: moved because constrained DP changed the remaining path.

For example:

```text
Decline E2
  -> E2 changes directly
  -> E3 changes indirectly through re-alignment
```

### 9.3 Transition metadata

`changed` is not a persistent event state. It is transition metadata.

The latest transition may show:

- old candidate thumbnail/timestamp;
- new candidate thumbnail/timestamp;
- `New proposal` for the directly acted event;
- `Adjusted by path` for indirect changes.

Only the latest old->new transition is surfaced in the main UI. Full interaction history belongs in logs.

### 9.4 Player focus rules

- Decline on selected event and successful replacement: seek to that event's new candidate.
- Approve: keep player at the approved event.
- Use: keep player at the user's selected frame.
- Indirect event changes never auto-seek the player.

The system never steals focus because an unrelated event moved.

### 9.5 Review queue

If additional events changed indirectly, the UI may expose a lightweight `N other events updated` review affordance. Reviewing changed events is optional and must not block further interaction.

---

## 10. ET-4: Relationship to Ranked Results

### 10.1 Local-only feedback

Trail constraints are scoped strictly to the selected result/video. They do not rerank the corpus or mutate other videos.

A local rejection means "this event is not here in this video," not "this pattern is globally irrelevant."

### 10.2 Open from exact result identity

EventTrail opens from `snapshot_id + result_id`, not merely `video_id`.

The initial Trail path must exactly equal the ranked result that the user clicked:

```text
TrailPath(revision=0) == RankedResult.initial_path
```

The backend must not silently run an unconstrained decoder and replace it with a different path at open time.

### 10.3 One active hypothesis

A Trail session contains one active temporal hypothesis. Alternative paths emerge through user feedback and constrained re-decoding. There is no Path #1/#2/#3 selector in v1.

### 10.4 Grid preservation

Opening EventTrail does not destroy the ranked-grid state. Returning preserves:

- scroll position;
- selected result;
- KIS revision;
- snapshot ID if still valid;
- the existing in-memory ranked-result data; the implementation must not explicitly clear/recreate the result collection merely because EventTrail was opened;
- per-result interaction annotations.

No search is rerun merely because the user returns from EventTrail.

### 10.5 Per-result interaction status

The grid may annotate results as:

```text
unvisited | explored | exhausted
```

These interaction states do not modify the original retrieval score or ranking order.

### 10.6 Submission

EventTrail provides a Trail-level `Submit` action using the existing competition submission workflow.

The system does not infer which event should be submitted. A frame selected through event-level `Use` becomes `submission_selection`. If no submission frame has been explicitly selected, Submit requires the user to select one rather than choosing E1/midpoint/another heuristic automatically.

---

## 11. ET-5: Backend State Model and API

### 11.1 Storage model

Two bounded in-memory stores are sufficient for v1:

```text
EvidenceSnapshotStore
EventTrailSessionStore
```

Both use TTL plus bounded capacity/LRU eviction.

Recommended initial operational defaults are:

```text
snapshot_ttl_seconds = 900
session_ttl_seconds  = 1800
```

These are configuration defaults, not research semantics.

A process restart may drop all snapshots and Trail sessions. No persistence guarantee is required. Snapshot TTL and session TTL are fixed from object creation in v1; successful reads/actions do not extend expiration.

### 11.2 Snapshot expiration

An expired snapshot cannot open a new Trail session and returns `410 SNAPSHOT_EXPIRED`.

An already-open Trail session remains usable after its source snapshot expires because the session owns or safely references the frozen selected-video evidence required for continued decoding.

### 11.3 Opening a session

Endpoint:

```http
POST /api/v1/event-trail/open
```

Request:

```json
{
  "snapshot_id": "es_...",
  "result_id": "r_...",
  "expected_kis_revision": 5
}
```

Validation:

1. snapshot exists and is not expired;
2. result exists inside snapshot;
3. expected KIS revision equals snapshot KIS revision;
4. result's video evidence exists;
5. event IDs/count match the snapshot scoring generation;
6. the ranked initial path references frames in the frozen selected-video evidence.

Opening creates `trail_revision = 0`, initializes the effective temporal window to the full timestamp span of the selected `VideoEvidence`, and sets `current_path` to the stored ranked result path without performing a second full-corpus retrieval. The public `window` field may remain `null` to mean this full-video default; the decoder adapter must resolve `null` to the full canonical span before building `Conditions`.

### 11.4 EventTrailSession

Conceptual state:

```python
EventTrailSession:
    session_id: str
    snapshot_id: str
    result_id: str
    video_id: str
    kis_revision: int
    scoring_revision: str

    event_ids: tuple[str, ...]
    video_evidence: VideoEvidence
    decoder_config: DecoderConfigSnapshot

    trail_revision: int
    anchors: tuple[str | None, ...]          # frame IDs by event
    rejected_cells: tuple[tuple[Interval, ...], ...]
    window: Interval | None

    current_path: tuple[str, ...] | None
    last_valid_path: tuple[str, ...] | None
    operation_history: tuple[ConstraintSnapshot, ...]

    submission_selection: str | None
    status: "active" | "exhausted"
```

The concrete implementation may use immutable internal dataclasses; the semantics above are normative.

### 11.5 Single mutation endpoint

Endpoint:

```http
POST /api/v1/event-trail/{session_id}/actions
```

Request:

```python
EventTrailActionRequest:
    expected_trail_revision: int
    action: EventTrailAction
```

`EventTrailAction` is a discriminated union of:

```text
ApproveEvent(event_id)
UseFrame(event_id, frame_id)
DeclineCandidate(event_id)
ClearAnchor(event_id)
SetWindow(start_ms, end_ms)
ClearWindow()
Undo()
```

`Explore` is not a mutation API action.

A read endpoint is also required:

```http
GET /api/v1/event-trail/{session_id}
```

It returns the current `EventTrailState` without incrementing `trail_revision`. This endpoint is the recovery path after a revision conflict and the read path for UI restoration while the in-memory session still exists.

### 11.6 Action ownership

For `ApproveEvent` and `DeclineCandidate`, the client sends only `event_id`. The backend resolves the current candidate from the session's current path. This prevents a stale client from approving/declining an arbitrary timestamp that is no longer current.

For `UseFrame`, the backend validates that `frame_id` is canonical and belongs to the selected video. `DeclineCandidate` against an anchored event is rejected with `409 CONSTRAINT_CONFLICT`; backend validation does not rely only on the UI disabling the action.

### 11.7 Transaction execution

A mutation executes under a per-session local lock:

```text
1. load session
2. verify expected trail revision
3. build candidate constraint state
4. decode under candidate constraints
5. apply action-specific commit rule
6. if committed, increment trail revision once
7. produce UI-ready transition diff
8. publish session
```

The store is process-local, so distributed locking is out of scope.

### 11.8 UI-ready response

The action response exposes semantic state, not decoder internals:

```python
EventTrailState:
    session_id: str
    result_id: str
    video_id: str
    kis_revision: int
    trail_revision: int
    status: "active" | "exhausted"

    path: list[EventCandidate] | None
    last_valid_path: list[EventCandidate] | None

    approved_event_ids: list[str]
    rejected_counts: dict[str, int]
    window: tuple[int, int] | None
    submission_selection: FrameRef | None

    transition: TrailTransition | None
```

`EventCandidate` contains canonical event/frame/timestamp/thumbnail metadata needed by the UI.

`TrailTransition` contains:

```text
action_event_id
old/new candidate pairs for changed events
direct_changed_event_ids
indirect_changed_event_ids
latency_ms
```

The frontend must not have to diff two large response trees to discover what changed.

### 11.9 Error contract

Public errors:

```text
404 SNAPSHOT_NOT_FOUND
404 RESULT_NOT_FOUND
404 TRAIL_SESSION_NOT_FOUND

409 KIS_REVISION_MISMATCH
409 TRAIL_REVISION_CONFLICT
409 CONSTRAINT_CONFLICT

410 SNAPSHOT_EXPIRED
410 TRAIL_SESSION_EXPIRED

422 INVALID_EVENT
422 INVALID_FRAME
422 INVALID_WINDOW
422 NOTHING_TO_UNDO
```

Infrastructure failures use the existing service-level error taxonomy where appropriate and must not be disguised as constraint conflicts.

### 11.10 KIS revision binding

A Trail session is permanently bound to one KIS revision. It is never rebound to a newer semantic revision.

When the Query Composer produces a new KIS revision, the frontend leaves/discards the active Trail session and uses the new KIS result snapshot. The backend does not require a browser-global "latest KIS revision" registry.

A stale browser tab may continue interacting with an explicit older frozen Trail session without corrupting a newer KIS revision.

---

## 12. Reuse of Existing Temporal Code

EventTrail must migrate the existing public temporal-exploration product flow rather than introducing a permanent parallel engine.

Current useful algorithmic primitives already exist in:

- `src/hcmai/temporal/constraints.py` (`Conditions`, interval merging, masks);
- `src/hcmai/orchestration/workflows/temporal_search.py` (`decode_video`, decoder config snapshot);
- `src/hcmai/orchestration/workflows/temporal_exploration.py` (selected-video score retention, guarded revisions, change detection, undo concept);
- `src/hcmai/api/contracts/exploration.py` and `src/hcmai/api/routers/exploration.py` (legacy public contracts to be migrated/deleted after consumers move).

Target responsibility split:

```text
TemporalConstraintDecoder
    algorithm only: frozen VideoEventScores + Conditions -> decoded path

EventTrailService
    product orchestration: session state, actions, revisions, diffs

EvidenceSnapshotStore / EventTrailSessionStore
    bounded short-lived state

EventTrail API
    HTTP contract only
```

The exact internal class name may differ, but there must be one reusable constraint-decoding responsibility and one EventTrail orchestration responsibility.

After frontend/API migration, the old public `TemporalExploration` router/contracts must be deleted rather than kept as a second product interface. Any still-useful decoding logic is extracted/reused internally.

Production code must not branch behavior based on `unittest.mock.Mock` or test-double identity.

---

## 13. Search Artifact Integration

`TemporalSearchService.search_plan()` currently returns only materialized paths and timings while `score_plan()` exposes per-video scores separately. EventTrail requires KIS retrieval to preserve the exact evidence and exact ranked paths from one scoring generation.

The target search orchestration therefore introduces an internal search artifact equivalent to:

```python
TemporalSearchArtifact:
    result: TemporalSearchResult
    video_scores: tuple[VideoEventScores, ...]
    decoder_config: DecoderConfigSnapshot
```

The implementation may choose a different type name, but it must satisfy these invariants:

1. scoring happens once per KIS search;
2. ranked paths and snapshot evidence derive from the same score objects/generation;
3. EventTrail snapshot creation does not call `score_plan()` again;
4. the exact result path is retained under a stable `result_id`;
5. only per-video evidence for returned results is materialized into the snapshot store;
6. the normal KIS product response does not expose raw matrices.

---

## 14. UI Navigation and Interaction State

The UI architecture is conceptually:

```text
SearchWorkspace
  |- Query Composer
  |- Ranked Result Grid
  |- optional EventTrail Workspace
```

The Trail workspace is progressive disclosure, not a permanent second large sidebar.

The implementation plan must inspect the actual frontend repository before binding these concepts to concrete component files. The current reviewed `src_v1.2.zip` contains backend source only, so this spec intentionally fixes behavior and state boundaries but not unverified frontend file paths.

Required frontend state includes:

```text
active KIS revision
active evidence snapshot ID
ranked results
per-result Trail status
optional active Trail session
selected EventTrail event
latest transition diff
submission selection
```

A new semantic KIS revision clears the active Trail reference and per-result Trail state belonging to the previous snapshot.

---

## 15. ET-6: Research Evaluation and Logging

### 15.1 Research question

The primary research question is:

> Can users correct an incorrect multi-event temporal retrieval hypothesis more efficiently by providing feedback directly on event-level evidence, rather than reformulating the whole query or inspecting additional ranked results?

### 15.2 Hypotheses

**H1 — Search efficiency**  
Providing event-level temporal feedback reduces interaction effort needed to reach a correct submission compared with conventional ranked-result browsing/query reformulation.

**H2 — Structured propagation**  
Correcting one erroneous event can cause the constrained temporal decoder to improve or change additional event positions without explicit feedback on those events.

A changed event is not automatically considered improved. Correctness claims require ground truth or suitable annotation.

**H3 — Interactive latency**  
Reusing frozen retrieval evidence makes Trail feedback substantially cheaper than rerunning full retrieval, because a Trail action performs only constraint update plus selected-video temporal decoding.

### 15.3 Evaluation layers

Evaluation is split into:

1. algorithmic/system behavior;
2. interaction behavior.

Do not conflate local decoder latency with human task efficiency.

### 15.4 Interaction conditions

Keep the retrieval stack fixed across conditions.

Recommended comparison:

```text
A. Grid
   ranked retrieval + ordinary browsing

B. Grid + Temporal Path
   show ordered retrieved event path but no corrective feedback

C. Grid + EventTrail
   temporal path + Approve/Decline/Use + constrained re-decoding + diffs
```

`A -> B` isolates value of temporal-path visibility.  
`B -> C` isolates value of interactive correction.

Users in condition C are not forced to open EventTrail on every query.

### 15.5 Primary interaction outcomes

The primary metrics are:

- successful correct submission;
- time to correct submission;
- number of videos inspected;
- number of semantic query reformulations.

### 15.6 Secondary Trail metrics

Secondary diagnostics may include:

- Trail usage rate;
- Trail actions per solved query;
- approve/decline/use/undo counts;
- exhausted videos before target;
- indirect changed events per feedback action;
- undo rate;
- per-action local decoding latency.

No secondary metric should be presented as correctness evidence unless the available labels support that interpretation.

### 15.7 Interaction log taxonomy

Use one common event envelope with typed payloads.

Families:

```text
QUERY
  query_initial
  query_event_patch
  query_global_rewrite
  query_search_only

RESULT
  result_impression
  result_open
  result_back

VIDEO
  video_seek
  video_play
  video_pause

TRAIL
  trail_open
  trail_approve
  trail_decline
  trail_use
  trail_undo
  trail_window
  trail_exhausted
  trail_close

SUBMISSION
  submission_select
  submission_send
  submission_result
```

Common envelope:

```python
InteractionEvent:
    timestamp
    search_session_id
    kis_revision
    type

    snapshot_id: str | None
    result_id: str | None
    video_id: str | None
    trail_session_id: str | None
    trail_revision: int | None
    event_id: str | None

    payload: dict
```

Trail mutation payloads record at least:

- target event;
- path before and after using canonical frame IDs;
- direct changed event IDs;
- indirect changed event IDs;
- action latency;
- exhausted/active outcome where applicable.

Do not log full score matrices or embeddings in interaction logs.

### 15.8 Latency stages

KIS search logging retains the existing stage timings and adds snapshot materialization timing.

Trail action timing is separated into at least:

```text
constraint_ms
dp_ms
diff_ms
total_ms
```

### 15.9 Claim boundaries

The architecture itself may support claims that EventTrail:

- exposes event-level temporal evidence correction;
- preserves confirmed evidence under local constrained re-decoding;
- reuses retrieval evidence instead of rerunning full-corpus retrieval.

Claims that EventTrail improves retrieval accuracy, user speed, or interaction effort require corresponding empirical evidence.

---

## 16. Error and Failure UX

The UI must distinguish recoverable interaction outcomes from infrastructure errors.

### Constraint conflict

Example: a new hard anchor conflicts with an already approved earlier/later event.

Behavior:

- action is not committed;
- current path remains unchanged;
- user is told which new action conflicts;
- UI may offer navigation to the relevant approved event;
- no existing approval is silently changed.

### Exhausted selected video

Behavior:

- Decline remains committed;
- retain dimmed `last_valid_path`;
- show no valid path remains in this video;
- offer Undo and Back to Results;
- no automatic global rerank or video switch.

### Expired snapshot

Behavior:

- existing open Trail sessions remain valid;
- new Trail open from expired snapshot fails with `410`;
- user may rerun the current KIS search to obtain a fresh snapshot.

### Expired Trail session

Behavior:

- return to preserved ranked results;
- reopening creates a new Trail session from a still-valid snapshot if available;
- otherwise rerun search on explicit user action.

---

## 17. Testing Strategy

The implementation plan must use targeted tests rather than broad product-test overengineering.

### 17.1 Constraint decoder tests

Verify:

- hard anchor remains fixed while neighboring events re-align;
- repeated rejected cells accumulate;
- midpoint-derived rejection excludes the intended canonical occurrence;
- approved frames never appear in changed indices under unrelated actions;
- contradictory hard constraint produces no state mutation;
- decline may commit an exhausted state;
- window composition respects hard anchors;
- clear/undo relaxes constraints correctly.

### 17.2 Snapshot tests

Verify:

- search scoring executes once;
- snapshot stores only returned-result video slices;
- multiple results for one video share one evidence slice;
- open uses stored evidence and does not call full-corpus scoring;
- initial Trail path exactly equals ranked result path;
- expired snapshot cannot open a new session;
- already-open session survives source snapshot eviction/expiration.

### 17.3 Revision/concurrency tests

Verify:

- every committed mutation increments Trail revision once;
- rejected transactional conflict does not increment revision;
- Undo increments revision rather than decrementing;
- stale `expected_trail_revision` returns 409 without mutation;
- per-session locking prevents out-of-order mutation publication.

### 17.4 API contract tests

Verify every documented error code and discriminated action contract, especially stale result IDs, invalid event IDs, invalid frame IDs, expired handles, and `NOTHING_TO_UNDO`.

### 17.5 KIS integration tests

Through the actual KIS orchestration path, verify:

```text
KIS search
  -> ranked result + snapshot ID
  -> EventTrail open by result ID
  -> Decline event
  -> constrained path changes
  -> Back/reopen semantics
```

Also verify a KIS semantic revision creates a new snapshot world rather than mutating an existing Trail session.

### 17.6 Frontend acceptance tests

When the full frontend source is available, verify:

- Explore Path opens the exact selected result;
- grid state/scroll is retained after Back;
- Approve/Decline/Use map to the correct backend operations;
- direct and indirect changed events are visually distinct;
- indirect changes do not steal player focus;
- exhausted state preserves the last valid path;
- Submit requires an explicit `Use` selection;
- a new KIS revision closes/discards the active Trail UI state.

---

## 18. Migration and Deletion Strategy

Migration sequence:

```text
1. finish S0/S1 readiness gate
2. introduce reusable search artifact/evidence snapshot
3. introduce internal temporal constraint decoder boundary
4. introduce EventTrail stores/service/API
5. migrate UI from public TemporalExploration flow to EventTrail
6. verify equivalent/more precise behavior
7. delete old public TemporalExploration API/contracts/unused models
8. remove dead compatibility branches and generated artifacts
```

There must not be two long-lived public APIs for the same interaction concept.

The existing constraint and DP algorithms are reused; the old product/session orchestration is replaced.

---

## 19. Completion Criteria

EventTrail v1 is implementation-complete only when all of the following are true:

1. S0/S1 prerequisite gate is green.
2. KIS search can produce ranked results plus an optional evidence snapshot without rescoring.
3. Every returned result covered by a snapshot can open an exact matching Trail path by `result_id`.
4. EventTrail hot-loop actions perform no LLM call, embedding call, BM25 search, or full-corpus score pass.
5. Approve and Use create hard event anchors.
6. Decline derives and accumulates canonical local rejection cells.
7. Conflicting hard actions are transactional and never silently change existing approvals.
8. Decline can produce a committed exhausted-video state with a preserved last valid path.
9. Trail revisions are independent of KIS revisions and monotonic through Undo.
10. Responses expose direct and indirect event diffs without raw retrieval internals.
11. Ranked-grid ordering remains unchanged by local Trail feedback.
12. Returning to the grid preserves the search workspace state and per-result interaction status.
13. Submission uses an explicit frame selected with Use.
14. Snapshot/session TTL and error behavior match this spec.
15. Interaction and latency logging captures the ET-6 taxonomy needed for evaluation.
16. The legacy public TemporalExploration product API is removed after consumer migration.
17. Backend tests and frontend build/tests for the affected flow pass on the implementation revision.
18. No EventTrail code path depends on test-mock identity, silent provider fallback, or duplicated scoring.

---

## 20. Design Summary

EventTrail has one deliberately narrow responsibility:

```text
Expose the system's current multi-event temporal hypothesis,
let the user correct evidence at the event level,
and propagate that correction through a constrained decoder.
```

The final architecture is:

```text
KIS Search
  -> TemporalSearchArtifact
       -> ranked results
       -> short-lived EvidenceSnapshot

ranked result + snapshot
  -> EventTrailSession(one video, one KIS revision)
       -> Event Rail + Evidence Inspector
       -> Approve / Decline / Use
       -> TemporalConstraintDecoder
       -> UI-ready transition diff
       -> optional Submit

semantic query change
  -> new KIS revision
  -> new retrieval snapshot
  -> old Trail world is not reused
```

This keeps the competition fast path simple, isolates research novelty in event-level evidence correction, and gives the later paper a measurable distinction between retrieval, temporal explanation, and interactive structured feedback.
