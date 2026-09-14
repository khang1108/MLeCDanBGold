# Scoped temporal feedback: frontend/DRES handoff

This document hands the implemented Task 4/5 backend core to the frontend and
DRES integration owner. It describes callable Python objects and existing
contracts only. Transport, UI, session storage, and DRES integration are not
implemented in this checkout.

## Current boundary

`POST /api/v1/search` remains backward-compatible. Its `SearchResponse` carries
the original `events`, optional `dense_events` and `bm25_caption_events`, and
`use_dense`/`use_bm25` flags. The additive local exploration transport is now
available at `/api/v1/exploration`; it owns branch feedback and does not claim
candidate or DRES integration. Feedback must
not be put in `retrieval_events` or routed through `/trake`; the integration PR
must add an explicitly reviewed additive transport contract.

The core callable is
`hcmai.orchestration.workflows.temporal_exploration.TemporalExploration`.
It owns one selected video's copied score matrix and revisioned conditions. It
does not own API sessions, persistence, full-corpus caches, answer submission,
or DRES logging.

## Flow mapping

| User flow | Backend mapping | Integration owner / rule |
| --- | --- | --- |
| Open selected result | Build `QueryBinding` from the search response event/scoring snapshot; call `open(binding, video_id, window)` with an explicit integer-ms window. | Frontend retains the global search response. Backend supplies `event_version` and `scoring_revision`. |
| Confirm/reject | Call `apply(expected_revision=..., event_version=..., scoring_revision=..., action="confirm", event_index=..., interval=...)` or the same call with `action="reject"`. | Controls call only after explicit user action. Stale guards reject old work. |
| Search around timestamp | Call `apply(expected_revision=..., event_version=..., scoring_revision=..., action="window", event_index=None, interval=...)`. | Show bounds. This changes the window only; it does not confirm an event. |
| Unknown | Call `apply(expected_revision=..., event_version=..., scoring_revision=..., action="unknown", event_index=...)`, or do not mutate. | Unknown is distinct from negative/reject. Any interaction log needs its own label. |
| Undo | Call `undo(expected_revision=..., event_version=..., scoring_revision=...)`. | Use the current exploration binding/revision. Never use `AnswerWorkspace` revision. |
| Query/decomposition refresh | Preserve the old view for comparison, call `close(expected_revision=...)`, then `open()` with a new binding. | Do not remap old feedback by event index. |
| Return global results | Call `close(expected_revision=...)`, then restore the retained global snapshot. | Do not call the retriever merely to restore the UI. |
| Select answer | Pass the canonical exploration result/inspector selection into the existing editable candidate dialog. | Core performs no workspace mutation and no auto-submit. |
| Result logging | On successful materialization, use the teammate's shared result logger. | Do not add a DRES client or double-log. Logging failure must not erase the branch. |

## Callable and identity contract

`QueryBinding` fields are `query`, `event_version`, normalized `events`,
normalized `retrieval_events`, optional normalized `caption_events`,
`use_dense`, `use_bm25`, and backend-owned `scoring_revision`. The event arrays
must have matching lengths. The first successful open starts at revision 1. Later opens on the same instance continue increasing the revision; close does not reset it. Requests from a closed lifecycle must not mutate or close a reopened branch.

`apply()` accepts only `confirm`, `reject`, `window`, or `unknown`. Every
mutation checks exploration revision, event version, and scoring revision
before validation/evaluation. `unknown` is a no-op. `undo()` creates a new
revision while restoring the prior condition snapshot. `current()` returns the
immutable `ExplorationView`; `close()` requires the current exploration
revision.

`ExplorationView` exposes `revision`, `event_version`, `video_id`, `conditions`,
`status`, `paths`, `changed_event_indices`, `comparison_available`, and
`can_undo`. Status is one of `ok`, `contradictory_conditions`,
`no_indexed_frames`, or `no_valid_path`; these are content/evidence outcomes,
not `video_valid` verdicts. Known scoring/evaluation I/O failures raise
`ExplorationUnavailable`; stale state raises `ExplorationConflict`.

`AlignedPath` is the canonical handoff result: `video_id`, `score`,
`frame_ids`, `frame_idxs`, and `timestamps_ms`. Preserve `video_id`, `frame_id`,
`frame_idx`, and `timestamp_ms` through every UI action. `frame_idx` preserves the original HCMAI frame coordinate as metadata. VBS/DRES submissions use the organizer media-ID mapping and the selected `timestamp_ms`, with equal start/end under the agreed point-answer contract. Never derive the submission timestamp from `frame_idx`. `changed_event_indices` compares `(frame_id,
timestamp_ms)` pairs, not scores, so a confirmed interval can retain its badge
while its selected frame moves.

## Session and deployment constraints

- Allocate one exploration handle per tab/session identity. It is independent
  of whether a DRES connection exists.
- When the integration layer creates a new exploration instance, issue a fresh handle and retire the old handle. Do not bind delayed requests from an old handle to the current instance. Responses are matched by handle and revision.
- Remove handles on close, disconnect, or expiry using the integration
  infrastructure. Bound active sessions and enforce one matrix/video/active
  branch per handle; this is a capacity bound, not an optimized throughput
  cache.
- `scoring_revision` is backend-owned. The client echoes it for stale-write
  detection and never manufactures or changes it.
- With multiple workers, reuse an existing session store/routing mechanism. If
  none exists, run exploration on one worker. Un-routed in-memory state across
  workers is invalid.
- `dataclasses.asdict(view)` is suitable for an internal replay fixture only.
  It is not a declared production JSON API; production serialization requires
  the schema agreed in the integration PR.

## Task 5 fixture and UI rehearsal

The replay fixture uses three events, five canonical frames, timestamps
`[10, 20, 30, 40, 50]`, a fake scorer, and the real decoder. Expected path
trace: open → `(0,1,3)`; confirm event 0 `[10,20]` → `(0,1,3)`; reject event 2
`[40,40]` → `(0,1,4)`; undo → `(0,1,3)`; confirm event 0 `[20,20]` →
`(1,2,3)`; window `[0,15]` → `contradictory_conditions`; undo → `(1,2,3)`.
Use `dataclasses.asdict(view)` only to exchange this internal fixture.

| Rehearsal case | Expected behavior |
| --- | --- |
| Revision 4 response arrives after revision 5 | Keep revision 5; key responses by branch handle plus revision, not integer revision alone. |
| Confirm `[80000,85000]`, frame moves 81000 → 83000 | Keep the condition badge; report the canonical frame diff. |
| Reject event B | Event A and C remain independently usable. |
| Failure, then undo | Restore branch conditions/path; retain the global result list. |
| Inspector `currentTime=12.3456s` | Pass `12346ms` to the dialog; do not snap. |
| Card `timestamp_ms=12000` | Pass `12000ms`; never derive it from `frame_idx`. |
| Select answer from exploration | Open editable dialog; Enter saves candidate; submission remains separate. |
| DRES/logging failure | Keep retrieval and branch state; expose logger status per the teammate's plan. |
| Two tabs | Isolate feedback; no cross-tab overwrite unless a separate sharing feature is designed. |

## Current-checkout audit

Audit commands from the plan were run on this checkout:

```text
rg -n 'create_search_router|SearchRequest|SearchResponse|X-VBS-User-ID' src/hcmai
rg -n 'AnswerCandidateDialog|onAddCandidate|workspaceAction|X-DRES-Log-Status' frontend/src
rg -n 'revision|session|service_container' src/hcmai/api src/hcmai/orchestration
```

Results: the first audit finds `create_search_router`, `SearchRequest`, and
`SearchResponse` in the existing search route/contracts, but no
`X-VBS-User-ID`. The second audit returns no matches. The third finds existing
workspace/history revisions, router `service_container` wiring, and the
Task 4/5 exploration revisions, but no exploration session/transport layer.
Therefore the frontend owner must map real component, endpoint, session, and
logger names on their current checkout; this document invents none. UI
integration and end-to-end rehearsal are not complete here.

## Next-owner checklist

- [x] Define and review additive local exploration transport; keep `/api/v1/search`
      backward-compatible.
- [ ] Map the transport to the real frontend result/inspector and candidate
      dialog components; preserve the global snapshot.
- [ ] Implement per-tab/session lifecycle, expiry, capacity limits, and
      multi-worker routing or single-worker affinity.
- [ ] Serialize `ExplorationView`/`AlignedPath` through the agreed canonical
      schema; do not publish the `asdict` fixture as API v1.
- [ ] Wire explicit confirm/reject/window/unknown/undo controls and stale
      response handling.
- [ ] Rehearse every case in the table, including millisecond conversion and
      two-tab isolation.
- [ ] Reuse the shared result logger; verify DRES/logging failure isolation.
- [ ] Run existing backend contracts plus frontend/DRES tests after merge;
      report UI and transport gaps separately from Task 5 functional evidence.
