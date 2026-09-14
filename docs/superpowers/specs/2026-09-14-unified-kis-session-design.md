# Design Specification: Unified Revisioned KIS Session

**Date:** 2026-09-14  
**Status:** Approved  
**Topic:** Multimodal Video Retrieval - Unified Revisioned KIS Session (KIS-T and KIS-C)

---

## 1. Context & User Goal

In video retrieval competitions (such as HCMAI / Ho Chi Minh City AI Challenge and Video Browser Showdown):
- **KIS-T (Known-Item Search - Text)** typically starts with one sufficiently complete natural language description of a scene.
- **KIS-C (Known-Item Search - Clues)** begins with an incomplete text clue and incrementally receives additional clues over time as the round progresses.
- Both types of searches frequently undergo human query refinement during competition rounds.

Previously, each text submission behaved as an isolated one-shot query to `POST /api/v1/search`. For KIS-C, participants had to manually concatenate clues into a single string in the input box, which conflated timeline moments and lacked structured clue lineage.

**Goal:**
1. Unify KIS-T and KIS-C under a single revisioned interaction flow without introducing mutable server-side session stores (no Redis, SQLite session tables, or TTL workers).
2. The frontend owns the ordered clue list and sends the complete list along with the expected revision on each submission to `POST /api/v1/kis/search`.
3. The backend deterministically builds a `KISIntent` from ordered clues, maintaining clear clue boundaries for temporal event planning, then executes retrieval via the shared `KISPipeline` and `TemporalSearchService`.
4. The frontend maintains live session state in a dedicated `KisPanel` displaying the current clue draft, committed clue history, and active event breakdown pills, while keeping live search distinct from Query History replay.
5. Existing `/api/v1/search`, image search, and literal filters remain strictly unchanged for backward compatibility.

---

## 2. Architecture & Data Flow

```text
[User Input]
     │
     ▼
[Frontend: KisPanel] ──(draft text)──► [kisSession.js State Model]
                                                │
                                                ▼
                                   POST /api/v1/kis/search
                                   {
                                     inputs: [ {text: "..."}, {text: "..."} ],
                                     expected_revision: 1,
                                     use_dense, use_bm25, top_k
                                   }
                                                │
                                                ▼
                                    [FastAPI Router: /kis/search]
                                                │  (409 Conflict if revision mismatch)
                                                │  (422 Unprocessable if inputs invalid)
                                                ▼
                                      [KISIntentBuilder]
                                                │  (deterministic query_text & events)
                                                ▼
                                      [KISPipeline.execute_events]
                                                │
                                                ▼
                                    [TemporalSearchService]
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                              Dense Retrieval         BM25 Caption
                                    └───────────┬───────────┘
                                                ▼
                                      Dynamic Programming Alignment
                                                ▼
                                      [SearchMaterializer]
                                                ▼
                                      KISRevisionSearchResponse
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
         [DRES Result Logging]                                      [Frontend SearchWorkspace]
         (event_value = intent.query_text)                                      │
                                                                       ┌───────┴───────┐
                                                                       ▼               ▼
                                                                  [KisPanel]     [Query History]
                                                               (commit revision)  (persist snapshot)
```

---

## 3. Backend Contracts & Boundary

### 3.1 DTOs (`src/hcmai/api/contracts/kis.py`)

- `KISInput`:
  - `text: str` (non-blank, stripped).
- `KISIntent`:
  - `revision: int` (positive integer >= 1).
  - `inputs: list[str]` (ordered list of non-blank clues).
  - `query_text: str` (deterministic joined text).
  - `events: list[str]` (planned timeline moments).
- `KISRevisionSearchRequest`:
  - `inputs: list[KISInput]` (min 1 item).
  - `expected_revision: int` (>= 0).
  - `use_dense: bool = True`, `use_bm25: bool = True` (at least one must be True).
  - `top_k: int = 20` (>= 1).
  - Property `previous_revision`: `len(inputs) - 1`.
  - Property `has_revision_conflict`: `expected_revision != previous_revision`.
- `KISRevisionSearchResponse`:
  - `revision: int`, `inputs: list[KISInput]`, `intent: KISIntent`.
  - `query: str`, `events: list[str]`, `dense_events: list[str] | None`, `bm25_caption_events: list[str] | None`.
  - `use_dense: bool`, `use_bm25: bool`.
  - `results: list[SearchResult]`, `latency: SearchLatency`.

### 3.2 Endpoint `POST /api/v1/kis/search`

1. Checks `service_container.get("service")`: returns 503 if unavailable.
2. Checks `request.has_revision_conflict`: returns `HTTP 409 Conflict` if client expected revision does not match `previous_revision`.
3. Executes `service.search_kis_revision(request)` on threadpool.
4. Catches `(ValueError, InvalidQueryInputError)`: returns `HTTP 422 Unprocessable Content`.
5. Logs to DRES via shared `record_dres_result_log` with `event_value = result.intent.query_text` and `category = "TEXT"`.
6. Attaches `X-DRES-Log-Status` header and returns `KISRevisionSearchResponse`.

---

## 4. Deterministic KISIntent Construction

`KISIntentBuilder` transforms the raw input sequence into structured intent:
- Strips and normalizes whitespace for each input.
- Rejects empty inputs or empty lists.
- Uses `"\n".join(normalized)` when invoking `plan_query_events` so that each clue boundary is recognized as a discrete timeline moment even if the user did not type terminal punctuation.
- Stores `query_text = " ".join(normalized)` for display and DRES logging.
- Enforces `DEFAULT_MAX_TEMPORAL_EVENT_COUNT` (default 8).

---

## 5. Frontend Session State Model & UI

### 5.1 Pure Session Model (`frontend/src/features/kis/kisSession.js`)

- `createKisSession({ draft = '', inputs = [], revision = 0, intent = null })`
- `withKisDraft(session, draft)`
- `buildKisRevisionRequest(session)`:
  - If `draft.trim()` is present: builds `{ inputs: [...session.inputs, { text: draft.trim() }], expectedRevision: session.revision }`.
  - If `draft.trim()` is empty but `session.inputs.length > 0`: allows re-searching existing clues with `{ inputs: session.inputs, expectedRevision: session.revision - 1 }`.
  - Otherwise returns `null`.
- `commitKisRevision(session, response)`:
  - Advances session revision, clears `draft`, records committed `inputs` and `intent`.
- `resetKisSession(options)`: Resets to revision 0 or replay draft.

### 5.2 `KisPanel` Presentation Component (`frontend/src/features/kis/components/KisPanel.jsx`)

- Textarea for typing query / clues with auto-height adjustment and Enter (submit) vs Shift+Enter (multiline).
- Clue History list (`<ol className="kis-clue-history">`): displays numbered clues (`#1`, `#2`, ...).
- Current Intent pills (`<div className="kis-current-intent">`): displays planned events (`E1`, `E2`, ...).
- Dark-theme compliant styling in `frontend/src/styles/workspace.css`.

---

## 6. Query History & Replay

- History snapshots (`buildKisSnapshot`) optionally persist `kis_revision` and `kis_inputs`.
- When replaying from Query History, the UI displays the replayed clues and intent events in `KisPanel` in visual inspection mode.
- Clicking `New Search` aborts any inflight request and resets `kisSession` to revision 0.

---

## 7. Invariants & Compatibility

- `/api/v1/search` remains 100% backward compatible:
  - Empty or whitespace query returns empty `SearchResponse` without raising `ValueError`.
  - Valid queries continue to plan events and query temporal retrieval identically.
- Canonical frame identity (`video_id`, `frame_id`, `frame_idx`, `timestamp_ms`) is strictly preserved across all layers.
