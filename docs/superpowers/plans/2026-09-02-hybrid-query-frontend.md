# Hybrid Query Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the retired Filter workspace with one Query workspace that supports Original + five generated retrieval hypotheses, independent Dense/BM25 toggles, and stateless KIS/TRAKE candidate retrieval.

**Architecture:** `SearchWorkspace` owns query/candidate/search state; API modules remain transport-only. `ToolBox` exposes Dense/BM25 capability switches plus Top-K, and a dedicated `QueryCandidatePanel` renders the literal translation and five candidate bundles. Mode detection always uses the original textarea, never the selected English candidate.

**Tech Stack:** React, Jest/React Testing Library, existing `requestJson` API client, existing CSS files/components.

**Spec:** `docs/superpowers/specs/2026-09-02-hybrid-dense-bm25-query-preparation-design.md`

## Global Constraints

- Remove Filter navigation/page/API completely; do not repurpose it.
- Dense and BM25 default ON when their backend capabilities are available; both OFF disables every Retrieve action.
- Generate Candidates never runs retrieval.
- Original retrieval sends no `retrieval_events`.
- Candidate retrieval sends concrete candidate `events`, never `candidate_id`.
- Literal English is displayed for debugging but is not a sixth candidate and has no independent Retrieve button.
- Editing original input immediately invalidates literal/candidates/candidate errors.
- KIS/TRAKE mode is determined from original input only.
- Enter on ordinary KIS input keeps the fast `Retrieve Original` path.
- Health disables unavailable toggles/actions; do not silently submit a mode backend says is unavailable.
- Existing KIS alignment, TRAKE path visualization, and submission behavior remain unchanged.

---

## File Map

**Create:**
- `src/api/queryCandidates.js`
- `src/api/queryCandidates.test.js`
- `src/features/search/components/QueryCandidatePanel.jsx`
- `src/features/search/components/QueryCandidatePanel.test.jsx`
- `src/styles/query-candidates.css`

**Modify:**
- `src/App.jsx`
- `src/App.test.jsx`
- `src/api/search.js`
- `src/api/search.test.js`
- `src/features/search/components/SearchWorkspace.jsx`
- `src/features/search/components/SearchWorkspace.test.jsx`
- `src/features/search-controls/components/ToolBox.jsx`
- `src/features/search-controls/components/ToolBox.test.jsx`
- `src/features/docs/components/ApiDocsModal.jsx`
- `src/features/docs/components/ApiDocsModal.test.jsx`
- `src/styles/index.css`
- `src/styles/controls.css`

**Delete:**
- `src/api/filter.js`
- `src/api/filter.test.js`
- `src/features/filter/` entire directory.
- `src/styles/filter.css`

---

### Task 1: Delete the Filter Workspace and Make Query the Only Workspace

**Files:**
- Modify: `src/App.jsx`
- Modify: `src/App.test.jsx`
- Delete: `src/api/filter.js`
- Delete: `src/api/filter.test.js`
- Delete: `src/features/filter/`
- Delete: `src/styles/filter.css`
- Modify: `src/styles/index.css`

**Interfaces:**
- App renders `SearchWorkspace` directly as the only retrieval workspace.
- No `activePage`, Filter nav, Filter frame handler, or filter stylesheet import remains.

- [ ] **Step 1: Rewrite App tests to assert Query exists and Filter navigation does not**

```javascript
expect(screen.getByText(/HCMAI 2026 Frame Retrieval/i)).toBeInTheDocument();
expect(screen.queryByRole('button', { name: /^Filter$/i })).not.toBeInTheDocument();
```

- [ ] **Step 2: Run test to verify it fails against current UI**

Run: `npm test -- --watchAll=false src/App.test.jsx`

Expected: FAIL because Filter button still exists.

- [ ] **Step 3: Remove Filter state/import/nav/panel and dead handler from `App.jsx`**

Render `SearchWorkspace` directly and pass existing `healthData` into it for capability gating added in Task 6. Update Vim Top-K enablement so it no longer depends on `activePage`.

- [ ] **Step 4: Delete Filter files and stylesheet import**

```bash
rm -rf src/features/filter
rm -f src/api/filter.js src/api/filter.test.js src/styles/filter.css
```

Remove the `filter.css` import from `src/styles/index.css`.

- [ ] **Step 5: Run App test and closure grep**

```bash
npm test -- --watchAll=false src/App.test.jsx
rg -n "FilterWorkspace|/api/v1/filter|features/filter|styles/filter" src
```

Expected: test PASS; grep returns no live Filter workflow reference.

- [ ] **Step 6: Commit**

```bash
git add -A src
git commit -m "refactor: remove filter workspace"
```

---

### Task 2: Extend Search API and Add Query-Candidate API Client

**Files:**
- Modify: `src/api/search.js`
- Modify: `src/api/search.test.js`
- Create: `src/api/queryCandidates.js`
- Create: `src/api/queryCandidates.test.js`

**Interfaces:**
- `searchFrames({ query, retrievalEvents, useDense, useBm25, topK, signal })`
- `searchTrake({ events, retrievalEvents, useDense, useBm25, topK, signal })`
- `generateQueryCandidates({ query, events, signal })`

- [ ] **Step 1: Write payload tests for Original and Candidate search**

Original KIS must omit `retrieval_events`:

```javascript
expect(lastBody).toEqual({
  query: 'mot co gai',
  use_dense: true,
  use_bm25: false,
  top_k: 20,
});
```

Candidate KIS must include concrete events:

```javascript
expect(lastBody.retrieval_events).toEqual(['a woman']);
```

Repeat the same assertions for TRAKE while preserving original Vietnamese `events`.

- [ ] **Step 2: Write candidate-client contract tests**

KIS sends `{ query }`; TRAKE sends `{ events }`. Reject responses unless:

```javascript
Array.isArray(payload.original_events)
Array.isArray(payload.literal_en)
payload.candidates.length === 5
payload.candidates.every((item, i) => item.index === i + 1 && Array.isArray(item.events))
typeof payload.query_preparation_ms === 'number'
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `npm test -- --watchAll=false src/api/search.test.js src/api/queryCandidates.test.js`

Expected: FAIL against old API functions.

- [ ] **Step 4: Extend API functions**

Only include `retrieval_events` when `retrievalEvents` is a concrete array. Always send `use_dense` and `use_bm25`. Do not send null candidate IDs/indexes.

- [ ] **Step 5: Run tests and commit**

```bash
npm test -- --watchAll=false src/api/search.test.js src/api/queryCandidates.test.js
git add src/api/search.js src/api/search.test.js src/api/queryCandidates.js src/api/queryCandidates.test.js
git commit -m "feat: add hybrid query api clients"
```

---

### Task 3: Add Dense/BM25 Controls to ToolBox

**Files:**
- Modify: `src/features/search-controls/components/ToolBox.jsx`
- Modify: `src/features/search-controls/components/ToolBox.test.jsx`
- Modify: `src/styles/controls.css`

**Interfaces:**
- New controlled props: `useDense`, `setUseDense`, `useBm25`, `setUseBm25`, `denseAvailable`, `bm25Available`.
- Parent owns mode state; ToolBox only renders/changes it.

- [ ] **Step 1: Write control and availability tests**

Assert both switches reflect controlled props, an unavailable capability disables its switch, and changing Dense never changes BM25 or vice versa.

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- --watchAll=false src/features/search-controls/components/ToolBox.test.jsx`

Expected: FAIL because switches are absent.

- [ ] **Step 3: Add accessible switch controls above Top-K**

Use visible labels exactly `Dense Retrieval` and `BM25`. Use native checkbox semantics or `role="switch"` with `aria-checked`; tests must query by accessible role/name rather than CSS class.

- [ ] **Step 4: Run tests and commit**

```bash
npm test -- --watchAll=false src/features/search-controls/components/ToolBox.test.jsx
git add src/features/search-controls/components/ToolBox.jsx src/features/search-controls/components/ToolBox.test.jsx src/styles/controls.css
git commit -m "feat: add dense and bm25 controls"
```

---

### Task 4: Build the Candidate Panel as a Pure UI Component

**Files:**
- Create: `src/features/search/components/QueryCandidatePanel.jsx`
- Create: `src/features/search/components/QueryCandidatePanel.test.jsx`
- Create: `src/styles/query-candidates.css`
- Modify: `src/styles/index.css`

**Interfaces:**
- Props:

```javascript
{
  originalEvents,
  literalEn,
  candidates,
  onRetrieveOriginal,
  onRetrieveCandidate,
  retrieveDisabled,
  isSearching,
}
```

- `onRetrieveCandidate` receives concrete `candidate.events` only.

- [ ] **Step 1: Write rendering tests**

Assert Original section, Literal English section, exactly five Candidate headings, preserved event order, no Retrieve button for Literal English, and five candidate Retrieve buttons plus one Original Retrieve button.

- [ ] **Step 2: Write interaction test proving concrete events are returned**

```javascript
fireEvent.click(screen.getByRole('button', { name: /Retrieve Candidate 3/i }));
expect(onRetrieveCandidate).toHaveBeenCalledWith(['event 1 en', 'event 2 en']);
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npm test -- --watchAll=false src/features/search/components/QueryCandidatePanel.test.jsx`

Expected: FAIL because component is absent.

- [ ] **Step 4: Implement component and compact styles**

Render event lines as `E1`, `E2`, ... based purely on array order. Do not re-parse English candidate text. The candidate index is presentation-only.

- [ ] **Step 5: Run tests and commit**

```bash
npm test -- --watchAll=false src/features/search/components/QueryCandidatePanel.test.jsx
git add src/features/search/components/QueryCandidatePanel.jsx src/features/search/components/QueryCandidatePanel.test.jsx src/styles/query-candidates.css src/styles/index.css
git commit -m "feat: render retrieval query candidates"
```

---

### Task 5: Refactor SearchWorkspace Into Explicit Generate and Retrieve Actions

**Files:**
- Modify: `src/features/search/components/SearchWorkspace.jsx`
- Modify: `src/features/search/components/SearchWorkspace.test.jsx`

**Interfaces:**
- `Retrieve Original` uses original textarea/mode and no `retrievalEvents`.
- Candidate retrieve calls the same search executor with concrete selected English events.
- Generate Candidates calls only `/query-candidates`.
- Search and candidate generation own separate `AbortController` refs.

- [ ] **Step 1: Add state-behavior tests**

Cover Dense/BM25 defaults, both-off disabled retrieval, Generate Candidates does not call search API, exactly five cards after generation, editing input clears candidate state, and current search results may remain visible after edit.

- [ ] **Step 2: Add KIS routing tests**

Original:

```javascript
expect(searchFrames).toHaveBeenCalledWith(expect.objectContaining({
  query: original,
  retrievalEvents: undefined,
  useDense: true,
  useBm25: true,
}));
```

Candidate:

```javascript
expect(searchFrames).toHaveBeenCalledWith(expect.objectContaining({
  query: original,
  retrievalEvents: candidateEvents,
}));
```

- [ ] **Step 3: Add TRAKE routing tests**

Mode detection comes from `parseTrakeEvents(originalInput)`. Candidate Retrieve calls `searchTrake` with original Vietnamese `events` plus selected English `retrievalEvents`.

- [ ] **Step 4: Run test to verify failures**

Run: `npm test -- --watchAll=false src/features/search/components/SearchWorkspace.test.jsx`

Expected: FAIL until new actions/state exist.

- [ ] **Step 5: Split internal actions**

```javascript
const retrieve = useCallback(async (retrievalEvents) => {
  const raw = eventDescription.trim();
  if (!raw || isSearching || (!useDense && !useBm25)) return;

  const events = parseTrakeEvents(raw);
  const isTrakeMode = events !== null;
  const controller = new AbortController();
  searchRequestRef.current?.abort();
  searchRequestRef.current = controller;
  setIsSearching(true);
  setError(null);

  try {
    const response = isTrakeMode
      ? await searchTrake({
          events,
          retrievalEvents,
          useDense,
          useBm25,
          topK,
          signal: controller.signal,
        })
      : await searchFrames({
          query: raw,
          retrievalEvents,
          useDense,
          useBm25,
          topK,
          signal: controller.signal,
        });
    applySearchResponse(response, isTrakeMode);
  } finally {
    if (searchRequestRef.current === controller) {
      searchRequestRef.current = null;
      setIsSearching(false);
    }
  }
}, [eventDescription, isSearching, topK, useDense, useBm25]);

const retrieveOriginal = useCallback(() => retrieve(undefined), [retrieve]);

const generateCandidates = useCallback(async () => {
  const raw = eventDescription.trim();
  if (!raw || isGeneratingCandidates || !queryPreparationAvailable) return;
  const events = parseTrakeEvents(raw);
  const controller = new AbortController();
  candidateRequestRef.current?.abort();
  candidateRequestRef.current = controller;
  setIsGeneratingCandidates(true);
  setCandidateError(null);
  try {
    const response = await generateQueryCandidates({
      query: events === null ? raw : undefined,
      events: events === null ? undefined : events,
      signal: controller.signal,
    });
    setCandidateSet(response);
  } finally {
    if (candidateRequestRef.current === controller) {
      candidateRequestRef.current = null;
      setIsGeneratingCandidates(false);
    }
  }
}, [eventDescription, isGeneratingCandidates, queryPreparationAvailable]);
```

Extract `applySearchResponse(response, isTrakeMode)` from the existing response-state block so Original and candidate retrieval share exactly one result-materialization path. Preserve the current AbortError handling when moving the existing catch logic into these callbacks.

- [ ] **Step 6: Make form submit/Enter call `retrieveOriginal` only**

Generating candidates is button-only and never runs from Enter.

- [ ] **Step 7: Invalidate candidate state synchronously on original text edit**

Create one `handleDescriptionChange` that sets `eventDescription` and clears `candidateSet` plus generation error immediately. Do not wait for an effect that leaves stale candidate buttons clickable for a render.

- [ ] **Step 8: Reset Parameters**

Reset must set Dense=true, BM25=true, Top-K=20 for capabilities that are currently available. It does not erase the original query.

- [ ] **Step 9: Run tests and commit**

```bash
npm test -- --watchAll=false src/features/search/components/SearchWorkspace.test.jsx
git add src/features/search/components/SearchWorkspace.jsx src/features/search/components/SearchWorkspace.test.jsx
git commit -m "feat: retrieve original or generated queries"
```

---

### Task 6: Wire Health-Based Capability Gating

**Files:**
- Modify: `src/App.jsx`
- Modify: `src/features/search/components/SearchWorkspace.jsx`
- Modify: `src/App.test.jsx`
- Modify: `src/features/search/components/SearchWorkspace.test.jsx`

**Interfaces:**
- Reads `healthData.capabilities.dense_temporal`, `.bm25`, `.query_preparation` from the existing App health request.

- [ ] **Step 1: Write tests for partial availability**

Cases:
- Qwen unavailable: Generate Candidates disabled; Dense-only Original remains possible.
- BM25 unavailable: BM25 switch disabled/off; Dense can run.
- Dense unavailable: Dense switch disabled/off; BM25 can run if available.
- neither retrieval family available: all Retrieve actions disabled.

- [ ] **Step 2: Run tests to verify failure**

Run: `npm test -- --watchAll=false src/App.test.jsx src/features/search/components/SearchWorkspace.test.jsx`

Expected: FAIL before health props are wired.

- [ ] **Step 3: Pass capabilities from existing `useHealthCheck()` in App**

Do not start a second health polling hook inside SearchWorkspace.

- [ ] **Step 4: Reconcile state when capability becomes unavailable**

On available→unavailable, turn that mode OFF. Do not silently send it. On initial healthy load, enable every available retrieval family; if availability later returns, leave explicit user mode choice unchanged until Reset/user action.

- [ ] **Step 5: Run tests and commit**

```bash
npm test -- --watchAll=false src/App.test.jsx src/features/search/components/SearchWorkspace.test.jsx
git add src/App.jsx src/App.test.jsx src/features/search/components/SearchWorkspace.jsx src/features/search/components/SearchWorkspace.test.jsx
git commit -m "feat: gate retrieval controls by backend health"
```

---

### Task 7: Update API Documentation and Response Contract Guards

**Files:**
- Modify: `src/features/docs/components/ApiDocsModal.jsx`
- Modify: `src/features/docs/components/ApiDocsModal.test.jsx`
- Modify: `src/api/search.js`
- Modify: `src/api/search.test.js`

**Interfaces:**
- Docs list `/api/v1/query-candidates`, updated `/search` and `/trake` fields, and no `/filter`.
- Search response validation accepts `dense_events`, `bm25_caption_events`, `use_dense`, `use_bm25` with nullability tied to enabled modes.

- [ ] **Step 1: Write docs and response tests**

Assert query-candidates appears, filter does not, and candidate fields are described as concrete event arrays. Test Dense-only response with `bm25_caption_events=null` and BM25-only response with `dense_events=null`.

- [ ] **Step 2: Run tests to verify failure**

Run: `npm test -- --watchAll=false src/features/docs/components/ApiDocsModal.test.jsx src/api/search.test.js`

Expected: FAIL until docs/client guards are updated.

- [ ] **Step 3: Update docs and response guards**

Require `use_dense`/`use_bm25` booleans. If `use_dense` is true, require `dense_events` array; if false, allow null. Apply the same rule to `bm25_caption_events` and `use_bm25`.

- [ ] **Step 4: Run tests and commit**

```bash
npm test -- --watchAll=false src/features/docs/components/ApiDocsModal.test.jsx src/api/search.test.js
git add src/features/docs/components/ApiDocsModal.jsx src/features/docs/components/ApiDocsModal.test.jsx src/api/search.js src/api/search.test.js
git commit -m "docs: describe hybrid query retrieval api"
```

---

### Task 8: Frontend Regression and Build Gate

**Files:**
- No production changes unless verification exposes a defect.

- [ ] **Step 1: Run targeted feature tests**

```bash
npm test -- --watchAll=false \
  src/api/queryCandidates.test.js \
  src/api/search.test.js \
  src/features/search-controls/components/ToolBox.test.jsx \
  src/features/search/components/QueryCandidatePanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/App.test.jsx
```

Expected: all PASS.

- [ ] **Step 2: Run existing KIS/TRAKE/submission regressions**

```bash
npm test -- --watchAll=false \
  src/features/alignment/components/AlignmentAccordion.test.jsx \
  src/features/search/components/TrakePathCard.test.jsx \
  src/features/frames/components/FramesBox.test.jsx \
  src/features/frames/components/ImageModal.test.jsx \
  src/features/submission/components/SubmissionWorktree.test.jsx
```

Expected: all PASS.

- [ ] **Step 3: Run complete frontend test suite**

Run: `npm test -- --watchAll=false`

Expected: 0 failed suites/tests.

- [ ] **Step 4: Build production bundle**

Run: `npm run build`

Expected: exit 0.

- [ ] **Step 5: Verify retired/stateful concepts are absent**

```bash
rg -n "FilterWorkspace|/api/v1/filter|candidate_id|candidate_session|search_id" src
```

Expected: no matching live workflow.

- [ ] **Step 6: Manual browser smoke matrix**

Run one KIS and one TRAKE query through:

```text
Original + Dense only
Original + BM25 only
Original + Hybrid
Generate Candidates (must not mutate results)
Candidate #1 + Dense only
Candidate #1 + BM25 only
Candidate #1 + Hybrid
```

For TRAKE, confirm every generated bundle retains original event count/order and every returned path remains independent with path-level submission semantics.
