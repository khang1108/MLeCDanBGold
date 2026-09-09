# UI Search Reranking Toggle Design Specification

**Date:** 2026-09-04  
**Feature:** User-tunable "Reranking" Toggle Button in Search UI  
**Target Surface:** `frontend/src/features/search-controls/`, `frontend/src/features/search/`, `frontend/src/api/search.js`, and `src/hcmai/api/contracts/search.py`  
**Status:** PROPOSED FOR REVIEW  

---

## 1. Context & Motivation

A backend agent is implementing the P2a High-Recall Video-Level Reranking pipeline (`docs/superpowers/plans/2026-09-04-hcmai-p2a-high-recall-video-reranking.md`). 
Because VLM reranking involves remote model inference over video candidate frames, it should not run unconditionally on every query. Users need an explicit, intuitive UI toggle on the Search page to decide whether to activate deep VLM reranking or maintain fast candidate retrieval.

---

## 2. Requirements & Scope

1. **User Interface:**
   - Add a toggle switch labeled **"VLM Rerank"** (with copy description *"Deep candidate reranking"*) inside the search sidebar options (`ToolBox.jsx`).
   - Default state must be **OFF (`false`)** to keep retrieval fast by default.
   - The toggle is interactive and decoupled from Dense/BM25 disabling rules (can be toggled on/off independently).
2. **Search Invocation:**
   - Both KIS (`searchFrames` $\to$ `/api/v1/search`) and TRAKE (`searchTrake` $\to$ `/api/v1/trake`) must forward the `use_rerank` boolean in the request body.
3. **Backend Schema Compatibility:**
   - Because `SearchRequest` and `TRAKERequest` specify `ConfigDict(extra="forbid")`, the backend contracts in `src/hcmai/api/contracts/search.py` must declare `use_rerank: bool = False` so incoming requests with the toggle are accepted cleanly without throwing HTTP 422 errors.
4. **History & Session Replay:**
   - Query history snapshots will record whether `use_rerank` was active for that search session.

---

## 3. Detailed Component Design

### A. Frontend: `ToolBox.jsx` (`frontend/src/features/search-controls/components/ToolBox.jsx`)

Add props `useRerank = false` and `setUseRerank = NOOP`.

Add a dedicated section or list item under options:
```jsx
<fieldset className="toolbox-section toolbox-rerank-section">
  <legend className="toolbox-label">Post-processing</legend>
  <div className="toolbox-toggle-list">
    <label className="toolbox-toggle-row">
      <span className="toolbox-toggle-copy">
        <span className="toolbox-toggle-name">VLM Rerank</span>
        <span className="toolbox-toggle-description">Deep candidate reranking</span>
      </span>
      <input
        type="checkbox"
        role="switch"
        className="toolbox-switch"
        checked={useRerank}
        onChange={(event) => setUseRerank(event.target.checked)}
        aria-label="Use VLM Reranking"
      />
    </label>
  </div>
</fieldset>
```

### B. Frontend: `SearchWorkspace.jsx` (`frontend/src/features/search/components/SearchWorkspace.jsx`)

1. State initialization:
   ```javascript
   const [useRerank, setUseRerank] = useState(false);
   ```
2. Pass `useRerank` and `setUseRerank` to `<ToolBox />`.
3. In `submit`, pass `useRerank` into `searchFrames` and `searchTrake`:
   ```javascript
   const response = isTrakeMode
     ? await searchTrake({
         events,
         topK,
         useDense,
         useBm25,
         useRerank,
         signal: controller.signal,
       })
     : await searchFrames({
         query: retrieval.query,
         topK,
         useDense,
         useBm25,
         useRerank,
         signal: controller.signal,
       });
   ```

### C. Frontend API Client: `search.js` (`frontend/src/api/search.js`)

1. Update `searchFrames`:
   ```javascript
   export const searchFrames = async ({
     query,
     topK,
     useDense = true,
     useBm25 = true,
     useRerank = false,
     signal,
   }) => {
     ...
     const payload = await requestJson('/api/v1/search', {
       method: 'POST',
       body: {
         query: query.trim(),
         top_k: topK,
         use_dense: useDense,
         use_bm25: useBm25,
         use_rerank: useRerank,
       },
       signal,
     });
     ...
   };
   ```
2. Update `searchTrake` similarly to send `use_rerank: useRerank`.

### D. Backend Contracts: `search.py` (`src/hcmai/api/contracts/search.py`)

Add the non-breaking field to both `SearchRequest` and `TRAKERequest`:
```python
class SearchRequest(BaseModel):
    ...
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = False
    top_k: int = Field(default=20, ge=1)
```

```python
class TRAKERequest(BaseModel):
    ...
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = False
    top_k: int = Field(default=20, ge=1)
```

When the backend P2a pipeline reads `request.use_rerank`, it can conditionally branch into `VideoCandidateRerankingService`. If `use_rerank=False`, it retains the fast standard retrieval path.

---

## 4. Verification Plan

1. **Frontend Unit Tests:**
   - `ToolBox.test.jsx`: Verify that the toggle renders, toggles `setUseRerank`, and respects accessibility attributes (`role="switch"`, `aria-label`).
   - `SearchWorkspace.test.jsx`: Verify that `useRerank` defaults to `false` and is forwarded to `searchFrames` / `searchTrake`.
   - `search.test.js`: Verify payload JSON serialization contains `use_rerank: false` or `use_rerank: true`.
2. **Backend Contract Tests:**
   - Run `pytest tests/test_search.py` or contract tests to ensure `SearchRequest` and `TRAKERequest` accept `use_rerank: true` without validation errors.
3. **End-to-End Visual Verification:**
   - Open browser at `http://localhost:3000`, test clicking the switch, verify state change and network request payload in browser devtools / fetch logs.
