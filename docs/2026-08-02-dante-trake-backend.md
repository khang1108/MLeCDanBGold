# DANTE TRAKE Backend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an exact DANTE path for TRAKE that retrieves one video and exactly one chronologically ordered canonical `frame_idx` per query event, without ANN pruning or accuracy-risking shortcuts.

**Architecture:** The existing encoder and exact FAISS `IndexFlatIP` produce event-to-frame similarities. A pure DANTE aligner performs dynamic programming independently inside each video's canonical frame sequence. The orchestration layer materializes official video names and `frame_idx` values through `FrameStore`; the existing dispatcher exposes TRAKE through `POST /api/v1/search`. TRAKE does not pass through KISC, VQA, or an LLM.

**Tech Stack:** Python 3, NumPy, FAISS, Pydantic, FastAPI, pytest, existing `hcmai` schemas/configuration.

---

## 1. Input contract and decisions that block implementation

Keep the existing public `SearchRequest` shape. Do not write the recurrence until the unresolved mathematical and metadata decisions below are resolved.

### 1.1 Confirmed ordered-event parsing

`SearchRequest` remains unchanged. For `query_type == TRAKE`, the backend splits `request.query` on the literal `|` character and passes the resulting ordered event strings to DANTE.

```python
def parse_trake_query(query: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in query.split("|"))
    if any(not part for part in parts):
        raise InvalidTrakeQueryError("TRAKE query contains an empty event")
    return parts
```

Parsing rules:

- Split only TRAKE queries; KIS, KISC, VKIS, and VQA keep their existing query behavior.
- Use literal `|` as the only delimiter and preserve segment order.
- Trim leading/trailing whitespace around every segment without rewriting its internal text.
- Reject empty segments such as `|event`, `event|`, or `event A||event B` with the existing HTTP validation/error convention.
- Do not support an escape syntax for a literal `|` inside event text unless a later public contract explicitly adds one.
- The minimum/maximum event count must come from a confirmed competition or product contract. After parsing, `validate_trake_event_count` enforces those configured bounds and raises `InvalidTrakeQueryError` on violation.
- Do not use punctuation heuristics or an LLM to split events.
- A query without `|` parses to one event. Whether one event is a valid TRAKE request is decided by the confirmed event-count contract, not by the delimiter parser.

`request.query` is the single public source of truth. The parsed tuple is request-local internal data and must not be added to `SearchRequest` or persisted as a competing input.

**Remaining user decision:** define filter behavior. Recommended first version rejects non-empty `SearchFilters` for TRAKE until same-video and temporal filter semantics are specified; it must never silently ignore them.

### 1.2 Exact transition equations

The supplied Algorithm 1 references Eq. (3), Eq. (4), and Eq. (5), but their definitions are absent. A generic temporal penalty is not sufficient to reproduce DANTE.

Before implementation, record in this document or a linked source:

- the exact recurrence, including strict (`t_prev < t`) versus non-strict ordering;
- the definition and units of the temporal distance;
- the sign and normalization of penalty `lambda`;
- the video-level final score and any length normalization;
- the tie-breaking rule.

**User input required:** provide the paper/equations or approve a precisely written recurrence. Until then, tests can define interfaces but must not encode a guessed formula.

## 2. Planned file map

| Action | File | Responsibility |
|---|---|---|
| Create | `notebooks/dante_exact_baseline.ipynb` | Prove the exact recurrence against a brute-force reference and select `lambda` on labeled validation data. |
| Create | `src/hcmai/retriever/trake/__init__.py` | Export the TRAKE implementation only. |
| Create | `src/hcmai/retriever/trake/dante.py` | Pipe-delimited query parser, pure DP aligner, and exact event-to-frame retrieval adapter; keep at or below 200 lines. |
| Create | `tests/test_dante.py` | Tiny fake-data smoke tests; keep at or below 100 lines and never load corpus/checkpoints. |
| Modify | `src/hcmai/common/schemas/search.py` | Keep `SearchRequest` unchanged; add only the nested TRAKE response result using existing authoritative contracts. |
| Modify | `src/hcmai/common/schemas/__init__.py` | Export the new result contract. |
| Modify | `src/hcmai/common/schemas/README.md` | Document TRAKE request/response invariants and unresolved 2026 assumptions. |
| Modify | `src/hcmai/data/loader.py` | Add public, read-only canonical per-video iteration; do not expose private dictionaries. |
| Modify | `src/hcmai/retriever/dense/index.py` | Add exact unsorted all-frame scoring so DANTE avoids an unnecessary full ranking. |
| Modify | `src/hcmai/common/config.py` | Add validated DANTE configuration under the existing `SearchConfig`. |
| Modify | `configs/baseline.yaml` | Store the benchmark-selected `lambda`; do not add runtime profiles. |
| Modify | `src/hcmai/orchestration/search.py` | Add a bounded `search_trake` path that converts alignments to the API response. |
| Modify | `src/hcmai/bootstrap/engine.py` | Construct DANTE once from the already loaded encoder, exact dense index, and `FrameStore`. |
| Modify | `src/hcmai/routers/search.py` | Route `TaskType.TRAKE` to its handler and report accurate capability state. |
| Modify | `src/hcmai/routers/system.py` | Report readiness per task instead of reusing ordinary retriever readiness for TRAKE. |
| Modify | `src/hcmai/app.py` | Wire the TRAKE closure without changing KIS/VKIS behavior. |
| Create | `src/hcmai/retriever/evaluation/trake.py` | Evaluate official Mean Top-k R-Score and TRAKE-specific metrics. This is a second component and needs explicit approval under the research limits. |
| Modify | `src/hcmai/retriever/evaluation/__init__.py` | Export the evaluator after it is approved. |
| Modify | `src/hcmai/README.md` | Describe the implemented TRAKE path and known contract status. |

Do not restore, overwrite, or delete currently removed/modified test files while executing this plan. Stage only the files named by the active task.

## 3. Public and internal contracts

### Public result

Extend the existing search contract instead of creating a parallel API schema:

```python
class TrakeSearchResult(ContractModel):
    rank: int = Field(ge=1, le=100)
    video_id: NonEmptyString
    frame_ids: list[NonEmptyString]
    frame_indices: list[int]
    score: float



class SearchResponse(ContractModel):
    # existing fields remain
    results: list[SearchResult] = Field(default_factory=list)
    trake_results: list[TrakeSearchResult] = Field(default_factory=list)
```

Required validators:

- For TRAKE, `results` is empty and `trake_results` contains at most `top_k` rows.
- For non-TRAKE tasks, `trake_results` is empty.
- Each row internally validates equal non-zero lengths for frame IDs and frame indices. `search_trake` validates that this length equals `len(parse_trake_query(request.query))`.
- Every frame belongs to the same returned video.
- Frame order is chronological according to `FrameStore`, not numeric assumptions about filenames/FPS.
- `total_results` equals the active result collection length.

If the API must expose the official video name separately from internal `video_id`, add an optional `video_name` field only after locating the authoritative mapping. Do not assume they are identical.

### Internal values

```python
@dataclass(frozen=True, slots=True)
class DanteAlignment:
    video_id: str
    frame_ids: tuple[str, ...]
    score: float

```

Keep `frame_id` throughout retrieval and DP. Convert to `frame_idx` only through `FrameStore` when producing `TrakeSearchResult` or submission output.

`event_scores` is intentionally absent: the confirmed competition contract does not require it, and the supplied equations do not define whether it means raw similarity, transition-adjusted similarity, or a DP contribution.

## 4. Classes and functions

### `src/hcmai/retriever/trake/dante.py`

```python
class InvalidTrakeQueryError(ValueError):
    """Raised when a pipe-delimited TRAKE query cannot be parsed."""

    ...
```


```python
class DanteAligner:
    def __init__(self, penalty_lambda: float) -> None: ...

    def align_video(
        self,
        similarities: np.ndarray,       # [event_count, video_frame_count]
        temporal_positions: np.ndarray, # canonical positions or confirmed Eq. units
    ) -> tuple[float, np.ndarray]: ...  # score and local frame positions


class DanteRetriever:
    def __init__(
        self,
        encoder: TextEncoder,
        index: DenseIndex,
        frame_store: FrameStore,
        aligner: DanteAligner,
    ) -> None: ...

    def search(
        self,
        events: Sequence[str],
        top_k: int,
    ) -> list[DanteAlignment]: ...
```

Pure helpers, each below 40 lines:

```python
def parse_trake_query(query: str) -> tuple[str, ...]: ...
def validate_trake_event_count(
    events: Sequence[str],
    minimum_events: int,
    maximum_events: int,
) -> None: ...
def _validate_inputs(similarities, temporal_positions) -> None: ...
def _advance_dp(previous_row, similarities_row, positions, penalty) -> tuple[np.ndarray, np.ndarray]: ...
def _backtrack(backpointers, last_position) -> np.ndarray: ...
def _rank_alignments(alignments, top_k) -> list[DanteAlignment]: ...
```

Implementation invariants:

- Encode all ordered events in one batch.
- Query the existing exact dense index for all corpus frames. Do not switch to ANN.
- Scatter each score by the index's authoritative embedding-to-`frame_id` mapping.
- Run an independent DP for every video; reset all state at video boundaries.
- Reject videos with fewer eligible frames than required by strict ordering.
- Use deterministic ties: the exact rule must be specified in Section 1.2.
- Return at most 100 ranked videos, one alignment row per video.
- Never combine frames from different videos.

### `src/hcmai/data/loader.py`

Add only read-only public accessors:

```python
def video_ids(self) -> tuple[str, ...]: ...
def get_video_frames(self, video_id: str) -> tuple[FrameRecord, ...]: ...
```

`get_video_frames` must reuse the store's already sorted canonical records. It must not sort by filename, infer a frame index, or expose `_records_by_video` for mutation.

### `src/hcmai/common/config.py`

```python
class DanteConfig(BaseModel):
    enabled: bool = False
    penalty_lambda: float | None = Field(default=None, ge=0.0)
    minimum_events: int | None = Field(default=None, ge=1)
    maximum_events: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_complete_enabled_config(self) -> DanteConfig:
        """Require lambda and ordered count bounds; enforce min <= max."""
        ...


class SearchConfig(BaseModel):
    # existing fields remain
    dante: DanteConfig = Field(default_factory=DanteConfig)
```

The production `penalty_lambda` is populated only after Task 2 records a validation result. `enabled` remains false until contract, smoke test, and benchmark gates pass.

### `src/hcmai/retriever/dense/index.py`

Add an exact, unsorted scoring API instead of calling `search(..., top_k=ntotal)`, which sorts/selects every corpus score even though DANTE reconstructs its own path:

```python
def score_all(self, query_vectors: np.ndarray) -> np.ndarray:
    """Return exact inner-product scores shaped [query_count, vector_count]."""
    ...
```

It must reuse vectors from the loaded `IndexFlatIP`, return `float32`, and preserve `embedding_index` order. Reconstruct and multiply corpus vectors in bounded blocks, filling only the required final `event_count × vector_count` score matrix; never allocate a second full `vector_count × embedding_dim` corpus copy. Select the block size from a peak-memory/latency benchmark and record both metrics. Use only a public FAISS reconstruction API supported by the pinned version. If that API is unavailable, stop and benchmark the correctness-preserving `search(top_k=vector_count)` fallback; do not reach into undocumented SWIG pointers.

### `src/hcmai/orchestration/search.py`

Extend `SearchEngine` without adding a factory/base-class hierarchy:

```python
def __init__(
    self,
    frame_store: Any,
    retriever: Any,
    reranker: Any | None = None,
    config: Mapping[str, Any] | None = None,
    evidence_stores: Mapping[RetrievalSource, Any] | None = None,
    dante_retriever: DanteRetriever | None = None,
) -> None: ...

def search_trake(self, request: SearchRequest) -> SearchResponse: ...
def _materialize_trake_result(self, alignment, rank) -> TrakeSearchResult: ...
```

`dante_retriever=None` is the explicit disabled/unavailable state. Bootstrap passes a constructed instance only after metadata, exact index, encoder, and enabled configuration are all valid.

`search_trake` calls `parse_trake_query(request.query)`, validates the tuple with `validate_trake_event_count` and the confirmed configured bounds, then passes it to `DanteRetriever.search`. It resolves every `frame_id` through `FrameStore` and returns the authoritative response. It does not mutate `SearchRequest` or call the conversational resolver. Both malformed delimiter syntax and invalid event counts raise `InvalidTrakeQueryError`; the HTTP boundary maps that exception to status 422 with a stable, concise message.

### `src/hcmai/retriever/evaluation/trake.py`

After explicit approval to add the evaluation component:

```python
def trake_row_score(prediction, ground_truth) -> float: ...
def mean_top_k_r_score(rows, ground_truth, cutoffs=(1, 5, 20, 50, 100)) -> float: ...

class TrakeBenchmark:
    def run(self, queries, output_dir: Path) -> TrakeBenchmarkReport: ...
```

The evaluator records `metrics.json`, predictions, failures, config/checkpoint provenance, video accuracy, per-event interval accuracy, full-sequence accuracy, Recall@1/5, MRR, and warm P50/P95 latency. Ground-truth interval parsing must follow an inspected artifact, not a guessed schema.

## 5. Execution tasks

The complete plan exceeds the repository's default research limit of one implementation file plus one smoke-test file and 300 changed lines. Execute it in approval-gated phases:

1. notebook proof only;
2. exact core phase—`dense/index.py`, `trake/__init__.py`, `dante.py`, and `tests/test_dante.py`—only after explicit user approval for exceeding the default implementation + smoke-test file limit;
3. API/config/bootstrap integration only after explicit user approval;
4. evaluator only after separate explicit approval.

Before running any command, perform a non-mutating readiness check for `aic/bin/python`, pytest, pyright, and the current test directory. This worktree currently lacks the mandated `aic/` environment and has user-deleted tests; do not recreate unrelated deleted tests. If the environment is absent, stop with setup instructions rather than claiming verification.

### Task 0: Freeze the missing contract and mathematics

- [ ] Record the confirmed pipe-delimited query contract and examples such as `event A | event B | event C`.
- [ ] Obtain Eq. (3), Eq. (4), Eq. (5), distance units, normalization, and tie behavior.
- [ ] Record whether official video names differ from `video_id`.
- [ ] Record the confirmed 2026 event-count and scoring differences, if any; otherwise label the 2025 rules as prior evidence.
- [ ] Keep `request.query` as the only public source of truth; parsed events remain request-local internal data.
- [ ] Decide whether TRAKE rejects or implements each existing filter field; never ignore filters silently.
- [ ] Obtain explicit approval for each implementation phase that exceeds the research limits.
- [ ] Stop if any of these choices materially changes schema or DP semantics.

Expected result: a hand-computable example with at least two events, two videos, expected winning video, expected path, and expected exact score.

### Task 1: Write failing contract and canonical mapping tests

- [ ] Create `tests/test_dante.py` with tiny `FrameRecord` fixtures and no real models.
- [ ] Assert `"events" not in SearchRequest.model_fields` so the request contract cannot regress.
- [ ] Assert `parse_trake_query("event A | event B") == ("event A", "event B")`.
- [ ] Assert a query without `|` parses to a one-item tuple; enforce minimum count separately after the official/product count is confirmed.
- [ ] Assert whitespace around segments is trimmed while internal text is preserved.
- [ ] Assert `|event`, `event|`, and `event A||event B` raise `InvalidTrakeQueryError`.
- [ ] Assert `validate_trake_event_count` accepts both configured boundaries and raises `InvalidTrakeQueryError` below/above them.
- [ ] Assert non-TRAKE search paths do not call the pipe parser, even if their text contains `|`.
- [ ] Assert `FrameStore.get_video_frames` returns canonical order and immutable output.
- [ ] Assert one TRAKE result contains one video and exactly `N` mapped frame indices.

Run:

```bash
PYTHONPATH=src aic/bin/pytest tests/test_dante.py -q
```

Expected before implementation: FAIL because the contracts/accessors do not exist.

### Task 2: Prove the exact DP before extraction

- [ ] Create `notebooks/dante_exact_baseline.ipynb`.
- [ ] Implement a slow exhaustive path enumerator for tiny matrices.
- [ ] Implement the exact running-maximum recurrence copied from the approved equations.
- [ ] Compare score and path against exhaustive enumeration on deterministic cases and randomized tiny matrices.
- [ ] Benchmark asymptotic/runtime behavior against the direct recurrence.
- [ ] Sweep `lambda` only on labeled validation data and report official Mean Top-k R-Score plus TRAKE metrics and P50/P95 latency.
- [ ] Do not choose a production value if labeled validation data is unavailable.

Acceptance: zero mismatch against exhaustive enumeration; a recorded validation result selects `lambda` without lowering accuracy relative to the exact unpenalized/reference configuration.

### Task 3: Add the authoritative TRAKE response contract

- [ ] Keep `SearchRequest` unchanged and add only `TrakeSearchResult` plus response validators in `src/hcmai/common/schemas/search.py`.
- [ ] Export only the public result type from `src/hcmai/common/schemas/__init__.py`.
- [ ] Document invariants in `src/hcmai/common/schemas/README.md`.
- [ ] Run the focused test and schema compile commands.

```bash
aic/bin/python -m py_compile src/hcmai/common/schemas/search.py
PYTHONPATH=src aic/bin/pytest tests/test_dante.py -q
```

Expected: contract tests PASS; DP tests still FAIL/not yet present.

### Task 4: Expose canonical per-video frames

- [ ] Add `video_ids()` and `get_video_frames()` to `FrameStore`.
- [ ] Reuse existing sorted storage; return tuples.
- [ ] Test unknown video behavior explicitly (`KeyError` or empty tuple, matching the approved existing-store convention).

```bash
aic/bin/python -m py_compile src/hcmai/data/loader.py
PYTHONPATH=src aic/bin/pytest tests/test_dante.py -q
```

Expected: mapping tests PASS with exact stored `frame_idx` values.

### Task 5: Add exact scoring and extract the exact DANTE implementation

- [ ] Add `DenseIndex.score_all(query_vectors)` using a supported public FAISS reconstruction API.
- [ ] Reconstruct/score in bounded blocks and verify peak memory excludes a second full corpus-vector copy.
- [ ] Benchmark block size, peak RAM/VRAM, and latency on the target L40/A6000 environment; block size may change performance but never scores.
- [ ] Test score and embedding-order equality against direct NumPy inner products.
- [ ] Confirm it performs no full top-k sorting, approximation, or undocumented pointer access.
- [ ] Validate that the dense mapping covers the intended DANTE frame corpus; fail DANTE initialization on mismatches instead of silently dropping frames.
- [ ] Create `src/hcmai/retriever/trake/__init__.py`.
- [ ] Create `src/hcmai/retriever/trake/dante.py` from the proven notebook recurrence.
- [ ] Add tests for brute-force equivalence, strict order, same-video isolation, too-short videos, deterministic ties, and top-k ranking.
- [ ] Use fake encoder/index objects in tests.
- [ ] Confirm the implementation file is no more than 200 lines and each function no more than 40 lines.

```bash
aic/bin/python -m py_compile src/hcmai/retriever/dense/index.py
aic/bin/python -m py_compile src/hcmai/retriever/trake/dante.py
PYTHONPATH=src aic/bin/pytest tests/test_dante.py -q
pyright src/hcmai/retriever/trake/dante.py
```

Expected: all focused tests PASS; no model/corpus/network access.

### Task 6: Configure and construct DANTE once

- [ ] Add `DanteConfig` to `src/hcmai/common/config.py`, including nullable disabled defaults and confirmed minimum/maximum event bounds required when enabled.
- [ ] Put the measured `lambda` and `enabled: false` into `configs/baseline.yaml`.
- [ ] Modify `src/hcmai/bootstrap/engine.py` to reuse the loaded encoder, exact dense index, and `FrameStore`.
- [ ] Pass the constructed object through the explicit `SearchEngine(dante_retriever=...)` dependency; use `None` when disabled/unavailable.
- [ ] Do not build/load an index per request.
- [ ] Do not use an RRF aggregate as DANTE's event-frame similarity matrix unless an experiment explicitly defines and validates that score contract.

```bash
aic/bin/python -m py_compile src/hcmai/common/config.py
aic/bin/python -m py_compile src/hcmai/bootstrap/engine.py
PYTHONPATH=src aic/bin/pytest tests/test_dante.py -q
```

Expected: bootstrap with a fake/tiny index constructs one reusable DANTE instance.

### Task 7: Wire orchestration and the HTTP route

- [ ] Add `SearchEngine.search_trake`: parse `request.query`, validate the configured event-count bounds, then perform canonical response materialization.
- [ ] Map `InvalidTrakeQueryError` to HTTP 422 at the router boundary; do not return a 500 for malformed pipe syntax.
- [ ] Add a `trake_search` callable to `StandaloneSearchDispatcher` and map `TaskType.TRAKE` to it.
- [ ] Wire the callable in `create_app`; preserve existing KIS/VKIS paths and leave VQA unchanged.
- [ ] Return the existing unavailable error when DANTE is disabled or not initialized.
- [ ] Update `src/hcmai/routers/system.py` and dispatcher capability APIs to receive per-task readiness; a non-null closure alone is not proof that DANTE loaded.
- [ ] Keep the raw pipe-delimited `request.query` in `SearchEngine._request_id` so event order is represented; add relevant filters deterministically if TRAKE filters are supported.
- [ ] Measure/populate `query_encoding`, `candidate_retrieval`, `temporal_refinement`, `materialization`, and `total` latency fields for DANTE.
- [ ] Add TestClient assertions that malformed pipe syntax and out-of-range event counts return HTTP 422 with stable detail.
- [ ] Add a TestClient assertion that a non-TRAKE query containing `|` retains its existing behavior and is not parsed.
- [ ] Keep these smoke cases in `tests/test_dante.py` if it remains below the 100-line limit; otherwise request approval for a separate API test file.

```bash
aic/bin/python -m py_compile src/hcmai/orchestration/search.py
aic/bin/python -m py_compile src/hcmai/routers/search.py
aic/bin/python -m py_compile src/hcmai/routers/system.py
aic/bin/python -m py_compile src/hcmai/app.py
PYTHONPATH=src aic/bin/pytest tests/test_dante.py -q
```

Expected: a TRAKE request returns ranked same-video paths; KIS/VKIS behavior remains unchanged.

### Task 8: Add official evaluation and enable only after gates pass

- [ ] Ask for explicit approval to add the second research component `evaluation/trake.py`.
- [ ] Inspect the actual label artifact and add/extend an authoritative evaluation schema.
- [ ] Implement 2025-confirmed row score and Mean Top-k R-Score at `{1,5,20,50,100}`; isolate any confirmed 2026 override.
- [ ] Record the required metrics/artifacts under `runs/` without committing run outputs.
- [ ] Compare exact DANTE with the current retrieval baseline on the same queries and hardware.
- [ ] Enable DANTE only if correctness passes and the chosen configuration does not reduce the accepted validation accuracy metric.

Expected artifacts (uncommitted):

```text
runs/<run-id>/metrics.json
runs/<run-id>/predictions.*
runs/<run-id>/failures.*
runs/<run-id>/config.yaml
```

### Task 9: Documentation and final verification

- [ ] Update `src/hcmai/README.md` with request/response examples and the distinction between confirmed 2025 rules and unresolved 2026 behavior.
- [ ] Run focused tests, compile checks, and type checks for every touched Python file.
- [ ] Inspect `git diff --check` and `git status --short`; do not modify unrelated changes.
- [ ] Verify no dataset, embeddings, index, weights, credentials, or run outputs are staged.

```bash
PYTHONPATH=src aic/bin/pytest tests/test_dante.py -q
git diff --check
git status --short
```

Expected: focused test suite PASS, no whitespace errors, only explicitly planned files changed.

## 6. Commit gate

Do not create commits unless the user explicitly authorizes commits. If authorized, commit task-sized changes and stage explicit paths only; never use `git add .` in the dirty worktree. Every AI-authored commit includes:

```text
Co-Authored-By: GPT-5 <noreply@openai.com>
```

Suggested commit boundaries:

1. `test: specify exact DANTE TRAKE behavior`
2. `feat: add exact DANTE temporal aligner`
3. `feat: expose DANTE through TRAKE search`
4. `eval: add TRAKE official-score benchmark`
5. `docs: document DANTE contracts and evidence`

## 7. Explicit non-goals

- No ANN/HNSW/IVF, frame pruning, candidate gating, beam search, or sparse score truncation in the first implementation.
- No Milvus migration; the repository currently has an exact FAISS foundation.
- No automatic event segmentation or LLM on the TRAKE critical path.
- No inferred `frame_idx`, timestamp-to-frame conversion, or filename parsing.
- No KISC routing, cross-video paths, or independent per-event final answers.
- No production enablement without exact-reference tests and recorded accuracy/latency metrics.

