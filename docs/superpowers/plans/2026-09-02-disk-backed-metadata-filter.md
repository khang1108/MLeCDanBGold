# Disk-Backed Metadata Filter Implementation Plan

> **Status: RETIRED on 2026-09-02.** The implemented exact Filter backend was
> intentionally removed. This plan is historical and must not be executed;
> only the stable endpoint placeholder remains.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the exact, deterministic `/api/v1/filter` feature over a read-only SQLite catalog, integrate its complete page response into Filter Workspace, and keep filter memory bounded while FAISS remains loaded.

**Architecture:** An offline builder joins canonical frame identity with title, Caption, OCR, exact Object counts, and timestamp-containing ASR, then atomically publishes `artifacts/filter/filter_catalog.sqlite`. A standalone runtime `FilterService` uses a four-connection read-only SQLite pool for parameterized AND filtering, stable ordering, counting, and pagination; FastAPI only validates and transports the result. The frontend consumes complete result rows directly, removes its frame-detail fan-out, and freezes the selected Auto/12/24/48 page size for each applied filter session.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI, standard-library `sqlite3`, PyArrow-backed existing artifact stores, pytest, React, Jest/Testing Library

**Spec:** `docs/superpowers/specs/2026-09-01-disk-backed-metadata-filter-design.md`

**Execution Status:** Implemented on `main` through task-scoped TDD commits;
real-corpus and FAISS-resident acceptance benchmark completed 2026-09-02.

## Global Constraints

- Preserve canonical `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms`; never derive identity or submission coordinates from SQLite row order.
- Populated Title, ASR, Caption, OCR, Object, folder, and video predicates combine with AND.
- Text matching is normalized substring matching; object matching is exact count equality for every requested label.
- ASR attached to a frame comes only from segments satisfying `start_ms <= timestamp_ms < end_ms`.
- Keep the frontend-compatible field name `frames_per_pages`; require `page_id >= 1` and `1 <= frames_per_pages <= 48`.
- Order results by `video_id`, `timestamp_ms`, `frame_idx`, then `frame_id`; an out-of-range page returns `results=[]` with the true totals.
- If an entire modality is unavailable, omit that predicate and log it; if the modality is available but a frame lacks evidence, that frame does not match.
- Serving never creates or repairs the catalog. A missing/unopenable catalog makes only Filter return 503; corrupt runtime state returns JSON 500.
- Open SQLite read-only with at most four connections, about 8 MiB cache per connection, `query_only=ON`, `temp_store=FILE`, and `mmap_size=0`.
- Do not add FAISS, embeddings, FTS, fuzzy matching, reranking, LLM, VLM, auth, or rate limiting to Filter V1.
- Do not expose database paths, source paths, image paths, or lineage internals in filter results.
- Every non-trivial Python module and every public class/function receives a useful ownership/invariant docstring.
- Use small hand-checkable fixtures and TDD. Run focused tests before broad suites. Do not claim the P95/RAM targets until measured on the real corpus with FAISS loaded.

## File Map

### Shared contract and runtime

- Create `src/hcmai/filtering/__init__.py`: public Filter package exports only.
- Create `src/hcmai/filtering/normalization.py`: the one text/label normalization function shared by offline materialization and online requests.
- Create `src/hcmai/filtering/schema.py`: catalog schema version, DDL, required-table validation, and deterministic index definitions.
- Create `src/hcmai/filtering/catalog.py`: validated catalog metadata plus bounded read-only SQLite connection pool.
- Create `src/hcmai/filtering/service.py`: predicate construction, count/page queries, result mapping, and Filter-specific exceptions.
- Create `src/hcmai/filtering/setup.py`: load Filter configuration/catalog independently from Search startup.
- Create `src/hcmai/api/contracts/filter.py`: strict HTTP request and response models.
- Create `src/hcmai/api/routers/filter.py`: thin `/api/v1/filter` transport and error mapping.
- Modify `src/hcmai/api/contracts/__init__.py` and `src/hcmai/api/routers/__init__.py`: export new public boundary types/factory.
- Modify `src/hcmai/common/config.py`: add bounded Filter runtime settings.
- Modify `configs/baseline.yaml` and `.env.example`: document the default catalog and its local override.
- Modify `src/hcmai/app.py`: independently initialize/close Filter and register its router.
- Modify `src/hcmai/api/routers/system.py`: report Filter capability/catalog facts without changing search readiness.

### Offline catalog production

- Create `offline/filtering/__init__.py`: offline package boundary.
- Create `offline/filtering/builder.py`: stream source records, validate joins, project frame-local ASR, insert catalog rows, validate, and atomically publish.
- Create `scripts/build_filter_catalog.py`: explicit CLI over the builder; never imported by serving.

### Frontend

- Modify `frontend/src/api/filter.js`: enforce the complete response contract including `frames_per_pages` and full display metadata.
- Modify `frontend/src/features/filter/components/FilterWorkspace.jsx`: consume complete rows, remove N+1 detail state/effects, and own page-size mode/session state.
- Create `frontend/src/features/filter/components/FilterPageSize.jsx`: accessible Auto/12/24/48 selector.
- Modify the existing Filter stylesheet that owns `.filter-workspace`: style the selector without introducing another global stylesheet.
- Keep `frontend/src/api/frames.js` unchanged because non-Filter consumers still use frame details.

### Tests and measurement

- Create `tests/filtering/test_normalization.py`, `tests/filtering/test_catalog.py`, and `tests/filtering/test_service.py`.
- Create `tests/offline/filtering/test_builder.py` and `tests/scripts/test_build_filter_catalog.py`.
- Create `tests/api/test_filter_contracts.py`, `tests/api/test_filter_routes.py`, and extend `tests/api/test_system_routes.py`.
- Modify `frontend/src/api/filter.test.js`, `frontend/src/features/filter/components/FilterWorkspace.test.jsx`, and `frontend/src/features/filter/filterPagination.test.js`.
- Create `frontend/src/features/filter/components/FilterPageSize.test.jsx`.
- Create `scripts/benchmark_filter_catalog.py` and `tests/scripts/test_benchmark_filter_catalog.py`.

---

### Task 1: Freeze Filter Contracts and Normalization

**Files:**
- Create: `src/hcmai/filtering/__init__.py`
- Create: `src/hcmai/filtering/normalization.py`
- Create: `src/hcmai/api/contracts/filter.py`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Test: `tests/filtering/test_normalization.py`
- Test: `tests/api/test_filter_contracts.py`

**Interfaces:**
- Produces: `normalize_filter_text(value: str) -> str`.
- Produces: `FilterMetadataFilters`, `FilterRequest`, `FilterResult`, and `FilterResponse`.
- `FilterRequest.metadata_filters` defaults to an empty `FilterMetadataFilters`; scope fields default to `None`; `frames_per_pages=12`; `page_id=1`.
- `FilterResult.objects` is `dict[str, int]`, not a presence-only label list.

- [ ] **Step 1: Write failing normalization tests**

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Cảnh  có ÁO đỏ ", "canh co ao do"),
        ("ĐƯỜNG phố", "duong pho"),
        ("person\n\tcar", "person car"),
    ],
)
def test_normalize_filter_text_matches_frontend_contract(raw, expected):
    assert normalize_filter_text(raw) == expected
```

- [ ] **Step 2: Run normalization tests and verify the module is missing**

Run: `pytest tests/filtering/test_normalization.py -q`

Expected: FAIL during import because `hcmai.filtering.normalization` does not exist.

- [ ] **Step 3: Implement the single shared normalizer**

Implement `normalize_filter_text()` with `unicodedata.normalize("NFD", value)`, explicit `đ/Đ -> d/D`, removal of combining characters, `lower()`, whitespace collapse, and a final trim. Do not reuse embedding-query normalization because it intentionally preserves diacritics and has different semantics.

- [ ] **Step 4: Write failing strict-contract tests**

Cover all of these cases explicitly:

```python
def test_filter_request_defaults_and_keeps_frontend_page_field():
    request = FilterRequest.model_validate({})
    assert request.frames_per_pages == 12
    assert request.page_id == 1
    assert request.metadata_filters.objects == {}

@pytest.mark.parametrize("value", [0, 49])
def test_filter_request_rejects_page_sizes_outside_v1_bound(value):
    with pytest.raises(ValidationError):
        FilterRequest(frames_per_pages=value)

def test_filter_request_normalizes_object_labels_and_rejects_fractional_counts():
    request = FilterRequest(metadata_filters={"objects": {" Người ": 3}})
    assert request.metadata_filters.objects == {"nguoi": 3}
    with pytest.raises(ValidationError):
        FilterRequest(metadata_filters={"objects": {"person": 1.5}})
```

Also test `page_id=0`, unknown fields, blank object labels, negative counts, text longer than 500 characters, and duplicate labels that collide after normalization. Reject normalized duplicates instead of silently changing exact-count intent.

- [ ] **Step 5: Implement the HTTP models**

Use `ConfigDict(extra="forbid")` on every model. Define optional Title/ASR/Caption/OCR request strings with `max_length=500`; normalize nonblank values in a field validator and convert blank strings to `None`. Define the response exactly as:

```python
class FilterResult(BaseModel):
    frame_id: str
    video_id: str
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    folder_id: str
    title: str | None = None
    caption: str | None = None
    ocr: str | None = None
    objects: dict[str, int] = Field(default_factory=dict)
    asr: str | None = None

class FilterResponse(BaseModel):
    page_id: int = Field(ge=1)
    frames_per_pages: int = Field(ge=1, le=48)
    total_results: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    results: list[FilterResult] = Field(default_factory=list)
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/filtering/test_normalization.py tests/api/test_filter_contracts.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the contract slice**

```bash
git add src/hcmai/filtering src/hcmai/api/contracts tests/filtering/test_normalization.py tests/api/test_filter_contracts.py
git commit -m "feat(filter): define exact filter contracts"
```

### Task 2: Define and Open the Read-Only Catalog Safely

**Files:**
- Create: `src/hcmai/filtering/schema.py`
- Create: `src/hcmai/filtering/catalog.py`
- Test: `tests/filtering/test_catalog.py`

**Interfaces:**
- Consumes: `normalize_filter_text()` only in fixture setup; catalog opening never mutates stored data.
- Produces: `CatalogAvailability`, `FilterCatalogInfo`, `FilterCatalog`, `FilterCatalogUnavailableError`, and `FilterCatalogCorruptError`.
- Produces: `FilterCatalog.open(path: Path, *, pool_size: int = 4, cache_kib: int = 8192) -> FilterCatalog`, `connection()` context manager, `info`, and idempotent `close()`.

- [ ] **Step 1: Write a hand-checkable SQLite fixture and failing open tests**

Create a fixture with one metadata row, two frame rows, and object rows. Assert that opening it returns the version/frame count/availability, `PRAGMA query_only` is `1`, `PRAGMA mmap_size` is `0`, and a write through a borrowed connection raises `sqlite3.OperationalError`.

Also assert:

```python
with pytest.raises(FilterCatalogUnavailableError):
    FilterCatalog.open(tmp_path / "missing.sqlite")

with pytest.raises(FilterCatalogCorruptError, match="schema_version"):
    FilterCatalog.open(wrong_schema_path)
```

- [ ] **Step 2: Run the focused catalog tests**

Run: `pytest tests/filtering/test_catalog.py -q`

Expected: FAIL because the schema/catalog modules do not exist.

- [ ] **Step 3: Implement schema V1 and its indexes**

Use a single-row `catalog_metadata` table with `id=1`, `schema_version`, `catalog_version`, `built_at`, `frame_count`, `source_lineage_json`, and boolean availability columns for title/caption/OCR/objects/ASR. Define `frames` with canonical fields plus raw and `_norm` text columns, and `frame_objects(frame_id, label_norm, object_count)` with `(frame_id, label_norm)` primary key and a foreign key to frames.

Create these minimum indexes:

```sql
CREATE INDEX idx_frames_order
ON frames(video_id, timestamp_ms, frame_idx, frame_id);
CREATE INDEX idx_frames_folder_order
ON frames(folder_id, video_id, timestamp_ms, frame_idx, frame_id);
CREATE INDEX idx_frame_objects_match
ON frame_objects(label_norm, object_count, frame_id);
```

Do not add an FTS virtual table or substring index in V1.

- [ ] **Step 4: Implement the bounded read-only pool**

Open each connection using SQLite URI mode `mode=ro`, `check_same_thread=False`, and `timeout=30`. Configure each connection with `query_only=ON`, `temp_store=FILE`, `mmap_size=0`, and negative `cache_size=-8192` by default. Put exactly `pool_size` connections in `queue.Queue(maxsize=pool_size)`; callers block on `get()` and always return the connection in `finally`.

Validate the schema version, exact required columns, one metadata row, nonnegative frame count, and actual `COUNT(*) == frame_count` before exposing the catalog.

- [ ] **Step 5: Add pool bound and close tests**

Borrow all four connections, start a fifth borrower in a thread, assert it waits, return one connection, and assert it proceeds. Assert `close()` closes all idle connections and a second `close()` is harmless.

- [ ] **Step 6: Run catalog tests**

Run: `pytest tests/filtering/test_catalog.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the catalog runtime boundary**

```bash
git add src/hcmai/filtering/schema.py src/hcmai/filtering/catalog.py tests/filtering/test_catalog.py
git commit -m "feat(filter): add bounded readonly catalog"
```

### Task 3: Build and Atomically Publish the Offline Catalog

**Files:**
- Create: `offline/filtering/__init__.py`
- Create: `offline/filtering/builder.py`
- Create: `scripts/build_filter_catalog.py`
- Test: `tests/offline/filtering/test_builder.py`
- Test: `tests/scripts/test_build_filter_catalog.py`

**Interfaces:**
- Consumes: existing `FrameStore`, `VideoMetadataStore`, `CaptionStore`, `OCRStore`, `ObjectCountsStore`, and `TranscriptStore` readers plus `CATALOG_SCHEMA_VERSION`, schema DDL, and `normalize_filter_text()`.
- Produces: `FilterCatalogBuildConfig`, `FilterCatalogBuildReport`, and `build_filter_catalog(config) -> FilterCatalogBuildReport`.
- `FilterCatalogBuildConfig` contains explicit input paths, `output_path`, `catalog_version`, and source-lineage values; optional modality paths use `None` to mean globally unavailable.

- [ ] **Step 1: Write failing builder tests using six canonical frames**

The fixture must make expected rows obvious:

- two videos (`L21_V001`, `L22_V002`) and derived folders (`L21`, `L22`);
- Vietnamese accented Title/Caption/OCR values with expected normalized columns;
- three `person` detections on one frame and one on another;
- ASR segments `[0, 1000)` and `[1000, 2000)` with frames at 999 and 1000 ms;
- one frame with no Caption while Caption is globally available.

Assert canonical IDs and coordinates are unchanged, counts are `3` and `1`, the 999 ms frame gets only the first ASR text, the 1000 ms frame gets only the second, and the metadata availability flags reflect supplied paths.

- [ ] **Step 2: Add rejection and publication tests**

Test that a Caption/Object/OCR record referring to an unknown `frame_id` raises `FilterCatalogBuildError` before publication. Seed an existing output file, force validation failure, and assert the old file bytes remain unchanged. On success, assert no temporary file remains beside the output.

- [ ] **Step 3: Run builder tests and verify failure**

Run: `pytest tests/offline/filtering/test_builder.py -q`

Expected: FAIL because the offline builder does not exist.

- [ ] **Step 4: Implement streaming joins and ASR projection**

Iterate canonical frames in stable order and insert in batches (default 2,000). Build only bounded per-video ASR state and per-frame evidence lookups supplied by existing stores; do not concatenate the full corpus into a second pandas DataFrame. For each frame:

```python
folder_id = frame.video_id.split("_", 1)[0]
asr_segments = transcript_store.get_at_time(frame.video_id, frame.timestamp_ms)
asr_text = " ".join(segment.text for segment in asr_segments) or None
```

Validate the organizer `video_id` shape before deriving `folder_id`. Insert raw display values and normalized values separately. Insert every object label/count pair; never reduce counts to presence.

- [ ] **Step 5: Implement same-directory atomic publication**

Create the output parent, write to a uniquely named temporary file in that same directory, commit, run `PRAGMA integrity_check`, validate schema and frame count via the runtime validator, close the temporary catalog, then call `os.replace(temp_path, output_path)`. On any exception, unlink only the known temporary path and leave the previous output untouched.

- [ ] **Step 6: Implement the explicit CLI**

Expose arguments for every source path, `--output`, and `--catalog-version`; default output to `artifacts/filter/filter_catalog.sqlite`. Print a JSON report containing catalog version, frame count, availability, output size, and build seconds. Do not start models or import `SearchService`.

- [ ] **Step 7: Run builder and CLI tests**

Run: `pytest tests/offline/filtering/test_builder.py tests/scripts/test_build_filter_catalog.py -q`

Expected: PASS.

- [ ] **Step 8: Commit offline production**

```bash
git add offline/filtering scripts/build_filter_catalog.py tests/offline/filtering tests/scripts/test_build_filter_catalog.py
git commit -m "feat(filter): build atomic sqlite catalog"
```

### Task 4: Implement Exact Filtering, Counting, and Pagination

**Files:**
- Create: `src/hcmai/filtering/service.py`
- Test: `tests/filtering/test_service.py`

**Interfaces:**
- Consumes: `FilterCatalog`, `FilterRequest`, `FilterResponse`, and normalized request fields.
- Produces: `FilterService(catalog: FilterCatalog)`, `filter(request: FilterRequest) -> FilterResponse`, `health() -> dict[str, object]`, `close()`, and `FilterServiceUnavailableError`.

- [ ] **Step 1: Write failing semantic tests against a tiny real catalog**

Cover normalized substring matching, AND across four text fields, exact object count, multiple-object AND, exact video scope, folder scope, empty-filter all-catalog behavior, available-but-missing evidence, and globally unavailable predicate omission with `caplog`.

Use a decisive exact-count assertion:

```python
response = service.filter(FilterRequest(
    metadata_filters={"objects": {"person": 3, "car": 1}}
))
assert [row.frame_id for row in response.results] == ["L21_V001_0003"]
```

- [ ] **Step 2: Write failing ordering/pagination tests**

Insert rows out of order and assert returned identities follow `(video_id, timestamp_ms, frame_idx, frame_id)`. With five matches and page size two, assert page 2 contains rows 3–4, `total_results=5`, `total_pages=3`; page 4 returns `[]` with totals unchanged. With zero matches, assert `total_pages=0`.

- [ ] **Step 3: Run service tests and verify failure**

Run: `pytest tests/filtering/test_service.py -q`

Expected: FAIL because `FilterService` does not exist.

- [ ] **Step 4: Implement one parameterized predicate builder**

Return `(where_sql, parameters)` and reuse it for both count and page SQL. Use `instr(frames.<field>_norm, ?) > 0` for each active text predicate, equality for `folder_id`/`video_id`, and one correlated `EXISTS` per object:

```sql
EXISTS (
  SELECT 1 FROM frame_objects fo
  WHERE fo.frame_id = frames.frame_id
    AND fo.label_norm = ?
    AND fo.object_count = ?
)
```

Never interpolate user values into SQL. Sort requested object labels before building SQL so diagnostics and query plans are deterministic.

- [ ] **Step 5: Implement result mapping without path leakage**

Fetch object counts for only the current page in one additional query using its frame IDs, group them into dictionaries, and construct strict `FilterResult` values. Do not `SELECT *`; name only public columns. Compute offset as `(page_id - 1) * frames_per_pages`.

- [ ] **Step 6: Add ten-caller concurrency coverage**

Run ten `service.filter()` calls through `ThreadPoolExecutor(max_workers=10)` against a four-connection catalog. Assert every response is identical, no SQLite thread error occurs, and the catalog never creates a fifth connection.

- [ ] **Step 7: Run service and catalog suites**

Run: `pytest tests/filtering/test_service.py tests/filtering/test_catalog.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the query service**

```bash
git add src/hcmai/filtering/service.py tests/filtering/test_service.py
git commit -m "feat(filter): query exact paginated metadata"
```

### Task 5: Wire Filter Independently into FastAPI and Health

**Files:**
- Create: `src/hcmai/api/routers/filter.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/common/config.py`
- Create: `src/hcmai/filtering/setup.py`
- Modify: `src/hcmai/app.py`
- Modify: `src/hcmai/api/routers/system.py`
- Modify: `configs/baseline.yaml`
- Modify: `.env.example`
- Test: `tests/api/test_filter_routes.py`
- Test: `tests/api/test_system_routes.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `FilterConfig(catalog_path, connection_pool_size=4, sqlite_cache_kib=8192)` with bounds `1..4` and `1024..8192` respectively.
- Produces: `load_filter_service(messages: list[str]) -> FilterService | None`.
- Produces: `create_filter_router(service_container) -> APIRouter`.
- Extends: `create_app(search_service=None, filter_service=None)` for isolated API tests.

- [ ] **Step 1: Write failing route tests**

Use a fake Filter service and assert `POST /api/v1/filter` passes the validated `FilterRequest` and serializes `FilterResponse`. Assert invalid pagination returns 422 before calling the service. Assert a missing service and `FilterServiceUnavailableError` return 503, while an unexpected error is handled by the application JSON 500 middleware.

- [ ] **Step 2: Write failing independent-health tests**

Assert all four combinations of Search ready/unready and Filter ready/unready. In particular, a missing Filter catalog must leave the existing `ready` and `capabilities.search/kis/trake` values unchanged while adding:

```json
"filter_catalog": {"ready": false, "catalog_version": null, "frame_count": 0}
```

and `capabilities.filter=false`.

- [ ] **Step 3: Run API/config tests and verify failure**

Run: `pytest tests/api/test_filter_routes.py tests/api/test_system_routes.py tests/test_config.py -q`

Expected: FAIL because Filter wiring/configuration is absent.

- [ ] **Step 4: Add bounded runtime configuration**

Add `FilterConfig` to `AppConfig` with repository-relative default `artifacts/filter/filter_catalog.sqlite` and add the matching `filter:` section to `configs/baseline.yaml`. `load_filter_service()` loads the root `.env` with the same precedence as Search startup and resolves the explicit `HCMAI_FILTER_CATALOG_PATH` override through `resolve_repository_path`; the service itself never reads environment variables. Document that override in `.env.example`.

- [ ] **Step 5: Add independent lifespan initialization**

During startup, attempt `FilterCatalog.open()` and wrap it in `FilterService`. On missing/unopenable catalog, store `filter_service=None`, append/log one Filter-specific startup message, and continue Search startup. On shutdown, close Search and Filter independently exactly once.

- [ ] **Step 6: Add the thin synchronous router**

Use a normal `def filter_frames(request: FilterRequest) -> FilterResponse` so FastAPI runs blocking SQLite work in its threadpool. The router may map only Filter availability to 503; allow invariant/corruption exceptions to reach the existing JSON 500 middleware and server logger.

- [ ] **Step 7: Merge Filter facts into health**

Build on the existing Search health dictionary. Add only `capabilities.filter` and `filter_catalog`; never use Filter state to recalculate top-level Search readiness.

- [ ] **Step 8: Run API, config, and existing search-route tests**

Run: `pytest tests/api/test_filter_routes.py tests/api/test_system_routes.py tests/api/test_search_routes.py tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 9: Commit API integration**

```bash
git add src/hcmai/api src/hcmai/common/config.py src/hcmai/app.py tests/api tests/test_config.py
git commit -m "feat(filter): expose independent filter api"
```

### Task 6: Make the Frontend Consume Complete Filter Rows

**Files:**
- Modify: `frontend/src/api/filter.js`
- Modify: `frontend/src/api/filter.test.js`
- Modify: `frontend/src/features/filter/components/FilterWorkspace.jsx`
- Modify: `frontend/src/features/filter/components/FilterWorkspace.test.jsx`

**Interfaces:**
- Consumes: backend `FilterResponse` with full flat metadata and echoed `frames_per_pages`.
- Produces: Filter cards/modal/submission behavior with zero calls to `getFrameDetail`.

- [ ] **Step 1: Extend failing API response tests**

Return a page row containing all canonical fields plus title/caption/OCR/object dictionary/ASR. Assert it is preserved. Reject responses that omit `total_results`, `frames_per_pages`, or any canonical identity field; reject an echoed page size different from the request. Assert unexpected path-like fields (`image_path`, `database_path`) are not required or consumed.

- [ ] **Step 2: Extend failing workspace tests for no N+1**

Mock only `filterFrames`, return 12 complete rows, apply a filter, and assert one Filter request renders 12 cards. Remove the `getFrameDetail` mock and assert opening a card uses its response metadata directly.

- [ ] **Step 3: Run focused frontend tests and observe failure**

Run: `cd frontend && npm test -- --runInBand src/api/filter.test.js src/features/filter/components/FilterWorkspace.test.jsx`

Expected: FAIL because the workspace still imports and fans out `getFrameDetail`, and the API does not validate all echoed totals/page-size fields.

- [ ] **Step 4: Remove detail fan-out from Filter Workspace**

Delete the `getFrameDetail` import, `detailCache` state/ref, detail controller ref, abort helper, `loadFrameDetail`, and loading effect. Remove `scopeResults` as a second client-side filter: the backend page is already scoped and must be rendered directly so counts and pagination cannot diverge. Folder/video edits affect the next submitted request; any video IDs derived from the current page are suggestions only. Render each card with the complete Filter row and invoke `onFrameClick(frame)` directly. Keep `frontend/src/api/frames.js` intact.

- [ ] **Step 5: Tighten response validation**

Require nonnegative `total_results`, nonnegative `total_pages`, positive `frames_per_pages <= 48`, and equality with the requested size. Validate nullable display strings and an object dictionary whose values are nonnegative integers. Keep canonical URL derivation in existing frame components; do not accept filesystem paths from this endpoint.

- [ ] **Step 6: Run focused frontend tests**

Run: `cd frontend && npm test -- --runInBand src/api/filter.test.js src/features/filter/components/FilterWorkspace.test.jsx`

Expected: PASS.

- [ ] **Step 7: Commit complete-page consumption**

```bash
git add frontend/src/api/filter.js frontend/src/api/filter.test.js frontend/src/features/filter/components/FilterWorkspace.jsx frontend/src/features/filter/components/FilterWorkspace.test.jsx
git commit -m "refactor(filter): remove frame detail fanout"
```

### Task 7: Add Auto/12/24/48 Page-Size Sessions

**Files:**
- Create: `frontend/src/features/filter/components/FilterPageSize.jsx`
- Create: `frontend/src/features/filter/components/FilterPageSize.test.jsx`
- Modify: `frontend/src/features/filter/components/FilterWorkspace.jsx`
- Modify: `frontend/src/features/filter/components/FilterWorkspace.test.jsx`
- Modify: `frontend/src/features/filter/filterPagination.js`
- Modify: `frontend/src/features/filter/filterPagination.test.js`
- Modify: existing Filter workspace stylesheet

**Interfaces:**
- Produces: `FILTER_PAGE_SIZE_OPTIONS = ['auto', 12, 24, 48]` and `resolveFramesPerPage(mode, viewport) -> int`.
- `FilterPageSize` receives `value`, `onChange`, and `disabled`.
- A filter session stores the resolved integer; resize changes only the next Auto session, never the currently applied session.

- [ ] **Step 1: Write failing resolver and selector tests**

Assert fixed modes return themselves, Auto delegates to the existing viewport calculation and remains within 1–48, and an invalid mode falls back to Auto/12 deterministically. Render the selector and assert accessible options `Auto`, `12`, `24`, `48` and a change callback.

- [ ] **Step 2: Write failing session-behavior tests**

Assert a new Auto filter uses the current calculated integer; resizing after the response does not refetch or change page size; navigating to page 2 reuses the applied integer; changing from Auto to 24 immediately applies the current filters at page 1 with size 24; a new filter after resize recalculates Auto and starts at page 1.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `cd frontend && npm test -- --runInBand src/features/filter/filterPagination.test.js src/features/filter/components/FilterPageSize.test.jsx src/features/filter/components/FilterWorkspace.test.jsx`

Expected: FAIL because page-size mode/session semantics are absent.

- [ ] **Step 4: Implement page-size mode separately from measured Auto size**

Keep `pageSizeMode` (`'auto' | 12 | 24 | 48`) separate from `autoFramesPerPage`. On apply, resolve one integer and copy it into `appliedFramesPerPage`. Resize observers may update `autoFramesPerPage`, but must not call `requestFilterPage` or mutate `appliedFramesPerPage`.

- [ ] **Step 5: Implement selector reset behavior and styling**

On selector change, set page 1 and, when a result session exists, reissue the applied filter/scope using the newly resolved size. Disable the selector while a request is active. Place it beside result pagination/summary with an accessible label and reuse existing input colors, borders, and focus states.

- [ ] **Step 6: Run all Filter frontend tests**

Run: `cd frontend && npm test -- --runInBand src/api/filter.test.js src/features/filter`

Expected: PASS.

- [ ] **Step 7: Commit page-size controls**

```bash
git add frontend/src/features/filter
git commit -m "feat(filter): add stable page size sessions"
```

### Task 8: Add a Reproducible Filter Benchmark

**Files:**
- Create: `scripts/benchmark_filter_catalog.py`
- Create: `tests/scripts/test_benchmark_filter_catalog.py`
- Modify: `KNOWLEDGE.md` only after real-corpus measurements exist

**Interfaces:**
- Consumes: a published catalog and a JSON query fixture containing named global/folder/video/object/combined cases.
- Produces: JSON with catalog version/size, process RSS baseline and delta, warmup count, concurrency, per-case P50/P95, error count, and run timestamp.

- [ ] **Step 1: Write failing benchmark CLI tests**

Use a tiny catalog and two query cases. Run with concurrency `1` and `4`; assert stable JSON keys, successful sample counts, millisecond percentiles, and nonnegative RSS delta. Assert invalid concurrency and an empty query list fail with a nonzero exit.

- [ ] **Step 2: Run the focused benchmark test**

Run: `pytest tests/scripts/test_benchmark_filter_catalog.py -q`

Expected: FAIL because the benchmark script does not exist.

- [ ] **Step 3: Implement measurement without changing runtime behavior**

Use `time.perf_counter_ns()`, `statistics`, `ThreadPoolExecutor`, and Linux `/proc/self/status` for RSS. Execute each case once as warmup, then the configured samples. Do not add caching to `FilterService` and do not mutate SQLite pragmas beyond the production catalog settings.

- [ ] **Step 4: Run benchmark-script tests**

Run: `pytest tests/scripts/test_benchmark_filter_catalog.py -q`

Expected: PASS.

- [ ] **Step 5: Commit benchmark tooling**

```bash
git add scripts/benchmark_filter_catalog.py tests/scripts/test_benchmark_filter_catalog.py
git commit -m "perf(filter): add catalog benchmark harness"
```

- [ ] **Step 6: Build and measure the real catalog before optimization**

Run the builder against the active approximately 470,000-frame artifacts, then benchmark cold/warm global, folder, video, object, and combined filters on page 1 and the final page at concurrency 1, 4, and 10. Run once without FAISS and once with the normal backend/FAISS loaded. Save the command, catalog/source versions, output JSON, build time, catalog size, and machine state in the experiment record.

Acceptance: representative P95 `< 2000 ms`, filter-specific RSS growth `< 64 MiB`, zero query errors, and unchanged KIS/TRAKE behavior. If global substring cases miss the target, open a separate measured FTS5-trigram experiment; do not silently add it here.

- [ ] **Step 7: Update research memory with measured status**

Change the existing Filter entry in `KNOWLEDGE.md` from PROPOSED to VERIFIED only if the recorded measurements meet the targets. Otherwise retain PROPOSED and record the failing cases and the separate FTS hypothesis. Never describe unmeasured latency as an improvement.

### Task 9: End-to-End Regression and Release Gate

**Files:**
- Modify only files required by failures caused by this feature; do not fold unrelated cleanup into this task.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: a release-ready Filter slice with evidence that existing Search/KIS/TRAKE and frontend flows still pass.

- [ ] **Step 1: Run Python syntax compilation**

No formatter or linter is configured in the current `pyproject.toml`. Run `python -m compileall -q src/hcmai/filtering src/hcmai/api offline/filtering scripts/build_filter_catalog.py scripts/benchmark_filter_catalog.py` and fix only failures introduced by this feature.

- [ ] **Step 2: Run focused backend suites**

Run:

```bash
pytest tests/filtering tests/offline/filtering tests/scripts/test_build_filter_catalog.py tests/scripts/test_benchmark_filter_catalog.py tests/api/test_filter_contracts.py tests/api/test_filter_routes.py tests/api/test_system_routes.py -q
```

Expected: PASS.

- [ ] **Step 3: Run backend regression suites**

Run: `pytest tests/api tests/orchestration tests/temporal tests/retrieval -q`

Expected: PASS with no KIS/TRAKE contract or canonical-identity regression.

- [ ] **Step 4: Run frontend tests and production build**

Run:

```bash
cd frontend
npm test -- --runInBand
npm run build
```

Expected: all tests PASS and the production build succeeds.

- [ ] **Step 5: Perform local API smoke checks**

With a fixture or published catalog, verify empty filters, one text field, exact object count, combined predicates, folder/video scope, page 1, out-of-range page, and missing-catalog 503. Inspect JSON to confirm complete metadata and absence of filesystem/database paths. Confirm `/health` reports Filter independently and Search endpoints still respond.

- [ ] **Step 6: Review the diff and commit the release gate**

Run `git diff --check` and inspect `git status --short`. If validation required feature-specific fixes, commit them with their tests using a precise message. Do not stage the user's unrelated files and do not resolve any merge conflict without first asking the user and explaining both sides.

## Completion Evidence

Before calling the feature complete, report:

- catalog schema/version, source lineage, frame count, build time, and file size;
- exact files changed and the public request/response behavior;
- focused and regression test commands with pass counts;
- benchmark P50/P95 at concurrency 1/4/10 and RSS delta with FAISS loaded;
- confirmation that missing Filter catalog leaves KIS/TRAKE active;
- confirmation that Filter Workspace makes one page request and zero frame-detail requests;
- known limitations, especially global substring performance and the absence of FTS/auth in V1;
- whether `KNOWLEDGE.md` remains PROPOSED or became VERIFIED based on actual measurements.
