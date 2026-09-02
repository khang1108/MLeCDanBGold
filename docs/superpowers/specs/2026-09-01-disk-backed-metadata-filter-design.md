# Disk-Backed Metadata Filter Design

**Date:** 2026-09-01  
**Status:** RETIRED on 2026-09-02

> This design is retained only as historical context. The exact Filter backend,
> catalog, and offline builder were removed by product decision. Only
> `POST /api/v1/filter` remains as a `501 Not Implemented` development stub.

**Implementation:** See
`docs/superpowers/plans/2026-09-02-disk-backed-metadata-filter.md`. The
470,804-frame catalog built in 127.46 seconds and occupied 1,316,909,056 bytes.
With the real Visual FAISS index resident, the worst concurrency-10 P95 was
1,905.20 ms and cumulative Filter RSS growth was about 36.3 MiB.

## 1. Understanding Summary

HCMAI needs a `POST /api/v1/filter` endpoint that serves the merged Filter
Workspace without involving semantic retrieval, FAISS, reranking, or VLM
inference. The feature is an exact, deterministic metadata filter over
canonical frames.

Only populated fields participate. Title, ASR, Caption, OCR, Object, folder,
and video conditions combine with AND. Text fields use normalized substring
matching. Object counts use exact equality. ASR is frame-relevant only when a
matching transcript segment contains the frame timestamp.

The backend owns filtering, stable ordering, result counts, and pagination.
The response contains full display metadata for only the requested page so the
frontend does not issue one detail request per frame.

The deployment host has 16 GiB RAM and must reserve memory for FAISS and its
runtime mappings. Therefore the filter view is an offline SQLite artifact,
opened read-only during serving, rather than a second in-memory copy of the
corpus evidence.

## 2. Goals

- Preserve canonical `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms`.
- Match the existing frontend request entrypoint and pagination behavior.
- Keep filter behavior deterministic, inspectable, and independent of search.
- Avoid material RAM growth that competes with FAISS.
- Support approximately ten concurrent filter requests with bounded resources.
- Keep KIS and TRAKE available when the optional filter catalog is missing.

## 3. Non-Goals

- Semantic similarity, fuzzy matching, learned scoring, or relevance ranking.
- Using FAISS, embeddings, rerankers, LLMs, or VLMs for Filter V1.
- Rebuilding filter artifacts during application startup or an HTTP request.
- Adding authentication or public-traffic rate limiting in the local V1.
- Replacing specialist evidence artifacts with the derived filter catalog.
- Changing canonical identity or deriving submission coordinates from row order.

## 4. Assumptions and Constraints

- The active corpus contains roughly 470,000 canonical frames.
- The first deployment is local; a future deployment may use Cloudflare Tunnel.
- `folder_id` is derived from the organizer prefix in `video_id`, for example
  `L21_V001 -> L21`, without rewriting the original ID.
- Empty filters return all frames within any supplied folder/video scope.
- A requested page beyond the last page returns an empty `results` array while
  retaining the real `total_pages` value.
- If an entire modality was unavailable when the catalog was built, its
  predicate is ignored and a backend warning is logged. The response does not
  expose an ignored-filter warning, by explicit product decision.
- If a modality is available but one frame has no usable evidence, that frame
  does not match a requested filter for the modality.

## 5. Architecture

### 5.1 Offline build path

```text
canonical frames
+ video metadata
+ Caption
+ OCR
+ Object counts
+ timestamped ASR segments
        -> Filter Catalog Builder
        -> validated temporary SQLite file
        -> atomic publish
        -> artifacts/filter/filter_catalog.sqlite
```

The builder belongs to the offline package. It reads already-published source
artifacts, validates identity and available lineage, writes a temporary catalog,
validates row counts and invariants, then atomically replaces the published
catalog. A failed build must leave the previous catalog intact.

ASR projection is derived offline. A frame receives only text from transcript
segments whose half-open interval contains its timestamp. Original transcript
segments remain authoritative and independently available.

### 5.2 Online path

```text
POST /api/v1/filter
        -> thin filter router
        -> FilterService
        -> bounded read-only SQLite pool
        -> FilterResponse
```

`FilterService` owns predicate construction, ordering, counts, and pagination.
It does not own artifact creation and does not call the retrieval subsystem.
The canonical Corpus remains authoritative for search and frame viewing.

## 6. Catalog Model

The initial schema contains three logical tables.

### `catalog_metadata`

Stores catalog version, build timestamp, canonical frame count, source artifact
versions/checksums when available, and modality availability flags.

### `frames`

Stores:

- canonical identity and coordinates;
- derived `folder_id`;
- raw/display Title, Caption, OCR, and ASR text;
- separately normalized Title, Caption, OCR, and ASR text.

No filesystem image or thumbnail path is published through the API.

### `frame_objects`

Stores `frame_id`, normalized object label, and exact integer count. Repeated
detections remain represented through their count and are never collapsed into
presence-only labels.

B-tree indexes cover folder/video scope and deterministic frame ordering.
The baseline intentionally does not add FTS or trigram indexes. Those are an
experiment only if measured global substring queries miss the latency target.

## 7. HTTP Contract

The request keeps the frontend's existing `frames_per_pages` field name.

```json
{
  "metadata_filters": {
    "title": "video",
    "asr": "xin chao",
    "caption": null,
    "ocr": null,
    "objects": {"person": 3}
  },
  "folder_id": "L21",
  "video_id": "L21_V001",
  "frames_per_pages": 12,
  "page_id": 1
}
```

Validation requires `page_id >= 1`, `1 <= frames_per_pages <= 48`, bounded text
lengths, nonblank object labels, and nonnegative integer object counts. The
backend normalizes input again even though the frontend already normalizes it.

The response contains `page_id`, `frames_per_pages`, `total_results`,
`total_pages`, and the current page's results. Each result contains canonical
identity, `folder_id`, Title, Caption, OCR, object counts, and ASR. Keyframe URLs
remain frontend-derived from canonical `frame_id`.

## 8. Matching and Pagination

1. Read modality availability from catalog metadata.
2. Omit unavailable modality predicates and log the omission.
3. Apply folder and exact canonical video scope.
4. Apply normalized substring predicates for populated text fields.
5. Require an exact count match for every requested object label.
6. Combine all active predicates with AND.
7. Order by `video_id`, `timestamp_ms`, `frame_idx`, then `frame_id`.
8. Run a count query and calculate total pages.
9. Fetch only the page through `LIMIT` and `OFFSET`.

Normalization uses Unicode decomposition, removes combining marks, maps
Vietnamese `đ/Đ` consistently, lowercases, trims, and collapses whitespace.

## 9. Frontend Integration

Filter Workspace consumes the complete page response directly. Its
`getFrameDetail` fan-out, detail cache, and per-frame abort controllers are
removed from this flow. The general frame detail endpoint remains available to
other consumers.

The page-size selector offers `Auto`, `12`, `24`, and `48`. Auto calculates an
effective integer from the viewport. That effective size is fixed for the
current filter session so resizing does not mutate pagination unexpectedly.
Changing the selector or applying a new filter requests page 1. Page navigation
reuses the applied filter, scope, and effective page size.

## 10. Reliability, Security, and Resource Bounds

- Missing/unopenable catalog: Filter returns HTTP 503; KIS/TRAKE remain active.
- Invalid input: HTTP 422.
- Corrupt catalog or violated runtime invariant: JSON HTTP 500 with server log.
- Builder publication is atomic and serving never regenerates the catalog.
- SQLite is opened read-only with `query_only=ON`, `temp_store=FILE`, and
  `mmap_size=0`.
- Use at most four read-only connections, each with about an 8 MiB page cache.
- Approximately ten concurrent requests may wait for the bounded pool rather
  than creating more connections.
- Target filter-specific RAM growth is below roughly 64 MiB.
- Responses never expose database paths, filesystem paths, or source internals.
- V1 adds no auth; input bounds are retained for a future tunneled deployment.

Health reporting adds a Filter capability plus catalog version/frame count.
Filter degradation does not change the search readiness contract.

## 11. Testing Strategy

Use small, hand-checkable fixtures for contracts, normalization, multi-field
AND, exact counts, ASR containment, folder parsing, missing modalities, stable
ordering, pagination, empty filters, out-of-range pages, and no path leakage.

Builder tests cover canonical identity/lineage rejection, modality availability,
ASR projection, exact object multiplicity, validation, and atomic publication.
API tests prove that a missing catalog returns 503 without breaking KIS/TRAKE.
Frontend tests prove full page consumption, removal of N+1 detail calls, page
size behavior, and reset to page 1.

## 12. Acceptance Benchmark

Benchmark the real approximately 470,000-frame catalog with cold and warm
global, folder, video, object, and combined filters. Test page 1 and final page
at 1, 4, and 10 concurrent requests. Record P50/P95, catalog size, build time,
process RSS before/after Filter initialization, and RSS with FAISS loaded.

Targets are representative P95 below two seconds, filter-specific RAM growth
below roughly 64 MiB, and no KIS/TRAKE behavior change. If global substring
queries fail, evaluate SQLite FTS5 trigram as a separate measured experiment.

## 13. Decision Log

| Decision | Alternatives | Rationale |
|---|---|---|
| AND across populated fields | OR; mixed text OR/scope AND | Matches separate UI fields and exact-filter intent. |
| Normalized substring text | token-anywhere; fuzzy | Deterministic and easy to debug. |
| Exact object counts | minimum; presence-only | Explicit user requirement. |
| ASR segment contains timestamp | temporal window; whole video | Preserves timeline semantics and provenance. |
| Video/time stable ordering | artifact order; scoring | Deterministic pagination without inventing relevance. |
| Separate FilterService | SearchService; router logic | Keeps metadata filtering out of temporal retrieval and routers thin. |
| Disk-backed SQLite | full RAM projection; direct corpus scan | Preserves scarce RAM for FAISS while staying operationally simple. |
| Full page metadata response | N+1 detail calls; click-only details | Avoids up to 48 extra requests per page. |
| Bounded four-connection pool | connection per request; one serial connection | Supports about ten callers with predictable memory. |
| No FTS baseline | immediate trigram index | Measure before adding complexity. |
| Missing modality silently omitted to UI | 503; response warning | Explicit product choice; backend still logs it. |
