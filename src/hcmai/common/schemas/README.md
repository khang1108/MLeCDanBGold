# Shared schemas

This package contains the Pydantic models and enums shared by the retrieval
pipeline, evaluation tools, and public API. The schemas are intentionally
strict: `ContractModel` rejects unknown fields and strips surrounding
whitespace from strings.

Only cross-module exchange contracts belong here. A configuration, backend
protocol, intermediate result, or report model used by one feature stays in
that feature package. In particular, caption/OCR types belong under
`enrichment/`, and reranker configuration belongs under `reranking/`.
`common.schemas` must not import feature modules.

## Modules and definitions

### `base.py`

- `NonEmptyString`: an `Annotated[str, ...]` type that strips whitespace and
  requires at least one character.
- `ContractModel`: base Pydantic model for shared contracts. Extra fields are
  forbidden and string fields are stripped of surrounding whitespace.

### `enum.py`

- `ProcessingStatus`: offline processing state: `pending`, `processing`,
  `completed`, or `failed`.
- `RetrievalSource`: evidence source used during retrieval: `visual`,
  `caption`, `ocr`, or `asr`.
- `QueryLanguage`: query language: Vietnamese (`vi`), English (`en`), or
  mixed (`mixed`).
- `TaskType`: public task type: `kis` or `trake`.
- `QueryDifficulty`: evaluation difficulty: `easy`, `medium`, or `hard`.

### `frame.py`

- `FrameRecord`: canonical metadata for one searchable frame. It preserves the
  authoritative `frame_idx`, optional official `keyframe_order`, relative
  `image_path`, source dimensions, and `timestamp_ms`. Submission identifiers
  are never derived from another field.
- `FrameEnrichment`: offline caption, OCR, ASR, object-label, model, and
  processing-status metadata associated with a `frame_id`. Duplicate object
  labels are removed while preserving order.

### `catalog.py`

- `FrameCatalogEntry`: API projection of canonical identity plus loaded caption,
  OCR, object-count, video metadata, and timestamp-containing ASR segments.

### `transcript.py`

- `TranscriptSegment`: canonical text, language, dominant speaker, and
  millisecond boundaries for one ordered spoken segment in a source video.

### `retrieval.py`

- `RetrievalCandidate`: internal candidate passed between retrieval stages. It
  stores per-source scores and ranks, fusion/reranker/final scores, and
  optional metadata. Source ranks must be positive one-based integers.
- `SearchScores`: scores exposed for a returned frame, including visual,
  caption, OCR, ASR, fusion, reranker, and final scores.
- `RetrievalResult`: one request-owned candidate list, retrieval trace, and
  warning list. Sequence access remains available for compatibility, while
  production callers consume `.candidates` and `.trace` explicitly.

### `alignment.py`

- `AlignmentEvent`: one deterministic, non-empty semantic event with its
  ordered position in an alignment request.
- `AlignmentPlan`: task, ordered events, and optional search filters consumed
  by the shared temporal alignment service.
- `AlignmentPath`: a same-video, strictly chronological canonical frame path.
  `frame_ids`, `frame_idxs`, and `timestamps_ms` are parallel immutable
  sequences; `frame_idx` remains the competition coordinate.

### `telemetry.py`

- `StageTrace`: one stage's monotonic start/end, duration, status, attempt
  count, cache state, and optional error category.
- `PipelineTrace`: uniquely named stages with deterministic merge and duration
  aggregation helpers.
- `RetrievalTrace`: request-scoped retrieval specialization of `PipelineTrace`.

### `search.py`

- `SearchFilters`: optional video and time-range restrictions. Video IDs are
  deduplicated, and `end_time_ms` cannot precede `start_time_ms`.
- `SearchRequest`: public standalone-search request containing a typed
  `query_type`, a non-empty query, bounded `top_k`, optional filters, and an
  optional compatibility `search_id`. Alignment execution is stateless.
- `SearchLatency`: non-negative latency measurements for each search stage and
  the total request, in milliseconds.
- `SearchResult`: one ranked result with required singular canonical
  `frame_id`, bounded scene `frame_ids`, BTC `frame_idx`, preview URLs,
  enrichment text, and scores.
- `SearchResponse`: complete search response with request metadata, latency,
  results, warnings, and the optional echoed `search_id`. `total_results` must
  match the result list and cannot exceed `top_k`.

### `trake.py`

- `TRAKERequest`: raw query and optional caller-supplied ordered events.
- `TRAKESubmission`: one same-video canonical frame mapping per ordered event.
  Aligned `frame_ids`, BTC `frame_idxs`, and `timestamps_ms` preserve both
  exact internal identity and competition coordinates for UI playback.
- `TRAKEResponse`: bounded ranked temporal-alignment submissions.

### `task.py`

- `TaskRequest` and `TaskResponse`: discriminated KIS and TRAKE unions for
  internal task routing and typed API adapters.

### `evaluation.py`

- `EvaluationQuery`: labelled query for offline evaluation. It contains the
  query metadata, gold frame/video IDs, temporal tolerance, tags, and optional
  notes. ID and tag lists are deduplicated while preserving order.

### `submission.py`

- `SubmissionResult`: official KIS submission output format containing
  `frame_id`, `video_id`, `frame_idx`, and a validated
  `video_id,frame_idx` submission code.

## Importing

Import shared base types and the public package exports like this:

```python
from hcmai.common.schemas import ContractModel, NonEmptyString
```

For models that are defined in a module but are not re-exported by
`__init__.py`, import them from their defining module:

```python
from hcmai.common.schemas.frame import FrameRecord
from hcmai.common.schemas.search import SearchRequest, SearchResponse
```

`__all__` in `__init__.py` documents the symbols intentionally re-exported
from the package root. Prefer explicit imports in application code.
