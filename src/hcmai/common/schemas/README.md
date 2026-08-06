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
- `TaskType`: frontend query type: `kis`, `kisc`, `vkis`, `vqa`, or `trake`.
- `ExecutionProfile`: bounded task profile: `fast`, `balanced`, `accurate`, or
  `competition_anytime`.
- `QueryDifficulty`: evaluation difficulty: `easy`, `medium`, or `hard`.

### `frame.py`

- `FrameRecord`: canonical metadata for one searchable frame. It preserves the
  authoritative `frame_idx`, optional official `keyframe_order`, relative
  `image_path`, source dimensions, and `timestamp_ms`. Submission identifiers
  are never derived from another field.
- `FrameEnrichment`: offline caption, OCR, ASR, object-label, model, and
  processing-status metadata associated with a `frame_id`. Duplicate object
  labels are removed while preserving order.

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
  `query_type`, a non-empty query, bounded `top_k`, optional filters, optional
  legacy session context, and human `feedback`. Feedback is valid only with a
  session ID. The HTTP router accepts KIS, VKIS, VQA, and TRAKE here; KISC uses
  `KISCSearchRequest`.
- `SearchLatency`: non-negative latency measurements for each search stage and
  the total request, in milliseconds.
- `SearchResult`: one ranked result with the canonical frame identifiers,
  preview URLs, enrichment text, and scores.
- `SearchResponse`: complete search response with request metadata, latency,
  results, warnings, and optional KISC context. Conversational responses contain
  a session ID, user `turn_id`, `assistant_turn_id`, and `ai_message` together.
  `total_results` must match the result list and cannot exceed `top_k`.

### `kisc.py`

- `KISCSearchRequest`: typed `kisc` request with browser-owned ordered
  history, current message, prior interpreted state, feedback, and filters for one stateless
  KISC turn.
- `KISCSearchResponse`: complete interpreted state, resolution latency, and a
  canonical nested `SearchResponse`.

### `vqa.py`

- `VQARequest`: competition event description, question, Top-k, optional
  filters, language hint, and execution profile.
- `VQASubmission`: ranked canonical video/frame/answer row with retrieval,
  grounding, answer, and joint scores.
- `VQAResponse`: bounded ranked competition submissions.
- `VQAInferenceRequest`, `VQAInferenceEvidence`, and `VQAInferenceResponse`:
  explicitly provider-scoped one-frame inference contracts.

### `trake.py`

- `TRAKERequest`: raw query and optional caller-supplied ordered events.
- `TRAKESubmission`: one same-video canonical frame mapping per ordered event.
- `TRAKEResponse`: bounded ranked temporal-alignment submissions.

### `task.py`

- `TaskRequest` and `TaskResponse`: discriminated KIS/VKIS, VQA, and TRAKE
  unions for internal task routing and typed API adapters.

### `evaluation.py`

- `EvaluationQuery`: labelled query for offline evaluation. It contains the
  query metadata, gold frame/video IDs, temporal tolerance, tags, and optional
  notes. ID and tag lists are deduplicated while preserving order.

### `conversation.py`

- `ConversationConstraint`: one resolver fact with a semantic slot, value,
  positive/negative/uncertain polarity, and source turn ID.
- `ConversationState`: complete interpreted KISC state containing the
  standalone query, positive/negative/uncertain constraints, and accepted or
  rejected frame IDs.
- `ConversationTurn`: one user or AI message in a KIS conversation with
  turn ID, typed sender (`user` or `ai`), timestamp, optional reply target, and
  non-empty message.
- `FrameFeedback`: ordered, deduplicated accepted/rejected frame IDs. One update
  cannot contain the same frame in both decisions.
- `ConversationSession`: active KISC conversation session state containing
  session ID, optional problem ID, creation timestamp, turns, and cumulative
  feedback.
- `SubmissionResult`: official competition submission output format containing
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
