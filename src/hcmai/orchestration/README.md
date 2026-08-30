# Search orchestration

`hcmai.orchestration` owns the online application flow. FastAPI routers call
`SearchService` from `pipeline.py`; they do not wire retrieval or temporal
alignment internals themselves.

```text
orchestration/
├── pipeline.py              # SearchService public facade
├── task_router.py           # Task-pipeline registry
├── workflows/               # Executable task-specific orchestration
├── setup.py                 # Single application composition root
└── materializer.py          # Canonical SearchResponse construction
```

## Runtime path

```text
FastAPI
    → SearchService
    → task workflow (KIS or TRAKE)
    → TemporalSearchService + RetrievalService
    → DataService-backed canonical materialization
```

`SearchService` resolves each request through `PipelineRegistry`. The current
registry exposes KIS and TRAKE. Both consume one stateless ordered
event-to-frame alignment service; KIS projects a path to its existing
single-frame public response and TRAKE returns the full path.

`setup.py` loads configuration and artifacts once, constructs the selected
services, and injects them into `SearchService`. Cross-component imports in
this package target only another component's `pipeline.py` or shared schemas.

Canonical identity is immutable: retrieval and alignment may score or order
candidate frames, while materialization alone resolves each exact
`frame_id` to its official `video_id` and integer `frame_idx` through
`DataService`.

Retrieval latency and warnings are returned in a request-scoped
`RetrievalResult`. KIS derives query-encoding, index-search, and fusion latency
from that trace; no timing is read from a mutable singleton service field.
Completed stages also emit a JSON record containing `request_id`, `task_type`,
`stage`, `duration_ms`, and `status`.

## Verification

```bash
PYTHONPATH=.:src aic/bin/pytest tests/test_search_service.py
pyright src/hcmai/orchestration
```
