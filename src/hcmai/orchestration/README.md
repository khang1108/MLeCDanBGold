# Search orchestration

`hcmai.orchestration` owns the online application flow. FastAPI routers and
KISC composition call `SearchService` from `pipeline.py`; they do not wire
retrieval or reranking internals themselves.

```text
orchestration/
├── pipeline.py              # SearchService public task router
├── setup.py                 # Single application composition root
├── ranking.py               # Bounded retrieval/reranking sequence
└── materializer.py          # Canonical SearchResponse construction
```

## Runtime path

```text
FastAPI or KISC
    → SearchService
    → RetrievalService
    → optional RerankingService
    → DataService-backed canonical materialization
```

`SearchService` routes Textual KIS, initial VKIS, and resolved KISC queries.
It recognizes VQA and TRAKE but raises `SearchPipelineUnavailableError` until
their real end-to-end stages exist. The comments in `_validate_task` document
the intended stages; do not replace them with placeholder services.

`setup.py` loads configuration and artifacts once, constructs the selected
services, and injects them into `SearchService`. Cross-component imports in
this package target only another component's `pipeline.py` or shared schemas.

Canonical identity is immutable: retrieval and reranking may score or reorder
`RetrievalCandidate` objects, while materialization alone resolves each exact
`frame_id` to its official `video_id` and integer `frame_idx` through
`DataService`.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest tests/test_search_service.py
pyright src/hcmai/orchestration
```
