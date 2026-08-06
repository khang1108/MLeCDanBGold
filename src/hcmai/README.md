# `hcmai` Package Reference

HCMAI is a modular monolith for multimodal video-frame retrieval. Each domain
service package shown below exposes one public service from its top-level
`pipeline.py`; model and provider implementations stay behind feature-owned
adapters. `api`, `common`, research agents, and the internal baseline evaluator
are explicit exceptions.

## Package layout

```text
src/hcmai/
├── app.py                    # FastAPI lifecycle and router assembly
├── api/routers/              # Thin HTTP adapters over SearchService
├── orchestration/
│   ├── pipeline.py           # SearchService: online task routing
│   ├── setup.py              # Application composition root
│   └── materializer.py       # Canonical candidate materialization
├── data/                     # DataService and canonical frame stores
├── embedding/                # EmbeddingService and model adapters
├── retriever/                # RetrievalService, indexes, and fusion
├── reranking/                # RerankingService and scoring adapters
├── enrichment/               # Caption/OCR EnrichmentService
├── transcripts/              # TranscriptService and ASR adapters
├── llm/                      # LLMService and local/HTTP adapters
├── query_suggestions/        # SuggestionService and provider adapters
├── submission/               # Optional DRES mini-challenge client
├── agents/kisc/              # Bounded conversational KIS research code
└── common/                   # Shared config, schemas, and generic utilities
```

## Dependency rule

Production code crossing a component boundary imports only its public
`pipeline.py` or a shared schema from `common`:

```python
from hcmai.data.pipeline import DataService
from hcmai.embedding.pipeline import EmbeddingService
from hcmai.retriever.pipeline import RetrievalService
from hcmai.reranking.pipeline import RerankingService
from hcmai.orchestration.pipeline import SearchService
```

`models/` contains contracts, entities, metadata, statistics, and value
objects. Concrete SigLIP, BGE, Qwen, remote HTTP, ASR, and enrichment backends
belong in `adapters/`. Unit tests may import internals for focused testing;
scripts and integration code call the owning service.

## Runtime paths

Online traffic uses one configured path:

```text
FastAPI → SearchService → RetrievalService → optional RerankingService
        → canonical response materialization
```

KIS and initial VKIS queries use frame retrieval directly. KISC research code
resolves only context-dependent turns and then calls `SearchService`. VQA and
TRAKE are recognized but return `501` until their real pipelines exist.

Offline research jobs call their owning service directly, for example
`DataService` for frame preparation, `EmbeddingService` for vector artifacts,
and `RetrievalService` for index construction. This keeps experiments modular
without adding a production-style dependency-injection framework.

There is no public `EvaluationService` yet. The existing
`retriever/evaluation/benchmark.py` is an internal dense-baseline tool; an
end-to-end evaluator remains deferred until its contract and second use are
demonstrated.

## Canonical identity invariant

Every displayed or submitted result resolves through the authoritative
`frame_id → video_id → frame_idx` mapping. Never derive `frame_idx` from a
timestamp, FPS, filename, array position, or neighboring frame.

## Running the API

```bash
PYTHONPATH=src aic/bin/python -m uvicorn hcmai.app:app \
  --host 127.0.0.1 --port 8000
```

The public endpoints remain:

- `GET /health`
- `POST /api/v1/search`
- `POST /api/v1/query-suggestions`
- `GET /api/v1/frames/{frame_id}` and frame asset/neighbor routes
- `POST /api/v1/submit`
- `GET /api/v1/minichallenge/evaluations`
- `GET /api/v1/minichallenge/evaluations/{id}/current-task`
- `POST /api/v1/minichallenge/evaluations/{id}/submit`

The mini-challenge routes require the DRES session token in the
`X-DRES-Session` header. The browser keeps this token only in memory. Submission
requests contain a canonical `frame_id`; the backend resolves its authoritative
`video_id` and sends the DRES `answerSets` payload with `start=end=0`. Configure
the upstream with `HCMAI_MINICHALLENGE_BASE_URL` and bound network calls with
`HCMAI_MINICHALLENGE_TIMEOUT_SECONDS`.
