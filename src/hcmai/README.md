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
├── submission/               # Optional DRES mini-challenge client
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
FastAPI → SearchService → query embedding → RetrievalService
        → optional RerankingService
        → canonical response materialization
```

KIS and VKIS queries use the original operator text directly; online query
suggestion and conversational resolution are not part of the search path.
VQA retains its separate grounded multimodal answering path.

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
- `POST /api/v1/vqa`
- `GET /api/v1/frames/{frame_id}` and frame asset/neighbor routes
- `POST /api/v1/submit`
- `GET /api/v1/minichallenge/evaluations`
- `GET /api/v1/minichallenge/evaluations/{id}/current-task`
- `POST /api/v1/minichallenge/evaluations/{id}/submit`

Run `PYTHONPATH=src aic/bin/python scripts/doctor.py` before starting a
competition session. It validates that metadata, canonical frame assets, and
the visual index agree without loading model weights. See
`docs/ARCHITECTURE.md` for the dependency and debugging map.

The mini-challenge routes use the latest in-memory backend session when one is
available, with `X-DRES-Session` and the static environment session retained as
fallbacks. This prevents a browser-held token from overriding a newer automatic
refresh. Submission requests contain a canonical `frame_id`; the backend resolves
its authoritative `video_id` and sends the DRES `answerSets` payload with
`start=end=0`. Configure the upstream with
`HCMAI_MINICHALLENGE_BASE_URL` and bound network calls with
`HCMAI_MINICHALLENGE_TIMEOUT_SECONDS`. Set `DES_USERNAME` and `DES_PASSWORD` to
log in once at startup and refresh the in-memory session automatically every
`HCMAI_MINICHALLENGE_SESSION_REFRESH_SECONDS` (300 seconds by default).
