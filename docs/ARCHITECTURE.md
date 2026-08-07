# HCMAI Runtime Architecture

HCMAI is a modular monolith. Keep one API process and one optional inference
process; do not introduce microservices, a DI container, or an event bus for
the competition runtime.

## Online flow

```text
FastAPI router
  -> SearchService facade
  -> PipelineRegistry
  -> KISPipeline or VQAPipeline
  -> DataService / RetrievalService / RerankingService / LLMService
  -> provider adapter or local artifact
  -> canonical response materialization
```

Routers validate HTTP contracts and map application errors. Task pipelines
describe the ordered workflow. Service facades own long-lived resources and
provider boundaries. Pure ranking, parsing, fusion, and normalization logic
stays as functions.

## Public boundaries

Cross-component production imports target either the owning package's
`pipeline.py` or authoritative contracts under `hcmai.common.schemas`.
`hcmai.orchestration.setup` is the online composition root. The private
inference server has a separate model-runtime composition boundary.

## Canonical identity and assets

`frame_id -> video_id -> frame_idx` is immutable. `DataService` owns frame
metadata and `FrameAssetResolver`; API serving, reranking, and VQA resolve the
same canonical `image_path` against the configured dataset root. A missing
asset degrades optional reranking and VQA answering without changing frame
identity.

## Debug map

```text
KIS: /api/v1/search -> routers/search.py -> pipelines/kis.py
     -> retriever/pipeline.py -> optional reranking/pipeline.py

VQA: /api/v1/vqa -> routers/vqa.py -> pipelines/vqa.py
     -> vqa/{parser,candidates,windows,evidence,localizer,answerer,ranking}.py
     -> llm/pipeline.py
```

Every request stage records request ID, task type, duration, status, backend,
fallback, warning, and error category. Run `scripts/doctor.py` before a session
to validate metadata, frame assets, visual-index alignment, evidence artifacts,
and optional remote inference readiness.

## Research boundary

Reusable implementations live in `src/hcmai`. Thin scripts and experiments
call public services. Outputs live under gitignored `runs/<run-id>/` with the
resolved config, predictions, failures, warnings, latency, and `metrics.json`.
Notebook-only production logic is not allowed.

## Deliberately avoided patterns

- microservices and service discovery;
- generalized plugin frameworks;
- Repository/Unit of Work without a database boundary;
- deep inheritance trees;
- abstract interfaces with only one implementation;
- hidden mutable request state.

TRAKE remains an externally owned pipeline behind stable shared contracts.
Shared schema changes require coordination and must not modify its internals.
