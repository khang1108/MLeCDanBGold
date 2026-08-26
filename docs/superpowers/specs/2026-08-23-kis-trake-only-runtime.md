# KIS + TRAKE-only runtime specification

**Status:** Proposed from repository trace on 2026-08-23

## Goal

Retire the VQA workflow and hosted VQA model completely. The active product
runtime must expose exactly two competition tasks: KIS and TRAKE.

## Decisions

1. `TaskType.KIS` and `TaskType.TRAKE` are the only supported task types.
2. `TaskType.VQA` and `TaskType.VKIS` are removed from public contracts,
   registry construction, health capabilities, and config. The frontend is
   deliberately scoped more narrowly: remove its VQA API client/docs and
   direct VQA API callsite, but leave the existing shared KIS/TRAKE workspace
   and presentation code in place.
3. `/api/v1/vqa`, `/v1/vqa`, and `/v1/vqa/multi` are removed. Old clients do
   not receive a compatibility VQA response.
4. The shared inference service remains responsible for embeddings, captions,
   OCR, ASR/diarization, and optional KIS reranking. Only grounded answer
   generation and its model are removed.
5. KIS and TRAKE workflow files remain in their current locations during this
   migration. Moving them into a new task-package hierarchy is a separate
   change because it adds import and regression risk without being required
   for VQA retirement.
6. Historical research documents may retain VQA findings. Active runtime
   documentation, configuration, and operator instructions must describe only
   KIS and TRAKE.

## Current runtime trace

```text
POST /api/v1/search
  -> create_search_router
  -> SearchService.search
  -> PipelineRegistry
  -> KISPipeline
  -> TemporalEvidenceCore.localize
  -> ProgressiveEvidenceProvider + ProgressiveSceneAligner
  -> SearchMaterializer
  -> SearchResponse

POST /api/v1/trake
  -> create_trake_router
  -> SearchService.search
  -> PipelineRegistry
  -> TRAKEPipeline
  -> TemporalEvidenceCore.ordered_plan/align_ordered
  -> DenseOrderedEvidenceProvider + MonotonicOrderedPathAligner
  -> TRAKEResponse

POST /api/v1/vqa (removed)
  -> VQAPipeline
  -> question-aware evidence selection
  -> LLMService.answer_vqa[_multi]
  -> grounded answer submission
```

## Target backend architecture

```text
src/hcmai/
├── api/routers/
│   ├── search.py             # KIS
│   ├── trake.py             # TRAKE
│   ├── frames.py            # shared canonical frame/assets
│   └── system.py            # health/readiness
├── common/
│   ├── config.py             # dataset, index, search, inference
│   └── schemas/
│       ├── enum.py           # KIS, TRAKE
│       ├── search.py         # KIS request/response
│       ├── trake.py          # TRAKE request/response
│       ├── task.py           # SearchRequest | TRAKERequest unions
│       ├── inference.py     # shared model-service contracts
│       └── ...               # frame/evidence/retrieval/temporal contracts
├── data/                     # canonical data and specialist evidence
├── retrieval/                # embeddings, indexes, fusion, reranking
├── temporal/                 # progressive KIS + ordered TRAKE alignment
├── orchestration/
│   ├── setup.py
│   ├── pipeline.py           # SearchService facade
│   ├── task_router.py        # KIS/TRAKE registry
│   ├── materializer.py       # shared KIS response materialization
│   └── workflows/
│       ├── kis.py
│       └── trake.py
└── app.py
```

`thundercompute/` remains a shared inference gateway, but contains no VQA
adapter, config, endpoint, or model. The current frontend workspace is kept
because it is the existing entrypoint for both KIS and TRAKE despite its
legacy VQA-oriented filename. This migration removes only its VQA endpoint
client/callsite and API documentation; renaming, splitting, deleting, or
reworking the workspace UI is a separate frontend task.

## Invariants

- `video_id`, internal `frame_id`, official `frame_idx`, and `timestamp_ms`
  remain distinct and preserved.
- KIS materialization continues through `DataService.get_frame()` and
  `official_frame_idx()`.
- TRAKE keeps aligned `frame_ids`, `frame_idxs`, and `timestamps_ms` for each
  ordered event.
- Specialist evidence and provenance remain available to retrieval and
  temporal scoring.
- Expensive shared model/index loading remains offline or in the existing
  inference service lifecycle; no VQA model is loaded at startup or request
  time.

## Non-goals

- Do not remove shared caption, OCR, ASR, diarization, embedding, retrieval,
  or reranking code.
- Do not move KIS/TRAKE workflow packages in this migration.
- Do not purge historical VQA research notes or old design records merely
  because VQA is no longer an active runtime task.
- Do not restore the currently deleted ThunderCompute launcher, delete helper,
  or bootstrap example; those are existing user-owned working-tree changes.

## Acceptance criteria

1. Backend imports and starts with only KIS and TRAKE registered.
2. `/api/v1/search`, `/api/v1/trake`, frame routes, and `/health` remain usable.
3. `/api/v1/vqa` is absent; no VQA pipeline or schema is importable from the
   active package.
4. The private inference service has no VQA route, model checkpoint, model
   flag, or `multi_image_vqa` readiness field.
5. Frontend no longer calls or advertises `/api/v1/vqa`; the existing shared
   KIS/TRAKE workspace remains available and is not broadly refactored.
6. Focused KIS/TRAKE/backend/frontend tests pass, followed by the full
   deterministic repository gate.
