# HCMAI runtime architecture

`hcmai` is the online runtime for HCMAI 2026 multimodal video retrieval. It
opens already-built corpus and index artifacts, searches them, aligns temporal
evidence, and shapes competition responses. Corpus-scale ingestion,
enrichment, embedding, and index construction belong to `offline` and the
repository scripts; they never run during application startup or a request.

## Runtime path

```text
FastAPI router
  -> SearchService
  -> KIS or TRAKE workflow
  -> TemporalSearchService
  -> RetrievalService load/search
  -> Corpus canonical materialization
  -> competition-compatible response
```

`Corpus.open(...)` is the read-only boundary for existing canonical frames,
Caption, OCR, object-count, transcript, and media-metadata artifacts. It does
not create, migrate, or republish artifacts. `RetrievalService` similarly owns
runtime index loading and search; builders live under `offline.embeddings` and
`offline.indexes` and are invoked by offline commands.

## Package owners

```text
api/             HTTP validation and response shaping
common/          configuration, logging, and cross-cutting utilities
corpus/          read-only canonical frames, evidence, and asset resolution
retrieval/       query encoding, index loading/search, fusion, and ranking
temporal/        query planning and ordered temporal alignment
orchestration/   application composition and thin KIS/TRAKE workflows
llm/             model gateway contracts and adapters
```

Offline construction ownership is documented in
[`offline/README.md`](../../offline/README.md). Runtime code must not import
offline producers, and offline code must not import API or orchestration
packages.

## Canonical identity

Every runtime stage preserves:

```text
video_id
frame_id
frame_idx
timestamp_ms
```

`frame_id` is the internal join identity. `frame_idx` is the BTC
competition-facing coordinate and is never inferred from keyframe order,
filename number, decode position, or an array index. Retrieval and alignment
may rank candidates but cannot invent or alter canonical identity.

Specialist evidence remains independently traceable. Missing or unevaluated
evidence is not converted into a negative score, object multiplicity remains
available in the offline artifact, and ASR remains timestamped timeline
evidence rather than frame-native truth.

## Startup and verification

The existing artifact paths remain controlled by `configs/baseline.yaml` and
environment overrides. Phase B does not introduce another artifact layout.
Start the backend only after the configured frame store and required visual
index are present:

```bash
PYTHONPATH=.:src aic/bin/python -m uvicorn hcmai.app:app \
  --host 127.0.0.1 --port 8000
```

`GET /health` reports frame-store, retrieval, evidence, and frame-asset
readiness. The principal task routes are `POST /api/v1/search` and
`POST /api/v1/trake`; frame and submission routes materialize identity through
the same `Corpus` instance.
