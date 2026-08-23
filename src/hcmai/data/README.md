# HCMAI 2026 BTC-native data preparation

`hcmai.data` owns canonical frame ingestion, specialist evidence artifacts,
derived frame context, and typed data access. The active competition profile
uses organizer-provided BTC keyframes and objects; it does not decode videos to
extract another frame set.

## Competition flow

```text
BTC keyframes ──> Caption ──────┐
              └─> OCR ──────────┤
BTC objects ────> Object Import ├─> FrameContext V1
Videos ─────────> ASR segments  │   (ASR excluded)
                                 └─> specialist artifacts
```

Caption, OCR, object import, and ASR produce independent specialist artifacts.
FrameContext V1 is a deterministic derived view of Caption, usable normalized
OCR, and the object summary, in that order. ASR remains timestamped timeline
evidence and is intentionally absent from FrameContext dependency identity and
text.

`src/hcmai/data/preprocessing/**` remains available for non-BTC experiments.
It is not used by the HCMAI 2026 competition preparation profile.

## Artifact ownership

The specialist outputs remain independently inspectable. A context or legacy
projection never replaces them.

| Artifact | Role |
| --- | --- |
| `captions.parquet` | Caption source of truth |
| `ocr/frames.parquet` | Frame-level OCR source of truth |
| `ocr/regions.parquet` | Raw OCR regions and boxes; source of truth |
| `objects/frames.parquet` | Frame-level object counts/summary; source of truth |
| `objects/detections.parquet` | Every valid BTC detection and box; source of truth |
| `transcripts/*.parquet` | Timestamped ASR segment source of truth |
| `context/frame_context_v1.parquet` | Derived Caption + OCR + Object view |
| `frame_enrichment.parquet` | Temporary compatibility projection only |

The compatibility projection exists for retrieval code that has not yet moved
to the typed specialist stores. It must not be treated as an authoritative
enrichment artifact.

All frame-aligned evidence preserves `video_id`, `frame_id`, `frame_idx`,
`timestamp_ms`, and `frame_store_id`. `frame_idx` is the competition-facing
coordinate; keyframe order and array position must never replace it.

The identity fields are required in specialist V1 Parquet. Existing Caption,
OCR, Object, or FrameContext artifacts without `timestamp_ms` (or OCR region
artifacts without `video_id`) must be regenerated or explicitly migrated before
they can be loaded.

## Run the V1 stages

Run commands from the repository root. Paths and versions shared by the
specialist stages are pinned in `configs/enrichment.yaml`.

```bash
# 1. Import organizer keyframes as the canonical frame store.
PYTHONPATH=.:src aic/bin/python scripts/ingest_btc_keyframes.py \
  --btc-root data \
  --data-root data \
  --output-root artifacts/frame_store \
  --frame-store-id btc-keyframes-v1

# 2. Generate captions.
PYTHONPATH=.:src aic/bin/python scripts/generate_enrichment.py \
  --config configs/enrichment.yaml

# 3. Generate structured OCR evidence.
PYTHONPATH=.:src aic/bin/python scripts/generate_ocr_enrichment.py \
  --config configs/enrichment.yaml

# 4. Import organizer-provided object detections; do not re-detect objects.
PYTHONPATH=.:src aic/bin/python scripts/generate_object_enrichment.py \
  --config configs/enrichment.yaml

# 5. Generate timestamped ASR segments from the source videos.
PYTHONPATH=.:src aic/bin/python scripts/prepare_transcripts.py \
  --config configs/enrichment.yaml \
  --videos-root data/videos

# 6. Build FrameContext only from the existing specialist artifacts.
PYTHONPATH=.:src aic/bin/python scripts/build_frame_context.py \
  --config configs/enrichment.yaml
```

If the videos live elsewhere, change only `--videos-root`. The V1 enrichment
sequence ends at FrameContext; retrieval/index construction belongs to a
separate plan.

## Public boundaries

- `src/hcmai/data/pipeline.py`: `DataService` imports the configured BTC frame
  store and exposes typed frame/evidence stores.
- `src/hcmai/data/enrichment/pipeline.py`: `EnrichmentService` runs Caption,
  OCR, Object Import, and FrameContext through independent stage boundaries.
- `src/hcmai/data/enrichment/transcripts/pipeline.py`: `TranscriptService`
  owns video-level ASR and diarization.
- `src/hcmai/data/stores/`: typed readers for specialist and derived artifacts.

Serving code reads prepared artifacts. It must not regenerate captions, OCR,
objects, transcripts, context, or other corpus-scale outputs in a request.
