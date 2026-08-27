# HCMAI 2026 BTC-native data preparation

`hcmai.data` owns canonical frame ingestion, specialist evidence artifacts,
derived frame context, and typed data access. The active competition baseline
uses organizer-provided BTC keyframes and offline YOLOE object enrichment. The
custom raw-video path is
a separately invoked offline corpus with its own run root and frame_store_id;
it does not overwrite or replace BTC preparation.

## Competition flow

```text
BTC keyframes ──> Caption ──────┐
              └─> OCR ──────────┤
              └─> YOLOE Objects ├─> FrameContext V1
Videos ─────────> ASR segments  │   (ASR excluded)
                                 └─> specialist artifacts
```

Caption, OCR, YOLOE object detection, and ASR produce independent specialist
artifacts.
FrameContext V1 is a deterministic derived view of Caption, usable normalized
OCR, and the object summary, in that order. ASR remains timestamped timeline
evidence and is intentionally absent from FrameContext dependency identity and
text.

`src/hcmai/data/preprocessing/**` remains available for non-BTC experiments.
The custom raw-video corpus instead uses the isolated C++17/FFmpeg package at
`src/hcmai/data/cpp/keyframes_extraction/` and the validation boundary under
`hcmai.data.ingestion`.

## Artifact ownership

The specialist outputs remain independently inspectable. A context or legacy
projection never replaces them.

| Artifact | Role |
| --- | --- |
| `captions.parquet` | Caption source of truth |
| `ocr/frames.parquet` | Frame-level OCR source of truth |
| `ocr/regions.parquet` | Raw OCR regions and boxes; source of truth |
| `objects/frames.parquet` | Frame-level object counts/summary; source of truth |
| `objects/detections.parquet` | Every valid YOLOE detection and box; source of truth |
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

# 4. Run YOLOE object detection and publish canonical object artifacts.
PYTHONPATH=.:src aic/bin/python scripts/detect_objects.py \
  --frames artifacts/frame_store/frames.parquet \
  --output artifacts/enrichment/objects_yoloe \
  --dataset-root data

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

## Custom raw-video 1-FPS workflow

This optional offline path retains a separate custom corpus and never modifies
BTC frames, BTC mappings, or the BTC frame_store_id. It uses the organizer
competition coordinate exactly as
`floor(ceil(avg_fps) * timestamp_ms / 1000)`, while its `frame_id` is an
internal `{video_id}_raw1fps_{sample_index}` identity.

```text
prepare_custom_extraction.py
  -> native extract
  -> validate staging frames.jsonl
  -> per-video Caption / Objects / visual on durable JPEGs
  -> per-video OCR on temporary high-resolution JPEGs
  -> per-video ASR on retained source video
  -> write_enrichment_handoff
  -> native state mark-enriched
  -> native state mark-published
  -> native cleanup
  -> materialize corpus/frames.parquet
  -> build FrameContext, embeddings, and indexes
```

Prepare metadata and strict native settings without downloading a video:

```bash
PYTHONPATH=.:src aic/bin/python scripts/prepare_custom_extraction.py \
  --media-info-dir data/media-info-aic25-b1/media-info \
  --run-root runs/custom-raw1fps-v1 \
  --native-executable build/keyframes_extraction/keyframe_extractor \
  --frame-store-id custom-raw1fps-v1 \
  --yt-dlp-binary yt-dlp
```

Run the native extractor only after the local/release gate is accepted:

```bash
build/keyframes_extraction/keyframe_extractor extract \
  --manifest runs/custom-raw1fps-v1/input/media_manifest.jsonl \
  --run-root runs/custom-raw1fps-v1 \
  --config runs/custom-raw1fps-v1/input/extraction_config.json \
  --video-id L01_V001 \
  --fail-fast
```

`materialize_video_enrichment_frames(..., image_variant="durable")` supplies
Caption, Objects, and visual stages. Its `image_variant="enrichment"` output
is OCR-only scratch data and must not be sent to the global materializer.
After the handoff validator accepts all specialist artifacts, call the Python
native-state wrappers rather than editing `state/{video_id}.json` directly.
`cleanup_video` is allowed only after native publication and retains
`published/{video_id}` durable JPEGs and manifest while removing its temporary
`published/{video_id}/enrichment_images` OCR scratch directory.

### Local release gate

Before downloading any organizer video, run the complete synthetic gate. It
builds a three-second local source and exercises extraction, image validation,
identity-only enrichment handoff validation, guarded publication, scoped
cleanup, and final FrameStore loading. It never invokes yt-dlp, models, or a
remote provider.

```bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction --output-on-failure
PYTHONPATH=.:src aic/bin/python -m pytest -q \
  tests/data/test_custom_manifest.py \
  tests/data/test_custom_frames.py \
  tests/data/test_custom_enrichment.py \
  tests/data/test_custom_state.py \
  tests/scripts/test_custom_extraction_cli.py
PYTHONPATH=.:src aic/bin/python -m compileall -q src/hcmai/data/ingestion scripts
```

Passing this gate is a code-correctness prerequisite, not an approval for an
unbounded corpus run. A separate bounded pilot must measure download/decode,
storage, enrichment quality, and cleanup behaviour before selecting the full
corpus.

Finally materialize only selected validated published bundles:

```bash
PYTHONPATH=.:src aic/bin/python scripts/materialize_custom_frames.py \
  --run-root runs/custom-raw1fps-v1 \
  --output-root runs/custom-raw1fps-v1/corpus \
  --frame-store-id custom-raw1fps-v1 \
  --video-id L01_V001
```

Custom `FrameRecord.image_path` values are relative to the custom run root, so
downstream image consumers must configure that run root as their dataset root.

## Public boundaries

- `src/hcmai/data/pipeline.py`: `DataService` imports the configured BTC frame
  store and exposes typed frame/evidence stores.
- `src/hcmai/data/enrichment/pipeline.py`: `EnrichmentService` runs Caption,
  OCR, YOLOE Object Detection, and FrameContext through independent stage boundaries.
- `src/hcmai/data/enrichment/transcripts/pipeline.py`: `TranscriptService`
  owns video-level ASR and diarization.
- `src/hcmai/data/stores/`: typed readers for specialist and derived artifacts.
- `src/hcmai/data/ingestion/custom_state.py`: safe Python argv wrappers for the
  native `mark-enriched`, `mark-published`, and `cleanup` lifecycle commands.

Serving code reads prepared artifacts. It must not regenerate captions, OCR,
objects, transcripts, context, or other corpus-scale outputs in a request.
