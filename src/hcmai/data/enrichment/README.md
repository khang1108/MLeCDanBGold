# BTC-native media enrichment

`hcmai.data.enrichment` owns independently runnable Caption, OCR, BTC Object
Import, ASR, and deterministic FrameContext stages. Model-specific code stays
behind feature-local adapters; production scripts call the public service
boundaries.

```text
BTC keyframes ──> Caption ──────┐
              └─> OCR ──────────┤
BTC objects ────> Object Import ├─> FrameContext V1
Videos ─────────> ASR segments  │   (ASR excluded)
                                 └─> specialist artifacts
```

`src/hcmai/data/preprocessing/**` supports non-BTC experiments only. The
competition profile starts from the BTC frame store and never invokes that
preprocessing path.

## Layout and ownership

```text
enrichment/
├── pipeline.py              # Caption/OCR/Object/Context service boundary
├── caption/                 # caption generation, resume, and artifacts
├── ocr/                     # raw regions plus derived normalized frame text
├── objects/                 # lossless BTC detection import and summary
├── context/                 # deterministic Caption/OCR/Object serializer
└── transcripts/             # timestamped ASR/diarization timeline evidence
```

The source-of-truth and derived outputs are:

```text
captions.parquet                source of truth
ocr/frames.parquet             source of truth
ocr/regions.parquet            source of truth
objects/frames.parquet         source of truth
objects/detections.parquet     source of truth
transcripts/*.parquet          source of truth
context/frame_context_v1.parquet derived cross-modal view
frame_enrichment.parquet       temporary compatibility projection only
```

Stage-local `manifest.json` and failure reports carry version and resume
metadata. `frame_enrichment.parquet` may be emitted beside a specialist output
for existing retrieval consumers, but it is never the source of truth.

Every frame and flat region/detection row carries exact `frame_id`, `video_id`,
`frame_idx`, and `timestamp_ms`. These fields are required: specialist V1
Parquet created before this identity contract must be regenerated or migrated.

## Commands

Run the complete V1 preparation sequence from the repository root:

```bash
PYTHONPATH=src aic/bin/python scripts/ingest_btc_keyframes.py \
  --btc-root data --data-root data \
  --output-root artifacts/frame_store \
  --frame-store-id btc-keyframes-v1

PYTHONPATH=src aic/bin/python scripts/generate_enrichment.py \
  --config configs/enrichment.yaml

PYTHONPATH=src aic/bin/python scripts/generate_ocr_enrichment.py \
  --config configs/enrichment.yaml

PYTHONPATH=src aic/bin/python scripts/generate_object_enrichment.py \
  --config configs/enrichment.yaml

PYTHONPATH=src aic/bin/python scripts/prepare_transcripts.py \
  --config configs/enrichment.yaml --videos-root data/videos

PYTHONPATH=src aic/bin/python scripts/build_frame_context.py \
  --config configs/enrichment.yaml
```

Caption and OCR may use local pinned models or configured hosted adapters.
Object Import always reads BTC JSON and does not run detection. FrameContext
only reads existing specialist parquet files and never triggers upstream model
inference. Index building is outside Enrichment V1.

## Resume and dependency rules

- Completed compatible Caption/OCR rows are skipped; failed or incomplete rows
  are retried without rewriting canonical identity.
- Object Import preserves every valid raw detection and bounding box. Derived
  counts may apply the configured confidence threshold without deleting raw
  rows.
- Context rebuild identity includes Caption, OCR, Object, frame-store lineage,
  and serializer policy. It excludes ASR.
- Changing only a Context policy rebuilds Context without touching specialist
  artifacts. Changing an OCR artifact version invalidates Context while leaving
  Caption, Object, and ASR artifacts untouched.

## Verification

```bash
pyright src/hcmai/common/schemas src/hcmai/data/enrichment \
  src/hcmai/data/stores src/hcmai/data/pipeline.py
PYTHONPATH=src aic/bin/python -m pytest tests/data tests/data/enrichment -q
```

Real-corpus artifacts are local experiment evidence and must not be committed.
