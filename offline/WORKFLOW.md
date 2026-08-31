# Offline ingestion and enrichment workflow

This guide records ownership and dependency direction for construction of the
existing HCMAI artifacts. It does not define a new artifact layout.

## Ownership flow

```text
BTC keyframes and mappings
  -> offline.ingestion
  -> canonical frame artifact

canonical frames
  -> offline.enrichment.caption
  -> offline.enrichment.ocr
  -> offline.enrichment.objects
  -> offline.enrichment.context

source videos
  -> offline.enrichment.transcripts

canonical frames + completed evidence
  -> offline.embeddings / offline.indexes
  -> runtime-compatible indexes
```

Caption, OCR, object, transcript, and FrameContext artifacts remain separate.
FrameContext is a deterministic derived view; it does not replace specialist
evidence. ASR remains segment-native timeline evidence, and any frame-aligned
projection is derived compatibility data.

All frame-aligned producers preserve `video_id`, `frame_id`, `frame_idx`, and
`timestamp_ms`. `frame_idx` is the competition-facing coordinate; keyframe
order and array position must not replace it. Missing and not-evaluated
evidence remain distinguishable from a completed empty result.

## Public boundaries

- `offline.ingestion` validates organizer inputs and publishes canonical frame
  metadata.
- `offline.enrichment` constructs specialist evidence and deterministic
  context artifacts.
- `offline.embeddings` constructs embedding matrices and canonical mappings.
- `offline.indexes` constructs indexes compatible with runtime retrieval.
- `offline.keyframes` owns the optional native custom-video extractor.
- `hcmai.corpus.Corpus.open(...)` reads existing artifacts for serving.
- `hcmai.retrieval.RetrievalService` loads and searches existing indexes.

Offline modules may reuse runtime-compatible value types and index loaders,
but they must not depend on runtime HTTP or application-composition packages.
Runtime modules must not depend on offline producers.

## Commands

Run commands from the repository root. Their defaults remain in
`configs/prepare.yaml`; use `--help` to inspect inputs without starting model
work.

```bash
PYTHONPATH=.:src aic/bin/python scripts/ingest_btc_keyframes.py --help
PYTHONPATH=.:src aic/bin/python scripts/generate_enrichment.py --help
PYTHONPATH=.:src aic/bin/python scripts/generate_ocr_enrichment.py --help
PYTHONPATH=.:src aic/bin/python scripts/detect_objects.py --help
PYTHONPATH=.:src aic/bin/python scripts/prepare_transcripts.py --help
PYTHONPATH=.:src aic/bin/python scripts/build_frame_context.py --help
PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py --help
```

The optional custom-video extractor is configured and built from
`offline/keyframes/keyframes_extraction`. Serving never invokes these commands
or rebuilds their outputs.
