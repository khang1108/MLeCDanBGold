> **Legacy note:** This document describes the first local refactor. For the current two-domain V3C pipeline, use `docs/VBS_LOCAL_PIPELINE_V2.md`.

# VBS local data layout

The VBS preparation path has one local source of truth:

```text
data/
├── videos/                  # only input: <video_id>.mp4
├── frames/                  # sampled JPEGs
│   └── <video_id>/
│       ├── 000000.jpg
│       └── ...
├── artifacts/
│   ├── frames.parquet       # canonical frame identity
│   ├── manifest.json
│   ├── captions/
│   ├── ocr/
│   ├── objects/
│   ├── asr/
│   └── context/
├── indexes/
│   ├── visual/
│   ├── context/
│   └── asr/
└── state/                   # resumable local checkpoints only
```

There are intentionally no `archives`, `source`, `staging`, `published`, S3,
`media-info`, or yt-dlp directories in this execution path.

## Local/remote boundary

The data machine owns filesystem I/O, ffmpeg sampling/audio extraction,
validation, checkpointing, artifact materialization, and FAISS publication.
Caption, OCR, YOLOE object detection, ASR, and dense embedding inference are
performed through the model API. No model stage needs an S3/public URL.

## Frame preparation

```bash
PYTHONPATH=.:src python -m scripts.corpus.prepare_vbs_frames --data data
```

The command scans only `data/videos/*.mp4`, extracts periodic JPEGs locally
using ffmpeg, and writes `data/artifacts/frames.parquet`. `video_id` is exactly
the MP4 filename stem. Frame images use short sample names, while the Parquet
artifact retains `video_id`, `frame_id`, `frame_idx`, `timestamp_ms`, `fps`, and
`image_path` for retrieval and VBS/DRES submission.

## Remote enrichment

Point all model clients at the same private inference service:

```bash
export HCMAI_INFERENCE_BASE_URL=http://127.0.0.1:8100
```

Then the VBS-specific remote stages are:

```bash
# Caption (existing remote adapter)
PYTHONPATH=.:src python -m scripts.enrichment.generate_enrichment \
  --config configs/vbs_prepare.yaml --execution-backend remote

# OCR (existing remote adapter)
PYTHONPATH=.:src python -m scripts.enrichment.generate_ocr_enrichment \
  --config configs/vbs_prepare.yaml

# Objects (new direct image-upload adapter)
PYTHONPATH=.:src python -m scripts.enrichment.prepare_vbs_objects \
  --config configs/vbs_prepare.yaml --data data

# ASR (new direct FLAC-upload adapter; no S3/audio URL)
PYTHONPATH=.:src python -m scripts.enrichment.prepare_vbs_transcripts \
  --config configs/vbs_prepare.yaml --data data

# Deterministic local context
PYTHONPATH=.:src python -m scripts.enrichment.build_frame_context \
  --config configs/vbs_prepare.yaml
```

The model server enables only the capabilities it should host, for example:

```bash
HCMAI_ENABLE_CAPTION=true \
HCMAI_ENABLE_OCR=true \
HCMAI_ENABLE_OBJECTS=true \
HCMAI_ENABLE_ASR=true \
python -m llm.main
```

Use separate model-server processes if the checkpoints do not fit on one GPU;
the local artifact layout does not change.

## Indexes

`build_retrieval_indexes.py` already supports remote SigLIP/BGE inference and
local index publication. In VBS mode, never pass `--s3`; publish only to:

- `data/indexes/visual`
- `data/indexes/context`
- `data/indexes/asr`

The index bundles already contain their embedding arrays, so there is no
separate top-level `data/embeddings/` directory.
