# VBS local pipeline refactor v2

This refactor targets the 1,000-video V3C paper run with source MP4 files on HDD and all generated artifacts on local SSD storage.

## Storage boundary

- Read-only videos: `/home/phuckhang/hdd/V3C/videos`
- Generated frames/artifacts/indexes/state: `./data`
- No S3, archive download, publication, or public media URLs are used by the canonical V3C path.

## Inference boundary

- Core API: `https://api.iamphuckhang.dev`
  - `/v1/embeddings/images`
  - `/v1/embeddings/text`
  - `/v1/chat/completions`
- GPU API: `https://gpu.iamphuckhang.dev`
  - `/v1/captions`
  - `/v1/enrichment/ocr`
  - `/v1/enrichment/objects`
  - `/v1/transcripts/asr-file`

The GPU server uses `HCMAI_LLM_PROFILE=gpu` plus `HCMAI_GPU_TASK=caption|ocr|objects|asr`, so one A6000 only loads the capability needed for the current stage.

## Fast paper profile

Caption generation defaults to `Qwen/Qwen3-VL-2B-Instruct`, 512 px input, deterministic decoding, and 64 output tokens. The client batch size is 16.

## Canonical data preparation modules

- `offline/corpus/`: external video scanning and frame extraction
- `offline/clients/`: core/GPU endpoint resolution
- `offline/pipeline/`: stage runner and local state journal
- `llm/contracts/`: HTTP transport contracts owned by the serving layer
- `scripts/prepare.py`: canonical stage CLI

`offline/ingestion/` and `offline/vbs/` remain only for compatibility during the paper run and are not the canonical V3C execution path.

## Canonical CLI

```bash
PYTHONPATH=.:src python -m scripts.prepare frames
PYTHONPATH=.:src python -m scripts.prepare visual-index
PYTHONPATH=.:src python -m scripts.prepare caption
```

See `docs/VBS_LOCAL_PIPELINE_V2.md` for deployment and stage details.
