# VBS / V3C local-first preparation v2

## Storage boundary

Source videos are read-only and stay on the HDD:

```text
/home/phuckhang/hdd/V3C/videos/*.mp4
```

Generated data stays under the repository-side work root (SSD):

```text
data/
├── frames/
├── artifacts/
├── indexes/
└── state/
```

No VBS stage downloads, copies, publishes, or uploads the corpus through S3.
Only inference requests cross the network boundary.

## Model API boundary

`https://api.iamphuckhang.dev` is the core API:

- `POST /v1/embeddings/images`
- `POST /v1/embeddings/text`
- `POST /v1/chat/completions` when text generation is explicitly enabled

`https://gpu.iamphuckhang.dev` is the single-A6000 API:

- `POST /v1/captions`
- `POST /v1/enrichment/ocr`
- `POST /v1/enrichment/objects`
- `POST /v1/transcripts/asr-file`

Both roots are configured in `configs/vbs_prepare.yaml`. Override them without
editing YAML using `VBS_CORE_API_URL` and `VBS_GPU_API_URL`.

## Single A6000 model lifecycle

The GPU app deliberately enables one heavy task per process. Start the task you
need, point the `gpu.iamphuckhang.dev` tunnel at port 8200, run that offline
stage, then restart with the next task.

```bash
HCMAI_GPU_TASK=caption llm/scripts/run_gpu.sh
HCMAI_GPU_TASK=ocr     llm/scripts/run_gpu.sh
HCMAI_GPU_TASK=objects llm/scripts/run_gpu.sh
HCMAI_GPU_TASK=asr     llm/scripts/run_gpu.sh
```

The paper caption profile uses `Qwen/Qwen3-VL-2B-Instruct`, deterministic
decoding, `max_new_tokens: 64`, and offline batch size 16.

## Core API

Start the embedding service (default port 8100):

```bash
llm/scripts/run_core.sh
```

The core profile enables visual SigLIP and BGE text embeddings. Text generation
stays disabled unless explicitly enabled, so a CPU core host does not
accidentally load the CUDA text-generation model.

## Canonical preparation CLI

Always run from the repository root:

```bash
PYTHONPATH=.:src python -m scripts.prepare frames
PYTHONPATH=.:src python -m scripts.prepare visual-index
PYTHONPATH=.:src python -m scripts.prepare caption
PYTHONPATH=.:src python -m scripts.prepare ocr
PYTHONPATH=.:src python -m scripts.prepare objects
PYTHONPATH=.:src python -m scripts.prepare asr
PYTHONPATH=.:src python -m scripts.prepare context
PYTHONPATH=.:src python -m scripts.prepare context-index
PYTHONPATH=.:src python -m scripts.prepare asr-index
```

For the earliest usable paper/demo baseline while the caption worker is active:

```bash
PYTHONPATH=.:src python -m scripts.prepare paper-fast
```

This runs:

```text
frames -> visual-index -> caption
```

The visual index no longer waits for Context or ASR preflight.

## Resume

Frame extraction checkpoints each video under `data/state/frames/`. Specialist
writers keep their existing artifact-level resume logic. The orchestration
journal is `data/state/pipeline.json`; it is diagnostic only, while completed
artifacts remain authoritative.

## Legacy code

`offline/ingestion/` remains temporarily for compatibility with the old
archive/S3/BTC path. The V3C local pipeline does not use its archive, S3,
media-info, yt-dlp, staging, or publication orchestration. Shared historical
imports are being kept as shims until the paper run is stable.
