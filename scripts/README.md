# Root scripts

These thin CLIs call reusable code from `src/hcmai`. Run them from the
repository root with the `aic/` virtual environment.

Scripts use the owning public service in each component's `pipeline.py`:
`DataService`, `EmbeddingService`, `EnrichmentService`, `RetrievalService`, or
`TranscriptService`. They must not compose another component's adapters,
stores, or generators directly.

`prepare_transcripts.py` additionally validates source/config/model manifests,
supports explicitly disabled diarization, and materializes frame-aligned ASR
through the canonical `FrameEnrichment`/`ASRStore` contract.

The root `thundercompute/` directory is the shared hosted-inference component.
See [`thundercompute/README.md`](../thundercompute/README.md) for its endpoint
contracts and the manual `tnr create -> scp -> connect -> run -> delete` flow.

## Validate the repository

There is currently no maintained release-wrapper script. Run the maintained
checks explicitly from the repository root and investigate or record every
non-zero result:

```bash
PYTHONPATH=.:src aic/bin/python -m compileall -q src/hcmai thundercompute
PYTHONPATH=.:src aic/bin/python -m pytest -q
docker compose config --quiet
CI=true npm --prefix frontend test -- --watchAll=false --runInBand
npm --prefix frontend run build
git diff --check
```

These checks cover Python imports, the complete backend suite, Compose
rendering, the complete frontend suite, the frontend production build, and
whitespace validation. Tests use local fixtures and deterministic fakes; the
verification commands must not invoke remote inference or rebuild corpus
artifacts.

## HCMAI 2026 BTC-native enrichment V1

The active competition preparation path starts from BTC keyframes and YOLOE
object enrichment. Custom video frame extraction is not part of this profile.

```text
BTC keyframes ──> Caption ──────┐
              └─> OCR ──────────┤
              └─> YOLOE Objects ├─> FrameContext V1
Videos ─────────> ASR segments  │   (ASR excluded)
                                 └─> specialist artifacts
```

Run these commands from the repository root:

```bash
# BTC frame-store ingest
PYTHONPATH=.:src aic/bin/python scripts/ingest_btc_keyframes.py \
  --btc-root data --data-root data \
  --output-root artifacts/frame_store \
  --frame-store-id btc-keyframes-v1

# Caption
PYTHONPATH=.:src aic/bin/python scripts/generate_enrichment.py \
  --config configs/enrichment.yaml

# OCR
PYTHONPATH=.:src aic/bin/python scripts/generate_ocr_enrichment.py \
  --config configs/enrichment.yaml

# YOLOE objects; publishes raw JSON plus canonical object Parquet artifacts.
PYTHONPATH=.:src aic/bin/python scripts/detect_objects.py \
  --frames artifacts/frame_store/frames.parquet \
  --output artifacts/enrichment/objects_yoloe \
  --dataset-root data

# Timestamped ASR segments; change data/videos if the source lives elsewhere.
PYTHONPATH=.:src aic/bin/python scripts/prepare_transcripts.py \
  --config configs/enrichment.yaml --videos-root data/videos

# Deterministic Caption + OCR + Object context; ASR is excluded.
PYTHONPATH=.:src aic/bin/python scripts/build_frame_context.py \
  --config configs/enrichment.yaml
```

The authoritative and compatibility outputs are:

```text
captions.parquet                  source of truth
ocr/frames.parquet               source of truth
ocr/regions.parquet              source of truth
objects/frames.parquet           source of truth
objects/detections.parquet       source of truth
transcripts/*.parquet            source of truth
context/frame_context_v1.parquet derived cross-modal view
frame_enrichment.parquet         temporary compatibility projection only
```

This V1 sequence intentionally ends at FrameContext. It does not build a
retrieval index.

## YOLOE object detection

`detect_objects.py` is the object-enrichment entry point. It runs YOLOE over
canonical frames, stores resumable raw JSON under `<output>/raw/`, and commits
the canonical `frames.parquet`, `detections.parquet`, and `manifest.json`
bundle in the same output directory. No second importer step is required.

Install the official Ultralytics headless variant through the `objects` extra.
The standard `ultralytics` package declares `opencv-python`; this server
workflow intentionally resolves `ultralytics-opencv-headless`, which exposes
the same Ultralytics API without requiring a manual uninstall/reinstall swap:

```bash
aic/bin/python -m pip install -e ".[objects]"
```

```bash
# Detect and publish; the run resumes from raw JSON. --limit is a degraded
# smoke artifact and should not be used for a final corpus.
PYTHONPATH=src aic/bin/python scripts/detect_objects.py --limit 200
PYTHONPATH=src aic/bin/python scripts/detect_objects.py

# Context rebuild; the wider budget holds a finer-grained label vocabulary
PYTHONPATH=src aic/bin/python scripts/build_frame_context.py \
  --config configs/enrichment.yaml --object-token-budget 80 \
  --object-frames artifacts/enrichment/objects_yoloe/frames.parquet \
  --output artifacts/enrichment/context_yoloe
```

## Fast-track multimodal index build

`build_retrieval_indexes.py` is the explicit offline entry point for the
competition retrieval profile. It consumes the dedicated
`configs/indexing.yaml` and `configs/indexing.models.yaml` files and publishes
validated `Visual + FrameContext + segment-native ASR` bundles. The stages are
independently runnable for diagnosis:

```bash
PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --stage preflight \
  --config configs/indexing.yaml \
  --model-config configs/indexing.models.yaml

PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --stage all \
  --config configs/indexing.yaml \
  --model-config configs/indexing.models.yaml
```

Preflight requires the external BTC `map_keyframes` CSV directory and writes
only an indexing-specific keyframe-path projection; it never rewrites the
canonical frame store. Heavy stages run sequentially and validation writes
`artifacts/indexes/build_report.json` only after all three checksummed bundles
pass identity and encoder-contract checks.

To offload only SigLIP and BGE inference to an already running private GPU
service, pass its URL explicitly. The builder intentionally does not infer a
remote endpoint from environment variables. It checks `/ready` and the exact
pinned model/revisions before it starts an embedding stage, while local file
I/O, FAISS publication, and validation remain local:

```bash
PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --stage all \
  --config configs/indexing.yaml \
  --model-config configs/indexing.models.yaml \
  --inference-url "$HCMAI_INFERENCE_BASE_URL"
```

Export the appropriate Cloudflare Access client credentials for the HTTP client
separately; do not place them in a config file or command history.

For the environment-driven local-to-A6000 transfer commands and safe staged
workflow, follow
[`docs/runbooks/thundercompute-index-build.md`](../docs/runbooks/thundercompute-index-build.md).

### Build directly from S3 on ThunderCompute

When the canonical inputs and enrichment artifacts are already in S3, use the
explicit S3 mode instead of copying the full repository over SSH. It downloads
only the five required prefixes (including the BTC mapping), resumes files
whose byte size already matches, runs the local SigLIP+BGE build, and uploads
only a passed Visual + Context + segment-ASR bundle under an immutable version.
The `latest.json` pointer is advanced last, so an interrupted build is never
advertised as serving data:

```bash
PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py \
  --s3 \
  --stage all \
  --config configs/indexing.yaml \
  --model-config configs/indexing.models.yaml \
  --s3-config configs/preparation.s3.yaml \
  --s3-sync-workers 16 \
  --s3-upload-workers 8
```

The local model config starts both SigLIP and BGE at batch size `128`, which is
the A6000 starting point. The command intentionally does not accept
`--inference-url` in S3 mode: inference and FAISS/index publication happen on
the ThunderCompute GPU, while S3 is used only for input/output transfer. Use
`--s3-dry-run` first to list the remote inputs without downloading them.

The default S3 mappings are:

```text
data/keyframes/                         -> data/keyframes/
data/features/map-keyframes/             -> data/map_keyframes/
data/artifacts/frame_store/              -> artifacts/frame_store/
data/artifacts/enrichment/context/       -> artifacts/enrichment/context/
data/artifacts/enrichment/transcripts/   -> artifacts/enrichment/transcripts/
```

Raw videos and unrelated enrichment/index prefixes are not downloaded. AWS
credentials remain outside the repository and are resolved through boto3's
standard credential chain.

## Separate retrieval utilities

The commands below are retained for retrieval development and are not part of
the BTC-native Enrichment V1 preparation sequence.

## Rebuild only the index

```bash
PYTHONPATH=.:src aic/bin/python scripts/build_index.py \
  --config configs/baseline.yaml \
  --model-config thundercompute/config.yaml \
  --embeddings artifacts/embeddings/visual_embeddings.npy \
  --mapping artifacts/embeddings/frame_mapping.parquet \
  --output artifacts/indexes/visual
```
