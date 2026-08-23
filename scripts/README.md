# Root scripts

These thin CLIs call reusable code from `src/hcmai`. Run them from the
repository root with the `aic/` virtual environment.

Scripts use the owning public service in each component's `pipeline.py`:
`DataService`, `EmbeddingService`, `EnrichmentService`, `RetrievalService`, or
`TranscriptService`. They must not compose another component's adapters,
stores, or generators directly. `build_benchmark.py` is the sole deliberate
exception: it uses the internal dense-baseline `RetrievalBenchmark`, which is
not an end-to-end `EvaluationService`.

`prepare_transcripts.py` additionally validates source/config/model manifests,
supports explicitly disabled diarization, and materializes frame-aligned ASR
through the canonical `FrameEnrichment`/`ASRStore` contract.

The `thundercompute/` subdirectory is the exception to the data CLI pattern:
it owns the laptop-side GPU lifecycle controller. See
[`thundercompute/README.md`](thundercompute/README.md) for the
`tnr create -> scp -> SSH deploy -> delete` flow and secret boundary.

## Validate the repository

Run the complete deterministic release gate from the repository root:

```bash
scripts/validate_repository.sh
```

The command runs focused temporal/VQA/TRAKE tests, the complete backend suite,
frontend tests, the frontend production build, and whitespace validation in
that order. It uses `aic/bin/python` when available; set `HCMAI_PYTHON` to an
alternative interpreter when validating another supported environment.

Plan 01 deliberately retires three historical suites instead of skipping
them:

- `tests/test_minichallenge.py` covered removed KISC/MiniChallenge behavior;
- `tests/test_removed_conversation.py` asserted deleted conversation APIs;
- `tests/unit/evaluation/test_vqa_metrics.py` targeted an orphan evaluator with
  no runtime consumer; evaluator construction remains deferred.

All other active tests are tracked and collected. Tests use local fixtures and
deterministic fakes; this validation command must not invoke remote inference or
rebuild corpus artifacts.

The Plan 01 baseline recorded on 2026-08-13 is 324 passing backend tests with
two deterministic skips, plus 21 passing frontend tests and a successful
production build. Both backend skips generate tiny video fixtures and run when
the optional preprocessing extra supplies PyAV.

The current deterministic baseline after the BTC-native preparation migration
is 367 passing backend tests with the same two optional PyAV skips,
plus 27 passing frontend tests and a successful production build. The focused
temporal/VQA/TRAKE/data-reliability gate contains 120 passing tests.

## HCMAI 2026 BTC-native enrichment V1

The active competition preparation path starts from BTC keyframes and BTC
objects. Custom video frame extraction is not part of this profile.

```text
BTC keyframes ──> Caption ──────┐
              └─> OCR ──────────┤
BTC objects ────> Object Import ├─> FrameContext V1
Videos ─────────> ASR segments  │   (ASR excluded)
                                 └─> specialist artifacts
```

Run these commands from the repository root:

```bash
# BTC frame-store ingest
PYTHONPATH=src aic/bin/python scripts/ingest_btc_keyframes.py \
  --btc-root data --data-root data \
  --output-root artifacts/frame_store \
  --frame-store-id btc-keyframes-v1

# Caption
PYTHONPATH=src aic/bin/python scripts/generate_enrichment.py \
  --config configs/enrichment.yaml

# OCR
PYTHONPATH=src aic/bin/python scripts/generate_ocr_enrichment.py \
  --config configs/enrichment.yaml

# BTC Object Import
PYTHONPATH=src aic/bin/python scripts/generate_object_enrichment.py \
  --config configs/enrichment.yaml

# Timestamped ASR segments; change data/videos if the source lives elsewhere.
PYTHONPATH=src aic/bin/python scripts/prepare_transcripts.py \
  --config configs/enrichment.yaml --videos-root data/videos

# Deterministic Caption + OCR + Object context; ASR is excluded.
PYTHONPATH=src aic/bin/python scripts/build_frame_context.py \
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

## Fast-track multimodal index build

`build_retrieval_indexes.py` is the explicit offline entry point for the
competition retrieval profile. It consumes the dedicated
`configs/indexing.yaml` and `configs/indexing.models.yaml` files and publishes
validated `Visual + FrameContext + segment-native ASR` bundles. The stages are
independently runnable for diagnosis:

```bash
PYTHONPATH=src aic/bin/python scripts/build_retrieval_indexes.py \
  --stage preflight \
  --config configs/indexing.yaml \
  --model-config configs/indexing.models.yaml

PYTHONPATH=src aic/bin/python scripts/build_retrieval_indexes.py \
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
PYTHONPATH=src aic/bin/python scripts/build_retrieval_indexes.py \
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
PYTHONPATH=src aic/bin/python scripts/build_retrieval_indexes.py \
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
PYTHONPATH=src aic/bin/python scripts/build_index.py \
  --config configs/baseline.yaml \
  --model-config llm/config.yaml \
  --embeddings artifacts/embeddings/visual_embeddings.npy \
  --mapping artifacts/embeddings/frame_mapping.parquet \
  --output artifacts/indexes/visual
```

## Benchmark retrieval

`build_benchmark.py` is a legacy dense-baseline harness, not the selected
competition evaluator. It still expects `models.embedding` inside the YAML
passed as `--config`; the current `configs/baseline.yaml` no longer owns that
checkpoint. Do not treat the command below as runnable with the current
baseline config until the script is migrated to `llm/config.yaml`.

```bash
PYTHONPATH=src aic/bin/python scripts/build_benchmark.py \
  --config path/to/legacy-benchmark-config.yaml \
  --index artifacts/indexes/visual \
  --queries data/eval/queries.jsonl \
  --output runs/dense_model_comparison
```
