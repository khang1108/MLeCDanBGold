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

`preprocess_videos.py` accepts either a local `videos_root` or an S3 source in
its YAML. For S3 it stages one object at a time, validates the complete local
FrameStore, publishes an immutable artifact version, writes `_SUCCESS.json`,
and advances `latest.json` last. See
`src/hcmai/data/preprocessing/README.md` and
`configs/preprocessing.s3.example.yaml`.

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

The current deterministic baseline after Plans 03–04 and the S3 preprocessing
transport is 367 passing backend tests with the same two optional PyAV skips,
plus 27 passing frontend tests and a successful production build. The focused
temporal/VQA/TRAKE/data-reliability gate contains 120 passing tests.

## HCMAI 2026 BTC-native enrichment V1

The active competition preparation path starts from BTC keyframes and BTC
objects. Video preprocessing remains available for non-BTC experiments, but it
is not part of this profile.

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

For the environment-driven local-to-A6000 transfer commands and safe staged
workflow, follow
[`docs/runbooks/thundercompute-index-build.md`](../docs/runbooks/thundercompute-index-build.md).

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
