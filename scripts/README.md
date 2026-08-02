# Root scripts

These thin CLIs call reusable code from `src/hcmai`. Run them from the
repository root with the `aic/` virtual environment.

## Build embeddings and index

`build_embeddings.py` resolves canonical relative image paths against
`--dataset-root`, generates normalized visual embeddings, writes their exact
frame mapping, and builds the FAISS index.

```bash
PYTHONPATH=src aic/bin/python scripts/build_embeddings.py \
  --config configs/baseline.yaml \
  --model-config llm/config.yaml \
  --dataset-root data \
  --frames data/metadata/frames.parquet \
  --output artifacts
```

Outputs:

```text
artifacts/embeddings/visual_embeddings.npy
artifacts/embeddings/frame_mapping.parquet
artifacts/embeddings/metadata.yaml
artifacts/indexes/visual/dense.index
artifacts/indexes/visual/frame_mapping.parquet
artifacts/indexes/visual/metadata.json
```

The dataset and index paths come from `configs/baseline.yaml`; the visual
checkpoint comes from `llm/config.yaml`. The first non-empty batch loads that
checkpoint. Ensure the machine has enough memory and that the checkpoint is
already cached or network access is available.

## Generate captions

Caption generation reads dataset, model, decoding, and output settings from
`configs/enrichment.yaml` by default:

```bash
PYTHONPATH=src aic/bin/python scripts/generate_enrichment.py
```

When `inference.enabled` is true in `configs/baseline.yaml`, the command sends
bounded JPEG batches to the hosted `/v1/captions` endpoint. It validates the
hosted checkpoint and immutable revision before writing resumable local
artifacts. Set `HCMAI_INFERENCE_BASE_URL` to override the configured URL.
The `Generating captions` progress bar includes already completed frames when
resuming and reports the current failure count.

Pass `--config`, `--frames`, `--dataset-root`, or `--output` only when a run
needs to override those values.

## Build caption, OCR, and ASR text indexes

Build each frame-aligned text source with the BGE-M3 configuration:

```bash
PYTHONPATH=src aic/bin/python scripts/build_caption_index.py --source caption
PYTHONPATH=src aic/bin/python scripts/build_caption_index.py --source ocr
PYTHONPATH=src aic/bin/python scripts/build_caption_index.py --source asr
```

The command reads the selected enrichment and index paths from
`configs/baseline.yaml`, including `index.text_embedding_filenames`, and the
shared text encoder from `llm/config.yaml`. Override them only when needed:

```bash
PYTHONPATH=src aic/bin/python scripts/build_caption_index.py \
  --source asr \
  --config configs/baseline.yaml \
  --model-config llm/config.yaml \
  --enrichment artifacts/enrichment/asr/frame_enrichment.parquet \
  --frames data/metadata/frames.parquet \
  --output artifacts/indexes/asr
```

It writes the configured text embedding filename, `dense.index`,
`frame_mapping.parquet`, and `metadata.json` under the output directory.
With hosted inference enabled, all three text channels are embedded remotely in
batches of
at most 64 with the configured BGE-M3 encoder while all vectors, mappings, and
FAISS files remain local. Each source preserves the canonical frame mapping.
The ASR smoke test uses a tiny frame-aligned transcript artifact and does not
decode or generate transcripts for the full video corpus.

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

```bash
PYTHONPATH=src aic/bin/python scripts/build_benchmark.py \
  --config configs/baseline.yaml \
  --index artifacts/indexes/visual \
  --queries data/eval/queries.jsonl \
  --output runs/dense_model_comparison
```
