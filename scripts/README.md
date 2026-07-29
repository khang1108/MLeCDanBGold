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

Pass `--config`, `--frames`, `--dataset-root`, or `--output` only when a run
needs to override those values.

## Build caption embeddings and index

After caption generation completes, build the caption retrieval artifacts:

```bash
PYTHONPATH=src aic/bin/python scripts/build_caption_index.py
```

The command reads caption/frame/index paths from `configs/baseline.yaml` and
the caption encoder from `llm/config.yaml`. Override them only when needed:

```bash
PYTHONPATH=src aic/bin/python scripts/build_caption_index.py \
  --config configs/baseline.yaml \
  --model-config llm/config.yaml \
  --captions artifacts/enrichment/caption/frame_enrichment.parquet \
  --frames data/metadata/frames.parquet \
  --output artifacts/indexes/caption
```

It writes `caption_embeddings.npy`, `dense.index`,
`frame_mapping.parquet`, and `metadata.json` under the output directory.

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
