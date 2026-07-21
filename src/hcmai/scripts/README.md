# Scripts

This package holds the `build_*` command-line entry points that drive the
offline pipeline. Each script contains only its run logic; all argument parsing
lives in one place so the scripts stay thin.

These are source entry points, not run outputs — the evaluation artifacts they
produce belong under `runs/` and `artifacts/`, which are not committed to Git.

## Argument parsing

`parse.py` exposes a single `parse_arguments(command)` function that builds the
right parser for each command. Every script shares `--config` and `--log-level`;
the rest of the arguments are added per command.

## Entry points

Run each script as a module with `PYTHONPATH=src`.

### `build_embeddings.py`

Generate the frame-embedding corpus with `EmbeddingPipeline`.

```bash
PYTHONPATH=src python -m hcmai.scripts.build_embeddings \
    --config configs/baseline.yaml \
    --frames data/metadata/frames.parquet \
    --output artifacts
```

### `build_index.py`

Build an exact FAISS `IndexFlatIP` from generated embeddings and their mapping.

```bash
PYTHONPATH=src python -m hcmai.scripts.build_index \
    --config configs/baseline.yaml \
    --embeddings artifacts/embeddings/visual_embeddings.npy \
    --mapping artifacts/embeddings/frame_mapping.parquet \
    --output artifacts/indexes
```

### `build_benchmark.py`

Benchmark the dense-retrieval baseline on labelled `EvaluationQuery` records and
freeze `config.yaml`, `metrics.json`, `per_query.csv`, and `failures.csv`.

```bash
PYTHONPATH=src python -m hcmai.scripts.build_benchmark \
    --config configs/baseline.yaml \
    --index artifacts/indexes \
    --queries data/eval/queries.jsonl \
    --output runs/dense_model_comparison
```

The pipeline logic these scripts invoke lives in the
[embedding](../embedding) and [retriever](../retriever) packages.
