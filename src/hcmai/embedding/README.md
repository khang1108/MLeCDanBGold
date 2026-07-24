# Embedding

This package generates the offline visual-embedding corpus that the
[retriever](../retriever) indexes and searches. It reads canonical frame
metadata, resolves relative image paths against the dataset root, encodes each
frame with the dense encoder, and writes versioned artifacts.

## Dependencies

```bash
aic/bin/python -m pip install -e ".[embedding]"
```

`numpy`, `pandas`, and a Parquet engine such as `pyarrow` are always needed;
`transformers`, `torch`, and `pillow` are used through `DenseEncoder`.

## Pipeline

`embedding.py` provides `EmbeddingPipeline`. Given a `frames.parquet` of
`FrameRecord` rows, a dataset root, and an `EncoderConfig`, `run()` resolves
and encodes images in batches, then writes artifacts under
`<output_dir>/embeddings/`:

Canonical `image_path` values are relative to the dataset root. Resolve them
as `dataset_root / frame.image_path` when integrating the builder with an
embedding consumer; do not rewrite the canonical Parquet with absolute paths.

| File | Format | Purpose |
|---|---|---|
| `visual_embeddings.npy` | NumPy | L2-normalized visual embedding matrix |
| `frame_mapping.parquet` | Parquet | `embedding_index` → frame identifiers |
| `metadata.yaml` | YAML | Corpus provenance (`EmbeddingMetadata`) |

The embedding and mapping filenames match the offline artifact contracts in the
root [`README`](../../../README.md). Frames that fail to load are collected and
skipped rather than aborting the run.

```python
from pathlib import Path

from hcmai.embedding.embedding import EmbeddingPipeline
from hcmai.retriever.encoder import EncoderConfig

pipeline = EmbeddingPipeline(
    frames_path=Path("data/metadata/frames.parquet"),
    dataset_root=Path("data"),
    output_dir=Path("artifacts"),
    encoder_config=EncoderConfig(device="cuda"),
    dataset_version="hcmai2026",
)
metadata = pipeline.run()
```

## Metadata

`metadata.py` holds `EmbeddingMetadata`, the provenance record for a generated
corpus (dataset version, model, preprocessing size, dtype, embedding dimension,
frame counts, normalization, device, batch size, processing time). It is kept
apart from `embedding.py` so the descriptor lives separately from the code that
reads frames and writes artifacts. `to_dict`/`from_dict` round-trip through YAML
or JSON.

The encoder and its `EncoderConfig`/`EncodingStats` are imported from the
[retriever](../retriever) package rather than duplicated here. Root-level
commands are documented in the [scripts guide](../../../scripts/README.md).
