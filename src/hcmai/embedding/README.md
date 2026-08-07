# Embedding

`hcmai.embedding` owns visual and text encoding. Other components use the
public `EmbeddingService`; checkpoint- and provider-specific implementations
remain private adapters.

```text
embedding/
├── pipeline.py                 # EmbeddingService public facade
├── artifacts.py                # Offline visual artifact builder
├── models/
│   ├── contracts.py            # Text/Image adapter protocols
│   ├── artifacts.py            # EmbeddingRun result entity
│   ├── metadata.py             # Artifact provenance
│   └── stats.py                # Encoding measurements
└── adapters/
    ├── siglip.py               # Local image and text encoder
    ├── bge.py                  # Local evidence-text encoder
    └── remote.py               # Remote text encoder
```

`models/` contains contracts and data only. Model loading and framework code
belong in `adapters/`. Adapters load lazily on the first non-empty request.

## Public service

```python
from hcmai.embedding.pipeline import EmbeddingService

vectors = embedding_service.encode_visual_query(["một người đang đi bộ"])
evidence_vectors = embedding_service.encode_evidence_query(["biển số xe"])
```

`EmbeddingService` owns the configured adapter instances and exposes:

- `encode_visual_images` for canonical frame images;
- `encode_visual_query` for queries against the visual index;
- `encode_evidence_query` for caption, OCR, and ASR indexes;
- `build_visual_artifacts` for the offline corpus job.

Cross-component production code must not import an embedding adapter directly.
Composition code creates adapters through the service factory methods; unit
tests may inject fake adapters through the constructor.

## Offline artifacts

```python
from pathlib import Path

from hcmai.common.config import EncoderConfig
from hcmai.embedding.pipeline import EmbeddingService

run = EmbeddingService.build_visual_artifacts(
    frames_path=Path("artifacts/frame_store/frames.parquet"),
    dataset_root=Path("artifacts/frame_store"),
    output_dir=Path("artifacts"),
    encoder_config=EncoderConfig(device="cuda"),
    dataset_version="hcmai2026",
)
```

The builder writes these versioned files under `<output_dir>/embeddings/`:

| File | Purpose |
|---|---|
| `visual_embeddings.npy` | L2-normalized visual embedding matrix |
| `frame_mapping.parquet` | `embedding_index` to canonical frame mapping |
| `metadata.yaml` | Model, dataset, shape, normalization, and run provenance |

Canonical `image_path` values stay relative to the dataset root. The builder
resolves them at read time and never rewrites `frames.parquet`. Frames that
cannot be loaded are recorded and skipped rather than changing the identity of
the remaining rows.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest \
  tests/test_embedding_pipeline.py \
  tests/test_encoder.py \
  tests/test_bge_encoder.py
pyright src/hcmai/embedding
```
