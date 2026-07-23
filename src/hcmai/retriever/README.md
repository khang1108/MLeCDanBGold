# Retriever

This package holds the dense visual-retrieval baseline: the image-text encoder,
the exact FAISS index, the online retriever that the `SearchEngine` depends on,
and the reproducible benchmark. Data classes are split by concept so inputs,
provenance, and measurements never mix.

## Dependencies

The heavy libraries are imported lazily inside the methods that need them, so
importing a data-class module (`config.py`, `stats.py`, `metadata.py`) never
pulls in a model runtime.

```bash
pip install numpy pandas pyarrow faiss-cpu transformers torch pillow
```

- `transformers`, `torch`, and `pillow` are used by `DenseEncoder`.
- `faiss-cpu`, `numpy`, and a Parquet engine such as `pyarrow` are used by
  `VisualIndex` and the retriever.

## Concept split: config vs stats vs metadata

These three modules are pure data classes with no heavy imports, kept apart on
purpose:

### `config.py` — inputs / knobs

- `RECALL_CUTOFFS`: the recall cut-offs `(1, 5, 10, 100)` frozen for the
  baseline comparison.
- `EncoderConfig`: encoder inputs (model name, device, batch size, image size,
  dtype, precision). `from_dict` builds it from a config mapping, reading the
  model name from the `name` key.
- `BenchmarkConfig`: the settings recorded alongside benchmark results for
  reproducibility.

### `stats.py` — outputs / measurements

- `EncodingStats`: counters and per-batch timings produced while encoding, with
  `throughput_samples_per_sec`, `avg_batch_time_ms`, `p95_batch_time_ms`
  properties and a human-readable `report()`.

### `metadata.py` — provenance / artifact descriptors

- `IndexMetadata`: provenance and shape of a serialized FAISS index (dataset
  version, model, metric, normalization, embedding dim, vector count, build
  time, on-disk size, timestamp). `to_dict`/`from_dict` round-trip through JSON.

## Encoder

`encoder.py` provides `DenseEncoder`, which loads a SigLIP2-style
vision-language model once, on its first non-empty encoding call, and exposes
`encode_images` and `encode_text`. Embeddings are L2-normalized so inner product
equals cosine similarity. For convenience it re-exports `EncoderConfig` and
`EncodingStats`.

```python
from hcmai.retriever.encoder import DenseEncoder, EncoderConfig

encoder = DenseEncoder(EncoderConfig(device="cuda"))
vectors = encoder.encode_text(["một người đang đi bộ"])
```

## Index

`index.py` provides `VisualIndex`, an exact FAISS `IndexFlatIP`. `build`
validates that embeddings and mapping describe the same corpus (matching count,
`embedding_index` a permutation of `0..N-1`, no duplicate `frame_id`). `save`
writes `visual.index`, `frame_mapping.parquet`, and `metadata.json`; `load`
rejects mismatched artifacts. Only `flat_ip` is supported — the exact baseline
must be measured before any IVF/PQ approximation.

```python
from hcmai.retriever.index import VisualIndex

index = VisualIndex.build(
    embeddings, mapping, dataset_version="hcmai2026", model_name="siglip2"
)
index.save("artifacts/indexes")
```

## Online retriever

`dense.py` provides `DenseRetriever`, which pairs an encoder with an index and
exposes the `search(query, top_k, filters)` contract used by
`hcmai.search.SearchEngine`. It rejects an encoder/index model or dimension
mismatch, returns score-sorted, deduplicated `RetrievalCandidate` objects, and
records query-encoding and index-search latency separately. Video/time filters
trigger a full-index scan so a full `top_k` survives filtering; a `min_score`
filter is applied as a threshold.

## Benchmark

`benchmark.py` provides `RetrievalBenchmark`, which runs a retriever over
labelled `EvaluationQuery` records and writes four frozen baseline artifacts to
the output directory: `config.yaml`, `metrics.json`, `per_query.csv`, and
`failures.csv`. Metrics include recall at each cut-off, P50/P95/mean latency,
throughput, index build time and size, and GPU peak memory.

The `build_*.py` entry points that drive this package live in
[`../scripts`](../scripts).
