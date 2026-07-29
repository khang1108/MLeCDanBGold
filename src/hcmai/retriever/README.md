# Retriever

This package groups retrieval work by capability so each modality or retrieval
stage has a clear owner:

```text
retriever/
├── dense/
│   ├── encoder.py       # SigLIP2-style image/text encoding
│   ├── index.py         # Modality-neutral exact FAISS index
│   ├── retriever.py     # Online dense retrieval contract
│   └── models/          # Dense artifact metadata and encoding stats
├── caption/
│   ├── pipeline.py      # Configured caption artifact build
│   └── retriever.py     # Caption corpus, index builder, and retriever
├── fusion/
│   └── rrf.py           # Cross-source candidate union and RRF
└── evaluation/
    └── benchmark.py     # Reproducible retrieval experiments
```

The root `hcmai.retriever` package re-exports the small public API. New
capability-specific code belongs in its owning folder instead of adding
another flat root-level `.py` file.

## Dependencies

The heavy libraries are imported lazily inside the methods that need them, so
importing `hcmai.common.config` or `dense.models` never pulls in a model
runtime.

```bash
aic/bin/python -m pip install -e ".[embedding]"
```

- `transformers`, `torch`, and `pillow` are used by `DenseEncoder`.
- `faiss-cpu`, `numpy`, and a Parquet engine such as `pyarrow` are used by
  `DenseIndex` and the retriever.

## Concept split: config vs stats vs metadata

These three modules are pure data classes with no heavy imports, kept apart on
purpose:

### `hcmai.common.config` — inputs / knobs

- `RECALL_CUTOFFS`: the recall cut-offs `(1, 5, 10, 100)` frozen for the
  baseline comparison.
- `EncoderConfig`: encoder inputs (model name, device, batch size, image size,
  dtype, precision). `from_dict` builds it from a config mapping, reading the
  model name from the `name` key.
- `BenchmarkConfig`: the settings recorded alongside benchmark results for
  reproducibility.

### `dense/models/stats.py` — outputs / measurements

- `EncodingStats`: counters and per-batch timings produced while encoding, with
  `throughput_samples_per_sec`, `avg_batch_time_ms`, `p95_batch_time_ms`
  properties and a human-readable `report()`.

### `dense/models/metadata.py` — provenance / artifact descriptors

- `IndexMetadata`: provenance and shape of a serialized FAISS index (dataset
  version, model, metric, normalization, embedding dim, vector count, build
  time, on-disk size, timestamp). `to_dict`/`from_dict` round-trip through JSON.

## Encoder

`dense/encoder.py` provides `DenseEncoder`, which loads a SigLIP2-style
vision-language model once, on its first non-empty encoding call, and exposes
`encode_images` and `encode_text`. Embeddings are L2-normalized so inner product
equals cosine similarity. For convenience it re-exports `EncoderConfig` and
`EncodingStats`.

```python
from hcmai.common.config import EncoderConfig
from hcmai.retriever.dense import DenseEncoder

encoder = DenseEncoder(EncoderConfig(device="cuda"))
vectors = encoder.encode_text(["một người đang đi bộ"])
```

## Index

`dense/index.py` provides `DenseIndex`, an exact FAISS `IndexFlatIP`. `build`
validates that embeddings and mapping describe the same corpus (matching count,
`embedding_index` a permutation of `0..N-1`, no duplicate `frame_id`). `save`
writes `dense.index`, `frame_mapping.parquet`, and `metadata.json`; `load`
rejects mismatched artifacts. Only `flat_ip` is supported — the exact baseline
must be measured before any IVF/PQ approximation.

```python
from hcmai.retriever.dense import DenseIndex

index = DenseIndex.build(
    embeddings, mapping, dataset_version="hcmai2026", model_name="siglip2"
)
index.save("artifacts/indexes")
```

## Online retriever

`dense/retriever.py` provides `DenseRetriever`, which pairs an encoder with an
index and exposes the `search(query, top_k, filters)` contract used by
`hcmai.search.SearchEngine`. It rejects an encoder/index model or dimension
mismatch, returns score-sorted, deduplicated `RetrievalCandidate` objects, and
records query-encoding and index-search latency separately. Video/time filters
trigger a full-index scan so a full `top_k` survives filtering; a `min_score`
filter is applied as a threshold.

`caption/pipeline.py` loads the pipeline and model configs used by
`scripts/build_caption_index.py`. `caption/retriever.py` builds a caption
index by joining every usable caption back to the canonical `FrameStore`, then
exposes it through `CaptionRetriever`. `fusion/rrf.py` provides
`RRFFusionRetriever`: it searches
independent retrievers, unions candidates by exact `frame_id`, accumulates
their source scores/ranks, and returns the RRF-ranked top-K. Disjoint visual or
caption hits remain in the pool; frames retrieved by both sources receive
contributions from both ranks.

At application startup, caption fusion is optional. It activates only when the
caption index has the same dataset version and is compatible with the
configured query encoder; missing or incompatible caption artifacts leave the
visual path available.

## Benchmark

`evaluation/benchmark.py` provides `RetrievalBenchmark`, which runs a retriever
over labelled `EvaluationQuery` records and writes four frozen baseline
artifacts to the output directory: `config.yaml`, `metrics.json`,
`per_query.csv`, and `failures.csv`. Metrics include recall at each cut-off,
P50/P95/mean latency, throughput, index build time and size, and GPU peak
memory.

The `build_*.py` entry points that drive this package live in
[`../scripts`](../scripts).
