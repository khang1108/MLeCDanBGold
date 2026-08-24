# Retriever

`hcmai.retrieval.retriever` owns visual, caption, OCR, and frame-aligned ASR search,
plus task-weighted reciprocal-rank fusion. Other components call the public
`RetrievalService`; index and modality implementations remain internal.

```text
retriever/
├── pipeline.py                 # RetrievalService public facade
├── models/
│   ├── contracts.py            # Retriever protocol
│   └── metadata.py             # FAISS artifact provenance
├── dense/
│   ├── index.py                # Exact FAISS index
│   └── retriever.py            # Visual dense search
├── text/
│   ├── artifacts.py            # Caption/OCR/ASR artifact builder
│   └── retriever.py            # Frame-aligned text search
├── fusion/rrf.py               # Four-source task-weighted RRF
└── evaluation/benchmark.py     # Internal dense-baseline benchmark
```

Embedding models and their adapters belong to `hcmai.retrieval.embedding`, not this
package. Shared inputs and outputs use `SearchFilters`, `TaskType`,
`RetrievalCandidate`, and `RetrievalResult` from `hcmai.common.schemas`.

## Public service

```python
from hcmai.retrieval.retriever.pipeline import RetrievalService

result = retrieval_service.search(
    query="một người đang đi bộ",
    top_k=100,
    query_type=task_type,
)
candidates = result.candidates
encoding_ms = result.trace.duration_for("query_encoding")
```

Every `search` call returns its own candidates, warnings, and named stage
trace. Timing is never read from mutable service fields, so concurrent calls
cannot overwrite each other's telemetry. `RetrievalService` exposes online
search and read-only index loading, while its build helpers are used only by
offline artifact jobs. Production code outside this component must not
instantiate dense/text retrievers, FAISS indexes, or fusion implementations
directly. The application composition root constructs the service once and
reuses its encoders and indexes.

`from_index` creates a single-index service for the frozen dense baseline.
`from_fast_track_indexes` wires the current competition path: one visual index,
an optional frame-native Context index, and optional segment-native ASR followed
by RRF fusion. All configured indexes must have compatible model, dimension,
and dataset provenance; an invalid or missing required artifact leaves online
search unavailable rather than silently selecting a rollback architecture.

## Index artifacts

The exact baseline uses FAISS `IndexFlatIP` with L2-normalized vectors. Index
building validates that embeddings and mappings describe the same corpus:

- row counts match;
- `embedding_index` is a permutation of `0..N-1`;
- `frame_id` values are unique;
- serialized index and mapping metadata remain compatible when loaded.

Every deployable dense-index directory is a complete bundle containing
`dense.index`, `frame_mapping.parquet`, `metadata.json`, `vectors.npy`, and the
persisted video-posting/timestamp arrays. Offline text-enrichment builders may
use filenames selected by `index.text_embedding_filenames` in
`configs/baseline.yaml`; those artifacts are not loaded by the online
composition root.

### Offline generation and local serving

Embedding generation and index persistence are offline batch operations run
on a remote GPU VM. That job must validate, version, and synchronize a complete
immutable bundle to the local serving machine. The local KIS and TRAKE runtime
only validates and memory-maps those artifacts. It must never generate,
reconstruct, migrate, or save embeddings during application startup or a
request. An incomplete bundle leaves retrieval unavailable with an actionable
startup diagnostic and must be rebuilt or resynchronized from the offline job.

## Four-modal retrieval and fusion

```mermaid
flowchart LR
    Q["Query + filters"]
    Q --> V["SigLIP visual index"]
    Q --> C["BGE caption index"]
    Q --> O["BGE OCR index"]
    Q --> A["BGE ASR index"]
    V --> U["Union by exact frame_id"]
    C --> U
    O --> U
    A --> U
    U --> F["Task-weighted RRF"]
    F --> R["RetrievalResult: candidates + trace + warnings"]
```

The visual branch embeds the query in the visual frame space. Caption, OCR,
and ASR search separate BGE text spaces; ASR is frame-aligned transcript text,
not a raw-audio branch. Every branch returns canonical candidates with its raw
similarity in `source_scores` and one-based rank in `source_ranks`.

Candidates are unioned by exact `frame_id`. A missing source contributes no
score and no penalty. Fusion never compares raw SigLIP and BGE similarities or
rewrites canonical identity. For task `t`, frame `d`, RRF constant `k`, and
the sources that returned that frame:

$$
\operatorname{WRRF}_{t}(d)
=
\sum_{s \in S(d)}
\frac{w_{t,s}}{k + r_s(d)}
$$

Results are ordered by descending fused score, best individual source rank,
then stable lexical `frame_id`. The current equal task weights are explicit
untuned baselines. Future weights must be selected on labeled development data
using official Mean Top-k R-Score and latency, not Recall@1/5 alone.

## Benchmark status

`evaluation/benchmark.py` deliberately remains an internal
`RetrievalBenchmark` for single-index dense-baseline measurements. It records
baseline recall, latency, throughput, build time/size, and GPU memory; it does
not evaluate the full retrieval-fusion-reranking-materialization path or prove
the official competition score.

Do not expose it as `EvaluationService`. That service is deferred until an
end-to-end evaluator has a second demonstrated use and confirmed 2026 scoring
semantics. Official competition experiments must eventually run through
`SearchService` and record predictions, failures, task row scores, Mean Top-k
R-Score at `{1,5,20,50,100}`, and P50/P95 latency under `runs/`.

## Verification

```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/test_dense_retriever.py \
  tests/test_faiss_index.py \
  tests/test_caption_retriever.py \
  tests/test_caption_index_pipeline.py \
  tests/test_fusion_retriever.py
pyright src/hcmai/retrieval/retriever
```

The offline build entry points live under [`scripts/`](../../../scripts/).
