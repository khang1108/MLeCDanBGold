# Retriever

`hcmai.retrieval.retriever` owns online index loading, visual/Context/segment-
ASR search, score fusion, and per-video event scoring. Other runtime packages
call the public `RetrievalService`; concrete dense, text, segment, and fusion
implementations remain internal.

Shared immutable ranking values are `RetrievalCandidate`, `RetrievalResult`,
and `RetrievalSource` from `hcmai.retrieval.models`. They preserve canonical
identity and modality provenance without depending on offline artifact
contracts.

## Runtime boundary

```python
from hcmai.retrieval import RetrievalService

result = retrieval_service.search(query="một người đang đi bộ", top_k=100)
```

Application composition uses `RetrievalService.from_fast_track_indexes(...)`
to combine the required visual index with optional frame-native Context and
segment-native ASR indexes. Search and `score_event_videos(...)` are online;
index construction and publication are not.

Every candidate preserves `frame_id`, `video_id`, `frame_idx`, and
`timestamp_ms`. Fusion may combine ranks and scores but cannot synthesize or
rewrite identity. A source that does not return a frame contributes neither a
score nor a penalty.

## Offline construction

Embedding and index construction belongs to `offline.embeddings` and
`offline.indexes`. The operator-facing command is
`scripts/build_retrieval_indexes.py`; it validates inputs, builds the
configured bundles, and checks that every mapping remains aligned with the
canonical corpus. The runtime only loads the resulting existing artifacts.

The configured paths and bundle names remain the pre-Phase-B paths. Startup
does not migrate, reconstruct, or republish them. Missing or invalid required
artifacts leave search unavailable with an explicit health diagnostic.

## Retrieval composition

```text
query/event text
  -> visual query encoder -> visual frame index
  -> evidence encoder     -> optional Context frame index
                          -> optional ASR segment index -> frame projection
  -> source-aware fusion
  -> canonical candidates and per-video score matrices
```

The visual branch uses the visual embedding space. Context and ASR use the
configured evidence-text embedding family. Segment ASR is projected through
canonical video/timestamp evidence and does not become frame-native truth.

## Verification

```bash
PYTHONPATH=.:src aic/bin/python -m pytest \
  tests/test_dense_retriever.py \
  tests/test_faiss_index.py \
  tests/retrieval/test_context_index.py \
  tests/retrieval/test_segment_dense_index.py
```

Use `PYTHONPATH=.:src aic/bin/python scripts/build_retrieval_indexes.py --help`
to inspect the non-serving build command without starting artifact work.
