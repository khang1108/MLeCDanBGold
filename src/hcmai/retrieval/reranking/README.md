# Multimodal reranking

`hcmai.retrieval.reranking` reorders only the bounded `RetrievalCandidate` list supplied
by retrieval. It does not search the corpus and cannot create or rewrite a
candidate's `frame_id`, `video_id`, or `frame_idx`.

```text
reranking/
├── pipeline.py                 # RerankingService public facade
├── config.py                   # Batching and score policy
├── models/
│   └── contracts.py            # RerankingAdapter protocol
└── adapters/
    ├── qwen.py                 # Lazy local Qwen scorer
    └── remote.py               # Remote inference scorer
```

## Public service

```python
from hcmai.retrieval.reranking.pipeline import RerankingService

ranked = reranking_service.rerank(query, candidates)
```

`RerankingService` resolves canonical image paths through the injected frame
authority, loads a bounded batch, delegates scoring to its adapter, validates
the returned count and finite scores, and returns score-enriched candidate
copies. Equal scores preserve input order.

The adapter contract is intentionally small: it scores an ordered image batch
without owning candidate identity. `QwenAdapter` and `RemoteAdapter` implement
that contract. Production callers select them through composition or the
service's `remote` constructor; they do not import adapter modules directly.

Missing frames or images, adapter failures, score-count mismatches, and invalid
scores fail the request. Tests inject fake adapters and never download model
weights.

## Verification

```bash
PYTHONPATH=.:src aic/bin/pytest tests/test_reranker.py tests/test_qwen_reranker.py
pyright src/hcmai/retrieval/reranking
```

Real experiments must record the selected checkpoint, configuration,
predictions, failures, official Mean Top-k R-Score, Recall, MRR, and P50/P95
latency under `runs/`.
