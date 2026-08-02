# Multimodal reranking

`hcmai.reranking` reorders only the retrieval candidates supplied to it. It
does not search FAISS, enumerate the corpus, or change authoritative frame
identifiers.

## Structure

```text
reranking/
├── multimodal/
│   ├── config.py       # Candidate-pipeline configuration
│   ├── protocols.py    # Model-agnostic scoring boundary
│   └── reranker.py     # Bounded fail-fast reranking
└── qwen/
    ├── config.py       # Qwen checkpoint/runtime configuration
    └── scorer.py       # Lazy native Qwen3-VL relevance scorer
```

Configuration and scoring protocols are owned by this package because they
are used only by reranking. Input and output candidates use the shared
`hcmai.common.schemas.RetrievalCandidate` contract. The root
`hcmai.reranking` package re-exports the public classes; a new model backend
gets its own folder rather than another flat root-level module.

## Components

- `MultimodalReranker` resolves images for the supplied candidates, invokes a
  `ScoreBatch`, maps scores back in input order, and returns validated copies.
- `RerankerConfig` controls batching and final-score policy.
- `QwenRerankerScorer` lazily loads one Qwen3-VL model/processor pair and
  returns the official yes/no relevance probability.
- `QwenRerankerConfig` records checkpoint, revision, device, dtype, token, and
  image limits.

Missing images, backend failures, count mismatches, and invalid scores abort
the request. Candidate count and exact `frame_id`, `video_id`, and `frame_idx`
mappings must never change.

## Dependency direction

```text
retriever -> RetrievalCandidate -> reranking -> RetrievalCandidate
                                      |
                                      -> FrameStore image lookup
```

Reranking-specific configuration, protocols, and model adapters stay here.
Only contracts exchanged with other modules belong in `common`.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest tests/test_reranker.py tests/test_qwen_reranker.py
```

Tests inject fake scorers and must not download model weights. A real
experiment must record checkpoint, configuration, predictions, failures,
Recall@1/5, MRR, and P50/P95 latency under `runs/`.
