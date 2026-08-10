# Weighted RRF Fusion: Research and Implementation

Date: 2026-07-30

Status: four-source implementation complete; task weights are not optimized
because labeled validation queries are not currently available.

## Evidence boundary

- [MMMORRF](https://arxiv.org/abs/2503.20698) supports modality-aware weighted
  reciprocal-rank fusion over visual, text, and audio retrieval results.
- [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934)
  reports that tuned score combinations can outperform RRF when judged
  relevance data is available.
- [CONQUER](https://arxiv.org/abs/2109.10016) motivates query-adaptive fusion,
  but its learned weighting is outside the current no-training constraint.

The selected first implementation is therefore task-specific weighted RRF.
It does not add raw SigLIP2 and BGE-M3 cosine values.

## Implemented retrieval path

```text
query
  -> SigLIP2 visual rank
  -> BGE-M3 caption rank
  -> BGE-M3 OCR rank
  -> BGE-M3 ASR-transcript rank
  -> union by canonical frame_id
  -> task-specific weighted RRF
  -> multimodal reranking
```

For task `t`, frame `d`, source set `S`, and source rank `r_s(d)`:

```text
WRRF_t(d) = sum_{s in S(d)} w[t,s] / (k + r_s(d))
```

The implementation requires positive weights for visual, caption, OCR, and ASR
for every task configuration. The baseline uses `1.0` for every source. These
numbers are neutral placeholders, not a research conclusion.

## How weights will be selected

Weights must be tuned separately for KIS, KISC, VQA, and TRAKE on labeled
development queries. VKIS remains a working 2026 extension until official
evaluation data is confirmed.

1. Freeze indexes, model checkpoints, candidate counts, RRF `k`, and reranker.
2. Split labeled queries into tuning and held-out validation sets by video.
3. Search a normalized weight grid because multiplying every weight by the
   same constant does not change ranking.
4. Optimize the official mean Top-k R-Score at `{1,5,20,50,100}` for each task.
5. Break score ties with MRR, then warm P95 latency and operator throughput.
6. Record every configuration, prediction, failure, metric, and checkpoint
   under `runs/`; select once and keep the competition path fixed.

Retrieval rankings are deterministic for fixed artifacts, so repeated accuracy
runs do not provide new evidence. Repetition is useful for warm latency
measurement. Until labeled queries exist, the repository must not claim an
optimal modality ratio.

## Test boundary

Caption, OCR, and ASR indexes are tested with fake embeddings and tiny
frame-aligned Parquet fixtures. The ASR fixture represents transcript text
already aligned to canonical frames. Tests do not process videos, extract
audio, call remote services, or load real checkpoints.

## Next research phase

The dataset strategy, query-conditioned utility router, confidence-aware
gating experiments, and SoICT-to-VBS roadmap are documented in
[Query-Adaptive Multimodal Fusion](QUERY_ADAPTIVE_MULTIMODAL_FUSION_RESEARCH.md).
