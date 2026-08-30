# HCMAI Research Knowledge

## Unified ordered event-to-frame alignment baseline

**Date:** 2026-08-30
**Problem:** KIS currently uses progressive scene localization while TRAKE uses
monotonic dynamic programming over dense event/frame scores. The two paths
duplicate temporal ownership and make a clean baseline difficult to ablate.

### Sources

- [CrossTask: Cross-Task Learning for Instructional Videos](https://arxiv.org/abs/1903.08225)
- [Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers](https://arxiv.org/abs/2108.11996)
- [A Survey on Temporal Sentence Grounding in Videos](https://arxiv.org/abs/2109.08043)
- Repository source trace on 2026-08-30: `KISPipeline` calls
  `TemporalEvidenceCore.localize()`; `TRAKEPipeline` calls
  `TemporalEvidenceCore.align_ordered()`; the latter uses
  `score_visual_videos()` and `monotonic_dp.py`.

### Findings

- **PAPER:** Ordered instructional-step alignment and monotonic temporal
  sequence alignment are established problem formulations. CrossTask and
  Drop-DTW are supporting precedents, but their models and datasets are not
  the HCMAI task or current visual-only implementation.
- **PAPER:** Temporal grounding requires semantic localization in time; the
  survey supports treating localization as an explicit subsystem rather than
  an HTTP/UI concern.
- **SOURCE:** The current DP is a deterministic, strict-increasing keyframe
  decoder. It scores a same-video event-by-frame matrix and materializes
  returned `frame_id` values through `DataService`. The KIS pipeline instead
  owns process-local progressive state, scene clustering, and a bounded
  single-frame reranker.

### Relevance to HCMAI

- **PROPOSED:** One task-agnostic ordered-alignment service could eliminate the
  duplicated temporal facade and expose KIS/TRAKE as thin output projections.
  KIS would select a deterministic representative from the full path while
  retaining every canonical `frame_id` for evidence inspection.
- **PROPOSED:** This is a semantic migration, not a behavior-preserving
  refactor. There is no measured evidence that visual-only monotonic alignment
  improves KIS, and strict no-frame-reuse may not match the current organizer
  scorer.

### Status

**PAPER-SUPPORTED** problem formulation; **PROPOSED** HCMAI architecture;
no HCMAI accuracy result measured. The local 2026 preliminary-round document
confirms complete KIS queries and ordered TRAKE event frames, but does not
settle whether one frame may satisfy more than one event.

### Decision or Experiment

The user explicitly authorized the structural migration and removal of the
progressive KIS path on 2026-08-30 before an evaluation record was available.
Before treating that implementation as a competition cut-over, freeze a
versioned development set and record the current/proposed outputs, relevant
official metric or proxy, canonical identities, index/model/config versions,
and P50/P95 latency. The record template and current no-release-claim decision
are in `docs/research/2026-08-30-temporal-migration-gate.md`. The organizer
contract or scorer must resolve whether strict chronological paths and
non-reused frames are valid. Accept, revise, or reject the migration from that
record; do not infer an improvement from literature alone.

## Segment-native ASR retrieval with canonical-frame fusion

**Date:** 2026-08-21

**Problem:** ASR is timestamped timeline evidence, while HCMAI retrieval and
competition submission require canonical frame identities. A frame-aligned ASR
index would conflate these two coordinate systems and make provenance harder to
inspect.

### Sources

- [Unified Interactive Multimodal Moment Retrieval (UIMR)](https://arxiv.org/abs/2512.12935v1)
- [Everything at Once: Multi-modal Fusion Transformer for Video Retrieval](https://arxiv.org/abs/2112.04446)
- [M2HF: Multi-Modal Hierarchical Fusion for Video-Text Retrieval](https://arxiv.org/abs/2208.07664)
- [Video-ColBERT: Contextualized Late Interaction for Video Retrieval](https://arxiv.org/abs/2503.19009)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [BGE-M3](https://arxiv.org/abs/2402.03216)

### Findings

- **PAPER:** UIMR treats a retrieval unit as a temporal moment and combines
  heterogeneous multimodal evidence rather than assuming every modality is
  natively frame-aligned.
- **PAPER:** Everything at Once and M2HF support retaining modality-specific
  representations and combining them with fusion, instead of destructively
  replacing every source with one undifferentiated text representation.
- **PAPER:** Video-ColBERT supports late interaction as a useful retrieval
  pattern when fine-grained evidence should retain its own provenance.
- **PAPER:** SigLIP2 and BGE-M3 are suitable research references for separate
  visual and broad text embedding families. They do not establish an HCMAI
  ranking improvement by themselves.

### Relevance to HCMAI

- **SOURCE:** BTC keyframes carry canonical `video_id`, `frame_id`,
  `frame_idx`, and `timestamp_ms`; transcripts are timestamped segments rather
  than frame-native evidence.
- **PAPER-SUPPORTED / SOURCE:** The implemented direction keeps ASR indexed by
  `segment_id`, projects a returned segment onto canonical frames only at the
  retrieval boundary, and preserves segment metadata alongside the canonical
  candidate. Frame-native Visual and FrameContext channels remain independent
  inputs to late fusion.
- **PROPOSED:** The initial fixed, neutral RRF weights are an engineering
  baseline, not a literature-derived optimum. They must remain configurable and
  be evaluated per HCMAI task.

### Status

**PAPER-SUPPORTED** architecture direction; **PROPOSED** HCMAI fusion weights;
no HCMAI accuracy result has been measured.

### Decision or Experiment

Use the segment-native ASR index and canonical-frame projection for the
fast-track profile while preserving specialist evidence and canonical identity.
Evaluate Visual-only (B0), Visual + FrameContext (B1), and Visual +
FrameContext + projected ASR (B2) on a versioned, labelled HCMAI query set.
Record recall/ranking, temporal-localization evidence, latency, artifact/model
versions, and failure cases before changing fusion weights or claiming an
improvement.
