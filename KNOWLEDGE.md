# HCMAI Research Knowledge

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
