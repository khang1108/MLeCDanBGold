# HCMAI Research Knowledge

## Disk-backed exact metadata filtering under a shared FAISS RAM budget

**Date:** 2026-09-01 (verified 2026-09-02)
**Problem:** The merged Filter Workspace needs exact metadata filtering and
backend-owned pagination over roughly 470,000 frames, while the 16 GiB serving
host must reserve most practical memory headroom for FAISS and retrieval maps.

### Findings

- **SOURCE:** Filter V1 now exposes `/api/v1/filter` independently from Search
  over a read-only SQLite catalog with a four-connection bounded pool.
- **SOURCE:** The frontend sends field-specific metadata filters, folder/video
  scope, `frames_per_pages`, and `page_id`; it expects canonical identities and
  pagination counts.
- **SOURCE:** A second corpus-wide normalized Python projection risks
  avoidable RAM duplication and contention with FAISS.
- **VERIFIED:** The offline builder published 470,804 canonical frames with all
  five modalities in 127.46 seconds. The resulting catalog is 1,316,909,056
  bytes; online serving materializes only the requested page.
- **VERIFIED:** With the real 470,804-vector Visual FAISS index resident,
  concurrency-10 P95 was 1,905.20 ms for the slowest exact-object case and
  1,358.23 ms for the combined folder/title/object case. The same run produced
  zero query errors.
- **VERIFIED:** Filter RSS grew cumulatively by about 36.3 MiB while FAISS was
  resident across concurrency 1, 4, and 10 benchmark phases, below the 64 MiB
  target. `mmap_size=0` and the bounded connection pool remained active.

### Status

**REJECTED / RETIRED (2026-09-02).** The measurements remain valid historical
evidence, but the exact Filter product semantics did not match the desired
ranked, search-engine-like experience. The catalog, runtime service, and
offline builder were intentionally removed.

### Decision or Experiment

Do not restore exact AND filtering as the Filter product without a new approved
design. The stable endpoint currently returns HTTP 501 with an explicit
under-development message. Historical design details remain in
`docs/superpowers/specs/2026-09-01-disk-backed-metadata-filter-design.md`.

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
  decoder. It scores a same-video event-by-frame matrix; the alignment service
  validates returned metadata against `DataService`, and task workflows
  resolve full frame records only when shaping responses. The KIS pipeline
  instead owns process-local progressive state, scene clustering, and a
  bounded single-frame reranker.

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

## Minimal aligned-path contract

**Date:** 2026-08-30
**Problem:** The first cleanup implementation introduced common Pydantic
alignment DTOs and wrapped a numerical DP path in a second materialized path
schema.

### Findings

- **SOURCE:** The temporal baseline only needs a video identity, ordered
  canonical frame IDs, and a path score. `frame_idx` and timestamps remain
  available in `VideoEventScores` and are resolved by KIS/TRAKE output
  adapters when constructing competition responses.
- **SOURCE:** `AlignedPath` now lives beside the pure DP implementation and is
  returned directly by `align_video()`, `rank_paths()`, and
  `TemporalAlignmentService.align()`. There is no `DPPath -> AlignmentPath`
  conversion and no alignment schema under `common.schemas`.

### Decision

**VERIFIED (contract/tests):** Keep query planning as a normalized tuple of
event strings and keep the path contract to
`AlignedPath(video_id, frame_ids, score)`. DataService canonical validation and
task-specific materialization stay at their existing boundaries without adding
DTO fields to the temporal baseline.

## RAM-bounded custom-corpus finalization

**Date:** 2026-09-01
**Problem:** Finalizing 114 committed custom batches attempted to concatenate
470,804 Visual and Context vectors in RAM before constructing global exact
FAISS indexes. On a 16 GiB host the run was killed while building the Visual
`IndexFlatIP` after specialist Parquet compaction completed.

### Sources

- [NumPy memory-mapped arrays](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [NumPy `open_memmap`](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html)
- [FAISS getting started: incremental `Index.add`](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [FAISS exact flat-index memory contract](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Apache Arrow incremental `ParquetWriter`](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html)

### Findings

- **SOURCE:** NumPy `.npy` memmaps allow the complete vector matrix to remain
  disk-backed while bounded slices are copied from committed batch artifacts.
- **SOURCE:** FAISS flat indexes assign ordinal IDs in `add()` order, so adding
  validated vector chunks sequentially preserves the canonical global
  `embedding_index` contract and exact `IndexFlatIP` scoring.
- **SOURCE:** `IndexFlatIP` itself remains RAM-resident at four bytes per
  dimension per vector. Chunking removes avoidable vector concatenation and
  duplicate build/load peaks; it does not make the final exact index disk-only.
- **SOURCE:** Arrow `ParquetWriter` supports incremental row-group writes, so
  specialist tables do not need corpus-wide pandas concatenation.

### Relevance to HCMAI

- **SOURCE:** The failed run contained 470,804 frame vectors; Visual float32
  vectors alone are about 1.35 GiB and Context vectors about 1.80 GiB before
  FAISS-owned storage and temporary copies.
- **PROPOSED:** Bound finalization by committed-batch chunks selected through
  `--finalize-batch-chunk-size`, defaulting to 16 and allowing 32 on hosts with
  more available RAM. Preserve batch order, canonical identities, checksum
  validation, exact flat-index semantics, and specialist provenance.

### Status

**VERIFIED (contract/tests + two-batch real-artifact smoke test).** The full
114-batch corpus run remains pending, so its peak RSS and end-to-end completion
are not yet verified.

### Decision or Experiment

Compare chunk sizes 16 and 32 on the same 114-batch corpus. Record peak RSS,
wall time, disk high-water mark, output checksums/counts, and successful runtime
loads. Treat lower peak RSS as the goal; do not claim retrieval improvement
because ranking semantics are intentionally unchanged.

## Hosted embedding and multimodal-reranker VRAM sizing

**Date:** 2026-09-01
**Problem:** Size one GPU for the resident SigLIP2 Base, BGE-M3, and
Qwen3-VL-Reranker-2B services configured in `llm/config.yaml`.

### Sources

- Official Hugging Face model cards/files for
  `google/siglip2-base-patch16-224`, `BAAI/bge-m3`, and
  `Qwen/Qwen3-VL-Reranker-2B`.
- Official NVIDIA specifications for L4 24 GB, RTX A6000 48 GB, and L40S
  48 GB.

### Findings

- **SOURCE:** The configured resident weights are approximately 1.5 GB
  SigLIP2 F32, 2.12 GB BGE-M3 F32, and 4 GB Qwen reranker BF16 before CUDA,
  allocator, input, and activation memory.
- **SOURCE:** The local adapter eagerly retains every enabled model. Current
  limits allow embedding batches of 128, BGE sequences up to 8192 tokens, and
  reranker batches of 16 images at up to 262,144 pixels each.
- **PROPOSED:** A 48 GB GPU is the safe initial target without changing these
  limits. A 24 GB GPU is a practical lower-cost target only after reducing
  embedding batches to 16-32, reranker batches to 4-8, and online BGE query
  length to roughly 512 tokens.
- **SOURCE:** No finite VRAM recommendation makes the exposed worst case of
  128 BGE inputs each padded near 8192 tokens safe; API/config limits must
  reflect the online query workload.

### Status

**SOURCE-SUPPORTED / PROPOSED sizing.** Peak VRAM and latency have not yet been
measured on HCMAI requests.

### Decision or Experiment

Prefer one 48 GB A6000/L40S-class GPU for unchanged configuration. For a 24 GB
L4-class deployment, lower the batch/sequence limits first, then capture
`torch.cuda.max_memory_allocated()` for Visual query embedding, BGE query
embedding, reranking at candidate depths 4/8/16, concurrent mixed requests,
and startup resident memory before declaring the profile verified.

## Segment-projected ASR Dense temporal

**Date:** 2026-09-02

**Problem:** Dense temporal was wired to a frame-native
`artifacts/indexes/asr` artifact that is not part of the production artifact
pipeline. Production ASR already exists as timestamped transcript segments and a
segment-native Dense index at `artifacts/indexes/asr_segments`.

**Decision:** Reuse the existing `SegmentDenseIndex` and
`SegmentFrameProjector` at runtime. Score each event against all ASR segments,
project segment scores onto canonical visual frames, max-aggregate collisions,
and assign uncovered frames the event floor before the existing per-event
min-max normalization.

**Preserved contracts:** Transcript artifacts, segment-ASR generic retrieval,
BM25 frame-ASR projection, Dense weights, Hybrid fusion, and monotonic DP remain
unchanged.

**Artifact contract:** No frame-native Dense ASR index is required. Dense ASR
uses `artifacts/indexes/asr_segments`; BM25 ASR may continue to use
`artifacts/enrichment/asr/frame_enrichment.parquet`.

### Status

**SOURCE (code/tests/artifact contract):** This is a production-artifact
correction, not a measured retrieval-quality change. No HCMAI accuracy
improvement is claimed.

## Vietnamese context and direct BM25 routing

**Date:** 2026-09-02

**Problem:** Caption and FrameContext artifacts are being replaced with
Vietnamese versions, so translating each original query to English for caption
BM25 is both unnecessary and inconsistent with the new corpus language.

**Decision:** BM25 uses the original Vietnamese event for title, caption, OCR,
and ASR. Generated English candidates affect Dense retrieval only. Runtime
Context defaults move to `artifacts/indexes/context_vi` and
`artifacts/enrichment/context_vi/frame_context_v1.parquet`.

**Artifact contract:** The BM25 artifact must be rebuilt from the Vietnamese
FrameContext/caption source after synchronization; online serving continues to
load the precomputed `artifacts/indexes/bm25` artifact and never rebuilds it.

### Status

**SOURCE (routing/config/tests); PENDING DATA:** Runtime behavior is verified by
tests. Full artifact startup remains pending completion of the AWS sync and a
Vietnamese BM25 rebuild. No retrieval-quality improvement is claimed yet.

## Direct image-to-keyframe retrieval with SigLIP2

**Date:** 2026-09-03

**Problem:** KIS users may provide a reference image instead of text. The
runtime already has a SigLIP2 visual frame index, but only text queries were
exposed through the public search service.

### Sources

- [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic
  Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)

### Findings

- **PAPER:** SigLIP2 is a dual vision-language encoder evaluated for
  image-text retrieval and visual-representation transfer. Its image encoder
  produces normalized vectors compatible with similarity retrieval when the
  query and corpus use the same pinned checkpoint and preprocessing.
- **SOURCE:** HCMAI visual keyframes are already embedded with
  `google/siglip2-base-patch16-224` and searched through the canonical visual
  FAISS mapping.

### Relevance to HCMAI

- **SOURCE / PAPER-SUPPORTED:** Encode an uploaded image with the same pinned
  SigLIP2 visual adapter, search the existing visual index directly, and
  materialize identities only through `Corpus`. This reuses the established
  index and preserves `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms`.
- **PROPOSED:** Direct image-to-image nearest-neighbour search is the initial
  baseline. It does not establish improved KIS accuracy and intentionally does
  not add text, BM25, or temporal-event fusion without query-specific evidence.

### Status

**PAPER-SUPPORTED / SOURCE-IMPLEMENTED; NOT YET BENCHMARKED.**

### Decision or Experiment

Measure image-query Recall@K and latency on a hand-labeled HCMAI reference-image
set. Compare the direct SigLIP2 baseline against optional visual reranking and
multi-crop query embeddings before adopting any more complex method.
