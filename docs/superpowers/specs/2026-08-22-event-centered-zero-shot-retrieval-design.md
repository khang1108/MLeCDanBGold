# HCMAI Event-Centered Zero-Shot Retrieval Redesign

**Date:** 2026-08-22  
**Status:** Draft for user review  
**Scope:** Architecture and experiment strategy for replacing BTC-only visual coverage with a custom raw-video timeline while preserving a safe competition baseline.  
**Applies to:** KIS first, then VQA/TRAKE through shared retrieval and temporal evidence.  
**Implementation:** Not started. This document is a design spec, not an implementation plan.

---

## 1. Problem Statement

HCMAI currently uses 177,321 BTC-provided keyframes from 873 news videos as the canonical visual corpus. Each source video is an episode/clip of the Vietnamese news program **“60 giây”**, typically about 25–35 minutes long.

The BTC keyframes were useful to ship an initial competition system quickly, but they are sparse relative to the target query semantics. Across roughly 1.57 million seconds of video, 177,321 keyframes correspond to about one BTC keyframe every 8.9 seconds on average. KIS and QA queries may describe a coherent scene/event lasting roughly 5–15 seconds, so a frame-only retrieval system over this sparse set can miss important short events entirely.

The repository also contains a custom raw-video keyframe pipeline based on TransNetV2, EfficientGEBD, camera-compensated optical flow, adaptive temporal coverage, DINOv2 deduplication, and gap repair. That pipeline is sophisticated but reflects an older design goal: decide offline which frames are important **before the query is known**.

The redesign changes the objective:

> **Offline extraction should preserve enough temporal evidence for future queries; online retrieval should decide which event and frames are important for the current query.**

The resulting system is event-centered and training-free at first because HCMAI has only about 20 labeled benchmark queries across KIS and QA. Those labels are too scarce to justify training a task-specific Event↔Query attention network without severe overfitting risk.

---

## 2. Design Status Labels

This document uses the project research labels:

- **[VERIFIED]** — confirmed in the current repository or organizer data/competition behavior.
- **[PAPER]** — supported by published or scholarly prior work.
- **[PROPOSED]** — design selected for HCMAI but not yet validated experimentally.
- **[REJECTED]** — considered and intentionally not selected for the current competition path.

---

## 3. Current-State Findings

### 3.1 BTC path

**[VERIFIED]** The active production configuration currently uses BTC keyframes as its `FrameStore` source and preserves organizer mapping metadata.

**[VERIFIED]** BTC-provided object evidence is filename/keyframe aligned. It cannot be silently transferred to newly extracted raw-video frames.

**[VERIFIED]** BTC keyframes remain valuable as a safety baseline because Batch 1 has already been submitted using the BTC-based system.

### 3.2 Existing custom preprocessing path

**[VERIFIED]** The repository contains a raw-video preprocessing pipeline under `src/hcmai/data/preprocessing/` with the following major stages:

1. PyAV decode with PTS/time-base timestamps.
2. Camera-compensated optical-flow motion signal.
3. TransNetV2 shot-boundary signal.
4. EfficientGEBD semantic/event-boundary signal.
5. Motion-adaptive temporal candidate generation.
6. Burst sampling near boundaries and motion peaks.
7. DINOv2 semantic deduplication.
8. Hard maximum-gap restoration.
9. `FrameRecord` materialization.

**[VERIFIED]** This path is currently treated by the repository as a legacy/alternate producer rather than the active production path.

**[VERIFIED]** The local parallel custom-preprocessing path shares mutable EfficientGEBD detector state across worker threads and therefore has a static-analysis concurrency risk.

**[VERIFIED]** Its current `frame_idx` computation is not reconciled with the organizer mapping contract and therefore cannot be treated as competition-authoritative without a dedicated conversion layer.

### 3.3 Existing runtime legacy branches

**[VERIFIED]** The P2 cleanup removed the VQA legacy-localization runtime path,
the `temporal | legacy` switch, the `legacy_specialists` startup profile, and
the online standalone specialist-index loader. Caption/OCR/ASR evidence
contracts and stores remain because they are still used by enrichment and
evidence materialization. Historical plans may still mention the removed paths.

---

## 4. Constraints

### 4.1 Competition schedule

**[VERIFIED]** Competition scoring is accumulated across four Saturday batches. Batch 1 has already been submitted, leaving three scored batches.

Therefore:

> The remaining competition batches are not A/B-test slots. New architecture must be validated offline before it is allowed to replace or augment the live submission path.

### 4.2 Label scarcity

**[VERIFIED]** Only approximately 20 labeled benchmark queries are available for KIS and QA.

Consequences:

- do not train a HCMAI-specific cross-attention network from these labels;
- do not tune a large number of modality/event parameters against these labels;
- preserve the official benchmark as an evaluation set;
- use synthetic/pseudo queries only for regression diagnostics, not as proof of competition accuracy.

### 4.3 Serving constraints

The design keeps the established serving objective:

- retrieval and thumbnail access should remain fast;
- expensive VLM/LLM reasoning may occur after candidate narrowing;
- no online request may silently decode all source videos or rebuild large indexes;
- expensive extraction, embedding, event construction, and index creation are offline jobs.

---

## 5. Core Architectural Decision

### 5.1 Retrieval hierarchy

**[PROPOSED]** HCMAI will distinguish three semantic levels:

```text
Video
  └── Event / temporal scene        ← primary semantic retrieval unit
        └── FrameObservation        ← precise evidence/localization unit
```

A future `Story`/news-topic level may sit between Video and Event, but it is explicitly out of scope for the first migration.

The responsibilities are:

- **Video:** corpus routing/grouping and provenance, not the primary semantic representation of a 25–35 minute news clip.
- **Event:** coherent temporal scene, usually the level most closely matching a 5–15 second natural-language query.
- **FrameObservation:** exact visual observation used for detailed evidence and final competition `frame_idx` selection.

### 5.2 Event-centered search

The online mental model becomes:

```text
query
  → query normalization / optional decomposition
  → coarse frame/event retrieval
  → query-conditioned temporal aggregation
  → top Event candidates
  → frame-level refinement inside/around those events
  → optional frozen reranker/VLM reasoning
  → competition output
```

Frame retrieval remains useful as evidence and as a cheap first-stage search, but an independent frame is no longer assumed to be the complete semantic answer.

---

## 6. Offline Visual Timeline

### 6.1 Version 1 sampling policy

**[PROPOSED]** The first custom visual corpus uses a deterministic **1 FPS** timeline extracted from the raw videos.

Rationale:

- approximately 1.57 million observations across the current corpus;
- roughly 5–15 visual observations for a typical 5–15 second target event;
- materially denser than the BTC set without exploding to 3M+ observations;
- deterministic and simple enough to diagnose;
- provides a clean baseline against which future adaptive sampling can be measured.

### 6.2 Rejected initial alternatives

**[REJECTED for v1] 2 FPS fixed timeline.** It doubles storage/index/enrichment before 1 FPS has demonstrated a coverage bottleneck.

**[REJECTED for v1] 1 FPS + adaptive bursts.** This may eventually help around short/high-information transitions, but introducing it immediately would confound the experiment: it would be unclear whether gains came from denser coverage or the adaptive policy.

**[REJECTED] Offline “perfect keyframe” selection as the primary corpus.** The current selection/dedup pipeline makes query-independent importance decisions that can permanently delete evidence before the query is known.

### 6.3 Enrichment tiers

**[PROPOSED]** Not every 1 FPS observation receives every expensive specialist artifact immediately.

Minimum dense timeline record:

```text
FrameObservation
- frame_id
- video_id
- timestamp_ms
- source_pts / decode provenance when available
- submission_fps
- submission_frame_idx
- visual_embedding
- image/thumbnail reference according to storage policy
```

Specialist evidence remains separate and can be attached selectively:

- CaptionEvidence
- OCREvidence
- ObjectEvidence
- ASRSegment
- FrameContext

The design does **not** destructively flatten these evidence types.

ASR remains timeline-native. Speech near a frame is not automatically treated as speech about that frame.

---

## 7. Canonical Identity and Submission Coordinates

### 7.1 Internal identity

**[PROPOSED]** `frame_id` is an internal immutable identity and is never derived from an array position, FAISS row, or competition coordinate.

A custom `FrameObservation` must preserve at least:

```text
frame_id
video_id
timestamp_ms
source_pts or equivalent decode provenance
submission_fps
submission_frame_idx
```

### 7.2 Organizer/BTC frame index

**[VERIFIED]** The existing BTC `frame_idx` values are valid only for the 177,321 BTC-extracted keyframes because they are organizer-provided mappings.

**[VERIFIED from competition behavior]** Custom-extracted frames may still be used if HCMAI computes a valid competition `frame_idx` using the organizer convention. The videos use integer submission FPS values of 25 or 30.

### 7.3 Authoritative conversion layer

**[PROPOSED]** Custom submission coordinates must be produced by one dedicated mapping service, conceptually:

```text
SubmissionFrameMapper(video_id, timestamp/decode provenance)
    → submission_fps
    → submission_frame_idx
```

The exact rounding/indexing formula is **not duplicated across extraction, retrieval, or pipelines**.

Before custom-frame submission is enabled, the mapper must be reverse-validated against the organizer data:

1. load all BTC records for which timestamp/FPS/mapping are available;
2. apply the candidate conversion rule;
3. require 100% exact agreement with organizer `frame_idx`, or explicitly explain every mismatch;
4. only then use the same rule for custom observations.

This preserves the existing invariant that competition coordinates come from an authoritative mapping layer rather than from arbitrary downstream inference code.

---

## 8. Event Representation

### 8.1 Event contract

**[PROPOSED]** Introduce a first-class temporal event representation:

```text
TemporalEvent
- event_id
- video_id
- start_ms
- end_ms
- frame_ids[]
- boundary/provenance evidence
- visual representation(s)
- optional textual/specialist references
- adjacency to previous/next events
```

An Event is a temporal evidence grouping, not a destructive replacement for its constituent frames or specialist evidence.

### 8.2 Soft boundaries

**[PROPOSED]** Event boundaries are soft for retrieval purposes.

If `E18` is retrieved strongly, the runtime may consider:

```text
E18
E17 + E18
E18 + E19
```

or another bounded neighbor expansion based on temporal coherence and query relevance.

This avoids treating one GEBD/shot boundary as an infallible semantic split.

### 8.3 Boundary signals

The first Event builder may use the following offline signals already available or compatible with the repository:

- TransNetV2 shot boundaries;
- EfficientGEBD semantic/event boundaries;
- visual semantic change from frozen visual embeddings;
- ASR semantic/topic shifts;
- later: OCR change, if it proves useful.

**Important scope decision:** this spec defines the Event contract and allowed signals but does **not** lock an unvalidated learned fusion model or a hand-tuned weighted formula. Exact deterministic boundary fusion is an experiment decision to be fixed before Batch 3 after the 1 FPS foundation is measured.

---

## 9. Training-Free Query↔Event Interaction

### 9.1 No HCMAI-trained attention network in v1

**[REJECTED]** Training a new Event↔Query Transformer/cross-attention model using the ~20 HCMAI labels.

Reason: the experiment would be dominated by overfitting and would not provide trustworthy evidence that the architecture generalizes.

### 9.2 Frozen similarity as query-conditioned attention

**[PROPOSED]** Attention-like selection can still be query-conditioned without learned HCMAI parameters.

For an event with frozen frame embeddings `f_i` and query embedding `q`:

```math
s_i = cos(q, f_i)
```

A soft query-conditioned weighting can be computed as:

```math
a_i = exp(s_i / τ) / Σ_j exp(s_j / τ)
```

and an event representation/score can use soft pooling, LogSumExp, top-m pooling, or another fixed aggregation selected by offline ablation.

The key principle is:

> event relevance is computed from query-conditioned evidence inside the event, rather than from one static 30-minute video embedding.

### 9.3 Query decomposition and late interaction

**[PROPOSED]** For compositional queries, optional query decomposition may produce several semantic clauses, for example:

```text
Q1: person/appearance
Q2: exits vehicle
Q3: walks toward building
Q4: enters building
```

The system can score a clause-by-frame/event matrix with frozen encoders and then enforce or reward temporal coverage/order when appropriate.

This is preferable to materializing separate learned `query→frame`, `frame→query`, `query→video`, and `video→query` networks in the label-scarce setting.

---

## 10. Multimodal Retrieval and Evidence

### 10.1 Keep specialist evidence

**[VERIFIED / KEEP]** Caption, OCR, Object, ASR, and FrameContext are not legacy concepts. They remain independently accessible for debugging, ablation, reranking, exact matching, and task-specific reasoning.

### 10.2 Dense custom timeline enrichment

**[PROPOSED]** The initial 1 FPS corpus is guaranteed a visual embedding but is **not** guaranteed full Caption/OCR/Object enrichment for every observation.

This prevents a 9× increase in visual density from automatically creating a 9× increase in every expensive artifact.

### 10.3 BTC and custom corpora coexist during migration

**[PROPOSED]** The BTC corpus remains available as a safety index while the custom 1 FPS corpus is introduced.

Online retrieval may search both in parallel and combine candidates using a deterministic, non-learned method such as rank fusion until the custom path is proven safe.

BTC-specific Objects remain attached only to BTC frames unless an explicit mapping or new detector creates valid object evidence for custom frames.

---

## 11. Competition Rollout Strategy

The remaining three scored batches are treated as release milestones, not research slots.

### Batch 2 — Coverage foundation

Goal: safely raise the visual recall ceiling.

Candidate release:

```text
BTC indexes                         (safety baseline)
+
custom 1 FPS visual timeline/index
+
deterministic dual-corpus fusion
```

No learned event-query network. No adaptive burst sampler. No full custom-frame multimodal enrichment requirement.

Release gate:

- submission-frame mapping validated;
- custom index reproducible;
- latency acceptable;
- official benchmark has no unexplained severe regressions versus BTC;
- synthetic regression suite shows no systematic retrieval collapse.

### Batch 3 — Event-centered retrieval

Goal: improve temporal semantic matching once the dense timeline is trusted.

Candidate additions:

```text
1 FPS timeline
→ TemporalEvent grouping
→ query-conditioned event aggregation
→ soft neighbor expansion
→ frame refinement
```

Exact Event boundary fusion must be decided by controlled offline ablation before this batch.

### Batch 4 — Exploitation, not redesign

Goal: fix observed failure modes without replacing the foundation.

Allowed focus areas:

- reranking;
- query decomposition/refinement;
- query-conditioned modality routing;
- VLM verification on a small candidate set;
- temporal boundary/frame selection refinement;
- latency and robustness fixes.

**[REJECTED]** Introducing an entirely new extraction architecture or large trained model for the first time in Batch 4.

---

## 12. Evaluation Discipline

### 12.1 Official benchmark

The ~20 labeled KIS/QA queries are used as a small paired evaluation set, not as a training set.

For each query, record at minimum:

```text
BTC rank
1 FPS frame rank
Event rank
Event + late-interaction rank
correct video retrieved?
correct temporal region retrieved?
best temporal distance to target
failure category
latency
```

Aggregate metrics include:

- Recall@1/5/10/100;
- video Recall@k;
- frame/temporal localization distance;
- mAP where the benchmark definition supports it;
- latency;
- index/storage size.

### 12.2 Synthetic regression suite

**[PROPOSED]** Build a larger diagnostic suite from known ASR/caption/event content:

```text
known temporal region
→ generate/rewrite a natural query offline
→ retrieve
→ verify whether the source region is recovered
```

This suite is used to detect engineering regressions and compare architecture variants. It is not treated as unbiased competition accuracy because the query-generation process creates synthetic distribution bias.

---

## 13. Primary Hypotheses

### H1 — Dense visual coverage

**[PROPOSED]** A deterministic 1 FPS custom visual timeline will improve retrieval coverage/Recall@k over the BTC-only 177k keyframe set on short scene-level queries.

Baseline:

```text
BTC visual/text retrieval
```

Treatment:

```text
BTC + custom 1 FPS visual retrieval
```

### H2 — Event aggregation

**[PROPOSED]** Query-conditioned temporal event aggregation will outperform independent frame ranking when the query describes a coherent 5–15 second scene.

Baseline:

```text
1 FPS independent frame retrieval
```

Treatment:

```text
1 FPS → Event grouping → query-conditioned event score → frame refinement
```

### H3 — Query decomposition / late interaction

**[PROPOSED]** Decomposing compositional queries and matching clauses against ordered event/frame evidence will improve retrieval for multi-step descriptions without requiring HCMAI-specific supervised training.

This hypothesis is evaluated only after H1/H2 establish a stable foundation.

---

## 14. Research Basis

The design is informed by the following prior work:

1. **Adaptive Keyframe Sampling for Long Video Understanding (AKS)** — Xi Tang et al., CVPR 2025. Frames are selected by balancing query relevance and video coverage, supporting the principle that relevance-only or uniform-only selection is insufficient.
2. **Q-Frame: Query-Aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs** — Shaojie Zhang et al., ICCV 2025. Supports training-free query-aware frame selection with frozen image-text models.
3. **LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding** — Xiaoqian Shen et al., ICML 2025. Uses DINOv2 redundancy removal plus text-guided selection, supporting separation between redundancy reduction and query-conditioned selection.
4. **EfficientGEBD** — Ziwei Zheng et al., ACM Multimedia 2024. Supports retaining EfficientGEBD as an efficient semantic-boundary signal rather than treating it as a final keyframe oracle.
5. **Event-Anchored Frame Selection for Effective Long-Video Understanding** — Wang Chen et al., 2026. Training-free event partitioning + query-relevant anchors + MMR directly supports event coverage, relevance, and diversity as separate objectives.
6. **TAG: A Simple Yet Effective Temporal-Aware Approach for Zero-Shot Video Temporal Grounding** — Jin-Seop Lee et al., 2025. Supports temporal coherence and avoiding semantic fragmentation across adjacent frames.
7. **Point to Span: Zero-Shot Moment Retrieval for Navigating Unseen Hour-Long Videos** — Mingyu Jeon et al., 2025. Supports a zero-shot search-then-refine architecture for long videos.
8. **HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling** — Joungbin An and Kristen Grauman, CVPR 2026. Supports hierarchical temporal representations that preserve global context and fine-grained temporal detail.
9. **Hierarchical Prototype Alignment for Video Temporal Grounding** — Yun Tian et al., 2026. Supports event–sentence alignment rather than relying only on global video–sentence correspondence.

These papers motivate the architecture but do not remove the need for HCMAI-specific ablation on the competition corpus.

---

## 15. Repository Migration and Legacy Cleanup

Cleanup is staged so that competition safety is not traded for codebase tidiness.

### 15.1 Delete immediately: generated/non-source artifacts

Safe hygiene cleanup:

```text
**/__pycache__/
**/*.pyc
frontend/node_modules/
frontend/build/
```

Environment secrets should not live in source archives; keep an example configuration instead.

### 15.2 Preserve and repurpose from old preprocessing

From `src/hcmai/data/preprocessing/`:

**KEEP / MOVE INTO NEW OWNERSHIP**

- PyAV decode/timestamp primitives from `video.py`;
- TransNetV2 and EfficientGEBD model adapters from `models.py`;
- S3 helpers that are still used by the canonical offline artifact flow.

**RETIRE AFTER 1 FPS TIMELINE REPLACEMENT IS VERIFIED**

- `selection.py` as the canonical frame-selection policy;
- burst/adaptive-gap candidate generation;
- DINO-based frame deletion as the primary corpus policy;
- gap-repair logic whose only purpose is to repair the old dedup sampler;
- old monolithic `prepare.py` responsibilities that mix decode, selection, dedup, and FrameStore publication.

DINO itself is not rejected; it may be reused for Event semantic change/diversity.

### 15.3 VQA/runtime legacy path — COMPLETED

The generic temporal-window helper now lives under VQA reasoning. The
`pipelines/vqa/legacy_localization/` package, the `temporal | legacy` runtime
switch, and the obsolete compatibility workflow import have been removed.

### 15.4 Retrieval profile cleanup — COMPLETED

`context_asr_segment` is now the sole supported runtime composition. The
`legacy_specialists` profile, its online loader helpers, and the rollback
factory have been removed. Offline text-index builders may remain for explicit
artifact/evaluation jobs, while CaptionEvidence, OCREvidence, ObjectEvidence,
ASRSegment, and FrameContext stores remain supported.

ASR retrieval should remain segment/timeline native rather than pretending ASR is inherently frame-native.

### 15.5 Old scripts

Consolidate toward one canonical index builder and one canonical evaluation runner.

Candidates for retirement after dependency checks:

```text
scripts/build_benchmark.py
scripts/benchmark_btc_keyframes.py
scripts/build_caption_index.py
scripts/build_embeddings.py
scripts/build_index.py
```

Keep `scripts/build_retrieval_indexes.py` only if it remains the canonical builder after the new timeline/event indexes are introduced; otherwise replace it with a single clearly named successor.

### 15.6 Explicitly keep BTC baseline support

Do **not** delete BTC ingestion/mapping during the competition:

```text
data/ingestion/btc.py
data/ingestion/keyframe_map.py
BTC ingestion scripts/config needed to reproduce the Batch-1 baseline
```

Git history is the archive for removed runtime legacy code; the active branch should not preserve every rollback architecture indefinitely.

---

## 16. Target Package Boundaries

The desired conceptual structure is:

```text
src/hcmai/
├── common/
├── data/
│   ├── ingestion/          # BTC baseline and canonical external imports
│   ├── timeline/           # raw video → deterministic FrameObservation timeline
│   ├── events/             # TemporalEvent construction and artifacts
│   ├── enrichment/         # caption/OCR/objects/ASR/context
│   └── stores/
├── retrieval/
│   ├── embedding/
│   ├── retriever/
│   │   ├── frame/
│   │   ├── event/
│   │   ├── segment/
│   │   └── fusion/
│   └── reranking/
├── temporal/
├── pipelines/
│   ├── kis/
│   ├── vqa/
│   └── trake/
├── orchestration/
├── api/
└── llm/
```

Ownership remains:

- `data/` owns extraction, canonical records, event artifacts, and evidence lineage;
- `retrieval/` owns embeddings, indexes, retrieval, fusion, and retrieval evaluation;
- `temporal/` owns shared temporal alignment primitives;
- `pipelines/` own task semantics;
- `orchestration/` composes services;
- `api/` remains transport-only.

---

## 17. Non-Goals for the First Migration

The following are intentionally outside the first implementation cycle:

- training a new attention network from the ~20 benchmark labels;
- pseudo-label/distillation training before a strong training-free baseline exists;
- a Story/news-topic hierarchy;
- adaptive burst sampling before the fixed 1 FPS baseline is measured;
- 2 FPS full-corpus extraction before 1 FPS proves insufficient;
- Caption/OCR/Object generation for every 1 FPS observation by default;
- deleting the BTC baseline during the competition;
- online raw-video decoding or online large-artifact regeneration;
- a broad unrelated repository refactor.

---

## 18. Implementation Planning Boundary

This is the master architecture spec, but the **next implementation plan is intentionally limited to the Batch-2 foundation**:

```text
raw video → deterministic 1 FPS timeline
          → validated SubmissionFrameMapper
          → visual embeddings/index
BTC baseline index in parallel
          → deterministic fusion
          → unified evaluation/regression harness
```

The first implementation plan may include only the minimal legacy cleanup required to make this path unambiguous and testable. It must not attempt the full repository cleanup in the same change set.

Event-boundary fusion, Event indexing, and query-conditioned Event refinement are a **Batch-3 follow-up design checkpoint** informed by H1 results. The Event contracts in this document constrain that future design, but they do not force an unvalidated fusion formula into the Batch-2 implementation.

This boundary keeps the next plan small enough to verify and roll back before a scored competition batch.

---

## 19. Acceptance Criteria for the Design

The design is considered ready to transition into implementation planning when the user approves all of the following decisions:

1. **1 FPS deterministic custom timeline** is the v1 extraction foundation.
2. **BTC remains a parallel safety baseline** through the remaining competition batches.
3. **Event is the primary temporal semantic retrieval unit; frame is the precise localization/output unit.**
4. **No HCMAI-supervised attention model is trained from the ~20 labels.**
5. **Query-conditioned scoring is initially training-free**, using frozen encoders, temporal aggregation, and optional query decomposition/late interaction.
6. **Event boundaries are soft** and permit bounded neighbor expansion.
7. **Custom `frame_idx` is generated only through one validated submission-coordinate mapper.**
8. **Legacy cleanup is staged**, preserving useful TransNet/GEBD/decode primitives and the BTC baseline while retiring rollback-only runtime paths after replacement tests pass.
9. **Remaining competition batches are release milestones**, with Batch 2 focused on coverage, Batch 3 on event retrieval, and Batch 4 on exploitation/reranking rather than architectural redesign.

After approval of this spec, the next Superpowers step is to write an implementation plan. No implementation should begin before that approval.
