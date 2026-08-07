# AIC 2026 Frame Preprocessing Architecture

## Status

This document defines the proposed frame preprocessing architecture for AIC
2026. It is an implementation target, not proof that every component already
exists in the repository.

The current data pipeline imports organizer-provided keyframes and canonical
mapping files into `frames.parquet`. It does not yet run TransNetV2, optical
flow, DINOv3, or adaptive frame extraction on raw videos.

## Objective

Build one compact, informative frame collection from approximately 120 GB of
raw video and reuse it for Textual KIS, Q&A, and TRAKE.

The architecture must:

- avoid storing every decoded frame;
- preserve short events and important state changes;
- keep canonical video and frame identity;
- support fast multilingual retrieval;
- allow TRAKE to recover exact frames from raw video;
- avoid separate image banks for individual tasks.

## System Overview

```text
                         OFFLINE

Raw Videos
    |
    v
Sequential Decode
    |
    +-- TransNetV2 ----------- shot-boundary signal
    +-- Optical Flow --------- motion signal
    +-- Dynamic Maximum Gap -- temporal coverage signal
    |
    v
Candidate Frames
    |
    v
DINOv3 Features
    |
    v
Semantic Deduplication
    |
    v
SigLIP2 Embeddings
    |
    v
Shared Informative Frame Store
    |
    +-- FAISS or Qdrant ------ vector retrieval
    +-- Canonical Metadata --- identity and temporal lookup


                         ONLINE

Query
    |
    +-- KIS   -> SigLIP2 retrieval -> rerank -> frame_id
    +-- Q&A   -> SigLIP2 retrieval -> evidence selection -> VLM -> answer
    +-- TRAKE -> region retrieval -> temporal matching
                 -> native-FPS local decode -> exact ordered frames
```

FAISS and Qdrant are alternative vector backends. A deployment selects one;
the pipeline does not require both simultaneously.

## Offline Stage 1: Candidate Selection

Each video is decoded sequentially once. Frames are analyzed during decoding,
but only selected candidates are retained.

### TransNetV2

TransNetV2 detects shot transitions. Frames around detected boundaries are
protected because a new shot usually introduces new visual information.

It is a shot-boundary detector, not a complete semantic event detector.

### Optical Flow

Optical flow measures motion between adjacent frames. Motion peaks increase
sampling density around actions and state changes. Low-motion intervals can be
sampled more sparsely.

The implementation should use a configurable modern optical-flow model. Model
name, resolution, batch size, thresholds, and device must not be hardcoded.

### Dynamic Maximum Gap

The maximum allowed time between retained candidates changes with local video
dynamics:

- high motion produces a shorter maximum gap;
- low motion permits a longer maximum gap;
- reaching the current gap always creates a candidate.

This is the recall safety net for intervals where the learned signals are weak.

### Candidate Rule

```text
candidate = shot_boundary
         OR motion_peak
         OR elapsed_time >= dynamic_maximum_gap
```

Thresholds and minimum/maximum gaps are configuration values selected through
corpus experiments. They are not fixed architectural constants.

## Offline Stage 2: Semantic Reduction and Indexing

### DINOv3 Semantic Deduplication

DINOv3 produces visual representations for candidate frames. Nearby candidates
with very similar representations are grouped as semantic duplicates, while
boundary-protected and temporally important candidates remain eligible to be
kept.

Deduplication must be deterministic. Its similarity threshold and
representative-selection policy must be recorded with the generated artifact.

### SigLIP2 Retrieval Embeddings

Every retained frame receives a SigLIP2 image embedding. Query text is encoded
with the matching SigLIP2 text encoder, providing one multilingual image-text
space for Vietnamese and English retrieval.

DINOv3 and SigLIP2 have different responsibilities:

- DINOv3 decides whether candidate images are visually redundant.
- SigLIP2 makes retained images searchable with natural-language queries.

### Shared Informative Frame Store

The shared store is the single frame source used by all tasks. Its logical
record contains at least:

```text
frame_id
video_id
official frame_idx, when supplied by canonical organizer mapping
timestamp_ms
PTS or another source decode position
image_path
shot-boundary and motion metadata
```

The vector index stores or references SigLIP2 embeddings by `frame_id`.
Metadata resolves each retrieved `frame_id` back to the canonical frame record.

Never derive an official `frame_idx` from timestamp, FPS, filename, array
position, or internal `frame_id`. Submission identifiers must come from the
canonical organizer mapping.

## Online Task Pipelines

### Textual KIS

```text
Text query -> SigLIP2 search -> task reranking -> canonical frame result
```

The goal is to rank representative frames that best match the description.

### Q&A

```text
Question -> SigLIP2 search -> evidence selection -> VLM -> answer
```

Retrieval first finds the relevant video region. Evidence selection then keeps
the frames that visibly support the answer. The VLM consumes only this bounded
evidence set.

### TRAKE

```text
Event descriptions -> region retrieval -> same-video temporal matching
                   -> native-FPS local decode -> exact ordered frames
```

The frame store provides coarse temporal localization. Raw video is decoded
only within small candidate intervals to recover short events and exact frames.
All predicted events must remain in one video and preserve event order.

## Component Responsibilities

| Component | Responsibility |
| --- | --- |
| TransNetV2 | Detect shot transitions |
| Optical flow | Measure local motion and action intensity |
| Dynamic maximum gap | Prevent temporal blind spots |
| DINOv3 | Remove visually redundant candidate frames |
| SigLIP2 | Produce multilingual image-text retrieval embeddings |
| FAISS or Qdrant | Return candidate `frame_id` values quickly |
| Shared Frame Store | Preserve canonical identity, paths, and timestamps |
| Raw video | Recover exact frames through local native-FPS decoding |
| Task rerankers | Apply KIS, Q&A, or TRAKE-specific ranking constraints |

## Hardware Model

Offline preprocessing runs in batches on an NVIDIA L40 or A6000 GPU. Models are
loaded once and videos are processed sequentially, so the complete corpus is
never loaded into memory at the same time.

Online retrieval searches the prepared index and metadata. GPU work is reserved
for measured reranking, VLM inference, or short local video refinement.

## Evaluation Requirements

Do not select models or thresholds by model age or public benchmark alone.
Evaluate this pipeline on representative AIC videos using:

- retained-frame and storage ratio;
- ground-truth interval coverage;
- short-event recall;
- official Mean Top-k R-Score at 1, 5, 20, 50, and 100;
- TRAKE video, per-event, and full-sequence accuracy;
- warm P50/P95 retrieval and end-to-end latency;
- failures and unexplained canonical-mapping mismatches.

The selected competition configuration must record all model checkpoints,
thresholds, predictions, metrics, and latency results.

