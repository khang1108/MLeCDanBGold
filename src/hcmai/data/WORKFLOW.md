# Data Ingestion and Enrichment Workflows

This document outlines the BTC-native workflow for producing the canonical
FrameStore and its multimodal enrichments in the `hcmai` system. BTC remains
the active competition baseline. Custom raw-video extraction is an explicitly
separate offline workflow with its own frame_store_id and never replaces BTC
artifacts in place.

## 1. Canonical FrameStore Ingestion

`import_btc_frame_store` validates BTC metadata, joins the organizer's
keyframe mapping, preserves `video_id`, `frame_id`, `frame_idx`, and
`timestamp_ms`, and atomically publishes `frames.parquet` plus its manifest.
The resulting FrameStore is the source of truth for Caption, OCR, Object, and
retrieval stages.

### Ingestion Strategy

1. **BTC metadata validation**: verify required canonical identity fields and
   the keyframe-to-submission mapping.
2. **Canonical publication**: write the validated frame table and manifest
   atomically without redetecting or selecting replacement frames.
3. **Timeline evidence**: read source videos only for timestamped ASR; ASR is
   kept separate from frame-native evidence.

### BTC-Native Workflow Diagram

```mermaid
flowchart LR
    BTCFrames["BTC keyframes + mapping"] --> FrameStore["Canonical frames.parquet"]
    BTCObjects["BTC object JSON"] --> Objects["Object evidence"]
    FrameStore --> Caption["Caption"]
    FrameStore --> OCR["OCR"]
    FrameStore --> Context["FrameContext V1"]
    Objects --> Context
    SourceVideos["Source videos"] --> ASR["Timestamped ASR"]
```

---

## 2. Enrichment Workflows

Enrichments add searchable text layers (Caption, OCR, ASR) that are rigidly aligned to the canonical `frame_id` coordinates.

### 2.1 Caption & OCR Pipeline

Caption and OCR share a very similar pipeline architecture (`EnrichmentService`).

- **Captioning**: Feeds canonical images to Qwen VL via `QwenVLCaptionAdapter` to generate retrieval-oriented descriptive natural language.
- **OCR**: Uses optical character recognition (e.g., `FlorenceAdapter`) to extract dense text blocks found on the screen.

**Algorithm:**
1. Loads the canonical `frames.parquet` to locate every selected image path.
2. Iterates over frame batches and sends them to the specialized adapter.
3. Structures the response alongside `frame_id`, model name, and processing status.
4. Writes an atomic `frame_enrichment.parquet`.

```mermaid
flowchart LR
    subgraph Canonical Data
        FramesParquet[frames.parquet]
        ImagesDir[Frame Images/]
    end

    subgraph Enrichment Service
        Batches[Batching Strategy]
        Adapter{Adapter Selection}
        VLMCaption[Qwen VL Caption Adapter]
        OCRModel[Florence OCR Adapter]
    end

    subgraph Artifacts
        CaptionParquet[caption/frame_enrichment.parquet]
        OCRParquet[ocr/frame_enrichment.parquet]
    end

    FramesParquet --> Batches
    ImagesDir --> Batches
    Batches --> Adapter
    Adapter --> VLMCaption
    Adapter --> OCRModel
    
    VLMCaption --> CaptionParquet
    OCRModel --> OCRParquet
```

### 2.2 ASR (Audio Speech Recognition) Pipeline

The ASR pipeline is inherently temporal and handles continuous audio streams, distinguishing it from independent frame-based inferences.

**Algorithm:**
1. **Audio Extraction & ASR**: The pipeline runs models like Whisper/FastWhisper (`ASRAdapter`) across the entire video audio track to extract raw transcripts.
2. **Diarization (Optional)**: Segments are augmented with speaker identities (`DiarizationAdapter`).
3. **Temporal Materialization**: 
   - ASR outputs are fundamentally independent of frame sampling rates.
   - `materialize_asr_enrichment()` aligns the half-open temporal ASR segments to the canonical frames.
   - For every canonical frame, a time window (`timestamp_ms ± window_ms`) is calculated.
   - Any ASR transcript overlapping that window is collected, deduplicated (ignoring casing and whitespace anomalies), and attached to the `frame_id`.
4. **Validation & Storage**: The output exactly mirrors the canonical frame order and is published atomically.

```mermaid
flowchart TD
    Video["Raw Video (Audio Track)"]
    ASRAdapter["ASR Adapter (Whisper)"]
    Diarization["Diarization Adapter (Optional)"]
    RawTranscripts["Raw Transcript Segments (JSON)"]
    
    FramesParquet["Canonical frames.parquet"]
    Materializer["Temporal Materialization (± window_ms)"]
    Deduplication["Text Normalization & Deduplication"]
    ASREnrichment["asr/frame_enrichment.parquet"]

    Video --> ASRAdapter
    ASRAdapter --> Diarization
    Diarization --> RawTranscripts
    
    RawTranscripts --> Materializer
    FramesParquet --> Materializer
    Materializer --> Deduplication
    Deduplication --> ASREnrichment
```

## Immutable Integrity Guarantee

Across all three pipelines (Preprocessing, Caption/OCR, ASR), data integrity is rigidly enforced:
- **Foreign Key Stability**: Enrichments strictly depend on `frame_id`. If an enrichment process observes an unknown `frame_id` or reorders the dataset, the write is aborted.
- **Atomic Operations**: All artifacts (Parquet tables and JSON manifests) write to a `.partial` staging path and are defensively moved/renamed at the end to prevent corruption during an interrupted run.

---

## 3. Custom raw-video 1-FPS workflow

The custom workflow is opt-in and bounded at video granularity. It preserves
`video_id`, internal `frame_id`, competition-facing `frame_idx`, and actual
`timestamp_ms` throughout native extraction, enrichment, and materialization.
`frame_idx` is always checked against
`floor(ceil(avg_fps) * timestamp_ms / 1000)`; it is never replaced with
keyframe order or an array position.

1. Run `extract_custom_keyframes.py` with a media-info folder and an explicit
   `--video-id`, `--limit`, or `--all` selection. It writes the sorted metadata
   manifest/config and invokes native extraction one video at a time.
2. Native extraction downloads each selected source with shell-free `yt-dlp`,
   retains it for ASR, and creates durable JPEGs plus temporary high-resolution
   OCR JPEGs in `staging/{video_id}`. The wrapper validates the bundle and emits
   durable/OCR FrameRecord tables for specialist enrichment.
3. Validate the staging bundle. Caption, Object, and visual stages consume the
   durable table; OCR consumes the high-resolution temporary table; ASR reads
   the retained source video as independent timeline evidence.
4. `write_enrichment_handoff` validates Caption/OCR/Object frame identity and
   ASR video/timeline identity. Each modality may be explicit
   `not_evaluated`; that status is not treated as negative evidence.
5. Call `mark_video_enriched`, `mark_video_published`, then `cleanup_video`.
   Only the native executable writes the state JSON. Cleanup retains
   `published/{video_id}` durable images and native manifest, while removing
   the selected video's `published/{video_id}/enrichment_images` OCR scratch
   directory.
6. Run `materialize_custom_frames.py` only over validated published IDs. The
   global custom FrameStore contains durable image paths only, with the custom
   run root configured as its dataset root for downstream image consumers.

The full corpus remains gated by the synthetic/local acceptance test and a
separate operational pilot decision. The custom workflow does not authorize
unbounded download, model inference, or cloud execution from serving paths.
