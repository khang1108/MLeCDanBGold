# C++/FFmpeg Deterministic 1-FPS Extraction Design

**Date:** 2026-08-24  
**Status:** Design for user review  
**Scope:** HCMAI custom raw-video timeline extraction and its per-video artifact lifecycle

## Decision summary

The custom timeline will be extracted by a Linux C++17 executable linked to
FFmpeg. The executable owns source acquisition, decoding, timestamp selection,
image encoding, custom frame metadata, and per-video checkpoint/state. Python
remains responsible for validating the native manifest and materializing the
canonical `frames.parquet` table used by the existing repository.

The full corpus is the 873-video `watch_url` manifest in
`data/media-info-aic25-b1/media-info`. The current local metadata estimates
470,428 seconds and 470,428 samples at 1 FPS. The old 1.57M-second statement in
the 22 Aug draft is superseded for this run by the measured local manifest.

The durable image representation is a configurable retrieval-sized JPEG. The
initial configuration is a 1024-pixel maximum long edge at JPEG quality 92.
Source-resolution images used for OCR are temporary and are deleted only after
the per-video enrichment stages have completed and their artifacts have been
verified.

## Goals

- Produce one deterministic visual observation per target second.
- Preserve actual FFmpeg PTS/time-base metadata and canonical identity.
- Generate the required custom submission coordinate:

  ```text
  frame_idx = floor(ceil(avg_fps_of_video) * timestamp_ms / 1000)
  ```

- Make extraction resumable at video granularity without reusing partial
  output as if it were complete.
- Bound scratch storage by processing one video or a small explicit batch at a
  time.
- Keep the raw video available until per-video enrichment has consumed any
  temporary high-resolution frames.
- Keep BTC keyframes and BTC frame mappings untouched.
- Leave the final Parquet/schema/index contracts in Python so existing runtime
  code does not need an Arrow C++ dependency.

## Non-goals

- Adaptive keyframe selection, DINO deduplication, motion-based gap repair, or
  2-FPS extraction.
- Event-boundary detection or temporal scene fusion.
- Caption, OCR, Object, ASR, embedding, or FAISS implementation inside the C++
  binary.
- Writing Parquet directly from C++.
- Replacing the BTC frame store or modifying BTC `frame_idx` values.
- Online video decoding or artifact generation during search serving.

## Current-repository delta

The 22 Aug plan described a Python/PyAV producer in preprocessing modules that
were removed by the later BTC-native refactor. The current checkout retains
only lightweight helpers in `src/hcmai/data/preprocessing/video.py`; it has no
active raw-video producer, C++ build system, or custom timeline configuration.

This design therefore replaces the plan's Task 2 implementation boundary:

```text
old plan:  Python/PyAV producer -> FrameRecord/Parquet
new plan:  C++/FFmpeg producer -> JSONL manifest -> Python validation/Parquet
```

The current `FrameMeta.frame_idx` calculation, which rounds native timestamp
times FPS, is decode metadata only and must not be reused as the custom
submission coordinate.

## Runtime architecture

```text
media-info JSON manifest
        |
        v
C++ native runner
  - invokes yt-dlp with an explicit argv (no shell interpolation)
  - downloads one source video to a `.part` path
  - decodes one video stream with FFmpeg
  - samples target timestamps at 1-second intervals
  - writes durable JPEGs and temporary enrichment frames
  - writes per-video state and frames.jsonl atomically
        |
        v
per-video native manifest + images
        |
        v
per-video Caption/OCR/Object/ASR stages
        |
        v
Python validator/materializer
  - validates native identity and output files
  - builds FrameRecord rows
  - publishes frames.parquet and corpus manifest
        |
        v
embedding/index stages
```

The C++ runner may stop after `enrichment_pending` for an extraction-only
smoke run, but it must retain the source and temporary staging directory in
that mode. Full-corpus cleanup is allowed only after enrichment completion and
artifact verification.

## C++/Python ownership boundary

### C++ owns

- FFmpeg format, codec, scaler, and JPEG encoder lifecycle;
- source download subprocess lifecycle and source-file cleanup;
- average FPS and PTS extraction;
- deterministic frame selection;
- `frame_id` and custom `frame_idx` generation;
- durable image and temporary enrichment-image paths;
- per-video state transitions and atomic checkpoint writes;
- native `frames.jsonl` publication.

### Python owns

- conversion of media-info JSON into the native input manifest;
- validation of C++ JSONL rows against `FrameRecord`;
- verification of the C++-generated `frame_idx` formula without replacing it;
- atomic corpus-level Parquet/manifest publication;
- downstream specialist enrichment adapters, evidence stores, embeddings, and
  indexes.

The C++ manifest is the offline authority for custom `frame_idx`. Python
recomputes the formula only as a rejection check, so a C++/Python rounding
disagreement cannot silently change the submission coordinate.

## Native input manifest

Python will generate a deterministic JSONL input manifest from the 873
media-info records. Each row contains:

```json
{
  "video_id": "L21_V001",
  "watch_url": "https://youtube.com/watch?v=...",
  "metadata_length_s": 1262
}
```

The video ID is the media-info filename stem. Duplicate video IDs or URLs are
rejected before native execution. The local metadata length is a planning
fallback; the downloaded stream's duration and average FPS are authoritative
for the extracted rows.

## Sampling and timestamp contract

The native decoder uses the first video stream and FFmpeg presentation
timestamps. For each decoded frame:

1. Read `best_effort_timestamp`; fail the video if no usable timestamp exists.
2. Convert the timestamp to milliseconds with FFmpeg rational rescaling using
   nearest rounding and an explicit `AV_ROUND_PASS_MINMAX` policy.
3. Maintain the next target timestamp `0, 1000, 2000, ...`.
4. When the decoded timestamp crosses a target, compare the previous and
   current decoded frames and select the one nearest to the target.
5. On an exact tie, select the earlier frame.
6. Emit each target at most once and advance targets until the next target is
   ahead of the current decoded timestamp.

The native row records both `target_timestamp_ms` and the selected frame's
actual `timestamp_ms`. `FrameRecord.timestamp_ms` receives the actual selected
frame timestamp. The target is retained in the native manifest for audit and
sampling validation.

Sampling stops when the next target is outside the decoded stream duration.
For a positive duration `D`, the expected target count is `ceil(D)`.

## FPS and submission identity

The source FPS is read from `AVStream::avg_frame_rate` as a rational. If it is
missing, non-positive, or non-finite after conversion, the video fails rather
than receiving an invented FPS. The raw numerator, denominator, and converted
value are retained in the native manifest.

For every emitted row, C++ computes:

```text
frame_idx = floor(ceil(avg_fps) * timestamp_ms / 1000)
```

The custom `frame_id` is independent of this coordinate:

```text
{video_id}_raw1fps_{sample_index:09d}
```

`sample_index` is the deterministic target order, not a retrieval-array
position. Multiple custom frames may share a `frame_idx`; they must not be
deduplicated by that field.

The native JSONL row contains at least:

```text
frame_id
video_id
sample_index
target_timestamp_ms
timestamp_ms
frame_idx
avg_fps
avg_fps_num
avg_fps_den
pts
time_base_num
time_base_den
width
height
image_path
```

The Python materializer maps these fields to the existing `FrameRecord`, with
`keyframe_order=None`, raw `fps=avg_fps`, and no selector-derived
`shot_id`/`event_id`/`selection_reasons` values.

## Image policy

The C++ extractor writes two representations only while a video is active:

### Durable retrieval image

- JPEG;
- maximum long edge: 1024 pixels;
- initial quality: 92;
- path is retained in `FrameRecord.image_path`;
- used by Caption, Objects, and visual embedding by default.

### Temporary enrichment image

- source-resolution or configured high-resolution JPEG;
- quality: 95;
- not referenced by the final `FrameRecord`;
- used for OCR when the durable image would remove small text;
- deleted after the per-video enrichment artifacts pass validation.

The 1024/92 values are starting configuration, not accuracy claims. A pilot
will compare durable versus temporary inputs using actual Caption/OCR/Object/
visual models before the full corpus run. The decision metric for OCR is
usable text/region recall and confidence diagnostics; visual embedding uses
embedding cosine similarity and retrieval ranking on a hand-checkable sample.

## State and checkpoint contract

C++ owns one state file per video. State writes use a temporary file followed
by an atomic rename. A completed state is never inferred from the existence of
one image; the manifest, image count, checksums/byte sizes, and status marker
must all validate.

Allowed lifecycle states are:

```text
pending
  -> downloading
  -> extracting
  -> extracted
  -> enrichment_pending
  -> enriched
  -> published
  -> cleaned
```

Any active state may transition to `failed` with an error payload. A failed or
interrupted extraction is restarted at video granularity from a clean staging
directory. The state records `last_completed_sample_index` for diagnosis, but
the first implementation does not seek into a partially decoded GOP to resume
mid-video.

Each state file records:

```text
run_id
video_id
watch_url
source_path
extractor_version
config_hash
status
started_at
updated_at
last_completed_sample_index
emitted_frame_count
native_manifest_path
enrichment_manifest_path
error
```

The `enrichment_pending -> enriched` transition is performed only after the
per-video Caption/OCR/Object/ASR artifact manifests pass identity and count
validation. Python enrichment never edits the state JSON directly. It invokes
the native executable's state-transition command with the validated artifact
manifest; the executable checks the allowed predecessor state and performs the
atomic write. The same native transition command is used to move an enriched
video through `published` and `cleaned`. Cleanup removes the raw source and
temporary high-resolution images only after the `published` transition
succeeds.

The native command surface includes:

```text
extract --manifest <path> --run-root <path>
state mark-enriched --run-root <path> --video-id <id> --artifacts <manifest>
state mark-published --run-root <path> --video-id <id> --manifest <manifest>
state cleanup --run-root <path> --video-id <id>
```

Each state command is idempotent for the same validated manifest and rejects
invalid transitions or a changed extractor/config hash.

## Storage layout

The run root is remote scratch/durable storage on ThunderCompute:

```text
run_root/
├── input/media_manifest.jsonl
├── state/{video_id}.json
├── source/{video_id}.part
├── staging/{video_id}/
│   ├── images/
│   ├── enrichment_images/
│   └── frames.jsonl
├── published/{video_id}/
│   ├── images/
│   ├── frames.jsonl
│   └── manifest.json
└── corpus/
    ├── frames.parquet
    └── manifest.json
```

Only one video or an explicitly bounded batch may occupy `source/` and
`staging/` concurrently. Publication moves or copies only verified outputs to
durable storage. The source video is never copied into the durable frame-store
image directory.

## Error handling and observability

- A failed URL/download records stderr and exits the video with `failed`; other
  videos continue unless `--fail-fast` is explicitly enabled.
- A missing PTS, invalid FPS, decoder error, image-encoder error, or manifest
  mismatch prevents publication for that video.
- Every state transition records timestamps and the extractor/config version.
- A run summary reports completed, failed, skipped, and pending videos plus
  emitted frame counts.
- The corpus publisher refuses to publish a global `frames.parquet` if any
  selected video lacks a validated native manifest.

## Build and test strategy

The native component will be an isolated CMake project under
`src/hcmai/data/cpp/keyframes_extraction/`. It is source-owned by the data
layer, but it is not imported as a Python module at runtime. It will use
`pkg-config` to locate FFmpeg's `libavformat`, `libavcodec`, `libavutil`, and
`libswscale`. It will not add an Arrow C++ dependency.

The proposed native package layout is:

```text
src/hcmai/data/cpp/keyframes_extraction/
├── CMakeLists.txt
├── include/hcmai/keyframes_extraction/
├── src/
└── tests/
```

Tests will include:

- C++ frame-index tests for the exact custom formula, including 25, 29.97, and
  30 FPS inputs;
- C++ timestamp-selection tests for previous/current nearest-frame behavior,
  exact ties, monotonic targets, and duplicate prevention;
- a synthetic FFmpeg video integration test proving three 1-FPS samples,
  encoded images, JSONL rows, and expected state transitions;
- interruption/failure tests proving incomplete staging is not published and a
  rerun starts safely at video granularity;
- Python JSONL-to-`FrameRecord` validation tests, including `keyframe_order=None`
  and mismatch rejection;
- a ThunderCompute smoke benchmark comparing durable-image and temporary
  high-resolution enrichment inputs before full-corpus execution.

The first implementation gate is a small representative video set. Full
corpus execution is allowed only after the smoke run validates counts,
timestamps, image readability, state resume, and storage growth.

## Compatibility and rollout

The BTC frame store remains the active baseline and is not modified. The
custom corpus receives a distinct frame-store/run identity and is later exposed
to retrieval as another `RetrievalSource.VISUAL` corpus. Existing online code
continues to consume canonical `FrameRecord` rows and stored `frame_idx` values.

The rollout order is:

1. Build and test the C++ extractor on a synthetic video.
2. Run a one-video ThunderCompute smoke test with temporary OCR-image policy.
3. Run a small representative pilot and measure storage/model quality.
4. Publish the complete custom FrameStore with resume enabled.
5. Run full per-video Caption/OCR/Object/ASR enrichment.
6. Build embeddings and indexes only after the FrameStore and evidence
   manifests pass validation.
