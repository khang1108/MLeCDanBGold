# AIC 2026 End-to-End Video Retrieval Implementation Plan

> **Execution rule:** Implement task-by-task in dependency order. Keep each stage resumable and artifact-driven. Do not couple expensive model stages to core data contracts. Benchmark/ablation is intentionally deferred; only contract tests, smoke tests, and end-to-end sanity checks are required in this plan.

## Goal

Build a production-target AIC 2026 system that starts from raw news videos and supports all three preliminary-round tasks:

- **KIS:** query-time uses only the precomputed `FrameStore` and indexed evidence.
- **Q&A / VQA:** query-time uses only the `FrameStore` plus precomputed OCR, ASR, caption, and retrieval indexes.
- **TRAKE:** use the `FrameStore` for coarse video/region retrieval, then use `TemporalMicroIndex` + raw-video local decode to refine each event to an exact native frame.

The primary engineering objective is not “minimum frame count.” It is:

> Preserve enough visual/textual/temporal evidence for KIS and Q&A while keeping a compact searchable frame bank, and preserve enough temporal structure to let TRAKE zoom back into raw video efficiently.

The official AIC 2026 preliminary-round document makes the critical constraints explicit: KIS/Q&A require a submitted `frame_id` inside the accepted ground-truth interval, while TRAKE requires one ordered semantic keyframe per event and notes that those semantic intervals are commonly shorter than 10 frames.

---

## Architecture

```text
                                  OFFLINE

 Raw Videos ──────────────────────────────────────────────────────────────┐
    │                                                                    │
    │                                                                    ├── Audio
    │                                                                    │    ↓
    │                                                                    │ Timestamped ASR
    │                                                                    │    ↓
    │                                                                    │ Transcript artifacts
    │                                                                    │
    ↓                                                                    │
 Native-temporal sequential decode                                      │
 low spatial resolution                                                  │
    │                                                                    │
    ├── Always-on cheap signals                                          │
    │     ├── global visual change                                       │
    │     ├── regional visual change                                     │
    │     ├── edge/layout change                                         │
    │     ├── cheap text-region change                                   │
    │     ├── codec motion/residual when available                       │
    │     └── blur/quality signal                                        │
    │                                                                    │
    ├── TransNetV2 shot structure                                        │
    ├── ESTimator event-boundary proposals                               │
    │                                                                    │
    ↓                                                                    │
 Candidate Window Generator                                              │
    ├── shot burst                                                        │
    ├── event burst                                                       │
    ├── motion / appearance burst                                        │
    ├── text-change burst                                                 │
    └── hard Maximum-Gap coverage floor                                  │
    │                                                                    │
    ↓                                                                    │
 Candidate Frames                                                        │
    │                                                                    │
    ↓                                                                    │
 Conservative local dedup                                                │
    ├── DINOv2 semantic similarity                                       │
    ├── regional/edge/text/motion equivalence                            │
    └── protected frames are never removed                               │
    │                                                                    │
    ↓                                                                    │
 Full-quality materialization                                            │
    │                                                                    │
    ├── SigLIP2 image embeddings                                         │
    ├── OCR                                                               │
    ├── selective captions                                                │
    └── ASR alignment ◄───────────────────────────────────────────────────┘
    │
    ↓
 Canonical FrameStore
    │
    ├── Visual index
    ├── OCR index
    ├── ASR index
    ├── Caption index
    ├── Story/shot index
    └── Temporal Micro-Index

                                  QUERY TIME

 KIS
   query → multimodal retrieval → RRF fusion → rerank → diversify → FrameStore result

 Q&A
   event + question → event retrieval → local FrameStore evidence
   → OCR/ASR/caption/VLM reasoning → frame + answer

 TRAKE
   ordered events → batch multimodal retrieval → candidate videos
   → temporal beam/DP path → Temporal Micro-Index
   → GOP seek → coarse local decode → native-FPS zoom
   → exact ordered frame sequence
```

---

## Evidence-backed design decisions

These are architectural decisions, not claims that the cited paper directly solves AIC 2026.

| Decision | Evidence | Why it transfers |
|---|---|---|
| Keep shot boundaries as a structural signal, not an event detector | **TransNet V2**, Souček & Lokoč, shot-transition detection | Useful for news shot structure and to forbid dedup across cuts; does not solve same-shot semantic moments. |
| Add generic event-boundary proposals | **ESTimator / Online Generic Event Boundary Detection, ICCV 2025** | Closer to semantic event transitions than shot boundaries; use only to open candidate bursts, not as exact TRAKE keyframes. |
| Use cheap compressed-domain temporal cues when possible | **End-to-End Compressed Video Representation Learning for GEBD, CVPR 2022** | Motion vectors, residuals, and GOP structure retain useful temporal information at much lower cost than full dense flow. |
| Preserve coverage in addition to informativeness | **KFS-Bench, WACV 2026** | Shows frame-sampling quality depends on coverage/balance, not only per-frame precision. Supports a hard coverage floor. |
| Use adaptive motion-sensitive sampling only as a cue | **MGSampler, ICCV 2021** | Shows motion-guided sampling can improve action recognition over fixed sampling, but its objective is not exact AIC interval retention. |
| Use DINO for local redundancy evidence, not as a sole deletion rule | **LongVU, ICML 2025** uses DINOv2 similarity for temporal compression | Supports semantic redundancy measurement, but AIC KIS/Q&A exact intervals require conservative multi-evidence dedup. |
| Use SigLIP2 for text-image retrieval | **SigLIP 2, 2025** | Multilingual image-text representation is well aligned with Vietnamese textual retrieval. |
| Treat scene text as an independent evidence channel | **TextVQA, CVPR 2019; ST-VQA, ICCV 2019; EgoTextVQA, CVPR 2025** | News videos contain lower-thirds, names, locations, numbers, and graphics that may change with little global motion. |
| Make ASR a first-class retrieval modality | **TVR, ECCV 2020; QVHighlights/Moment-DETR, NeurIPS 2021** | Spoken content often identifies news stories/events even when visuals are generic b-roll. |
| Add story-level hierarchy for long news programs | **NewsNet, CVPR 2023** | Long-form news naturally contains multiple semantic stories above the shot level; story priors can improve coarse retrieval without hard filtering. |
| Use coarse-to-fine raw-video refinement for TRAKE | **Re-thinking Temporal Search / T*, CVPR 2025** | Sparse retrieval can locate a candidate temporal region; adaptive zoom is suitable for finding needle-like moments at native frame resolution. |

---

## Existing repository: reuse instead of rewrite

The current source already contains useful boundaries. The new design should extend them rather than replace the whole codebase.

### Keep and extend

- `src/hcmai/common/schemas/frame.py`
  - Keep `FrameRecord` as the canonical searchable-frame contract.
  - Add auxiliary preprocessing schemas instead of bloating every runtime record.

- `src/hcmai/data/stores/frame.py`
  - Keep `FrameStore` as the runtime authority for KIS/Q&A.
  - Continue using `get()`, `get_many()`, `get_neighbors()`, `filter_frame_ids()`.

- `src/hcmai/data/pipeline.py`
  - Keep `DataService` as the public data facade.
  - Replace only the legacy `prepare()` path after the new preprocessor is stable.

- `src/hcmai/embedding/`
  - Reuse SigLIP adapters/artifact builders.

- `src/hcmai/data/enrichment/transcripts/`
  - Reuse `TranscriptService`, ASR adapter, diarization, and store.

- `src/hcmai/enrichment/ocr/` and `src/hcmai/enrichment/caption/`
  - Reuse adapters, resume logic, artifact generation, and reports.

- `src/hcmai/retriever/`
  - Reuse dense indexes, text retrievers, batching, cache, and `RRFFusionRetriever`.
  - Existing RRF already supports visual/caption/OCR/ASR fusion.

- `src/hcmai/orchestration/pipelines/kis.py`
  - Reuse current KIS orchestration/materialization path.

- `src/hcmai/common/schemas/trake.py`
  - Keep current public TRAKE request/response contracts.
  - Add internal event/path schemas separately.

### New subsystem

```text
src/hcmai/preprocessing/
├── __init__.py
├── config.py
├── pipeline.py
├── artifacts.py
├── video/
│   ├── probe.py
│   ├── decoder.py
│   ├── identity.py
│   └── seek.py
├── signals/
│   ├── global_change.py
│   ├── regional_change.py
│   ├── edge_change.py
│   ├── text_change.py
│   ├── compressed_motion.py
│   └── quality.py
├── shot/
│   └── transnet.py
├── event/
│   └── estimator.py
├── sampling/
│   ├── candidate_selector.py
│   ├── burst.py
│   └── coverage.py
├── dedup/
│   ├── encoder.py
│   └── deduplicator.py
├── materialize.py
└── micro_index.py
```

### New task modules

```text
src/hcmai/orchestration/pipelines/
├── kis.py                  # existing, extend only where needed
├── vqa.py                  # new/complete implementation
└── trake.py                # new executable pipeline

src/hcmai/trake/
├── __init__.py
├── parser.py
├── retrieval.py
├── video_ranker.py
├── path_search.py
├── refine.py
├── micro_index.py
└── diversify.py

src/hcmai/story/
├── __init__.py
├── segmenter.py
├── artifacts.py
└── retriever.py
```

---

# Task 1 — Add preprocessing contracts and configuration

**Files**

- Create: `src/hcmai/preprocessing/__init__.py`
- Create: `src/hcmai/preprocessing/config.py`
- Create: `src/hcmai/preprocessing/artifacts.py`
- Modify: `src/hcmai/common/config.py`
- Create: `tests/preprocessing/test_config.py`
- Create: `tests/preprocessing/test_artifacts.py`

## Step 1: Define config contracts

Add config groups for:

```python
PreprocessingConfig
AnalysisConfig
ChangeSignalConfig
TextChangeConfig
ShotDetectionConfig
EventDetectionConfig
CoverageConfig
BurstConfig
DedupConfig
MaterializationConfig
MicroIndexConfig
```

Minimum fields:

```python
class AnalysisConfig(BaseModel):
    width: int = 320
    height: int = 180
    native_temporal: bool = True

class CoverageConfig(BaseModel):
    maximum_gap_ms: int

class BurstConfig(BaseModel):
    shot_radius_ms: int
    event_radius_ms: int
    motion_radius_ms: int
    text_radius_ms: int
    default_step_ms: int

class DedupConfig(BaseModel):
    model_name: str = "facebook/dinov2-base"
    window_ms: int
    semantic_threshold: float
    regional_threshold: float
    edge_threshold: float
    text_threshold: float
    motion_threshold: float
```

Add `preprocessing: PreprocessingConfig` to `AppConfig` without breaking existing config loading.

## Step 2: Define internal artifact schemas

Use dataclasses/Pydantic only for inter-module contracts; bulk frame rows can be written column-wise.

Required logical records:

```python
VideoRecord
DecodedFrameMeta
FrameSignalRecord
CandidateFrameRecord
MicroIndexRecord
DedupDecision
PreprocessingReport
```

`MicroIndexRecord` minimum fields:

```text
video_id
internal_decode_index
frame_idx
pts
timestamp_ms
gop_seek_pts
shot_id
shot_score
event_score
global_change
regional_change
edge_change
text_change
codec_motion_score
codec_residual_score
blur_score
candidate
protected
protected_reasons
```

## Step 3: Tests

Test:

- configs reject negative gaps/windows;
- `native_temporal=False` is allowed only as an explicit experimental setting, never default;
- serialization round-trips;
- protected reasons preserve multiple triggers.

Run:

```bash
pytest tests/preprocessing/test_config.py tests/preprocessing/test_artifacts.py -q
```

## Acceptance criteria

- Existing `AppConfig.from_yaml()` still loads current config.
- New preprocessing config has no model import side effects.
- All new contracts serialize deterministically.

---

# Task 2 — Probe raw videos and build the video manifest

**Files**

- Create: `src/hcmai/preprocessing/video/probe.py`
- Create: `tests/preprocessing/video/test_probe.py`

## Step 1: Implement `probe_video()`

Preferred backend: `ffprobe` JSON or PyAV stream metadata.

Return:

```python
VideoRecord(
    video_id,
    video_path,
    width,
    height,
    fps_num,
    fps_den,
    duration_ms,
    num_frames,
    codec,
    time_base_num,
    time_base_den,
    has_audio,
)
```

Do not reduce rational FPS to a rounded integer.

## Step 2: Implement corpus manifest command

```python
build_video_manifest(videos_root, output_path)
```

Output:

```text
artifacts/video_manifest.parquet
```

## Step 3: Tests

Use a tiny generated fixture video with known FPS/duration.

Run:

```bash
pytest tests/preprocessing/video/test_probe.py -q
```

## Acceptance criteria

- Correctly distinguishes 25 FPS and 30 FPS videos.
- Stores rational frame rate/time base.
- No assumption that all L21-L30 videos share the same FPS.

---

# Task 3 — Implement canonical frame identity resolution

**Files**

- Create: `src/hcmai/preprocessing/video/identity.py`
- Create: `tests/preprocessing/video/test_identity.py`

## Step 1: Create resolver interface

```python
class FrameIdentityResolver(Protocol):
    def resolve(
        self,
        *,
        video_id: str,
        decode_index: int,
        pts: int | None,
        timestamp_ms: int,
    ) -> int | None: ...
```

Implement:

```python
DecodeOrdinalIdentityResolver
OfficialAnchorValidatedResolver
```

The second wrapper exists so official BTC keyframes/mappings can be used to validate the indexing convention without making official keyframes a production dependency.

## Step 2: Never compute identity as `round(timestamp * fps)`

Timestamp remains navigation metadata; canonical submission identity is explicit.

## Step 3: Tests

- exact decode-index mapping;
- resolver can mark unresolved frames;
- no timestamp-to-index fallback occurs silently.

## Acceptance criteria

All downstream components obtain `frame_idx` only through the resolver.

---

# Task 4 — Native-temporal low-resolution sequential decoder

**Files**

- Create: `src/hcmai/preprocessing/video/decoder.py`
- Create: `tests/preprocessing/video/test_decoder.py`

## Step 1: Implement streaming decoder

Interface:

```python
@dataclass(slots=True)
class AnalysisFrame:
    video_id: str
    decode_index: int
    frame_idx: int | None
    pts: int | None
    timestamp_ms: int
    rgb: np.ndarray


def iter_analysis_frames(
    video: VideoRecord,
    config: AnalysisConfig,
    resolver: FrameIdentityResolver,
) -> Iterator[AnalysisFrame]: ...
```

Requirements:

- decode every presented frame temporally by default;
- resize spatially to configured analysis resolution;
- no disk write for analysis RGB;
- preserve PTS and decode index;
- bounded memory.

## Step 2: Tests

- expected frame count from fixture;
- monotonically nondecreasing timestamps;
- low-res output dimensions;
- no skipped frames in native-temporal mode.

Run:

```bash
pytest tests/preprocessing/video/test_decoder.py -q
```

## Acceptance criteria

A full 15–20 minute video can be streamed without retaining all frames in RAM.

---

# Task 5 — Build Temporal Micro-Index writer/reader

**Files**

- Create: `src/hcmai/preprocessing/micro_index.py`
- Create: `tests/preprocessing/test_micro_index.py`

## Step 1: Writer

Implement incremental buffered writer:

```python
class TemporalMicroIndexWriter:
    append(record)
    flush()
    close()
```

Write one shard per video:

```text
artifacts/temporal_micro_index/<video_id>.parquet
```

## Step 2: Reader

```python
class TemporalMicroIndex:
    load(video_id)
    nearest_timestamp(ms)
    window(start_ms, end_ms)
    nearest_gop_seek(timestamp_ms)
```

## Step 3: Tests

- ordered rows;
- temporal-window lookup;
- nearest seek anchor;
- nullable model scores supported before later stages fill them.

## Acceptance criteria

TRAKE can later locate a small temporal window without scanning global metadata.

---

# Task 6 — Implement always-on global, regional, edge, and quality signals

**Files**

- Create: `src/hcmai/preprocessing/signals/global_change.py`
- Create: `src/hcmai/preprocessing/signals/regional_change.py`
- Create: `src/hcmai/preprocessing/signals/edge_change.py`
- Create: `src/hcmai/preprocessing/signals/quality.py`
- Create: `tests/preprocessing/signals/test_change_signals.py`

## Step 1: Global change

Low-res grayscale mean absolute difference:

```python
global_change(prev, curr) -> float
```

## Step 2: Regional change

Split into configurable grid, e.g. 4×4. Return:

```python
RegionalChangeResult(
    region_scores,
    max_score,
    top_k_mean,
)
```

Do not collapse to only global average.

## Step 3: Edge change

Compute a cheap Sobel/Canny-like edge map and compare adjacent frames.

## Step 4: Blur quality

Use a cheap sharpness metric such as Laplacian variance. It is for representative selection, not candidate rejection.

## Step 5: Tests

Synthetic fixtures:

- global scene change;
- small local square appears;
- text-like edge pattern changes;
- blurred vs sharp image.

## Acceptance criteria

A small local change can trigger regional/edge score even when global score stays low.

---

# Task 7 — Implement independent cheap text-change channel

**Files**

- Create: `src/hcmai/preprocessing/signals/text_change.py`
- Create: `tests/preprocessing/signals/test_text_change.py`

## Step 1: Define configurable ROIs

Support normalized rectangles:

```text
main_content
lower_third
ticker
logo_mask
```

## Step 2: Cheap text/layout representation

V1 should avoid OCR on every native frame. Use a combination of:

- edge density;
- binary text-like mask or lightweight detector if already available;
- crop perceptual hash / SSIM-like similarity;
- separate scores per ROI.

Output:

```python
TextChangeResult(
    main_score,
    lower_third_score,
    ticker_score,
    aggregate_score,
)
```

Ticker must have separate thresholds so moving ticker text does not flood candidates.

## Step 3: Tests

- lower-third appears while main visual remains static;
- ticker-only motion does not trigger the same policy as main text;
- logo region can be ignored.

## Acceptance criteria

Text change can independently open a candidate window without requiring visual-motion/ESTimator triggers.

---

# Task 8 — Add compressed-motion adapter, but keep it optional

**Files**

- Create: `src/hcmai/preprocessing/signals/compressed_motion.py`
- Create: `tests/preprocessing/signals/test_compressed_motion.py`

## Step 1: Adapter contract

```python
class CompressedMotionAdapter(Protocol):
    def analyze_video(...) -> Iterator[CompressedMotionRecord]: ...
```

Fields:

```text
codec_motion_score
codec_residual_score
gop_seek_pts
```

## Step 2: FFmpeg-backed implementation

Implement only if motion-vector/residual extraction is reliable in current environment. Otherwise return `UnavailableCompressedMotionAdapter` and continue without blocking preprocessing.

## Acceptance criteria

The pipeline works with or without compressed motion signals.

---

# Task 9 — Integrate TransNetV2 as shot-structure adapter

**Files**

- Create: `src/hcmai/preprocessing/shot/transnet.py`
- Create: `src/hcmai/preprocessing/shot/contracts.py`
- Create: `tests/preprocessing/shot/test_transnet_adapter.py`

## Step 1: Adapter interface

```python
class ShotDetector(Protocol):
    def detect(video_path) -> ShotDetectionResult: ...
```

Result:

```text
per-frame shot_score
boundaries
shot_id assignment
```

## Step 2: Keep model isolated

All TransNet imports/load calls must be inside adapter implementation so preprocessing contracts remain importable on CPU-only machines.

## Step 3: Candidate behavior

Each boundary opens a configurable burst and marks all burst members:

```text
protected_reason += ["shot_boundary"]
```

## Acceptance criteria

- no dedup across distinct `shot_id`;
- adapter failure can degrade gracefully to cheap change signals rather than abort the corpus.

---

# Task 10 — Integrate ESTimator as event-boundary proposal adapter

**Files**

- Create: `src/hcmai/preprocessing/event/contracts.py`
- Create: `src/hcmai/preprocessing/event/estimator.py`
- Create: `tests/preprocessing/event/test_estimator_adapter.py`

## Step 1: Adapter interface

```python
class EventBoundaryDetector(Protocol):
    def score(...) -> EventDetectionResult: ...
```

Output:

```text
event_score[t]
event_peak_candidates
```

## Step 2: Treat score as proposal only

Never convert event score directly into a semantic answer frame.

Rule:

```text
event_peak → open burst → protect burst
```

## Step 3: Failure handling

If checkpoint/model unavailable:

```text
log warning
continue with cheap signals + shot + coverage
```

## Acceptance criteria

The whole offline pipeline never depends structurally on ESTimator being online.

---

# Task 11 — Candidate window generator and hard coverage floor

**Files**

- Create: `src/hcmai/preprocessing/sampling/coverage.py`
- Create: `src/hcmai/preprocessing/sampling/burst.py`
- Create: `src/hcmai/preprocessing/sampling/candidate_selector.py`
- Create: `tests/preprocessing/sampling/test_candidate_selector.py`

## Step 1: Candidate union

Do not implement one weighted master score.

```python
candidate = any([
    shot_trigger,
    event_trigger,
    global_trigger,
    regional_trigger,
    edge_trigger,
    text_trigger,
    codec_motion_trigger,
    coverage_trigger,
])
```

## Step 2: Hard coverage

Invariant:

```python
if timestamp_ms - last_candidate_ms >= maximum_gap_ms:
    keep(reason="coverage")
```

Adaptive logic may add samples; it must never create a gap larger than `maximum_gap_ms`.

## Step 3: Burst expansion

Trigger-specific radii:

```text
shot → shot_radius_ms
event → event_radius_ms
motion → motion_radius_ms
text → text_radius_ms
```

Multiple reasons merge into one candidate record.

## Step 4: Protected policy

Protected reasons:

```text
shot_boundary
event_boundary
motion_peak
text_change
coverage_anchor
```

## Tests

- every trigger can independently create a candidate;
- union preserves all reasons;
- maximum gap is never violated;
- bursts merge without duplicates.

## Acceptance criteria

Candidate generation is deterministic for the same signal arrays/config.

---

# Task 12 — Conservative DINOv2 local dedup

**Files**

- Create: `src/hcmai/preprocessing/dedup/encoder.py`
- Create: `src/hcmai/preprocessing/dedup/deduplicator.py`
- Create: `tests/preprocessing/dedup/test_deduplicator.py`

## Step 1: Encoder protocol

```python
class FrameSimilarityEncoder(Protocol):
    def encode(images: list[Image.Image]) -> np.ndarray: ...
```

Implement DINOv2 adapter first. Keep DINOv3 swappable later.

## Step 2: Multi-evidence duplicate rule

A pair/group is duplicate only when all safety conditions are satisfied:

```python
same_shot
and temporal_distance <= dedup_window
and dino_similarity >= semantic_threshold
and regional_change <= regional_threshold
and edge_change <= edge_threshold
and text_change <= text_threshold
and motion_change <= motion_threshold
and not protected
```

Do not dedup across shots.

Do not remove any protected candidate.

## Step 3: Representative selection

For non-protected duplicate group choose by lexicographic policy:

```text
more trigger reasons
higher sharpness
better exposure if available
better text readability score if available
closer to group temporal center
lower frame_idx as final deterministic tie-break
```

## Step 4: Audit output

Write:

```text
artifacts/preprocessing/dedup_decisions.parquet
```

Columns:

```text
video_id
removed_frame_idx
representative_frame_idx
semantic_similarity
regional_change
edge_change
text_change
motion_change
reason
```

## Acceptance criteria

Uncertain cases stay retained. Dedup is a storage optimization, never a prerequisite for correctness.

---

# Task 13 — Materialize retained frames and canonical `frames.parquet`

**Files**

- Create: `src/hcmai/preprocessing/materialize.py`
- Modify: `src/hcmai/common/schemas/frame.py`
- Create: `tests/preprocessing/test_materialize.py`

## Step 1: Full-quality extraction

Only retained frames are materialized at source resolution/quality.

Output:

```text
artifacts/frame_images/<group>/<video_id>/<frame_id>.jpg
```

## Step 2: Canonical identity

Generate stable frame ID independent of keyframe order, e.g.:

```text
<video_id>_frame_<frame_idx:09d>
```

Keep `keyframe_order=None` for custom frames.

## Step 3: `FrameRecord`

Reuse current fields:

```text
frame_id
video_id
frame_idx
keyframe_order
timestamp_ms
image_path
thumbnail_path
width
height
shot_id
is_anchor
```

`is_anchor` should mean “strong protected candidate,” not “all extracted frames.”

Keep detailed preprocessing scores/reasons in Micro-Index or an auxiliary artifact rather than making runtime `FrameRecord` huge.

## Step 4: Write canonical artifact

```text
artifacts/frames.parquet
```

## Acceptance criteria

Existing `FrameStore` can load the new custom frames without knowledge of BTC keyframe folder layout.

---

# Task 14 — Create the new preprocessing orchestrator

**Files**

- Create: `src/hcmai/preprocessing/pipeline.py`
- Create: `tests/preprocessing/test_pipeline_smoke.py`

## Step 1: Pipeline stages

```python
PreprocessingPipeline.prepare_video(video_path)
```

Order:

```text
probe
→ decode cheap signals
→ shot detection
→ event detection
→ candidate selection/bursts
→ DINO dedup
→ full-quality materialization
→ micro-index finalize
```

## Step 2: Resumability

Every stage receives a configuration hash and model/version metadata.

Stage outputs must be independently reusable.

Proposed manifest:

```python
ArtifactManifest:
    stage
    version
    config_hash
    source_video_size
    source_video_mtime_ns
    model_name
    checkpoint
    completed
```

## Step 3: CLI entry point

Add a root command later through existing project CLI mechanism, conceptually:

```bash
python -m hcmai.preprocessing --config configs/baseline.yaml --videos data/L21
```

## Acceptance criteria

Interrupting after one video and rerunning does not redo completed work unless relevant config/source changed.

---

# Task 15 — Switch `DataService.prepare()` from legacy BTC keyframes to the new preprocessor

**Files**

- Modify: `src/hcmai/data/pipeline.py`
- Keep legacy code temporarily: `src/hcmai/data/prepare.py`
- Create: `tests/data/test_custom_prepare_integration.py`

## Step 1: Do not delete legacy path yet

Add an explicit preparation mode:

```python
DataService.prepare_raw_videos(...)
```

Keep old `prepare()` temporarily for regression/backward compatibility.

## Step 2: Load the produced `frames.parquet`

Verify:

```python
DataService.load(custom_frames_path)
```

works with current `FrameStore`, KIS materializer, and API frame lookup.

## Acceptance criteria

The rest of runtime code no longer cares whether frames came from BTC keyframes or custom extraction.

---

# Task 16 — Reuse and harden full-video ASR as a first-class index

**Files**

- Modify: `src/hcmai/data/enrichment/transcripts/pipeline.py`
- Modify: `src/hcmai/data/enrichment/transcripts/store.py`
- Modify/create text artifact builder under `src/hcmai/retriever/text/`
- Create: `tests/transcripts/test_frame_alignment.py`

## Step 1: Preserve timestamped ASR segments

Canonical transcript record should retain:

```text
segment_id
video_id
start_ms
end_ms
text
speaker_id if available
```

## Step 2: Frame alignment

Implement query-time helper:

```python
TranscriptStore.text_for_window(video_id, start_ms, end_ms)
```

And offline frame-aligned evidence generation when needed:

```text
frame timestamp ± ASR alignment window
→ asr_text for frame
```

## Step 3: Build both semantic and lexical search artifacts

Semantic index is already aligned with current retrieval architecture.

Also preserve raw text for exact names, organizations, locations, dates, and numbers.

## Acceptance criteria

A frame can retrieve temporally overlapping transcript text without duplicating ASR generation.

---

# Task 17 — OCR retained frames and preserve structured OCR evidence

**Files**

- Modify: `src/hcmai/enrichment/ocr/models/entities.py`
- Modify: `src/hcmai/enrichment/ocr/artifacts.py`
- Modify: `src/hcmai/data/stores/evidence.py` only if necessary
- Create: `tests/enrichment/ocr/test_structured_artifacts.py`

## Step 1: Keep recognized text and boxes

Do not flatten away coordinates/confidence during generation.

Canonical OCR artifact should preserve:

```text
frame_id
full_text
regions[]:
    text
    bbox
    confidence
```

A frame-level flattened string can still be exported for the current `OCRStore`/retriever.

## Step 2: Avoid rerunning OCR when only indexing changes

Separate recognition artifact from embedding/index artifact.

## Acceptance criteria

Current `DataService.get_evidence(frame_id, OCR)` remains possible while structured OCR is available to Q&A.

---

# Task 18 — Build SigLIP2 visual artifacts on all retained frames

**Files**

- Reuse/modify: `src/hcmai/embedding/adapters/siglip.py`
- Modify: `src/hcmai/embedding/artifacts.py`
- Create: `tests/embedding/test_custom_frame_artifacts.py`

## Step 1: Input source

Read only canonical `frames.parquet`.

Do not re-enumerate image directories independently.

## Step 2: Output

```text
artifacts/visual/siglip2_embeddings.npy
artifacts/visual/mapping.parquet
artifacts/indexes/visual/
```

Mapping identity must use `frame_id`.

## Step 3: Retrieval role only

SigLIP2 must not participate in frame-retention decisions.

## Acceptance criteria

`RetrievalService.from_index()` can search custom retained frames with no other runtime change.

---

# Task 19 — Selective caption enrichment

**Files**

- Modify: `src/hcmai/enrichment/caption/pipeline.py`
- Modify: `src/hcmai/enrichment/caption/config.py`
- Create: `tests/enrichment/caption/test_selection_policy.py`

## Step 1: Add caption eligibility policy

Priority candidates:

```text
is_anchor/protected
shot boundary burst
event boundary burst
text-change frame
high-information frame
```

Coverage-only frames may initially skip expensive captioning.

## Step 2: Keep missing captions legal

Current `FrameEnrichment.caption` is nullable; use that instead of forcing caption coverage.

## Acceptance criteria

Frame retention is never reduced merely to afford captioning cost.

---

# Task 20 — Add lightweight news-story segmentation

**Files**

- Create: `src/hcmai/story/__init__.py`
- Create: `src/hcmai/story/segmenter.py`
- Create: `src/hcmai/story/artifacts.py`
- Create: `tests/story/test_segmenter.py`

## Step 1: Story boundary features

V1 heuristic hierarchy:

```text
strong shot/graphic boundary
+
ASR semantic topic shift
+
optional long visual-semantic discontinuity
```

Do not train a custom NewsNet-style model under current deadline.

## Step 2: Story artifact

```text
story_id
video_id
start_ms
end_ms
representative_frame_ids
transcript_text
ocr_keywords
caption_text
```

## Step 3: Story is a soft prior

Never hard-filter a frame solely because another story ranked higher.

## Acceptance criteria

Every frame maps to zero or one story; story boundaries remain inspectable and editable.

---

# Task 21 — Finish multimodal retrieval around existing RRF

**Files**

- Modify: `src/hcmai/retriever/pipeline.py`
- Reuse: `src/hcmai/retriever/fusion/rrf.py`
- Add if needed: `src/hcmai/retriever/text/lexical.py`
- Create: `tests/retriever/test_multimodal_sources.py`

## Step 1: Keep current dense RRF path

The repository already supports:

```text
visual
caption
OCR
ASR
```

through `RetrievalService.from_indexes()` and `RRFFusionRetriever`.

Do not replace it.

## Step 2: Add lexical fallback/parallel retrieval

Names, locations, dates, and numbers benefit from literal text search.

Add lexical results as either:

- an extra internal rank source folded into the text modality; or
- a new retrieval source only if public schemas can be extended safely.

Prefer minimum schema disruption under deadline.

## Step 3: Story prior

After RRF, optionally add:

```text
candidate.final_score = rrf_score + λ * story_score
```

Story score is a soft boost.

## Acceptance criteria

If OCR/ASR/caption index is missing, visual search still works and existing graceful-degradation behavior remains intact.

---

# Task 22 — Improve KIS result clustering/diversification

**Files**

- Modify: `src/hcmai/orchestration/pipelines/kis.py`
- Create: `src/hcmai/orchestration/diversification.py`
- Create: `tests/orchestration/test_kis_diversification.py`

## Step 1: Add temporal cluster suppression after reranking

Do not allow the top 100 to be filled by dozens of near-adjacent frames from one temporal cluster.

Cluster key:

```text
same video
same shot/story when available
within configurable temporal radius
```

## Step 2: Preserve strongest representative first

Then allow a small number of alternate frames per strong cluster before expanding to other hypotheses.

## Step 3: Do not over-diversify Top-1/Top-5

The official metric rewards strong early ranking. Diversification should mostly protect Top-20/50/100 from duplicate hypotheses.

## Acceptance criteria

Top-100 contains multiple plausible temporal/video hypotheses while keeping the highest-scoring result at rank 1.

---

# Task 23 — Implement the VQA task pipeline using retrieved evidence packs

**Files**

- Inspect/reuse: `src/hcmai/common/schemas/vqa.py`
- Create/complete: `src/hcmai/orchestration/pipelines/vqa.py`
- Create: `src/hcmai/vqa/evidence.py`
- Create: `src/hcmai/vqa/answerer.py`
- Create: `tests/vqa/test_pipeline.py`

## Step 1: Event retrieval

Use event description as primary retrieval query; use question text as auxiliary evidence query.

## Step 2: Build local evidence pack

```python
EvidencePack:
    center_frame
    neighbor_frames
    ocr_regions
    asr_segments
    captions
    story_context
```

Neighbors come only from `FrameStore`; do not open raw video for Q&A.

## Step 3: VLM reasoning

Require structured output:

```json
{
  "answer": "...",
  "supporting_frame_id": "...",
  "confidence": 0.0
}
```

The selected supporting frame must be one of the canonical FrameStore frames provided to the model.

## Step 4: Return public `VQAResponse`

The answer and frame identity must stay tied to the same evidence hypothesis.

## Acceptance criteria

Q&A never depends on raw-video decode at query time.

---

# Task 24 — Implement TRAKE event parser and internal contracts

**Files**

- Create: `src/hcmai/trake/__init__.py`
- Create: `src/hcmai/trake/parser.py`
- Create: `src/hcmai/trake/contracts.py`
- Create: `tests/trake/test_parser.py`

## Step 1: Internal event schema

```python
class ParsedTRAKEEvent(BaseModel):
    index: int
    description: str
    entities: list[str]
    action: str | None
    state: str | None
    relations: list[str]
```

## Step 2: Respect caller-supplied events

If `TRAKERequest.events` is present, do not re-segment them; only enrich/normalize each event.

If absent, parse ordered events from free-form query.

## Step 3: Preserve order invariant

Event indices are immutable after parsing.

## Acceptance criteria

Parser output always contains at least 2 ordered events or returns a clear structured failure.

---

# Task 25 — Batch coarse retrieval for TRAKE events

**Files**

- Create: `src/hcmai/trake/retrieval.py`
- Create: `tests/trake/test_retrieval.py`

## Step 1: Use existing batch retrieval

The repository already exposes:

```python
RetrievalService.search_batch(queries, ...)
```

Use it for all ordered events in one request.

## Step 2: Internal candidate

```python
EventCandidate:
    event_index
    frame_id
    video_id
    frame_idx
    timestamp_ms
    retrieval_score
    source_scores
```

Materialize frame metadata through `DataService`, never infer identity from mapping arrays inside TRAKE.

## Acceptance criteria

N TRAKE events are encoded/searched in batch and produce ordered candidate lists without N independent model reloads.

---

# Task 26 — Rank candidate videos for TRAKE

**Files**

- Create: `src/hcmai/trake/video_ranker.py`
- Create: `tests/trake/test_video_ranker.py`

## Step 1: Aggregate event evidence by video

Initial score:

```text
sum over events of best candidate score in video
+
coverage bonus for number of events represented
```

Do not allow one extremely strong event to dominate a video that has no evidence for the remaining events.

## Step 2: Keep multiple videos

Return configurable Top-V candidate videos for temporal path search.

## Acceptance criteria

A video with candidates for all N events ranks above a video that matches only one event unless score evidence overwhelmingly indicates otherwise.

---

# Task 27 — Temporal beam/DP path search

**Files**

- Create: `src/hcmai/trake/path_search.py`
- Create: `tests/trake/test_path_search.py`

## Step 1: Path validity

For one video choose one candidate per event such that:

```text
frame_idx_1 <= frame_idx_2 <= ... <= frame_idx_N
```

Use non-decreasing order to match current public `TRAKESubmission` validation.

## Step 2: Score

Start with:

```text
path_score = Σ event_candidate_score
             + modality_agreement_bonus
             - pathological_gap_penalty
```

Do not enforce uniform spacing.

## Step 3: Beam search

Keep `beam_width` best partial paths after each event.

This avoids combinatorial explosion for large candidate sets.

## Step 4: Keep alternate paths

Return multiple coarse paths per candidate video so Top-100 TRAKE generation and refinement have alternatives.

## Acceptance criteria

All generated paths satisfy same-video and event-order constraints before raw refinement.

---

# Task 28 — Raw-video TRAKE temporal refinement service

**Files**

- Create: `src/hcmai/preprocessing/video/seek.py`
- Create: `src/hcmai/trake/refine.py`
- Create: `tests/trake/test_refine.py`

## Step 1: Seek using Micro-Index

Input:

```text
video_id
coarse timestamp
outer window
```

Use `gop_seek_pts` to open raw video near the target, then decode only the required local span.

## Step 2: Coarse local pass

Example policy:

```text
outer window: ±5–10 s
sample local frames at 2–5 FPS
score event relevance
choose top temporal peaks
```

These are config defaults, not hardcoded scientific constants.

## Step 3: Native-FPS inner pass

Around each coarse peak:

```text
inner window: ±1–2 s
native FPS decode
```

Collect exact canonical frame indices through the same frame-identity resolver.

## Step 4: Semantic local scorer

Combine:

```text
SigLIP2 event-frame score
local temporal motion/event signals
OCR/ASR alignment when relevant
optional VLM short-sequence judgment
```

For transition-style descriptions, judge short tuples such as `(t-1, t, t+1)` rather than isolated images.

## Step 5: Return alternatives

```python
RefinedEventCandidate:
    frame_idx
    timestamp_ms
    score
    evidence
```

Keep top-M per event, not only top-1, so global ordering can be repaired.

## Acceptance criteria

Refinement does not scan the whole 15–20 minute video and can return native-frame candidates around a coarse event region.

---

# Task 29 — Global TRAKE post-refinement path consistency

**Files**

- Modify: `src/hcmai/trake/path_search.py`
- Create: `tests/trake/test_refined_path_search.py`

## Step 1: Re-run temporal path selection on refined candidates

Do not simply replace each coarse event independently with its local best frame.

Choose refined sequence satisfying:

```text
same video
correct order
one frame per event
```

## Step 2: Fallback

If best local choices conflict temporally, use second/third local candidates rather than discarding the whole path.

## Acceptance criteria

All final `frame_idxs` pass `TRAKESubmission.validate_frame_sequence()`.

---

# Task 30 — TRAKE Top-100 diversified hypothesis generation

**Files**

- Create: `src/hcmai/trake/diversify.py`
- Create: `tests/trake/test_diversify.py`

## Step 1: Candidate hypotheses

Mix:

```text
same video + alternate refined path
alternate candidate video + best path
alternate coarse region + refined path
```

## Step 2: Avoid duplicate rows

Deduplicate identical `(video_id, frame_idxs...)` submissions.

## Step 3: Preserve rank strength

Top-1 is the maximum final path score. Diversity constraints should become progressively stronger deeper in the list.

## Acceptance criteria

Can produce up to 100 unique valid TRAKE submissions without filling the list with near-identical paths.

---

# Task 31 — Create executable `TRAKEPipeline`

**Files**

- Create: `src/hcmai/orchestration/pipelines/trake.py`
- Modify: `src/hcmai/orchestration/pipelines/__init__.py`
- Modify task registry/setup where current KIS pipelines are registered
- Create: `tests/orchestration/test_trake_pipeline.py`

## Step 1: Dependencies

```python
TRAKEPipeline(
    data: DataService,
    retrieval: RetrievalService,
    micro_index_store,
    raw_video_resolver,
    parser,
    path_searcher,
    refiner,
    config,
)
```

## Step 2: Execute stages

```text
parse
→ batch retrieve events
→ rank videos
→ coarse path search
→ raw refine selected paths
→ global refined path search
→ diversify
→ TRAKEResponse
```

## Step 3: Observability

Reuse current tracing style and add stage names for:

```text
trake_parse
trake_event_retrieval
trake_video_ranking
trake_path_search
trake_raw_refinement
trake_diversification
```

## Acceptance criteria

Current placeholder/501 TRAKE behavior is replaced by an executable path returning valid `TRAKEResponse` objects.

---

# Task 32 — Pipeline-wide resumability and artifact manifests

**Files**

- Create: `src/hcmai/preprocessing/resume.py`
- Reuse patterns from: `src/hcmai/enrichment/caption/resume.py`
- Modify preprocessing/enrichment entrypoints to write manifests
- Create: `tests/preprocessing/test_resume.py`

## Step 1: Reuse existing project resume patterns

Do not invent a second incompatible mechanism if caption pipeline already has working resume semantics.

## Step 2: Stage-specific invalidation

Examples:

- Change SigLIP2 model → rebuild visual embeddings/index only.
- Change caption model → rebuild captions/caption index only.
- Change dedup threshold → rebuild retained-frame selection/materialization and all downstream frame-dependent artifacts.
- Change ASR model → rebuild transcripts/ASR index, not frame extraction.

## Acceptance criteria

100+ GB corpus preprocessing is restartable and changes invalidate the minimum necessary artifact subtree.

---

# Task 33 — Observability and operational reports

**Files**

- Modify/reuse: `src/hcmai/observability/metrics.py`
- Modify/reuse: `src/hcmai/observability/stages.py`
- Create: `src/hcmai/preprocessing/report.py`
- Create: `tests/preprocessing/test_report.py`

## Required per-video metrics

```text
native_frame_count
candidate_count
retained_frame_count
removed_by_dedup

candidate_by_reason:
  coverage
  shot
  event
  global_change
  regional_change
  edge_change
  text_change
  codec_motion

runtime_ms:
  probe
  decode
  cheap_signals
  transnet
  estimator
  dino
  materialize
  siglip
  OCR
  caption
  ASR
```

Do not treat these as benchmark metrics yet; they are operational diagnostics.

## Acceptance criteria

A failed/slow video can be diagnosed without re-running with debug prints.

---

# Task 34 — End-to-end L21 smoke command

**Files**

- Add CLI/script at repository root according to existing project CLI conventions.
- Create: `tests/integration/test_l21_smoke.py` with tiny fixture, not real L21 payload.

## Required command sequence

Conceptually:

```bash
# 1. Prepare frames + micro-index
hcmai preprocess --input data/L21 --config configs/aic2026.yaml

# 2. ASR
hcmai transcripts prepare --input data/L21 ...

# 3. OCR/caption
hcmai enrich ...

# 4. SigLIP/text embeddings + indexes
hcmai index build ...

# 5. Start API / run query
hcmai serve ...
```

If the project has a different root CLI, preserve that convention rather than adding multiple ad-hoc scripts.

## Smoke acceptance criteria

For one real L21 video:

- custom `frames.parquet` exists;
- `FrameStore` loads it;
- SigLIP index builds;
- ASR/OCR/caption evidence can be loaded;
- KIS returns ranked canonical frames;
- Q&A can build an evidence pack without raw-video access;
- TRAKE can generate an ordered coarse path and invoke local raw-video refinement;
- no stage requires BTC keyframe images for production execution.

---

# Task 35 — Root configuration for the target solution

**Files**

- Create/modify: `configs/aic2026.yaml`

Use config similar to:

```yaml
preprocessing:
  analysis:
    width: 320
    height: 180
    native_temporal: true

  change:
    global_threshold: null
    regional_threshold: null
    edge_threshold: null

  text_change:
    enabled: true
    lower_third_enabled: true
    ticker_enabled: true

  shot:
    backend: transnetv2
    enabled: true
    threshold: null

  event:
    backend: estimator
    enabled: true
    threshold: null

  compressed_motion:
    enabled: true
    required: false

  coverage:
    maximum_gap_ms: null

  burst:
    shot_radius_ms: null
    event_radius_ms: null
    motion_radius_ms: null
    text_radius_ms: null
    default_step_ms: null

  dedup:
    backend: dinov2
    model_name: facebook/dinov2-base
    window_ms: null
    semantic_threshold: null
    regional_threshold: null
    edge_threshold: null
    text_threshold: null
    motion_threshold: null

  materialization:
    image_format: jpg
    quality: 95

search:
  candidate_count: 500
  rerank_count: 100
  fusion:
    method: rrf

trake:
  candidate_videos: 10
  event_top_k: 100
  beam_width: 50
  coarse_paths_per_video: 10
  refinement:
    outer_window_ms: 10000
    coarse_fps: 3
    inner_window_ms: 1500
    refined_candidates_per_event: 5
```

`null` threshold values above mean “must be deliberately chosen before corpus-scale run”; do not hide arbitrary constants inside Python code.

---

# Dependency graph

```text
Task 1 contracts/config
   │
   ├── Task 2 probe
   │      └── Task 3 identity
   │             └── Task 4 decoder
   │                    ├── Task 5 micro-index
   │                    ├── Task 6 cheap visual signals
   │                    ├── Task 7 text-change
   │                    └── Task 8 compressed motion
   │
   ├── Task 9 TransNetV2
   └── Task 10 ESTimator

Tasks 5–10
   ↓
Task 11 candidate selection
   ↓
Task 12 conservative dedup
   ↓
Task 13 materialization
   ↓
Task 14 preprocessing pipeline
   ↓
Task 15 DataService integration
   ↓
Task 18 SigLIP2 ───────────────┐
Task 16 ASR ───────────────────┤
Task 17 OCR ───────────────────┤
Task 19 captions ──────────────┤
Task 20 stories ───────────────┤
                               ↓
                      Task 21 multimodal retrieval
                         │                  │
                         ↓                  ↓
                    Task 22 KIS       Task 24 TRAKE parser
                         │                  ↓
                         ↓             Task 25 event retrieval
                    Task 23 Q&A            ↓
                                      Task 26 video rank
                                           ↓
                                      Task 27 path search
                                           ↓
                                      Task 28 raw refine
                                           ↓
                                      Task 29 global refine
                                           ↓
                                      Task 30 diversify
                                           ↓
                                      Task 31 TRAKE pipeline

Task 32 resumability + Task 33 observability span all tracks.
Task 34 smoke integration starts once first vertical slice is available.
```

---

# Four-person parallelization

## Track A — Video preprocessing / FrameStore

Own:

```text
Tasks 1–15
Task 32 preprocessing resume
Task 33 preprocessing metrics
```

Priority:

```text
identity → decoder → cheap signals → candidates → dedup → materialize
```

TransNetV2/ESTimator adapters may be developed in parallel after contracts exist.

## Track B — Multimodal enrichment

Own:

```text
Task 16 ASR
Task 17 OCR
Task 18 SigLIP2
Task 19 captions
Task 20 story artifacts
```

Can start immediately using current BTC keyframes or temporary sampled frames; all final artifacts must join by canonical `frame_id` once custom FrameStore lands.

## Track C — KIS / Q&A / retrieval

Own:

```text
Task 21 retrieval integration
Task 22 KIS diversification
Task 23 Q&A
```

Reuse current `RetrievalService`, RRF, reranking, `DataService`, and materializer rather than rebuilding retrieval infrastructure.

## Track D — TRAKE

Own:

```text
Tasks 24–31
```

Can begin with mocked `EventCandidate` lists and generated video fixtures before the final custom FrameStore exists.

---

# Critical-path implementation order

If team capacity is limited, prioritize this exact vertical slice:

```text
1. Video contracts/config
2. Probe + identity + decoder
3. Micro-index
4. Global + regional + edge + text-change signals
5. TransNetV2
6. ESTimator
7. Coverage + burst candidate selector
8. Conservative DINOv2 dedup
9. Materialize custom FrameStore
10. SigLIP2
11. ASR
12. OCR
13. Existing RRF multimodal retrieval
14. KIS
15. Q&A evidence pipeline
16. TRAKE batch event retrieval
17. TRAKE video ranking + beam path
18. TRAKE raw refinement
19. TRAKE Top-100 generation
20. Story/caption enhancements where time remains
```

Compressed motion is useful but not allowed to block the critical path. SEA-RAFT, custom training, DINOv3, and learned story segmentation are explicitly outside the first production slice.

---

# Do not implement yet

Under the current deadline, defer:

- SEA-RAFT as a mandatory preprocessing stage;
- camera-motion compensation pipeline;
- DINOv3 migration before DINOv2 path is stable;
- custom trained GEBD model;
- learned offline frame sampler;
- query-aware offline frame extraction;
- end-to-end learned story segmentation;
- heavy caption generation for every coverage frame;
- full benchmark/ablation harness.

Keep adapters/interfaces so these can be added later without changing canonical artifact contracts.

---

# System invariants

These invariants should be encoded in tests and code review checklists.

1. **Canonical identity:** all public results originate from canonical `FrameRecord` or the same validated resolver used by TRAKE native refinement.
2. **KIS/Q&A no raw-video dependency:** query-time KIS and Q&A must work if raw videos are unavailable.
3. **TRAKE raw refinement is local:** raw-video access is allowed for TRAKE only after coarse retrieval; do not rescan entire videos.
4. **Native-temporal safety pass:** low spatial resolution is allowed; silently dropping temporal frames before cheap safety signals is not the default.
5. **Candidate generation is a union:** one strong signal can protect a frame; weak signals do not dilute each other in an untrained weighted score.
6. **Coverage ceiling:** no candidate-free temporal gap may exceed configured `maximum_gap_ms`.
7. **Protected frames survive dedup.**
8. **No cross-shot dedup.**
9. **DINO similarity alone never deletes a frame.**
10. **SigLIP2 is retrieval-only, not a retention oracle.**
11. **OCR/text change is an independent modality, not subordinate to visual motion.**
12. **ASR is timestamp-native and generated independently from frame extraction.**
13. **Story score is a soft prior, never a hard retrieval filter.**
14. **TRAKE paths preserve one video and event order before and after refinement.**
15. **Every corpus-scale stage is resumable.**

---

# Definition of Done for the end-to-end solution

The target solution is considered implemented when all conditions below hold.

## Offline

- Raw L21 videos can be processed without BTC keyframe images.
- `video_manifest.parquet` is produced.
- Per-video `TemporalMicroIndex` shards are produced.
- Custom retained frames are materialized.
- Canonical `frames.parquet` loads in existing `FrameStore`.
- SigLIP2 visual embeddings/index exist.
- ASR transcript/index exists.
- OCR artifact/index exists.
- Caption artifact/index can be partial.
- Pipeline is resumable by stage and video.

## KIS

- Existing KIS API/pipeline searches the custom FrameStore.
- Retrieval can fuse available visual/OCR/ASR/caption evidence.
- Final results always materialize canonical `video_id/frame_idx`.
- Top-100 is not dominated by duplicate adjacent frames.
- No raw-video access is required at query time.

## Q&A

- Event localization is performed on FrameStore/evidence indexes.
- Local evidence pack includes retained neighboring frames + OCR + ASR + captions when available.
- Answer is coupled to a specific canonical supporting frame.
- No raw-video access is required at query time.

## TRAKE

- Free-form query can be converted to ordered events or caller-supplied events are preserved.
- Events are retrieved in batch.
- Candidate videos are ranked by multi-event evidence.
- Coarse same-video ordered paths are generated.
- `TemporalMicroIndex` seeks local raw-video windows.
- Native-FPS refinement returns exact frame candidates.
- Final global path remains ordered.
- Up to 100 unique valid `TRAKESubmission` rows can be generated.

---

# Research references

1. Tomáš Souček, Jakub Lokoč. **TransNet V2: An Effective Deep Network Architecture for Fast Shot Transition Detection.** https://arxiv.org/abs/2008.04838
2. **Online Generic Event Boundary Detection (ESTimator), ICCV 2025.** https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Online_Generic_Event_Boundary_Detection_ICCV_2025_paper.html
3. **End-to-End Compressed Video Representation Learning for Generic Event Boundary Detection, CVPR 2022.** https://openaccess.thecvf.com/content/CVPR2022/html/Li_End-to-End_Compressed_Video_Representation_Learning_for_Generic_Event_Boundary_Detection_CVPR_2022_paper.html
4. Yuan Zhi et al. **MGSampler: An Explainable Sampling Strategy for Video Action Recognition, ICCV 2021.** https://openaccess.thecvf.com/content/ICCV2021/html/Zhi_MGSampler_An_Explainable_Sampling_Strategy_for_Video_Action_Recognition_ICCV_2021_paper.html
5. Zongyao Li et al. **KFS-Bench: Comprehensive Evaluation of Key Frame Sampling in Long Video Understanding, WACV 2026.** https://openaccess.thecvf.com/content/WACV2026/html/Li_KFS-Bench_Comprehensive_Evaluation_of_Key_Frame_Sampling_in_Long_Video_WACV_2026_paper.html
6. **LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding, ICML 2025.** https://proceedings.mlr.press/v267/shen25j.html
7. **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.** https://arxiv.org/abs/2502.14786
8. Amanpreet Singh et al. **Towards VQA Models That Can Read (TextVQA), CVPR 2019.** https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html
9. Ali Furkan Biten et al. **Scene Text Visual Question Answering (ST-VQA), ICCV 2019.** https://openaccess.thecvf.com/content_ICCV_2019/html/Biten_Scene_Text_Visual_Question_Answering_ICCV_2019_paper.html
10. **EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering, CVPR 2025.** https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_EgoTextVQA_Towards_Egocentric_Scene-Text_Aware_Video_Question_Answering_CVPR_2025_paper.html
11. **TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval, ECCV 2020.** https://arxiv.org/abs/2001.09099
12. **QVHighlights / Detecting Moments and Highlights in Videos via Natural Language Queries (Moment-DETR), NeurIPS 2021.** https://arxiv.org/abs/2107.09609
13. **NewsNet: A Novel Dataset for Hierarchical Temporal Segmentation, CVPR 2023.** https://openaccess.thecvf.com/content/CVPR2023/html/Wu_NewsNet_A_Novel_Dataset_for_Hierarchical_Temporal_Segmentation_CVPR_2023_paper.html
14. **Re-thinking Temporal Search for Long-Form Video Understanding (T*), CVPR 2025.** https://openaccess.thecvf.com/content/CVPR2025/html/Ye_Re-thinking_Temporal_Search_for_Long-Form_Video_Understanding_CVPR_2025_paper.html

---

# Local project sources used for this plan

- `Thong tin vong So tuyen AIC2026.pdf` — official preliminary-round task definitions, scoring, and dataset notes.
- `Shared Video Frame Extraction Baseline.md` — teammate survey/proposed preprocessing architecture.
- Current `src.zip` — existing HCMAI source structure, especially `FrameStore`, `DataService`, SigLIP embedding, ASR, OCR/caption enrichment, RRF retrieval, KIS orchestration, and TRAKE public schemas.
